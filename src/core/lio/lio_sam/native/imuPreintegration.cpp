#include "utility.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include <cmath>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImuOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLaserOdometry;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;

    Eigen::Isometry3d lidarOdomAffine;
    Eigen::Isometry3d imuOdomAffineFront;
    Eigen::Isometry3d imuOdomAffineBack;

    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    tf2::Stamped<tf2::Transform> lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::msg::Odometry> imuOdomQueue;

    TransformFusion(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_transformFusion", options)
    {
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

        callbackGroupImuOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupLaserOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOdomOpt = rclcpp::SubscriptionOptions();
        imuOdomOpt.callback_group = callbackGroupImuOdometry;
        auto laserOdomOpt = rclcpp::SubscriptionOptions();
        laserOdomOpt.callback_group = callbackGroupLaserOdometry;

        subLaserOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry", qos,
            std::bind(&TransformFusion::lidarOdometryHandler, this, std::placeholders::_1),
            laserOdomOpt);
        subImuOdometry = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic+"_incremental", qos_imu,
            std::bind(&TransformFusion::imuOdometryHandler, this, std::placeholders::_1),
            imuOdomOpt);

        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic, qos_imu);
        pubImuPath = create_publisher<nav_msgs::msg::Path>("lio_sam/imu/path", qos);

        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    Eigen::Isometry3d odom2affine(nav_msgs::msg::Odometry odom)
    {
        tf2::Transform t;
        tf2::fromMsg(odom.pose.pose, t);
        return tf2::transformToEigen(tf2::toMsg(t));
    }

    void lidarOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = stamp2Sec(odomMsg->header.stamp);
    }

    void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (stamp2Sec(imuOdomQueue.front().header.stamp) <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        if (imuOdomQueue.empty())
            return;

        Eigen::Isometry3d imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Isometry3d imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Isometry3d imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Isometry3d imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        auto t = tf2::eigenToTransform(imuOdomAffineLast);
        tf2::Stamped<tf2::Transform> tCur;
        tf2::convert(t, tCur);

        // publish latest odometry
        nav_msgs::msg::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = t.transform.translation.x;
        laserOdometry.pose.pose.position.y = t.transform.translation.y;
        laserOdometry.pose.pose.position.z = t.transform.translation.z;
        laserOdometry.pose.pose.orientation = t.transform.rotation;
        pubImuOdometry->publish(laserOdometry);
        
        if (lidarFrame != baselinkFrame)
        {
            tf2::Transform lidar2BaselinkTf;

            // 1) 平移：直接用 extTrans
            lidar2BaselinkTf.setOrigin(
                tf2::Vector3(extTrans.x(), extTrans.y(), extTrans.z()));

            // 2) 旋转：用 extRot 构造
            Eigen::Quaterniond q(extRot);
            lidar2BaselinkTf.setRotation(
                tf2::Quaternion(q.x(), q.y(), q.z(), q.w()));

            tf2::Stamped<tf2::Transform> tb(
                tCur * lidar2BaselinkTf,
                tf2_ros::fromMsg(odomMsg->header.stamp),
                odometryFrame);
            tCur = tb;
        }
	/* 0428
        // publish tf
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tf2::fromMsg(tfBuffer->lookupTransform(
                    lidarFrame, baselinkFrame, rclcpp::Time(0)), lidar2Baselink);
            }
            catch (tf2::TransformException ex)
            {
                RCLCPP_ERROR(get_logger(), "%s", ex.what());
            }
            tf2::Stamped<tf2::Transform> tb(
                tCur * lidar2Baselink, tf2_ros::fromMsg(odomMsg->header.stamp), odometryFrame);
            tCur = tb;
        }
        */
        geometry_msgs::msg::TransformStamped ts;
        tf2::convert(tCur, ts);
        ts.child_frame_id = baselinkFrame;
        tfBroadcaster->sendTransform(ts);

        // publish IMU path
        static nav_msgs::msg::Path imuPath;
        static double last_path_time = -1;
        double imuTime = stamp2Sec(imuOdomQueue.back().header.stamp);
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && stamp2Sec(imuPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath->get_subscription_count() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath->publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::msg::Imu> imuQueOpt;
    std::deque<sensor_msgs::msg::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;
    int skippedPoseFactorCount = 0;
    double firstSkippedPoseFactorTime = -1.0;

    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imu_preintegration", options)
    {
        if (!useImuPreintegration)
        {
            RCLCPP_WARN(get_logger(),
                "[MAPPING_MODE] lioMode=mapping. IMUPreintegration node is disabled.");
            return;
        }

        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1),
            imuOpt);
        // odometry_incremental
        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry", qos,    
            std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1),
            odomOpt);

        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic+"_incremental", qos_imu);
        
        auto p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        lastImuT_opt = -1;
        imuQueOpt.clear();
        imuQueImu.clear();
        doneFirstOpt = false;
        systemInitialized = false;
        skippedPoseFactorCount = 0;
        firstSkippedPoseFactorTime = -1.0;
    }

    bool resetOptimizationWithState(const gtsam::NavState& state,
                                    const gtsam::imuBias::ConstantBias& bias,
                                    const std::string& reason)
    {
        resetOptimization();

        prevPose_ = state.pose();
        prevVel_ = state.v();
        prevState_ = state;
        prevBias_ = bias;

        gtsam::noiseModel::Diagonal::shared_ptr rebasePoseNoise =
            gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.5, 0.5, 1.0, 5.0, 5.0, 5.0).finished());
        gtsam::noiseModel::Diagonal::shared_ptr rebaseVelNoise =
            gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 5.0, 5.0, 5.0).finished());
        gtsam::noiseModel::Diagonal::shared_ptr rebaseBiasNoise =
            gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.01, 0.01, 0.01).finished());

        graphFactors.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), prevPose_, rebasePoseNoise));
        graphFactors.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), prevVel_, rebaseVelNoise));
        graphFactors.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), prevBias_, rebaseBiasNoise));

        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);

        try
        {
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(get_logger(),
                "[IMU_REBASE_EXCEPTION][%s] %s. Reset IMU-preintegration.",
                reason.c_str(), e.what());
            graphFactors.resize(0);
            graphValues.clear();
            resetParams();
            return false;
        }

        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        key = 1;
        systemInitialized = true;
        return true;
    }

    void dropImuBeforeCorrection(double currentCorrectionTime)
    {
        while (!imuQueOpt.empty() && stamp2Sec(imuQueOpt.front().header.stamp) < currentCorrectionTime - delta_t)
        {
            lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp);
            imuQueOpt.pop_front();
        }

        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t)
            imuQueImu.pop_front();
    }

    void repropagateImuOdometry(double currentCorrectionTime)
    {
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;

        double lastImuQT = -1;
        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t)
        {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp);
            imuQueImu.pop_front();
        }

        if (imuQueImu.empty())
            return;

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
        for (int i = 0; i < (int)imuQueImu.size(); ++i)
        {
            sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
            double imuTime = stamp2Sec(thisImu->header.stamp);
            double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);
            if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.1)
            {
                RCLCPP_WARN(get_logger(),
                    "[IMU_DT_REPROP] abnormal dt=%.6f imuTime=%.6f lastImuQT=%.6f "
                    "correctionTime=%.6f imuQueImu=%zu",
                    dt, imuTime, lastImuQT, currentCorrectionTime, imuQueImu.size());
            }

            imuIntegratorImu_->integrateMeasurement(
                gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
            lastImuQT = imuTime;
        }
    }
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = stamp2Sec(odomMsg->header.stamp);

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;

        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        int correctionFlag = static_cast<int>(std::round(odomMsg->pose.covariance[0]));
        bool degenerate = correctionFlag == 1;
        bool skipPoseFactor = correctionFlag >= 2;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

        static bool hasLastCorrection = false;
        static double lastCorrectionTime = -1.0;
        static gtsam::Pose3 lastLidarPose;

        if (hasLastCorrection)
        {
            double dt_corr = currentCorrectionTime - lastCorrectionTime;
            double dx = p_x - lastLidarPose.translation().x();
            double dy = p_y - lastLidarPose.translation().y();
            double dz = p_z - lastLidarPose.translation().z();
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            double speed_lidar = dist / (dt_corr > 1e-3 ? dt_corr : 1e-3);
            gtsam::Rot3 rotDelta = lastLidarPose.between(lidarPose).rotation();

            RCLCPP_WARN(get_logger(),
                "[LIDAR_CORR_DELTA] dt=%.3f dist=%.3f speed=%.3f "
                "cur=(%.3f %.3f %.3f) rpy=(%.2f %.2f %.2f) "
                "drpy=(%.2f %.2f %.2f) degenerate=%d flag=%d skipPose=%d imuQueOpt=%zu imuQueImu=%zu",
                dt_corr, dist, speed_lidar,
                p_x, p_y, p_z,
                lidarPose.rotation().roll() * 180.0 / M_PI,
                lidarPose.rotation().pitch() * 180.0 / M_PI,
                lidarPose.rotation().yaw() * 180.0 / M_PI,
                rotDelta.roll() * 180.0 / M_PI,
                rotDelta.pitch() * 180.0 / M_PI,
                rotDelta.yaw() * 180.0 / M_PI,
                int(degenerate),
                correctionFlag,
                int(skipPoseFactor),
                imuQueOpt.size(),
                imuQueImu.size());
        }

        lastCorrectionTime = currentCorrectionTime;
        lastLidarPose = lidarPose;
        hasLastCorrection = true;


        // 0. initialize system
        if (systemInitialized == false)
        {
            if (skipPoseFactor)
            {
                dropImuBeforeCorrection(currentCorrectionTime);
                RCLCPP_WARN(get_logger(),
                    "[IMU_INIT_SKIP_POSE] flag=%d imuQueOpt=%zu imuQueImu=%zu. "
                    "Wait for a trusted LiDAR correction before initializing the IMU graph.",
                    correctionFlag, imuQueOpt.size(), imuQueImu.size());
                return;
            }

            resetOptimization();

            // pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (stamp2Sec(imuQueOpt.front().header.stamp) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp);
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            try
            {
                optimizer.update(graphFactors, graphValues);
                graphFactors.resize(0);
                graphValues.clear();
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(get_logger(),
                    "[IMU_OPT_EXCEPTION][INIT] %s. Reset IMU-preintegration.",
                    e.what());
                resetParams();
                return;
            }

            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        if (key == 100)
        {
            try
            {
                // get updated noise before reset
                gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
                gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
                gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
                // reset graph
                resetOptimization();
                // add pose
                gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
                graphFactors.add(priorPose);
                // add velocity
                gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
                graphFactors.add(priorVel);
                // add bias
                gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
                graphFactors.add(priorBias);
                // add values
                graphValues.insert(X(0), prevPose_);
                graphValues.insert(V(0), prevVel_);
                graphValues.insert(B(0), prevBias_);
                // optimize once
                optimizer.update(graphFactors, graphValues);
                graphFactors.resize(0);
                graphValues.clear();

                key = 1;
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(get_logger(),
                    "[IMU_OPT_EXCEPTION][RESET_WINDOW] %s. Reset IMU-preintegration.",
                    e.what());
                resetParams();
                return;
            }
        }


        // 1. integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::msg::Imu *thisImu = &imuQueOpt.front();
            double imuTime = stamp2Sec(thisImu->header.stamp);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.1)
                {
                    RCLCPP_WARN(get_logger(),
                        "[IMU_DT_OPT] abnormal dt=%.6f imuTime=%.6f lastImuT_opt=%.6f "
                        "correctionTime=%.6f imuQueOpt=%zu",
                        dt, imuTime, lastImuT_opt, currentCorrectionTime, imuQueOpt.size());
                }
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        if (skipPoseFactor)
        {
            if (skippedPoseFactorCount == 0)
                firstSkippedPoseFactorTime = currentCorrectionTime;
            skippedPoseFactorCount++;

            double skippedDuration = currentCorrectionTime - firstSkippedPoseFactorTime;
            RCLCPP_WARN(get_logger(),
                "[IMU_POSE_FACTOR_SKIP] key=%d flag=%d skipped=%d duration=%.3f. "
                "Rebase at IMU prediction, but do not add a graph key.",
                key, correctionFlag, skippedPoseFactorCount, skippedDuration);

            bool tooManySkips = imuPoseFactorSkipMaxConsecutive > 0 &&
                                skippedPoseFactorCount > imuPoseFactorSkipMaxConsecutive;
            bool tooLongSkip = imuPoseFactorSkipMaxTime > 0.0 &&
                               skippedDuration > imuPoseFactorSkipMaxTime;
            if (tooManySkips || tooLongSkip)
            {
                RCLCPP_WARN(get_logger(),
                    "[IMU_POSE_FACTOR_SKIP_RESET] skipped=%d duration=%.3f maxSkipped=%d maxDuration=%.3f. "
                    "Reset IMU-preintegration to avoid an overlong unconstrained chain.",
                    skippedPoseFactorCount,
                    skippedDuration,
                    imuPoseFactorSkipMaxConsecutive,
                    imuPoseFactorSkipMaxTime);
                resetParams();
                return;
            }

            gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
            Eigen::Vector3f vel(propState_.v().x(), propState_.v().y(), propState_.v().z());
            Eigen::Vector3f ba(prevBias_.accelerometer().x(), prevBias_.accelerometer().y(), prevBias_.accelerometer().z());
            Eigen::Vector3f bg(prevBias_.gyroscope().x(), prevBias_.gyroscope().y(), prevBias_.gyroscope().z());
            if (!vel.allFinite() || !ba.allFinite() || !bg.allFinite() ||
                vel.norm() > imuFailureVelocityThreshold ||
                ba.norm() > imuFailureBiasThreshold ||
                bg.norm() > imuFailureBiasThreshold)
            {
                RCLCPP_WARN(get_logger(),
                    "[IMU_SKIP_PRED_RESET] vel=(%.3f %.3f %.3f) |v|=%.3f "
                    "ba=(%.4f %.4f %.4f) |ba|=%.4f bg=(%.4f %.4f %.4f) |bg|=%.4f. "
                    "Reset IMU-preintegration.",
                    vel.x(), vel.y(), vel.z(), vel.norm(),
                    ba.x(), ba.y(), ba.z(), ba.norm(),
                    bg.x(), bg.y(), bg.z(), bg.norm());
                resetParams();
                return;
            }
            if (!resetOptimizationWithState(propState_, prevBias_, "SKIP_POSE"))
                return;

            repropagateImuOdometry(currentCorrectionTime);
            doneFirstOpt = true;
            return;
        }
        else
        {
            skippedPoseFactorCount = 0;
            firstSkippedPoseFactorTime = -1.0;
        }

        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        if (!skipPoseFactor)
        {
            gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
            graphFactors.add(pose_factor);
        }
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        try
        {
            optimizer.update(graphFactors, graphValues);
            optimizer.update();
            graphFactors.resize(0);
            graphValues.clear();
            // Overwrite the beginning of the preintegration for the next step.
            gtsam::Values result = optimizer.calculateEstimate();
            prevPose_  = result.at<gtsam::Pose3>(X(key));
            prevVel_   = result.at<gtsam::Vector3>(V(key));
            prevState_ = gtsam::NavState(prevPose_, prevVel_);
            prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(get_logger(),
                "[IMU_OPT_EXCEPTION] key=%d flag=%d skipPose=%d %s. Reset IMU-preintegration.",
                key, correctionFlag, int(skipPoseFactor), e.what());
            graphFactors.resize(0);
            graphValues.clear();
            if (imuResetOnOptimizationFailure)
                resetParams();
            return;
        }
        // Reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t)
        {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp);
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
                double imuTime = stamp2Sec(thisImu->header.stamp);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);
                if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.1)
                {
                    RCLCPP_WARN(get_logger(),
                        "[IMU_DT_REPROP] abnormal dt=%.6f imuTime=%.6f lastImuQT=%.6f "
                        "correctionTime=%.6f imuQueImu=%zu",
                        dt, imuTime, lastImuQT, currentCorrectionTime, imuQueImu.size());
                }

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());

        RCLCPP_WARN(get_logger(),
            "[IMU_PREINT_CHECK] key=%d vel=(%.3f %.3f %.3f) |v|=%.3f "
            "ba=(%.4f %.4f %.4f) |ba|=%.4f "
            "bg=(%.4f %.4f %.4f) |bg|=%.4f",
            key,
            vel.x(), vel.y(), vel.z(), vel.norm(),
            ba.x(), ba.y(), ba.z(), ba.norm(),
            bg.x(), bg.y(), bg.z(), bg.norm());

        if (!vel.allFinite() || !ba.allFinite() || !bg.allFinite())
        {
            RCLCPP_WARN(get_logger(), "Non-finite velocity or bias, reset IMU-preintegration!");
            return true;
        }

        if (vel.norm() > imuFailureVelocityThreshold)
        {
            RCLCPP_WARN(get_logger(),
                "Large velocity %.3f > %.3f, reset IMU-preintegration!",
                vel.norm(), imuFailureVelocityThreshold);
            return true;
        }

        if (ba.norm() > imuFailureBiasThreshold || bg.norm() > imuFailureBiasThreshold)
        {
            RCLCPP_WARN(get_logger(),
                "Large bias ba=%.3f bg=%.3f limit=%.3f, reset IMU-preintegration!",
                ba.norm(), bg.norm(), imuFailureBiasThreshold);
            return true;
        }

        return false;
    }

    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        sensor_msgs::msg::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false)
            return;

        double imuTime = stamp2Sec(thisImu.header.stamp);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.1)
        {
            RCLCPP_WARN(get_logger(),
                "[IMU_DT_ODOM] abnormal dt=%.6f imuTime=%.6f lastImuT_imu=%.6f imuQueImu=%zu",
                dt, imuTime, lastImuT_imu, imuQueImu.size());
        }
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry->publish(odometry);
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor e;

    auto ImuP = std::make_shared<IMUPreintegration>(options);
    auto TF = std::make_shared<TransformFusion>(options);
    e.add_node(ImuP);
    e.add_node(TF);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
