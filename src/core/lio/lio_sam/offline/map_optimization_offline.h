#include "utility_offline.hpp"
#pragma once

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    LioSamCloudInfo cloudInfo;
    bool createdNewKeyframe = false;
    bool hasLatestLaserOdometryIncremental = false;
    LioSamOdometryState latestLaserOdometryIncremental;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> rawCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    std::vector<int> surroundingKeyFrameIndices;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudRawFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudRawFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterICP;

    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    
    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    struct MotionGateResult
    {
        bool initialized = false;
        bool continuous = true;
        bool predictedReference = false;
        bool badSpeed = false;
        bool badAcceleration = false;
        bool badAngularVelocity = false;
        bool badAngularAcceleration = false;
        bool badCurvature = false;
        bool badPositionInnovation = false;
        bool badYawInnovation = false;
        bool badRollPitch = false;
        bool badFinite = false;
        double dt = 0.0;
        double ds = 0.0;
        double speed = 0.0;
        double acceleration = 0.0;
        double rollDeg = 0.0;
        double pitchDeg = 0.0;
        double angleDeg = 0.0;
        double yawDeg = 0.0;
        double omega = 0.0;
        double alpha = 0.0;
        double curvature = 0.0;
    };

    enum class MappingTrackingState
    {
        TRACKING,
        LOST,
        RECOVERY_CANDIDATE
    };

    bool mappingPoseReliable = true;
    int lidarCorrectionFlag = 0; // 0: reliable pose, 2: failed pose, IMU side skips LiDAR pose factor
    std::string mappingPoseSource = "INIT";
    MappingTrackingState mappingTrackingState = MappingTrackingState::TRACKING;
    int mappingFailureCount = 0;
    double mappingFirstFailureTime = -1.0;
    bool motionGateHasLast = false;
    double motionGateLastTime = -1.0;
    double motionGateLastSpeed = 0.0;
    double motionGateLastOmega = 0.0;
    float motionGateLastTransform[6] = {0, 0, 0, 0, 0, 0};
    bool lowSpeedGuessHasPrevTrusted = false;
    bool lowSpeedGuessHasLastTrusted = false;
    double lowSpeedGuessPrevTrustedTime = -1.0;
    double lowSpeedGuessLastTrustedTime = -1.0;
    float lowSpeedGuessPrevTrustedTransform[6] = {0, 0, 0, 0, 0, 0};
    float lowSpeedGuessLastTrustedTransform[6] = {0, 0, 0, 0, 0, 0};
    bool hasLastTrustedMappingTransform = false;
    float lastTrustedMappingTransform[6] = {0, 0, 0, 0, 0, 0};
    bool mappingPredictedPoseAvailable = false;
    double mappingPredictedPoseTime = -1.0;
    float mappingPredictedTransform[6] = {0, 0, 0, 0, 0, 0};
    bool pendingRecoveryAvailable = false;
    double pendingRecoveryTime = -1.0;
    float pendingRecoveryTransform[6] = {0, 0, 0, 0, 0, 0};
    std::string pendingRecoverySource = "NONE";
    int lastLMCloudSelNum = 0;
    int lastLMIterationCount = 0;
    bool lastLMRan = false;
    bool lastLMConverged = false;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;
    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options)
    {
        cout << "------------build version - 20260513-1100 -------------------\n" << endl;
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());


        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudRawFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudRawFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
    }

    bool Run(LioSamCloudInfo& msgIn)
    {
        createdNewKeyframe = false;
        hasLatestLaserOdometryIncremental = false;

        // extract time stamp
        timeLaserInfoCur = msgIn.timestamp;

        // extract info and feature cloud
        cloudInfo = msgIn;
        laserCloudCornerLast = msgIn.cloud_corner;
        laserCloudSurfLast = msgIn.cloud_surface;

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            resetFrameQuality();

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();
            performLoopClosure();
            const size_t keyframeCountBefore = cloudKeyPoses6D->size();
            saveKeyFramesAndFactor();
            createdNewKeyframe = cloudKeyPoses6D->size() > keyframeCountBefore;

            correctPoses();

            updateOdometryState();

            int keyframeAllowed =int(mappingPoseReliable && !isDegenerate && transformIsFinite(transformTobeMapped));
            
            RCLCPP_WARN(get_logger(),
                "[T2M] time=%.6f roll=%.3f pitch=%.3f yaw=%.3f "
                "x=%.3f y=%.3f z=%.3f degenerate=%d reliable=%d keyframeAllowed=%d "
                "state=%s source=%s keyposes=%zu corner=%zu surf=%zu",
                timeLaserInfoCur,
                transformTobeMapped[0] * 180.0 / M_PI,
                transformTobeMapped[1] * 180.0 / M_PI,
                transformTobeMapped[2] * 180.0 / M_PI,
                transformTobeMapped[3],
                transformTobeMapped[4],
                transformTobeMapped[5],
                int(isDegenerate),
                int(mappingPoseReliable),
                keyframeAllowed,
                trackingStateName(),
                mappingPoseSource.c_str(),
                cloudKeyPoses3D->size(),
                laserCloudCornerLastDS->size(),
                laserCloudSurfLastDS->size());

            return true;
        }

        return true;
    }

    double TimeLaserInfoCur() const { return timeLaserInfoCur; }

    const float* TransformTobeMapped() const { return transformTobeMapped; }

    bool CreatedNewKeyframe() const { return createdNewKeyframe; }

    void ClearCreatedNewKeyframe() { createdNewKeyframe = false; }

    size_t KeyPoseSize() const
    {
        return cloudKeyPoses6D ? cloudKeyPoses6D->size() : 0;
    }

    PointTypePose KeyPose(size_t idx) const
    {
        return cloudKeyPoses6D->points[idx];
    }

    pcl::PointCloud<PointType>::Ptr LatestRawCloudKeyFrame() const
    {
        if (rawCloudKeyFrames.empty())
            return nullptr;
        return rawCloudKeyFrames.back();
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    void copyTransform(const float src[6], float dst[6])
    {
        for (int i = 0; i < 6; ++i)
            dst[i] = src[i];
    }

    bool transformIsFinite(const float transformIn[6])
    {
        for (int i = 0; i < 6; ++i)
        {
            if (!std::isfinite(static_cast<double>(transformIn[i])))
                return false;
        }
        return true;
    }

    const char* trackingStateName() const
    {
        if (mappingTrackingState == MappingTrackingState::LOST)
            return "LOST";
        if (mappingTrackingState == MappingTrackingState::RECOVERY_CANDIDATE)
            return "RECOVERY_CANDIDATE";
        return "TRACKING";
    }

    void fillMotionDeltaMetrics(const float fromTransform[6], const float toTransform[6],
                                MotionGateResult& result)
    {
        Eigen::Affine3f transFrom = pcl::getTransformation(
            fromTransform[3], fromTransform[4], fromTransform[5],
            fromTransform[0], fromTransform[1], fromTransform[2]);
        Eigen::Affine3f transTo = pcl::getTransformation(
            toTransform[3], toTransform[4], toTransform[5],
            toTransform[0], toTransform[1], toTransform[2]);
        Eigen::Affine3f transDelta = transFrom.inverse() * transTo;

        float dx, dy, dz, droll, dpitch, dyaw;
        pcl::getTranslationAndEulerAngles(transDelta, dx, dy, dz, droll, dpitch, dyaw);

        result.ds = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double angleRad = std::sqrt(droll * droll + dpitch * dpitch + dyaw * dyaw);
        result.rollDeg = std::fabs(droll) * 180.0 / M_PI;
        result.pitchDeg = std::fabs(dpitch) * 180.0 / M_PI;
        result.angleDeg = angleRad * 180.0 / M_PI;
        result.yawDeg = std::fabs(dyaw) * 180.0 / M_PI;
        result.curvature = result.ds > 1e-3 ? std::fabs(dyaw) / result.ds : 0.0;
    }

    void syncMappingPrediction(const float acceptedTransform[6])
    {
        if (!transformIsFinite(acceptedTransform))
            return;

        copyTransform(acceptedTransform, mappingPredictedTransform);
        mappingPredictedPoseTime = timeLaserInfoCur;
        mappingPredictedPoseAvailable = true;
    }

    bool estimateLowSpeedVelocity(double& vx, double& vy, double& vz)
    {
        vx = 0.0;
        vy = 0.0;
        vz = 0.0;

        if (!lowSpeedGuessHasPrevTrusted || !lowSpeedGuessHasLastTrusted)
            return false;

        const double dtHist = lowSpeedGuessLastTrustedTime - lowSpeedGuessPrevTrustedTime;
        if (!std::isfinite(dtHist) || dtHist <= 1e-3)
            return false;

        vx = (lowSpeedGuessLastTrustedTransform[3] - lowSpeedGuessPrevTrustedTransform[3]) / dtHist;
        vy = (lowSpeedGuessLastTrustedTransform[4] - lowSpeedGuessPrevTrustedTransform[4]) / dtHist;
        vz = (lowSpeedGuessLastTrustedTransform[5] - lowSpeedGuessPrevTrustedTransform[5]) / dtHist;
        const double speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (mappingLowSpeedMaxTranslationSpeed > 0.0 && speed > mappingLowSpeedMaxTranslationSpeed)
        {
            const double scale = mappingLowSpeedMaxTranslationSpeed / std::max(speed, 1e-6);
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }

        return true;
    }

    void resetFrameQuality()
    {
        mappingPoseReliable = false;
        lidarCorrectionFlag = 2;
        mappingPoseSource = "FAIL";
        lastLMCloudSelNum = 0;
        lastLMIterationCount = 0;
        lastLMRan = false;
        lastLMConverged = false;
    }

    bool acceptMappingPose(const std::string& source)
    {

        const bool recovering = mappingTrackingState != MappingTrackingState::TRACKING;
        mappingPoseReliable = true;
        lidarCorrectionFlag = 0;
        mappingPoseSource = source;
        mappingFailureCount = 0;
        mappingFirstFailureTime = -1.0;
        pendingRecoveryAvailable = false;
        pendingRecoverySource = "NONE";
        mappingTrackingState = MappingTrackingState::TRACKING;

        if (recovering)
        {
            motionGateHasLast = false;
            motionGateLastSpeed = 0.0;
            motionGateLastOmega = 0.0;
        }

        return true;
    }

    void rejectMappingPose(const float fallbackTransform[6], const std::string& source)
    {
        const bool keepPendingRecovery =
            source.find("RECOVERY_PENDING") != std::string::npos;
        if (!keepPendingRecovery)
        {
            mappingTrackingState = MappingTrackingState::LOST;
            pendingRecoveryAvailable = false;
            pendingRecoverySource = "NONE";
        }

        copyTransform(fallbackTransform, transformTobeMapped);
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);

        mappingPoseReliable = false;
        lidarCorrectionFlag = 2;
        mappingPoseSource = source;

        if (mappingFailureCount == 0)
            mappingFirstFailureTime = timeLaserInfoCur;
        mappingFailureCount++;

        RCLCPP_WARN(get_logger(),
            "[MAPPING_FAIL] state=%s source=%s failedCount=%d failedDuration=%.3f. "
            "Use last reliable pose only; do not add keyframe or LiDAR pose factor.",
            trackingStateName(),
            mappingPoseSource.c_str(),
            mappingFailureCount,
            mappingFirstFailureTime >= 0.0 ? timeLaserInfoCur - mappingFirstFailureTime : 0.0);
    }

    MotionGateResult evaluateMotionGate(const float candidateTransform[6])
    {
        MotionGateResult result;

        if (!transformIsFinite(candidateTransform))
        {
            result.badFinite = true;
            result.continuous = false;
            return result;
        }

        if (!mappingMotionGateEnable)
            return result;

        const bool usePredictedReference =
            mappingTrackingState != MappingTrackingState::TRACKING &&
            mappingPredictedPoseAvailable &&
            transformIsFinite(mappingPredictedTransform);

        if (usePredictedReference)
        {
            result.initialized = true;
            result.predictedReference = true;

            // time since the last accepted tracking reference.
            // In LOST/RECOVERY this can be large; do not use it for velocity here,
            // but use it to relax the allowed prediction innovation.
            result.dt = motionGateHasLast ? timeLaserInfoCur - motionGateLastTime : 0.0;
            if (!std::isfinite(result.dt) || result.dt < 0.0)
                result.dt = 0.0;

            fillMotionDeltaMetrics(mappingPredictedTransform, candidateTransform, result);

            // Base position innovation gate.
            double maxPositionInnovation = mappingRecoveryMaxPositionError;

            // LOST/RECOVERY 状态下，prediction 只靠 IMU 旋转 + 低速外推，
            // 时间越久，平移预测误差越可能累积。
            // 所以这里允许 position innovation 随丢失时间适度放宽。
            if (mappingTrackingState != MappingTrackingState::TRACKING)
            {
                // TODO: 后面建议做成 yaml 参数
                const double recoveryDriftSpeed = 0.10;  // m/s
                const double recoveryExtraMax   = 2.5;   // m

                const double extraAllowance =
                    std::min(recoveryExtraMax, recoveryDriftSpeed * result.dt);

                maxPositionInnovation += extraAllowance;
            }

            result.badPositionInnovation =
                maxPositionInnovation > 0.0 &&
                result.ds > maxPositionInnovation;

            // yaw 仍然用固定阈值，不随 LOST 时间放宽太多；
            // 否则很容易把错误朝向的 LM 解放进 recovery。
            result.badYawInnovation =
                mappingRecoveryMaxYawDeg > 0.0 &&
                result.yawDeg > mappingRecoveryMaxYawDeg;

            // roll/pitch 对地面车仍然是硬约束。
            result.badRollPitch =
                mappingMotionMaxRollPitchDeg > 0.0 &&
                (result.rollDeg > mappingMotionMaxRollPitchDeg ||
                result.pitchDeg > mappingMotionMaxRollPitchDeg);

            result.continuous = !(result.badPositionInnovation ||
                                result.badYawInnovation ||
                                result.badRollPitch ||
                                result.badFinite);

            return result;
        }

        if (!motionGateHasLast)
            return result;

        result.initialized = true;
        result.dt = timeLaserInfoCur - motionGateLastTime;
        if (!std::isfinite(result.dt) || result.dt <= 1e-3)
        {
            result.badFinite = true;
            result.continuous = false;
            return result;
        }

        fillMotionDeltaMetrics(motionGateLastTransform, candidateTransform, result);

        result.speed = result.ds / result.dt;
        result.acceleration = std::fabs(result.speed - motionGateLastSpeed) / result.dt;
        result.omega = result.angleDeg / result.dt;
        result.alpha = std::fabs(result.omega - motionGateLastOmega) / result.dt;

        result.badSpeed =
            mappingMotionMaxSpeed > 0.0 &&
            result.speed > mappingMotionMaxSpeed;

        result.badAcceleration =
            mappingMotionMaxAcceleration > 0.0 &&
            result.acceleration > mappingMotionMaxAcceleration;

        result.badAngularVelocity =
            mappingMotionMaxAngularVelocity > 0.0 &&
            result.omega > mappingMotionMaxAngularVelocity;

        result.badAngularAcceleration =
            mappingMotionMaxAngularAcceleration > 0.0 &&
            result.alpha > mappingMotionMaxAngularAcceleration;

        result.badCurvature =
            mappingMotionMaxCurvature > 0.0 &&
            result.ds > 0.5 &&
            result.curvature > mappingMotionMaxCurvature;

        result.badRollPitch =
            mappingMotionMaxRollPitchDeg > 0.0 &&
            (result.rollDeg > mappingMotionMaxRollPitchDeg ||
            result.pitchDeg > mappingMotionMaxRollPitchDeg);

        result.continuous = !(result.badSpeed ||
                            result.badAcceleration ||
                            result.badAngularVelocity ||
                            result.badAngularAcceleration ||
                            result.badCurvature ||
                            result.badPositionInnovation ||
                            result.badYawInnovation ||
                            result.badRollPitch ||
                            result.badFinite);

        return result;
    }

    void logMotionGate(const std::string& source, const MotionGateResult& motion)
    {
        if (!motion.initialized && !motion.badFinite)
            return;

        RCLCPP_WARN(get_logger(),
            "[MOTION_GATE][%s] ok=%d state=%s ref=%s dt=%.3f ds=%.3f drp=(%.2f %.2f)deg dyaw=%.2fdeg "
            "v=%.3f(prev=%.3f,a=%.3f) omega=%.2f(prev=%.2f,alpha=%.2f) "
            "kappa=%.3f bad(speed=%d accel=%d omega=%d alpha=%d kappa=%d posInnov=%d yawInnov=%d rollpitch=%d finite=%d)",
            source.c_str(),
            int(motion.continuous),
            trackingStateName(),
            motion.predictedReference ? "PREDICTED" : "TRUSTED",
            motion.dt,
            motion.ds,
            motion.rollDeg,
            motion.pitchDeg,
            motion.yawDeg,
            motion.speed,
            motionGateLastSpeed,
            motion.acceleration,
            motion.omega,
            motionGateLastOmega,
            motion.alpha,
            motion.curvature,
            int(motion.badSpeed),
            int(motion.badAcceleration),
            int(motion.badAngularVelocity),
            int(motion.badAngularAcceleration),
            int(motion.badCurvature),
            int(motion.badPositionInnovation),
            int(motion.badYawInnovation),
            int(motion.badRollPitch),
            int(motion.badFinite));
    }

    void commitMotionGateReference(const float acceptedTransform[6])
    {
        if (!transformIsFinite(acceptedTransform))
            return;

        MotionGateResult motion = evaluateMotionGate(acceptedTransform);
        if (motion.initialized)
        {
            motionGateLastSpeed = motion.speed;
            motionGateLastOmega = motion.omega;
        }
        else
        {
            motionGateLastSpeed = 0.0;
            motionGateLastOmega = 0.0;
        }

        copyTransform(acceptedTransform, motionGateLastTransform);
        motionGateLastTime = timeLaserInfoCur;
        motionGateHasLast = true;
    }

    void commitLowSpeedTrustedPose(const float acceptedTransform[6])
    {
        if (!transformIsFinite(acceptedTransform))
            return;

        if (lowSpeedGuessHasLastTrusted && timeLaserInfoCur > lowSpeedGuessLastTrustedTime + 1e-3)
        {
            copyTransform(lowSpeedGuessLastTrustedTransform, lowSpeedGuessPrevTrustedTransform);
            lowSpeedGuessPrevTrustedTime = lowSpeedGuessLastTrustedTime;
            lowSpeedGuessHasPrevTrusted = true;
        }

        copyTransform(acceptedTransform, lowSpeedGuessLastTrustedTransform);
        lowSpeedGuessLastTrustedTime = timeLaserInfoCur;
        lowSpeedGuessHasLastTrusted = true;
    }

    void advanceMappingPrediction(bool hasImuRotationIncrement, const Eigen::Affine3f& imuRotationIncrement)
    {
        if (!mappingPredictedPoseAvailable)
        {
            if (hasLastTrustedMappingTransform)
                copyTransform(lastTrustedMappingTransform, mappingPredictedTransform);
            else
                copyTransform(transformTobeMapped, mappingPredictedTransform);
            mappingPredictedPoseTime = timeLaserInfoCur;
            mappingPredictedPoseAvailable = true;
        }

        Eigen::Affine3f predictedAffine = trans2Affine3f(mappingPredictedTransform);
        if (hasImuRotationIncrement)
            predictedAffine = predictedAffine * imuRotationIncrement;

        pcl::getTranslationAndEulerAngles(predictedAffine,
            mappingPredictedTransform[3], mappingPredictedTransform[4], mappingPredictedTransform[5],
            mappingPredictedTransform[0], mappingPredictedTransform[1], mappingPredictedTransform[2]);

        const double dtRaw = timeLaserInfoCur - mappingPredictedPoseTime;
        if (std::isfinite(dtRaw) && dtRaw > 0.0)
        {
            double dt = dtRaw;
            if (mappingTrackingState == MappingTrackingState::TRACKING)
            {
                if (mappingLowSpeedMaxExtrapolationTime > 0.0)
                    dt = std::min(dt, static_cast<double>(mappingLowSpeedMaxExtrapolationTime));
            }
            else
            {
                // LOST / RECOVERY 时允许预测追上时间跨度
                const double lostMaxExtrapolationTime = 30.0;  // 先写死，后面做参数
                dt = std::min(dt, lostMaxExtrapolationTime);
            }

            double vx, vy, vz;
            if (estimateLowSpeedVelocity(vx, vy, vz))
            {
                mappingPredictedTransform[3] += vx * dt;
                mappingPredictedTransform[4] += vy * dt;
                mappingPredictedTransform[5] += vz * dt;
            }
        }

        mappingPredictedPoseTime = timeLaserInfoCur;
        copyTransform(mappingPredictedTransform, transformTobeMapped);
    }

    bool pendingRecoveryMotionConsistent(const float candidateTransform[6])
    {
        if (!pendingRecoveryAvailable || !transformIsFinite(candidateTransform))
            return false;

        const double dt = timeLaserInfoCur - pendingRecoveryTime;
        if (!std::isfinite(dt) || dt <= 1e-3)
            return false;

        MotionGateResult motion;
        fillMotionDeltaMetrics(pendingRecoveryTransform, candidateTransform, motion);

        const double maxDs =
            std::max(0.0f, mappingLowSpeedMaxTranslationSpeed) * dt +
            std::max(0.0f, mappingRecoveryMaxPositionError);
        const double maxYaw = mappingRecoveryMaxYawDeg;

        const bool badDs = maxDs > 0.0 && motion.ds > maxDs;
        const bool badYaw = maxYaw > 0.0 && motion.yawDeg > maxYaw;
        const bool badRollPitch = mappingMotionMaxRollPitchDeg > 0.0 &&
                                  (motion.rollDeg > mappingMotionMaxRollPitchDeg ||
                                   motion.pitchDeg > mappingMotionMaxRollPitchDeg);

        RCLCPP_WARN(get_logger(),
            "[RECOVERY][CHECK] pending=%s dt=%.3f ds=%.3f/%.3f dyaw=%.2f/%.2f "
            "drp=(%.2f %.2f) bad(ds=%d yaw=%d rollpitch=%d)",
            pendingRecoverySource.c_str(),
            dt,
            motion.ds,
            maxDs,
            motion.yawDeg,
            maxYaw,
            motion.rollDeg,
            motion.pitchDeg,
            int(badDs),
            int(badYaw),
            int(badRollPitch));

        return !(badDs || badYaw || badRollPitch);
    }

    bool acceptOrStageMappingCandidate(const std::string& source,
                                       const float candidateTransform[6],
                                       const float fallbackTransform[6])
    {

        if (mappingTrackingState == MappingTrackingState::TRACKING)
        { 
            return acceptMappingPose(source);
        }

        if (mappingTrackingState == MappingTrackingState::RECOVERY_CANDIDATE &&
            pendingRecoveryMotionConsistent(candidateTransform))
        {
            RCLCPP_WARN(get_logger(),
                "[RECOVERY][ACCEPT] source=%s confirmed by two consecutive candidates.",
                source.c_str());
            return acceptMappingPose(source);
        }

        copyTransform(candidateTransform, pendingRecoveryTransform);
        pendingRecoveryTime = timeLaserInfoCur;
        pendingRecoverySource = source;
        pendingRecoveryAvailable = true;
        mappingTrackingState = MappingTrackingState::RECOVERY_CANDIDATE;

        RCLCPP_WARN(get_logger(),
            "[RECOVERY][PENDING] source=%s accepted as first recovery candidate only. "
            "Wait one more consistent candidate before adding keyframe.",
            source.c_str());
        rejectMappingPose(fallbackTransform, "RECOVERY_PENDING_" + source);
        return false;
    }

    bool runIcpFallback(const float initialGuess[6], float resultTransform[6],
                        double& fitnessScore, int& sourceSize, int& targetSize)
    {
        fitnessScore = std::numeric_limits<double>::infinity();
        sourceSize = 0;
        targetSize = 0;

        if (!mappingIcpFallbackEnable)
        {
            RCLCPP_WARN(get_logger(), "[ICP][SKIP] fallback disabled by mappingIcpFallbackEnable=false.");
            return false;
        }

        pcl::PointCloud<PointType>::Ptr sourceCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr targetCloud(new pcl::PointCloud<PointType>());
        if (cloudInfo.cloud_deskewed)
            pcl::copyPointCloud(*cloudInfo.cloud_deskewed, *sourceCloud);

        laserCloudRawFromMap->clear();
        laserCloudRawFromMapDS->clear();
        for (int thisKeyInd : surroundingKeyFrameIndices)
        {
            if (thisKeyInd < 0 ||
                thisKeyInd >= (int)rawCloudKeyFrames.size() ||
                thisKeyInd >= (int)cloudKeyPoses6D->size())
                continue;

            *laserCloudRawFromMap += *transformPointCloud(rawCloudKeyFrames[thisKeyInd],
                                                          &cloudKeyPoses6D->points[thisKeyInd]);
        }

        if (mappingFallbackIcpLeafSize > 0.0)
        {
            pcl::VoxelGrid<PointType> downSizeFilterRawMap;
            downSizeFilterRawMap.setLeafSize(
                mappingFallbackIcpLeafSize,
                mappingFallbackIcpLeafSize,
                mappingFallbackIcpLeafSize);
            downSizeFilterRawMap.setInputCloud(laserCloudRawFromMap);
            downSizeFilterRawMap.filter(*laserCloudRawFromMapDS);
        }
        else
        {
            pcl::copyPointCloud(*laserCloudRawFromMap, *laserCloudRawFromMapDS);
        }

        *targetCloud += *laserCloudRawFromMapDS;

        std::vector<int> indices;
        pcl::PointCloud<PointType>::Ptr sourceClean(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr targetClean(new pcl::PointCloud<PointType>());
        pcl::removeNaNFromPointCloud(*sourceCloud, *sourceClean, indices);
        pcl::removeNaNFromPointCloud(*targetCloud, *targetClean, indices);
        sourceCloud = sourceClean;
        targetCloud = targetClean;

        if (mappingFallbackIcpLeafSize > 0.0)
        {
            pcl::VoxelGrid<PointType> downSizeFilterFallbackIcp;
            pcl::PointCloud<PointType>::Ptr sourceDS(new pcl::PointCloud<PointType>());
            downSizeFilterFallbackIcp.setLeafSize(
                mappingFallbackIcpLeafSize,
                mappingFallbackIcpLeafSize,
                mappingFallbackIcpLeafSize);
            downSizeFilterFallbackIcp.setInputCloud(sourceCloud);
            downSizeFilterFallbackIcp.filter(*sourceDS);
            sourceCloud = sourceDS;
        }

        sourceSize = static_cast<int>(sourceCloud->size());
        targetSize = static_cast<int>(targetCloud->size());

        if (sourceSize < mappingFallbackIcpMinSourcePoints ||
            targetSize < mappingFallbackIcpMinTargetPoints)
        {
            RCLCPP_WARN(get_logger(),
                "[ICP][FAIL] not enough points. source=%d target=%d minSource=%d minTarget=%d",
                sourceSize, targetSize,
                mappingFallbackIcpMinSourcePoints,
                mappingFallbackIcpMinTargetPoints);
            return false;
        }

        if ((mappingFallbackIcpMaxSourcePoints > 0 && sourceSize > mappingFallbackIcpMaxSourcePoints) ||
            (mappingFallbackIcpMaxTargetPoints > 0 && targetSize > mappingFallbackIcpMaxTargetPoints))
        {
            RCLCPP_WARN(get_logger(),
                "[ICP][SKIP] too many points. source=%d target=%d maxSource=%d maxTarget=%d leaf=%.2f",
                sourceSize, targetSize,
                mappingFallbackIcpMaxSourcePoints,
                mappingFallbackIcpMaxTargetPoints,
                mappingFallbackIcpLeafSize);
            return false;
        }

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(mappingFallbackIcpMaxCorrespondenceDistance);
        icp.setMaximumIterations(mappingFallbackIcpMaxIterations);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
        icp.setInputSource(sourceCloud);
        icp.setInputTarget(targetCloud);

        Eigen::Affine3f initialAffine = pcl::getTransformation(
            initialGuess[3], initialGuess[4], initialGuess[5],
            initialGuess[0], initialGuess[1], initialGuess[2]);
        pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
        icp.align(*alignedCloud, initialAffine.matrix());

        if (!icp.hasConverged())
        {
            RCLCPP_WARN(get_logger(),
                "[ICP][FAIL] did not converge. source=%d target=%d",
                sourceSize, targetSize);
            return false;
        }

        fitnessScore = icp.getFitnessScore(mappingFallbackIcpFitnessScoreMaxRange);
        Eigen::Affine3f finalAffine;
        finalAffine = icp.getFinalTransformation();
        pcl::getTranslationAndEulerAngles(
            finalAffine,
            resultTransform[3], resultTransform[4], resultTransform[5],
            resultTransform[0], resultTransform[1], resultTransform[2]);

        return true;
    }

    void performLoopClosure()
    {
        if (loopClosureEnableFlag == false)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
            return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }
    



    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        static bool lastImuTransformationAvailable = false;
        // initialization
        if (cloudKeyPoses3D->points.empty())
        {
            transformTobeMapped[0] = cloudInfo.imu_roll_init;
            transformTobeMapped[1] = cloudInfo.imu_pitch_init;
            transformTobeMapped[2] = cloudInfo.imu_yaw_init;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            lastImuTransformationAvailable = true;
            syncMappingPrediction(transformTobeMapped);
            return;
        }

        Eigen::Affine3f imuRotationIncrement;
        bool hasImuRotationIncrement = false;
        if (cloudInfo.imu_available == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
            if (lastImuTransformationAvailable)
            {
                imuRotationIncrement = lastImuTransformation.inverse() * transBack;
                hasImuRotationIncrement = true;
            }

            lastImuTransformation = transBack;
            lastImuTransformationAvailable = true;
        }

        advanceMappingPrediction(hasImuRotationIncrement, imuRotationIncrement);
        return;
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        surroundingKeyFrameIndices.clear();
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (thisKeyInd < 0 ||
                thisKeyInd >= (int)cornerCloudKeyFrames.size() ||
                thisKeyInd >= (int)surfCloudKeyFrames.size() ||
                thisKeyInd >= (int)rawCloudKeyFrames.size())
                continue;

            surroundingKeyFrameIndices.push_back(thisKeyInd);
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // 0428 新增
        if(laserCloudCornerFromMapDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudCornerFromMap, *laserCloudCornerFromMapDS);
            laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        }
        //  0428 end
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
        //0428 
        if(laserCloudSurfFromMapDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudSurfFromMap, *laserCloudSurfFromMapDS);
            laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        }
        //0428 end

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();
        if(laserCloudCornerLastDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudCornerLast, *laserCloudCornerLastDS);
            laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();
        }


        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        if(laserCloudSurfLastDSNum < 500)
        {
            pcl::copyPointCloud(*laserCloudSurfLast, *laserCloudSurfLastDS);
            laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
        }
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        lastLMCloudSelNum = laserCloudSelNum;
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        float initialGuessTransform[6];
        copyTransform(transformTobeMapped, initialGuessTransform);

        float keepaliveTransform[6];
        if (hasLastTrustedMappingTransform)
            copyTransform(lastTrustedMappingTransform, keepaliveTransform);
        else
            copyTransform(initialGuessTransform, keepaliveTransform);

        if (cloudKeyPoses3D->points.empty())
        {
            acceptMappingPose("INIT");
            return;
        }

        bool lmUsable = false;
        bool lmMotionOk = true;
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            lastLMRan = true;
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                {
                    lastLMConverged = true;
                    lastLMIterationCount = iterCount + 1;
                    break;              
                }

                lastLMIterationCount = iterCount + 1;
            }

            transformUpdate();
            MotionGateResult lmMotion = evaluateMotionGate(transformTobeMapped);
            logMotionGate("LM", lmMotion);
            lmMotionOk = lmMotion.continuous;
            lmUsable = lastLMConverged && lmMotion.continuous && transformIsFinite(transformTobeMapped);

            if (lmUsable)
            {
                acceptOrStageMappingCandidate("LM", transformTobeMapped, keepaliveTransform);
                return;
            }
        } else {
            RCLCPP_WARN(get_logger(), "Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }

        RCLCPP_WARN(get_logger(),
            "[LM][SUSPECT] ran=%d converged=%d degenerate=%d coeff=%d iter=%d motionOK=%d. Evaluate ICP fallback.",
            int(lastLMRan),
            int(lastLMConverged),
            int(isDegenerate),
            lastLMCloudSelNum,
            lastLMIterationCount,
            int(lmMotionOk));

        if (mappingFallbackIcpSkipOnBadLmMotion && lastLMRan && !lmMotionOk)
        {
            RCLCPP_WARN(get_logger(),
                "[ICP][SKIP] LM motion gate rejected the pose. Go directly to failed pose handling.");
            rejectMappingPose(keepaliveTransform, "FAIL_LM_MOTION");
            return;
        }

        float icpTransform[6];
        double icpFitness = std::numeric_limits<double>::infinity();
        int icpSourceSize = 0;
        int icpTargetSize = 0;
        bool icpRan = runIcpFallback(initialGuessTransform, icpTransform, icpFitness, icpSourceSize, icpTargetSize);
        MotionGateResult icpMotion;
        if (icpRan)
        {
            icpMotion = evaluateMotionGate(icpTransform);
            logMotionGate("ICP", icpMotion);
        }

        bool icpUsable = icpRan &&
                         transformIsFinite(icpTransform) &&
                         icpMotion.continuous &&
                         icpFitness < mappingFallbackIcpFitnessScore;

        if (icpUsable)
        {
            copyTransform(icpTransform, transformTobeMapped);
            isDegenerate = false;
            transformUpdate();
            const bool icpAccepted = acceptOrStageMappingCandidate("ICP", transformTobeMapped, keepaliveTransform);
            RCLCPP_WARN(get_logger(),
                "[ICP][%s] fitness=%.6f < %.6f source=%d target=%d",
                icpAccepted ? "OK" : "PENDING",
                icpFitness, mappingFallbackIcpFitnessScore,
                icpSourceSize, icpTargetSize);
            return;
        }

        if (icpRan)
        {
            RCLCPP_WARN(get_logger(),
                "[ICP][FAIL] fitness %.6f >= %.6f or motionOK=%d. source=%d target=%d",
                icpFitness, mappingFallbackIcpFitnessScore,
                int(icpMotion.continuous),
                icpSourceSize, icpTargetSize);
        }

        rejectMappingPose(keepaliveTransform, "FAIL_LM_ICP");
    }

    void transformUpdate()
    {
        if (cloudInfo.imu_available == true)
        {
            if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }


    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise =
                noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (!mappingPoseReliable  || !transformIsFinite(transformTobeMapped)) //  || isDegenerate
        {
            RCLCPP_WARN(get_logger(),
                "[KEYFRAME_SKIP_FINAL] trackingAccepted=%d source=%s degenerate=%d finite=%d. "
                "Skip keyframe and pose factor only.",
                int(mappingPoseReliable),
                mappingPoseSource.c_str(),
                int(isDegenerate),
                int(transformIsFinite(transformTobeMapped)));
            return;
        }

        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisRawKeyFrameRaw(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisRawKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        if (cloudInfo.cloud_deskewed)
            pcl::copyPointCloud(*cloudInfo.cloud_deskewed, *thisRawKeyFrameRaw);
        std::vector<int> rawCloudIndices;
        pcl::removeNaNFromPointCloud(*thisRawKeyFrameRaw, *thisRawKeyFrame, rawCloudIndices);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        rawCloudKeyFrames.push_back(thisRawKeyFrame);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
            }

            aLoopIsClosed = false;
            motionGateHasLast = false;
            lowSpeedGuessHasPrevTrusted = false;
            lowSpeedGuessHasLastTrusted = false;
            hasLastTrustedMappingTransform = false;
            mappingPredictedPoseAvailable = false;
            mappingPredictedPoseTime = -1.0;
            pendingRecoveryAvailable = false;
            pendingRecoverySource = "NONE";
            mappingTrackingState = MappingTrackingState::TRACKING;
            mappingFailureCount = 0;
            mappingFirstFailureTime = -1.0;
            surroundingKeyFrameIndices.clear();
        }
    }

    void updateOdometryState()
    {
        int correctionFlag = lidarCorrectionFlag;

        static bool lastIncreOdomPubFlag = false;
        static LioSamOdometryState laserOdomIncremental; // incremental odometry state
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental.timestamp = timeLaserInfoCur;
            laserOdomIncremental.x = transformTobeMapped[3];
            laserOdomIncremental.y = transformTobeMapped[4];
            laserOdomIncremental.z = transformTobeMapped[5];
            laserOdomIncremental.roll = transformTobeMapped[0];
            laserOdomIncremental.pitch = transformTobeMapped[1];
            laserOdomIncremental.yaw = transformTobeMapped[2];
            laserOdomIncremental.correction_flag = correctionFlag;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imu_available == true)
            {
                if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf2::Quaternion imuQuaternion;
                    tf2::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.timestamp = timeLaserInfoCur;
            laserOdomIncremental.x = x;
            laserOdomIncremental.y = y;
            laserOdomIncremental.z = z;
            laserOdomIncremental.roll = roll;
            laserOdomIncremental.pitch = pitch;
            laserOdomIncremental.yaw = yaw;
            laserOdomIncremental.correction_flag = correctionFlag;
        }
        latestLaserOdometryIncremental = laserOdomIncremental;
        hasLatestLaserOdometryIncremental = true;

        if (mappingPoseReliable &&
            transformIsFinite(transformTobeMapped))
        {
            copyTransform(transformTobeMapped, lastTrustedMappingTransform);
            hasLastTrustedMappingTransform = true;
            commitLowSpeedTrustedPose(transformTobeMapped);
            commitMotionGateReference(transformTobeMapped);
            syncMappingPrediction(transformTobeMapped);
        }
    }

};

