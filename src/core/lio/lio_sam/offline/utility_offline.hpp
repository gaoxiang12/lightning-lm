#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>
#include <deque>
#include <mutex>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/kdtree/kdtree_flann.h>

namespace flann {
namespace serialization {

template <typename Key, typename Value, typename Hash, typename Eq, typename Alloc>
struct Serializer<std::unordered_map<Key, Value, Hash, Eq, Alloc>> {
    template <typename InputArchive>
    static inline void load(InputArchive& ar,
                            std::unordered_map<Key, Value, Hash, Eq, Alloc>& map)
    {
        size_t size = 0;
        ar & size;
        map.clear();

        for (size_t i = 0; i < size; ++i) {
            Key key;
            Value value;
            ar & key;
            ar & value;
            map.emplace(std::move(key), std::move(value));
        }
    }

    template <typename OutputArchive>
    static inline void save(OutputArchive& ar,
                            const std::unordered_map<Key, Value, Hash, Eq, Alloc>& map)
    {
        size_t size = map.size();
        ar & size;

        for (const auto& kv : map) {
            Key key = kv.first;
            Value value = kv.second;
            ar & key;
            ar & value;
        }
    }
};

}  // namespace serialization
}  // namespace flann

#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std;

typedef pcl::PointXYZI PointType;

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x) (float, y, y)
                                  (float, z, z) (float, intensity, intensity)
                                  (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                  (double, time, time))

typedef PointXYZIRPYT PointTypePose;

struct LioSamCloudInfo
{
    double timestamp = 0.0;

    bool imu_available = false;
    float imu_roll_init = 0.0f;
    float imu_pitch_init = 0.0f;
    float imu_yaw_init = 0.0f;

    std::vector<int> start_ring_index;
    std::vector<int> end_ring_index;
    std::vector<int> point_col_ind;
    std::vector<float> point_range;

    pcl::PointCloud<PointType>::Ptr cloud_deskewed{new pcl::PointCloud<PointType>()};
    pcl::PointCloud<PointType>::Ptr cloud_corner{new pcl::PointCloud<PointType>()};
    pcl::PointCloud<PointType>::Ptr cloud_surface{new pcl::PointCloud<PointType>()};
};

enum class SensorType { VELODYNE, OUSTER, LIVOX };

class ParamServer : public rclcpp::Node
{
public:
    std::string robot_id;

    string lioMode;

    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // IMU initialization
    bool useImuHeadingInitialization;

    // Lidar Sensor Configuration
    SensorType sensor = SensorType::OUSTER;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    bool useImuAccelRollPitchInitialization;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;
    float mappingLowSpeedMaxTranslationSpeed;


    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    // Loop closure
    bool  loopClosureEnableFlag;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // Mapping robustness
    bool mappingMotionGateEnable;
    bool mappingIcpFallbackEnable;
    float mappingMotionMaxSpeed;
    float mappingMotionMaxAngularVelocity;
    float mappingMotionMaxCurvature;
    float mappingMotionMaxRollPitchDeg;
    bool mappingFallbackIcpSkipOnBadLmMotion;
    float mappingFallbackIcpMaxCorrespondenceDistance;
    int mappingFallbackIcpMaxIterations;
    float mappingFallbackIcpLeafSize;
    float mappingFallbackIcpFitnessScore;
    float mappingFallbackIcpFitnessScoreMaxRange;
    int mappingFallbackIcpMinSourcePoints;
    int mappingFallbackIcpMinTargetPoints;
    int mappingFallbackIcpMaxSourcePoints;
    int mappingFallbackIcpMaxTargetPoints;

    ParamServer(std::string node_name, const rclcpp::NodeOptions & options) : Node(node_name, options)
    {
        declare_parameter("lioMode", "mapping");
        get_parameter("lioMode", lioMode);
        std::transform(lioMode.begin(), lioMode.end(), lioMode.begin(), ::tolower);

        declare_parameter("lidarFrame", "laser_data_frame");
        get_parameter("lidarFrame", lidarFrame);
        declare_parameter("baselinkFrame", "base_link");
        get_parameter("baselinkFrame", baselinkFrame);
        declare_parameter("odometryFrame", "odom");
        get_parameter("odometryFrame", odometryFrame);
        declare_parameter("mapFrame", "map");
        get_parameter("mapFrame", mapFrame);

        declare_parameter("useImuHeadingInitialization", false);
        get_parameter("useImuHeadingInitialization", useImuHeadingInitialization);

        std::string sensorStr;
        declare_parameter("sensor", "ouster");
        get_parameter("sensor", sensorStr);
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else
        {
            RCLCPP_ERROR_STREAM(
                get_logger(),
                "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'livox'): " << sensorStr);
            rclcpp::shutdown();
        }

        declare_parameter("N_SCAN", 64);
        get_parameter("N_SCAN", N_SCAN);
        declare_parameter("Horizon_SCAN", 512);
        get_parameter("Horizon_SCAN", Horizon_SCAN);
        declare_parameter("downsampleRate", 1);
        get_parameter("downsampleRate", downsampleRate);
        declare_parameter("lidarMinRange", 5.5);
        get_parameter("lidarMinRange", lidarMinRange);
        declare_parameter("lidarMaxRange", 1000.0);
        get_parameter("lidarMaxRange", lidarMaxRange);

        declare_parameter("imuAccNoise", 9e-4);
        get_parameter("imuAccNoise", imuAccNoise);
        declare_parameter("imuGyrNoise", 1.6e-4);
        get_parameter("imuGyrNoise", imuGyrNoise);
        declare_parameter("imuAccBiasN", 5e-4);
        get_parameter("imuAccBiasN", imuAccBiasN);
        declare_parameter("imuGyrBiasN", 7e-5);
        get_parameter("imuGyrBiasN", imuGyrBiasN);
        declare_parameter("imuGravity", 9.80511);
        get_parameter("imuGravity", imuGravity);
        declare_parameter("imuRPYWeight", 0.01);
        get_parameter("imuRPYWeight", imuRPYWeight);
        declare_parameter("useImuAccelRollPitchInitialization", true);
        get_parameter("useImuAccelRollPitchInitialization", useImuAccelRollPitchInitialization);

        double ida[] = { 1.0,  0.0,  0.0,
                         0.0,  1.0,  0.0,
                         0.0,  0.0,  1.0};
        std::vector < double > id(ida, std::end(ida));
        declare_parameter("extrinsicRot", id);
        get_parameter("extrinsicRot", extRotV);
        declare_parameter("extrinsicRPY", id);
        get_parameter("extrinsicRPY", extRPYV);
        double zea[] = {0.0, 0.0, 0.0};
        std::vector < double > ze(zea, std::end(zea));
        declare_parameter("extrinsicTrans", ze);
        get_parameter("extrinsicTrans", extTransV);

        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        declare_parameter("edgeThreshold", 1.0);
        get_parameter("edgeThreshold", edgeThreshold);
        declare_parameter("surfThreshold", 0.1);
        get_parameter("surfThreshold", surfThreshold);
        declare_parameter("edgeFeatureMinValidNum", 10);
        get_parameter("edgeFeatureMinValidNum", edgeFeatureMinValidNum);
        declare_parameter("surfFeatureMinValidNum", 100);
        get_parameter("surfFeatureMinValidNum", surfFeatureMinValidNum);

        declare_parameter("odometrySurfLeafSize", 0.4);
        get_parameter("odometrySurfLeafSize", odometrySurfLeafSize);
        declare_parameter("mappingCornerLeafSize", 0.2);
        get_parameter("mappingCornerLeafSize", mappingCornerLeafSize);
        declare_parameter("mappingSurfLeafSize", 0.4);
        get_parameter("mappingSurfLeafSize", mappingSurfLeafSize);

        declare_parameter("z_tollerance", 1000.0);
        get_parameter("z_tollerance", z_tollerance);
        declare_parameter("rotation_tollerance", 1000.0);
        get_parameter("rotation_tollerance", rotation_tollerance);

        declare_parameter("numberOfCores", 4);
        get_parameter("numberOfCores", numberOfCores);
        declare_parameter("mappingProcessInterval", 0.15);
        get_parameter("mappingProcessInterval", mappingProcessInterval);
        declare_parameter("mappingLowSpeedMaxTranslationSpeed", 0.8);
        get_parameter("mappingLowSpeedMaxTranslationSpeed", mappingLowSpeedMaxTranslationSpeed);

        declare_parameter("surroundingkeyframeAddingDistThreshold", 1.0);
        get_parameter("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold);
        declare_parameter("surroundingkeyframeAddingAngleThreshold", 0.2);
        get_parameter("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold);
        declare_parameter("surroundingKeyframeDensity", 2.0);
        get_parameter("surroundingKeyframeDensity", surroundingKeyframeDensity);
        declare_parameter("surroundingKeyframeSearchRadius", 50.0);
        get_parameter("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius);

        declare_parameter("loopClosureEnableFlag", true);
        get_parameter("loopClosureEnableFlag", loopClosureEnableFlag);
        declare_parameter("surroundingKeyframeSize", 50);
        get_parameter("surroundingKeyframeSize", surroundingKeyframeSize);
        declare_parameter("historyKeyframeSearchRadius", 15.0);
        get_parameter("historyKeyframeSearchRadius", historyKeyframeSearchRadius);
        declare_parameter("historyKeyframeSearchTimeDiff", 30.0);
        get_parameter("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff);
        declare_parameter("historyKeyframeSearchNum", 25);
        get_parameter("historyKeyframeSearchNum", historyKeyframeSearchNum);
        declare_parameter("historyKeyframeFitnessScore", 0.3);
        get_parameter("historyKeyframeFitnessScore", historyKeyframeFitnessScore);

        declare_parameter("mappingMotionGateEnable", true);
        get_parameter("mappingMotionGateEnable", mappingMotionGateEnable);
        declare_parameter("mappingIcpFallbackEnable", false);
        get_parameter("mappingIcpFallbackEnable", mappingIcpFallbackEnable);
        declare_parameter("mappingMotionMaxSpeed", 3.0);
        get_parameter("mappingMotionMaxSpeed", mappingMotionMaxSpeed);
        declare_parameter("mappingMotionMaxAngularVelocity", 90.0);
        get_parameter("mappingMotionMaxAngularVelocity", mappingMotionMaxAngularVelocity);
        declare_parameter("mappingMotionMaxCurvature", 2.0);
        get_parameter("mappingMotionMaxCurvature", mappingMotionMaxCurvature);
        declare_parameter("mappingMotionMaxRollPitchDeg", 20.0);
        get_parameter("mappingMotionMaxRollPitchDeg", mappingMotionMaxRollPitchDeg);
        declare_parameter("mappingFallbackIcpSkipOnBadLmMotion", true);
        get_parameter("mappingFallbackIcpSkipOnBadLmMotion", mappingFallbackIcpSkipOnBadLmMotion);
        declare_parameter("mappingFallbackIcpMaxCorrespondenceDistance", 2.0);
        get_parameter("mappingFallbackIcpMaxCorrespondenceDistance", mappingFallbackIcpMaxCorrespondenceDistance);
        declare_parameter("mappingFallbackIcpMaxIterations", 10);
        get_parameter("mappingFallbackIcpMaxIterations", mappingFallbackIcpMaxIterations);
        declare_parameter("mappingFallbackIcpLeafSize", 0.80);
        get_parameter("mappingFallbackIcpLeafSize", mappingFallbackIcpLeafSize);
        declare_parameter("mappingFallbackIcpFitnessScore", 0.14);
        get_parameter("mappingFallbackIcpFitnessScore", mappingFallbackIcpFitnessScore);
        declare_parameter("mappingFallbackIcpFitnessScoreMaxRange", 2.0);
        get_parameter("mappingFallbackIcpFitnessScoreMaxRange", mappingFallbackIcpFitnessScoreMaxRange);
        declare_parameter("mappingFallbackIcpMinSourcePoints", 300);
        get_parameter("mappingFallbackIcpMinSourcePoints", mappingFallbackIcpMinSourcePoints);
        declare_parameter("mappingFallbackIcpMinTargetPoints", 1000);
        get_parameter("mappingFallbackIcpMinTargetPoints", mappingFallbackIcpMinTargetPoints);
        declare_parameter("mappingFallbackIcpMaxSourcePoints", 8000);
        get_parameter("mappingFallbackIcpMaxSourcePoints", mappingFallbackIcpMaxSourcePoints);
        declare_parameter("mappingFallbackIcpMaxTargetPoints", 30000);
        get_parameter("mappingFallbackIcpMaxTargetPoints", mappingFallbackIcpMaxTargetPoints);

        usleep(100);
    }

    sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu& imu_in)
    {
        sensor_msgs::msg::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        //Eigen::Quaterniond q_final = extQRPY ; // 0428
        Eigen::Quaterniond q_final = q_from * extQRPY ; 
        q_final.normalize(); //0428

        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        static int imuConverterDebugCount = 0;
        if (imuConverterDebugCount < 5)
        {
            double rawRoll, rawPitch, rawYaw;
            double outRoll, outPitch, outYaw;
            tf2::Quaternion rawOrientation;
            tf2::Quaternion outOrientation;
            tf2::fromMsg(imu_in.orientation, rawOrientation);
            tf2::fromMsg(imu_out.orientation, outOrientation);
            tf2::Matrix3x3(rawOrientation).getRPY(rawRoll, rawPitch, rawYaw);
            tf2::Matrix3x3(outOrientation).getRPY(outRoll, outPitch, outYaw);

            RCLCPP_WARN(get_logger(),
                "[IMU_CONVERTER] raw_rpy=(%.2f %.2f %.2f) out_rpy=(%.2f %.2f %.2f) "
                "raw_acc=(%.3f %.3f %.3f) out_acc=(%.3f %.3f %.3f)",
                rawRoll * 180.0 / M_PI,
                rawPitch * 180.0 / M_PI,
                rawYaw * 180.0 / M_PI,
                outRoll * 180.0 / M_PI,
                outPitch * 180.0 / M_PI,
                outYaw * 180.0 / M_PI,
                imu_in.linear_acceleration.x,
                imu_in.linear_acceleration.y,
                imu_in.linear_acceleration.z,
                imu_out.linear_acceleration.x,
                imu_out.linear_acceleration.y,
                imu_out.linear_acceleration.z);
            imuConverterDebugCount++;
        }

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            RCLCPP_ERROR(get_logger(), "Invalid quaternion, please use a 9-axis IMU!");
            rclcpp::shutdown();
        }

        return imu_out;
    }
};


template<typename T>
double stamp2Sec(const T& stamp)
{
    return rclcpp::Time(stamp).seconds();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::msg::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::msg::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf2::Quaternion orientation;
    tf2::fromMsg(thisImuMsg->orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

template<typename T>
bool imuAccel2rosRollPitch(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch)
{
    double ax = thisImuMsg->linear_acceleration.x;
    double ay = thisImuMsg->linear_acceleration.y;
    double az = thisImuMsg->linear_acceleration.z;
    double accNorm = std::sqrt(ax * ax + ay * ay + az * az);

    if (!std::isfinite(accNorm) || accNorm < 1.0)
        return false;

    *rosRoll = std::atan2(ay, az);
    *rosPitch = std::atan2(-ax, std::sqrt(ay * ay + az * az));
    return true;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

rmw_qos_profile_t qos_profile{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    1,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile.history,
        qos_profile.depth
    ),
    qos_profile);

rmw_qos_profile_t qos_profile_imu{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    2000,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_imu = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_imu.history,
        qos_profile_imu.depth
    ),
    qos_profile_imu);

rmw_qos_profile_t qos_profile_lidar{
    RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    5,
    RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    RMW_QOS_POLICY_DURABILITY_VOLATILE,
    RMW_QOS_DEADLINE_DEFAULT,
    RMW_QOS_LIFESPAN_DEFAULT,
    RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
    RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
    false
};

auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
        qos_profile_lidar.history,
        qos_profile_lidar.depth
    ),
    qos_profile_lidar);

#endif
