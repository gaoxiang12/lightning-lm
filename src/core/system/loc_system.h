//
// Created by xiang on 25-9-8.
//

#ifndef LIGHTNING_LOC_SYSTEM_H
#define LIGHTNING_LOC_SYSTEM_H

#include <deque>
#include <mutex>
#include <atomic>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"

#include "core/localization/localization_result.h"
#include "core/system/async_message_process.h"

namespace lightning {

namespace loc {
class Localization;
struct DualLidarCalibrationResult;
}

class LocSystem {
   public:
    struct Options {
        bool pub_tf_ = true;  // 是否发布tf
    };

    explicit LocSystem(Options options);
    ~LocSystem();

    /// 初始化，地图路径在yaml里配置
    bool Init(const std::string& yaml_path);

    /// 设置初始化位姿
    void SetInitPose(const SE3& pose);

    /// 处理IMU
    void ProcessIMU(const lightning::IMUPtr& imu);

    /// 处理点云
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    /// 实时模式下的spin
    void Spin();

    /// 发布base_link的TF
    void PublishBaseLinkTF(const lightning::loc::LocalizationResult& res);

    /// 发布双雷达标定 TF：front_lidar -> rear_lidar
    void PublishDualLidarCalibrationTF(const lightning::loc::DualLidarCalibrationResult& res);

   private:
    struct TimedCloud {
        double timestamp = 0.0;
        sensor_msgs::msg::PointCloud2::SharedPtr cloud = nullptr;
    };

    struct TimedCloudPair {
        sensor_msgs::msg::PointCloud2::SharedPtr front_cloud = nullptr;
        sensor_msgs::msg::PointCloud2::SharedPtr rear_cloud = nullptr;
    };

    void ProcessFrontLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessRearLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void TryProcessDualLidarPair();

    Options options_;

    std::shared_ptr<loc::Localization> loc_ = nullptr;  // 定位接口

    std::atomic_bool loc_started_ = false;  // 是否开启定位
    std::atomic_bool map_loaded_ = false;   // 地图是否已载入

    /// 实时模式下的ros2 node, subscribers
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_ = nullptr;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_ = nullptr;

    std::string imu_topic_;
    std::string cloud_topic_;
    std::string livox_topic_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;

    /// 双雷达在线标定旁路，不影响原定位分支
    bool dual_lidar_options_valid_ = false;
    std::string front_lidar_topic_;
    std::string rear_lidar_topic_;
    bool publish_dual_lidar_tf_ = false;
    double dual_lidar_sync_tolerance_ = 0.02;
    size_t dual_lidar_max_queue_size_ = 20;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr front_lidar_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr rear_lidar_sub_ = nullptr;

    std::mutex dual_lidar_mutex_;
    std::deque<TimedCloud> front_lidar_queue_;
    std::deque<TimedCloud> rear_lidar_queue_;
    sys::AsyncMessageProcess<TimedCloudPair> dual_lidar_pair_proc_;
};

}  // namespace lightning

#endif  // LIGHTNING_LOC_SYSTEM_H