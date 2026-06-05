//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include "core/localization/localization.h"
#include "io/yaml_io.h"
#include "wrapper/ros_utils.h"
#include <iomanip>
namespace lightning {

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() { loc_->Finish(); }

bool LocSystem::Init(const std::string &yaml_path) {
    loc::Localization::Options opt;
    opt.online_mode_ = true;
    loc_ = std::make_shared<loc::Localization>(opt);

    YAML_IO yaml(yaml_path);

    std::string map_path = yaml.GetValue<std::string>("system", "map_path");

    LOG(INFO) << "online mode, creating ros2 node ... ";

    /// subscribers
    node_ = std::make_shared<rclcpp::Node>("lightning_slam");

    imu_topic_ = yaml.GetValue<std::string>("common", "imu_topic");
    cloud_topic_ = yaml.GetValue<std::string>("common", "lidar_topic");
    livox_topic_ = yaml.GetValue<std::string>("common", "livox_lidar_topic");

    auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(2000));
    imu_qos.best_effort();
    imu_qos.durability_volatile();

    auto lidar_qos = rclcpp::QoS(rclcpp::KeepLast(10));
    lidar_qos.best_effort();
    lidar_qos.durability_volatile();
    // 在线定位模式下，稳定发布位姿
    imu_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    lidar_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    pub_timer_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions imu_sub_options;
    imu_sub_options.callback_group = imu_cb_group_;

    rclcpp::SubscriptionOptions lidar_sub_options;
    lidar_sub_options.callback_group = lidar_cb_group_;

    using namespace std::chrono_literals;

    loc_pub_timer_ = node_->create_wall_timer(
        100ms,
        [this]() {
            if (loc_started_ && loc_) {
                loc_->PublishLatestResult();
            }
        },
        pub_timer_cb_group_);

    imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, imu_qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
            IMUPtr imu = std::make_shared<IMU>();
            imu->timestamp = ToSec(msg->header.stamp);
            imu->linear_acceleration =
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            imu->angular_velocity =
                Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
            imu->orientation =
                Quatd(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

            static int imu_count = 0;
            if (++imu_count % 200 == 0) {
                LOG(INFO) << "[IMU_RECV] t=" << std::setprecision(14) << imu->timestamp;
            }

            ProcessIMU(imu);
        },
        imu_sub_options);
        
    cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, lidar_qos,
        [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        },
        lidar_sub_options);

    livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        livox_topic_, lidar_qos,
        [this](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        },
        lidar_sub_options);
        
    // 发布定位结果
    loc_odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>(
        "/lightning/localization/odom", 10);

    loc_pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/lightning/localization/pose", 10);    

    if (options_.pub_tf_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }

    loc_->SetTFCallback([this](const geometry_msgs::msg::TransformStamped& tf_msg) {
        if (options_.pub_tf_ && tf_broadcaster_) {
            tf_broadcaster_->sendTransform(tf_msg);
        }

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = tf_msg.header;
        pose_msg.pose.position.x = tf_msg.transform.translation.x;
        pose_msg.pose.position.y = tf_msg.transform.translation.y;
        pose_msg.pose.position.z = tf_msg.transform.translation.z;
        pose_msg.pose.orientation = tf_msg.transform.rotation;

        if (loc_pose_pub_) {
            loc_pose_pub_->publish(pose_msg);
        }

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header = tf_msg.header;
        odom_msg.child_frame_id = tf_msg.child_frame_id;
        odom_msg.pose.pose = pose_msg.pose;

        if (loc_odom_pub_) {
            loc_odom_pub_->publish(odom_msg);
        }
    });
    
    bool ret = loc_->Init(yaml_path, map_path);
    if (ret) {
        LOG(INFO) << "online loc node has been created.";
    }

    return ret;
}

void LocSystem::SetInitPose(const SE3 &pose) {
    LOG(INFO) << "set init pose: " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();

    loc_->SetExternalPose(pose.unit_quaternion(), pose.translation());
    loc_started_ = true;
}

void LocSystem::ProcessIMU(const IMUPtr &imu) {
    if (loc_started_) {
        loc_->ProcessIMUMsg(imu);
    }
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud) {
    if (loc_started_) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr &cloud) {
    if (loc_started_) {
        loc_->ProcessLivoxLidarMsg(cloud);
    }
}
/*
void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}
*/
void LocSystem::Spin() {
    if (node_ == nullptr) {
        return;
    }

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 4);

    executor.add_node(node_);
    executor.spin();
}
}  // namespace lightning