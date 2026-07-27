//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include "core/localization/localization.h"
#include "io/yaml_io.h"
#include "wrapper/ros_utils.h"

#include <cmath>

namespace lightning {

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() {
    if (loc_) {
        loc_->Finish();
    }
}

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

    rclcpp::SensorDataQoS qos;

    imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
            IMUPtr imu = std::make_shared<IMU>();
            imu->timestamp = ToSec(msg->header.stamp);
            imu->linear_acceleration =
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            imu->angular_velocity = Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

            ProcessIMU(imu);
        });

    cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        });

    livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg ::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        });

    odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    localization_pose_pub_ =
        node_->create_publisher<geometry_msgs::msg::PoseStamped>("/lightning/localization_pose", 10);
    localization_status_pub_ =
        node_->create_publisher<std_msgs::msg::UInt8>("/lightning/localization_status", 10);
    initial_pose_sub_ = node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10, [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
            if (!msg->header.frame_id.empty() && msg->header.frame_id != "map") {
                RCLCPP_WARN(node_->get_logger(), "Ignoring /initialpose in frame '%s'; expected map",
                            msg->header.frame_id.c_str());
                return;
            }
            const auto& p = msg->pose.pose.position;
            const auto& q = msg->pose.pose.orientation;
            const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(norm) ||
                norm < 1e-6) {
                RCLCPP_WARN(node_->get_logger(), "Ignoring invalid /initialpose");
                return;
            }
            SetInitPose(SE3(Eigen::Quaterniond(q.w / norm, q.x / norm, q.y / norm, q.z / norm),
                            Vec3d(p.x, p.y, p.z)));
        });

    if (options_.pub_tf_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }
    loc_->SetResultCallback([this](const loc::LocalizationResult& result) {
        if (tf_broadcaster_) {
            tf_broadcaster_->sendTransform(result.ToOdomBaseMsg());
            if (result.valid_) {
                tf_broadcaster_->sendTransform(result.ToMapOdomMsg());
            }
        }
        odom_pub_->publish(result.ToOdomMsg());
        if (result.valid_) {
            localization_pose_pub_->publish(result.ToPoseMsg());
        }
        std_msgs::msg::UInt8 status;
        status.data = static_cast<uint8_t>(result.status_);
        localization_status_pub_->publish(status);
    });

    bool ret = loc_->Init(yaml_path, map_path);
    if (ret) {
        loc_started_ = true;
        LOG(INFO) << "online loc node has been created.";
    }

    return ret;
}

void LocSystem::SetInitPose(const SE3 &pose) {
    LOG(INFO) << "set init pose: " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();

    std_msgs::msg::UInt8 status;
    status.data = static_cast<uint8_t>(loc::LocalizationStatus::INITIALIZING);
    localization_status_pub_->publish(status);
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

void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}

}  // namespace lightning
