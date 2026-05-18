//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include "core/localization/dual_lidar_online_calibration.h"
#include "core/localization/localization.h"
#include "wrapper/ros_utils.h"
#include <algorithm>
#include <cmath>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>

namespace lightning {

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() {
    dual_lidar_pair_proc_.Quit();
    if (loc_) {
        loc_->Finish();
    }
}

bool LocSystem::Init(const std::string &yaml_path) {
    loc::Localization::Options opt;
    opt.online_mode_ = true;
    loc_ = std::make_shared<loc::Localization>(opt);

    LOG(INFO) << "online mode, creating ros2 node ... ";

    node_ = std::make_shared<rclcpp::Node>("lightning_slam");

    YAML::Node yaml_node;
    try {
        yaml_node = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to load yaml " << yaml_path << ": " << e.what();
        return false;
    }

    std::string map_path;
    if (yaml_node["system"] && yaml_node["system"]["map_path"]) {
        map_path = yaml_node["system"]["map_path"].as<std::string>();
    }

    imu_topic_ = yaml_node["common"]["imu_topic"].as<std::string>();
    cloud_topic_ = yaml_node["common"]["lidar_topic"].as<std::string>();
    livox_topic_ = yaml_node["common"]["livox_lidar_topic"].as<std::string>();

    rclcpp::QoS qos(10);

    bool ret = loc_->Init(yaml_path, map_path);
    if (!ret) {
        return false;
    }

    const bool runs_localization = loc_->RunsLocalization();
    const bool runs_dual_lidar_calibration = loc_->RunsDualLidarCalibration();

    if (runs_dual_lidar_calibration) {
        YAML::Node dual_lidar_cfg = yaml_node["dual_lidar_online_calibration"];
        if (!dual_lidar_cfg && yaml_node["localization"]) {
            dual_lidar_cfg = yaml_node["localization"]["dual_lidar_online_calibration"];
        }

        if (dual_lidar_cfg) {
            if (dual_lidar_cfg["front_lidar_topic"]) {
                front_lidar_topic_ = dual_lidar_cfg["front_lidar_topic"].as<std::string>();
            }
            if (dual_lidar_cfg["rear_lidar_topic"]) {
                rear_lidar_topic_ = dual_lidar_cfg["rear_lidar_topic"].as<std::string>();
            }
            if (dual_lidar_cfg["sync_tolerance"]) {
                dual_lidar_sync_tolerance_ = dual_lidar_cfg["sync_tolerance"].as<double>();
            }
            if (dual_lidar_cfg["max_queue_size"]) {
                dual_lidar_max_queue_size_ =
                    static_cast<size_t>(std::max(1, dual_lidar_cfg["max_queue_size"].as<int>()));
            }
            if (dual_lidar_cfg["publish_tf"]) {
                publish_dual_lidar_tf_ = dual_lidar_cfg["publish_tf"].as<bool>();
            }
        }
    }

    if (options_.pub_tf_ && (runs_localization || (runs_dual_lidar_calibration && publish_dual_lidar_tf_))) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
        if (runs_localization) {
            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        }
    }

    if (runs_localization && options_.pub_tf_) {
        loc_->SetTFCallback([this](const lightning::loc::LocalizationResult& pose) { PublishBaseLinkTF(pose); });
    }

    if (runs_dual_lidar_calibration && publish_dual_lidar_tf_) {
        loc_->SetDualLidarCalibrationCallback(
            [this](const lightning::loc::DualLidarCalibrationResult& res) { PublishDualLidarCalibrationTF(res); });
    }

    if (runs_dual_lidar_calibration) {
        dual_lidar_pair_proc_.SetName("dual lidar online calibration");
        dual_lidar_pair_proc_.SetMaxSize(1);
        dual_lidar_pair_proc_.SetProcFunc([this](const TimedCloudPair& pair) {
            if (pair.front_cloud && pair.rear_cloud && loc_) {
                loc_->ProcessDualLidarPointCloudPair(pair.front_cloud, pair.rear_cloud);
            }
        });
        dual_lidar_pair_proc_.Start();
    }

    if (runs_localization) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
                IMUPtr imu = std::make_shared<IMU>();
                imu->timestamp = ToSec(msg->header.stamp);
                imu->linear_acceleration =
                    Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
                imu->angular_velocity =
                    Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

                ProcessIMU(imu);
            });

        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });

        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });
    }

    if (runs_dual_lidar_calibration) {
        front_lidar_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            front_lidar_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessFrontLidar(cloud); }, "Dual Front Lidar", true);
            });

        rear_lidar_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            rear_lidar_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessRearLidar(cloud); }, "Dual Rear Lidar", true);
            });
    }

    LOG(INFO) << "online loc node has been created. run_localization=" << runs_localization
              << ", run_dual_lidar_calibration=" << runs_dual_lidar_calibration;
    return true;
}

void LocSystem::SetInitPose(const SE3 &pose) {
    if (!loc_ || !loc_->RunsLocalization()) {
        return;
    }

    LOG(INFO) << "set init pose: " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();

    loc_->SetExternalPose(pose.unit_quaternion(), pose.translation());
    loc_started_ = true;
}

void LocSystem::ProcessIMU(const IMUPtr &imu) {
    if (loc_ && loc_->RunsLocalization() && loc_started_) {
        loc_->ProcessIMUMsg(imu);
    }
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud) {
    if (loc_ && loc_->RunsLocalization() && loc_started_) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr &cloud) {
    if (loc_ && loc_->RunsLocalization() && loc_started_) {
        loc_->ProcessLivoxLidarMsg(cloud);
    }
}

void LocSystem::ProcessFrontLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (!loc_ || !loc_->RunsDualLidarCalibration() || !cloud) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(dual_lidar_mutex_);
        front_lidar_queue_.push_back({ToSec(cloud->header.stamp), cloud});
        while (front_lidar_queue_.size() > dual_lidar_max_queue_size_) {
            front_lidar_queue_.pop_front();
        }
    }
    TryProcessDualLidarPair();
}

void LocSystem::ProcessRearLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (!loc_ || !loc_->RunsDualLidarCalibration() || !cloud) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(dual_lidar_mutex_);
        rear_lidar_queue_.push_back({ToSec(cloud->header.stamp), cloud});
        while (rear_lidar_queue_.size() > dual_lidar_max_queue_size_) {
            rear_lidar_queue_.pop_front();
        }
    }
    TryProcessDualLidarPair();
}

void LocSystem::TryProcessDualLidarPair() {
    sensor_msgs::msg::PointCloud2::SharedPtr front_cloud = nullptr;
    sensor_msgs::msg::PointCloud2::SharedPtr rear_cloud = nullptr;

    {
        std::lock_guard<std::mutex> lock(dual_lidar_mutex_);
        while (!front_lidar_queue_.empty() && !rear_lidar_queue_.empty()) {
            const double dt = front_lidar_queue_.front().timestamp - rear_lidar_queue_.front().timestamp;
            if (std::abs(dt) <= dual_lidar_sync_tolerance_) {
                front_cloud = front_lidar_queue_.front().cloud;
                rear_cloud = rear_lidar_queue_.front().cloud;
                front_lidar_queue_.pop_front();
                rear_lidar_queue_.pop_front();
                break;
            }

            if (dt < 0.0) {
                front_lidar_queue_.pop_front();
            } else {
                rear_lidar_queue_.pop_front();
            }
        }
    }

    if (front_cloud && rear_cloud) {
        dual_lidar_pair_proc_.AddMessage({front_cloud, rear_cloud});
    }
}

void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}

void LocSystem::PublishBaseLinkTF(const lightning::loc::LocalizationResult& res) {
    if (!tf_broadcaster_ || !tf_buffer_) {
        return;
    }

    geometry_msgs::msg::TransformStamped tf_imu_base;

    try {
        // lookupTransform(target_frame, source_frame, time) - 返回 source_frame 到 target_frame 的变换
        // 我们需要 imu_link 到 base_link 的变换，所以 target_frame 是 base_link，source_frame 是 imu_link
        tf_imu_base = tf_buffer_->lookupTransform(
            "base_link",
            "imu_link",
            tf2::TimePointZero
        );
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(node_->get_logger(), "lookup imu_link->base_link failed: %s", ex.what());
        return;
    }

    tf2::Transform T_map_imu, T_imu_base, T_map_base;

    // res.pose_ 视为 map->imu_link
    T_map_imu.setOrigin(tf2::Vector3(
        res.pose_.translation().x(),
        res.pose_.translation().y(),
        res.pose_.translation().z()));

    const auto q = res.pose_.so3().unit_quaternion();
    T_map_imu.setRotation(tf2::Quaternion(q.x(), q.y(), q.z(), q.w()));

    tf2::fromMsg(tf_imu_base.transform, T_imu_base);

    // T_map_base = T_map_imu * T_imu_base
    // 因为 T_imu_base 是 imu_link 到 base_link 的变换
    T_map_base = T_map_imu * T_imu_base;

    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = "map";
    msg.header.stamp = lightning::math::FromSec(res.timestamp_);
    msg.child_frame_id = "base_link";
    msg.transform = tf2::toMsg(T_map_base);

    tf_broadcaster_->sendTransform(msg);
}

void LocSystem::PublishDualLidarCalibrationTF(const lightning::loc::DualLidarCalibrationResult& res) {
    if (!tf_broadcaster_) {
        return;
    }

    Eigen::Quaterniond q(res.T_front_rear.linear());
    q.normalize();

    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = "front_lidar";
    msg.header.stamp = lightning::math::FromSec(res.timestamp);
    msg.child_frame_id = "rear_lidar";
    msg.transform.translation.x = res.T_front_rear.translation().x();
    msg.transform.translation.y = res.T_front_rear.translation().y();
    msg.transform.translation.z = res.T_front_rear.translation().z();
    msg.transform.rotation.x = q.x();
    msg.transform.rotation.y = q.y();
    msg.transform.rotation.z = q.z();
    msg.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(msg);
}
}  // namespace lightning
