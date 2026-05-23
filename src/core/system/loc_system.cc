//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"
#include "core/localization/localization.h"
#include "core/localization/dual_lidar_online_calibration.h"
#include "io/yaml_io.h"
#include "wrapper/ros_utils.h"
#include <tf2/transform_datatypes.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cmath>
#include <csignal>

namespace lightning {

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() {
    dual_lidar_pair_proc_.Quit();
    loc_->Finish();

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
    localization_mode_ = yaml.GetValue<std::string>("localization", "mode");
    lidar_count_ = yaml.GetValue<int>("localization", "lidar_count");

    if (lidar_count_ != 1 && lidar_count_ != 2) {
        LOG(WARNING) << "invalid localization.lidar_count=" << lidar_count_ << ", use 1";
        lidar_count_ = 1;
    }

    pure_calibration_mode_ = localization_mode_ == "dual_lidar_online_calibration";
    normal_localization_mode_ = localization_mode_ == "localization";

    if (!pure_calibration_mode_ && !normal_localization_mode_) {
        LOG(WARNING) << "unknown localization.mode=" << localization_mode_ << ", use localization";
        localization_mode_ = "localization";
        normal_localization_mode_ = true;
        pure_calibration_mode_ = false;
    }

    rclcpp::QoS qos(10);



    if (!loc_->Init(yaml_path, map_path)) {
        LOG(ERROR) << "online loc node init failed.";
        return false;
    }
    LOG(INFO) << "online loc node has been created.";

    const bool subscribe_imu = normal_localization_mode_;
    const bool subscribe_single_lidar = normal_localization_mode_ && lidar_count_ == 1;
    const bool subscribe_dual_lidar_pair =
        pure_calibration_mode_ || (normal_localization_mode_ && lidar_count_ == 2);

    if (subscribe_imu) {
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
                IMUPtr imu = std::make_shared<IMU>();
                imu->timestamp = ToSec(msg->header.stamp);
                imu->linear_acceleration =
                    Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
                imu->angular_velocity = Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
                ProcessIMU(imu);
            });

    }

    if (options_.pub_tf_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        loc_->SetTFCallback(
            [this](const lightning::loc::LocalizationResult& pose) { PublishBaseLinkTF(pose); });
    }

    if (subscribe_single_lidar) {
        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });

        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg ::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });
    }

    if (subscribe_dual_lidar_pair) {
        const std::string pair_cfg_name =
            pure_calibration_mode_ ? "dual_lidar_online_calibration" : "dual_lidar_localization";

        front_lidar_topic_ = yaml.GetValue<std::string>(pair_cfg_name, "front_lidar_topic");
        rear_lidar_topic_ = yaml.GetValue<std::string>(pair_cfg_name, "rear_lidar_topic");
        dual_lidar_sync_tolerance_ = std::max(0.0, yaml.GetValue<double>(pair_cfg_name, "sync_tolerance"));
        const int max_queue_size = yaml.GetValue<int>(pair_cfg_name, "max_queue_size");
        dual_lidar_max_queue_size_ = max_queue_size > 0 ? static_cast<size_t>(max_queue_size) : 20;
        const int pair_process_queue_size = std::max(1, yaml.GetValue<int>(pair_cfg_name, "pair_process_queue_size"));

        dual_lidar_pair_proc_.SetName("dual lidar pair input");
        dual_lidar_pair_proc_.SetMaxSize(pair_process_queue_size);
        dual_lidar_pair_proc_.SetProcFunc([this](const TimedCloudPair& pair) {
            if (pair.front_cloud && pair.rear_cloud) {
                loc_->ProcessDualLidarPointCloudPair(pair.front_cloud, pair.rear_cloud);
            }
        });
        dual_lidar_pair_proc_.Start();

        front_lidar_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            front_lidar_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                ProcessFrontLidar(cloud);
            });

        rear_lidar_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            rear_lidar_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                ProcessRearLidar(cloud);
            });

        if (pure_calibration_mode_) {
            loc_->SetDualLidarCalibrationCallback([this](const lightning::loc::DualLidarCalibrationResult& res) {
                Eigen::Quaterniond q(res.T_front_rear.linear());
                const Eigen::Vector3d t = res.T_front_rear.translation();

                LOG(INFO) << "[DUAL_LIDAR_CALIB][RESULT] "
                          << "front_lidar->rear_lidar "
                          << "accepted=" << res.accepted_observations
                          << " fitness=" << res.fitness
                          << " translation=" << t.transpose()
                          << " quaternion_xyzw="
                          << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
            });
        }

        LOG(INFO) << "dual lidar pair input enabled. front_topic=" << front_lidar_topic_
                  << " rear_topic=" << rear_lidar_topic_
                  << " sync_tolerance=" << dual_lidar_sync_tolerance_
                  << " max_queue_size=" << dual_lidar_max_queue_size_
                  << " pair_process_queue_size=" << pair_process_queue_size;
        if (normal_localization_mode_ && lidar_count_ == 2) {
            LOG(INFO) << "dual lidar localization enabled, common/lidar_topic is not subscribed";
        }
    }

    return true;
}

void LocSystem::SetInitPose(const SE3 &pose) {
    LOG(INFO) << "set init pose: " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();

    if (loc_) {
        loc_->SetExternalPose(pose.unit_quaternion(), pose.translation());
    }
    loc_started_ = true;
}

void LocSystem::ProcessIMU(const IMUPtr &imu) {
    if (loc_ && normal_localization_mode_) {
        loc_->ProcessIMUMsg(imu);
    }
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud) {
    if (loc_ && normal_localization_mode_ && lidar_count_ == 1 && loc_started_) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr &cloud) {
    if (loc_ && normal_localization_mode_ && lidar_count_ == 1 && loc_started_) {
        loc_->ProcessLivoxLidarMsg(cloud);
    }
}

void LocSystem::ProcessFrontLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (!loc_ || !cloud) {
        return;
    }

    if (normal_localization_mode_ && lidar_count_ == 2 && !loc_started_) {
        return;
    }

    if (!pure_calibration_mode_ && !(normal_localization_mode_ && lidar_count_ == 2)) {
        return;
    }

    TimedCloudPair pair;
    bool has_pair = false;
    {
        std::lock_guard<std::mutex> lock(dual_lidar_mutex_);

        front_lidar_queue_.push_back({ToSec(cloud->header.stamp), cloud});
        while (front_lidar_queue_.size() > dual_lidar_max_queue_size_) {
            front_lidar_queue_.pop_front();
        }

        has_pair = TryPopDualLidarPairLocked(pair);
    }

    if (has_pair) {
        dual_lidar_pair_proc_.AddMessage(pair);
    }
}

void LocSystem::ProcessRearLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (!loc_ || !cloud) {
        return;
    }

    if (normal_localization_mode_ && lidar_count_ == 2 && !loc_started_) {
        return;
    }

    if (!pure_calibration_mode_ && !(normal_localization_mode_ && lidar_count_ == 2)) {
        return;
    }

    TimedCloudPair pair;
    bool has_pair = false;
    {
        std::lock_guard<std::mutex> lock(dual_lidar_mutex_);

        rear_lidar_queue_.push_back({ToSec(cloud->header.stamp), cloud});
        while (rear_lidar_queue_.size() > dual_lidar_max_queue_size_) {
            rear_lidar_queue_.pop_front();
        }

        has_pair = TryPopDualLidarPairLocked(pair);
    }

    if (has_pair) {
        dual_lidar_pair_proc_.AddMessage(pair);
    }
}

bool LocSystem::TryPopDualLidarPairLocked(TimedCloudPair& pair) {
    while (!front_lidar_queue_.empty() && !rear_lidar_queue_.empty()) {
        const double dt =
            front_lidar_queue_.front().timestamp - rear_lidar_queue_.front().timestamp;

        if (std::abs(dt) <= dual_lidar_sync_tolerance_) {
            pair.front_cloud = front_lidar_queue_.front().cloud;
            pair.rear_cloud = rear_lidar_queue_.front().cloud;

            front_lidar_queue_.pop_front();
            rear_lidar_queue_.pop_front();

            return true;
        }

        if (dt < 0.0) {
            LOG_EVERY_N(WARNING, 100)
                << "[DUAL_LIDAR_SYNC][DROP_FRONT] dt=" << dt
                << " tolerance=" << dual_lidar_sync_tolerance_;
            front_lidar_queue_.pop_front();
        } else {
            LOG_EVERY_N(WARNING, 100)
                << "[DUAL_LIDAR_SYNC][DROP_REAR] dt=" << dt
                << " tolerance=" << dual_lidar_sync_tolerance_;
            rear_lidar_queue_.pop_front();
        }
    }

    return false;
}

void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}

void LocSystem::PublishBaseLinkTF(const lightning::loc::LocalizationResult& res) {
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

}  // namespace lightning
