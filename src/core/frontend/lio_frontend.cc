#include "core/frontend/lio_frontend.h"
#include "wrapper/ros_utils.h"

#include <glog/logging.h>

namespace lightning {

void LIOFrontend::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    CloudPtr cloud(new PointCloudType());
    preprocess_->Process(msg, cloud);

    double timestamp = ToSec(msg->header.stamp);
    if (timestamp < last_timestamp_imu_ && !lidar_buffer_.empty()) {
        // 时间回跳，跳过
        return;
    }

    lidar_buffer_.push_back(cloud);
    time_buffer_.push_back(timestamp);
}

void LIOFrontend::ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    CloudPtr cloud(new PointCloudType());
    preprocess_->Process(msg, cloud);

    double timestamp = rclcpp::Time(msg->header.stamp).seconds();

    lidar_buffer_.push_back(cloud);
    time_buffer_.push_back(timestamp);
}

void LIOFrontend::ProcessPointCloud2(CloudPtr cloud) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    lidar_buffer_.push_back(cloud);
}

void LIOFrontend::ProcessIMU(const IMUPtr& msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (msg_in->timestamp < last_timestamp_imu_) {
        imu_buffer_.clear();
    }
    last_timestamp_imu_ = msg_in->timestamp;
    imu_buffer_.push_back(msg_in);
}

bool LIOFrontend::SyncPackages() {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    // 计算 lidar_end_time_
    if (!lidar_pushed_) {
        auto cloud = lidar_buffer_.front();
        double beg_time = time_buffer_.front();

        if (cloud->points.size() <= 1) {
            if (lidar_mean_scantime_ <= 0) {
                lidar_mean_scantime_ = 0.1;
            }
            lidar_end_time_ = beg_time + lidar_mean_scantime_;
        } else if (cloud->points.back().time / 1000.0 < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = beg_time + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = beg_time + cloud->points.back().time / 1000.0;

            lidar_mean_scantime_ +=
                (cloud->points.back().time / 1000.0 - lidar_mean_scantime_) / scan_num_;

            if ((lidar_end_time_ - beg_time) > 5.0 * lidar_mean_scantime_) {
                lidar_end_time_ = beg_time + lidar_mean_scantime_;
            }
        }

        lidar_pushed_ = true;
    }

    // 检查 IMU 数据是否覆盖
    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    // 收集 IMU 数据
    synced_imu_.clear();
    double imu_time = imu_buffer_.front()->timestamp;
    while (!imu_buffer_.empty() && imu_time < lidar_end_time_) {
        imu_time = imu_buffer_.front()->timestamp;
        if (imu_time > lidar_end_time_) break;
        synced_imu_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    return true;
}

}  // namespace lightning
