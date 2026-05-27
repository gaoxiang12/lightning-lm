#include "pointcloud_preprocess.h"
#include <algorithm>
#include <cmath>
#include <execution>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "wrapper/ros_utils.h"

namespace lightning {

bool PointCloudPreprocess::Init(const std::string& yaml_path) {
    try {
        const YAML::Node yaml = YAML::LoadFile(yaml_path);
        const YAML::Node params = yaml["fasterlio"];

        blind_ = params["blind"].as<double>();
        time_scale_ = params["time_scale"].as<float>();
        num_scans_ = params["scan_line"].as<int>();
        point_filter_num_ = params["point_filter_num"].as<int>();
        height_max_ = params["height_max"] ? params["height_max"].as<float>() : yaml["roi"]["height_max"].as<float>();
        height_min_ = params["height_min"] ? params["height_min"].as<float>() : yaml["roi"]["height_min"].as<float>();

        const int lidar_type = params["lidar_type"].as<int>();
        if (lidar_type < static_cast<int>(LidarType::AVIA) ||
            lidar_type > static_cast<int>(LidarType::ROBOSENSE)) {
            LOG(ERROR) << "unknown lidar_type: " << lidar_type;
            return false;
        }
        lidar_type_ = static_cast<LidarType>(lidar_type);
        first_lidar_timestamp_ = -1.0;
        first_imu_timestamp_ = -1.0;
        return true;
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to initialize point cloud preprocess: " << e.what();
        return false;
    }
}

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

void PointCloudPreprocess::Process(const sensor_msgs::msg::PointCloud2 ::SharedPtr &msg, PointCloudType::Ptr &pcl_out) {
    switch (lidar_type_) {
        case LidarType::OUST64:
            Oust64Handler(msg);
            break;

        case LidarType::VELO32:
            VelodyneHandler(msg);
            break;

        case LidarType::ROBOSENSE:
            RoboSenseHandler(msg);
            break;

        default:
            LOG(ERROR) << "Error LiDAR Type";
            break;
    }
    *pcl_out = cloud_out_;
    pcl_out->header.stamp = process_time(ToSec(msg->header.stamp));
}

void PointCloudPreprocess::Process(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg,
                                   PointCloudType::Ptr &pcl_out) {
    cloud_out_.clear();
    cloud_full_.clear();

    int plsize = msg->point_num;

    cloud_out_.reserve(plsize);
    cloud_full_.resize(plsize);

    std::vector<char> is_valid_pt(plsize, 0);
    std::vector<uint> index(plsize - 1);
    for (uint i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;  // 从1开始
    }

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        // if ((msg->points[i].line < num_scans_) &&
        // ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
        if (i % point_filter_num_ != 0) {
            return;
        }

        cloud_full_[i].x = msg->points[i].x;
        cloud_full_[i].y = msg->points[i].y;
        cloud_full_[i].z = msg->points[i].z;
        cloud_full_[i].intensity = msg->points[i].reflectivity;
        cloud_full_[i].ring = msg->points[i].line;

        // use curvature as time of each laser points, curvature unit: ms
        cloud_full_[i].time = msg->points[i].offset_time / double(1000000);

        if (cloud_full_[i].z < height_min_ || cloud_full_[i].z > height_max_) {
            return;
        }

        if ((abs(cloud_full_[i].x - cloud_full_[i - 1].x) > 1e-7) ||
            (abs(cloud_full_[i].y - cloud_full_[i - 1].y) > 1e-7) ||
            (abs(cloud_full_[i].z - cloud_full_[i - 1].z) > 1e-7) &&
                (cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
                     cloud_full_[i].z * cloud_full_[i].z >
                 (blind_ * blind_))) {
            is_valid_pt[i] = 1;
        }

        // }
    });

    for (uint i = 1; i < plsize; i++) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
    *pcl_out = cloud_out_;
    pcl_out->header.stamp = process_time(ToSec(msg->header.stamp));
}

uint64_t PointCloudPreprocess::process_time(double raw_timestamp) {
    if (first_lidar_timestamp_ < 0.0) {
        first_lidar_timestamp_ = raw_timestamp;
        LOG(INFO) << "first lidar timestamp: " << first_lidar_timestamp_;
    }
    const double relative_time = std::max(0.0, raw_timestamp - first_lidar_timestamp_);
    return static_cast<uint64_t>(std::llround(relative_time * 1e9));
}

IMUPtr PointCloudPreprocess::process_imu(const IMUPtr& imu) {
    if (!imu) {
        return nullptr;
    }
    if (first_imu_timestamp_ < 0.0) {
        first_imu_timestamp_ = imu->timestamp;
        LOG(INFO) << "first IMU timestamp: " << first_imu_timestamp_;
    }
    IMUPtr output = std::make_shared<IMU>(*imu);
    output->timestamp = imu->timestamp - first_imu_timestamp_;
    return output;
}

IMUPtr PointCloudPreprocess::process_imu(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (!imu) {
        return nullptr;
    }
    IMUPtr output = std::make_shared<IMU>();
    output->timestamp = ToSec(imu->header.stamp);
    output->angular_velocity =
        Vec3d(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
    output->linear_acceleration =
        Vec3d(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
    output->orientation =
        Quatd(imu->orientation.w, imu->orientation.x, imu->orientation.y, imu->orientation.z);
    return process_imu(output);
}

void PointCloudPreprocess::Oust64Handler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);

    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) {
            continue;
        }

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) {
            continue;
        }

        if (pl_orig.points[i].z < height_min_ || pl_orig.points[i].z > height_max_) {
            continue;
        }

        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.ring = pl_orig.points[i].ring;

        added_pt.time = pl_orig.points[i].t / 1e6;
        cloud_out_.points.push_back(added_pt);
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

void PointCloudPreprocess::RoboSenseHandler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<PointRobotSense> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);

    double head_time = msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;

    // Header stamp and per-point timestamp are seconds on the lidar clock.
    // The common PointType stores relative point time in milliseconds.

    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) {
            continue;
        }

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) {
            continue;
        }

        if (pl_orig.points[i].z < height_min_ || pl_orig.points[i].z > height_max_) {
            continue;
        }

        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.ring = pl_orig.points[i].ring;

        added_pt.time = (pl_orig.points[i].timestamp - head_time) * 1e3;  //  / 1e6;  // curvature unit: ms

        cloud_out_.points.push_back(added_pt);
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

void PointCloudPreprocess::VelodyneHandler(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    cloud_out_.reserve(plsize);
    if (plsize == 0) {
        cloud_out_.width = 0;
        cloud_out_.height = 1;
        cloud_out_.is_dense = false;
        return;
    }

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time_ = true;
    } else {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;

        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.ring = pl_orig.points[i].ring;
        added_pt.time = pl_orig.points[i].time * time_scale_;  // curvature unit: ms

        if (!given_offset_time_) {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.time = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.time;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.time = (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.time = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.time < time_last[layer]) {
                added_pt.time += 360.0 / omega_l;
            }

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.time;
        }

        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }

    cloud_out_.width = cloud_out_.size();
    cloud_out_.height = 1;
    cloud_out_.is_dense = false;
}

}  // namespace lightning
