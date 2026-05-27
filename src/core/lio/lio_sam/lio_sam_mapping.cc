#include "core/lio/lio_sam/lio_sam_mapping.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <utility>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

#include "ui/pangolin_window.h"
#include "core/lightning_math.hpp"
#include "wrapper/ros_utils.h"

// Header-only offline LIO-SAM stages: no ROS topic handoff, no included .cpp files.
#include "core/lio/lio_sam/offline/image_projection_offline.h"
#include "core/lio/lio_sam/offline/feature_extraction_offline.h"
#include "core/lio/lio_sam/offline/map_optimization_offline.h"

namespace {

template <typename T>
void SetParamOverride(std::vector<rclcpp::Parameter>& params, const std::string& name, const T& value) {
    params.erase(std::remove_if(params.begin(), params.end(),
                                [&](const rclcpp::Parameter& p) { return p.get_name() == name; }),
                 params.end());
    params.emplace_back(name, value);
}

builtin_interfaces::msg::Time ToRosStamp(double timestamp) {
    builtin_interfaces::msg::Time stamp;
    const double clamped = std::max(0.0, timestamp);
    stamp.sec = static_cast<int32_t>(std::floor(clamped));
    stamp.nanosec = static_cast<uint32_t>(std::llround((clamped - stamp.sec) * 1e9));
    if (stamp.nanosec >= 1000000000U) {
        ++stamp.sec;
        stamp.nanosec -= 1000000000U;
    }
    return stamp;
}

lightning::SO3 RpyToSO3(double roll, double pitch, double yaw) {
    Eigen::Matrix3d rot =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    return lightning::SO3(Eigen::Quaterniond(rot));
}

}  // namespace

namespace lightning {

LioSamMapping::LioSamMapping() : LioSamMapping(Options()) {}

LioSamMapping::LioSamMapping(Options options) : options_(options) {}

LioSamMapping::~LioSamMapping() {
    image_projection_.reset();
    feature_extraction_.reset();
    map_optimization_.reset();
    if (owns_rclcpp_context_ && rclcpp::ok()) {
        rclcpp::shutdown();
    }
}

bool LioSamMapping::Init(const std::string& config_yaml) {
    LOG(INFO) << "init lio-sam mapping from " << config_yaml;
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    if (!rclcpp::ok()) {
        int argc = 1;
        const char* argv[] = {"lightning_lio_sam_offline"};
        rclcpp::init(argc, argv);
        owns_rclcpp_context_ = true;
    }

    image_projection_ = std::make_unique<::ImageProjection>(node_options_);
    feature_extraction_ = std::make_unique<::FeatureExtraction>(node_options_);
    map_optimization_ = std::make_unique<::mapOptimization>(node_options_);

    return true;
}

bool LioSamMapping::LoadParamsFromYAML(const std::string& yaml_path) {
    try {
        const YAML::Node yaml = YAML::LoadFile(yaml_path);
        const YAML::Node common = yaml["common"];
        const YAML::Node params = yaml["lio_sam"];
        if (!params) {
            LOG(ERROR) << "lio_sam config section is missing";
            return false;
        }
        std::vector<rclcpp::Parameter> overrides;
        sensor_type_ = params["sensor"].as<std::string>();
        std::transform(sensor_type_.begin(), sensor_type_.end(), sensor_type_.begin(), ::tolower);

        SetParamOverride(overrides, "pointCloudTopic", common["lidar_topic"].as<std::string>());
        SetParamOverride(overrides, "imuTopic", common["imu_topic"].as<std::string>());
        SetParamOverride(overrides, "odomTopic", common["odom_topic"].as<std::string>());
        SetParamOverride(overrides, "gpsTopic", common["gps_topic"].as<std::string>());
        SetParamOverride(overrides, "lidarFrame", common["lidar_frame"].as<std::string>());
        SetParamOverride(overrides, "baselinkFrame", common["base_link_frame"].as<std::string>());
        SetParamOverride(overrides, "odometryFrame", common["odometry_frame"].as<std::string>());
        SetParamOverride(overrides, "mapFrame", common["map_frame"].as<std::string>());

        SetParamOverride(overrides, "sensor", sensor_type_);
        SetParamOverride(overrides, "useImuHeadingInitialization", params["useImuHeadingInitialization"].as<bool>());
        SetParamOverride(overrides, "useImuAccelRollPitchInitialization",
                         params["useImuAccelRollPitchInitialization"].as<bool>());
        SetParamOverride(overrides, "N_SCAN", params["N_SCAN"].as<int>());
        SetParamOverride(overrides, "Horizon_SCAN", params["Horizon_SCAN"].as<int>());
        SetParamOverride(overrides, "downsampleRate", params["downsampleRate"].as<int>());
        SetParamOverride(overrides, "lidarMinRange", params["lidarMinRange"].as<double>());
        SetParamOverride(overrides, "lidarMaxRange", params["lidarMaxRange"].as<double>());
        SetParamOverride(overrides, "imuAccNoise", params["imuAccNoise"].as<double>());
        SetParamOverride(overrides, "imuGyrNoise", params["imuGyrNoise"].as<double>());
        SetParamOverride(overrides, "imuAccBiasN", params["imuAccBiasN"].as<double>());
        SetParamOverride(overrides, "imuGyrBiasN", params["imuGyrBiasN"].as<double>());
        SetParamOverride(overrides, "imuGravity", params["imuGravity"].as<double>());
        SetParamOverride(overrides, "imuRPYWeight", params["imuRPYWeight"].as<double>());
        SetParamOverride(overrides, "extrinsicTrans", params["extrinsicTrans"].as<std::vector<double>>());
        SetParamOverride(overrides, "extrinsicRot", params["extrinsicRot"].as<std::vector<double>>());
        SetParamOverride(overrides, "extrinsicRPY", params["extrinsicRPY"].as<std::vector<double>>());
        SetParamOverride(overrides, "edgeThreshold", params["edgeThreshold"].as<double>());
        SetParamOverride(overrides, "surfThreshold", params["surfThreshold"].as<double>());
        SetParamOverride(overrides, "edgeFeatureMinValidNum", params["edgeFeatureMinValidNum"].as<int>());
        SetParamOverride(overrides, "surfFeatureMinValidNum", params["surfFeatureMinValidNum"].as<int>());
        SetParamOverride(overrides, "odometrySurfLeafSize", params["odometrySurfLeafSize"].as<double>());
        SetParamOverride(overrides, "mappingCornerLeafSize", params["mappingCornerLeafSize"].as<double>());
        SetParamOverride(overrides, "mappingSurfLeafSize", params["mappingSurfLeafSize"].as<double>());
        SetParamOverride(overrides, "z_tollerance", params["z_tollerance"].as<double>());
        SetParamOverride(overrides, "rotation_tollerance", params["rotation_tollerance"].as<double>());
        SetParamOverride(overrides, "numberOfCores", params["numberOfCores"].as<int>());
        SetParamOverride(overrides, "mappingProcessInterval", params["mappingProcessInterval"].as<double>());
        SetParamOverride(overrides, "mappingLowSpeedMaxTranslationSpeed",
                         params["mappingLowSpeedMaxTranslationSpeed"].as<double>());
        SetParamOverride(overrides, "surroundingkeyframeAddingDistThreshold",
                         params["surroundingkeyframeAddingDistThreshold"].as<double>());
        SetParamOverride(overrides, "surroundingkeyframeAddingAngleThreshold",
                         params["surroundingkeyframeAddingAngleThreshold"].as<double>());
        SetParamOverride(overrides, "surroundingKeyframeDensity", params["surroundingKeyframeDensity"].as<double>());
        SetParamOverride(overrides, "surroundingKeyframeSearchRadius",
                         params["surroundingKeyframeSearchRadius"].as<double>());
        SetParamOverride(overrides, "loopClosureEnableFlag",
                         options_.is_in_slam_mode_ && params["loopClosureEnableFlag"].as<bool>());
        SetParamOverride(overrides, "surroundingKeyframeSize", params["surroundingKeyframeSize"].as<int>());
        SetParamOverride(overrides, "historyKeyframeSearchRadius", params["historyKeyframeSearchRadius"].as<double>());
        SetParamOverride(overrides, "historyKeyframeSearchTimeDiff",
                         params["historyKeyframeSearchTimeDiff"].as<double>());
        SetParamOverride(overrides, "historyKeyframeSearchNum", params["historyKeyframeSearchNum"].as<int>());
        SetParamOverride(overrides, "historyKeyframeFitnessScore", params["historyKeyframeFitnessScore"].as<double>());
        SetParamOverride(overrides, "mappingMotionGateEnable", params["mappingMotionGateEnable"].as<bool>());
        SetParamOverride(overrides, "mappingIcpFallbackEnable", params["mappingIcpFallbackEnable"].as<bool>());
        SetParamOverride(overrides, "mappingMotionMaxSpeed", params["mappingMotionMaxSpeed"].as<double>());
        SetParamOverride(overrides, "mappingMotionMaxAngularVelocity",
                         params["mappingMotionMaxAngularVelocity"].as<double>());
        SetParamOverride(overrides, "mappingMotionMaxCurvature", params["mappingMotionMaxCurvature"].as<double>());
        SetParamOverride(overrides, "mappingMotionMaxRollPitchDeg", params["mappingMotionMaxRollPitchDeg"].as<double>());
        SetParamOverride(overrides, "mappingFallbackIcpSkipOnBadLmMotion",
                         params["mappingFallbackIcpSkipOnBadLmMotion"].as<bool>());

        node_options_ = rclcpp::NodeOptions();
        node_options_.use_intra_process_comms(true);
        node_options_.parameter_overrides(overrides);
        LOG(INFO) << "LIO-SAM params loaded: sensor=" << sensor_type_ << ", overrides=" << overrides.size();
        return true;
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to load LIO-SAM params from " << yaml_path << ": " << e.what();
        return false;
    }
}

void LioSamMapping::ProcessIMU(const IMUPtr& input) {
    sensor_msgs::msg::Imu imu;
    imu.header.stamp = ToRosStamp(input->timestamp);
    imu.angular_velocity.x = input->angular_velocity.x();
    imu.angular_velocity.y = input->angular_velocity.y();
    imu.angular_velocity.z = input->angular_velocity.z();
    imu.linear_acceleration.x = input->linear_acceleration.x();
    imu.linear_acceleration.y = input->linear_acceleration.y();
    imu.linear_acceleration.z = input->linear_acceleration.z();
    imu.orientation.x = input->orientation.x();
    imu.orientation.y = input->orientation.y();
    imu.orientation.z = input->orientation.z();
    imu.orientation.w = input->orientation.w();
    const double timestamp = input->timestamp;
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "lio-sam imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_count_++;
    imu_buffer_.push_back(imu);
}

void LioSamMapping::ProcessPointCloud2(CloudPtr cloud) {
    pcl::PointCloud<::VelodynePointXYZIRT> native_cloud;
    native_cloud.header = cloud->header;
    native_cloud.reserve(cloud->size());
    for (const auto& source : cloud->points) {
        ::VelodynePointXYZIRT point;
        point.x = source.x;
        point.y = source.y;
        point.z = source.z;
        point.intensity = source.intensity;
        point.ring = source.ring;
        point.time = static_cast<float>(source.time * 1e-3);
        native_cloud.push_back(point);
    }
    native_cloud.height = cloud->height;
    native_cloud.width = cloud->width;
    native_cloud.is_dense = cloud->is_dense;
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(native_cloud, msg);

    const double timestamp = math::ToSec(cloud->header.stamp);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lio-sam lidar loop back, clear buffer";
        lidar_buffer_.clear();
        time_buffer_.clear();
        lidar_pushed_ = false;

    }
    scan_count_++;
    last_timestamp_lidar_ = timestamp;
    lidar_buffer_.push_back(msg);
    time_buffer_.push_back(timestamp);

    LOG(INFO) << "lio-sam enqueue cloud at " << std::setprecision(14) << timestamp
              << ", latest imu: " << last_timestamp_imu_;
}

bool LioSamMapping::SyncPackages() {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_ = SyncedPackage();
        measures_.cloud = lidar_buffer_.front();
        measures_.lidar_begin_time = time_buffer_.front();

        double scan_duration = 0.0;
        if (sensor_type_ == "ouster") {
            pcl::PointCloud<::OusterPointXYZIRT> scan_for_time;
            pcl::fromROSMsg(measures_.cloud, scan_for_time);
            if (scan_for_time.points.size() <= 1) {
                LOG(WARNING) << "LIO-SAM input point cloud has too few points, drop scan";
                lidar_buffer_.pop_front();
                time_buffer_.pop_front();
                return false;
            }
            scan_duration = scan_for_time.points.back().t * 1e-9;
        } else {
            pcl::PointCloud<::PointXYZIRT> scan_for_time;
            pcl::fromROSMsg(measures_.cloud, scan_for_time);
            if (scan_for_time.points.size() <= 1) {
                LOG(WARNING) << "LIO-SAM input point cloud has too few points, drop scan";
                lidar_buffer_.pop_front();
                time_buffer_.pop_front();
                return false;
            }
            scan_duration = scan_for_time.points.back().time;
        }

        if (!std::isfinite(scan_duration) || scan_duration <= 0.0 || scan_duration > 0.5) {
            LOG(ERROR) << "invalid LIO-SAM scan duration: " << scan_duration;
            lidar_buffer_.pop_front();
            time_buffer_.pop_front();
            lidar_pushed_ = false;
            return false;
        }

        scan_num_for_mean_++;
        lidar_mean_scantime_ += (scan_duration - lidar_mean_scantime_) / scan_num_for_mean_;
        lidar_begin_time_ = measures_.lidar_begin_time;
        lidar_end_time_ = measures_.lidar_begin_time + scan_duration;

        measures_.lidar_end_time = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    while (imu_buffer_.size() >= 2 && ToSec(imu_buffer_[1].header.stamp) < lidar_begin_time_ - 0.05) {
        imu_buffer_.pop_front();
    }

    if (ToSec(imu_buffer_.front().header.stamp) > lidar_begin_time_) {
        LOG(WARNING) << "LIO-SAM IMU does not cover scan begin, drop lidar scan. scan="
                     << std::setprecision(14) << lidar_begin_time_
                     << ", first imu=" << ToSec(imu_buffer_.front().header.stamp);
        lidar_buffer_.pop_front();
        time_buffer_.pop_front();
        lidar_pushed_ = false;
        return false;
    }

    measures_.imus.clear();
    const double imu_collect_end = lidar_end_time_ + 0.01;
    bool imu_covers_scan_end = false;
    for (const auto& imu : imu_buffer_) {
        const double imu_time = ToSec(imu.header.stamp);
        measures_.imus.push_back(imu);
        if (imu_time >= lidar_end_time_) {
            imu_covers_scan_end = true;
        }
        if (imu_time >= imu_collect_end) {
            break;
        }
    }

    if (measures_.imus.empty() || !imu_covers_scan_end) {
        return false;
    }

    while (imu_buffer_.size() >= 2 && ToSec(imu_buffer_[1].header.stamp) <= lidar_end_time_) {
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;
    LOG(INFO) << "LIO-SAM sync: begin=" << std::setprecision(14) << measures_.lidar_begin_time
              << ", end=" << measures_.lidar_end_time
              << ", duration=" << (measures_.lidar_end_time - measures_.lidar_begin_time)
              << " s, imu=" << measures_.imus.size();

    return true;
}

bool LioSamMapping::Run() {
    if (!image_projection_ || !feature_extraction_ || !map_optimization_) {
        return false;
    }

    if (!SyncPackages()) {
        return false;
    }

    LioSamCloudInfo cloud_info;

    if (!image_projection_->Run(measures_.cloud, measures_.imus,
                                measures_.lidar_begin_time, measures_.lidar_end_time,
                                cloud_info)) {
        return false;
    }

    if (!feature_extraction_->Run(cloud_info)) {
        return false;
    }

    if (!map_optimization_->Run(cloud_info)) {
        return false;
    }

    state_.timestamp_ = map_optimization_->timeLaserInfoCur;
    state_.pos_ = Vec3d(map_optimization_->transformTobeMapped[3],
                        map_optimization_->transformTobeMapped[4],
                        map_optimization_->transformTobeMapped[5]);
    state_.rot_ = RpyToSO3(map_optimization_->transformTobeMapped[0],
                           map_optimization_->transformTobeMapped[1],
                           map_optimization_->transformTobeMapped[2]);

    scan_undistort_->clear();
    if (cloud_info.cloud_deskewed) {
        scan_undistort_->reserve(cloud_info.cloud_deskewed->size());
        for (const auto& p : cloud_info.cloud_deskewed->points) {
            PointType pt;
            pt.x = p.x;
            pt.y = p.y;
            pt.z = p.z;
            pt.intensity = p.intensity;
            pt.time = 0.0;
            scan_undistort_->push_back(pt);
        }
        scan_undistort_->height = 1;
        scan_undistort_->width = scan_undistort_->size();
        scan_undistort_->is_dense = cloud_info.cloud_deskewed->is_dense;
    }
    recent_cloud_ = scan_undistort_;

    if (ui_) {
        ui_->UpdateNavState(state_);
        ui_->UpdateScan(scan_undistort_, state_.GetPose());
    }

    MakeLightningKeyframeIfNeeded();
    SyncLightningKeyframePoses();
    return true;
}

bool LioSamMapping::MakeLightningKeyframeIfNeeded() {
    if (!map_optimization_ || !map_optimization_->createdNewKeyframe || !map_optimization_->cloudKeyPoses6D ||
        map_optimization_->cloudKeyPoses6D->empty() ||
        map_optimization_->cloudKeyPoses6D->size() <= native_keyframe_count_) {
        return false;
    }

    const auto& pose = map_optimization_->cloudKeyPoses6D->points.back();
    pcl::PointCloud<::PointType>::Ptr native_cloud;
    if (!map_optimization_->rawCloudKeyFrames.empty()) {
        native_cloud = map_optimization_->rawCloudKeyFrames.back();
    }

    CloudPtr cloud(new PointCloudType());
    if (native_cloud) {
        cloud->reserve(native_cloud->size());
        for (const auto& p : native_cloud->points) {
            PointType pt;
            pt.x = p.x;
            pt.y = p.y;
            pt.z = p.z;
            pt.intensity = p.intensity;
            pt.time = 0.0;
            cloud->push_back(pt);
        }
        cloud->height = 1;
        cloud->width = cloud->size();
        cloud->is_dense = native_cloud->is_dense;
    }

    state_.timestamp_ = pose.time;
    state_.pos_ = Vec3d(pose.x, pose.y, pose.z);
    state_.rot_ = RpyToSO3(pose.roll, pose.pitch, pose.yaw);

    auto kf = std::make_shared<Keyframe>(kf_id_++, cloud, state_);
    kf->SetLIOPose(state_.GetPose());
    kf->SetOptPose(state_.GetPose());
    kf->SetState(state_);

    all_keyframes_.push_back(kf);
    last_kf_ = kf;
    native_keyframe_count_ = map_optimization_->cloudKeyPoses6D->size();
    map_optimization_->createdNewKeyframe = false;

    LOG(INFO) << "LIO-SAM: create lightning keyframe " << kf->GetID() << ", pose: "
              << state_.pos_.transpose() << ", time: " << std::setprecision(14) << state_.timestamp_;
    return true;
}

void LioSamMapping::SyncLightningKeyframePoses() {
    if (!map_optimization_ || !map_optimization_->cloudKeyPoses6D) {
        return;
    }

    const size_t n = std::min(all_keyframes_.size(), map_optimization_->cloudKeyPoses6D->size());
    for (size_t i = 0; i < n; ++i) {
        const auto& pose = map_optimization_->cloudKeyPoses6D->points[i];
        NavState state;
        state.timestamp_ = pose.time;
        state.pos_ = Vec3d(pose.x, pose.y, pose.z);
        state.rot_ = RpyToSO3(pose.roll, pose.pitch, pose.yaw);
        all_keyframes_[i]->SetLIOPose(state.GetPose());
        all_keyframes_[i]->SetOptPose(state.GetPose());
        all_keyframes_[i]->SetState(state);
    }
}

CloudPtr LioSamMapping::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) {
    CloudPtr global_map(new PointCloudType);

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(res, res, res);

    for (auto& kf : all_keyframes_) {
        CloudPtr cloud = kf->GetCloud();
        CloudPtr cloud_filter(new PointCloudType);

        if (use_voxel) {
            voxel.setInputCloud(cloud);
            voxel.filter(*cloud_filter);
        } else {
            cloud_filter = cloud;
        }

        CloudPtr cloud_trans(new PointCloudType);
        if (use_lio_pose) {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetLIOPose().matrix());
        } else {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetOptPose().matrix());
        }
        *global_map += *cloud_trans;
    }

    CloudPtr global_map_filtered(new PointCloudType);
    if (use_voxel) {
        voxel.setInputCloud(global_map);
        voxel.filter(*global_map_filtered);
    } else {
        global_map_filtered = global_map;
    }

    global_map_filtered->height = 1;
    global_map_filtered->width = global_map_filtered->size();
    global_map_filtered->is_dense = false;
    return global_map_filtered;
}

}  // namespace lightning
