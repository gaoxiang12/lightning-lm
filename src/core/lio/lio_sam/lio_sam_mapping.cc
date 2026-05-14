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

template <typename T>
void ReadYamlParam(const YAML::Node& node,
                   const std::string& yaml_key,
                   const std::string& param_name,
                   std::vector<rclcpp::Parameter>& params) {
    if (node && node[yaml_key]) {
        SetParamOverride(params, param_name, node[yaml_key].as<T>());
    }
}

std::string LidarTypeToSensorName(int lidar_type) {
    if (lidar_type == 1) {
        return "livox";
    }
    if (lidar_type == 2) {
        return "velodyne";
    }
    return "ouster";
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
    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to load yaml " << yaml_path << ": " << e.what();
        return false;
    }

    std::vector<rclcpp::Parameter> overrides;

    const YAML::Node fasterlio = yaml["fasterlio"];
    if (fasterlio) {
        int lidar_type = 3;
        if (fasterlio["lidar_type"]) {
            lidar_type = fasterlio["lidar_type"].as<int>();
        }
        sensor_type_ = LidarTypeToSensorName(lidar_type);
        SetParamOverride(overrides, "sensor", sensor_type_);

        ReadYamlParam<int>(fasterlio, "scan_line", "N_SCAN", overrides);
        ReadYamlParam<double>(fasterlio, "blind", "lidarMinRange", overrides);
        ReadYamlParam<double>(fasterlio, "acc_cov", "imuAccNoise", overrides);
        ReadYamlParam<double>(fasterlio, "gyr_cov", "imuGyrNoise", overrides);
        ReadYamlParam<double>(fasterlio, "b_acc_cov", "imuAccBiasN", overrides);
        ReadYamlParam<double>(fasterlio, "b_gyr_cov", "imuGyrBiasN", overrides);
        ReadYamlParam<double>(fasterlio, "filter_size_scan", "odometrySurfLeafSize", overrides);
        ReadYamlParam<double>(fasterlio, "filter_size_map", "mappingSurfLeafSize", overrides);
        ReadYamlParam<std::vector<double>>(fasterlio, "extrinsic_T", "extrinsicTrans", overrides);
        ReadYamlParam<std::vector<double>>(fasterlio, "extrinsic_R", "extrinsicRot", overrides);
        ReadYamlParam<std::vector<double>>(fasterlio, "extrinsic_R", "extrinsicRPY", overrides);
    }

    YAML::Node lio_sam = yaml["lio_sam"];
    if (!lio_sam) {
        lio_sam = yaml["liosam"];
    }
    if (lio_sam) {
        ReadYamlParam<std::string>(lio_sam, "sensor", "sensor", overrides);
        if (lio_sam["sensor"]) {
            sensor_type_ = lio_sam["sensor"].as<std::string>();
            std::transform(sensor_type_.begin(), sensor_type_.end(), sensor_type_.begin(), ::tolower);
        }

        ReadYamlParam<std::string>(lio_sam, "lidarFrame", "lidarFrame", overrides);
        ReadYamlParam<std::string>(lio_sam, "baselinkFrame", "baselinkFrame", overrides);
        ReadYamlParam<std::string>(lio_sam, "odometryFrame", "odometryFrame", overrides);
        ReadYamlParam<std::string>(lio_sam, "mapFrame", "mapFrame", overrides);

        ReadYamlParam<bool>(lio_sam, "useImuHeadingInitialization", "useImuHeadingInitialization", overrides);
        ReadYamlParam<bool>(lio_sam, "useImuAccelRollPitchInitialization", "useImuAccelRollPitchInitialization",
                            overrides);

        ReadYamlParam<int>(lio_sam, "N_SCAN", "N_SCAN", overrides);
        ReadYamlParam<int>(lio_sam, "Horizon_SCAN", "Horizon_SCAN", overrides);
        ReadYamlParam<int>(lio_sam, "downsampleRate", "downsampleRate", overrides);
        ReadYamlParam<double>(lio_sam, "lidarMinRange", "lidarMinRange", overrides);
        ReadYamlParam<double>(lio_sam, "lidarMaxRange", "lidarMaxRange", overrides);

        ReadYamlParam<double>(lio_sam, "imuAccNoise", "imuAccNoise", overrides);
        ReadYamlParam<double>(lio_sam, "imuGyrNoise", "imuGyrNoise", overrides);
        ReadYamlParam<double>(lio_sam, "imuAccBiasN", "imuAccBiasN", overrides);
        ReadYamlParam<double>(lio_sam, "imuGyrBiasN", "imuGyrBiasN", overrides);
        ReadYamlParam<double>(lio_sam, "imuGravity", "imuGravity", overrides);
        ReadYamlParam<double>(lio_sam, "imuRPYWeight", "imuRPYWeight", overrides);
        ReadYamlParam<std::vector<double>>(lio_sam, "extrinsicRot", "extrinsicRot", overrides);
        ReadYamlParam<std::vector<double>>(lio_sam, "extrinsicRPY", "extrinsicRPY", overrides);
        ReadYamlParam<std::vector<double>>(lio_sam, "extrinsicTrans", "extrinsicTrans", overrides);

        ReadYamlParam<double>(lio_sam, "edgeThreshold", "edgeThreshold", overrides);
        ReadYamlParam<double>(lio_sam, "surfThreshold", "surfThreshold", overrides);
        ReadYamlParam<int>(lio_sam, "edgeFeatureMinValidNum", "edgeFeatureMinValidNum", overrides);
        ReadYamlParam<int>(lio_sam, "surfFeatureMinValidNum", "surfFeatureMinValidNum", overrides);
        ReadYamlParam<double>(lio_sam, "odometrySurfLeafSize", "odometrySurfLeafSize", overrides);
        ReadYamlParam<double>(lio_sam, "mappingCornerLeafSize", "mappingCornerLeafSize", overrides);
        ReadYamlParam<double>(lio_sam, "mappingSurfLeafSize", "mappingSurfLeafSize", overrides);
        ReadYamlParam<double>(lio_sam, "z_tollerance", "z_tollerance", overrides);
        ReadYamlParam<double>(lio_sam, "rotation_tollerance", "rotation_tollerance", overrides);
        ReadYamlParam<int>(lio_sam, "numberOfCores", "numberOfCores", overrides);
        ReadYamlParam<double>(lio_sam, "mappingProcessInterval", "mappingProcessInterval", overrides);
        ReadYamlParam<double>(lio_sam, "mappingLowSpeedMaxTranslationSpeed",
                              "mappingLowSpeedMaxTranslationSpeed", overrides);
        ReadYamlParam<double>(lio_sam, "mappingLowSpeedMaxExtrapolationTime",
                              "mappingLowSpeedMaxExtrapolationTime", overrides);
        ReadYamlParam<double>(lio_sam, "surroundingkeyframeAddingDistThreshold",
                              "surroundingkeyframeAddingDistThreshold", overrides);
        ReadYamlParam<double>(lio_sam, "surroundingkeyframeAddingAngleThreshold",
                              "surroundingkeyframeAddingAngleThreshold", overrides);
        ReadYamlParam<double>(lio_sam, "surroundingKeyframeDensity", "surroundingKeyframeDensity", overrides);
        ReadYamlParam<double>(lio_sam, "surroundingKeyframeSearchRadius", "surroundingKeyframeSearchRadius",
                              overrides);
        ReadYamlParam<bool>(lio_sam, "loopClosureEnableFlag", "loopClosureEnableFlag", overrides);
        ReadYamlParam<int>(lio_sam, "surroundingKeyframeSize", "surroundingKeyframeSize", overrides);
        ReadYamlParam<double>(lio_sam, "historyKeyframeSearchRadius", "historyKeyframeSearchRadius", overrides);
        ReadYamlParam<double>(lio_sam, "historyKeyframeSearchTimeDiff", "historyKeyframeSearchTimeDiff",
                              overrides);
        ReadYamlParam<int>(lio_sam, "historyKeyframeSearchNum", "historyKeyframeSearchNum", overrides);
        ReadYamlParam<double>(lio_sam, "historyKeyframeFitnessScore", "historyKeyframeFitnessScore", overrides);

        ReadYamlParam<bool>(lio_sam, "mappingMotionGateEnable", "mappingMotionGateEnable", overrides);
        ReadYamlParam<bool>(lio_sam, "mappingIcpFallbackEnable", "mappingIcpFallbackEnable", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxSpeed", "mappingMotionMaxSpeed", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxAngularVelocity", "mappingMotionMaxAngularVelocity",
                              overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxCurvature", "mappingMotionMaxCurvature", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxRollPitchDeg", "mappingMotionMaxRollPitchDeg", overrides);
        ReadYamlParam<bool>(lio_sam, "mappingFallbackIcpSkipOnBadLmMotion",
                            "mappingFallbackIcpSkipOnBadLmMotion", overrides);
        ReadYamlParam<double>(lio_sam, "mappingFallbackIcpMaxCorrespondenceDistance",
                              "mappingFallbackIcpMaxCorrespondenceDistance", overrides);
        ReadYamlParam<int>(lio_sam, "mappingFallbackIcpMaxIterations", "mappingFallbackIcpMaxIterations",
                           overrides);
        ReadYamlParam<double>(lio_sam, "mappingFallbackIcpLeafSize", "mappingFallbackIcpLeafSize", overrides);
        ReadYamlParam<double>(lio_sam, "mappingFallbackIcpFitnessScore", "mappingFallbackIcpFitnessScore",
                              overrides);
        ReadYamlParam<double>(lio_sam, "mappingFallbackIcpFitnessScoreMaxRange",
                              "mappingFallbackIcpFitnessScoreMaxRange", overrides);
        ReadYamlParam<int>(lio_sam, "mappingFallbackIcpMinSourcePoints", "mappingFallbackIcpMinSourcePoints",
                           overrides);
        ReadYamlParam<int>(lio_sam, "mappingFallbackIcpMinTargetPoints", "mappingFallbackIcpMinTargetPoints",
                           overrides);
        ReadYamlParam<int>(lio_sam, "mappingFallbackIcpMaxSourcePoints", "mappingFallbackIcpMaxSourcePoints",
                           overrides);
        ReadYamlParam<int>(lio_sam, "mappingFallbackIcpMaxTargetPoints", "mappingFallbackIcpMaxTargetPoints",
                           overrides);
    }

    const bool has_explicit_horizon = lio_sam && lio_sam["Horizon_SCAN"];
    if (!has_explicit_horizon && sensor_type_ == "velodyne") {
        SetParamOverride(overrides, "Horizon_SCAN", 1800);
    } else if (!has_explicit_horizon && sensor_type_ == "ouster") {
        SetParamOverride(overrides, "Horizon_SCAN", 1024);
    }

    node_options_ = rclcpp::NodeOptions();
    node_options_.use_intra_process_comms(true);
    node_options_.parameter_overrides(overrides);

    LOG(INFO) << "LIO-SAM params loaded: sensor=" << sensor_type_ << ", overrides=" << overrides.size();
    return true;
}

void LioSamMapping::ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu) {

    const double timestamp = ToSec(imu->header.stamp);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "lio-sam imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_count_++;
    imu_buffer_.push_back(*imu);
}

void LioSamMapping::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {

    const double timestamp = ToSec(msg->header.stamp);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lio-sam lidar loop back, clear buffer";
        lidar_buffer_.clear();
        time_buffer_.clear();
        lidar_pushed_ = false;

    }
    scan_count_++;
    last_timestamp_lidar_ = timestamp;
    lidar_buffer_.push_back(*msg);
    time_buffer_.push_back(timestamp);

    LOG(INFO) << "lio-sam enqueue cloud at " << std::setprecision(14) << timestamp
              << ", latest imu: " << last_timestamp_imu_;
}

void LioSamMapping::ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg) {
    if (!msg) {
        return;
    }

    pcl::PointCloud<::PointXYZIRT> cloud;
    cloud.reserve(msg->points.size());
    for (const auto& src : msg->points) {
        ::PointXYZIRT pt;
        pt.x = src.x;
        pt.y = src.y;
        pt.z = src.z;
        pt.intensity = src.reflectivity;
        pt.ring = src.line;
        pt.time = static_cast<float>(src.offset_time) * 1e-9f;
        cloud.push_back(pt);
    }
    cloud.height = 1;
    cloud.width = cloud.size();
    cloud.is_dense = false;

    auto ros_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(cloud, *ros_cloud);
    ros_cloud->header = msg->header;
    ProcessPointCloud2(ros_cloud);
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
