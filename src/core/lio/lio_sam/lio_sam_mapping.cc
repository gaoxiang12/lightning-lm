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

// The native LIO-SAM files are included as source-level core modules.  In this
// mode their ROS topic endpoints and standalone mains are disabled; data moves
// through LioSamMapping::Run() after lightning-lm style synchronization.
#define LIO_SAM_LIGHTNING_OFFLINE
#include "core/lio/lio_sam/native/imageProjection.cpp"
#include "core/lio/lio_sam/native/featureExtraction.cpp"
#include "core/lio/lio_sam/native/mapOptmization.cpp"

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
    SetParamOverride(overrides, "lioMode", std::string("mapping"));

    const YAML::Node common = yaml["common"];
    ReadYamlParam<std::string>(common, "lidar_topic", "pointCloudTopic", overrides);
    ReadYamlParam<std::string>(common, "imu_topic", "imuTopic", overrides);

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

        ReadYamlParam<std::string>(lio_sam, "pointCloudTopic", "pointCloudTopic", overrides);
        ReadYamlParam<std::string>(lio_sam, "imuTopic", "imuTopic", overrides);
        ReadYamlParam<std::string>(lio_sam, "odomTopic", "odomTopic", overrides);
        ReadYamlParam<std::string>(lio_sam, "gpsTopic", "gpsTopic", overrides);
        ReadYamlParam<std::string>(lio_sam, "lidarFrame", "lidarFrame", overrides);
        ReadYamlParam<std::string>(lio_sam, "baselinkFrame", "baselinkFrame", overrides);
        ReadYamlParam<std::string>(lio_sam, "odometryFrame", "odometryFrame", overrides);
        ReadYamlParam<std::string>(lio_sam, "mapFrame", "mapFrame", overrides);
        ReadYamlParam<std::string>(lio_sam, "lioMode", "lioMode", overrides);
        SetParamOverride(overrides, "lioMode", std::string("mapping"));

        ReadYamlParam<bool>(lio_sam, "useImuHeadingInitialization", "useImuHeadingInitialization", overrides);
        ReadYamlParam<bool>(lio_sam, "useImuAccelRollPitchInitialization", "useImuAccelRollPitchInitialization",
                            overrides);
        ReadYamlParam<bool>(lio_sam, "useGpsElevation", "useGpsElevation", overrides);
        ReadYamlParam<double>(lio_sam, "gpsCovThreshold", "gpsCovThreshold", overrides);
        ReadYamlParam<double>(lio_sam, "poseCovThreshold", "poseCovThreshold", overrides);
        ReadYamlParam<bool>(lio_sam, "savePCD", "savePCD", overrides);
        ReadYamlParam<std::string>(lio_sam, "savePCDDirectory", "savePCDDirectory", overrides);

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
        ReadYamlParam<double>(lio_sam, "loopClosureFrequency", "loopClosureFrequency", overrides);
        ReadYamlParam<int>(lio_sam, "surroundingKeyframeSize", "surroundingKeyframeSize", overrides);
        ReadYamlParam<double>(lio_sam, "historyKeyframeSearchRadius", "historyKeyframeSearchRadius", overrides);
        ReadYamlParam<double>(lio_sam, "historyKeyframeSearchTimeDiff", "historyKeyframeSearchTimeDiff", overrides);
        ReadYamlParam<int>(lio_sam, "historyKeyframeSearchNum", "historyKeyframeSearchNum", overrides);
        ReadYamlParam<double>(lio_sam, "historyKeyframeFitnessScore", "historyKeyframeFitnessScore", overrides);
        ReadYamlParam<double>(lio_sam, "globalMapVisualizationSearchRadius",
                              "globalMapVisualizationSearchRadius", overrides);
        ReadYamlParam<double>(lio_sam, "globalMapVisualizationPoseDensity",
                              "globalMapVisualizationPoseDensity", overrides);
        ReadYamlParam<double>(lio_sam, "globalMapVisualizationLeafSize", "globalMapVisualizationLeafSize",
                              overrides);

        ReadYamlParam<bool>(lio_sam, "mappingMotionGateEnable", "mappingMotionGateEnable", overrides);
        ReadYamlParam<bool>(lio_sam, "mappingIcpFallbackEnable", "mappingIcpFallbackEnable", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxSpeed", "mappingMotionMaxSpeed", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxAcceleration", "mappingMotionMaxAcceleration", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxAngularVelocity", "mappingMotionMaxAngularVelocity",
                              overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxAngularAcceleration",
                              "mappingMotionMaxAngularAcceleration", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxCurvature", "mappingMotionMaxCurvature", overrides);
        ReadYamlParam<double>(lio_sam, "mappingMotionMaxRollPitchDeg", "mappingMotionMaxRollPitchDeg", overrides);
        ReadYamlParam<bool>(lio_sam, "mappingFallbackIcpSkipOnBadLmMotion",
                            "mappingFallbackIcpSkipOnBadLmMotion", overrides);
        ReadYamlParam<double>(lio_sam, "mappingRecoveryMaxPositionError", "mappingRecoveryMaxPositionError",
                              overrides);
        ReadYamlParam<double>(lio_sam, "mappingRecoveryMaxYawDeg", "mappingRecoveryMaxYawDeg", overrides);
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
    if (!imu) {
        return;
    }

    const double timestamp = ToSec(imu->header.stamp);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "lio-sam imu loop back, clear buffer";
        imuQueue_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_count_++;
    imuQueue_.push_back(*imu);
}

void LioSamMapping::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    if (!msg) {
        return;
    }

    const double timestamp = ToSec(msg->header.stamp);
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lio-sam lidar loop back, clear buffer";
        cloudQueue_.clear();
    }

    scan_count_++;
    last_timestamp_lidar_ = timestamp;
    cloudQueue_.push_back(*msg);

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

double LioSamMapping::EstimateScanEndTime(const sensor_msgs::msg::PointCloud2& cloud_msg, double lidar_begin_time) {
    double scan_duration = lidar_mean_scantime_;

    try {
        if (sensor_type_ == "ouster") {
            pcl::PointCloud<::OusterPointXYZIRT> cloud;
            pcl::fromROSMsg(cloud_msg, cloud);
            if (!cloud.empty()) {
                scan_duration = static_cast<double>(cloud.points.back().t) * 1e-9;
            }
        } else {
            pcl::PointCloud<::PointXYZIRT> cloud;
            pcl::fromROSMsg(cloud_msg, cloud);
            if (!cloud.empty()) {
                scan_duration = static_cast<double>(cloud.points.back().time);
            }
        }
    } catch (const std::exception& e) {
        LOG(WARNING) << "failed to estimate LIO-SAM scan duration, use mean scan time: " << e.what();
    }

    if (!std::isfinite(scan_duration) || scan_duration <= 0.0 || scan_duration > 0.5) {
        scan_duration = lidar_mean_scantime_;
    } else {
        sync_scan_num_++;
        lidar_mean_scantime_ += (scan_duration - lidar_mean_scantime_) / sync_scan_num_;
    }

    return lidar_begin_time + scan_duration;
}

bool LioSamMapping::SyncPackages() {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (cloudQueue_.empty() || imuQueue_.empty()) {
        return false;
    }

    const auto& cloud_msg = cloudQueue_.front();
    const double lidar_begin_time = ToSec(cloud_msg.header.stamp);
    const double lidar_end_time = EstimateScanEndTime(cloud_msg, lidar_begin_time);

    if (last_timestamp_imu_ < lidar_end_time) {
        return false;
    }

    const double imu_front_time = ToSec(imuQueue_.front().header.stamp);
    if (imu_front_time > lidar_begin_time) {
        LOG(WARNING) << "lio-sam missing imu before scan start, drop cloud. imu_front="
                     << std::setprecision(14) << imu_front_time << ", lidar_begin=" << lidar_begin_time;
        cloudQueue_.pop_front();
        return false;
    }

    measures_ = SyncedPackage();
    measures_.cloud = cloud_msg;
    measures_.lidar_begin_time = lidar_begin_time;
    measures_.lidar_end_time = lidar_end_time;

    const double imu_start_time = lidar_begin_time - 0.05;
    const double imu_end_time = lidar_end_time + 0.01;
    for (const auto& imu : imuQueue_) {
        const double imu_time = ToSec(imu.header.stamp);
        if (imu_time < imu_start_time) {
            continue;
        }
        if (imu_time > imu_end_time) {
            break;
        }
        measures_.imus.push_back(imu);
    }

    if (measures_.imus.empty() || ToSec(measures_.imus.front().header.stamp) > lidar_begin_time ||
        ToSec(measures_.imus.back().header.stamp) < lidar_end_time) {
        return false;
    }

    while (imuQueue_.size() > 1 && ToSec(imuQueue_[1].header.stamp) <= lidar_end_time) {
        imuQueue_.pop_front();
    }
    cloudQueue_.pop_front();

    LOG(INFO) << "lio-sam sync cloud: begin=" << std::setprecision(14) << measures_.lidar_begin_time
              << ", end=" << measures_.lidar_end_time << ", imu=" << measures_.imus.size();
    return true;
}

bool LioSamMapping::RunImageProjection() {
    auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>(measures_.cloud);
    return image_projection_->processSyncedCloud(cloud_msg, measures_.imus);
}

bool LioSamMapping::RunFeatureExtraction() {
    auto cloud_info = std::make_shared<lio_sam::msg::CloudInfo>(image_projection_->latestCloudInfo);
    feature_extraction_->laserCloudInfoHandler(cloud_info);
    return feature_extraction_->hasFeatureInfo;
}

bool LioSamMapping::RunMapOptimization() {
    auto feature_info = std::make_shared<lio_sam::msg::CloudInfo>(feature_extraction_->latestFeatureInfo);
    map_optimization_->laserCloudInfoHandler(feature_info);
    if (map_optimization_->loopClosureEnableFlag) {
        map_optimization_->performLoopClosure();
    }
    state_.timestamp_ = map_optimization_->timeLaserInfoCur;
    state_.pos_ = Vec3d(map_optimization_->transformTobeMapped[3],
                        map_optimization_->transformTobeMapped[4],
                        map_optimization_->transformTobeMapped[5]);
    state_.rot_ = RpyToSO3(map_optimization_->transformTobeMapped[0],
                           map_optimization_->transformTobeMapped[1],
                           map_optimization_->transformTobeMapped[2]);
    return true;
}

bool LioSamMapping::Run() {
    if (!image_projection_ || !feature_extraction_ || !map_optimization_) {
        return false;
    }

    if (!SyncPackages()) {
        return false;
    }

    if (!RunImageProjection()) {
        return false;
    }

    if (!RunFeatureExtraction()) {
        return false;
    }

    if (!RunMapOptimization()) {
        return false;
    }

    pcl::fromROSMsg(image_projection_->latestCloudInfo.cloud_deskewed, *scan_undistort_);
    recent_cloud_ = scan_undistort_;

    return MakeLightningKeyframeIfNeeded();
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

    if (ui_) {
        ui_->UpdateNavState(state_);
        ui_->UpdateScan(scan_undistort_, state_.GetPose());
    }

    LOG(INFO) << "LIO-SAM: create lightning keyframe " << kf->GetID() << ", pose: "
              << state_.pos_.transpose() << ", time: " << std::setprecision(14) << state_.timestamp_;
    return true;
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
