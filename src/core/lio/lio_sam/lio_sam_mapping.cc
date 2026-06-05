#include "core/lio/lio_sam/lio_sam_mapping.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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

    ESKF::Options eskf_options;
    eskf_options.max_iterations_ = 0;
    eskf_options.epsi_ = ESKF::StateVecType::Zero();
    eskf_options.use_aa_ = false;
    kf_imu_.Init(eskf_options);

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
        //0603新增imu 外推功能
         // LIO-SAM 分支没有 p_imu_，所以这里直接构造给 ESKF::Predict 使用的 IMU 过程噪声 Q。
        // 参数沿用 fasterlio 配置，和 LaserMapping 读取的字段保持一致。
        float gyr_cov = yaml["fasterlio"]["gyr_cov"].as<float>();
        float acc_cov = yaml["fasterlio"]["acc_cov"].as<float>();
        float b_gyr_cov = yaml["fasterlio"]["b_gyr_cov"].as<float>();
        float b_acc_cov = yaml["fasterlio"]["b_acc_cov"].as<float>();

        imu_Q_.setZero();
        imu_Q_.block<3, 3>(0, 0).diagonal() = Vec3d(gyr_cov, gyr_cov, gyr_cov);
        imu_Q_.block<3, 3>(3, 3).diagonal() = Vec3d(acc_cov, acc_cov, acc_cov);
        imu_Q_.block<3, 3>(6, 6).diagonal() = Vec3d(b_gyr_cov, b_gyr_cov, b_gyr_cov);
        imu_Q_.block<3, 3>(9, 9).diagonal() = Vec3d(b_acc_cov, b_acc_cov, b_acc_cov);

        LOG(INFO) << "[LIO_SAM_DR] imu_Q loaded from fasterlio: "
                  << "gyr_cov=" << gyr_cov
                  << ", acc_cov=" << acc_cov
                  << ", b_gyr_cov=" << b_gyr_cov
                  << ", b_acc_cov=" << b_acc_cov
                  << ", Q_diag=" << imu_Q_.diagonal().transpose();

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

        // LIO-SAM 原始 IMU buffer
        imu_buffer_.clear();

        // 新增 DR buffer
        imu_dr_buffer_.clear();

        // DR 状态重新初始化
        imu_dr_inited_ = false;
        imu_mean_ready_ = false;
        imu_init_count_ = 0;
        imu_mean_acc_.setZero();
        imu_mean_gyr_.setZero();
        last_dr_imu_time_ = -1.0;
    }

    last_timestamp_imu_ = timestamp;
    imu_count_++;
    imu_buffer_.push_back(imu);
    // 0603新增imu外推
    // 3. 给新增 ESKF / DR 高频预测使用
    imu_dr_buffer_.push_back(input);
    while (imu_dr_buffer_.size() > 2000) {
        imu_dr_buffer_.pop_front();
    }
    // 4. 初始化阶段统计 IMU 均值，用于第一次设置 gravity / gyro bias
    if (!imu_mean_ready_) {
        imu_init_count_++;

        if (imu_init_count_ == 1) {
            imu_mean_acc_ = input->linear_acceleration;
            imu_mean_gyr_ = input->angular_velocity;
        } else {
            imu_mean_acc_ += (input->linear_acceleration - imu_mean_acc_) / static_cast<double>(imu_init_count_);
            imu_mean_gyr_ += (input->angular_velocity - imu_mean_gyr_) / static_cast<double>(imu_init_count_);
        }

        if (imu_init_count_ >= imu_init_min_count_) {
            imu_mean_ready_ = true;

            LOG(INFO) << "[LIO_SAM_DR] imu mean ready, acc="
                      << imu_mean_acc_.transpose()
                      << ", gyr=" << imu_mean_gyr_.transpose();
        }
    }

    // 5. 高频 DR 预测：只有 LIO-SAM Run() 成功锚定过 kf_imu_ 后才允许 Predict
    if (!imu_dr_inited_ || last_dr_imu_time_ <= 0.0) {
        return;
    }

    const double dt = timestamp - last_dr_imu_time_;

    if (dt <= 0.0) {
        return;
    }

    if (dt > 0.1) {
        LOG(WARNING) << "[LIO_SAM_DR] abnormal imu dt=" << dt
                     << ", skip predict";
        imu_dr_inited_ = false;
        last_dr_imu_time_ = -1.0;
        return;
    }

    kf_imu_.Predict(dt,
                    imu_Q_,
                    input->angular_velocity,
                    input->linear_acceleration);

    // Predict() 已经更新 x_，这里不再 ChangeX()。
    // 只显式更新时间，保证 PGO 插值用的 timestamp 正确。
    kf_imu_.SetTime(timestamp);
    last_dr_imu_time_ = timestamp;
}

void LioSamMapping::ProcessPointCloud2(CloudPtr cloud) {
    const double timestamp = math::ToSec(cloud->header.stamp);

    pcl::PointCloud<::VelodynePointXYZIRT> native_cloud;
    native_cloud.header.frame_id = cloud->header.frame_id;
    native_cloud.header.stamp = static_cast<std::uint64_t>(std::llround(timestamp * 1e6));
    native_cloud.reserve(cloud->size());
    for (const auto& source : cloud->points) {
        if (!std::isfinite(source.x) ||
            !std::isfinite(source.y) ||
            !std::isfinite(source.z) ||
            !std::isfinite(source.time)) {
            continue;
        }
        ::VelodynePointXYZIRT point;
        point.x = source.x;
        point.y = source.y;
        point.z = source.z;
        point.intensity = source.intensity;
        point.ring = source.ring;
        // PointCloudPreprocess::RoboSenseHandler 输出 source.time 单位是 ms；
        // LIO-SAM VelodynePointXYZIRT::time 需要 seconds。
        point.time = static_cast<float>(source.time * 1e-3);
        native_cloud.push_back(point);
    }
    native_cloud.height = 1;
    native_cloud.width = native_cloud.size();
    native_cloud.is_dense = true;
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(native_cloud, msg);
    msg.header.stamp = ToRosStamp(timestamp);
    msg.header.frame_id = cloud->header.frame_id;
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lio-sam lidar loop back, clear buffer";
        lidar_buffer_.clear();
        time_buffer_.clear();
        lidar_pushed_ = false;

    }
    //为了在线定位，不积攒旧帧，只处理最新的帧
    // 只在 run_loc_online 这类在线定位模式丢旧雷达。
    // run_loc_offline 离线定位不能丢，否则离线评估/回放不完整。
    // run_slam_offline 建图也不能丢。
    if (options_.online_mode_ && !options_.is_in_slam_mode_) {
        if (!lidar_buffer_.empty()) {
            LOG_EVERY_N(WARNING, 20)
                << "[LIO_SAM_ONLINE_DROP] drop stale lidar in internal buffer, size="
                << lidar_buffer_.size()
                << ", new_t=" << std::setprecision(14) << timestamp
                << ", latest_imu=" << last_timestamp_imu_;
        }

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
        lo::lidar_time_interval = scan_duration;
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

    const float* transform = map_optimization_->TransformTobeMapped();
    state_.timestamp_ = measures_.lidar_end_time;
    state_.pos_ = Vec3d(transform[3], transform[4], transform[5]);
    state_.rot_ = RpyToSO3(transform[0], transform[1], transform[2]);
    state_.pose_is_ok_ = map_optimization_->mappingPoseReliable;
    state_.lidar_odom_reliable_ = map_optimization_->mappingPoseReliable;
    //0603 imu外推,高频发布
    if (state_.pose_is_ok_) {
        std::lock_guard<std::mutex> lock(mtx_buffer_);
        NavState x;

        if (imu_dr_inited_) {
            x = kf_imu_.GetX();
        } else {
            x = NavState();

            if (imu_mean_ready_ && imu_mean_acc_.norm() > 1e-3) {
                x.grav_ = -imu_mean_acc_ / imu_mean_acc_.norm() * 9.81;
                x.bg_ = imu_mean_gyr_;
            } else {
                x.grav_ = Vec3d(0.0, 0.0, -9.81);
                x.bg_ = Vec3d::Zero();
            }

            x.vel_ = Vec3d::Zero();
        }

        // LIO-SAM 只给 pose，不给速度。
        // 所以速度用相邻 LIO-SAM pose 估一个，作为 IMU 外推初值。
        if (last_lio_anchor_time_ > 0.0) {
            const double dt_lio = state_.timestamp_ - last_lio_anchor_time_;
            if (dt_lio > 0.02 && dt_lio < 1.0) {
                Vec3d v_lio = (state_.pos_ - last_lio_anchor_pos_) / dt_lio;
                if (v_lio.norm() < 5.0) {
                    x.vel_ = v_lio;
                }
            }
        }

        // 用 LIO-SAM 低频优化位姿重置 DR 锚点
        x.timestamp_ = state_.timestamp_;
        x.pos_ = state_.pos_;
        x.rot_ = state_.rot_;
        x.pose_is_ok_ = true;
        x.lidar_odom_reliable_ = state_.lidar_odom_reliable_;

        kf_imu_.ChangeX(x);
        kf_imu_.SetTime(state_.timestamp_);

        imu_dr_inited_ = true;
        last_dr_imu_time_ = state_.timestamp_;
        last_lio_anchor_time_ = state_.timestamp_;
        last_lio_anchor_pos_ = state_.pos_;

        // replay scan end 之后已经收到的 IMU，把 kf_imu_ 追到最新 IMU 时刻
        double t = state_.timestamp_;
        for (const auto& imu_ptr : imu_dr_buffer_) {
            if (imu_ptr->timestamp <= t) {
                continue;
            }

            const double dt = imu_ptr->timestamp - t;
            if (dt <= 0.0) {
                continue;
            }
            if (dt > 0.1) {
                LOG(WARNING) << "[LIO_SAM_DR] replay abnormal imu dt=" << dt
                            << ", stop replay at t=" << std::setprecision(14) << t;
                break;
            }

            kf_imu_.Predict(dt, imu_Q_, imu_ptr->angular_velocity, imu_ptr->linear_acceleration);
            t = imu_ptr->timestamp;
        }

        kf_imu_.SetTime(t);
        last_dr_imu_time_ = t;

        static int dr_anchor_count = 0;
        if (++dr_anchor_count % 20 == 0) {
            const auto& dr = kf_imu_.GetX();
            LOG(INFO) << "[LIO_SAM_DR_ANCHOR] lidar_t=" << std::setprecision(14) << state_.timestamp_
                    << ", dr_t=" << dr.timestamp_
                    << ", pos=" << dr.pos_.transpose()
                    << ", vel=" << dr.vel_.transpose()
                    << ", grav=" << dr.grav_.transpose()
                    << ", bg=" << dr.bg_.transpose();
        }
    }

    // 显示
    scan_undistort_->clear();
    if (cloud_info.cloud_deskewed) {
        scan_undistort_->reserve(cloud_info.cloud_deskewed->size());
        for (const auto& p : cloud_info.cloud_deskewed->points) {
            PointType pt;
            pt.x = p.x;
            pt.y = p.y;
            pt.z = p.z;
            pt.intensity = p.intensity;
            pt.ring = 0.0;
            pt.time = 0.0;
            scan_undistort_->push_back(pt);
        }
    }
    scan_undistort_->header.stamp = static_cast<std::uint64_t>(std::llround(state_.timestamp_ * 1e9));
    scan_undistort_->header.frame_id = measures_.cloud.header.frame_id;
    scan_undistort_->height = 1;
    scan_undistort_->width = scan_undistort_->size();
    scan_undistort_->is_dense = true;
    recent_cloud_.reset(new PointCloudType(*scan_undistort_));

    if (ui_) {
        ui_->UpdateNavState(state_);
        ui_->UpdateScan(scan_undistort_, state_.GetPose());
    }

    MakeLightningKeyframeIfNeeded();
    //SyncLightningKeyframePoses();
    LOG(INFO) << "[LIO_SAM_OUTPUT] scan_header="
          << std::setprecision(14)
          <<  math::ToSec(scan_undistort_->header.stamp)
          << ", duration=" << lo::lidar_time_interval
          << ", loc_time="<< measures_.lidar_end_time
          << ", state_time=" << state_.timestamp_
          << ", begin=" << measures_.lidar_begin_time
          << ", end=" << measures_.lidar_end_time
          << ", scan_size=" << scan_undistort_->size()
          << ", reliable=" << state_.lidar_odom_reliable_;    
    return true;
}

bool LioSamMapping::MakeLightningKeyframeIfNeeded() {
    if (!map_optimization_ || !map_optimization_->CreatedNewKeyframe() ||
        map_optimization_->KeyPoseSize() == 0 ||
        map_optimization_->KeyPoseSize() <= native_keyframe_count_) {
        return false;
    }
    pcl::PointCloud<::PointType>::Ptr native_cloud = map_optimization_->LatestRawCloudKeyFrame();

    CloudPtr cloud(new PointCloudType());
    if (native_cloud) {
        cloud->header = scan_undistort_->header;
        cloud->reserve(native_cloud->size());
        for (const auto& p : native_cloud->points) {
            PointType pt;
            pt.x = p.x;
            pt.y = p.y;
            pt.z = p.z;
            pt.intensity = p.intensity;
            pt.ring = 0;
            pt.time = 0.0;
            cloud->push_back(pt);
        }
    }
    cloud->height = 1;
    cloud->width = cloud->size();
    cloud->is_dense = true;

    auto kf = std::make_shared<Keyframe>(kf_id_++, cloud, state_);
    kf->SetLIOPose(state_.GetPose());
    kf->SetOptPose(state_.GetPose());
    kf->SetState(state_);

    all_keyframes_.push_back(kf);
    last_kf_ = kf;
    native_keyframe_count_ = map_optimization_->KeyPoseSize();
    map_optimization_->ClearCreatedNewKeyframe();

    LOG(INFO) << "LIO-SAM: create lightning keyframe " << kf->GetID() << ", pose: "
              << state_.pos_.transpose() << ", time: " << std::setprecision(14) << state_.timestamp_;
    return true;
}

void LioSamMapping::SyncLightningKeyframePoses() {
    if (!map_optimization_) {
        return;
    }

    const size_t n = std::min(all_keyframes_.size(), map_optimization_->KeyPoseSize());
    for (size_t i = 0; i < n; ++i) {
        const auto pose = map_optimization_->KeyPose(i);
        SE3 opt_pose(
            RpyToSO3(pose.roll, pose.pitch, pose.yaw),
            Vec3d(pose.x, pose.y, pose.z));

        all_keyframes_[i]->SetLIOPose(opt_pose);
        all_keyframes_[i]->SetOptPose(opt_pose);
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
