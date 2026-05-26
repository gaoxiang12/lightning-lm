#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <utility>
#include <vector>

#include "common/constant.h"
#include "common/options.h"
#include "core/localization/lidar_loc/lidar_loc.h"
#include "core/localization/localization.h"

#include <opencv2/highgui.hpp>

#include "core/localization/pose_graph/pgo.h"
#include "core/lightning_math.hpp"
#include "io/yaml_io.h"
#include "ui/pangolin_window.h"
#include <yaml-cpp/yaml.h>

namespace lightning::loc {
namespace {

template <typename T>
void ReadPgoValueIfPresent(const YAML::Node& config, const char* key, T& value) {
    if (config[key]) {
        value = config[key].as<T>();
    }
}

void LoadPgoOptions(const YAML::Node& yaml) {
    const YAML::Node config = yaml["pgo"];
    if (!config) {
        LOG(WARNING) << "PGO config section is missing, using defaults";
        return;
    }

    ReadPgoValueIfPresent(config, "lidar_loc_pos_noise", pgo::lidar_loc_pos_noise);
    ReadPgoValueIfPresent(config, "lidar_loc_outlier_th", pgo::lidar_loc_outlier_th);
    ReadPgoValueIfPresent(config, "lidar_odom_pos_noise", pgo::lidar_odom_pos_noise);
    ReadPgoValueIfPresent(config, "lidar_odom_outlier_th", pgo::lidar_odom_outlier_th);
    ReadPgoValueIfPresent(config, "dr_pos_noise", pgo::dr_pos_noise);
    ReadPgoValueIfPresent(config, "dr_pos_noise_ratio", pgo::dr_pos_noise_ratio);
    ReadPgoValueIfPresent(config, "pgo_frame_converge_pos_th", pgo::pgo_frame_converge_pos_th);
    ReadPgoValueIfPresent(config, "smooth_factor", pgo::pgo_smooth_factor);

    double lidar_loc_ang_noise_deg = pgo::lidar_loc_ang_noise / constant::kDEG2RAD;
    double lidar_odom_ang_noise_deg = pgo::lidar_odom_ang_noise / constant::kDEG2RAD;
    double dr_ang_noise_deg = pgo::dr_ang_noise / constant::kDEG2RAD;
    double converge_ang_th_deg = pgo::pgo_frame_converge_ang_th / constant::kDEG2RAD;
    ReadPgoValueIfPresent(config, "lidar_loc_ang_noise", lidar_loc_ang_noise_deg);
    ReadPgoValueIfPresent(config, "lidar_odom_ang_noise", lidar_odom_ang_noise_deg);
    ReadPgoValueIfPresent(config, "dr_ang_noise", dr_ang_noise_deg);
    ReadPgoValueIfPresent(config, "pgo_frame_converge_ang_th", converge_ang_th_deg);
    pgo::lidar_loc_ang_noise = lidar_loc_ang_noise_deg * constant::kDEG2RAD;
    pgo::lidar_odom_ang_noise = lidar_odom_ang_noise_deg * constant::kDEG2RAD;
    pgo::dr_ang_noise = dr_ang_noise_deg * constant::kDEG2RAD;
    pgo::pgo_frame_converge_ang_th = converge_ang_th_deg * constant::kDEG2RAD;

    LOG(INFO) << "PGO config loaded, lidar loc pos noise: " << pgo::lidar_loc_pos_noise
              << ", lidar loc angular noise(deg): " << lidar_loc_ang_noise_deg
              << ", lidar loc outlier threshold: " << pgo::lidar_loc_outlier_th
              << ", smoother factor: " << pgo::pgo_smooth_factor;
}

SE3 ReadTransform(const YAML::Node& node) {
    Vec3d t = Vec3d::Zero();
    Mat3d R = Mat3d::Identity();

    if (node["translation"]) {
        const auto data = node["translation"].as<std::vector<double>>();
        if (data.size() == 3) {
            t = Vec3d(data[0], data[1], data[2]);
        }
    }

    if (node["rotation"]) {
        const auto data = node["rotation"].as<std::vector<double>>();
        if (data.size() == 9) {
            R = math::MatFromArray<double>(data);
        }
    } else if (node["rpy_deg"]) {
        const auto data = node["rpy_deg"].as<std::vector<double>>();
        if (data.size() == 3) {
            const double roll = data[0] * constant::kDEG2RAD;
            const double pitch = data[1] * constant::kDEG2RAD;
            const double yaw = data[2] * constant::kDEG2RAD;
            R = (Eigen::AngleAxisd(yaw, Vec3d::UnitZ()) *
                 Eigen::AngleAxisd(pitch, Vec3d::UnitY()) *
                 Eigen::AngleAxisd(roll, Vec3d::UnitX()))
                    .toRotationMatrix();
        }
    }

    return SE3(Eigen::Quaterniond(R).normalized(), t);
}

void ConfigurePreprocess(PointCloudPreprocess& preprocess, const YAML::Node& cfg) {
    const int lidar_type = cfg["lidar_type"].as<int>();
    preprocess.Blind() = cfg["blind"].as<double>();
    preprocess.TimeScale() = cfg["time_scale"].as<float>();
    preprocess.NumScans() = cfg["scan_line"].as<int>();
    preprocess.PointFilterNum() = cfg["point_filter_num"].as<int>();

    if (lidar_type == 1) {
        preprocess.SetLidarType(LidarType::AVIA);
    } else if (lidar_type == 2) {
        preprocess.SetLidarType(LidarType::VELO32);
    } else if (lidar_type == 3) {
        preprocess.SetLidarType(LidarType::OUST64);
    } else {
        LOG(WARNING) << "unknown lidar_type " << lidar_type << ", use VELO32";
        preprocess.SetLidarType(LidarType::VELO32);
    }
}

}  // namespace

// ！ 构造函数
Localization::Localization(Options options) { options_ = options; }

// ！初始化函数
bool Localization::Init(const std::string& yaml_path, const std::string& global_map_path) {
    UL lock(global_mutex_);
    if (lidar_loc_ != nullptr) {
        // 若已经启动，则变为初始化
        Finish();
        lidar_loc_.reset();
        lio_.reset();
        pgo_.reset();
        preprocess_.reset();
    }

    YAML_IO yaml(yaml_path);
    YAML::Node yaml_node = YAML::LoadFile(yaml_path);

    options_.with_ui_ = yaml.GetValue<bool>("system", "with_ui");
    lidar_count_ = yaml.GetValue<int>("localization", "lidar_count");

    /// lidar odom前端
    LaserMapping::Options opt_lio;
    opt_lio.is_in_slam_mode_ = false;

    lio_ = std::make_shared<LaserMapping>(opt_lio);
    if (!lio_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init lio";
        return false;
    }

    /// 激光定位
    LidarLoc::Options lidar_loc_options;
    lidar_loc_options.update_dynamic_cloud_ = yaml.GetValue<bool>("lidar_loc", "update_dynamic_cloud");
    lidar_loc_options.force_2d_ = yaml.GetValue<bool>("lidar_loc", "force_2d");
    lidar_loc_options.map_option_.enable_dynamic_polygon_ = false;
    lidar_loc_options.map_option_.map_path_ = global_map_path;
    lidar_loc_ = std::make_shared<LidarLoc>(lidar_loc_options);

    if (options_.with_ui_) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->SetCurrentScanSize(1);
        ui_->Init();

        lidar_loc_->SetUI(ui_);

        // lio_->SetUI(ui_);
    }

    lidar_loc_->Init(yaml_path);

    /// pose graph
    LoadPgoOptions(yaml_node);
    pgo_ = std::make_shared<PGO>();
    pgo_->SetDebug(false);

    ///  各模块的异步调用
    options_.enable_lidar_loc_skip_ = yaml.GetValue<bool>("system", "enable_lidar_loc_skip");
    options_.enable_lidar_loc_rviz_ = yaml.GetValue<bool>("system", "enable_lidar_loc_rviz");
    options_.lidar_loc_skip_num_ = yaml.GetValue<int>("system", "lidar_loc_skip_num");
    options_.enable_lidar_odom_skip_ = yaml.GetValue<bool>("system", "enable_lidar_odom_skip");
    options_.lidar_odom_skip_num_ = yaml.GetValue<int>("system", "lidar_odom_skip_num");
    options_.loc_on_kf_ = yaml.GetValue<bool>("lidar_loc", "loc_on_kf");

    lidar_odom_proc_cloud_.SetMaxSize(1);
    lidar_loc_proc_cloud_.SetMaxSize(1);

    lidar_odom_proc_cloud_.SetName("激光里程计");
    lidar_loc_proc_cloud_.SetName("激光定位");

    // 允许跳帧
    lidar_loc_proc_cloud_.SetSkipParam(options_.enable_lidar_loc_skip_, options_.lidar_loc_skip_num_);
    lidar_odom_proc_cloud_.SetSkipParam(options_.enable_lidar_odom_skip_, options_.lidar_odom_skip_num_);

    lidar_odom_proc_cloud_.SetProcFunc([this](CloudPtr cloud) { LidarOdomProcCloud(cloud); });
    lidar_loc_proc_cloud_.SetProcFunc([this](CloudPtr cloud) { LidarLocProcCloud(cloud); });

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.Start();
        lidar_loc_proc_cloud_.Start();
    }

    /// TODO: 发布
    pgo_->SetHighFrequencyGlobalOutputHandleFunction([this](const LocalizationResult& res) {
        // if (loc_result_.timestamp_ > 0) {
        //             double loc_fps = 1.0 / (res.timestamp_ - loc_result_.timestamp_);
        //             // LOG_EVERY_N(INFO, 10) << "loc fps: " << loc_fps;
        //         }

        loc_result_ = res;

        if (tf_callback_ && loc_result_.valid_) {
            tf_callback_(loc_result_);
        }

        if (ui_) {
            ui_->UpdateNavState(loc_result_.ToNavState());
            ui_->UpdateRecentPose(loc_result_.pose_);
        }
    });

    /// 预处理器
    preprocess_.reset(new PointCloudPreprocess());
    preprocess_->Blind() = yaml.GetValue<double>("fasterlio", "blind");
    preprocess_->TimeScale() = yaml.GetValue<double>("fasterlio", "time_scale");
    int lidar_type = yaml.GetValue<int>("fasterlio", "lidar_type");
    preprocess_->NumScans() = yaml.GetValue<int>("fasterlio", "scan_line");
    preprocess_->PointFilterNum() = yaml.GetValue<int>("fasterlio", "point_filter_num");
    float height_max = yaml.GetValue<float>("roi", "height_max");
    float height_min = yaml.GetValue<float>("roi", "height_min");

    preprocess_->SetHeightROI(height_max, height_min);

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else if (lidar_type == 4) {
        preprocess_->SetLidarType(LidarType::ROBOSENSE);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
    }

    if (lidar_count_ == 2) {
        const YAML::Node cfg = yaml_node["dual_lidar_localization"];
        if (!cfg) {
            LOG(ERROR) << "localization.lidar_count=2 but dual_lidar_localization config is missing";
            return false;
        }

        if (!cfg["T_front_rear"] || !cfg["front_preprocess"] || !cfg["rear_preprocess"]) {
            LOG(ERROR) << "dual_lidar_localization requires T_front_rear, front_preprocess and rear_preprocess";
            return false;
        }

        T_front_rear_ = ReadTransform(cfg["T_front_rear"]);
        allow_single_lidar_fallback_ = cfg["allow_single_lidar_fallback"].as<bool>();

        ConfigurePreprocess(front_lidar_preprocess_, cfg["front_preprocess"]);
        ConfigurePreprocess(rear_lidar_preprocess_, cfg["rear_preprocess"]);

        LOG(INFO) << "[DUAL_LIDAR_LOC] virtual cloud frame is front_lidar. "
                  << "fasterlio.extrinsic_T/R must be T_imu_front. "
                  << "dual_lidar_localization.T_front_rear maps rear_lidar points into front_lidar frame. "
                  << "T_front_rear t=[" << T_front_rear_.translation().transpose()
                  << "], allow_single_lidar_fallback=" << allow_single_lidar_fallback_;
    }

    return true;
}

void Localization::ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
    UL lock(global_mutex_);
    // 串行模式
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
    } else {
        LidarOdomProcCloud(laser_cloud);
    }
}

void Localization::ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
    UL lock(global_mutex_);
    // 串行模式
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(laser_cloud);
    } else {
        LidarOdomProcCloud(laser_cloud);
    }
}

void Localization::ProcessDualLidarPointCloudPair(const sensor_msgs::msg::PointCloud2::SharedPtr& front_msg,
                                                  const sensor_msgs::msg::PointCloud2::SharedPtr& rear_msg) {
    if (lidar_count_ == 2) {
        ProcessDualLidarLocalizationPair(front_msg, rear_msg);
    }
}

void Localization::ProcessDualLidarLocalizationPair(const sensor_msgs::msg::PointCloud2::SharedPtr& front_msg,
                                                    const sensor_msgs::msg::PointCloud2::SharedPtr& rear_msg) {
    UL lock(global_mutex_);
    if (!front_msg || !rear_msg) {
        return;
    }

    CloudPtr front_cloud(new PointCloudType);
    CloudPtr rear_cloud(new PointCloudType);
    front_lidar_preprocess_.Process(front_msg, front_cloud);
    rear_lidar_preprocess_.Process(rear_msg, rear_cloud);

    front_cloud->header.stamp =
        static_cast<uint64_t>(front_msg->header.stamp.sec) * 1000000000ull + front_msg->header.stamp.nanosec;
    rear_cloud->header.stamp =
        static_cast<uint64_t>(rear_msg->header.stamp.sec) * 1000000000ull + rear_msg->header.stamp.nanosec;

    const double front_time = static_cast<double>(front_msg->header.stamp.sec) +
                              static_cast<double>(front_msg->header.stamp.nanosec) * 1e-9;
    const double rear_time = static_cast<double>(rear_msg->header.stamp.sec) +
                             static_cast<double>(rear_msg->header.stamp.nanosec) * 1e-9;

    const bool front_empty = front_cloud->empty();
    const bool rear_empty = rear_cloud->empty();
    if (front_empty && rear_empty) {
        LOG(WARNING) << "[DUAL_LIDAR_LOC] both clouds are empty";
        return;
    }

    if (!allow_single_lidar_fallback_ && (front_empty || rear_empty)) {
        LOG(WARNING) << "[DUAL_LIDAR_LOC] one cloud is empty and fallback disabled. front="
                     << front_cloud->size() << " rear=" << rear_cloud->size();
        return;
    }

    CloudPtr virtual_cloud = BuildVirtualFrontCloud(front_cloud, rear_cloud, front_time, rear_time);
    if (!virtual_cloud || virtual_cloud->empty()) {
        LOG(WARNING) << "[DUAL_LIDAR_LOC] virtual front cloud is empty";
        return;
    }

    LOG_EVERY_N(INFO, 10) << std::fixed << std::setprecision(9)
                          << "[DUAL_LIDAR_LOC][INPUT] front_points=" << front_cloud->size()
                          << " rear_points=" << rear_cloud->size()
                          << " virtual_points=" << virtual_cloud->size()
                          << " frame=front_lidar"
                          << " dt_ms=" << std::abs(front_time - rear_time) * 1000.0
                          << " fallback=" << allow_single_lidar_fallback_;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(virtual_cloud);
    } else {
        LidarOdomProcCloud(virtual_cloud);
    }
}

CloudPtr Localization::BuildVirtualFrontCloud(const CloudPtr& front_cloud,
                                              const CloudPtr& rear_cloud,
                                              double front_time,
                                              double rear_time) const {
    const double base_time = std::min(front_time, rear_time);
    const double front_time_offset_ms = (front_time - base_time) * 1000.0;
    const double rear_time_offset_ms = (rear_time - base_time) * 1000.0;

    CloudPtr cloud(new PointCloudType);
    cloud->reserve(front_cloud->size() + rear_cloud->size());

    for (const auto& p : front_cloud->points) {
        PointType q = p;
        q.time += front_time_offset_ms;
        cloud->push_back(q);
    }

    for (const auto& p : rear_cloud->points) {
        const Vec3d pr = p.getVector3fMap().cast<double>();
        const Vec3d pf = T_front_rear_ * pr;

        PointType q = p;
        q.x = static_cast<float>(pf.x());
        q.y = static_cast<float>(pf.y());
        q.z = static_cast<float>(pf.z());
        q.time += rear_time_offset_ms;
        cloud->push_back(q);
    }

    std::sort(cloud->points.begin(), cloud->points.end(), [](const PointType& a, const PointType& b) {
        return a.time < b.time;
    });

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;
    cloud->header.stamp = static_cast<uint64_t>(base_time * 1e9);

    return cloud;
}

void Localization::LidarOdomProcCloud(CloudPtr cloud) {
    if (!cloud) {
        return;
    }

    /// NOTE: 在NCLT这种数据集中，lio内部是有缓存的，它拿到的点云不一定是最新时刻的点云
    lio_->ProcessPointCloud2(cloud);
    if (!lio_->Run()) {
        return;
    }

    auto lo_state = lio_->GetState();

    lidar_loc_->ProcessLO(lo_state);
    pgo_->ProcessLidarOdom(lo_state);

    // LOG(INFO) << "LO pose: " << std::setprecision(12) << lo_state.timestamp_ << " "
    //           << lo_state.GetPose().translation().transpose();

    /// 获得lio的关键帧

    auto scan = lio_->GetProjCloud();

    if (options_.loc_on_kf_) {
        auto kf = lio_->GetKeyframe();
        if (kf == lio_kf_) {
            /// 关键帧未更新，那就只更新IMU状态

            // auto dr_state = lio_->GetState();
            // lidar_loc_->ProcessDR(dr_state);
            // pgo_->ProcessDR(dr_state);
            return;
        }

        // if (ui_) {
        //     ui_->UpdateKF(kf);
        // }

        lio_kf_ = kf;
    }

    // auto scan = lio_->GetScanUndist();

    if (options_.online_mode_) {
        lidar_loc_proc_cloud_.AddMessage(scan);
    } else {
        LidarLocProcCloud(scan);
    }
}

void Localization::LidarLocProcCloud(CloudPtr scan_undist) {
    lidar_loc_->ProcessCloud(scan_undist);

    auto res = lidar_loc_->GetLocalizationResult();  // ndt 绝对位姿
    pgo_->ProcessLidarLoc(res);

    if (lidar_count_ == 2) {
        LOG_EVERY_N(INFO, 10) << "[DUAL_LIDAR_LOC][NDT_RESULT] input_points=" << scan_undist->size()
                              << " input_frame=virtual_front_lidar"
                              << " score=" << res.confidence_
                              << " valid=" << res.lidar_loc_valid_;
    }

    if (ui_) {
        // Twi with Til, here pose means Twl, thus Til=I
        ui_->UpdateScan(scan_undist, res.pose_);
    }

    if (loc_state_callback_) {
        auto loc_state = std::make_shared<std_msgs::msg::Int32>();
        loc_state->data = static_cast<int>(res.status_);
        LOG(INFO) << "loc_state: " << loc_state->data;
        loc_state_callback_(*loc_state);
    }

    // cv::Mat img(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    // cv::imshow("img", img);
    // cv::waitKey(0);
}

void Localization::ProcessIMUMsg(IMUPtr imu) {
    UL lock(global_mutex_);

    if (!imu) {
        return;
    }

    double this_imu_time = imu->timestamp;
    if (last_imu_time_ > 0 && this_imu_time < last_imu_time_) {
        LOG(WARNING) << "IMU 时间异常：" << this_imu_time << ", last: " << last_imu_time_;
    }
    last_imu_time_ = this_imu_time;

    /// 里程计处理IMU
    lio_->ProcessIMU(imu);

    /// 这里需要 IMU predict，否则没法process DR了
    auto dr_state = lio_->GetIMUState();

    if (!dr_state.pose_is_ok_) {
        return;
    }

    // /// 停车判定
    // constexpr auto kThVbrbStill = 0.05;  // 0.08;
    // constexpr auto kThOmegaStill = 0.05;

    // if (dr_state.GetVel().norm() < kThVbrbStill && imu->angular_velocity.norm() < kThOmegaStill) {
    //     dr_state.is_parking_ = true;
    //     dr_state.SetVel(Vec3d::Zero());
    // }

    /// 如果没有odm, 用lio替代DR

    // LOG(INFO) << "dr state: " << std::setprecision(12) << dr_state.timestamp_ << " "
    //           << dr_state.GetPose().translation().transpose()
    //           << ", q=" << dr_state.GetPose().unit_quaternion().coeffs().transpose();

    lidar_loc_->ProcessDR(dr_state);
    pgo_->ProcessDR(dr_state);
}

// void Localization::ProcessOdomMsg(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
//     UL lock(global_mutex_);
//
//     if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
//         return;
//     }
//     double this_odom_time = ToSec(odom_msg->header.stamp);
//     if (last_odom_time_ > 0 && this_odom_time < last_odom_time_) {
//         LOG(WARNING) << "Odom Time Abnormal:" << this_odom_time << ", last: " << last_odom_time_;
//     }
//     last_odom_time_ = this_odom_time;
//
//     lio_->ProcessOdometry(odom_msg);
//
//     if (!lio_->GetbOdomHF()) {
//         return;
//     }
//
//     auto dr_state = lio_->GetStateHF(mapping::FasterLioMapping::kHFStateOdomFiltered);
//
//     constexpr auto kThVbrbStill = 0.03;  // 0.08;
//     constexpr auto kThOmegaStill = 0.03;
//     if (dr_state.Getvwi().norm() < kThVbrbStill && dr_state.Getwii().norm() < kThOmegaStill) {
//         dr_state.is_parking_ = true;
//         dr_state.Setvwi(Vec3d::Zero());
//         dr_state.Setwii(Vec3d::Zero());
//     }
//
//     lidar_loc_->ProcessDR(dr_state);
//     pgo_->ProcessDR(dr_state);
// }

void Localization::Finish() {
    if (lidar_loc_) {
        lidar_loc_->Finish();
    }
    if (ui_) {
        ui_->Quit();
    }

    lidar_loc_proc_cloud_.Quit();
    lidar_odom_proc_cloud_.Quit();
}

void Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    UL lock(global_mutex_);
    /// 设置外部重定位的pose
    if (lidar_loc_) {
        lidar_loc_->SetInitialPose(SE3(q, t));
    }
}

void Localization::SetTFCallback(Localization::TFCallback&& callback) { tf_callback_ = callback; }

}  // namespace lightning::loc
