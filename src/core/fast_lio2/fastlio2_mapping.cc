// FAST-LIO2 适配器实现
// 实现 LIOFrontend 接口，封装 FASTLIO2Core

#include "core/fast_lio2/fastlio2_mapping.h"
#include "core/fast_lio2/fastlio2_core.h"
#include "common/keyframe.h"
#include "ui/pangolin_window.h"

#include <yaml-cpp/yaml.h>
#include <glog/logging.h>
#include <algorithm>

namespace lightning {

FASTLIO2Mapping::FASTLIO2Mapping() {
    core_ = std::make_shared<FASTLIO2Core>();
    preprocess_ = std::make_shared<PointCloudPreprocess>();  // 基类成员，子类初始化
}

FASTLIO2Mapping::~FASTLIO2Mapping() = default;

bool FASTLIO2Mapping::Init(const std::string& config_yaml) {
    auto yaml = YAML::LoadFile(config_yaml);

    // 读取调试开关
    if (yaml["common"] && yaml["common"]["debug"]) {
        debug_ = yaml["common"]["debug"].as<bool>();
    }

    // 公共参数：优先从 lio_common 读取，回退到 fast_lio2
    auto get_common = [&](const std::string& key, auto default_val) {
        if (yaml["lio_common"] && yaml["lio_common"][key]) {
            return yaml["lio_common"][key].as<decltype(default_val)>();
        }
        if (yaml["fast_lio2"] && yaml["fast_lio2"][key]) {
            return yaml["fast_lio2"][key].as<decltype(default_val)>();
        }
        return default_val;
    };

    // 读取 FAST-LIO2 配置
    FASTLIO2Core::Config config;
    config.lidar_type = get_common("lidar_type", 4);
    config.scan_line = get_common("scan_line", 128);
    config.blind = get_common("blind", 0.5);
    config.filter_size_scan = get_common("filter_size_scan", 0.5);
    config.filter_size_map = get_common("filter_size_map", 0.5);
    config.max_iteration = get_common("max_iteration", 4);
    config.acc_cov = get_common("acc_cov", 0.1);
    config.gyr_cov = get_common("gyr_cov", 0.1);
    config.b_acc_cov = get_common("b_acc_cov", 0.0001);
    config.b_gyr_cov = get_common("b_gyr_cov", 0.0001);

    auto get_extrinsic = [&](const std::string& key, std::vector<double> default_val) {
        return get_common(key, default_val);
    };
    auto T = get_extrinsic("extrinsic_T", {0, 0, 0.0});
    config.extrinsic_T = Vec3d(T[0], T[1], T[2]);
    auto R = get_extrinsic("extrinsic_R", {1, 0, 0, 0, 1, 0, 0, 0, 1});
    config.extrinsic_R = Eigen::Map<const Mat3d>(R.data());

    // 读取关键帧参数
    if (yaml["fasterlio"]) {
        kf_dis_th_ = yaml["fasterlio"]["kf_dis_th"].as<double>(2.0);
        kf_angle_th_ = yaml["fasterlio"]["kf_angle_th"].as<double>(15.0) * M_PI / 180.0;
    }

    if (!core_->Init(config)) {
        LOG(ERROR) << "failed to init FASTLIO2Core";
        return false;
    }

    // 设置点云预处理
    preprocess_->Set(static_cast<LidarType>(config.lidar_type), config.blind, 4);

    // 设置高度ROI（与aa_fasterlio一致）
    if (yaml["roi"]) {
        float height_max = yaml["roi"]["height_max"].as<float>(5.0);
        float height_min = yaml["roi"]["height_min"].as<float>(-2.0);
        preprocess_->SetHeightROI(height_max, height_min);
        LOG(INFO) << "FASTLIO2 height ROI: [" << height_min << ", " << height_max << "]";
    }

    LOG(INFO) << "FASTLIO2Mapping initialized successfully";
    return true;
}

bool FASTLIO2Mapping::Run() {
    static int run_cnt = 0;
    run_cnt++;
    LOG_EVERY_N(INFO, 50) << "FASTLIO2 Run called, cnt=" << run_cnt;

    FASTLIO2MeasureGroup measures;

    if (!SyncPackages()) {
        LOG_EVERY_N(WARNING, 50) << "FASTLIO2 SyncPackages FAILED, lidar_buf=" << lidar_buffer_.size()
                                   << " imu_buf=" << imu_buffer_.size();
        return false;
    }

    // 构建 MeasureGroup
    measures.lidar = lidar_buffer_.front();
    measures.lidar_beg_time = time_buffer_.front();
    measures.lidar_end_time = lidar_end_time_;

    // 使用基类同步的 IMU 数据
    measures.imu.insert(measures.imu.end(), synced_imu_.begin(), synced_imu_.end());

    // 弹出已处理的雷达帧
    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;

    // IMU 处理 + 去畸变
    CloudPtr scan_undistort;
    core_->IMUProcess(measures, scan_undistort);

    if (scan_undistort == nullptr || scan_undistort->empty()) {
        LOG_EVERY_N(WARNING, 50) << "FASTLIO2 scan_undistort empty";
        return false;
    }

    scan_undistort_ = scan_undistort;

    static int dbg_cnt = 0;
    if (dbg_cnt++ < 3) {
        LOG(INFO) << "FASTLIO2 scan_undistort pts=" << scan_undistort_->size();
    }

    // 降采样
    DownSample();

    if (core_->GetFeatsDownSize() < 5) {
        LOG_EVERY_N(WARNING, 50) << "FASTLIO2 feats too few: " << core_->GetFeatsDownSize()
                                   << " scan_body_pts=" << (core_->GetScanDownBody() ? core_->GetScanDownBody()->size() : 0);
        return false;
    }

    // 首帧初始化 ikd-Tree
    if (core_->GetKDTree().Root_Node == nullptr) {
        LOG(INFO) << "FASTLIO2 init ikd-Tree, pts=" << core_->GetFeatsDownSize();
        if (core_->GetFeatsDownSize() > 5) {
            auto scan_world = std::make_shared<PointCloudType>();
            scan_world->resize(core_->GetFeatsDownSize());
            auto scan_body = core_->GetScanDownBody();
            for (int i = 0; i < core_->GetFeatsDownSize(); i++) {
                core_->PointBodyToWorld(scan_body->points[i], scan_world->points[i]);
            }
            core_->GetKDTree().Build(scan_world->points);
            core_->GetKDTree().set_downsample_param(0.5);
        }
        return false;
    }

    // 观测更新
    core_->Observe(core_->GetScanDownBody());

    // 地图更新
    core_->UpdateMap(nullptr);

    // 获取结果
    auto core_state = core_->GetNativeState();
    native_state_.pos = core_state.pos;
    native_state_.rot = core_state.rot;
    native_state_.offset_R_L_I = core_state.offset_R_L_I;
    native_state_.offset_T_L_I = core_state.offset_T_L_I;
    native_state_.vel = core_state.vel;
    native_state_.bg = core_state.bg;
    native_state_.ba = core_state.ba;
    native_state_.grav = core_state.grav;

    if (debug_ && run_cnt <= 5) {
        LOG(INFO) << "FASTLIO2 frame " << run_cnt
                  << " pos: " << native_state_.pos.transpose()
                  << " vel: " << native_state_.vel.transpose()
                  << " bg: " << native_state_.bg.transpose()
                  << " ext_R: " << native_state_.offset_R_L_I.diagonal().transpose()
                  << " ext_T: " << native_state_.offset_T_L_I.transpose();
    }

    // 转换降采样点云到世界坐标系
    scan_down_world_->clear();
    scan_down_world_->resize(core_->GetFeatsDownSize());
    auto scan_body = core_->GetScanDownBody();
    for (int i = 0; i < core_->GetFeatsDownSize(); i++) {
        core_->PointBodyToWorld(scan_body->points[i], scan_down_world_->points[i]);
    }

    // 推送点云给 UI（传体坐标系点云，UI 内部会用 pose 变换到世界坐标）
    if (ui_) {
        ui_->UpdateScan(scan_down_body_, GetPose());
    }

    // 创建关键帧
    if (all_keyframes_.empty()) {
        LOG(INFO) << "FASTLIO2 first KF, pos=" << GetPose().translation().transpose();
        MakeKF();
    } else {
        // 检查是否需要创建新关键帧
        SE3 cur_pose = GetPose();
        SE3 last_pose = all_keyframes_.back()->GetOptPose();
        Vec3d delta_t = cur_pose.translation() - last_pose.translation();
        double delta_angle = (cur_pose.so3().inverse() * last_pose.so3()).log().norm();

        static int skip_cnt = 0;
        if (skip_cnt++ % 50 == 0) {
            LOG(INFO) << "FASTLIO2 KF check: delta_t=" << delta_t.norm()
                      << " delta_ang=" << delta_angle
                      << " th_dis=" << kf_dis_th_ << " th_ang=" << kf_angle_th_
                      << " kf_count=" << all_keyframes_.size();
        }

        if (delta_t.norm() > kf_dis_th_ || delta_angle > kf_angle_th_) {
            MakeKF();
        }
    }

    return true;
}


void FASTLIO2Mapping::DownSample() {
    auto scan_body = core_->GetScanDownBody();
    core_->GetKDTree().set_downsample_param(0.5);

    static int dbg_cnt = 0;
    if (dbg_cnt++ < 3 && scan_undistort_->size() > 0) {
        auto& pts = scan_undistort_->points;
        float min_x = pts[0].x, max_x = pts[0].x;
        float min_y = pts[0].y, max_y = pts[0].y;
        float min_z = pts[0].z, max_z = pts[0].z;
        for (size_t i = 1; i < pts.size(); i++) {
            if (pts[i].x < min_x) min_x = pts[i].x;
            if (pts[i].x > max_x) max_x = pts[i].x;
            if (pts[i].y < min_y) min_y = pts[i].y;
            if (pts[i].y > max_y) max_y = pts[i].y;
            if (pts[i].z < min_z) min_z = pts[i].z;
            if (pts[i].z > max_z) max_z = pts[i].z;
        }
        LOG(INFO) << "FASTLIO2 scan range: x[" << min_x << "," << max_x
                  << "] y[" << min_y << "," << max_y
                  << "] z[" << min_z << "," << max_z << "]"
                  << " size=" << scan_undistort_->size();
    }

    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(0.5, 0.5, 0.5);
    downSizeFilter.setInputCloud(scan_undistort_);
    downSizeFilter.filter(*scan_body);

    // 保存体坐标系点云（用于关键帧存储）
    *scan_down_body_ = *scan_body;

    // 同步 feats_down_size_（在 Observe 之前需要知道特征数来决定是否继续）
    core_->SetFeatsDownSize(scan_body->size());
}

void FASTLIO2Mapping::MakeKF() {
    auto state = core_->GetNativeState();
    NavState nav_state;
    nav_state.timestamp_ = lidar_end_time_;
    nav_state.pos_ = state.pos;
    nav_state.rot_ = SO3(state.rot);
    nav_state.vel_ = state.vel;
    nav_state.bg_ = state.bg;

    auto kf = std::make_shared<Keyframe>(kf_id_++, scan_down_body_, nav_state);

    LOG(INFO) << "LIO: create kf " << kf->GetID()
              << ", state: " << nav_state.pos_.transpose()
              << ", kf opt pose: " << kf->GetOptPose().translation().transpose()
              << ", lio pose: " << kf->GetLIOPose().translation().transpose()
              << ", time: " << std::setprecision(14) << lidar_end_time_;

    last_kf_ = kf;
    all_keyframes_.push_back(kf);
}

SE3 FASTLIO2Mapping::GetPose() const {
    return core_->GetPose();
}

CloudPtr FASTLIO2Mapping::GetScanUndist() const {
    return scan_undistort_;
}

CloudPtr FASTLIO2Mapping::GetScanDownWorld() const {
    return scan_down_world_;
}

Keyframe::Ptr FASTLIO2Mapping::GetKeyframe() const {
    return last_kf_;
}

std::vector<Keyframe::Ptr> FASTLIO2Mapping::GetAllKeyframes() {
    return all_keyframes_;
}

void FASTLIO2Mapping::SaveMap() {
    // TODO: 实现地图保存
    LOG(INFO) << "FASTLIO2Mapping SaveMap not implemented yet";
}

void FASTLIO2Mapping::PrintExtrinsic() {
    LOG(INFO) << "FASTLIO2 PrintExtrinsic: pos=" << native_state_.pos.transpose()
              << " offset_T=" << native_state_.offset_T_L_I.transpose();
    Vec3d t = native_state_.offset_T_L_I;
    Mat3d R = native_state_.offset_R_L_I;

    Eigen::Quaterniond q(R);
    q.normalize();

    LOG(INFO) << "========== FAST-LIO2 Extrinsic (LiDAR→IMU) ==========";
    LOG(INFO) << "Translation: " << t.transpose();
    LOG(INFO) << "Rotation (quaternion wxyz): " << q.w() << " " << q.x() << " " << q.y() << " " << q.z();
    LOG(INFO) << "Rotation (matrix): ";
    LOG(INFO) << R;
    LOG(INFO) << "======================================================";
}

CloudPtr FASTLIO2Mapping::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) {
    CloudPtr global_map(new PointCloudType());

    for (auto& kf : all_keyframes_) {
        auto cloud = kf->GetCloud();
        if (cloud == nullptr) continue;

        SE3 pose = use_lio_pose ? kf->GetLIOPose() : kf->GetOptPose();

        for (auto& pt : cloud->points) {
            PointType pt_world;
            Vec3d p_body(pt.x, pt.y, pt.z);
            Vec3d p_global = pose * p_body;
            pt_world.x = p_global(0);
            pt_world.y = p_global(1);
            pt_world.z = p_global(2);
            pt_world.intensity = pt.intensity;
            global_map->points.push_back(pt_world);
        }
    }

    // 降采样
    if (use_voxel && res > 0) {
        pcl::VoxelGrid<PointType> filter;
        filter.setLeafSize(res, res, res);
        filter.setInputCloud(global_map);
        CloudPtr filtered(new PointCloudType());
        filter.filter(*filtered);
        return filtered;
    }

    return global_map;
}

FASTLIO2Mapping::NativeState FASTLIO2Mapping::GetNativeState() const {
    return native_state_;
}

}  // namespace lightning
