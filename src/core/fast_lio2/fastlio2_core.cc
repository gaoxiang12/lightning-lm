// FAST-LIO2 核心实现
// 封装 ESEKF + ikd-Tree，适配 Lightning-LM 数据结构

#include "core/fast_lio2/fastlio2_core.h"
#include "core/frontend/undistortion.h"
#include <glog/logging.h>
#include <algorithm>
#include <cmath>

namespace lightning {

// 静态实例指针
FASTLIO2Core* FASTLIO2Core::instance_ = nullptr;

// 观测模型回调 (友元函数，访问 Core 内部状态)
void fastlio2_h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data) {
    auto* core = FASTLIO2Core::GetInstance();
    if (!core) {
        ekfom_data.valid = false;
        return;
    }

    core->laserCloudOri_->clear();
    core->corr_normvect_->clear();
    double total_residual = 0.0;

    for (int i = 0; i < core->feats_down_size_; i++) {
        PointType& point_body = core->feats_down_body_->points[i];
        PointType& point_world = core->feats_down_world_->points[i];

        // Transform to world frame
        Vec3d p_body(point_body.x, point_body.y, point_body.z);
        Vec3d p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<float> pointSearchSqDis(FASTLIO2Core::NUM_MATCH_POINTS);
        auto& points_near = core->Nearest_Points[i];

        if (ekfom_data.converge) {
            core->ikdtree_.Nearest_Search(point_world, FASTLIO2Core::NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            core->point_selected_surf_[i] = points_near.size() < FASTLIO2Core::NUM_MATCH_POINTS ? false :
                                            pointSearchSqDis[FASTLIO2Core::NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!core->point_selected_surf_[i]) continue;

        Eigen::Vector4f pabcd;
        core->point_selected_surf_[i] = false;

        // Plane estimation
        if (points_near.size() >= FASTLIO2Core::NUM_MATCH_POINTS) {
            Eigen::MatrixXf A(FASTLIO2Core::NUM_MATCH_POINTS, 3);
            Eigen::MatrixXf b(FASTLIO2Core::NUM_MATCH_POINTS, 1);
            b.setOnes();
            b *= -1.0f;

            for (int j = 0; j < FASTLIO2Core::NUM_MATCH_POINTS; j++) {
                A(j, 0) = points_near[j].x;
                A(j, 1) = points_near[j].y;
                A(j, 2) = points_near[j].z;
            }

            Eigen::Vector3f normvec = A.colPivHouseholderQr().solve(b);
            float n = normvec.norm();
            pabcd(0) = normvec(0) / n;
            pabcd(1) = normvec(1) / n;
            pabcd(2) = normvec(2) / n;
            pabcd(3) = 1.0f / n;

            bool plane_valid = true;
            for (int j = 0; j < FASTLIO2Core::NUM_MATCH_POINTS; j++) {
                if (std::fabs(pabcd(0) * points_near[j].x + pabcd(1) * points_near[j].y +
                              pabcd(2) * points_near[j].z + pabcd(3)) > 0.1f) {
                    plane_valid = false;
                    break;
                }
            }

            if (plane_valid) {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                            pabcd(2) * point_world.z + pabcd(3);
                float s = 1 - 0.9f * std::fabs(pd2) / sqrt(p_body.norm());

                if (s > 0.9f) {
                    core->point_selected_surf_[i] = true;
                    core->normvec_->points[i].x = pabcd(0);
                    core->normvec_->points[i].y = pabcd(1);
                    core->normvec_->points[i].z = pabcd(2);
                    core->normvec_->points[i].intensity = pd2;
                    core->res_last_[i] = std::fabs(pd2);
                }
            }
        }
    }

    core->effct_feat_num_ = 0;
    for (int i = 0; i < core->feats_down_size_; i++) {
        if (core->point_selected_surf_[i]) {
            core->laserCloudOri_->points[core->effct_feat_num_] = core->feats_down_body_->points[i];
            core->corr_normvect_->points[core->effct_feat_num_] = core->normvec_->points[i];
            total_residual += core->res_last_[i];
            core->effct_feat_num_++;
        }
    }

    if (core->effct_feat_num_ < 1) {
        ekfom_data.valid = false;
        return;
    }

    // Computation of Measurement Jacobian matrix H and measurement vector
    // 注意: IKFoM ESEKF 使用 12 维观测雅可比（仅位姿部分），外参和重力通过 MTK 流形机制处理
    ekfom_data.h_x = Eigen::MatrixXd::Zero(core->effct_feat_num_, 12);
    ekfom_data.h.resize(core->effct_feat_num_);

    for (int i = 0; i < core->effct_feat_num_; i++) {
        const PointType& laser_p = core->laserCloudOri_->points[i];
        Vec3d point_this_be(laser_p.x, laser_p.y, laser_p.z);
        Mat3d point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        Vec3d point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        Mat3d point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        const PointType& norm_p = core->corr_normvect_->points[i];
        Vec3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

        Vec3d C(s.rot.conjugate() * norm_vec);
        Vec3d A(point_crossmat * C);

        // pos(0-2), rot(3-5), vel(6-8), bg(9-11) — 仅位姿和速度/偏置部分
        ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
            A(0), A(1), A(2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        ekfom_data.h(i) = -norm_p.intensity;
    }
}

FASTLIO2Core::FASTLIO2Core() {
    instance_ = this;
    Lidar_R_wrt_IMU_ = Mat3d::Identity();
    Lidar_T_wrt_IMU_ = Vec3d::Zero();
    Q_ = process_noise_cov();
    last_lidar_end_time_ = 0.0;
}

FASTLIO2Core::~FASTLIO2Core() {
    if (instance_ == this) instance_ = nullptr;
}

bool FASTLIO2Core::Init(const Config& config) {
    config_ = config;

    // 设置外参
    Lidar_T_wrt_IMU_ = config.extrinsic_T;
    Lidar_R_wrt_IMU_ = config.extrinsic_R;

    // 设置 IMU 噪声
    Q_.block<3, 3>(0, 0).diagonal() = Vec3d(config.gyr_cov, config.gyr_cov, config.gyr_cov);
    Q_.block<3, 3>(3, 3).diagonal() = Vec3d(config.acc_cov, config.acc_cov, config.acc_cov);
    Q_.block<3, 3>(6, 6).diagonal() = Vec3d(config.b_gyr_cov, config.b_gyr_cov, config.b_gyr_cov);
    Q_.block<3, 3>(9, 9).diagonal() = Vec3d(config.b_acc_cov, config.b_acc_cov, config.b_acc_cov);

    // 初始化 ESEKF
    double epsi[23] = {0.001};
    std::fill(epsi, epsi + 23, 0.001);
    kf_.init_dyn_share(get_f, df_dx, df_dw, fastlio2_h_share_model, config.max_iteration, epsi);

    // 初始化降采样
    downSizeFilterSurf_.setLeafSize(config.filter_size_scan, config.filter_size_scan, config.filter_size_scan);
    downSizeFilterMap_.setLeafSize(config.filter_size_map, config.filter_size_map, config.filter_size_map);

    LOG(INFO) << "FASTLIO2Core initialized: scan_line=" << config.scan_line
              << " filter_size_scan=" << config.filter_size_scan
              << " filter_size_map=" << config.filter_size_map;

    return true;
}

void FASTLIO2Core::IMU_init(const FASTLIO2MeasureGroup& meas, int& N) {
    Vec3d cur_acc, cur_gyr;

    if (b_first_frame_) {
        mean_acc_ = Vec3d(0, 0, -1.0);
        mean_gyr_ = Vec3d(0, 0, 0);
        angvel_last_ = Vec3d::Zero();
        imu_need_init_ = true;
        init_iter_num_ = 1;

        N = 1;
        b_first_frame_ = false;
        first_lidar_time_ = meas.lidar_beg_time;
    }

    for (const auto& imu : meas.imu) {
        cur_acc = imu->linear_acceleration;
        cur_gyr = imu->angular_velocity;

        mean_acc_ += (cur_acc - mean_acc_) / N;
        mean_gyr_ += (cur_gyr - mean_gyr_) / N;

        Vec3d cov_acc_delta = (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_);
        Vec3d cov_gyr_delta = (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_);

        double k = (N - 1.0) / N;
        for (int i = 0; i < 3; i++) {
            cov_acc_(i) = cov_acc_(i) * k + cov_acc_delta(i) * (N - 1.0) / (N * N);
            cov_gyr_(i) = cov_gyr_(i) * k + cov_gyr_delta(i) * (N - 1.0) / (N * N);
        }

        N++;
    }

    state_ikfom init_state = kf_.get_x();
    init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * G_m_s2);
    init_state.bg = mean_gyr_;
    init_state.offset_T_L_I = Lidar_T_wrt_IMU_;
    init_state.offset_R_L_I = Lidar_R_wrt_IMU_;
    kf_.change_x(init_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_.get_P();
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_.change_P(init_P);
}

void FASTLIO2Core::IMUProcess(const FASTLIO2MeasureGroup& measures, CloudPtr& scan_out) {
    if (measures.imu.empty()) {
        return;
    }

    if (imu_need_init_) {
        IMU_init(measures, init_iter_num_);

        // Use first actual IMU timestamp so prepended sample is close to real data
        last_imu_pose_.offset_time = measures.imu.front()->timestamp;
        last_imu_pose_.acc = measures.imu.front()->linear_acceleration;
        last_imu_pose_.gyr = measures.imu.front()->angular_velocity;
        last_imu_pose_.vel = Vec3d::Zero();
        last_imu_pose_.pos = Vec3d::Zero();
        last_imu_pose_.rot = Mat3d::Identity();

        if (init_iter_num_ > MAX_INI_COUNT) {
            cov_acc_ = cov_acc_.cwiseProduct(Vec3d(G_m_s2 / mean_acc_.norm(),
                                                   G_m_s2 / mean_acc_.norm(),
                                                   G_m_s2 / mean_acc_.norm()));
            imu_need_init_ = false;

            // 从 ESEKF 状态同步 last_imu_pose_，避免使用零/单位阵的初始值
            state_ikfom imu_state = kf_.get_x();
            last_imu_pose_.rot = imu_state.rot.toRotationMatrix();
            last_imu_pose_.pos = imu_state.pos;
            last_imu_pose_.vel = imu_state.vel;
            // Sync offset_time to last IMU timestamp so first propagation has correct dt
            last_imu_pose_.offset_time = measures.imu.back()->timestamp;

            LOG(INFO) << "IMU Initial Done, gravity norm: " << mean_acc_.norm();
            if (FASTLIO2Core::GetInstance()) {
                LOG(INFO) << "  mean_acc: " << mean_acc_.transpose()
                          << " mean_gyr: " << mean_gyr_.transpose();
            }
        }

        return;
    }

    // Forward propagation at each IMU point
    auto v_imu = measures.imu;
    v_imu.push_front(std::make_shared<IMU>(IMU{
        last_imu_pose_.offset_time,
        last_imu_pose_.gyr,
        last_imu_pose_.acc}));

    const double pcl_beg_time = measures.lidar_beg_time;
    const double pcl_end_time = measures.lidar_end_time;
    const double imu_end_time = v_imu.back()->timestamp;

    // Sort point clouds by time (use .time field for PointXYZIT)
    scan_out = std::make_shared<PointCloudType>(*measures.lidar);

    // 检查 point.time 是否有效（对 RoboSense 等 timestamp 字段不可靠的雷达，
    // time 可能是垃圾值）。如果无效，基于点索引分配线性插值时间戳。
    bool time_valid = true;
    const double scan_duration = pcl_end_time - pcl_beg_time;
    if (scan_out->size() > 1 && scan_duration > 0) {
        // point.time 单位是毫秒，期望 max_time ≈ scan_duration * 1000
        double max_time = 0;
        for (const auto& pt : scan_out->points) {
            if (std::abs(pt.time) > max_time) max_time = std::abs(pt.time);
        }
        // 如果最大时间 > 3倍扫描周期(ms)，认为 time 字段不可靠
        double expected_max_ms = scan_duration * 1000.0 * 3.0;
        if (expected_max_ms > 1.0 && max_time > expected_max_ms) {
            time_valid = false;
            LOG_EVERY_N(WARNING, 50) << "FASTLIO2 point.time is garbage (max=" << max_time
                                     << "ms, expected <" << expected_max_ms
                                     << "ms), assigning linear timestamps";
            const int n = scan_out->size();
            for (int i = 0; i < n; i++) {
                scan_out->points[i].time = static_cast<float>(i * scan_duration * 1000.0 / (n - 1));
            }
        } else if (expected_max_ms <= 1.0) {
            // scan_duration 异常，也分配线性时间戳
            time_valid = false;
            const int n = scan_out->size();
            for (int i = 0; i < n; i++) {
                scan_out->points[i].time = static_cast<float>(i * 100.0 / (n - 1));  // 假设 100ms 扫描
            }
        }
    }

    std::sort(scan_out->points.begin(), scan_out->points.end(),
              [](const PointType& a, const PointType& b) { return a.time < b.time; });

    // Initialize IMU pose
    state_ikfom imu_state = kf_.get_x();
    std::vector<Pose6D> IMUpose;
    IMUpose.push_back(Pose6D(0.0, acc_s_last_, angvel_last_,
                             imu_state.vel, imu_state.pos,
                             imu_state.rot.toRotationMatrix()));

    // Forward propagation
    Vec3d angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    Mat3d R_imu;
    input_ikfom in;

    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto head = *it_imu;
        auto tail = *(it_imu + 1);

        if (tail->timestamp < last_lidar_end_time_) continue;

        angvel_avr = 0.5 * (head->angular_velocity + tail->angular_velocity);
        acc_avr = 0.5 * (head->linear_acceleration + tail->linear_acceleration);

        acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();

        double dt;
        if (head->timestamp < last_lidar_end_time_) {
            dt = tail->timestamp - last_lidar_end_time_;
        } else {
            dt = tail->timestamp - head->timestamp;
        }

        in.acc = acc_avr;
        in.gyro = angvel_avr;
        kf_.predict(dt, Q_, in);

        imu_state = kf_.get_x();
        angvel_last_ = angvel_avr - imu_state.bg;
        acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba);
        for (int i = 0; i < 3; i++) {
            acc_s_last_[i] += imu_state.grav[i];
        }

        double offs_t = tail->timestamp - pcl_beg_time;
        IMUpose.push_back(Pose6D(offs_t, acc_s_last_, angvel_last_,
                                 imu_state.vel, imu_state.pos,
                                 imu_state.rot.toRotationMatrix()));
    }

    // Propagation to frame end
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    double dt = note * (pcl_end_time - imu_end_time);
    kf_.predict(dt, Q_, in);

    imu_state = kf_.get_x();
    last_imu_pose_.offset_time = pcl_end_time;
    last_lidar_end_time_ = pcl_end_time;

    // Sync state_point_ after forward propagation so PointBodyToWorld() uses
    // the correct state (especially for the first frame before Observe() runs).
    state_point_ = imu_state;

    // Backward undistortion
    UndistortPointCloud(scan_out, IMUpose,
                        imu_state.rot.toRotationMatrix(), imu_state.pos,
                        imu_state.offset_R_L_I.toRotationMatrix(), imu_state.offset_T_L_I);
}

void FASTLIO2Core::PointBodyToWorld(const PointType& pi, PointType& po) {
    Vec3d p_body(pi.x, pi.y, pi.z);
    Vec3d p_global(state_point_.rot *
                   (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                   state_point_.pos);

    po.x = p_global(0);
    po.y = p_global(1);
    po.z = p_global(2);
    po.intensity = pi.intensity;
}

void FASTLIO2Core::Observe(CloudPtr scan_down) {
    feats_down_body_ = scan_down;
    feats_down_size_ = feats_down_body_->points.size();

    if (feats_down_size_ < 5) {
        LOG(WARNING) << "No enough points for observation: " << feats_down_size_;
        return;
    }

    normvec_->resize(feats_down_size_);
    feats_down_world_->resize(feats_down_size_);
    Nearest_Points.resize(feats_down_size_);

    // Run iterated EKF update
    double solve_H_time = 0;
    kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    state_point_ = kf_.get_x();

    // Convert MTK SO3 to Eigen Matrix3d for RotMtoEuler
    Mat3d rot_mat = state_point_.rot.toRotationMatrix();
    euler_cur_ = RotMtoEuler(rot_mat);
    pos_lid_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
}

void FASTLIO2Core::UpdateMap(CloudPtr scan_world) {
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size_);
    PointNoNeedDownsample.reserve(feats_down_size_);

    for (int i = 0; i < feats_down_size_; i++) {
        // Transform to world frame
        PointBodyToWorld(feats_down_body_->points[i], feats_down_world_->points[i]);

        // Decide if need to add to map
        if (!Nearest_Points[i].empty() && flg_EKF_inited_) {
            const PointVector& points_near = Nearest_Points[i];
            bool need_add = true;

            PointType mid_point;
            mid_point.x = std::floor(feats_down_world_->points[i].x / config_.filter_size_map) *
                           config_.filter_size_map + 0.5 * config_.filter_size_map;
            mid_point.y = std::floor(feats_down_world_->points[i].y / config_.filter_size_map) *
                           config_.filter_size_map + 0.5 * config_.filter_size_map;
            mid_point.z = std::floor(feats_down_world_->points[i].z / config_.filter_size_map) *
                           config_.filter_size_map + 0.5 * config_.filter_size_map;

            float dist = (feats_down_world_->points[i].x - mid_point.x) *
                         (feats_down_world_->points[i].x - mid_point.x) +
                         (feats_down_world_->points[i].y - mid_point.y) *
                         (feats_down_world_->points[i].y - mid_point.y) +
                         (feats_down_world_->points[i].z - mid_point.z) *
                         (feats_down_world_->points[i].z - mid_point.z);

            if (std::fabs(points_near[0].x - mid_point.x) > 0.5 * config_.filter_size_map &&
                std::fabs(points_near[0].y - mid_point.y) > 0.5 * config_.filter_size_map &&
                std::fabs(points_near[0].z - mid_point.z) > 0.5 * config_.filter_size_map) {
                PointNoNeedDownsample.push_back(feats_down_world_->points[i]);
                continue;
            }

            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                float near_dist = (points_near[readd_i].x - mid_point.x) *
                                  (points_near[readd_i].x - mid_point.x) +
                                  (points_near[readd_i].y - mid_point.y) *
                                  (points_near[readd_i].y - mid_point.y) +
                                  (points_near[readd_i].z - mid_point.z) *
                                  (points_near[readd_i].z - mid_point.z);
                if (near_dist < dist) {
                    need_add = false;
                    break;
                }
            }

            if (need_add) PointToAdd.push_back(feats_down_world_->points[i]);
        } else {
            PointToAdd.push_back(feats_down_world_->points[i]);
        }
    }

    ikdtree_.Add_Points(PointToAdd, true);
    ikdtree_.Add_Points(PointNoNeedDownsample, false);
}

SE3 FASTLIO2Core::GetPose() const {
    Mat3d rot_mat = state_point_.rot.toRotationMatrix();
    return SE3(SO3(rot_mat), state_point_.pos);
}

FASTLIO2Core::NativeState FASTLIO2Core::GetNativeState() const {
    NativeState ns;
    ns.pos = state_point_.pos;
    ns.rot = state_point_.rot.toRotationMatrix();
    ns.offset_R_L_I = state_point_.offset_R_L_I.toRotationMatrix();
    ns.offset_T_L_I = state_point_.offset_T_L_I;
    ns.vel = state_point_.vel;
    ns.bg = state_point_.bg;
    ns.ba = state_point_.ba;
    ns.grav = state_point_.grav;
    return ns;
}

}  // namespace lightning
