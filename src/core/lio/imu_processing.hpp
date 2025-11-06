#pragma once

#ifndef FASTER_LIO_IMU_PROCESSING_H
#define FASTER_LIO_IMU_PROCESSING_H

#include <glog/logging.h>
#include <cmath>
#include <deque>
#include <fstream>

#include "common/eigen_types.h"
#include "common/measure_group.h"
#include "common/point_def.h"
#include "core/lightning_math.hpp"
#include "core/lio/eskf.hpp"
#include "core/lio/pose6d.h"
#include "utils/timer.h"

namespace lightning {

/// IMU处理类
class ImuProcess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void SetExtrinsic(const Vec3d &transl, const Mat3d &rot);
    void SetGyrCov(const Vec3d &scaler);
    void SetAccCov(const Vec3d &scaler);
    void SetGyrBiasCov(const Vec3d &b_g);
    void SetAccBiasCov(const Vec3d &b_a);

    void Process(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &scan);

    bool IsIMUInited() const { return imu_need_init_ == false; }

    double GetMeanAccNorm() const { return mean_acc_.norm(); }

    Eigen::Matrix<double, 12, 12> Q_;
    Vec3d cov_acc_;
    Vec3d cov_gyr_;
    Vec3d cov_acc_scale_;
    Vec3d cov_gyr_scale_;
    Vec3d cov_bias_gyr_;
    Vec3d cov_bias_acc_;

   private:
    void IMUInit(const MeasureGroup &meas, ESKF &kf_state, int &N);
    void UndistortPcl(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &pcl_out);

    static inline constexpr int max_init_count_ = 20;

    PointCloudType::Ptr cur_pcl_un_ = nullptr;
    lightning::IMUPtr last_imu_ = nullptr;
    std::deque<lightning::IMUPtr> imu_queue_;

    std::vector<Pose6D> imu_pose_;
    Mat3d R_lidar_imu_ = Mat3d ::Identity();
    Vec3d t_lidar_mu_ = Vec3d ::Zero();
    Vec3d mean_acc_ = Vec3d::Zero();
    Vec3d mean_gyr_ = Vec3d::Zero();
    Vec3d angvel_last_ = Vec3d ::Zero();
    Vec3d acc_s_last_ = Vec3d ::Zero();

    double last_lidar_end_time_ = 0;
    int init_iter_num_ = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};

inline ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true) {
    init_iter_num_ = 1;
    Q_.setZero();
    Q_.diagonal() << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5;
    cov_acc_ = Vec3d(0.1, 0.1, 0.1);
    cov_gyr_ = Vec3d(0.1, 0.1, 0.1);
    cov_bias_gyr_ = Vec3d(0.0001, 0.0001, 0.0001);
    cov_bias_acc_ = Vec3d(0.0001, 0.0001, 0.0001);
    mean_acc_ = Vec3d(0, 0, -1.0);
    mean_gyr_ = Vec3d(0, 0, 0);
    last_imu_.reset(new lightning::IMU());
}

inline ImuProcess::~ImuProcess() {}

inline void ImuProcess::Reset() {
    mean_acc_ = Vec3d(0, 0, -1.0);
    mean_gyr_ = Vec3d(0, 0, 0);
    angvel_last_.setZero();

    imu_need_init_ = true;
    init_iter_num_ = 1;
    imu_queue_.clear();
    imu_pose_.clear();
    last_imu_.reset(new lightning::IMU());
    cur_pcl_un_.reset(new PointCloudType());
}

inline void ImuProcess::SetExtrinsic(const Vec3d &transl, const Mat3d &rot) {
    t_lidar_mu_ = transl;
    R_lidar_imu_ = rot;
}

inline void ImuProcess::SetGyrCov(const Vec3d &scaler) { cov_gyr_scale_ = scaler; }

inline void ImuProcess::SetAccCov(const Vec3d &scaler) { cov_acc_scale_ = scaler; }

inline void ImuProcess::SetGyrBiasCov(const Vec3d &b_g) { cov_bias_gyr_ = b_g; }

inline void ImuProcess::SetAccBiasCov(const Vec3d &b_a) { cov_bias_acc_ = b_a; }

inline void ImuProcess::IMUInit(const MeasureGroup &meas, ESKF &kf_state, int &N) {
    /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity_ **/

    Vec3d cur_acc, cur_gyr;

    if (b_first_frame_) {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu_.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu_.front()->angular_velocity;
        mean_acc_ = imu_acc;
        mean_gyr_ = gyr_acc;
    }

    for (const auto &imu : meas.imu_) {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc = imu_acc;
        cur_gyr = gyr_acc;

        mean_acc_ += (cur_acc - mean_acc_) / N;
        mean_gyr_ += (cur_gyr - mean_gyr_) / N;

        cov_acc_ =
            cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
        cov_gyr_ =
            cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

        N++;
    }

    auto init_state = kf_state.GetX();
    init_state.timestamp_ = meas.imu_.back()->timestamp;
    init_state.grav_ = -mean_acc_ / mean_acc_.norm() * G_m_s2;
    init_state.bg_ = mean_gyr_;
    init_state.offset_t_lidar_ = t_lidar_mu_;
    init_state.offset_R_lidar_ = R_lidar_imu_;
    kf_state.ChangeX(init_state);

    auto init_P = kf_state.GetP();
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.ChangeP(init_P);

    last_imu_ = meas.imu_.back();
}

inline void ImuProcess::UndistortPcl(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &pcl_out) {
    /*** add the imu_ of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu_;
    v_imu.push_front(last_imu_);
    const double &imu_end_time = v_imu.back()->timestamp;

    const double &pcl_beg_time = meas.lidar_begin_time_;
    const double &pcl_end_time = meas.lidar_end_time_;

    /*** Initialize IMU pose ***/
    auto imu_state = kf_state.GetX();
    imu_pose_.clear();
    imu_pose_.emplace_back(0.0, acc_s_last_, angvel_last_, imu_state.vel_, imu_state.pos_, imu_state.rot_.matrix());

    /*** forward propagation at each imu_ point ***/
    Vec3d angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    Mat3d R_imu;

    double dt = 0;
    Vec3d acc = Vec3d::Zero();
    Vec3d gyro = Vec3d::Zero();

    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->timestamp < last_lidar_end_time_) {
            continue;
        }

        angvel_avr = .5 * (head->angular_velocity + tail->angular_velocity);
        acc_avr = .5 * (head->linear_acceleration + tail->linear_acceleration);

        acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();  // - state_inout.ba;

        if (head->timestamp < last_lidar_end_time_) {
            dt = tail->timestamp - last_lidar_end_time_;
        } else {
            dt = tail->timestamp - head->timestamp;
        }

        acc = acc_avr;
        gyro = angvel_avr;

        if (dt > 0.1) {
            LOG(ERROR) << "get abnormal dt: " << dt;
            kf_state.SetTime((*it_imu)->timestamp);
            break;
        }

        Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
        Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
        Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
        Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
        kf_state.Predict(dt, Q_, gyro, acc);

        // LOG(INFO) << "gyro: " << gyro.transpose() << ", dt: " << dt;

        // LOG(INFO) << "acc: " << acc.transpose() << " grav: " << kf_state.GetX().grav_.vec_.norm()
        //           << ", vel: " << kf_state.GetX().vel_.transpose() << ", dt: " << dt;

        /* save the poses at each IMU measurements */
        imu_state = kf_state.GetX();
        angvel_last_ = angvel_avr - imu_state.bg_;
        acc_s_last_ = imu_state.rot_ * (acc_avr - imu_state.ba_);
        for (int i = 0; i < 3; i++) {
            acc_s_last_[i] += imu_state.grav_[i];
        }

        double &&offs_t = tail->timestamp - pcl_beg_time;
        imu_pose_.emplace_back(
            Pose6D(offs_t, acc_s_last_, angvel_last_, imu_state.vel_, imu_state.pos_, imu_state.rot_.matrix()));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.Predict(dt, Q_, gyro, acc);

    imu_state = kf_state.GetX();
    last_imu_ = meas.imu_.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** sort point clouds by offset time ***/
    pcl_out = meas.scan_;
    std::sort(pcl_out->points.begin(), pcl_out->points.end(),
              [](const PointType &p1, const PointType &p2) { return p1.time < p2.time; });

    /*** undistort each lidar point (backward propagation) ***/
    if (pcl_out->empty()) {
        return;
    }

    auto it_pcl = pcl_out->points.end() - 1;
    for (auto it_kp = imu_pose_.end() - 1; it_kp != imu_pose_.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu = (head->rot);
        vel_imu = (head->vel);
        pos_imu = (head->pos);
        acc_imu = (tail->acc);
        angvel_avr = (tail->gyr);

        for (; it_pcl->time / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->time / double(1000) - head->offset_time;

            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * p_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
            Mat3d R_i(R_imu * math::exp(angvel_avr, dt).matrix());

            Vec3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            Vec3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos_);
            Vec3d p_compensate = imu_state.offset_R_lidar_.inverse() *
                                 (imu_state.rot_.inverse() *
                                      (R_i * (imu_state.offset_R_lidar_ * P_i + imu_state.offset_t_lidar_) + T_ei) -
                                  imu_state.offset_t_lidar_);  // not accurate!

            // save Undistorted points and their rotation
            it_pcl->x = p_compensate(0);
            it_pcl->y = p_compensate(1);
            it_pcl->z = p_compensate(2);

            if (it_pcl == pcl_out->points.begin()) {
                break;
            }
        }
    }
}

inline void ImuProcess::Process(const MeasureGroup &meas, ESKF &kf_state, CloudPtr &scan) {
    if (meas.imu_.empty()) {
        return;
    }

    if (imu_need_init_) {
        /// The very first lidar frame
        IMUInit(meas, kf_state, init_iter_num_);

        imu_need_init_ = true;

        last_imu_ = meas.imu_.back();

        auto imu_state = kf_state.GetX();
        if (init_iter_num_ > max_init_count_) {
            cov_acc_ *= pow(G_m_s2 / mean_acc_.norm(), 2);
            imu_need_init_ = false;

            cov_acc_ = cov_acc_scale_;
            cov_gyr_ = cov_gyr_scale_;
            LOG(INFO) << "imu init done, bg: " << imu_state.bg_.transpose() << ", ba: " << imu_state.ba_.transpose();
        } else {
            LOG(INFO) << "waiting for imu init ... " << init_iter_num_;
        }

        return;
    }

    Timer::Evaluate([&, this]() { UndistortPcl(meas, kf_state, scan); }, "Undistort Pcl");
}
}  // namespace lightning

#endif
