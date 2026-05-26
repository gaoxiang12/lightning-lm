//
// Created by xiang on 2022/2/15.
//

#pragma once

#include "common/eigen_types.h"

#include <glog/logging.h>
#include <iomanip>

namespace lightning {
/**
 * 重构之后的状态变量
 * 显式写出各维度状态
 *
 * 虽然我觉得有些地方还是啰嗦了点。。
 *
 * 4个3维块，共12维
 */
struct NavState {
    constexpr static int dim = 12;       //  状态变量维度
    constexpr static int full_dim = 12;  // 误差变量维度
    constexpr static int kBlockDim = 3;
    constexpr static int kPosIdx = 0;
    constexpr static int kRotIdx = 3;
    constexpr static int kVelIdx = 6;
    constexpr static int kBgIdx = 9;

    using VectState = Eigen::Matrix<double, dim, 1>;           // 矢量形式
    using FullVectState = Eigen::Matrix<double, full_dim, 1>;  // 全状态矢量形式

    NavState() = default;

    bool operator<(const NavState& other) { return timestamp_ < other.timestamp_; }

    FullVectState ToState() {
        FullVectState ret;
        ret.block<kBlockDim, 1>(kPosIdx, 0) = pos_;
        ret.block<kBlockDim, 1>(kRotIdx, 0) = rot_.log();
        ret.block<kBlockDim, 1>(kVelIdx, 0) = vel_;
        ret.block<kBlockDim, 1>(kBgIdx, 0) = bg_;
        return ret;
    }

    void FromVectState(const FullVectState& state) {
        pos_ = state.block<kBlockDim, 1>(kPosIdx, 0);
        rot_ = SO3::exp(state.block<kBlockDim, 1>(kRotIdx, 0));
        vel_ = state.block<kBlockDim, 1>(kVelIdx, 0);
        bg_ = state.block<kBlockDim, 1>(kBgIdx, 0);
    }

    // 运动过程
    inline FullVectState get_f(const Vec3d& gyro, const Vec3d& acce) const {
        FullVectState res = FullVectState::Zero();
        // 减零偏
        Vec3d omega = gyro - bg_;
        Vec3d a_inertial = rot_ * acce;  // 加计零偏不再参与在线估计

        for (int i = 0; i < 3; i++) {
            res(i) = vel_[i];
            res(i + kRotIdx) = omega[i];
            res(i + kVelIdx) = a_inertial[i] + grav_[i];
        }
        return res;
    }

    /// 运动方程对状态的雅可比
    inline Eigen::Matrix<double, full_dim, dim> df_dx(const Vec3d& acce) const {
        Eigen::Matrix<double, full_dim, dim> cov = Eigen::Matrix<double, full_dim, dim>::Zero();
        cov.block<kBlockDim, kBlockDim>(kPosIdx, kVelIdx) = Mat3d::Identity();
        Vec3d acc = acce;
        // Vec3d omega = gyro - bg_;
        cov.block<kBlockDim, kBlockDim>(kVelIdx, kRotIdx) = -rot_.matrix() * SO3::hat(acc);
        cov.block<kBlockDim, kBlockDim>(kRotIdx, kBgIdx) = -Eigen::Matrix3d::Identity();
        return cov;
    }

    /// 运动方程对噪声的雅可比
    inline Eigen::Matrix<double, full_dim, 12> df_dw() const {
        Eigen::Matrix<double, full_dim, 12> cov = Eigen::Matrix<double, full_dim, 12>::Zero();
        cov.block<kBlockDim, kBlockDim>(kVelIdx, 3) = -rot_.matrix();
        cov.block<kBlockDim, kBlockDim>(kRotIdx, 0) = -Eigen::Matrix3d::Identity();
        cov.block<kBlockDim, kBlockDim>(kBgIdx, 6) = Eigen::Matrix3d::Identity();
        return cov;
    }

    /// 递推
    void oplus(const FullVectState& vec, double dt) {
        timestamp_ += dt;
        pos_ += vec.middleRows(kPosIdx, kBlockDim) * dt;
        rot_ = rot_ * SO3::exp(vec.middleRows(kRotIdx, kBlockDim) * dt);
        // vel_ += vec.middleRows(kVelIdx, kBlockDim) * dt;
        bg_ += vec.middleRows(kBgIdx, kBlockDim) * dt;
    }

    /**
     * 广义减法, this - other
     * @param result 减法结果
     * @param other 另一个状态变量
     */
    VectState boxminus(const NavState& other) {
        VectState result;
        result.block<kBlockDim, 1>(kPosIdx, 0) = pos_ - other.pos_;
        result.block<kBlockDim, 1>(kRotIdx, 0) = (other.rot_.inverse() * rot_).log();
        result.block<kBlockDim, 1>(kVelIdx, 0) = vel_ - other.vel_;
        result.block<kBlockDim, 1>(kBgIdx, 0) = bg_ - other.bg_;

        return result;
    }

    /**
     * 广义加法 this = this+dx
     * @param dx 增量
     */
    NavState boxplus(const VectState& dx) {
        NavState ret;
        ret.timestamp_ = timestamp_;
        ret.pos_ = pos_ + dx.middleRows(kPosIdx, kBlockDim);
        ret.rot_ = rot_ * SO3::exp(dx.middleRows(kRotIdx, kBlockDim));
        ret.vel_ = vel_ + dx.middleRows(kVelIdx, kBlockDim);
        ret.bg_ = bg_ + dx.middleRows(kBgIdx, kBlockDim);
        ret.grav_ = grav_;

        return ret;
    }

    /// 各个子变量所在维度信息
    struct MetaInfo {
        MetaInfo(int idx, int vdim, int dof) : idx_(idx), dim_(vdim), dof_(dof) {}
        int idx_ = 0;  // 变量所在索引
        int dim_ = 0;  // 变量维度
        int dof_ = 0;  // 自由度
    };

    static const std::vector<MetaInfo> vect_states_;  // 矢量变量的维度
    static const std::vector<MetaInfo> SO3_states_;   // SO3 变量的维度

    friend inline std::ostream& operator<<(std::ostream& os, const NavState& s) {
        os << std::setprecision(18) << s.pos_.transpose() << " " << s.rot_.unit_quaternion().coeffs().transpose() << " "
           << s.vel_.transpose() << " " << s.bg_.transpose() << " " << s.grav_.transpose();
        return os;
    }

    inline SE3 GetPose() const { return SE3(rot_, pos_); }
    inline SO3 GetRot() const { return rot_; }
    inline void SetPose(const SE3& pose) {
        rot_ = pose.so3();
        pos_ = pose.translation();
    }

    inline Vec3d Getba() const { return Vec3d::Zero(); }
    inline Vec3d Getbg() const { return bg_; }
    inline Vec3d GetVel() const { return vel_; }
    void SetVel(const Vec3d& v) { vel_ = v; }

    double timestamp_ = 0.0;           // 时间戳
    double confidence_ = 0.0;          // 定位置信度
    bool pose_is_ok_ = true;           // 定位是否有效
    bool lidar_odom_reliable_ = true;  // lio是否有效
    bool is_parking_ = false;          // 是否在停车

    Vec3d pos_ = Vec3d::Zero();             // 位置
    SO3 rot_;                               // 旋转
    Vec3d vel_ = Vec3d::Zero();             // 速度
    Vec3d bg_ = Vec3d::Zero();              // 陀螺零偏
    Vec3d grav_ = Vec3d(0.0, 0.0, -9.81);   // 固定重力向量（不参与状态估计）
};

}  // namespace lightning
