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
 */
struct NavState {
    constexpr static int dim = 24;  //  状态变量维度

    using VectState = Eigen::Matrix<double, dim, 1>;  // 矢量形式

    NavState() = default;

    bool operator<(const NavState& other) { return timestamp_ < other.timestamp_; }

    VectState ToState() {
        VectState ret;
        ret.segment<3>(0) = pos_;
        ret.segment<3>(3) = rot_.log();
        ret.segment<3>(6) = offset_R_lidar_.log();
        ret.segment<3>(9) = offset_t_lidar_;
        ret.segment<3>(12) = vel_;
        ret.segment<3>(15) = bg_;
        ret.segment<3>(18) = ba_;
        ret.segment<3>(21) = grav_;
        return ret;
    }

    void FromVectState(const VectState& state) {
        pos_ = state.segment<3>(0);
        rot_ = SO3::exp(state.segment<3>(3));
        offset_R_lidar_ = SO3::exp(state.segment<3>(6));
        offset_t_lidar_ = state.segment<3>(9);
        vel_ = state.segment<3>(12);
        bg_ = state.segment<3>(15);
        ba_ = state.segment<3>(18);
        grav_ = state.segment<3>(21);
    }

    // 运动过程
    inline VectState get_f(const Vec3d& gyro, const Vec3d& acce) const {
        VectState res = VectState::Zero();
        // 减零偏
        Vec3d omega = gyro - bg_;
        Vec3d a_inertial = rot_ * (acce - ba_);  // 加计读数-ba 并转到 世界系下

        res.segment<3>(0) = vel_;
        res.segment<3>(3) = omega;
        res.segment<3>(12) = a_inertial + grav_;
        return res;
    }

    /// 运动方程对状态的雅可比
    inline Eigen::Matrix<double, dim, dim> df_dx(const Vec3d& acce) const {
        Eigen::Matrix<double, dim, dim> cov = Eigen::Matrix<double, dim, dim>::Zero();
        cov.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
        cov.block<3, 3>(12, 3) = -rot_.matrix() * SO3::hat(acce - ba_);
        cov.block<3, 3>(12, 18) = -rot_.matrix();
        cov.block<3, 3>(12, 21) = Eigen::Matrix3d::Identity();
        cov.block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
        return cov;
    }

    /// 运动方程对噪声的雅可比
    inline Eigen::Matrix<double, 24, 12> df_dw() const {
        Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
        cov.block<3, 3>(12, 3) = -rot_.matrix();
        cov.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        cov.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
        cov.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
        return cov;
    }

    /// 递推
    void oplus(const VectState& vec, double dt) {
        timestamp_ += dt;
        pos_ += vec.segment<3>(0) * dt;
        rot_ = rot_ * SO3::exp(vec.segment<3>(3) * dt);
        offset_R_lidar_ = offset_R_lidar_ * SO3::exp(vec.segment<3>(6) * dt);
        offset_t_lidar_ = offset_t_lidar_ + vec.segment<3>(9) * dt;
        vel_ += vec.segment<3>(12) * dt;
        bg_ += vec.segment<3>(15) * dt;
        ba_ += vec.segment<3>(18) * dt;
        grav_ += vec.segment<3>(21) * dt;
    }

    /**
     * 广义减法, this - other
     * @param result 减法结果
     * @param other 另一个状态变量
     */
    VectState boxminus(const NavState& other) {
        VectState result;
        result.segment<3>(0) = pos_ - other.pos_;
        result.segment<3>(3) = (other.rot_.inverse() * rot_).log();
        result.segment<3>(6) = (other.offset_R_lidar_.inverse() * offset_R_lidar_).log();
        result.segment<3>(9) = offset_t_lidar_ - other.offset_t_lidar_;
        result.segment<3>(12) = vel_ - other.vel_;
        result.segment<3>(15) = bg_ - other.bg_;
        result.segment<3>(18) = ba_ - other.ba_;
        result.segment<3>(21) = grav_ - other.grav_;
        return result;
    }

    /**
     * 广义加法 this = this+dx
     * @param dx 增量
     */
    NavState boxplus(const VectState& dx) {
        NavState ret;
        ret.timestamp_ = timestamp_;
        ret.pos_ = pos_ + dx.segment<3>(0);
        ret.rot_ = rot_ * SO3::exp(dx.segment<3>(3));
        ret.offset_R_lidar_ = offset_R_lidar_ * SO3::exp(dx.segment<3>(6));
        ret.offset_t_lidar_ = offset_t_lidar_ + dx.segment<3>(9);
        ret.vel_ = vel_ + dx.segment<3>(12);
        ret.bg_ = bg_ + dx.segment<3>(15);
        ret.ba_ = ba_ + dx.segment<3>(18);
        ret.grav_ = grav_ + dx.segment<3>(21);
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
    static const std::vector<MetaInfo> S2_states_;    // S2 变量维度

    friend inline std::ostream& operator<<(std::ostream& os, const NavState& s) {
        os << std::setprecision(18) << s.pos_.transpose() << " " << s.rot_.unit_quaternion().coeffs().transpose() << " "
           << s.offset_R_lidar_.unit_quaternion().coeffs().transpose() << " " << s.offset_t_lidar_.transpose() << " "
           << s.vel_.transpose() << " " << s.bg_.transpose() << " " << s.ba_.transpose() << " " << s.grav_.transpose();
        return os;
    }

    inline SE3 GetPose() const { return SE3(rot_, pos_); }
    inline SO3 GetRot() const { return rot_; }
    inline void SetPose(const SE3& pose) {
        rot_ = pose.so3();
        pos_ = pose.translation();
    }

    inline Vec3d Getba() const { return ba_; }
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
    SO3 offset_R_lidar_;                    // 外参R
    Vec3d offset_t_lidar_ = Vec3d::Zero();  // 外参t
    Vec3d vel_ = Vec3d::Zero();             // 速度
    Vec3d bg_ = Vec3d::Zero();              // 陀螺零偏
    Vec3d ba_ = Vec3d::Zero();              // 加计零偏
    Vec3d grav_;                            // 重力
};

}  // namespace lightning
