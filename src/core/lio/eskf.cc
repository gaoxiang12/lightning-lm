//
// Created by xiang on 2022/2/15.
//

#include "core/lio/eskf.hpp"
#include "core/lightning_math.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>

namespace {

using CovType = lightning::ESKF::CovType;

void SymmetrizeAndFloorCovariance(CovType& P, double min_cov_diag) {
    P = 0.5 * (P + P.transpose()).eval();

    for (int i = 0; i < P.rows(); ++i) {
        if (P(i, i) < min_cov_diag) {
            P(i, i) = min_cov_diag;
        } else if (P(i, i) > 100.0) {
            P(i, i) = 100.0;
        }

        for (int j = 0; j < P.cols(); ++j) {
            if (std::isnan(P(i, j)) || std::isinf(P(i, j))) {
                LOG(WARNING) << "find nan or inf in P: " << P(i, j);
                P(i, j) = 1.0;
            }
        }
    }
}

}  // namespace

namespace lightning {

void ESKF::Predict(const double& dt, const ESKF::ProcessNoiseType& Q, const Vec3d& gyro, const Vec3d& acce) {
    Eigen::Matrix<double, NavState::full_dim, 1> f_ = x_.get_f(gyro, acce);  // 调用get_f 获取 速度 角速度 加速度
    Eigen::Matrix<double, NavState::full_dim, state_dim_> f_x_ = x_.df_dx(acce);
    Eigen::Matrix<double, NavState::full_dim, process_noise_dim_> f_w_ = x_.df_dw();
    Eigen::Matrix<double, state_dim_, process_noise_dim_> f_w_final =
        Eigen::Matrix<double, state_dim_, process_noise_dim_>::Zero();

    NavState x_before = x_;
    x_.oplus(f_, dt);

    F_x1_ = CovType::Identity();

    // set f_x_final
    CovType f_x_final = CovType::Zero();
    for (auto st : x_.vect_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        int dof = st.dof_;

        for (int i = 0; i < state_dim_; i++) {
            for (int j = 0; j < dof; j++) {
                f_x_final(idx + j, i) = f_x_(dim + j, i);
            }
        }

        for (int i = 0; i < process_noise_dim_; i++) {
            for (int j = 0; j < dof; j++) {
                f_w_final(idx + j, i) = f_w_(dim + j, i);
            }
        }
    }

    Mat3d res_temp_SO3;
    Vec3d seg_SO3;
    for (auto st : x_.SO3_states_) {
        int idx = st.idx_;
        int dim = st.dim_;
        for (int i = 0; i < 3; i++) {
            seg_SO3(i) = -1 * f_(dim + i) * dt;
        }

        F_x1_.block<3, 3>(idx, idx) = math::exp(seg_SO3, 0.5).matrix();

        res_temp_SO3 = math::A_matrix(seg_SO3);
        for (int i = 0; i < state_dim_; i++) {
            f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.block<3, 1>(dim, i));
        }

        for (int i = 0; i < process_noise_dim_; i++) {
            f_w_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_.block<3, 1>(dim, i));
        }
    }

    F_x1_ += f_x_final * dt;
    P_ = (F_x1_)*P_ * (F_x1_).transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();
    P_ *= options_.predict_cov_inflation_;
    SymmetrizeAndFloorCovariance(P_, options_.min_cov_diag_);
}

/**
 * 原版的迭代过程中，收敛次数大于1才会结果，所以需要两次收敛。
 * 在未收敛时，实际上不会计算最近邻，也就回避了一次ObsModel的计算
 * 如果这边对每次迭代都计算最近邻的话，时间明显会变长一些，并不是非常合理。。
 *
 * @param obs
 * @param R
 */
void ESKF::Update(ESKF::ObsType obs, const double& R) {
    custom_obs_model_.valid_ = true;
    custom_obs_model_.converge_ = true;

    CovType P_propagated = P_;

    Eigen::Matrix<double, state_dim_, 1> K_r;
    Eigen::Matrix<double, state_dim_, state_dim_> K_H;

    StateVecType dx_current = StateVecType::Zero();  // 本轮迭代的dx

    NavState start_x = x_;  // 迭代的起点
    NavState last_x = x_;

    int converged_times = 0;
    double last_lidar_res = 0;

    double init_res = 0.0;
    static double iterated_num = 0;
    static double update_num = 0;
    update_num += 1;
    for (int i = -1; i < maximum_iter_; i++) {
        custom_obs_model_.valid_ = true;

        /// 计算observation function，主要是residual_, h_x_, s_
        /// x_ 在每次迭代中都是更新的，线性化点也会更新
        if (obs == ObsType::LIDAR || obs == ObsType::WHEEL_SPEED_AND_LIDAR) {
            lidar_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::WHEEL_SPEED) {
            wheelspeed_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::ACC_AS_GRAVITY) {
            acc_as_gravity_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::GPS) {
            gps_obs_func_(x_, custom_obs_model_);
        } else if (obs == ObsType::BIAS) {
            bias_obs_func_(x_, custom_obs_model_);
        }

        if (custom_obs_model_.valid_ == false) {
            x_ = last_x;
            P_ = P_propagated;
            return;
        }

        if (use_aa_ && i > -1 && (obs == ObsType::LIDAR || obs == ObsType::WHEEL_SPEED_AND_LIDAR) &&
            custom_obs_model_.lidar_residual_mean_ >= last_lidar_res * 1.01) {
            x_ = last_x;
            break;
        }
        iterated_num += 1;

        if (!custom_obs_model_.valid_) {
            continue;
        }

        if (i == -1) {
            init_res = custom_obs_model_.lidar_residual_mean_;
            if (init_res < 1e-9) {
                init_res = 1e-9;  // 可能有零
            }
        }

        iterations_ = i + 2;  // i从-1开始计
        final_res_ = custom_obs_model_.lidar_residual_mean_ / init_res;

        StateVecType dx = x_.boxminus(start_x);  // 当前x与起点之间的dx
        dx_current = dx;                         //

        P_ = P_propagated;

        /// 更新P 和 dx
        /// P = J*P*J^T
        /// dx = J * dx
        for (auto it : x_.SO3_states_) {
            int idx = it.idx_;
            Vec3d seg_SO3 = dx.block<3, 1>(idx, 0);
            Mat3d res_temp_SO3 = math::A_matrix(seg_SO3).transpose();  // 小块的J阵, SO3上的雅可比？

            dx_current.block<3, 1>(idx, 0) = res_temp_SO3 * dx.block<3, 1>(idx, 0);

            /// P 上面有SO3的行 进行转换
            for (int j = 0; j < state_dim_; j++) {
                P_.block<3, 1>(idx, j) = res_temp_SO3 * (P_.block<3, 1>(idx, j));
            }
            /// P 上面有SO3的列 进行转换
            for (int j = 0; j < state_dim_; j++) {
                P_.block<1, 3>(j, idx) = (P_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
            }
        }

        Mat6d HTH = custom_obs_model_.HTH_;
        Vec6d HTr = custom_obs_model_.HTr_;
        Mat6d HTH_sym = 0.5 * (HTH + HTH.transpose());

        Eigen::SelfAdjointEigenSolver<Mat6d> eigen_solver(HTH_sym);
        if (eigen_solver.info() != Eigen::Success) {
            LOG(WARNING) << "Failed to decompose ESKF observation information matrix.";
            continue;
        }

        const Vec6d eigen_values = eigen_solver.eigenvalues();
        const Mat6d eigen_vectors = eigen_solver.eigenvectors();
        const double max_eigen_value = std::max(1e-12, eigen_values.maxCoeff());
        const double degeneracy_threshold = max_eigen_value * options_.degeneracy_threshold_ratio_;

        // LOG(INFO) << "eigen values of HTH: " << eigen_values.transpose();

        Vec6d observable_mask = Vec6d::Zero();
        int nullity = 0;
        for (int k = 0; k < observable_mask.size(); ++k) {
            if (eigen_values(k) > degeneracy_threshold) {
                observable_mask(k) = 1.0;
            } else {
                nullity++;
            }
        }

        const Mat6d observable_projector = eigen_vectors * observable_mask.asDiagonal() * eigen_vectors.transpose();
        const Mat6d HTH_eff = observable_projector * HTH_sym * observable_projector;
        const Vec6d HTr_eff = observable_projector * HTr;

        CovType P_temp = (P_ / R).inverse();  // P阵上面已经更新

        /// 现在问题是这个权重太大，导致整体过于依赖先验 ...
        // P_temp.setIdentity();

        P_temp.block<pose_obs_dim_, pose_obs_dim_>(0, 0) += HTH_eff;
        CovType Q_inv = P_temp.inverse();  // Q inv

        // Q*H^T * R^-1 * r = K * r
        // <-- K ----->
        K_r = Q_inv.template block<state_dim_, pose_obs_dim_>(0, 0) * HTr_eff;

        // K_H = Q^-1 H^T R^-1 H
        //       <--  K     ->
        K_H.setZero();
        K_H.template block<state_dim_, pose_obs_dim_>(0, 0) =
            Q_inv.template block<state_dim_, pose_obs_dim_>(0, 0) * HTH_eff;

        // dx = Kr + (KH-I) dx
        // LOG(INFO) << "K_r: " << K_r.transpose()
        //           << ", prior: " << ((K_H - Eigen::Matrix<double, state_dim_, state_dim_>::Identity()) * dx_current).transpose();

        dx_current = K_r + (K_H - Eigen::Matrix<double, state_dim_, state_dim_>::Identity()) * dx_current;

        // check nan
        for (int j = 0; j < state_dim_; ++j) {
            if (std::isnan(dx_current(j, 0))) {
                return;
            }
        }

        // Vec3d dv = dx_current.middleRows(NavState::kVelIdx, NavState::kBlockDim);
        // if (dv.norm() > options_.vel_clip_norm_) {
        //     dv = dv / dv.norm() * options_.vel_clip_norm_;
        // }

        // dv = dv * options_.dv_ratio_;
        // dx_current.middleRows(NavState::kVelIdx, NavState::kBlockDim) = dv;

        // dx_current.middleRows(18, 5).setZero();

        // LOG(INFO) << "iter " << iterations_ << ", dx: " << dx_current.transpose();
        const double dx_translation = dx_current.head<3>().norm();
        const double dx_rotation_deg = dx_current.segment<3>(3).norm() * 180.0 / M_PI;
        if (dx_translation > options_.max_update_translation_step_ ||
            dx_rotation_deg > options_.max_update_rotation_step_deg_) {
            LOG(ERROR) << "Reject ESKF iter update, dtrans: " << dx_translation << ", drot_deg: " << dx_rotation_deg
                       << ", dvel: " << dx_current.segment<NavState::kBlockDim>(NavState::kVelIdx).norm();
            x_ = start_x;
            P_ = P_propagated;
            return;
        }

        if (!use_aa_) {
            x_ = x_.boxplus(dx_current);
        } else {
            // 转到起点的线性空间
            x_ = x_.boxplus(dx_current);

            if (i == -1) {
                aa_.init(dx_current);  // 初始化AA
            } else {
                // 利用AA计算dx from start
                auto dx_all = x_.boxminus(start_x);
                auto new_dx_all = aa_.compute(dx_all);
                x_ = start_x.boxplus(new_dx_all);
            }
        }

        last_x = x_;

        // update last res
        last_lidar_res = custom_obs_model_.lidar_residual_mean_;
        custom_obs_model_.converge_ = true;

        for (int j = 0; j < state_dim_; j++) {
            if (std::fabs(dx_current[j]) > limit_[j]) {
                custom_obs_model_.converge_ = false;
                break;
            }
        }

        if (custom_obs_model_.converge_) {
            converged_times++;
        }

        if (!converged_times && i == maximum_iter_ - 2) {
            custom_obs_model_.converge_ = true;
        }

        if (converged_times > 0 || i == maximum_iter_ - 1) {
            /// 结束条件：已经收敛
            /// 更新P阵, using (45)
            L_ = P_;
            Mat3d res_temp_SO3;
            Vec3d seg_SO3;
            for (auto it : x_.SO3_states_) {
                int idx = it.idx_;
                for (int j = 0; j < 3; j++) {
                    seg_SO3(j) = dx_current(j + idx);
                }

                res_temp_SO3 = math::A_matrix(seg_SO3).transpose();
                for (int j = 0; j < state_dim_; j++) {
                    L_.block<3, 1>(idx, j) = res_temp_SO3 * (P_.block<3, 1>(idx, j));
                }

                for (int j = 0; j < pose_obs_dim_; j++) {
                    K_H.block<3, 1>(idx, j) = res_temp_SO3 * (K_H.block<3, 1>(idx, j));
                }

                for (int j = 0; j < state_dim_; j++) {
                    L_.block<1, 3>(j, idx) = (L_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
                    P_.block<1, 3>(j, idx) = (P_.block<1, 3>(j, idx)) * res_temp_SO3.transpose();
                }
            }

            P_ = L_ - K_H.block<state_dim_, pose_obs_dim_>(0, 0) * P_.template block<pose_obs_dim_, state_dim_>(0, 0);

            if (nullity > 0) {
                // LOG_EVERY_N(INFO, 50) << "ESKF observation degeneracy rank " << (pose_obs_dim_ - nullity) << "/"
                //                      << pose_obs_dim_;
                P_.block<pose_obs_dim_, pose_obs_dim_>(0, 0) *= options_.degeneracy_cov_inflation_;
            }

            break;
        }
    }

    SymmetrizeAndFloorCovariance(P_, options_.min_cov_diag_);
}

}  // namespace lightning
