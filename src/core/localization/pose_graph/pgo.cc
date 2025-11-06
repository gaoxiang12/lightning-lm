#include "pgo.h"
#include "pgo_impl.h"
#include "core/lightning_math.hpp"

#include <boost/format.hpp>

#include <glog/logging.h>

namespace lightning::loc {

PGO::PGO() : impl_(new PGOImpl), pose_extrapolator_(new PoseExtrapolator) {
    smoother_ = std::make_shared<PoseSmoother>(pgo::pgo_smooth_factor);
    LOG(INFO) << "smoother factor: " << pgo::pgo_smooth_factor;
}

PGO::~PGO() = default;

void PGO::SetGlobalOutputHandleFunction(PGO::GlobalOutputHandleFunction handle) {
    impl_->output_func_ = std::move(handle);
}

void PGO::SetHighFrequencyGlobalOutputHandleFunction(PGO::GlobalOutputHandleFunction handle) {
    high_freq_output_func_ = std::move(handle);
}

void PGO::PubResult() {
    // 在有需要时，高频外推然后向外发送数据
    if (high_freq_output_func_ && impl_->result_.valid_) {
        // 检查激光定位分值情况
        if (last_lidar_loc_time_ > 0 && impl_->result_.timestamp_ > last_lidar_loc_time_) {
            if (impl_->result_.confidence_ < lidar_loc_score_thd_) {
                localization_unusual_count_++;
                LOG(INFO) << "当前lidar_loc分值较低: " << impl_->result_.confidence_
                          << " , 当前次数: " << localization_unusual_count_;
            } else {
                localization_unusual_count_ = 0;
            }
        }

        last_lidar_loc_time_ = impl_->result_.timestamp_;

        if (localization_unusual_count_ > localization_unusual_thd_ && last_lidar_loc_time_ > 0) {
            localization_unusual_tag_ = true;
            // LOG(INFO) << "连续多次匹配分值过低, 需检查定位精度";
        } else {
            localization_unusual_tag_ = false;
        }

        auto result = impl_->result_;
        ExtrapolateLocResult(result);
        double dt = result.timestamp_ - impl_->result_.timestamp_;

        bool extrap_success = true;

        /// 利用激光定位的外推结果校验DR外推
        if (impl_->lidar_loc_pose_queue_.size() > 5 && impl_->result_.lidar_loc_smooth_flag_ &&
            (result.timestamp_ - impl_->lidar_loc_pose_queue_.back().timestamp_) < 0.3) {
            SE3 recent_lidar_loc_pose = impl_->lidar_loc_pose_queue_.back().pose_;
            /// 要求激光pose队列有效，且激光定位自身满足平滑性要求

            SE3 lidar_loc_extrap_pose;  // 外推之后的激光定位位姿
            TimedPose best_match;
            bool interp_succ = math::PoseInterp<TimedPose>(
                result.timestamp_, impl_->lidar_loc_pose_queue_, [](const TimedPose& tp) { return tp.timestamp_; },
                [](const TimedPose& tp) { return tp.pose_; }, lidar_loc_extrap_pose, best_match, 1.0);

            if (interp_succ) {
                /// 检查激光定位外推结果与DR外推的差异

                /// 条件1. 从激光定位指向外推结果的矢量方向 不应该与激光定位自身外推方向相差太多
                // Vec3d v1 = result.pose_.translation() - recent_lidar_loc_pose.translation();
                // Vec3d v2 = lidar_loc_extrap_pose.translation() - recent_lidar_loc_pose.translation();
                // if (v1.norm() > 0.1 && v2.norm() > 0.1) {
                //     v1.normalize();
                //     v2.normalize();
                //     const double theta_th = std::cos(15 * M_PI / 180.0);  // 角度阈值
                //     if (v1.dot(v2) <= theta_th) {
                //         /// 外推检查不通过
                //         LOG(WARNING) << "外推检查不通过：" << result.pose_.translation().transpose() << ", "
                //                      << lidar_loc_extrap_pose.translation().transpose() << ", dv: " << v1.dot(v2);
                //         extrap_success = false;
                //         result.pose_ = lidar_loc_extrap_pose;
                //     }
                // }

                // SE3 delta = result.pose_.inverse() * lidar_loc_extrap_pose;
                // if (fabs(delta.translation()[0]) > 0.5 || fabs(delta.translation()[1]) > 0.5 ||
                //     delta.so3().log().norm() > 2.0 * M_PI / 180.0) {
                //     /// 外推检查不通过
                //     LOG(WARNING) << "外推检查不通过：" << result.pose_.translation().transpose() << ", "
                //                  << lidar_loc_extrap_pose.translation().transpose()
                //                  << ", ang: " << delta.so3().log().norm();
                //     extrap_success = false;
                //     result.pose_ = lidar_loc_extrap_pose;
                // }
            }
        }

        if (!impl_->dr_pose_queue_.empty()) {
            smoother_->PushDRPose(impl_->dr_pose_queue_.back().GetPose());
        }

        impl_->result_.timestamp_ = result.timestamp_;
        impl_->result_.pose_ = result.pose_;

        SE3 extra_pose = result.pose_;
        smoother_->PushPose(result.pose_);
        result.pose_ = smoother_->GetPose();

        // 输出force 2d
        // common::PoseRPY RPYXYZ = common::math::SE3ToRollPitchYaw(smoother_->GetPose());
        // RPYXYZ.roll = 0;
        // RPYXYZ.pitch = 0;
        // RPYXYZ.z = 0;
        // result.pose_ = common::math::XYZRPYToSE3(RPYXYZ);

        high_freq_output_func_(result);

        impl_->output_pose_queue_.emplace_back(result.timestamp_, result.pose_);
        while (impl_->output_pose_queue_.size() > 1000) {
            impl_->output_pose_queue_.pop_front();
        }

        // LOG_EVERY_N(INFO, 10) << std::fixed << "extrap: " << extra_pose.translation().transpose()
        //                       << ", smoother: " << result.pose_.translation().transpose() << ", dt: " << dt << ", t:
        //                       " << std::setprecision(12) << result.timestamp_;

        if (!extrap_success) {
            /// 外推失效，那么认为DR queue外推失效，清空DR队列
            impl_->dr_pose_queue_.clear();

            /// 同时清空final_output_pose，因为长时间依靠final output可能会导致发散
            /// 所以当外推失效时，下个时刻就是激光定位+LO的pose。如果LO也失效，那就直接是激光定位的Pose
            /// final_output_pose_.clear();
        }

        return;
    }
}

bool PGO::ProcessDR(const NavState& dr_result) {
    UL lock(impl_->data_mutex_);
    /// 假定DR定位是按时间顺序到达的
    double delta_timestamp = 0;
    if (!impl_->dr_pose_queue_.empty()) {
        const double last_stamp = impl_->dr_pose_queue_.back().timestamp_;
        delta_timestamp = dr_result.timestamp_ - last_stamp;
        if (dr_result.timestamp_ < last_stamp) {
            LOG(WARNING) << "当前DR定位的结果的时间戳应当比上一个时间戳数值大，实际相减得"
                         << dr_result.timestamp_ - last_stamp;
            return false;
        }
    }

    impl_->dr_pose_queue_.emplace_back(dr_result);
    if (impl_->dr_pose_queue_.size() > 1) {
        auto curr_it = impl_->dr_pose_queue_.rbegin();
        auto last_it = curr_it;
        ++last_it;
        // curr_it->delta_t_ = curr_it->timestamp_ - last_it->timestamp_;
    }

    while (impl_->dr_pose_queue_.size() >= pgo::PGO_MAX_SIZE_OF_RELATIVE_POSE_QUEUE) {
        impl_->dr_pose_queue_.pop_front();
    }
    if (!impl_->dr_pose_queue_.empty() && !is_parking_) {
        PubResult();
    } else if (is_parking_ && high_freq_output_func_) {
        parking_result_.timestamp_ = dr_result.timestamp_;
        high_freq_output_func_(parking_result_);
    }

    return true;
}

bool PGO::ProcessLidarOdom(const NavState& lio_result) {
    UL lock(impl_->data_mutex_);
    /// 假定LidarOdom定位是按时间顺序到达的
    if (!impl_->lidar_odom_pose_queue_.empty()) {
        const double last_stamp = impl_->lidar_odom_pose_queue_.back().timestamp_;
        if (lio_result.timestamp_ < last_stamp) {
            LOG(WARNING) << "当前LidarOdom定位时间戳回退，实际相减得" << lio_result.timestamp_ - last_stamp;
        }
    }

    // 保存lidarOdom帧
    impl_->lidar_odom_pose_queue_.emplace_back(lio_result);
    if (impl_->lidar_odom_pose_queue_.size() > 1) {
        auto curr_it = impl_->lidar_odom_pose_queue_.rbegin();
        auto last_it = curr_it;
        ++last_it;
        // curr_it->delta_t_ = curr_it->timestamp_ - last_it->timestamp_;
    }

    while (impl_->lidar_odom_pose_queue_.size() >= pgo::PGO_MAX_SIZE_OF_RELATIVE_POSE_QUEUE) {
        impl_->lidar_odom_pose_queue_.pop_front();
    }

    if (!lio_result.lidar_odom_reliable_) {
        impl_->lidar_odom_conflict_with_dr_ = true;
        impl_->lidar_odom_conflict_with_dr_cnt_ = 200;
    } else {
        if (impl_->lidar_odom_conflict_with_dr_cnt_ > 0) {
            impl_->lidar_odom_conflict_with_dr_cnt_--;
        } else {
            impl_->lidar_odom_conflict_with_dr_ = false;
        }
    }

    /// 如果LO的时间比DR更新，则发布LO的递推结果
    if (!impl_->dr_pose_queue_.empty() && lio_result.timestamp_ >= impl_->dr_pose_queue_.back().timestamp_ &&
        !is_parking_) {
        PubResult();
    } else if (is_parking_ && high_freq_output_func_) {
        parking_result_.timestamp_ = lio_result.timestamp_;
        high_freq_output_func_(parking_result_);
    }

    return true;
}

bool PGO::ProcessLidarLoc(const LocalizationResult& loc_result) {
    UL lock(impl_->data_mutex_);
    is_parking_ = loc_result.is_parking_;
    if (is_parking_ && high_freq_output_func_) {
        parking_result_ = loc_result;
        high_freq_output_func_(loc_result);
        return true;
    }

    // 如果相对位姿(DR和LidarOdom有一个即可)还没来，也退出
    if (RelativePoseQueueEmpty()) {
        LOG(WARNING) << "PGO received LidarLoc, but is waiting for LO or DR ... ";
        return false;
    }

    if (!loc_result.lidar_loc_valid_) {
        return false;
    }

    // 不允许时间回退
    static double last_lidar_loc_timestamp = -1;
    double lidar_loc_delta_t = loc_result.timestamp_ - last_lidar_loc_timestamp;
    if (last_lidar_loc_timestamp > 0) {
        if (lidar_loc_delta_t < 0) {
            LOG(ERROR) << "lidar loc 时间回退: " << lidar_loc_delta_t;
            return false;
        } else {
            last_lidar_loc_timestamp = loc_result.timestamp_;
        }
    } else {
        last_lidar_loc_timestamp = loc_result.timestamp_;
    }

    // 增加一个PGO Frame并触发一次PGO
    auto new_frame = std::make_shared<PGOFrame>();
    new_frame->timestamp_ = loc_result.timestamp_;
    new_frame->opti_pose_ = loc_result.pose_;
    new_frame->last_opti_pose_ = loc_result.pose_;
    new_frame->lidar_loc_set_ = true;
    new_frame->lidar_loc_valid_ = loc_result.lidar_loc_valid_;
    new_frame->lidar_loc_pose_ = loc_result.pose_;
    new_frame->lidar_loc_delta_t_ = lidar_loc_delta_t;

    // 捕获lidarLoc给出的置信度
    new_frame->confidence_ = loc_result.confidence_;
    if (loc_result.confidence_ > 0.6) {
        new_frame->lidar_loc_normalized_weight_ = Vec6d::Ones();
    } else {
        const double clap_weight = std::max(loc_result.confidence_, 0.5);
        new_frame->lidar_loc_normalized_weight_ = clap_weight * Vec6d::Ones();
    }

    impl_->result_.lidar_loc_odom_error_normal_ = loc_result.lidar_loc_odom_error_normal_;
    impl_->result_.lidar_loc_smooth_flag_ = loc_result.lidar_loc_smooth_flag_;

    if (!loc_result.lidar_loc_odom_error_normal_) {
        // LOG(WARNING) << "PGO接收到LO失效";
        impl_->lidar_odom_valid_ = false;
        impl_->lidar_odom_valid_cnt_ = 10;
    } else {
        /// 如果Odom有效，也需要累计一段时间
        if (impl_->lidar_odom_valid_cnt_ > 0) {
            impl_->lidar_odom_valid_cnt_--;
        } else {
            // LOG(INFO) << "PGO认为LO生效";
            impl_->lidar_odom_valid_ = true;
        }
    }

    LOG(INFO) << std::setprecision(14) << std::fixed << "PGO received LidarLoc ["
              << new_frame->lidar_loc_pose_.translation().transpose() << "], t=" << new_frame->timestamp_;
    return ProcessPGOFrame(new_frame);
}

bool PGO::ProcessPGOFrame(std::shared_ptr<PGOFrame> frame) {
    impl_->AddPGOFrame(frame);

    impl_->lidar_loc_pose_queue_.emplace_back(impl_->result_.timestamp_, impl_->result_.pose_);
    while (impl_->lidar_loc_pose_queue_.size() > 50) {
        impl_->lidar_loc_pose_queue_.pop_front();
    }

    if (!impl_->dr_pose_queue_.empty() && frame->timestamp_ >= impl_->dr_pose_queue_.back().timestamp_) {
        PubResult();
    }

    return true;
}

std::shared_ptr<PGOFrame> PGO::GetCurrentPGOFrame() const { return impl_->current_frame_; }

bool PGO::Reset() {
    UL lock(impl_->data_mutex_);
    smoother_->Reset();
    return impl_->Reset();
}

void PGO::SetDebug(bool debug) { impl_->SetDebug(debug); }

void PGO::LogWindowState() {
    UL lock(impl_->data_mutex_);
    // 显示最近5个PGOFrame的信息
    auto& window = impl_->frames_;
    int idx1 = (window.size() - 5) >= 0 ? (window.size() - 5) : -1;
    int idx2 = (window.size() - 4) >= 0 ? (window.size() - 4) : -1;
    int idx3 = (window.size() - 3) >= 0 ? (window.size() - 3) : -1;
    int idx4 = (window.size() - 2) >= 0 ? (window.size() - 2) : -1;
    int idx5 = (window.size() - 1) >= 0 ? (window.size() - 1) : -1;
    boost::format fmt("--- %c --- %c --- %c --- %c --- %c ---");
    std::string lidar_info = idx1 >= 0 ? "info" : "empty";
    LOG(INFO) << "Show PGO window state: \n"
              << " ************************************************** \n"
              << " ** index    : --- 1st --- 2nd --- 3rd --- 4th --- 5th \n"
              << " ** lidar loc:"
              << " **   rtk loc:"
              << " **   rel loc:"
              << " ** "
              << " ** // + is valid/good, ? is invalid/bad, W is wrong, G is good \n"
              << " ************************************************** ";

    // LOG(INFO) << "PGO received lidar loc ["
    //     << new_frame->lidar_loc_pose_.translation().transpose() << "], let's go for one shot! ||||||||||||||||| \n"
    //     << "   ********* PGO status ********* \n"
    //     << "   ** current window size : " << impl_->frames_.size() << "\n"
    //     << "   ** gps      queue size : " << impl_->gps_queue_.size() << "\n"
    //     << "   ** LO       queue size : " << impl_->lidar_odom_pose_queue_.size() << "\n"
    //     << "   ** DR       queue size : " << impl_->dr_pose_queue_.size() << "\n"
    //     << "   ********* ********** ********* \n";
}

bool PGO::ExtrapolateLocResult(LocalizationResult& output_result) {
    if (!output_result.valid_) {
        return false;
    }

    auto& dr_pose_queue = impl_->dr_pose_queue_;
    auto& lo_pose_queue = impl_->lidar_odom_pose_queue_;
    SE3 interp_pose;
    NavState best_match;

    /// 确定外推的最远时间，由DR，GPS，LO中取最大者（因为这几个都可能断流）
    double latest_time = impl_->result_.timestamp_;
    if (!dr_pose_queue.empty() && dr_pose_queue.back().timestamp_ > latest_time) {
        latest_time = dr_pose_queue.back().timestamp_;
    }

    if (!lo_pose_queue.empty() && lo_pose_queue.back().timestamp_ > latest_time) {
        latest_time = lo_pose_queue.back().timestamp_;
    }

    // if (!impl_->gps_queue_.empty() && impl_->gps_queue_.back().timestamp_ > latest_time) {
    //     latest_time = impl_->gps_queue_.back().timestamp_;
    // }

    // 其他数据源时间检测imu时间
    if (!dr_pose_queue.empty() && latest_time - dr_pose_queue.back().timestamp_ > imu_interruption_time_thd_) {
        imu_interruption_tag_ = true;
        LOG(ERROR) << "长时间未获取到DR数据, IMU存在断流, 断流时间: " << latest_time - dr_pose_queue.back().timestamp_;
    } else {
        imu_interruption_tag_ = false;
    }

    // 用LO外推到最新时刻
    // if (impl_->lidar_odom_valid_ && !lo_pose_queue.empty() &&
    //     lo_pose_queue.back().timestamp_ > output_result.timestamp_) {
    //     bool lo_interp_success = roki::common::math::PoseInterp<NavState>(
    //         output_result.timestamp_, lo_pose_queue,
    //         [](const NavState& nav_state) { return nav_state.timestamp_; },
    //         [](const NavState& nav_state) { return nav_state.pose_; }, interp_pose, best_match);
    //     if (lo_interp_success) {
    //         SE3 pose_incre = interp_pose.inverse() * lo_pose_queue.back().pose_;
    //         output_result.pose_ = output_result.pose_ * pose_incre;
    //         const double time_incre = lo_pose_queue.back().timestamp_ - output_result.timestamp_;
    //         output_result.timestamp_ = lo_pose_queue.back().timestamp_;
    //         // 如果外推时间比较久(比如超过3s)，跟踪状态改为跟踪相对位姿
    //         if (time_incre > 3.0) {
    //             output_result.status_ = common::GlobalPoseStatus::FOLLOWING_LiDAR_ODOM;
    //         }
    //     }
    // }

    // if (impl_->lidar_odom_conflict_with_dr_ && impl_->lidar_odom_valid_) {
    //     /// 用LO外推到DR时刻
    //     SE3 interp_pose1, interp_pose2;
    //     bool interp1 = common::math::PoseInterp<NavState>(
    //         latest_time, lo_pose_queue, [](const NavState& nav_state) { return nav_state.timestamp_; },
    //         [](const NavState& nav_state) { return nav_state.pose_; }, interp_pose1, best_match);
    //     bool interp2 = common::math::PoseInterp<NavState>(
    //         output_result.timestamp_, lo_pose_queue,
    //         [](const NavState& nav_state) { return nav_state.timestamp_; },
    //         [](const NavState& nav_state) { return nav_state.pose_; }, interp_pose2, best_match);
    //     if (interp1 && interp2) {
    //         SE3 pose_incre = interp_pose2.inverse() * interp_pose1;
    //         output_result.pose_ = output_result.pose_ * pose_incre;
    //         output_result.timestamp_ = latest_time;
    //     }
    // }

    // 其次用DR做外推

    if (!dr_pose_queue.empty() && dr_pose_queue.back().timestamp_ > output_result.timestamp_) {
        double dr_extrap_time = dr_pose_queue.back().timestamp_ - output_result.timestamp_;  // DR 递推的时间

        /// NOTE 当LO失效，lidar loc有较大延时，可能需要DR递推较久的时间

        double dr_pose_inc_th = 5.0 * dr_extrap_time;  // 允许DR外推的距离
        if (!impl_->lidar_odom_valid_) {
            dr_pose_inc_th *= 2;  // lidar odom invalid，放宽
        }
        if (dr_pose_inc_th < 2.0) {
            dr_pose_inc_th = 2.0;
        }

        bool dr_interp_success = math::PoseInterp<NavState>(
            output_result.timestamp_, dr_pose_queue, [](const NavState& nav_state) { return nav_state.timestamp_; },
            [](const NavState& nav_state) { return nav_state.GetPose(); }, interp_pose, best_match);
        if (dr_interp_success) {
            SE3 pose_incre = interp_pose.inverse() * dr_pose_queue.back().GetPose();

            /// 限制此处的pose incre大小
            // if (pose_incre.translation().norm() > dr_pose_inc_th) {
            //     LOG(WARNING) << "pose increment is too large: " << pose_incre.translation().norm()
            //               << ", skip extrapolate, dt=" << dr_extrap_time << ", th: " << dr_pose_inc_th;
            //     return true;
            // }

            output_result.pose_ = output_result.pose_ * pose_incre;
            const double time_incre = dr_pose_queue.back().timestamp_ - output_result.timestamp_;
            output_result.timestamp_ = dr_pose_queue.back().timestamp_;
        }
    }

    if (output_result.timestamp_ < (latest_time - 0.05)) {
        /// 如果中间失效导致最新定位无法更新，那么用激光定位将其外推至最新位置
        SE3 interp_pose_final;
        TimedPose best_match_final;
        if (math::PoseInterp<TimedPose>(
                latest_time, impl_->lidar_loc_pose_queue_, [](const TimedPose& pose) { return pose.timestamp_; },
                [](const TimedPose& pose) { return pose.pose_; }, interp_pose_final, best_match_final, 5.0)) {
            output_result.pose_ = interp_pose_final;
            output_result.timestamp_ = latest_time;
        }
    }

    return true;
}

}  // namespace lightning::loc
