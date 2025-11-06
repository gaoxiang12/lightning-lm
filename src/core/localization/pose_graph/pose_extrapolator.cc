#include "core/localization/pose_graph/pose_extrapolator.h"
#include "core/lightning_math.hpp"
#include <memory>

namespace lightning::loc {

PoseExtrapolator::PoseExtrapolator() {}
PoseExtrapolator::~PoseExtrapolator() {}

bool PoseExtrapolator::AddDRLocAndExtrapolate(const NavState& dr_loc, SE3& output_pose) {
    UL lock(data_mutex_);
    if (!initialized) return false;

    // 预测源不能回退
    if (dr_loc.timestamp_ < current_time_) {
        return false;
    }

    // 常规处理流传1：根据DR增量，更新currentPose和currTime
    static std::shared_ptr<NavState> last_dr_loc_;
    if (last_dr_loc_ == nullptr) {
        last_dr_loc_.reset(new NavState);
    } else {
        SE3 pose_incre = last_dr_loc_->GetPose().inverse() * dr_loc.GetPose();
        current_pose_ *= pose_incre;
    }
    current_time_ = dr_loc.timestamp_;

    // 常规处理流传2：补偿pose（如有）
    if (pgo_compensate_trans_needed_) {
        const double delta_t = dr_loc.timestamp_ - last_dr_loc_->timestamp_;
        const double curr_dr_speed =
            (dr_loc.GetPose().translation() - last_dr_loc_->GetPose().translation()).norm() / delta_t;
        const double compensate_speed = std::max(kMinTransVelocity, 0.5 * curr_dr_speed);
        const double compensate_distance = compensate_speed * delta_t;
        if (compensate_distance > pgo_curr_trans_gap_.norm()) {
            current_pose_.translation() += pgo_curr_trans_gap_;
            pgo_compensate_trans_needed_ = false;
        } else {
            Vec3d compensate_trans = pgo_curr_trans_gap_.normalized() * compensate_distance;
            pgo_curr_trans_gap_ -= compensate_trans;
            current_pose_.translation() += compensate_trans;
            if ((pgo_curr_trans_gap_.array() < 0).any()) {
                current_pose_.translation() += pgo_curr_trans_gap_;
                pgo_curr_trans_gap_ = Vec3d{0, 0, 0};
                pgo_compensate_trans_needed_ = false;
            }
        }
        LOG(INFO) << "Extrapolator: compensated translation for " << compensate_distance << "m.";
    }

    if (pgo_compensate_rot_needed_) {
        //
    }

    // 常规处理流传3：输出与更新
    output_pose = current_pose_;
    *last_dr_loc_ = dr_loc;
    return true;
}

bool PoseExtrapolator::AddLidarOdomLoc(const NavState& lo_loc) {
    UL lock(data_mutex_);
    if (!initialized) {
        return false;
    }

    // 矫正源应该慢于预测源
    if (lo_loc.timestamp_ > current_time_) {
        //
    }
    //
    return true;
}

bool PoseExtrapolator::AddPGOLoc(const LocalizationResult& pgo_loc) {
    UL lock(data_mutex_);
    // 系统只能由PGOLoc来初始化
    if (!initialized) {
        current_time_ = pgo_loc.timestamp_;
        current_pose_ = pgo_loc.pose_;
        {
            NavState init_nav_state;
            init_nav_state.timestamp_ = current_time_;
            init_nav_state.SetPose(current_pose_);
            output_pose_queue_.push_back(init_nav_state);
        }
        last_pgo_loc_ = pgo_loc;
        initialized = true;
        return true;
    }

    // 时间戳回退的PGOLoc没有意义
    if (pgo_loc.timestamp_ < last_pgo_loc_.timestamp_) {
        return false;
    }

    // PGO源正常要慢于预测源，如果快了，我们单独处理
    if (pgo_loc.timestamp_ > current_time_) {
        current_time_ = pgo_loc.timestamp_;
        current_pose_ = pgo_loc.pose_;

        {
            NavState nav_state;
            nav_state.timestamp_ = current_time_;
            nav_state.SetPose(current_pose_);
            output_pose_queue_.push_back(nav_state);
        }

        last_pgo_loc_ = pgo_loc;
        return true;
    }

    // 以下为常规处理逻辑
    // 当前PGOLoc在history队列上插值，重置补偿量。
    SE3 interp_pose;
    NavState best_match;
    bool pgo_interp_success = math::PoseInterp<NavState>(
        pgo_loc.timestamp_, output_pose_queue_, [](const NavState& nav_state) { return nav_state.timestamp_; },
        [](const NavState& nav_state) { return nav_state.GetPose(); }, interp_pose, best_match);

    if (pgo_interp_success) {
        pgo_curr_trans_gap_ = pgo_loc.pose_.translation() - interp_pose.translation();
        pgo_curr_rot_gap_ = (interp_pose.inverse() * pgo_loc.pose_).so3().log();
        pgo_compensate_trans_needed_ = true;
        pgo_compensate_rot_needed_ = true;
        last_pgo_loc_ = pgo_loc;
        return true;
    }

    last_pgo_loc_ = pgo_loc;
    return false;
}

}  // namespace lightning::loc
