//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_NDT_POSE_ESTIMATOR_H
#define LIGHTNING_NDT_POSE_ESTIMATOR_H

#include "core/loop_closing/loop_pose_estimator.h"

namespace lightning {

/**
 * 基于 NDT 的回环位姿估计
 * 使用多分辨率 NDT 匹配估计回环候选的相对位姿
 */
class NDTPoseEstimator : public LoopPoseEstimator {
public:
    void Init(const std::string& yaml_path) override;
    void Estimate(LoopCandidate& c,
                  const std::vector<Keyframe::Ptr>& all_kfs) override;

private:
    int submap_idx_range_ = 40;
};

}  // namespace lightning

#endif  // LIGHTNING_NDT_POSE_ESTIMATOR_H
