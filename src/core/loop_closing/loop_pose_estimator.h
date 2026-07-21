//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_LOOP_POSE_ESTIMATOR_H
#define LIGHTNING_LOOP_POSE_ESTIMATOR_H

#include "common/keyframe.h"
#include "common/loop_candidate.h"

#include <string>
#include <vector>

namespace lightning {

/**
 * 回环位姿估计策略接口
 * 不同策略负责估计回环候选帧对的相对位姿
 */
class LoopPoseEstimator {
public:
    virtual ~LoopPoseEstimator() = default;

    /// 初始化（从 YAML 读取参数）
    virtual void Init(const std::string& yaml_path) = 0;

    /// 估计回环候选的相对位姿
    /// @param c 候选帧对（输入 Tij_ 为初始估计，输出更新后的 Tij_ 和分数）
    /// @param all_kfs 所有关键帧（用于构建子图等）
    virtual void Estimate(LoopCandidate& c,
                          const std::vector<Keyframe::Ptr>& all_kfs) = 0;
};

}  // namespace lightning

#endif  // LIGHTNING_LOOP_POSE_ESTIMATOR_H
