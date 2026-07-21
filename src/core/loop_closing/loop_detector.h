//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_LOOP_DETECTOR_H
#define LIGHTNING_LOOP_DETECTOR_H

#include "common/keyframe.h"
#include "common/loop_candidate.h"

#include <string>
#include <vector>

namespace lightning {

/**
 * 回环候选检测策略接口
 * 不同策略负责检测可能的回环候选帧对
 */
class LoopDetector {
public:
    virtual ~LoopDetector() = default;

    /// 初始化（从 YAML 读取参数）
    virtual void Init(const std::string& yaml_path) = 0;

    /// 添加新关键帧（维护数据库/索引）
    virtual void AddKeyframe(Keyframe::Ptr kf) = 0;

    /// 检测回环候选
    virtual std::vector<LoopCandidate> Detect(Keyframe::Ptr cur_kf) = 0;
};

}  // namespace lightning

#endif  // LIGHTNING_LOOP_DETECTOR_H
