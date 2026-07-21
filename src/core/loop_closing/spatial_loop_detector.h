//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_SPATIAL_LOOP_DETECTOR_H
#define LIGHTNING_SPATIAL_LOOP_DETECTOR_H

#include "core/loop_closing/loop_detector.h"

namespace lightning {

/**
 * 基于空间距离的回环候选检测
 * 通过 ID 间隔 + 2D 距离阈值过滤候选帧
 */
class SpatialLoopDetector : public LoopDetector {
public:
    void Init(const std::string& yaml_path) override;
    void AddKeyframe(Keyframe::Ptr kf) override;
    std::vector<LoopCandidate> Detect(Keyframe::Ptr cur_kf) override;

private:
    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_loop_kf_ = nullptr;

    int loop_kf_gap_ = 20;
    int min_id_interval_ = 20;
    int closest_id_th_ = 50;
    double max_range_ = 30.0;
};

}  // namespace lightning

#endif  // LIGHTNING_SPATIAL_LOOP_DETECTOR_H
