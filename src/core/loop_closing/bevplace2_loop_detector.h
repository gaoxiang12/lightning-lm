//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_BEVPLACE2_LOOP_DETECTOR_H
#define LIGHTNING_BEVPLACE2_LOOP_DETECTOR_H

#include "core/loop_closing/loop_detector.h"
#include "core/loop_closing/bev_generator.h"

#include <memory>
#include <string>
#include <vector>

namespace lightning {

/**
 * 基于 BEVPlace2 描述子检索的回环候选检测
 */
class BEVPlace2LoopDetector : public LoopDetector {
public:
    void Init(const std::string& yaml_path) override;
    void AddKeyframe(Keyframe::Ptr kf) override;
    std::vector<LoopCandidate> Detect(Keyframe::Ptr cur_kf) override;

private:
    bool StartInferenceBackend();

    /// 提取描述子，失败返回空 vector
    std::vector<float> ExtractDescriptor(CloudPtr cloud);

    BEVGenerator bev_gen_;

    std::vector<std::vector<float>> database_descs_;
    std::vector<Keyframe::Ptr> database_kfs_;

    // 当前帧缓存（避免重复推理）
    std::vector<float> cur_frame_desc_;
    Keyframe::Ptr cur_frame_kf_ = nullptr;

    FILE* python_stdin_ = nullptr;
    FILE* python_stdout_ = nullptr;
    int python_pid_ = -1;
    bool backend_ready_ = false;

    double match_threshold_ = 1.5;  // 欧氏距离阈值（L2 归一化描述子范围 [0,2]）
    int min_id_gap_ = 50;
};

}  // namespace lightning

#endif  // LIGHTNING_BEVPLACE2_LOOP_DETECTOR_H
