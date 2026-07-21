//
// Created by MiMo on 25-7-21.
//

#include "core/loop_closing/spatial_loop_detector.h"
#include "io/yaml_io.h"

namespace lightning {

void SpatialLoopDetector::Init(const std::string& yaml_path) {
    if (!yaml_path.empty()) {
        YAML_IO yaml(yaml_path);
        loop_kf_gap_ = yaml.GetValueOr<int>("loop_closing", "loop_kf_gap", 20);
        min_id_interval_ = yaml.GetValueOr<int>("loop_closing", "min_id_interval", 20);
        closest_id_th_ = yaml.GetValueOr<int>("loop_closing", "closest_id_th", 50);
        max_range_ = yaml.GetValueOr<double>("loop_closing", "max_range", 30.0);
    }
}

void SpatialLoopDetector::AddKeyframe(Keyframe::Ptr kf) {
    all_keyframes_.emplace_back(kf);
}

std::vector<LoopCandidate> SpatialLoopDetector::Detect(Keyframe::Ptr cur_kf) {
    std::vector<LoopCandidate> candidates;

    Keyframe::Ptr check_first = nullptr;

    if (last_loop_kf_ == nullptr) {
        last_loop_kf_ = cur_kf;
        return candidates;
    }

    if (last_loop_kf_ && (cur_kf->GetID() - last_loop_kf_->GetID()) <= loop_kf_gap_) {
        LOG(INFO) << "skip because last loop kf: " << last_loop_kf_->GetID();
        return candidates;
    }

    for (auto kf : all_keyframes_) {
        if (check_first != nullptr && abs(int(kf->GetID() - check_first->GetID())) <= min_id_interval_) {
            continue;
        }

        if (abs(int(kf->GetID() - cur_kf->GetID())) < closest_id_th_) {
            break;
        }

        Vec3d dt = kf->GetOptPose().translation() - cur_kf->GetOptPose().translation();
        double t2d = dt.head<2>().norm();

        if (t2d < max_range_) {
            LoopCandidate c(kf->GetID(), cur_kf->GetID());
            c.Tij_ = kf->GetLIOPose().inverse() * cur_kf->GetLIOPose();

            candidates.emplace_back(c);
            check_first = kf;
        }
    }

    if (!candidates.empty()) {
        last_loop_kf_ = cur_kf;
    }

    return candidates;
}

}  // namespace lightning
