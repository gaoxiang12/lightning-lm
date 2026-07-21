//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_BEVPLACE2_POSE_ESTIMATOR_H
#define LIGHTNING_BEVPLACE2_POSE_ESTIMATOR_H

#include "core/loop_closing/loop_pose_estimator.h"
#include "core/loop_closing/bev_generator.h"

namespace lightning {

/**
 * 基于 BEV 特征匹配的回环位姿估计
 * FAST 特征检测 + BFMatcher + 2D RANSAC
 */
class BEVPlace2PoseEstimator : public LoopPoseEstimator {
public:
    void Init(const std::string& yaml_path) override;
    void Estimate(LoopCandidate& c,
                  const std::vector<Keyframe::Ptr>& all_kfs) override;

private:
    static cv::Matx23d SvdICP(const std::vector<cv::Point2f>& src,
                               const std::vector<cv::Point2f>& dst);

    static cv::Matx23d RigidRansac(const std::vector<cv::Point2f>& pts1,
                                    const std::vector<cv::Point2f>& pts2,
                                    std::vector<uchar>& inlier_mask,
                                    int max_iterations = 1000,
                                    double inlier_threshold = 0.5);

    BEVGenerator bev_gen_;
};

}  // namespace lightning

#endif  // LIGHTNING_BEVPLACE2_POSE_ESTIMATOR_H
