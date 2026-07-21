//
// Created by MiMo on 25-7-21.
//

#include "core/loop_closing/loop_closing_factory.h"
#include "core/loop_closing/spatial_loop_detector.h"
#include "core/loop_closing/ndt_pose_estimator.h"
#include "core/loop_closing/bevplace2_loop_detector.h"
#include "core/loop_closing/bevplace2_pose_estimator.h"

#include <glog/logging.h>

namespace lightning {

std::unique_ptr<LoopDetector> CreateLoopDetector(
    const std::string& type, const std::string& yaml_path) {
    std::unique_ptr<LoopDetector> detector;

    if (type == "spatial") {
        LOG(INFO) << "using spatial loop detector";
        detector = std::make_unique<SpatialLoopDetector>();
    } else if (type == "bevplace2") {
        LOG(INFO) << "using BEVPlace2 loop detector";
        detector = std::make_unique<BEVPlace2LoopDetector>();
    } else {
        LOG(ERROR) << "unknown loop detector type: " << type;
        return nullptr;
    }

    detector->Init(yaml_path);
    return detector;
}

std::unique_ptr<LoopPoseEstimator> CreateLoopPoseEstimator(
    const std::string& type, const std::string& yaml_path) {
    std::unique_ptr<LoopPoseEstimator> estimator;

    if (type == "ndt") {
        LOG(INFO) << "using NDT loop pose estimator";
        estimator = std::make_unique<NDTPoseEstimator>();
    } else if (type == "bevplace2") {
        LOG(INFO) << "using BEVPlace2 loop pose estimator";
        estimator = std::make_unique<BEVPlace2PoseEstimator>();
    } else {
        LOG(ERROR) << "unknown loop pose estimator type: " << type;
        return nullptr;
    }

    estimator->Init(yaml_path);
    return estimator;
}

}  // namespace lightning
