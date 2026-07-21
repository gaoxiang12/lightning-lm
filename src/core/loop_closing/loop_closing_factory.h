//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_LOOP_CLOSING_FACTORY_H
#define LIGHTNING_LOOP_CLOSING_FACTORY_H

#include "core/loop_closing/loop_detector.h"
#include "core/loop_closing/loop_pose_estimator.h"

#include <memory>
#include <string>

namespace lightning {

/// 根据配置创建回环检测策略
std::unique_ptr<LoopDetector> CreateLoopDetector(
    const std::string& type, const std::string& yaml_path);

/// 根据配置创建位姿估计策略
std::unique_ptr<LoopPoseEstimator> CreateLoopPoseEstimator(
    const std::string& type, const std::string& yaml_path);

}  // namespace lightning

#endif  // LIGHTNING_LOOP_CLOSING_FACTORY_H
