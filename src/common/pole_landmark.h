#ifndef LIGHTNING_POLE_LANDMARK_H
#define LIGHTNING_POLE_LANDMARK_H

#include "common/eigen_types.h"

namespace lightning {

struct PoleLandmark {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id_ = -1;

    /// body系下轴线上一点和方向（用于构建雅可比）
    Vec3d axis_point_body_ = Vec3d::Zero();
    Vec3d axis_dir_body_ = Vec3d::UnitZ();

    /// world系下轴线上一点和方向（用于匹配与地图维护）
    Vec3d axis_point_ = Vec3d::Zero();
    Vec3d axis_dir_ = Vec3d::UnitZ();

    double radius_ = 0.0375;
    double length_ = 0.5;
    double mean_intensity_ = 0.0;
    int support_ = 0;
    double timestamp_ = 0.0;
};

}  // namespace lightning

#endif  // LIGHTNING_POLE_LANDMARK_H

