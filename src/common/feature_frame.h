#ifndef LIGHTNING_FEATURE_FRAME_H
#define LIGHTNING_FEATURE_FRAME_H

#include "common/point_def.h"

namespace lightning {

struct FeatureFrame {
    using Ptr = std::shared_ptr<FeatureFrame>;

    double timestamp_ = 0.0;

    CloudPtr organized_cloud_ = CloudPtr(new PointCloudType());
    CloudPtr corner_cloud_ = CloudPtr(new PointCloudType());
    CloudPtr surf_cloud_ = CloudPtr(new PointCloudType());
    CloudPtr corner_cloud_ds_ = CloudPtr(new PointCloudType());
    CloudPtr surf_cloud_ds_ = CloudPtr(new PointCloudType());

    std::vector<int> start_ring_index_;
    std::vector<int> end_ring_index_;
    std::vector<int> point_col_ind_;
    std::vector<float> point_range_;
};

}  // namespace lightning

#endif