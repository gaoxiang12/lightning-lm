//
// Created by MiMo on 25-7-21.
//

#include "core/loop_closing/ndt_pose_estimator.h"
#include "utils/pointcloud_utils.h"

#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>

namespace lightning {

void NDTPoseEstimator::Init(const std::string& yaml_path) {
    // submap_idx_range uses a fixed default; NDT params are hardcoded in Estimate
}

void NDTPoseEstimator::Estimate(LoopCandidate& c,
                                 const std::vector<Keyframe::Ptr>& all_kfs) {
    auto kf1 = all_kfs.at(c.idx1_), kf2 = all_kfs.at(c.idx2_);

    auto build_submap = [&](int given_id, bool build_in_world) -> CloudPtr {
        CloudPtr submap(new PointCloudType);
        for (int idx = -submap_idx_range_; idx < submap_idx_range_; idx += 4) {
            int id = idx + given_id;
            if (id < 0 || id >= static_cast<int>(all_kfs.size())) {
                continue;
            }

            auto kf = all_kfs[id];
            CloudPtr cloud = kf->GetCloud();

            if (cloud->empty()) {
                continue;
            }

            SE3 Twb = kf->GetOptPose();

            if (!build_in_world) {
                Twb = all_kfs.at(given_id)->GetOptPose().inverse() * Twb;
            }

            CloudPtr cloud_trans(new PointCloudType);
            pcl::transformPointCloud(*cloud, *cloud_trans, Twb.matrix());

            *submap += *cloud_trans;
        }
        return submap;
    };

    auto submap_kf1 = build_submap(kf1->GetID(), true);

    CloudPtr submap_kf2 = kf2->GetCloud();

    if (submap_kf1->empty() || submap_kf2->empty()) {
        c.ndt_score_ = 0;
        return;
    }

    Mat4f Tw2 = kf2->GetOptPose().matrix().cast<float>();

    CloudPtr output(new PointCloudType);
    std::vector<double> res{10.0, 5.0, 2.0, 1.0};

    CloudPtr rough_map1, rough_map2;

    for (auto& r : res) {
        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.05);
        ndt.setStepSize(0.7);
        ndt.setMaximumIterations(40);

        ndt.setResolution(r);
        rough_map1 = VoxelGrid(submap_kf1, r * 0.1);
        rough_map2 = VoxelGrid(submap_kf2, r * 0.1);
        ndt.setInputTarget(rough_map1);
        ndt.setInputSource(rough_map2);

        ndt.align(*output, Tw2);
        Tw2 = ndt.getFinalTransformation();

        c.ndt_score_ = ndt.getTransformationProbability();
    }

    Mat4d T = Tw2.cast<double>();
    Quatd q(T.block<3, 3>(0, 0));
    q.normalize();
    Vec3d t = T.block<3, 1>(0, 3);

    c.Tij_ = kf1->GetOptPose().inverse() * SE3(q, t);
}

}  // namespace lightning
