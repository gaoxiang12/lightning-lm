//
// Created by MiMo on 25-7-21.
//

#ifndef LIGHTNING_BEV_GENERATOR_H
#define LIGHTNING_BEV_GENERATOR_H

#include "common/eigen_types.h"
#include "common/point_def.h"

#include <opencv2/core.hpp>
#include <vector>

namespace lightning {

/**
 * BEV 图像生成器
 * 将 3D 点云转换为 2D 鸟瞰图灰度图像
 */
class BEVGenerator {
public:
    struct Options {
        double voxel_size_ = 0.4;
        double range_ = 40.0;
        int bev_size_ = 200;
    };

    BEVGenerator() = default;
    explicit BEVGenerator(Options options) : options_(options) {}

    cv::Mat Generate(CloudPtr cloud) const;
    cv::Mat Generate(const std::vector<Vec3f>& points) const;

    const Options& GetOptions() const { return options_; }

private:
    Options options_;
};

}  // namespace lightning

#endif  // LIGHTNING_BEV_GENERATOR_H
