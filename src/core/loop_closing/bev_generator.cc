//
// Created by MiMo on 25-7-21.
//

#include "core/loop_closing/bev_generator.h"

#include <opencv2/imgproc.hpp>
#include <unordered_map>
#include <cmath>

namespace lightning {

/// 简单的体素降采样
static std::vector<Vec3f> VoxelDownsample(const std::vector<Vec3f>& points, double voxel_size) {
    if (points.empty()) return {};

    struct VoxelHash {
        size_t operator()(const Eigen::Vector3i& v) const {
            size_t h = 0;
            h ^= std::hash<int>()(v[0]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(v[1]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(v[2]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<Eigen::Vector3i, Vec3f, VoxelHash> voxels;

    for (const auto& p : points) {
        Eigen::Vector3i idx;
        idx[0] = static_cast<int>(std::floor(p[0] / voxel_size));
        idx[1] = static_cast<int>(std::floor(p[1] / voxel_size));
        idx[2] = static_cast<int>(std::floor(p[2] / voxel_size));

        auto it = voxels.find(idx);
        if (it == voxels.end()) {
            voxels[idx] = p;
        } else {
            it->second = (it->second + p) * 0.5f;
        }
    }

    std::vector<Vec3f> result;
    result.reserve(voxels.size());
    for (const auto& [k, v] : voxels) {
        result.push_back(v);
    }
    return result;
}

cv::Mat BEVGenerator::Generate(CloudPtr cloud) const {
    if (!cloud || cloud->empty()) {
        return cv::Mat::zeros(options_.bev_size_, options_.bev_size_, CV_8UC1);
    }

    std::vector<Vec3f> points;
    points.reserve(cloud->size());
    for (const auto& pt : cloud->points) {
        points.emplace_back(pt.x, pt.y, pt.z);
    }
    return Generate(points);
}

cv::Mat BEVGenerator::Generate(const std::vector<Vec3f>& points) const {
    const double vs = options_.voxel_size_;
    const double range = options_.range_;
    const int bev_size = options_.bev_size_;

    // 体素降采样
    auto sampled = VoxelDownsample(points, vs);

    // 裁剪到范围
    std::vector<Vec3f> filtered;
    filtered.reserve(sampled.size());
    for (const auto& p : sampled) {
        if (std::abs(p[0]) < range && std::abs(p[1]) < range && std::abs(p[2]) < range) {
            filtered.push_back(p);
        }
    }

    // 创建 BEV 图像
    int max_ind = static_cast<int>(std::floor(range / vs));
    int grid_size = 2 * max_ind + 1;

    // 如果 bev_size != grid_size，需要缩放
    cv::Mat bev = cv::Mat::zeros(grid_size, grid_size, CV_32FC1);

    // 投影点到 BEV 图像
    // 与 BEVPlace2 一致: row = max_ind - floor(y/0.4), col = max_ind - floor(x/0.4)
    for (const auto& p : filtered) {
        int col = max_ind - static_cast<int>(std::floor(p[0] / vs));
        int row = max_ind - static_cast<int>(std::floor(p[1] / vs));

        if (col >= 0 && col < grid_size && row >= 0 && row < grid_size) {
            float& val = bev.at<float>(row, col);
            if (val < 10.0f) {
                val += 1.0f;
            }
        }
    }

    // 归一化到 0-255
    bev *= 10.0f;
    cv::Mat bev_clipped;
    cv::threshold(bev, bev_clipped, 255.0, 255.0, cv::THRESH_TRUNC);
    double max_val;
    cv::minMaxLoc(bev_clipped, nullptr, &max_val);
    if (max_val > 0) {
        bev_clipped = bev_clipped * (255.0 / max_val);
    }

    // 转换为 uint8
    cv::Mat bev_uint8;
    bev_clipped.convertTo(bev_uint8, CV_8UC1);

    // 缩放到目标尺寸
    if (grid_size != bev_size) {
        cv::Mat resized;
        cv::resize(bev_uint8, resized, cv::Size(bev_size, bev_size), 0, 0, cv::INTER_NEAREST);
        return resized;
    }

    return bev_uint8;
}

}  // namespace lightning
