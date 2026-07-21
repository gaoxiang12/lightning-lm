//
// Created by MiMo on 25-7-21.
//

#include "core/loop_closing/bevplace2_pose_estimator.h"
#include "io/yaml_io.h"

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace lightning {

void BEVPlace2PoseEstimator::Init(const std::string& yaml_path) {
    BEVGenerator::Options bev_opts;
    if (!yaml_path.empty()) {
        YAML_IO yio(yaml_path);
        bev_opts.voxel_size_ = yio.GetValueOr<double>("bevplace2", "bev_resolution", 0.4);
        bev_opts.range_ = yio.GetValueOr<double>("bevplace2", "bev_range", 40.0);
        bev_opts.bev_size_ = yio.GetValueOr<int>("bevplace2", "bev_size", 200);
    }
    bev_gen_ = BEVGenerator(bev_opts);
}

cv::Matx23d BEVPlace2PoseEstimator::SvdICP(const std::vector<cv::Point2f>& src,
                                             const std::vector<cv::Point2f>& dst) {
    if (src.size() < 2) return cv::Matx23d::eye();

    cv::Point2f mean_src(0, 0), mean_dst(0, 0);
    for (size_t i = 0; i < src.size(); i++) {
        mean_src += src[i];
        mean_dst += dst[i];
    }
    mean_src *= (1.0f / src.size());
    mean_dst *= (1.0f / dst.size());

    std::vector<cv::Point2f> src_n(src.size()), dst_n(dst.size());
    for (size_t i = 0; i < src.size(); i++) {
        src_n[i] = src[i] - mean_src;
        dst_n[i] = dst[i] - mean_dst;
    }

    // 构建 2x2 协方差矩阵
    cv::Matx22d S(0, 0, 0, 0);
    for (size_t i = 0; i < src.size(); i++) {
        S(0, 0) += src_n[i].x * dst_n[i].x;
        S(0, 1) += src_n[i].x * dst_n[i].y;
        S(1, 0) += src_n[i].y * dst_n[i].x;
        S(1, 1) += src_n[i].y * dst_n[i].y;
    }

    // SVD 分解
    cv::Mat S_mat(2, 2, CV_64F);
    S_mat.at<double>(0, 0) = S(0, 0); S_mat.at<double>(0, 1) = S(0, 1);
    S_mat.at<double>(1, 0) = S(1, 0); S_mat.at<double>(1, 1) = S(1, 1);

    cv::Mat U_mat, S_vec, Vt_mat;
    cv::SVDecomp(S_mat, U_mat, S_vec, Vt_mat);

    // 旋转矩阵 R = Vt^T * U^T
    cv::Mat R_mat = Vt_mat.t() * U_mat.t();

    // 修正反射
    double det = cv::determinant(R_mat);
    if (det < 0) {
        cv::Matx22d S_fix(1, 0, 0, -1);
        cv::Mat S_fix_mat(2, 2, CV_64F);
        S_fix_mat.at<double>(0, 0) = 1; S_fix_mat.at<double>(0, 1) = 0;
        S_fix_mat.at<double>(1, 0) = 0; S_fix_mat.at<double>(1, 1) = -1;
        R_mat = Vt_mat.t() * S_fix_mat * U_mat.t();
    }

    // 平移 t = mean_dst - R * mean_src
    cv::Vec2d ms(mean_src.x, mean_src.y);
    cv::Vec2d md(mean_dst.x, mean_dst.y);
    cv::Vec2d r00(R_mat.at<double>(0, 0), R_mat.at<double>(0, 1));
    cv::Vec2d r10(R_mat.at<double>(1, 0), R_mat.at<double>(1, 1));
    cv::Vec2d t(md[0] - r00.dot(ms), md[1] - r10.dot(ms));

    cv::Matx23d result;
    result(0, 0) = R_mat.at<double>(0, 0); result(0, 1) = R_mat.at<double>(0, 1); result(0, 2) = t[0];
    result(1, 0) = R_mat.at<double>(1, 0); result(1, 1) = R_mat.at<double>(1, 1); result(1, 2) = t[1];
    return result;
}

cv::Matx23d BEVPlace2PoseEstimator::RigidRansac(const std::vector<cv::Point2f>& pts1,
                                                  const std::vector<cv::Point2f>& pts2,
                                                  std::vector<uchar>& inlier_mask,
                                                  int max_iterations,
                                                  double inlier_threshold) {
    if (pts1.size() < 2) {
        inlier_mask.assign(pts1.size(), 0);
        return cv::Matx23d::eye();
    }

    // 与 BEVPlace2 一致: 交换 x/y
    std::vector<cv::Point2f> p1(pts1.size()), p2(pts2.size());
    for (size_t i = 0; i < pts1.size(); i++) {
        p1[i] = cv::Point2f(pts1[i].y, pts1[i].x);
        p2[i] = cv::Point2f(pts2[i].y, pts2[i].x);
    }

    int max_inliers = 0;
    cv::Matx23d best_mat = cv::Matx23d::eye();
    inlier_mask.assign(p1.size(), 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(p1.size()) - 1);

    for (int iter = 0; iter < max_iterations; iter++) {
        int idx1 = dist(rng);
        int idx2 = dist(rng);
        if (idx1 == idx2) continue;

        std::vector<cv::Point2f> ss = {p1[idx1], p1[idx2]};
        std::vector<cv::Point2f> sd = {p2[idx1], p2[idx2]};

        cv::Matx23d mat = SvdICP(ss, sd);

        int inlier_count = 0;
        std::vector<uchar> mask(p1.size());
        for (size_t i = 0; i < p1.size(); i++) {
            double x = mat(0, 0) * p1[i].x + mat(0, 1) * p1[i].y + mat(0, 2);
            double y = mat(1, 0) * p1[i].x + mat(1, 1) * p1[i].y + mat(1, 2);
            double ex = x - p2[i].x;
            double ey = y - p2[i].y;
            double err = std::sqrt(ex * ex + ey * ey);
            mask[i] = (err < inlier_threshold) ? 1 : 0;
            if (mask[i]) inlier_count++;
        }

        if (inlier_count > max_inliers) {
            max_inliers = inlier_count;
            best_mat = mat;
            inlier_mask = mask;
        }
    }

    // 用所有内点重新拟合
    std::vector<cv::Point2f> is, id;
    for (size_t i = 0; i < p1.size(); i++) {
        if (inlier_mask[i]) {
            is.push_back(p1[i]);
            id.push_back(p2[i]);
        }
    }
    if (is.size() >= 2) {
        best_mat = SvdICP(is, id);
    }

    return best_mat;
}

void BEVPlace2PoseEstimator::Estimate(LoopCandidate& c,
                                        const std::vector<Keyframe::Ptr>& all_kfs) {
    auto kf1 = all_kfs.at(c.idx1_);
    auto kf2 = all_kfs.at(c.idx2_);

    cv::Mat bev1 = bev_gen_.Generate(kf1->GetCloud());
    cv::Mat bev2 = bev_gen_.Generate(kf2->GetCloud());

    if (bev1.empty() || bev2.empty()) {
        c.ndt_score_ = 0;
        return;
    }

    // FAST 特征检测
    auto fast = cv::FastFeatureDetector::create();
    std::vector<cv::KeyPoint> kps1, kps2;
    fast->detect(bev1, kps1);
    fast->detect(bev2, kps2);

    if (kps1.empty() || kps2.empty()) {
        c.ndt_score_ = 0;
        return;
    }

    // 简化描述子：关键点周围 patch
    auto extractPatch = [](const cv::Mat& bev, const cv::KeyPoint& kp) -> cv::Mat {
        int x = static_cast<int>(kp.pt.x);
        int y = static_cast<int>(kp.pt.y);
        x = std::max(0, std::min(x, bev.cols - 1));
        y = std::max(0, std::min(y, bev.rows - 1));
        const int ps = 16;
        int x0 = std::max(0, x - ps / 2);
        int y0 = std::max(0, y - ps / 2);
        int x1 = std::min(bev.cols, x0 + ps);
        int y1 = std::min(bev.rows, y0 + ps);
        cv::Mat patch = bev(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
        cv::Mat desc;
        patch.reshape(1, 1).convertTo(desc, CV_32F);
        return desc;
    };

    std::vector<cv::Mat> descs1, descs2;
    for (const auto& kp : kps1) descs1.push_back(extractPatch(bev1, kp));
    for (const auto& kp : kps2) descs2.push_back(extractPatch(bev2, kp));

    // BFMatcher
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descs1, descs2, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
            good_matches.push_back(m[0]);
        }
    }

    if (good_matches.size() < 2) {
        c.ndt_score_ = 0;
        return;
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(kps1[m.queryIdx].pt);
        pts2.push_back(kps2[m.trainIdx].pt);
    }

    // 坐标转 metric
    int im_side = bev1.rows;
    float resolution = static_cast<float>(bev_gen_.GetOptions().voxel_size_);
    std::vector<cv::Point2f> pm1(pts1.size()), pm2(pts2.size());
    for (size_t i = 0; i < pts1.size(); i++) {
        pm1[i] = cv::Point2f((im_side / 2.0f - pts1[i].x) * resolution,
                              (im_side / 2.0f - pts1[i].y) * resolution);
        pm2[i] = cv::Point2f((im_side / 2.0f - pts2[i].x) * resolution,
                              (im_side / 2.0f - pts2[i].y) * resolution);
    }

    std::vector<uchar> inlier_mask;
    cv::Matx23d H = RigidRansac(pm1, pm2, inlier_mask);

    int inlier_count = 0;
    for (uchar m : inlier_mask) inlier_count += m;

    if (inlier_count < 3) {
        c.ndt_score_ = 0;
        return;
    }

    double theta = std::atan2(-H(0, 1), H(0, 0));
    double tx = H(0, 2);
    double ty = H(1, 2);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R(0, 0) = std::cos(theta); R(0, 1) = -std::sin(theta);
    R(1, 0) = std::sin(theta); R(1, 1) = std::cos(theta);

    // H 将 kf1 坐标变换到 kf2: p_kf2 = H * p_kf1
    // Tij_ 约定为 T_kf1_kf2，与图优化边 0→1 一致
    SE3 T(R, Eigen::Vector3d(tx, ty, 0.0));
    c.Tij_ = T;

    c.ndt_score_ = static_cast<double>(inlier_count) / pts1.size();

    LOG(INFO) << "BEVPlace2 pose: inliers=" << inlier_count << "/" << pts1.size()
              << " theta=" << theta * 180.0 / M_PI << " deg"
              << " t=(" << tx << "," << ty << ")";
}

}  // namespace lightning
