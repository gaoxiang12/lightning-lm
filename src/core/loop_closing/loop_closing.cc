//
// Created by xiang on 25-4-21.
//

#include "core/loop_closing/loop_closing.h"
#include "common/keyframe.h"
#include "common/loop_candidate.h"
#include "core/loop_closing/loop_closing_factory.h"

#include "core/opti_algo/algo_select.h"
#include "core/robust_kernel/cauchy.h"
#include "core/types/edge_se3.h"
#include "core/types/edge_se3_height_prior.h"
#include "core/types/vertex_se3.h"
#include "io/yaml_io.h"

#include <yaml-cpp/yaml.h>

namespace lightning {

LoopClosing::~LoopClosing() {
    if (options_.online_mode_) {
        kf_thread_.Quit();
    }
}

bool LoopClosing::Init(const std::string yaml_path) {
    /// setup miao
    miao::OptimizerConfig config(miao::AlgorithmType::LEVENBERG_MARQUARDT,
                                 miao::LinearSolverType::LINEAR_SOLVER_SPARSE_EIGEN, false);
    config.incremental_mode_ = true;
    optimizer_ = miao::SetupOptimizer<6, 3>(config);

    info_motion_.setIdentity();
    info_motion_.block<3, 3>(0, 0) =
        Mat3d::Identity() * 1.0 / (options_.motion_trans_noise_ * options_.motion_trans_noise_);
    info_motion_.block<3, 3>(3, 3) =
        Mat3d::Identity() * 1.0 / (options_.motion_rot_noise_ * options_.motion_rot_noise_);

    info_loops_.setIdentity();
    info_loops_.block<3, 3>(0, 0) = Mat3d::Identity() * 1.0 / (options_.loop_trans_noise_ * options_.loop_trans_noise_);
    info_loops_.block<3, 3>(3, 3) = Mat3d::Identity() * 1.0 / (options_.loop_rot_noise_ * options_.loop_rot_noise_);

    if (!yaml_path.empty()) {
        auto yaml = YAML::LoadFile(yaml_path);

        // 读取策略类型
        if (yaml["loop_closing"]["detector"]) {
            options_.detector_type_ = yaml["loop_closing"]["detector"].as<std::string>();
        }
        if (yaml["loop_closing"]["pose_estimator"]) {
            options_.pose_estimator_type_ = yaml["loop_closing"]["pose_estimator"].as<std::string>();
        }

        // 读取图优化参数
        YAML_IO yio(yaml_path);
        options_.with_height_ = yio.GetValueOr<bool>("loop_closing", "with_height", true);
        options_.ndt_score_th_ = yio.GetValueOr<double>("loop_closing", "ndt_score_th", 1.0);
        options_.height_noise_ = yio.GetValueOr<double>("loop_closing", "height_noise", 0.1);
    }

    // 创建策略
    detector_ = CreateLoopDetector(options_.detector_type_, yaml_path);
    if (!detector_) {
        LOG(ERROR) << "failed to create loop detector: " << options_.detector_type_;
        return false;
    }

    pose_estimator_ = CreateLoopPoseEstimator(options_.pose_estimator_type_, yaml_path);
    if (!pose_estimator_) {
        LOG(ERROR) << "failed to create loop pose estimator: " << options_.pose_estimator_type_;
        return false;
    }

    if (options_.online_mode_) {
        LOG(INFO) << "loop closing module is running in online mode";
        kf_thread_.SetProcFunc([this](Keyframe::Ptr kf) { HandleKF(kf); });
        kf_thread_.SetName("handle loop closure");
        kf_thread_.Start();
    }

    return true;
}

void LoopClosing::AddKF(Keyframe::Ptr kf) {
    if (options_.online_mode_) {
        kf_thread_.AddMessage(kf);
    } else {
        HandleKF(kf);
    }
}

void LoopClosing::HandleKF(Keyframe::Ptr kf) {
    if (kf == last_kf_) {
        return;
    }

    if (!detector_ || !pose_estimator_) {
        return;
    }

    cur_kf_ = kf;
    all_keyframes_.emplace_back(kf);

    // 1. 策略化的候选检测
    detector_->AddKeyframe(kf);
    candidates_ = detector_->Detect(kf);

    if (options_.verbose_) {
        LOG(INFO) << "lc: get kf " << cur_kf_->GetID() << " candi: " << candidates_.size();
    }

    // 2. 策略化的位姿估计
    for (auto& c : candidates_) {
        pose_estimator_->Estimate(c, all_keyframes_);
    }

    // 过滤成功的候选（使用 ndt_score_th 门控）
    std::vector<LoopCandidate> succ_candidates;
    for (auto& c : candidates_) {
        if (c.ndt_score_ > options_.ndt_score_th_) {
            succ_candidates.emplace_back(c);
        }
    }

    if (options_.verbose_ && !succ_candidates.empty()) {
        LOG(INFO) << "success: " << succ_candidates.size() << "/" << candidates_.size();
    }

    candidates_.swap(succ_candidates);

    // 3. 图优化
    PoseOptimization();

    last_kf_ = kf;
}

void LoopClosing::PoseOptimization() {
    auto v = std::make_shared<miao::VertexSE3>();
    v->SetId(cur_kf_->GetID());
    v->SetEstimate(cur_kf_->GetOptPose());

    optimizer_->AddVertex(v);
    kf_vert_.emplace_back(v);

    /// 上一个关键帧的运动约束
    for (int i = 1; i < 3; i++) {
        int id = cur_kf_->GetID() - i;
        if (id >= 0) {
            auto last_kf = all_keyframes_[id];
            auto e = std::make_shared<miao::EdgeSE3>();
            e->SetVertex(0, optimizer_->GetVertex(last_kf->GetID()));
            e->SetVertex(1, v);

            SE3 motion = last_kf->GetLIOPose().inverse() * cur_kf_->GetLIOPose();
            e->SetMeasurement(motion);
            e->SetInformation(info_motion_);
            optimizer_->AddEdge(e);
        }
    }

    if (options_.with_height_) {
        /// 高度约束
        auto e = std::make_shared<miao::EdgeHeightPrior>();
        e->SetVertex(0, v);
        e->SetMeasurement(0);
        e->SetInformation(Mat1d::Identity() * 1.0 / (options_.height_noise_ * options_.height_noise_));
        optimizer_->AddEdge(e);
    }

    /// 回环的约束
    for (auto& c : candidates_) {
        auto e = std::make_shared<miao::EdgeSE3>();
        e->SetVertex(0, optimizer_->GetVertex(c.idx1_));
        e->SetVertex(1, optimizer_->GetVertex(c.idx2_));
        e->SetMeasurement(c.Tij_);
        e->SetInformation(info_loops_);

        auto rk = std::make_shared<miao::RobustKernelCauchy>();
        rk->SetDelta(options_.rk_loop_th_);
        e->SetRobustKernel(rk);

        optimizer_->AddEdge(e);
        edge_loops_.emplace_back(e);
    }

    if (optimizer_->GetEdges().empty()) {
        return;
    }

    if (candidates_.empty()) {
        return;
    }

    optimizer_->InitializeOptimization();
    optimizer_->SetVerbose(false);

    optimizer_->Optimize(20);

    /// remove outliers
    int cnt_outliers = 0;
    for (auto& e : edge_loops_) {
        if (e->GetRobustKernel() == nullptr) {
            continue;
        }

        if (e->Chi2() > e->GetRobustKernel()->Delta()) {
            e->SetLevel(1);
            cnt_outliers++;
        } else {
            e->SetRobustKernel(nullptr);
        }
    }

    if (options_.verbose_) {
        LOG(INFO) << "loop outliers: " << cnt_outliers << "/" << edge_loops_.size();
    }

    /// get results
    for (auto& vert : kf_vert_) {
        SE3 pose = vert->Estimate();
        all_keyframes_[vert->GetId()]->SetOptPose(pose);
    }

    if (loop_cb_) {
        loop_cb_();
    }

    LOG(INFO) << "optimize finished, loops: " << edge_loops_.size();
}

std::vector<std::pair<SE3, SE3>> LoopClosing::GetLoopEdges() const {
    std::vector<std::pair<SE3, SE3>> edges;
    for (const auto& e : edge_loops_) {
        if (e->Level() != 0) continue;  // 跳过被标记为 outlier 的边
        int id0 = e->GetVertex(0)->GetId();
        int id1 = e->GetVertex(1)->GetId();
        if (id0 >= 0 && id0 < static_cast<int>(all_keyframes_.size()) &&
            id1 >= 0 && id1 < static_cast<int>(all_keyframes_.size())) {
            edges.emplace_back(all_keyframes_[id0]->GetOptPose(),
                               all_keyframes_[id1]->GetOptPose());
        }
    }
    return edges;
}

}  // namespace lightning
