#include "pgo_impl.h"
#include "core/lightning_math.hpp"
#include "core/opti_algo/algo_select.h"
#include "core/robust_kernel/robust_kernel_all.h"

#include <boost/format.hpp>
#include <cassert>
#include <cmath>
#include <iomanip>

namespace lightning::loc {

/**
 * 几个踩过的坑
 * - g2o fix某个vertex后，unary
 * edge的计算可能被跳过，导致拿chi2的时候可能拿到一个非法值，很神奇；所以最好是拿chi2前都重新计算一遍
 */

namespace {

/// SE3 转g2o的SE3Quat

/// 把约束边信息转化为字符串
template <typename T>
std::string print_info(const std::vector<T>& edges, double th = 0) {
    std::vector<double> chi2;
    for (auto& edge : edges) {
        if (edge->Level() == 0) {
            chi2.push_back(edge->Chi2());
        }
    }

    std::sort(chi2.begin(), chi2.end());
    double ave_chi2 = std::accumulate(chi2.begin(), chi2.end(), 0.0) / chi2.size();
    boost::format fmt("数量: %d, 均值: %f, 中位数: %f, 0.1分位: %f, 0.9分位: %f, 最大值: %f, 阈值: %f\n");
    if (!chi2.empty()) {
        std::string str = (fmt % chi2.size() % ave_chi2 % chi2[chi2.size() / 2] % chi2[int(chi2.size() * 0.1)] %
                           chi2[int(chi2.size() * 0.9)] % chi2.back() % th)
                              .str();
        return str;
    }
    return std::string("");
}

}  // namespace

/// alias for pgo option
PGOImpl::PGOImpl(Options options) {
    options_ = options;

    /// set noise params
    auto set6dnoise = [](Vec6d& noise_item, const double& pos_noise, const double& ang_noise) {
        noise_item.head<3>() = Vec3d::Ones() * pos_noise;
        noise_item.tail<3>() = Vec3d::Ones() * ang_noise;
    };

    set6dnoise(lidar_loc_noise_, options_.lidar_loc_pos_noise, options_.lidar_loc_ang_noise);
    set6dnoise(lidar_odom_rel_noise_, options_.lidar_odom_pos_noise, options_.lidar_odom_ang_noise);
    set6dnoise(dr_rel_noise_, options_.dr_pos_noise, options_.dr_ang_noise);

    // Setup solver
    miao::OptimizerConfig config(miao::AlgorithmType::LEVENBERG_MARQUARDT,
                                 miao::LinearSolverType::LINEAR_SOLVER_SPARSE_EIGEN, false);
    config.incremental_mode_ = true;
    config.max_vertex_size_ = options_.PGO_MAX_FRAMES;
    optimizer_ = miao::SetupOptimizer<6, 3>(config);
}

bool PGOImpl::Reset() {
    LOG(WARNING) << "PGO is reset";
    CleanProblem();
    frames_.clear();
    frames_by_id_.clear();
    current_frame_ = nullptr;
    last_frame_ = nullptr;
    return true;
}

void PGOImpl::AddPGOFrame(std::shared_ptr<PGOFrame> pgo_frame) {
    assert(pgo_frame != nullptr);
    if (last_frame_ != nullptr) {
        const double adjacent_dalta_t = pgo_frame->timestamp_ - last_frame_->timestamp_;
        if (adjacent_dalta_t < 0.) {
            LOG(WARNING) << "PGO received pgoframe, however timestamp rollback for " << adjacent_dalta_t << "senonds!";
            return;
        }
        pgo_frame->lidar_loc_delta_t_ = adjacent_dalta_t;
    }

    if (debug_) {
        LOG(INFO) << "inserting pgo frame, it's timestamp is " << std::setprecision(18) << pgo_frame->timestamp_;
    }

    // 这里尝试设置相对位姿观测，如果上游（通常是激光定位）给了就跳过；
    // 如果 LidarOdom 和 DR 都设置失败，结束本函数。
    bool interp_lio_success = AssignLidarOdomPoseIfNeeded(pgo_frame);
    bool interp_dr_success = AssignDRPoseIfNeeded(pgo_frame);
    if (!interp_lio_success && !interp_dr_success) {
        LOG(ERROR) << "PGO received pgo frame, but assign relative pose failed!";
        return;
    }

    is_in_map_ = pgo_frame->lidar_loc_set_ && pgo_frame->lidar_loc_valid_;
    if (!is_in_map_) {
        //
        LOG(ERROR) << "PGO received PGOFrame with lidar_loc_set_(" << pgo_frame->lidar_loc_set_
                   << "), lidar_loc_valid_(" << pgo_frame->lidar_loc_valid_ << "); Reject It!";
        return;
    }

    pgo_frame->frame_id_ = accumulated_frame_id_++;
    current_frame_ = pgo_frame;
    frames_by_id_.emplace(pgo_frame->frame_id_, pgo_frame);

    // 由于到达时间可能不一致，最好是插到正确的位置（仅限多线程，单线程没这问题）
    // 2023-02-16：在唯一触发源唯一的情况下，无需考虑到达时间不一致问题
    frames_.emplace_back(pgo_frame);

    /// 触发一次优化
    RunOptimization();

    // // 根据优化更新一些状态量
    // UpdatePoseGraphState();

    // 输出结果信息
    CollectOptimizationStatistics();

    // 清空
    // CleanProblem();

    /// 需要时，删除一部分
    SlideWindowAdaptively();

    last_frame_ = current_frame_;
}

bool PGOImpl::AssignLidarOdomPoseIfNeeded(std::shared_ptr<PGOFrame> frame) {
    assert(frame != nullptr);
    SE3 interp_pose;
    Vec3d interp_vel_b;
    NavState best_match;
    bool lo_interp_done = false;

    // 无论如何，我们都要寻找最近的lidarOdom帧，用于判断PGOFrame中lidarOdom的置信度状态
    lo_interp_done = math::PoseInterp<NavState>(
        frame->timestamp_, lidar_odom_pose_queue_, [](const NavState& nav_state) { return nav_state.timestamp_; },
        [](const NavState& nav_state) { return nav_state.GetPose(); }, interp_pose, best_match);
    if (lo_interp_done) {
        UpdateLidarOdomStatusInFrame(best_match, frame);
        // frame->lidar_odom_delta_t_ = best_match.delta_t_;
    }

    // 如果PGOFrame中还没有设置LO插值位姿，设置它。
    if (!frame->lidar_odom_set_) {
        if (lo_interp_done) {
            interp_vel_b = best_match.GetRot().matrix().transpose() * best_match.GetVel();
            frame->lidar_odom_set_ = true;
            frame->lidar_odom_valid_ = true;
            frame->lidar_odom_pose_ = interp_pose;
            frame->lidar_odom_vel_ = interp_vel_b;
            static int counts = 0;
            ++counts;
            return true;

        } else {
            // lidarodom 不见得一定比 lidarloc 快。
            LOG(WARNING) << "PGOFrame (frame_id " << frame->frame_id_ << ") Interpolate on lidarOdom Failed!";
            LOG(WARNING) << "PGOFrame time: " << std::fixed << std::setprecision(18) << frame->timestamp_
                         << ", latest lidarOdom time: " << lidar_odom_pose_queue_.back().timestamp_;
            return false;
        }
    } else {
        return true;
    }

    // 正常不会到这里
    return false;
}

bool PGOImpl::AssignDRPoseIfNeeded(std::shared_ptr<PGOFrame> frame) {
    assert(frame != nullptr);
    SE3 interp_pose;
    Vec3d interp_vel_b;
    NavState best_match;
    bool dr_interp_done = false;

    // 无论如何，我们都要寻找最近的DR帧，用于判断PGO收到的DR消息的时延
    dr_interp_done = math::PoseInterp<NavState>(
        frame->timestamp_, dr_pose_queue_, [](const NavState& nav_state) { return nav_state.timestamp_; },
        [](const NavState& nav_state) { return nav_state.GetPose(); }, interp_pose, best_match);

    if (!frame->dr_valid_) {
        if (dr_interp_done) {
            interp_vel_b = best_match.GetRot().matrix().transpose() * best_match.GetVel();
            frame->dr_set_ = true;
            frame->dr_valid_ = true;
            frame->dr_pose_ = interp_pose;
            frame->dr_vel_b_ = interp_vel_b;
            static int counts = 0;
            ++counts;
            // LOG(INFO) << "PGO interp DR  success - " << counts << " times.";
            return true;
        } else {
            // DR理论上应该比 lidarloc 快。
            LOG(WARNING) << "PGOFrame (frame_id" << frame->frame_id_ << ") Interpolate on DR Failed!";
            return false;
        }
    } else {
        return true;
    }

    // 正常不会到这里
    return false;
}

void PGOImpl::UpdateLidarOdomStatusInFrame(NavState& lio_result, std::shared_ptr<PGOFrame> frame) {
    // 把LidarOdom信息更新到PGOFrame
    frame->lidar_odom_normalized_weight_ = lio_result.confidence_ * Vec6d::Ones();
    Eigen::Array3d trans_confidence = {1.0, 1.0, 1.0};
    Eigen::Array3d rot_confidence = {1.0, 1.0, 1.0};

    if ((trans_confidence < kLidarOdomTransDegenThres).any()) {
        frame->lidar_odom_trans_degenerated = true;
    }
    if ((rot_confidence < kLidarOdomRotDegenThres).any()) {
        frame->lidar_odom_rot_degenerated = true;
    }
}

void PGOImpl::RunOptimization() {
    // if (frames_.size() < kMinNumRequiredForOptimization) {
    //     LOG(INFO) << "Skip optimization because frame size is " << frames_.size();
    //     return;
    // }

    if (debug_) {
        LOG(INFO) << "Run pose graph optimization: ";
    }

    // build g2o problem
    BuildProblem();

    // solve problems
    optimizer_->InitializeOptimization();
    optimizer_->SetVerbose(options_.verbose_);
    optimizer_->Optimize(5);

    // 确定inlier和outliers
    // RemoveOutliers();

    // solve again
    // optimizer_->InitializeOptimization();
    // optimizer_->Optimize(5);

    // get results
    for (const auto& frame : frames_) {
        auto v = std::dynamic_pointer_cast<miao::VertexSE3>(optimizer_->GetVertex(frame->frame_id_));
        if (v == nullptr) {
            continue;
        }

        ++frame->opti_times_;
        frame->last_opti_pose_ = frame->opti_pose_;
        frame->opti_pose_ = SE3(v->Estimate().matrix());
    }
}

void PGOImpl::BuildProblem() {
    // Add Vertex
    AddVertex();

    // Add Factors
    if (is_in_map_) {
        AddLidarLocFactors();
    }

    AddLidarOdomFactors();
    AddPriorFactors();
}

void PGOImpl::CleanProblem() {
    optimizer_->Clear();
    vertices_.clear();
    lidar_loc_edges_.clear();
    lidar_odom_edges_.clear();
    dr_edges_.clear();
    prior_edges_.clear();
}

void PGOImpl::AddVertex() {
    auto v = std::make_shared<miao::VertexSE3>();
    v->SetId(current_frame_->frame_id_);
    v->SetEstimate(current_frame_->opti_pose_);

    optimizer_->AddVertex(v);

    /// NOTE: 在incremental模式下，optimizer可能会修改顶点的ID（需要代替过去的某个顶点），
    /// 这导致顶点ID与frame_id对应不上。这里我们将frame_id改成顶点ID
    current_frame_->frame_id_ = v->GetId();

    vertices_.emplace_back(v);
}

void PGOImpl::AddLidarLocFactors() {
    // // 不支持未设置的或者无效的lidarLoc约束。
    if (!current_frame_->lidar_loc_set_) {
        return;
    }

    /// 不管lidarLoc是否为valid，factor都会加，只是可能会判为outlier
    SE3 loc_obs_pose = current_frame_->lidar_loc_pose_;
    Mat6d loc_obs_cov = Mat6d::Zero();
    loc_obs_cov(0, 0) = lidar_loc_noise_[0] * lidar_loc_noise_[0];
    loc_obs_cov(1, 1) = lidar_loc_noise_[1] * lidar_loc_noise_[1];
    loc_obs_cov(2, 2) = lidar_loc_noise_[2] * lidar_loc_noise_[2];
    loc_obs_cov(3, 3) = lidar_loc_noise_[3] * lidar_loc_noise_[3];
    loc_obs_cov(4, 4) = lidar_loc_noise_[4] * lidar_loc_noise_[4];
    loc_obs_cov(5, 5) = lidar_loc_noise_[5] * lidar_loc_noise_[5];

    Mat6d loc_obs_info = loc_obs_cov.inverse();
    Vec6d loc_obs_weight = current_frame_->lidar_loc_normalized_weight_;
    loc_obs_info(0, 0) *= loc_obs_weight[0];
    loc_obs_info(1, 1) *= loc_obs_weight[1];
    loc_obs_info(2, 2) *= loc_obs_weight[2];
    loc_obs_info(3, 3) *= loc_obs_weight[3];
    loc_obs_info(4, 4) *= loc_obs_weight[4];
    loc_obs_info(5, 5) *= loc_obs_weight[5];

    auto e = std::make_shared<miao::EdgeSE3Prior>();
    e->SetVertex(0, optimizer_->GetVertex(current_frame_->frame_id_));
    e->SetMeasurement(loc_obs_pose);
    e->SetInformation(loc_obs_info);

    auto rk = std::make_shared<miao::RobustKernelHuber>();
    rk->SetDelta(options_.lidar_loc_outlier_th);
    e->SetRobustKernel(rk);

    optimizer_->AddEdge(e);
    lidar_loc_edges_.emplace_back(e);
}

void PGOImpl::AddLidarOdomFactors() {
    size_t num = lo_relative_constraints_num_;

    // 该循环负责在不直接相邻的帧之间也添加相对位姿约束，每一帧最多向后找‘num’帧；这个策略让graph更稳定
    for (auto iter = frames_.rbegin(); iter != frames_.rend(); ++iter) {
        auto frame = *iter;
        if (frame == current_frame_) {
            continue;
        }

        if ((current_frame_->frame_id_ - frame->frame_id_) >= num) {
            continue;
        }

        auto pre_key_frame = frame;
        auto cur_key_frame = current_frame_;
        if (!pre_key_frame->lidar_odom_valid_ || !cur_key_frame->lidar_odom_valid_) {
            continue;
        }

        SE3 lo_obs_pose = pre_key_frame->lidar_odom_pose_.inverse() * cur_key_frame->lidar_odom_pose_;

        Mat6d lo_obs_cov = Mat6d::Zero();
        const Vec6d lo_noise = lidar_odom_rel_noise_;
        lo_obs_cov(0, 0) = lo_noise[0] * lo_noise[0];
        lo_obs_cov(1, 1) = lo_noise[1] * lo_noise[1];
        lo_obs_cov(2, 2) = lo_noise[2] * lo_noise[2];
        lo_obs_cov(3, 3) = lo_noise[3] * lo_noise[3];
        lo_obs_cov(4, 4) = lo_noise[4] * lo_noise[4];
        lo_obs_cov(5, 5) = lo_noise[5] * lo_noise[5];

        // 取两帧中更小的权重作为实际权重 —— 取“木桶短板效应”原理
        Vec6d lo_obs_weight = lo_noise;

        // 添加边
        auto e = std::make_shared<miao::EdgeSE3>();
        auto v1 = optimizer_->GetVertex(pre_key_frame->frame_id_);
        auto v2 = optimizer_->GetVertex(cur_key_frame->frame_id_);

        if (v1 == nullptr || v2 == nullptr) {
            continue;
        }

        e->SetVertex(0, v1);
        e->SetVertex(1, v2);
        e->SetMeasurement(lo_obs_pose);

        Mat6d rel_obs_info = lo_obs_cov.inverse();
        e->SetInformation(rel_obs_info);

        /// 这个e仍然需要robust kernel
        auto rk = std::make_shared<miao::RobustKernelCauchy>();
        rk->SetDelta(options_.lidar_odom_ang_noise);
        e->SetRobustKernel(rk);

        optimizer_->AddEdge(e);
        lidar_odom_edges_.emplace_back(e);
    }
}

void PGOImpl::AddDRFactors() {
    size_t num = dr_relative_constraints_num_;
    for (auto iter = frames_.rbegin(); iter != frames_.rend(); ++iter) {
        auto frame = *iter;
        if (frame == current_frame_) {
            continue;
        }

        if ((current_frame_->frame_id_ - frame->frame_id_) >= num) {
            continue;
        }

        auto pre_key_frame = frame;
        auto cur_key_frame = current_frame_;

        if (!pre_key_frame->dr_valid_ || !cur_key_frame->dr_valid_) {
            continue;
        }

        SE3 dr_rel_pose = pre_key_frame->dr_pose_.inverse() * cur_key_frame->dr_pose_;
        Mat6d dr_obs_cov = Mat6d::Zero();
        const Vec6d dr_noise = dr_rel_noise_;
        dr_obs_cov(0, 0) = dr_noise[0] * dr_noise[0];
        dr_obs_cov(1, 1) = dr_noise[1] * dr_noise[1];
        dr_obs_cov(2, 2) = dr_noise[2] * dr_noise[2];
        dr_obs_cov(3, 3) = dr_noise[3] * dr_noise[3];
        dr_obs_cov(4, 4) = dr_noise[4] * dr_noise[4];
        dr_obs_cov(5, 5) = dr_noise[5] * dr_noise[5];

        // 添加边
        auto e = std::make_shared<miao::EdgeSE3>();
        auto v1 = optimizer_->GetVertex(pre_key_frame->frame_id_);
        auto v2 = optimizer_->GetVertex(cur_key_frame->frame_id_);

        if (v1 == nullptr || v2 == nullptr) {
            continue;
        }

        e->SetVertex(0, optimizer_->GetVertex(pre_key_frame->frame_id_));
        e->SetVertex(1, optimizer_->GetVertex(cur_key_frame->frame_id_));

        e->SetMeasurement(dr_rel_pose);
        Mat6d dr_obs_info = dr_obs_cov.inverse();  // DR无法给出weight，我们直接tune noise即可。
        e->SetInformation(dr_obs_info);

        optimizer_->AddEdge(e);
        dr_edges_.emplace_back(e);
    }
}

void PGOImpl::AddPriorFactors() {
    auto frame = current_frame_;

    if (frame->prior_set_ && frame->prior_valid_) {
        SE3 prior_obs_pose = frame->prior_pose_;

        auto e = std::make_shared<miao::EdgeSE3Prior>();
        e->SetVertex(0, optimizer_->GetVertex(frame->frame_id_));
        e->SetMeasurement(prior_obs_pose);

        // 有必要实现noise的融合吗，毕竟还有权重项可以tune
        // Vec6d prior_noise = rtk_fix_noise_;
        // Vec6d prior_noise = lidar_odom_rel_noise_;
        Vec6d prior_noise = lidar_loc_noise_;

        // 先验（边缘化）约束的cov应该参考lidarLoc和lidarOdom，是一个可信任的观测
        Mat6d prior_obs_cov = Mat6d::Zero();
        prior_obs_cov(0, 0) = prior_noise[0] * prior_noise[0];
        prior_obs_cov(1, 1) = prior_noise[1] * prior_noise[1];
        prior_obs_cov(2, 2) = prior_noise[2] * prior_noise[2];
        prior_obs_cov(3, 3) = prior_noise[3] * prior_noise[3];
        prior_obs_cov(4, 4) = prior_noise[4] * prior_noise[4];
        prior_obs_cov(5, 5) = prior_noise[5] * prior_noise[5];

        Mat6d prior_obs_info = prior_obs_cov.inverse();
        const Vec6d prior_obs_weight = frame->prior_normalized_weight_;
        prior_obs_info(0, 0) *= prior_obs_weight[0];
        prior_obs_info(1, 1) *= prior_obs_weight[1];
        prior_obs_info(2, 2) *= prior_obs_weight[2];
        prior_obs_info(3, 3) *= prior_obs_weight[3];
        prior_obs_info(4, 4) *= prior_obs_weight[4];
        prior_obs_info(5, 5) *= prior_obs_weight[5];
        e->SetInformation(prior_obs_info);

        // 个人觉得应该是不用加核函数的（像lidarOdom一样）
        // auto* rk = new g2o::RobustKernelHuber();
        // rk->setDelta(pgo_option::lidar_loc_outlier_th/*使用lidarLoc的核参数*/);
        // e->setRobustKernel(rk);
        // e->setParameterId(0, 0);

        optimizer_->AddEdge(e);
        prior_edges_.emplace_back(e);
    }
}

void PGOImpl::RemoveOutliers() {
    int cnt_outliers = 0;
    auto remove_outlier = [&cnt_outliers, this](miao::Edge* e) {
        e->ComputeError();
        if (e->Chi2() > e->GetRobustKernel()->Delta()) {
            e->SetLevel(1);
            cnt_outliers++;

            // set rtk outlier in frame
            auto frame = frames_by_id_.find(e->GetVertex(0)->GetId())->second;

            if (debug_) {
                LOG(INFO) << "frame " << frame->frame_id_ << " has outlier gps, chi2 = " << e->Chi2() << " > "
                          << e->GetRobustKernel()->Delta();
            }
        } else {
            auto frame = frames_by_id_.find(e->GetVertex(0)->GetId())->second;
            if (debug_) {
                LOG(INFO) << "frame " << frame->frame_id_ << " has inlier gps, chi2 = " << e->Chi2() << " < "
                          << e->GetRobustKernel()->Delta();
            }
            e->SetRobustKernel(nullptr);
        }
    };
}

void PGOImpl::UpdatePoseGraphState() {}

void PGOImpl::CollectOptimizationStatistics() {
    /// 打印必要信息
    if (debug_) {
        LOG(INFO) << std::string("LidarLoc       -- ") + print_info(lidar_loc_edges_, options_.lidar_loc_outlier_th);
        LOG(INFO) << std::string("LidarOdom      -- ") + print_info(lidar_odom_edges_);
        LOG(INFO) << std::string("DR             -- ") + print_info(dr_edges_);
        LOG(INFO) << std::string("Marginal Prior -- ") + print_info(prior_edges_);
    }

    // 把滑窗中最新帧的信息更新到result中，在需要时输出到外部
    if (frames_.size() < kMinNumRequiredForOptimization) {
        // 只有一个frame的时候，优化被跳过，各种chi2也无效
        UpdateFinalResultByLastFrame();
    } else {
        UpdateFinalResultByWindow();
    }
    if (output_func_) {
        output_func_(result_);
    }

    // 打印必要的误差信息(lidarOdom的error大于0为有效，否则invalid)
    static std::stringstream ss;
    std::string lo_error_vert = "invalid", lo_error_hori = "invalid";
    ss << std::fixed << std::setprecision(3);
    if (result_.lidar_odom_error_vert_ > 0) {
        ss.str("");
        ss << result_.lidar_odom_error_vert_;
        lo_error_vert = ss.str();
    }
    if (result_.lidar_odom_error_hori_ > 0) {
        ss.str("");
        ss << result_.lidar_odom_error_hori_;
        lo_error_hori = ss.str();
    }

    const double loc_err = std::sqrt(result_.lidar_loc_error_vert_ * result_.lidar_loc_error_vert_ +
                                     result_.lidar_loc_error_hori_ * result_.lidar_loc_error_hori_);

    const double lo_err = std::sqrt(result_.lidar_odom_error_vert_ * result_.lidar_odom_error_vert_ +
                                    result_.lidar_odom_error_hori_ * result_.lidar_odom_error_hori_);
}

void PGOImpl::UpdateFinalResultByLastFrame() {
    auto& lf = frames_.back();

    PGOFrameToResult(lf, result_);

    // 近似计算lidarOdom与优化位姿的误差
    auto& ff = frames_.front();
    if (ff->lidar_odom_valid_ && lf->lidar_odom_valid_) {
        SE3 lo_pose_w = ff->opti_pose_ * ff->lidar_odom_pose_.inverse() * lf->lidar_odom_pose_;
        Vec3d lo_error = lo_pose_w.translation() - lf->opti_pose_.translation();
        Vec3d lo_err_body = lf->opti_pose_.so3().inverse() * lo_error;
        result_.lidar_odom_error_vert_ = fabs(lo_err_body[0]);
        result_.lidar_odom_error_hori_ = fabs(lo_err_body[1]);
    } else {
        result_.lidar_odom_error_vert_ = -999;
        result_.lidar_odom_error_hori_ = -999;
    }

    // 一些需要从PGOFrame以外获取的信息
    result_.valid_ = true;
    // result_.is_in_map_           = is_in_map_;
    // result_.absolute_pose_valid_ = lf->rtk_valid_;  // outlier也算valid
    // result_.rtk_status_ = gps_queue_.back().gps_status_;  // 发送给下游：rtk实时状态

    /// 确定整体定位状态
    // if (lf->rtk_valid_ && lf->lidar_loc_valid_ && lf->lidar_odom_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_ALL;
    // } else if (lf->rtk_valid_ && lf->lidar_loc_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_BOTH1;
    // } else if (lf->rtk_valid_ && lf->lidar_odom_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_BOTH2;
    // } else if (lf->lidar_loc_valid_ && lf->lidar_odom_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_LIDAR_FULLY;
    // } else if (lf->rtk_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_GNSS;
    // } else if (lf->lidar_loc_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_LIDAR_LOC;
    // } else if (lf->lidar_odom_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_LiDAR_ODOM;
    // } else if (lf->dr_valid_) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_DR;
    // } else {
    //   result_.status_ = common::GlobalPoseStatus::OTHER;
    // }

    SetPgoGraphVertexes();
}

void PGOImpl::UpdateFinalResultByWindow() {
    auto& lf = frames_.back();

    PGOFrameToResult(lf, result_);

    // 近似计算lidarOdom与优化位姿的误差
    auto& ff = frames_.front();
    if (ff->lidar_odom_valid_ && lf->lidar_odom_valid_) {
        SE3 lo_pose_w = ff->opti_pose_ * ff->lidar_odom_pose_.inverse() * lf->lidar_odom_pose_;
        Vec3d lo_error = lo_pose_w.translation() - lf->opti_pose_.translation();
        Vec3d lo_err_body = lf->opti_pose_.so3().inverse() * lo_error;
        result_.lidar_odom_error_vert_ = fabs(lo_err_body[0]);
        result_.lidar_odom_error_hori_ = fabs(lo_err_body[1]);
    } else {
        result_.lidar_odom_error_vert_ = -999;
        result_.lidar_odom_error_hori_ = -999;
    }

    // 一些需要从PGOFrame以外获取的信息
    result_.valid_ = true;
    // result_.is_in_map_           = is_in_map_;
    // result_.absolute_pose_valid_ = lf->rtk_valid_;  // outlier也算valid
    // result_.rtk_status_ = gps_queue_.back().gps_status_;  // 发送给下游：rtk实时状态

    // LOG(INFO) << "PGO assigned result with position [" << result_.pose_.translation().transpose() << "].";

    /// 确定各个source的定位状态
    bool following_gps = false;
    bool following_lidar_loc = false;
    bool following_lidar_odom = lf->lidar_odom_valid_;
    bool following_dr = lf->dr_valid_;
    // if (lf->lidar_odom_rot_degenerated && lf->lidar_odom_trans_degenerated) {
    //     following_lidar_odom = false;
    // }

    // 判定RTK状态【至多使用滑窗中最新的3帧】 & 保留chi2信息
    // for (const auto& edge : gps_edges_) {
    //   edge->computeError();
    //   frames_by_id_.find(edge->vertex(0)->id())->second->rtk_chi2_ = edge->chi2();
    // }
    // double min_rtk_chi2 = 999999.;
    // int used_frames     = 0;
    // for (auto it = gps_edges_.rbegin(); it != gps_edges_.rend(); ++it) {
    //   if ((*it)->chi2() < pgo_option::rtk_outlier_th) {
    //     following_gps = true;
    //   }
    //   if ((*it)->chi2() < min_rtk_chi2) {
    //     min_rtk_chi2 = (*it)->chi2();
    //   }
    //   ++used_frames;
    //   if (used_frames >= 3) {
    //     break;
    //   }
    // }
    // // LOG(INFO) << "------- recent 5 minimum rtk chi2 is " << min_rtk_chi2;
    // if (!lf->rtk_valid_) {
    //   // RTK 设置了但是RTK状态位无效，所以自动是outlier
    //   result_.rtk_loc_inlier_ = false;
    // } else {
    //   // RTK 给进来为有效，要看chi2是否满足
    //   if (lf->rtk_chi2_ < pgo_option::rtk_outlier_th) {
    //     result_.rtk_loc_inlier_ = true;
    //   } else {
    //     result_.rtk_loc_inlier_ = false;
    //     LOG(WARNING) << "rtk is outlier, chi2 = " << lf->rtk_chi2_ << ", > " << pgo_option::rtk_outlier_th;
    //   }
    // }

    // 判定lidarLoc状态 & 保留chi2信息
    // for (const auto& edge : lidar_loc_edges_) {
    //     edge->ComputeError();
    //     auto frame = frames_by_id_.find(edge->GetVertex(0)->GetId());
    //     if (edge->Chi2() < options_.lidar_loc_outlier_th && frame->second->lidar_loc_valid_) {
    //         following_lidar_loc = true;
    //     }

    //     frames_by_id_.find(edge->GetVertex(0)->GetId())->second->lidar_loc_chi2_ = edge->Chi2();
    // }
    // if (lf->lidar_loc_set_) {  // lidarKLoc有效性仅看chi2，和输入时是否有效无关
    //     // if (lf->lidar_loc_set_ && lf->lidar_loc_valid_) {
    //     result_.lidar_loc_inlier_ = lf->lidar_loc_chi2_ < options_.lidar_loc_outlier_th;
    //     if (!result_.lidar_loc_inlier_) {
    //         LOG(WARNING) << "lidar loc is outlier, chi2 = " << lf->lidar_loc_chi2_ << ", > "
    //                      << options_.lidar_loc_outlier_th;
    //     }
    // }

    // 给出全局定位状态
    // if (following_lidar_odom && following_lidar_loc && following_gps) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_ALL;
    // } else if (following_lidar_loc && following_gps) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_BOTH1;
    // } else if (following_lidar_odom && following_gps) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_BOTH2;
    // } else if (following_lidar_loc && following_lidar_odom) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_LIDAR_FULLY;
    // } else if (following_gps) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_GNSS;
    // } else if (following_lidar_loc) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_LIDAR_LOC;
    // } else if (following_lidar_odom) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_LiDAR_ODOM;
    // } else if (following_dr) {
    //   result_.status_ = common::GlobalPoseStatus::FOLLOWING_DR;
    // } else {
    //   result_.status_ = common::GlobalPoseStatus::OTHER;
    // }

    // LOG(INFO) << "PGO assigned result with global status [" << common::StatusToString(result_.status_) << "].";

    // // 判定是否需要尝试从RTK pose来搜索激光定位
    // /// 如果最近的rtk有效 (且与优化pose有明显差异)，无论如何都可以尝试一下（试试又不会怀孕）
    // if (lf->rtk_valid_ &&
    //     (lf->rtk_pose_.translation().head<2>() - lf->lidar_loc_pose_.translation().head<2>()).norm() > 0.2) {
    //     LOG(INFO) << "should try rtk pose: " << lf->rtk_pose_.translation().head<2>().transpose()
    //               << " and opti: " << lf->opti_pose_.translation().head<2>().transpose();
    //     should_try_rtk_pose_for_localization_ = true;
    // }

    // SetPgoGraphVertexes();
}

void PGOImpl::PGOFrameToResult(const PGOFramePtr& frame, LocalizationResult& result) {
    if ((result.timestamp_ - frame->timestamp_) > 0.3) {
        /// 激光定位与当前时刻相差太多，则放弃设置此结果
        return;
    }

    // 复制PGOFrame中的信息到Result中
    result.timestamp_ = frame->timestamp_;
    result.pose_ = frame->opti_pose_;

    result.lidar_loc_valid_ = frame->lidar_loc_valid_;
    result.lidar_loc_inlier_ = frame->lidar_loc_inlier_;
    result.lidar_loc_delta_t_ = frame->lidar_loc_delta_t_;
    result.confidence_ = frame->confidence_;
    Vec3d loc_error = frame->lidar_loc_pose_.translation() - result.pose_.translation();
    Vec3d loc_err_body = result.pose_.so3().inverse() * loc_error;
    result.lidar_loc_error_vert_ = fabs(loc_err_body[0]);
    result.lidar_loc_error_hori_ = fabs(loc_err_body[1]);

    // 用lidarOdom和DR一起给出相对观测
    result.rel_pose_set_ = frame->lidar_odom_set_;
    result.rel_pose_ = frame->lidar_odom_pose_;
    result.vel_b_ = frame->dr_vel_b_;
    result.lidar_odom_delta_t_ = frame->lidar_odom_delta_t_;
    result.dr_delta_t_ = frame->dr_delta_t_;

    /// 此时检查外推历史队列和融合定位队列
    // SE3 output_history_pose;
    // common::TimedPose match;
    // if (result.lidar_loc_valid_ &&
    //     common::math::PoseInterp<common::TimedPose>(
    //         result.timestamp_, output_pose_queue_, [](const common::TimedPose& p) { return p.time_; },
    //         [](const common::TimedPose& p) { return p.pose_; }, output_history_pose, match)) {
    //     /// 检查历史位姿和当前定位的差异。如果差异较少，则取一个平均以保证外推平滑性
    //     SE3 delta = result.pose_.inverse() * output_history_pose;
    //     if (delta.translation().norm() < 0.3 && delta.so3().log().norm() < 1.5 * M_PI / 180.0) {
    //         const double fusion_ratio = 0.7;  // 更偏向于历史外推？
    //         result.pose_ = result.pose_ * SE3::exp(fusion_ratio * delta.log());
    //     } else {
    //         const double fusion_ratio = 0.3;  // 更偏向于自身定位？
    //         result.pose_ = result.pose_ * SE3::exp(fusion_ratio * delta.log());
    //     }
    // }
}

void PGOImpl::SetPgoGraphVertexes() {
    if (frames_.empty() || current_frame_ == nullptr) {
        return;
    }
}

void PGOImpl::SlideWindowAdaptively() {
    // NOTE 由于这里的frames要和optimizer中的保持一致，所以只能移除最新的关键帧

    // 保持窗口有一定宽度，但关键帧间距也不要太长
    if (frames_.size() < options_.PGO_MAX_FRAMES) {
        return;
    }

    auto frame_to_remove = *(frames_.begin());
    auto frame_to_keep = *(++frames_.begin());
    Marginalize(frame_to_remove, frame_to_keep);
    frames_.erase(frames_.begin());
    frames_by_id_.erase(frame_to_remove->frame_id_);
    return;

    // 看是移除队头还是次新帧
    // int n = frames_.size();
    // auto last_frame = frames_[n - 1];
    // auto last_frame2 = frames_[n - 2];

    // double dp = (last_frame->opti_pose_.translation() - last_frame2->opti_pose_.translation()).norm();
    // double da = (last_frame->opti_pose_.so3().inverse() * last_frame2->opti_pose_.so3()).log().norm();

    // if (dp < options_.PGO_DISTANCE_TH_LAST_FRAMES && da < options_.PGO_ANGLE_TH_LAST_FRAMES) {
    //     // remove 次新帧
    //     auto frame_to_remove = frames_.back();
    //     frames_.pop_back();
    //     frames_by_id_.erase(frame_to_remove->frame_id_);
    // } else {
    //     assert(frames_.size() > 1);
    //     // remove 队头 -- 需要执行动态策略和边缘化
    //     auto frame_to_remove = *(frames_.begin());
    //     auto frame_to_keep = *(++frames_.begin());
    //     if (frame_to_remove->opti_times_ > 2) {
    //         SE3 pose_error = frame_to_remove->last_opti_pose_.inverse() * frame_to_remove->opti_pose_;
    //         const double delta_p = pose_error.translation().norm();
    //         const double delta_a = pose_error.so3().log().norm();
    //         bool is_converged =
    //             (delta_p < options_.pgo_frame_converge_pos_th) && (delta_a < options_.pgo_frame_converge_ang_th);
    //         if (is_converged) {
    //             // 队头帧已经收敛了，边缘化后丢掉它
    //             Marginalize(frame_to_remove, frame_to_keep);
    //             frames_.erase(frames_.begin());
    //             frames_by_id_.erase(frame_to_remove->frame_id_);
    //         }
    //     }

    //     if (frames_.size() > (options_.PGO_MAX_FRAMES + 5)) {
    //         // 最坏的情况，即使不收敛，我们也不能让滑窗过大
    //         auto frame_to_remove = *(frames_.begin());
    //         auto frame_to_keep = *(++frames_.begin());
    //         Marginalize(frame_to_remove, frame_to_keep);
    //         frames_.erase(frames_.begin());
    //         frames_by_id_.erase(frame_to_remove->frame_id_);
    //     }
    // }
}

void PGOImpl::Marginalize(const PGOFramePtr& frame_to_remove, const PGOFramePtr& frame_to_keep) {
    /// 严格来说，我们需要按照边缘化理论实现这套东西； /// TODO: @wgh 实现边缘化算法
    /// 但受限于时间，我们先极简处理一下，跑通流程要紧。

    frame_to_keep->prior_pose_ = frame_to_keep->opti_pose_;
    // frame_to_keep->prior_cov_;
    frame_to_keep->prior_normalized_weight_ = Vec6d::Ones();
    frame_to_keep->prior_set_ = true;
    frame_to_keep->prior_valid_ = true;
}

void PGOImpl::log_window_status(std::ostringstream& report) {
    report.str("");
    // report << std::fixed << std::setprecision(6);
    report << std::setprecision(6);

    report << "Pose Graph Vertices: ";
    for (auto& vertice : vertices_) {
        report << " ------ id=" << vertice->GetId() << " ------ | ";
    }
    report << "\n";

    report << "LidarLoc edges     : ";
    for (auto& edge : lidar_loc_edges_) {
        edge->ComputeError();
        report << " (id" << edge->GetVertex(0)->GetId() << ")chi2=" << edge->Chi2() << " | ";
    }
    report << "\n";

    report << "Prior edges        : ";
    for (auto& edge : prior_edges_) {
        edge->ComputeError();
        report << " (id" << edge->GetVertex(0)->GetId() << ")chi2=" << edge->Chi2() << " | ";
    }
    report << "\n";

    // 相对约束：仅打印异常边
    report << "LidarOdom edges    : ";
    for (auto& edge : lidar_odom_edges_) {
        edge->ComputeError();
        if (edge->Chi2() > 10) {
            report << " (id" << edge->GetVertex(0)->GetId() << "&" << edge->GetVertex(1)->GetId()
                   << ")chi2=" << edge->Chi2() << " | ";
        }
    }
    report << "\n";

    // 相对约束：仅打印异常边
    report << "DR edges           : ";
    for (auto& edge : dr_edges_) {
        edge->ComputeError();
        if (edge->Chi2() > 10) {
            report << " (id" << edge->GetVertex(0)->GetId() << "&" << edge->GetVertex(1)->GetId()
                   << ")chi2=" << edge->Chi2() << " | ";
        }
    }
    report << "\n";
}

}  // namespace lightning::loc