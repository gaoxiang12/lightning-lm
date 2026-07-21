//
// Created by xiang on 25-4-21.
//

#ifndef LIGHTNING_LOOP_CLOSING_H
#define LIGHTNING_LOOP_CLOSING_H

#include "common/keyframe.h"
#include "common/loop_candidate.h"
#include "core/loop_closing/loop_detector.h"
#include "core/loop_closing/loop_pose_estimator.h"
#include "utils/async_message_process.h"

#include "core/graph/optimizer.h"
#include "core/types/edge_se3.h"

#include <memory>

namespace lightning {

/**
 * 回环检测编排器
 * 通过策略接口组合不同的检测和位姿估计方法
 * 图优化逻辑保持不变
 */
class LoopClosing {
   public:
    struct Options {
        Options() {}

        bool verbose_ = true;       // 输出调试信息
        bool online_mode_ = false;  // 切换离线-在线模式

        // 策略类型（从 YAML 读取）
        std::string detector_type_ = "spatial";
        std::string pose_estimator_type_ = "ndt";

        /// 图优化权重
        double motion_trans_noise_ = 0.1;               // 位移权重
        double motion_rot_noise_ = 3.0 * M_PI / 180.0;  // 旋转权重

        double loop_trans_noise_ = 0.2;               // 位移权重
        double loop_rot_noise_ = 3.0 * M_PI / 180.0;  // 旋转权重

        double rk_loop_th_ = 5.2 / 5;  // 回环的RK阈值

        bool with_height_ = true;
        double height_noise_ = 0.1;

        double ndt_score_th_ = 1.0;  // NDT 分数阈值
    };

    LoopClosing(Options options = Options()) { options_ = options; }
    ~LoopClosing();

    bool Init(const std::string yaml_path);

    /// 向回环中添加一个关键帧
    void AddKF(Keyframe::Ptr kf);

    /// 如果检测到新地回环并发生了优化，则调用回调
    using LoopClosedCallback = std::function<void()>;
    void SetLoopClosedCB(LoopClosedCallback cb) { loop_cb_ = cb; }

    /// 获取当前回环边（用于 UI 可视化）
    std::vector<std::pair<SE3, SE3>> GetLoopEdges() const;

   protected:
    void HandleKF(Keyframe::Ptr kf);

    /// 优化位姿
    void PoseOptimization();

    Options options_;

    // 策略组件（多态）
    std::unique_ptr<LoopDetector> detector_;
    std::unique_ptr<LoopPoseEstimator> pose_estimator_;

    Keyframe::Ptr last_kf_ = nullptr;
    Keyframe::Ptr cur_kf_ = nullptr;
    std::vector<Keyframe::Ptr> all_keyframes_;
    std::vector<LoopCandidate> candidates_;

    AsyncMessageProcess<Keyframe::Ptr> kf_thread_;

    std::shared_ptr<miao::Optimizer> optimizer_ = nullptr;

    Mat6d info_motion_ = Mat6d::Identity();  // 关键帧间的运动信息阵
    Mat6d info_loops_ = Mat6d::Identity();   // 回环帧的信息矩阵

    std::vector<std::shared_ptr<miao::VertexSE3>> kf_vert_;
    std::vector<std::shared_ptr<miao::EdgeSE3>> edge_loops_;

    LoopClosedCallback loop_cb_;
};

}  // namespace lightning

#endif  // LIGHTNING_LOOP_CLOSING_H
