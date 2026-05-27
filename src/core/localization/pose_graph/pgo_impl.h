#pragma once

#include <boost/format.hpp>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "common/constant.h"
#include "common/eigen_types.h"
#include "common/nav_state.h"
#include "common/timed_pose.h"
#include "core/graph/optimizer.h"
#include "core/localization/localization_result.h"
#include "core/types/edge_se3.h"
#include "core/types/edge_se3_prior.h"

namespace lightning::loc {

using FrameId = unsigned long;

/**
 * 保存一个优化单元需要的所有观测信息
 */
struct PGOFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PGOFrame() = default;
    double timestamp_ = 0;
    FrameId frame_id_ = 0;

    // 优化后的位姿
    SE3 opti_pose_;       // 最新的优化位姿
    SE3 last_opti_pose_;  // 上一轮优化位姿，用于判断是否收敛
    int opti_times_ = 0;  // 当前帧被优化的次数

    // GNSS/RTK观测【插值量】 | 提供绝对约束
    // bool rtk_set_    = false;          // RTK是否已经设置（可以已设置但状态位无效）
    // bool rtk_valid_  = false;          // RTK在[状态位]角度来看是否有效
    // bool rtk_inlier_ = true;           // RTK在PGO看来是否为inlier
    // SE3 rtk_pose_;                     // 插值得到的RTK pose（roll&pitch为零，z为零）
    // double rtk_delta_t_          = 0;  // 插值时，bestmatch相对于上一帧rtk消息的时延
    // double rtk_interp_time_error = 0;  // 当前帧在rtk队列上做插值时的时间误差
    // common::UTMCoordinate rtk_utm_;    // 与PGO帧最接近的原始RTK观测
    // double rtk_chi2_ = 0.0;            // RTK卡方误差

    // 激光定位观测 | 提供绝对约束
    bool lidar_loc_set_ = false;                         // lidarLoc是否已经设置（PGO由lodarLoc触发，正常是有效的）
    bool lidar_loc_valid_ = false;                       // lidarLoc是否有效（只要设置了就有效）
    bool lidar_loc_inlier_ = true;                       // lidarLoc在PGO看来是否为inlier
    SE3 lidar_loc_pose_;                                 // lidarLoc定位观测
    double lidar_loc_delta_t_ = 0;                       // 插值时，bestmatch相对于上一帧lidarLoc消息的时延
    double confidence_ = 0;                              // lidarLoc给出的原始置信度
    Vec6d lidar_loc_normalized_weight_ = Vec6d::Ones();  // 归一化权重，表征退化情况，位于(0, 1]区间，平移在前旋转在后
    bool lidar_loc_rot_degenerated = false;              // 根据归一化权重来决定
    bool lidar_loc_trans_degenerated = false;            // 根据归一化权重来决定
    double lidar_loc_chi2_ = 0.0;                        // lidarLoc观测的卡方误差

    // 激光里程计观测 | 提供帧间相对约束
    bool lidar_odom_set_ = false;                         // LO是否已经设置（由lidarLoc设置或者PGO中插值）
    bool lidar_odom_valid_ = false;                       // LO是否有效（只要设置了就是有效）
    SE3 lidar_odom_pose_;                                 // LO自系位姿观测
    double lidar_odom_delta_t_ = 0;                       // 插值时，bestmatch相对于上一帧lidarOdom消息的时延
    Vec6d lidar_odom_normalized_weight_ = Vec6d::Ones();  // 归一化权重，表征退化情况，位于(0, 1]区间，平移在前旋转在后
    bool lidar_odom_rot_degenerated = false;              // 根据归一化权重来决定（无论是否退化都会添加约束）
    bool lidar_odom_trans_degenerated = false;            // 根据归一化权重来决定
    Vec3d lidar_odom_vel_ = Vec3d::Zero();                // 删除这个量？
    double lidar_odom_chi2_ = 0.;                         // LO观测的卡方误差
    bool lo_reliable_ = true;

    // DR观测 | 提供帧间相对约束
    bool dr_set_ = false;             // DR是否已经设置（正常都是设置了的）
    bool dr_valid_ = false;           // DR是否有效（正常跑是有效的，但边界上不一定）
    SE3 dr_pose_;                     // DR自系位姿观测
    Vec3d dr_vel_b_ = Vec3d::Zero();  // 来自DR的vel,车体系下的
    double dr_delta_t_ = 0;           // 插值时，bestmatch相对于上一帧DR消息的时延

    // 先验“观测”（对滑出帧【边缘化】之后导出的虚拟观测） | 提供绝对约束
    bool prior_set_ = false;                         // 边缘化约束是否设置（仅滑窗的第一帧会设置）
    bool prior_valid_ = false;                       // 边缘化约束是否有效（只要设置了就是有效）
    SE3 prior_pose_;                                 // 边缘化约束提供的位姿
    Mat6d prior_cov_ = Mat6d::Identity();            // 边缘化约束的协方差（来自边缘化计算）
    Vec6d prior_normalized_weight_ = Vec6d::Ones();  // 归一化权重，位于(0, 1]区间，平移在前旋转在后
    double prior_chi2_ = 0.;                         // 边缘化“观测”的卡方误差
};

using PGOFramePtr = std::shared_ptr<PGOFrame>;

/**
 * Pose Graph Implementation
 * 基于 Pose Graph 的多源信息融合：实现。
 */
struct PGOImpl {
    using UL = std::unique_lock<std::mutex>;

    struct Options {
        Options() {}
        bool verbose_ = false;                                                  // 调试打印
        static constexpr int PGO_MAX_FRAMES = 5;                               // PGO所持的最大帧数
        static constexpr int PGO_MAX_SIZE_OF_RELATIVE_POSE_QUEUE = 10000;      // PGO 相对定位队列最大长度
        static constexpr int PGO_MAX_SIZE_OF_RTK_POSE_QUEUE = 200;             // PGO RTK观测队列最大长度
        static constexpr double PGO_DISTANCE_TH_LAST_FRAMES = 2.5;             // PGO 滑窗时，最近两帧的最小距离
        static constexpr double PGO_ANGLE_TH_LAST_FRAMES = 10 * M_PI / 360.0;  // PGO 滑窗时，最近两帧的最小角度

        double lidar_loc_pos_noise = 0.3;                             // lidar定位位置噪声 // 0.3
        double lidar_loc_ang_noise = 1.0 * constant::kDEG2RAD;        // lidar定位角度噪声
        double lidar_loc_outlier_th = 30.0;                           // lidar定位异常阈值
        double lidar_odom_pos_noise = 0.3;                            // LidarOdom相对定位位置噪声
        double lidar_odom_ang_noise = 1.0 * constant::kDEG2RAD;       // LidarOdom相对定位角度噪声
        double lidar_odom_outlier_th = 0.01;                          // LidarOdom异常值检测
        double dr_pos_noise = 1.0;                                    // DR相对定位位置噪声 // 0.05
        double dr_ang_noise = 0.5 * constant::kDEG2RAD;               // DR相对定位角度噪声
        double dr_pos_noise_ratio = 1.0;                              // DR位置噪声倍率
        double pgo_frame_converge_pos_th = 0.05;                      // PGO帧位置收敛阈值
        double pgo_frame_converge_ang_th = 1.0 * constant::kDEG2RAD;  // PGO帧姿态收敛阈值
        double pgo_smooth_factor = 0.01;                              // PGO帧平滑因子
    };

    PGOImpl(Options options = {});
    ~PGOImpl() {}

    /// 总逻辑
    void AddPGOFrame(std::shared_ptr<PGOFrame> frame);

    /// 为frame分配相对定位pose，如果已经有了就什么也不做。
    /// 注意有 lidarOdom pose 和 DR pose 两个；要求至少有一个，否则返回 false。
    // bool AssignRelativePoseIfNeeded(std::shared_ptr<PGOFrame> frame);
    bool AssignLidarOdomPoseIfNeeded(std::shared_ptr<PGOFrame> frame);
    bool AssignDRPoseIfNeeded(std::shared_ptr<PGOFrame> frame);

    void UpdateLidarOdomStatusInFrame(NavState& lio_result, std::shared_ptr<PGOFrame> frame);

    /// 执行优化的逻辑
    void RunOptimization();

    // 建立g2o优化问题
    void BuildProblem();

    // 清空优化问题
    void CleanProblem();

    // 添加顶点
    void AddVertex();

    // 添加lidar定位约束
    void AddLidarLocFactors();

    // 添加帧间相对约束，我们更愿意相信LidarOdom，如果LO退化才会用DR
    void AddLidarOdomFactors();

    void AddDRFactors();

    // 如果滑窗第一帧存在边缘化约束，添加之
    void AddPriorFactors();

    // 剔除异常值：假设所有信息源都可能有异常值？目前仅实现了对RTK判outlier
    void RemoveOutliers();

    // 获取优化详细信息
    void CollectOptimizationStatistics();

    /// 是否打开调试输出
    void SetDebug(bool debug) { debug_ = debug; }

    /// 清空并重置
    bool Reset();

   private:
    /// 更新系统状态
    void UpdatePoseGraphState();

    /// 保存结果
    void UpdateFinalResultByLastFrame();
    void UpdateFinalResultByWindow();
    inline void PGOFrameToResult(const PGOFramePtr& frame, LocalizationResult& result);
    inline void SetPgoGraphVertexes();

    /// 自适应调整窗口大小：如果早期帧的优化结果已经充分收敛，删除一部分；否则，适当多保留一些
    void SlideWindowAdaptively();

    /// 边缘化处理：把即将被滑出窗口的PGOFrame的约束转化为下一个PGOFrame的先验约束
    void Marginalize(const PGOFramePtr& frame_to_remove, const PGOFramePtr& frame_to_keep);

    void log_window_status(std::ostringstream& report);

   public:
    Options options_;

    // 全局数据锁
    std::mutex data_mutex_;
    const size_t kMinNumRequiredForOptimization = 2;
    const double kLidarOdomTransDegenThres = 0.3;
    const double kLidarOdomRotDegenThres = 0.3;

    /// data
    std::vector<std::shared_ptr<PGOFrame>> frames_;              // PGO的帧队列
    std::shared_ptr<PGOFrame> last_frame_ = nullptr;             // 上一帧，用于处理延时
    std::map<FrameId, std::shared_ptr<PGOFrame>> frames_by_id_;  // PGO帧按照Id进行索引
    std::shared_ptr<PGOFrame> current_frame_ = nullptr;          // 记录当前PGO帧

    std::deque<NavState> dr_pose_queue_;          // DR位姿观测队列
    std::deque<NavState> lidar_odom_pose_queue_;  // LidarOdom观测队列
    std::deque<TimedPose> lidar_loc_pose_queue_;  // LidarLoc观测队列
    std::deque<TimedPose> output_pose_queue_;     // 输出位姿队列

    // internal variables
    double last_gps_time_ = 0;          // 上一个gps时间戳,用于判断重复/回流数据
    FrameId accumulated_frame_id_ = 0;  // 每次自加1，作为新进PGOFrame的Id

    /// the miao optimizer
    std::shared_ptr<miao::Optimizer> optimizer_ = nullptr;

    /// miao 里的指针接口是shared_ptr
    std::vector<std::shared_ptr<miao::VertexSE3>> vertices_;            // pose graph顶点
    std::vector<std::shared_ptr<miao::EdgeSE3>> lidar_odom_edges_;      // 相对运动观测（来自LidarOdom）
    std::vector<std::shared_ptr<miao::EdgeSE3>> dr_edges_;              // 相对运动观测（来自DR）
    std::vector<std::shared_ptr<miao::EdgeSE3Prior>> lidar_loc_edges_;  // 激光定位边
    std::vector<std::shared_ptr<miao::EdgeSE3Prior>> prior_edges_;      // 边缘化约束

    // output variables
    LocalizationResult result_;  // pgo融合定位结果

    // output call back
    std::function<void(const LocalizationResult& output_result)> output_func_;

    /// params
    Vec6d rtk_fix_noise_ = Vec6d::Zero();         // 固定解RTK噪声
    Vec6d rtk_other_noise_ = Vec6d::Zero();       // 其他解RTK噪声
    Vec6d lidar_loc_noise_ = Vec6d::Zero();       // lidarLoc噪声
    Vec6d lidar_odom_rel_noise_ = Vec6d::Zero();  // lidarOdom噪声
    Vec6d dr_rel_noise_ = Vec6d::Zero();          // DR噪声
    int lo_relative_constraints_num_ = 5;         // LidarOdom相对位姿提供几个连续观测
    int dr_relative_constraints_num_ = 5;         // DR相对位姿提供几个连续观测

    bool debug_ = true;  // debug模式将打印更多信息

    int invalid_lidar_num_ = 0;  // 未启用
    int valid_lidar_num_ = 0;    // 未启用
    bool is_in_map_ = false;     // 未启用
    bool gps2lidar_ = false;     // 未启用
    bool lidar2gps_ = false;     // 未启用

    /// 判定Lidar odom是否有效
    bool lidar_odom_valid_ = true;
    bool lidar_odom_conflict_with_dr_ = false;  // LO和DR是否有冲突
    int lidar_odom_conflict_with_dr_cnt_ = 0;
    int lidar_odom_valid_cnt_ = 0;  // 需要缓冲一个计时之后才算有效

    std::ostringstream oss;  // log相关
};

}  // namespace lightning::loc
