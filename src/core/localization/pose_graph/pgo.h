#pragma once

#include "common/eigen_types.h"
#include "common/nav_state.h"
#include "core/localization/localization_result.h"

#include "pgo_impl.h"
#include "pose_extrapolator.h"
#include "smoother.h"

namespace lightning::loc {

/**
 * Pose Graph Optimization
 * 基于 Pose Graph 的多源信息融合。
 *
 * 输入：
 * -- DR，需要降低对此的依赖（轮速失效case较多），兼容没有DR
 * -- LiDAR Odom，频率可能不高，比如5Hz；lidarOdom应该提供自己的hessian矩阵的PCA分析，提供退化情况参考
 * -- LiDAR Loc，频率可能也不高，可动态配置
 *
 * 输出：
 * -- 融合定位结果，与IMU同频率发布
 * -- 判断各个信息源的outlier，并剔除之
 * -- 盲走一段时间误差控制在容许范围内
 *
 * 调用：
 * -- 外部可以调用所有接口塞数据和获取结果，是多线程安全的；
 * -- 但内部只有一个线程，不保证一定不阻塞，外部应当handle这种情况。
 */
class PGO {
   public:
    PGO();
    ~PGO();

    /// 向外输出全局定位结果
    using GlobalOutputHandleFunction = std::function<void(const LocalizationResult& output_result)>;
    void SetGlobalOutputHandleFunction(GlobalOutputHandleFunction handle);
    void SetHighFrequencyGlobalOutputHandleFunction(GlobalOutputHandleFunction handle);

    /// 处理lidarOdom信息
    bool ProcessDR(const NavState& dr_result);

    /// 处理lidarOdom信息
    bool ProcessLidarOdom(const NavState& lio_result);

    /// 接收激光定位信息（触发PGO优化）
    bool ProcessLidarLoc(const LocalizationResult& loc_result);

    /// 处理外部组装好的一个PGO frame，将触发优化（仅在单测时外部直接调用）
    bool ProcessPGOFrame(std::shared_ptr<PGOFrame> frame);

    std::shared_ptr<PGOFrame> GetCurrentPGOFrame() const;

    // 高频查询定位，使用DR做递推，多线程安全
    bool GetCurrentLocalization() const;

    /// 重置PGO
    bool Reset();

    /// 向外发布位姿
    void PubResult();

    /// debug stuffs
    void SetDebug(bool debug = true);
    void LogWindowState();

   public:
    bool localization_unusual_tag_ = false;  // 定位失效标志位
    bool imu_interruption_tag_ = false;      // imu断流标志位

   private:
    inline bool RelativePoseQueueEmpty() {
        return (impl_->dr_pose_queue_.empty() && impl_->lidar_odom_pose_queue_.empty());
    }

    /**
     * 为当前定位结果尽可能实时地外推位姿，考虑LO和DR两个信息源
     * 在当前激光定位的基础之上，先用LO进行外推，再用DR进行外推
     *
     * 同时和自身外推做校验，防止DR的异常
     *
     * @param @in/out output_result 当前融合定位结果（激光定位时间）
     * @return
     */
    bool ExtrapolateLocResult(LocalizationResult& output_result);

    std::unique_ptr<PGOImpl> impl_;

    // 在本层放一个extrapolator，被三个地方访问：DR，lidarOdom，LidarLoc，RTK
    std::unique_ptr<PoseExtrapolator> pose_extrapolator_;
    std::function<void(const LocalizationResult& output_result)> high_freq_output_func_;
    LocalizationResult high_freq_result_;
    LocalizationResult parking_result_;

    std::shared_ptr<PoseSmoother> smoother_;

    float imu_interruption_time_thd_ = 1.0;  // imu数据断流时间阈值 s
    float lidar_loc_score_thd_ = 0.5;        // 激光定位分值阈值
    int localization_unusual_thd_ = 10;      // 激光定位失效次数阈值
    int localization_unusual_count_ = 0;     // 定位异常次数
    double last_lidar_loc_time_ = 0.;        // 上次激光定位时间戳
    bool is_parking_ = false;
};

}  // namespace lightning::loc
