#pragma once

#include <pcl/registration/icp.h>
#include <chrono>
#include <deque>
#include <iostream>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>

#include "common/nav_state.h"
#include "common/timed_pose.h"
#include "core/localization/localization_result.h"
#include "core/maps/tiled_map.h"

#include "pclomp/ndt_omp_impl.hpp"

namespace lightning::ui {
class PangolinWindow;
}

namespace lightning::loc {

/// 激光定位对外接口类
class LidarLoc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using UL = std::unique_lock<std::mutex>;

    /// 定位方法
    enum class LocMethod {
        NDT_OMP,  // OMP版本NDT
    };

    struct Options {
        Options() {}

        bool display_realtime_cloud_ = false;          // 是否显示实时点云
        bool debug_ = false;                           // 是否使用单测模式
        LocMethod match_method_ = LocMethod::NDT_OMP;  // 匹配方式
        bool try_self_extrap_ = false;                 // 是否尝试自己的外推pose
        bool with_height_ = true;                      // 建图期间是否带有高度约束？
        bool force_2d_ = true;                         // 强制在2D空间
        float min_init_confidence_ = 0.1;              // 初始化时要求的最小分值
        bool init_with_fp_ = true;                     // 是否使用功能点进行初始化
        bool enable_parking_static_ = false;           // 是否在静止时输出固定位置
        bool enable_icp_adjust_ = false;               // 是否使用icp调整ndt匹配结果提高定位精度

        /// 点云过滤
        // float filter_z_min_ = -1.0;
        float filter_z_max_ = 30.0;
        // float filter_intensity_min_ = 10.0;
        // float filter_intensity_max_ = 100.0;

        /// 地图配置
        TiledMap::Options map_option_ = TiledMap::Options();

        // 动态图层更新相关
        bool update_dynamic_cloud_ = true;     // 是否使用定位后的点云来更新动态图层
        double update_kf_dis_ = 5.0;           // 每隔多少米更新一次
        double update_kf_time_ = 10.0;         // 每隔多少时间更新一次
        double update_lidar_loc_score_ = 2.2;  // 更新时激光定位匹配分值阈值
        double lidar_loc_odom_th_ = 0.3;       // 激光两帧匹配结果与对应lidarodom结果差的阈值，超过则认为lidarodom异常

        double max_update_cache_dis_ = 30.0;  // 更新动态图层的缓冲距离
        std::string recover_pose_path_ = "./data/recover_pose.txt";
    };

    explicit LidarLoc(Options options = Options());
    virtual ~LidarLoc();

    /// 初始化
    bool Init(const std::string& config_path);

    /// 处理 lidar odom 消息
    bool ProcessLO(const NavState& state);

    /// 处理拼接后点云
    bool ProcessCloud(CloudPtr cloud_input);

    /// 处理DR状态
    bool ProcessDR(const NavState& state);

    /// 获取位姿
    NavState GetState();

    /// 设置当前定位状态标志位
    void SetInitRltState();

    /**
     * 尝试在另一个位置进行定位
     * @param input
     * @param pose
     * @return
     */
    bool TryOtherSolution(CloudPtr input, SE3& pose);

    /// 使用功能点初始化
    bool InitWithFP(CloudPtr input, const SE3& fp_pose);

    /// 更新全局地图
    bool UpdateGlobalMap();

    /// @brief 定位是否已经成功初始化
    bool LocInited();

    /// @brief 激光定位重置接口
    void ResetLastPose(const SE3& last_pose);

    /**
     * @brief 激光地图匹配
     * @param pose       预测位姿
     * @param confidence    得分
     * @param input 输入点云
     * @param output 输出点云
     * @param use_rough_res 是否使用粗分辨率
     * @return
     */
    bool Localize(SE3& pose, double& confidence, CloudPtr input, CloudPtr output, bool use_rough_res = false);

    /// 设置UI
    void SetUI(std::shared_ptr<ui::PangolinWindow> ui) { ui_ = ui; }

    /// 设置init pose
    void SetInitialPose(SE3 init_pose);

    /// 获取定位结果
    LocalizationResult GetLocalizationResult() {
        UL lock(result_mutex_);
        return localization_result_;
    }

    void Finish();

    /// 激光定位是否认为LO有效
    bool LidarLocThinkLOReliable() { return lo_reliable_; }

   private:
    // 内部函数  ==========================================================================
    /**
     * 对点云进行配准
     * @param input
     */
    void Align(const CloudPtr& input);

    /**
     * 寻找当前帧对应的LO相对位姿
     * @param timestamp
     * @return
     */
    bool AssignLOPose(double timestamp);

    /**
     * 寻找当前帧对应的DR相对位姿
     * @param timestamp
     * @return
     */
    bool AssignDRPose(double timestamp);

    /**
     * 检查车辆是否静止
     * @param timestamp
     * @return
     */
    bool CheckStatic(double timestamp);

    /**
     * 更新自身状态
     * @param input
     */
    void UpdateState(const CloudPtr& input);

    /**
     * 更新地图
     */
    void UpdateMapThread();

    /**
     * 使用网格搜索best yaw
     */
    bool YawSearch(SE3& pose, double& confidence, CloudPtr input, CloudPtr output);

    bool CheckLidarOdomValid(const SE3& current_pose_esti, double& delta_posi);

    // 成员变量  ==========================================================================
    Options options_;

    std::mutex match_mutex_;  // 锁定pcl_ndt指针

    using NDTType = pclomp::NormalDistributionsTransform<PointType, PointType>;
    NDTType::Ptr pcl_ndt_ = nullptr;
    NDTType::Ptr pcl_ndt_rough_ = nullptr;  // 粗分辨率

    using ICPType = pcl::IterativeClosestPoint<PointType, PointType>;
    ICPType::Ptr pcl_icp_ = nullptr;

    CloudPtr current_scan_ = nullptr;                   // 当前扫描
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;  // ui
    SE3 last_loc_pose_;

    std::mutex initial_pose_mutex_;  // 初始定位锁
    bool initial_pose_set_ = false;  // 定位是否被手动设置
    SE3 initial_pose_;               // 手动设置的初始位姿
    bool loc_inited_ = false;        // 定位是否初始化成功

    double current_timestamp_ = 0;  // 本次输入的时间戳
    double last_timestamp_ = 0;     // 上次输入的时间戳

    bool last_lo_pose_set_ = false;
    bool current_lo_pose_set_ = false;
    SE3 last_lo_pose_;     // 上一次相对位置，相对位置来自LO
    SE3 current_lo_pose_;  // 本次的LO相对位置

    bool last_dr_pose_set_ = false;
    bool current_dr_pose_set_ = false;
    SE3 last_dr_pose_;     // 上一次相对位置，相对位置来自DR
    SE3 current_dr_pose_;  // 本次的DR相对位置
    bool parking_ = false;

    double try_other_guess_trans_th_ = 0.3;               // 在初始估计相差多少时，尝试其他的解
    double try_other_guess_rot_th_ = 0.5 * M_PI / 180.0;  // 在初始估计相差多少时，尝试其他的解
    // double low_vel_th_ = 1.0;                             // 低速阈值m/s
    // double update_cache_dis_ = 0;                         // 动态图层的更新缓冲距离

    std::deque<double> ave_scores_;  // 近期匹配的分值情况

    Vec3d current_vel_b_ = Vec3d::Zero();  // 本次的车体系下的速度
    Vec3d current_vel_ = Vec3d::Zero();    // 本次dr的速度

    std::deque<TimedPose> lidar_loc_pose_queue_;  // lidar odom pose 轨迹

    bool lo_reliable_ = true;
    int lo_reliable_cnt_ = 0;

    SE3 last_abs_pose_;            // 上一次绝对定位，绝对定位来自于地图匹配得到的位姿
    TimedPose last_dyn_upd_pose_;  // 上次更新动态图层时使用的位姿
    SE3 current_abs_pose_;         // 本次的绝对定位
    bool last_abs_pose_set_ = false;
    double current_score_ = 1e5;  /// 设一个大分值，若定位一开始就匹配失败，则可以直接用GPS重置
    int match_fail_count_ = 0;
    int static_count_ = 0;

    int rtk_reset_cnt_ = 0;  // RTK重置计数

    std::mutex result_mutex_;
    LocalizationResult localization_result_;  // 输出结果

    // 相对运动观测队列
    std::mutex lo_pose_mutex_;
    std::deque<NavState> lo_pose_queue_;

    std::mutex dr_pose_mutex_;
    std::deque<NavState> dr_pose_queue_;

    /// 功能点初始化的记录
    std::vector<SE3> fp_init_fail_pose_vec_;
    double fp_last_tried_time_ = 0;

    bool update_map_quit_ = false;
    std::thread update_map_thread_;            // 地图更新
    std::shared_ptr<TiledMap> map_ = nullptr;  // 地图
    double map_height_ = 0;

    bool has_set_pose_ = false;  // 外部set_pose标志位，若存在则本次动态图层不落盘

    std::ofstream recover_pose_out_;
};

}  // namespace lightning::loc
