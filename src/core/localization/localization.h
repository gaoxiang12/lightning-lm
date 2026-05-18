#pragma once

#include <functional>
#include <memory>
#include <mutex>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "std_msgs/msg/int32.hpp"

#include "common/imu.h"
#include "core/lio/laser_mapping.h"
#include "core/localization/localization_result.h"
#include "core/system/async_message_process.h"

/// 预声明
namespace lightning {
namespace ui {
class PangolinWindow;
}

namespace loc {

class LidarLoc;
class PGO;
class DualLidarOnlineCalibration;
struct DualLidarCalibrationResult;

/**
 * 实时定位接口实现
 */
class Localization {
   public:
    enum class Mode {
        LOCALIZATION,
        DUAL_LIDAR_ONLINE_CALIBRATION,
        LOCALIZATION_WITH_DUAL_LIDAR_CALIB_MONITOR
    };

    struct Options {
        Options() {}

        bool online_mode_ = false;  // 在线模式还是离线模式
        bool with_ui_ = false;      // 是否带ui

        /// 参数
        SE3 T_body_lidar_;

        bool enable_lidar_odom_skip_ = false;  // 是否允许激光里程计跳帧
        int lidar_odom_skip_num_ = 1;          // 如果允许跳帧，跳多少帧
        bool enable_lidar_loc_skip_ = true;    // 是否允许激光定位跳帧
        bool enable_lidar_loc_rviz_ = false;   // 是否允许调试用rviz
        int lidar_loc_skip_num_ = 4;           // 如果允许跳帧，跳多少帧
        bool loc_on_kf_ = false;
        Mode mode_ = Mode::LOCALIZATION;
    };

    Localization(Options options = Options());
    ~Localization() = default;

    /**
     * 初始化，读配置参数
     * @param yaml_path
     * @param global_map_path
     * @param init_reloc_pose
     */
    bool Init(const std::string& yaml_path, const std::string& global_map_path);

    /// 处理lidar消息
    void ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr laser_msg);
    void ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr laser_msg);
    void ProcessDualLidarPointCloudPair(const sensor_msgs::msg::PointCloud2::SharedPtr& front_msg,
                                        const sensor_msgs::msg::PointCloud2::SharedPtr& rear_msg);

    /// 处理IMU消息
    void ProcessIMUMsg(IMUPtr imu);

    // void ProcessOdomMsg(const nav_msgs::msg::Odometry::SharedPtr odom_msg) override;

    /// 由外部设置pose，适用于手动重定位
    void SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);

    /// TODO: 其他初始化逻辑

    /// TODO: 处理odom消息

    /// 结束，保存临时地图
    void Finish();

    /// 异步处理函数
    void LidarOdomProcCloud(CloudPtr);
    void LidarLocProcCloud(CloudPtr);

    using TFCallback = std::function<void(const LocalizationResult& res)>;
    using LocStateCallback = std::function<void(const std_msgs::msg::Int32& state)>;
    using PointcloudBodyCallback = std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>;
    using PointcloudWorldCallback = std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>;
    using DualLidarCalibrationCallback = std::function<void(const DualLidarCalibrationResult& res)>;

    void SetTFCallback(TFCallback&& callback);
    void SetDualLidarCalibrationCallback(DualLidarCalibrationCallback&& callback);
    bool RunsLocalization() const { return options_.mode_ != Mode::DUAL_LIDAR_ONLINE_CALIBRATION; }
    bool RunsDualLidarCalibration() const { return dual_lidar_calib_ != nullptr; }

    // void SetPathCallback(std::function<void(const nav_msgs::msg::Path& path)>&& callback);
    // void SetPointcloudWorldCallback(std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>&& callback);
    // void SetPointcloudBodyCallback(std::function<void(const sensor_msgs::msg::PointCloud2& pointcloud)>&& callback);
    // void SetLocStateCallback(std::function<void(const std_msgs::msg::Int32& state)>&& callback);
    // void SetHealthDiagNormalCallback(interface::health_diag_normal_callback&& callback);

   private:
    /// 模块  ========================================================================================================
    std::mutex global_mutex_;  // 防止处理过程中被重复init
    Options options_;

    /// 预处理
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess

    /// 前端
    std::shared_ptr<LaserMapping> lio_ = nullptr;
    Keyframe::Ptr lio_kf_ = nullptr;

    // ui
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;

    // pose graph
    std::shared_ptr<PGO> pgo_ = nullptr;

    // lidar localization
    std::shared_ptr<LidarLoc> lidar_loc_;
    std::shared_ptr<DualLidarOnlineCalibration> dual_lidar_calib_;

    /// TODO async 处理
    sys::AsyncMessageProcess<CloudPtr> lidar_odom_proc_cloud_;  // lidar odom 处理点云
    sys::AsyncMessageProcess<CloudPtr> lidar_loc_proc_cloud_;   // lidar loc 处理点云

    /// 结果数据 =====================================================================================================
    LocalizationResult loc_result_;

    /// 框架相关
    TFCallback tf_callback_;
    LocStateCallback loc_state_callback_;
    PointcloudBodyCallback pointcloud_body_callback_;
    PointcloudWorldCallback pointcloud_world_callback_;

    /// 输入检查
    double last_imu_time_ = 0;
    double last_odom_time_ = 0;
    double last_cloud_time_ = 0;
};
}  // namespace loc

}  // namespace lightning
