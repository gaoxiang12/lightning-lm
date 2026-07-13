//
// Created by MiMo on 25-7-10.
//

#ifndef LIGHTNING_LIO_FRONTEND_H
#define LIGHTNING_LIO_FRONTEND_H

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "common/measure_group.h"
#include "core/frontend/pointcloud_preprocess.h"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace lightning {

namespace ui {
class PangolinWindow;
}

/**
 * LIO 前端抽象基类
 * 统一接口: 预处理 → IMU处理 → Run → 获取结果
 *
 * 设计原则:
 * - 不同里程计返回自己的原生状态，不强制统一
 * - 后端只使用 SE3 pose + CloudPtr
 * - 各前端独立演进，互不影响
 *
 * 公共功能 (基类实现):
 * - ProcessPointCloud2: 点云预处理 + 缓冲区管理
 * - ProcessIMU: IMU 缓冲区管理
 * - SyncPackages: lidar_end_time_ 计算 + IMU 收集
 */
class LIOFrontend {
public:
    virtual ~LIOFrontend() = default;

    /// 初始化
    virtual bool Init(const std::string& config_yaml) = 0;

    /// 主循环: 一次完整的 LIO 处理
    virtual bool Run() = 0;

    // ====== 点云输入 (基类实现公共预处理+缓冲) ======

    /// 处理 ROS2 PointCloud2 (预处理 + 推入缓冲区)
    void ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg);

    /// 处理 Livox CustomMsg (预处理 + 推入缓冲区)
    void ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg);

    /// 处理已预处理的点云 (直接推入缓冲区)
    void ProcessPointCloud2(CloudPtr cloud);

    /// 处理 IMU (推入缓冲区，子类可重写以添加实时预测)
    virtual void ProcessIMU(const IMUPtr& msg_in);

    // ====== 结果获取 (统一接口) ======

    /// 获取 LIO 位姿
    virtual SE3 GetPose() const = 0;

    /// 获取去畸变点云
    virtual CloudPtr GetScanUndist() const = 0;

    /// 获取降采样后的点云 (世界坐标系)
    virtual CloudPtr GetScanDownWorld() const = 0;

    /// 获取关键帧
    virtual Keyframe::Ptr GetKeyframe() const { return nullptr; }

    /// 获取所有关键帧
    virtual std::vector<Keyframe::Ptr> GetAllKeyframes() { return {}; }

    // ====== 可选功能 ======

    /// 设置 UI
    virtual void SetUI(std::shared_ptr<ui::PangolinWindow> ui) {}

    /// 保存地图
    virtual void SaveMap() {}

    /// 获取全局地图
    virtual CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1) {
        return nullptr;
    }

    /// 打印外参
    virtual void PrintExtrinsic() {}

    /// 外参相关 (可选，子类按需实现)
    virtual bool IsExtrinsicEstEnabled() const { return false; }
    virtual Vec3d GetExtrinsicT() const { return Vec3d::Zero(); }
    virtual Mat3d GetExtrinsicR() const { return Mat3d::Identity(); }

protected:
    // ====== 公共缓冲区管理 (子类在 Run/SyncPackages 中使用) ======

    /// 从缓冲区同步数据，计算 lidar_end_time_，收集 IMU
    /// 返回 true 表示数据就绪
    bool SyncPackages();

    /// 获取预处理器 (子类 Init 中设置参数)
    std::shared_ptr<PointCloudPreprocess>& Preprocessor() { return preprocess_; }

    // ====== 公共成员变量 ======
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;

    // 数据缓冲
    std::deque<IMUPtr> imu_buffer_;
    std::deque<CloudPtr> lidar_buffer_;
    std::deque<double> time_buffer_;
    std::mutex mtx_buffer_;

    // 同步状态
    bool lidar_pushed_ = false;
    double last_timestamp_imu_ = -1.0;
    double lidar_end_time_ = 0;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;

    // 同步结果（SyncPackages 填充，子类在 Run 中读取）
    std::deque<IMUPtr> synced_imu_;

    // 输出
    CloudPtr scan_undistort_{new PointCloudType()};

    // 调试开关
    bool debug_ = false;

    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;
};

}  // namespace lightning

#endif  // LIGHTNING_LIO_FRONTEND_H
