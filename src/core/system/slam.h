#ifndef LIGHTNING_SLAM_H
#define LIGHTNING_SLAM_H

#include <atomic>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "lightning/srv/save_map.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning {

class LaserMapping;
class LioSamMapping;
class LoopClosing;

namespace ui {
class PangolinWindow;
}

namespace g2p5 {
class G2P5;
}

/**
 * SLAM 系统调用接口
 */
class SlamSystem {
   public:
    struct Options {
        bool online_mode_ = true;

        bool with_cc_ = true;
        bool with_gridmap_ = true;
        bool with_loop_closing_ = true;
        bool with_visualization_ = true;
        bool with_2dvisualization_ = true;

        bool step_on_kf_ = true;
    };

    using SaveMapService = srv::SaveMap;

    explicit SlamSystem(Options options);
    ~SlamSystem();

    /// 初始化
    bool Init(const std::string& yaml_path);

    /// 对外部交互接口
    /// 开始建图，输入地图名称
    void StartSLAM(std::string map_name);

    /// 保存地图，默认保存至./data/地图名/ 下方
    void SaveMap(const std::string& path = "");

    /// 处理IMU
    void ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu);
    void ProcessIMU(const lightning::IMUPtr& imu);

    /// 处理点云
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    /// 实时模式下的spin
    void Spin();

   private:
    /// ros端保存地图的实现
    void SaveMap(const SaveMapService::Request::SharedPtr request, SaveMapService::Response::SharedPtr response);
    void PublishKeyframeToBackends(const Keyframe::Ptr& kf);

    Options options_;
    std::atomic_bool running_ = false;
    bool use_lio_sam_ = false;

    rclcpp::Service<SaveMapService>::SharedPtr savemap_service_ = nullptr;

    std::string map_name_;  // 地图名
    
    std::shared_ptr<LioSamMapping> lio_sam_ = nullptr;
    std::shared_ptr<LaserMapping> lio_ = nullptr;       // lio 前端
    std::shared_ptr<LoopClosing> lc_ = nullptr;         // 回环检测
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;  // ui
    std::shared_ptr<g2p5::G2P5> g2p5_ = nullptr;        // 栅格地图

    Keyframe::Ptr cur_kf_ = nullptr;

    /// 实时模式下的ros2 node, subscribers
    rclcpp::Node::SharedPtr node_;
    std::string imu_topic_;
    std::string cloud_topic_;
    std::string livox_topic_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;
};
}  // namespace lightning

#endif  // LIGHTNING_SLAM_H
