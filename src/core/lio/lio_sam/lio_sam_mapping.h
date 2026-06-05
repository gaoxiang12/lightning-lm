#ifndef LIGHTNING_LIO_SAM_MAPPING_H
#define LIGHTNING_LIO_SAM_MAPPING_H

#include <deque>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <rclcpp/rclcpp.hpp>

#include "common/keyframe.h"
#include "common/imu.h"
#include "common/nav_state.h"
#include "common/options.h"
#include "core/lio/eskf.hpp"

class ImageProjection;
class FeatureExtraction;
class mapOptimization;

namespace lightning {

namespace ui {
class PangolinWindow;
}

class LioSamMapping {
   public:
    struct Options {
        Options() : is_in_slam_mode_(true),online_mode_(false), kf_dis_th_(2.0), kf_angle_th_(15.0 * M_PI / 180.0) {}

        bool is_in_slam_mode_;
        bool online_mode_;
        double kf_dis_th_;
        double kf_angle_th_;
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LioSamMapping();
    explicit LioSamMapping(Options options);
    ~LioSamMapping();

    bool Init(const std::string& config_yaml);
    bool Run();

    void ProcessIMU(const IMUPtr& imu);
    void ProcessPointCloud2(CloudPtr cloud);

    void SetUI(std::shared_ptr<ui::PangolinWindow> ui) { ui_ = std::move(ui); }

    Keyframe::Ptr GetKeyframe() const { return last_kf_; }
    std::vector<Keyframe::Ptr> GetAllKeyframes() const { return all_keyframes_; }
    NavState GetState() const { return state_; }
    // 0603 新增imu 外推
    NavState GetIMUState() const {
        std::lock_guard<std::mutex> lock(mtx_buffer_);
        if (!imu_dr_inited_) {
            NavState s;
            s.pose_is_ok_ = false;
            return s;
        }

        NavState s = kf_imu_.GetX();
        s.pose_is_ok_ = true;
        return s;
    }

    CloudPtr GetScanUndist() const { return scan_undistort_; }
    CloudPtr GetProjCloud() const { return scan_undistort_; }
    CloudPtr GetRecentCloud() const { return recent_cloud_; }
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1);

   private:
    struct SyncedPackage {
        sensor_msgs::msg::PointCloud2 cloud;
        std::vector<sensor_msgs::msg::Imu> imus;
        double lidar_begin_time = 0.0;
        double lidar_end_time = 0.0;
    };

    bool LoadParamsFromYAML(const std::string& yaml_path);
    bool SyncPackages();
    bool MakeLightningKeyframeIfNeeded();
    void SyncLightningKeyframePoses();

    Options options_;
    rclcpp::NodeOptions node_options_;
    bool owns_rclcpp_context_ = false;

    std::unique_ptr<::ImageProjection> image_projection_;
    std::unique_ptr<::FeatureExtraction> feature_extraction_;
    std::unique_ptr<::mapOptimization> map_optimization_;

    mutable std::mutex mtx_buffer_;
    std::deque<sensor_msgs::msg::PointCloud2> lidar_buffer_;
    std::deque<double> time_buffer_;
    std::deque<sensor_msgs::msg::Imu> imu_buffer_; //给 LIO-SAM 的 ImageProjection 去畸变用的 ROS IMU 消息

    //0603新增imu 预测
    // LIO-SAM 分支给 PGO 高频发布用的 IMU/DR 状态
    ESKF kf_imu_;
    ESKF::ProcessNoiseType imu_Q_ = ESKF::ProcessNoiseType::Zero();

    bool imu_dr_inited_ = false;
    bool imu_mean_ready_ = false;

    int imu_init_count_ = 0;
    int imu_init_min_count_ = 100;

    Vec3d imu_mean_acc_ = Vec3d::Zero();
    Vec3d imu_mean_gyr_ = Vec3d::Zero();

    double last_dr_imu_time_ = -1.0;
    double last_lio_anchor_time_ = -1.0;
    Vec3d last_lio_anchor_pos_ = Vec3d::Zero();
    //给 ESKF::Predict() 用的 lightning 原始 IMUPtr。
    std::deque<IMUPtr> imu_dr_buffer_;

    SyncedPackage measures_;
    CloudPtr scan_undistort_{new PointCloudType()};
    CloudPtr recent_cloud_{new PointCloudType()};
    NavState state_;

    double last_timestamp_imu_ = -1.0;
    double last_timestamp_lidar_ = -1.0;
    double lidar_begin_time_ = 0.0;
    double lidar_end_time_ = 0.0;
    double lidar_mean_scantime_ = 0.1;
    bool lidar_pushed_ = false;
    int scan_count_ = 0;
    int imu_count_ = 0;
    int scan_num_for_mean_ = 0;
    std::string sensor_type_ = "ouster";

    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;
    size_t native_keyframe_count_ = 0;

    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;
};

}  // namespace lightning

#endif  // LIGHTNING_LIO_SAM_MAPPING_H
