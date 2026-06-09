//
// Created by xiang on 25-3-12.
//

#ifndef LIGHTNING_PARAMS_H
#define LIGHTNING_PARAMS_H

#include <string>
#include <vector>

#include "common/eigen_types.h"

namespace lightning {

constexpr double GRAVITY_ = 9.7946;  // Gravity const in Shanghai/China

/// 激光类型
enum class SensorType {
    VELODYNE,  // 威力登
    OUSTER,    // ouster
    LIVOX,     // 览沃
    MID360
};

enum class FrontEndType {
    FASTER_LIO,  // 使用faster-lio作为前端
};

struct Params {
    Params() {}                            // not set
    Params(const std::string& yaml_path);  // load from yaml

    std::string yaml_path_;                             // yaml文件路径
    FrontEndType frontend_ = FrontEndType::FASTER_LIO;  // 前端类型
    bool process_cloud_in_step = false;                 // 是否单步调试
    bool online_mode = false;                           // 是否在线模式
    bool with_ui_ = false;                              // 是否带可视化UI
    int max_imu_init_count_ = 20;                       // IMU初始化使用的数据
    bool use_fasterlio_undistort_ = true;               // 是否使用fasterlio中的去畸变（与predict过程绑定）
    bool relative_cloud_pt_time_ =
        false;                         // if true: it_cloud->time(some point tm) should not - pcl_beg_time(0th point tm)
    bool enable_backend_ = false;      // 是否打开后端回环
    bool enable_frontend_log_ = true;  // 是否打开前端log
    bool enable_backend_log_ = true;   // 是否打开后端log
    bool is_vis_occupancy_map_ = false;  // 是否显示occu map
    bool enable_skip_cloud_ = false;
    bool frontend_record_kf_ = true;  // 前端是否记录关键帧（仅建图时需要，定位时不需要）
    int ivox_capacity_ = 1000000;     // 前端局部地图的缓存大小（定位时取更小的值)
    int skip_cloud_number_ = 4;
    bool enable_balm_ = false;

    // Topics
    std::string point_cloud_topic_;  // 输入点云话题
    std::string imu_topic_;          // 输入imu话题
    std::string odom_topic_;         // 输入odom话题

    // Frames
    std::string lidar_frame_;
    std::string baselink_frame_;
    std::string odom_frame_;
    std::string map_frame_;

    // Save pcd
    std::string save_pcd_dir_;
    float save_map_resolution_ = 0.2;

    // Lidar Sensor Configuration
    SensorType sensor_;
    float lidar_min_range_ = 0;
    float lidar_max_range_ = 0;

    float kf_dist_th_ = 0;
    float kf_angle_th_ = 0;

    // IMU
    double imu_scale_ = 1.0;

    Mat3d R_imu_lidar_;  //; R_imu_lidar, IMU->Lidar, or same P in Lidar->in IMU
    Vec3d t_imu_lidar_;  //; t_imu_lidar, IMU->Lidar

    // for Livox_imu
    int imu_type_ = 0;
    Quatd ext_qrpy_;

    // odom to imu
    SE3 Toi_;

    // trans input data(l means lidar)
    //  only used in input stage, will update Til/T_imu_lidar before also in input stage
    SE3 Tllold_used_input_;
};

}  // namespace lightning

#endif  // LIGHTNING_PARAMS_H
