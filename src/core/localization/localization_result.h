//
// Created by xiang on 2021/11/17.
//
#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "common/eigen_types.h"
#include "common/nav_state.h"

namespace lightning::loc {

/// 定位结果，包含所有的信息
/// 点云定位和最终结果都可以按照这个来输出

/// 定位状态
enum class LocalizationStatus {
    IDLE,          // 0 无定位，也没有准备初始化
    INITIALIZING,  // 1 初始化过程中，位姿无效
    GOOD,          // 2 正常定位
    FOLLOWING_DR,  // 3 异常定位，输出DR递推
    FAIL,          // 4 定位失败
};

/// 定位结果
struct LocalizationResult {
    /// 对外部分
    double timestamp_ = 0;                                  // 时间戳
    SE3 pose_;                                              // 定位的位置和姿态（lidarLoc和PGO共用此变量）
    bool valid_ = false;                                    // 标志此定位是否有效
    LocalizationStatus status_ = LocalizationStatus::IDLE;  // 定位状态位

    /// 对内部分
    bool lidar_loc_valid_ = false;        // 该时刻lidarLoc是否有效
    bool lidar_loc_inlier_ = false;       // 该时刻lidarLoc在PGO中是否是内点
    double confidence_ = 1.0;             // lidarLoc的confidence(若有效)
    double lidar_loc_error_vert_ = 0;     // lidarLoc相对于PGO定位的误差（纵向）
    double lidar_loc_error_hori_ = 0;     // lidarLoc相对于PGO定位的误差（横向）
    double lidar_loc_delta_t_ = 0;        // 相对于上一帧lidarLoc消息的时延
    double lidar_loc_odom_delta_ = 0;     // lidarLoc与lidarOdom对应时间两帧相差的距离
    bool lidar_loc_smooth_flag_ = false;  // Lidar loc是否满足平滑性要求（自身定位结果与外推结果类似）

    bool lidar_loc_odom_error_normal_ = true;  // lidarLoc与lidarOdom对应时间两帧相差的距离超过一定阈值则为false
    bool lidar_loc_odom_reliable_ = true;      // LO 认为自己的定位可不可信

    double lidar_odom_error_vert_ = 0;  // lidarOdom(通过滑窗首帧转换到map系下)相对于PGO定位的误差（纵向）
    double lidar_odom_error_hori_ = 0;  // lidarOdom(通过滑窗首帧转换到map系下)相对于PGO定位的误差（横向）

    bool rel_pose_set_ = false;      // 相对定位是否被设置（通常是lidarOdom）
    SE3 rel_pose_;                   // 相对定位的位置和姿态
    Vec3d vel_b_ = Vec3d::Zero();    // 相对定位的速度
    double lidar_odom_delta_t_ = 0;  // 相对于上一帧lidarOdom消息的时延
    double dr_delta_t_ = 0;          // 相对于上一帧DR消息的时延
    double is_parking_ = false;

    geometry_msgs::msg::TransformStamped ToMapOdomMsg() const;
    geometry_msgs::msg::TransformStamped ToOdomBaseMsg() const;
    nav_msgs::msg::Odometry ToOdomMsg() const;
    geometry_msgs::msg::PoseStamped ToPoseMsg() const;
    NavState ToNavState() const;                            // 转到navstate
};

}  // namespace lightning::loc
