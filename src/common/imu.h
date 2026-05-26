//
// Created by xiang on 25-3-12.
//

#ifndef LIGHTNING_IMU_H
#define LIGHTNING_IMU_H

#include "common/eigen_types.h"

namespace lightning {

/// 内部定义的里程计观测，如果需要ROS，则从sensor_msg/Imu 的消息转换过来
struct IMU {
    double timestamp = 0;
    Vec3d angular_velocity = Vec3d::Zero();
    Vec3d linear_acceleration = Vec3d::Zero();
};

using IMUPtr = std::shared_ptr<IMU>;

}  // namespace lightning

#endif  // LIGHTNING_IMU_H
