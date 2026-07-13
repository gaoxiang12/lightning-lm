#ifndef LIGHTNING_UNDISTORTION_H
#define LIGHTNING_UNDISTORTION_H

#include "common/eigen_types.h"
#include "common/point_def.h"
#include "core/lightning_math.hpp"
#include "core/lio/pose6d.h"
#include <vector>

namespace lightning {

/**
 * 点云去畸变（后向传播）
 * 将点云从各自时刻变换到帧末尾时刻，消除运动畸变。
 *
 * 两个 LIO 前端 (AA-FasterLIO, FAST-LIO2) 使用相同的公式，可共享此函数。
 *
 * @param scan        输入/输出点云（原地修改）
 * @param imu_poses   IMU 传播过程中记录的位姿序列
 * @param R_end       帧末尾时刻的 IMU 旋转
 * @param t_end       帧末尾时刻的 IMU 位置
 * @param R_L_I       LiDAR→IMU 旋转外参
 * @param t_L_I       LiDAR→IMU 平移外参
 */
inline void UndistortPointCloud(
    CloudPtr& scan,
    const std::vector<Pose6D>& imu_poses,
    const Mat3d& R_end,
    const Vec3d& t_end,
    const Mat3d& R_L_I,
    const Vec3d& t_L_I)
{
    if (scan == nullptr || scan->empty() || imu_poses.size() < 2) {
        return;
    }

    auto it_pcl = scan->points.end() - 1;
    for (auto it_kp = imu_poses.end() - 1; it_kp != imu_poses.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;

        Mat3d R_imu = head->rot;
        Vec3d vel_imu = head->vel;
        Vec3d pos_imu = head->pos;
        Vec3d acc_imu = tail->acc;
        Vec3d angvel_avr = tail->gyr;

        for (; it_pcl->time / 1000.0 > head->offset_time; it_pcl--) {
            double dt = it_pcl->time / 1000.0 - head->offset_time;

            // 时刻 i 的 IMU 旋转
            Mat3d R_i = R_imu * math::exp(angvel_avr, dt).matrix();

            // 点坐标
            Vec3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);

            // 时刻 i 到帧末尾的位移（全局坐标系）
            Vec3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - t_end;

            // 后向去畸变：将点从时刻 i 变换到帧末尾
            // P_comp = R_L_I^T * (R_end^T * (R_i * (R_L_I * P + t_L_I) + T_ei) - t_L_I)
            Vec3d P_compensate = R_L_I.transpose() *
                (R_end.transpose() *
                 (R_i * (R_L_I * P_i + t_L_I) + T_ei) - t_L_I);

            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == scan->points.begin()) break;
        }
    }
}

}  // namespace lightning

#endif  // LIGHTNING_UNDISTORTION_H
