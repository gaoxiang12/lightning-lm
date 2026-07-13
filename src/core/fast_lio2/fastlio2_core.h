// FAST-LIO2 核心封装
// 封装 ESEKF + ikd-Tree，提供 Init/IMUProcess/Observe/UpdateMap 接口
// 去除 ROS 依赖，使用 Lightning-LM 的点类型和数据结构

#ifndef LIGHTNING_FASTLIO2_CORE_H
#define LIGHTNING_FASTLIO2_CORE_H

#include <deque>
#include <memory>
#include <vector>

#include <pcl/filters/voxel_grid.h>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "common/point_def.h"
#include "core/lio/pose6d.h"

// ikd-Tree
#include "ikd-Tree/ikd_Tree.h"

// ESEKF (via IKFoM_toolkit)
#include "core/fast_lio2/fastlio2_use_ikfom.hpp"
#include "core/fast_lio2/fastlio2_so3_math.h"

namespace lightning {

/// FAST-LIO2 内部使用的数据同步结构
struct FASTLIO2MeasureGroup {
    FASTLIO2MeasureGroup() {
        lidar_beg_time = 0.0;
        lidar = std::make_shared<PointCloudType>();
    }
    double lidar_beg_time;
    double lidar_end_time;
    CloudPtr lidar;
    std::deque<IMUPtr> imu;
};

class FASTLIO2Core {
    // 观测模型需要访问内部状态，作为友元函数
    friend void fastlio2_h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data);

public:
    struct Config {
        int lidar_type = 4;          // 1=Livox, 2=Velodyne, 3=Ouster, 4=RoboSense
        int scan_line = 64;
        double blind = 0.5;
        double filter_size_scan = 0.5;
        double filter_size_map = 0.5;
        int max_iteration = 4;
        double acc_cov = 0.1;
        double gyr_cov = 0.1;
        double b_acc_cov = 0.0001;
        double b_gyr_cov = 0.0001;
        Vec3d extrinsic_T = Vec3d::Zero();
        Mat3d extrinsic_R = Mat3d::Identity();
    };

    struct NativeState {
        Vec3d pos;
        Mat3d rot;
        Mat3d offset_R_L_I;  // rotation from IMU to LiDAR
        Vec3d offset_T_L_I;  // translation from IMU to LiDAR
        Vec3d vel;
        Vec3d bg;
        Vec3d ba;
        Vec3d grav;
    };

    FASTLIO2Core();
    ~FASTLIO2Core();

    bool Init(const Config& config);

    /// IMU 传播 + 去畸变
    void IMUProcess(const FASTLIO2MeasureGroup& measures, CloudPtr& scan_out);

    /// 观测更新 (点到面 ICP + ESEKF)
    void Observe(CloudPtr scan_down);

    /// 地图更新 (添加点到 ikd-Tree)
    void UpdateMap(CloudPtr scan_world);

    SE3 GetPose() const;
    NativeState GetNativeState() const;

    /// 获取 ikd-Tree 引用
    KD_TREE<PointType>& GetKDTree() { return ikdtree_; }

    /// 点云变换到世界坐标系
    void PointBodyToWorld(const PointType& pi, PointType& po);

    /// 获取降采样后的点云 (body 坐标系)
    CloudPtr GetScanDownBody() const { return feats_down_body_; }

    /// 获取降采样后的点云 (world 坐标系)
    CloudPtr GetScanDownWorld() const { return feats_down_world_; }

    /// 获取最近邻搜索结果
    std::vector<PointVector>& GetNearestPoints() { return Nearest_Points; }

    /// 获取有效特征点数
    int GetEffFeatNum() const { return effct_feat_num_; }

    /// 获取降采样后的点云大小
    int GetFeatsDownSize() const { return feats_down_size_; }

    /// 手动设置降采样后特征数（在 mapping 层 DownSample 后调用）
    void SetFeatsDownSize(int n) { feats_down_size_ = n; }

    bool IsInitialized() const { return flg_EKF_inited_; }

    /// 获取当前实例指针 (用于观测模型回调)
    static FASTLIO2Core*GetInstance() { return instance_; }

private:
    void IMU_init(const FASTLIO2MeasureGroup& meas, int& N);

    static FASTLIO2Core* instance_;  // 用于观测模型回调

    Config config_;

    // ESEKF
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;
    state_ikfom state_point_;
    Eigen::Matrix<double, 12, 12> Q_;

    // ikd-Tree
    KD_TREE<PointType> ikdtree_;

    // 点云
    CloudPtr feats_undistort_{new PointCloudType()};
    CloudPtr feats_down_body_{new PointCloudType()};
    CloudPtr feats_down_world_{new PointCloudType()};
    CloudPtr normvec_{new PointCloudType(100000, 1)};
    CloudPtr laserCloudOri_{new PointCloudType(100000, 1)};
    CloudPtr corr_normvect_{new PointCloudType(100000, 1)};

    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterSurf_;
    pcl::VoxelGrid<PointType> downSizeFilterMap_;

    // IMU 处理
    Pose6D last_imu_pose_;
    Vec3d mean_acc_{0, 0, -1.0};
    Vec3d mean_gyr_{0, 0, 0};
    Vec3d cov_acc_{0.1, 0.1, 0.1};
    Vec3d cov_gyr_{0.1, 0.1, 0.1};
    Vec3d angvel_last_{Vec3d::Zero()};
    Vec3d acc_s_last_{Vec3d::Zero()};
    int init_iter_num_ = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
    double first_lidar_time_ = 0.0;
    double last_lidar_end_time_ = -1.0;

    // 状态
    bool flg_EKF_inited_ = false;
    int effct_feat_num_ = 0;
    int feats_down_size_ = 0;
    Vec3d pos_lid_;
    Vec3d euler_cur_;

    // 最近邻搜索
    std::vector<PointVector> Nearest_Points;
    bool point_selected_surf_[100000] = {false};
    float res_last_[100000] = {0.0};

    // 常量
    static constexpr int MAX_INI_COUNT = 10;
    static constexpr double INIT_TIME = 0.1;
    static constexpr double LASER_POINT_COV = 0.001;
    static constexpr int NUM_MATCH_POINTS = 5;

    // 外参
    Mat3d Lidar_R_wrt_IMU_;
    Vec3d Lidar_T_wrt_IMU_;
};

/// 观测模型回调函数 (ESEKF 需要的全局/静态函数)
void fastlio2_h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data);

}  // namespace lightning

#endif  // LIGHTNING_FASTLIO2_CORE_H
