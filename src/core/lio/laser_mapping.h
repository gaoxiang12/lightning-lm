#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <thread>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "common/options.h"
#include "core/ivox3d/ivox3d.h"
#include "core/lio/eskf.hpp"
#include "core/lio/imu_processing.hpp"
#include "core/frontend/lio_frontend.h"

namespace lightning {

/**
 * laser mapping (AA-FasterLIO)
 * 基于 12D ESKF 的 LiDAR-Inertial 里程计
 *
 * 继承 LIOFrontend 接口，支持多态调用
 * 公共功能（预处理、缓冲区管理、数据同步）由基类 LIOFrontend 实现
 */
class LaserMapping : public LIOFrontend {
   public:
    struct Options {
        Options() {}

        bool is_in_slam_mode_ = true;

        bool enable_icp_part_ = true;
        double plane_icp_weight_ = 1.0;
        double icp_weight_ = 100;

        int min_pts = 300;

        double kf_dis_th_ = 2.0;
        double kf_angle_th_ = 15 * M_PI / 180.0;

        bool proj_kfs_ = false;
        int max_proj_kfs_ = 5;
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

    LaserMapping(Options options = Options());
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_down_world_ = nullptr;
        LOG(INFO) << "laser mapping deconstruct";
    }

    // ====== LIOFrontend 接口实现 ======

    bool Init(const std::string &config_yaml) override;
    bool Run() override;
    void ProcessIMU(const IMUPtr &msg_in) override;  // 重写：添加实时 IMU 预测

    SE3 GetPose() const override { return state_point_.GetPose(); }
    CloudPtr GetScanUndist() const override { return scan_undistort_; }
    CloudPtr GetScanDownWorld() const override { return scan_down_world_; }
    Keyframe::Ptr GetKeyframe() const override { return last_kf_; }
    std::vector<Keyframe::Ptr> GetAllKeyframes() override { return all_keyframes_; }

    void SetUI(std::shared_ptr<ui::PangolinWindow> ui) override { ui_ = ui; }
    void SaveMap() override;
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1) override;
    void PrintExtrinsic() override;

    // ====== 原有接口 (向后兼容) ======

    NavState GetState() const { return state_point_; }

    NavState GetIMUState() const {
        if (p_imu_->IsIMUInited()) {
            return kf_imu_.GetX();
        } else {
            NavState s;
            s.pose_is_ok_ = false;
            return s;
        }
    }

    CloudPtr GetProjCloud();
    CloudPtr GetRecentCloud();

    bool IsExtrinsicEstEnabled() const override { return extrinsic_est_en_; }
    Vec3d GetExtrinsicT() const override { return offset_t_lidar_fixed_; }
    Mat3d GetExtrinsicR() const override { return offset_R_lidar_fixed_; }

   private:
    bool SyncPackages();

    void ObsModel(NavState &s, ESKF::CustomObservationModel &obs);

    inline void PointBodyToWorld(const PointType &pi, PointType &po) {
        Vec3d p_global(state_point_.rot_ *
                           (offset_R_lidar_fixed_ * pi.getVector3fMap().cast<double>() + offset_t_lidar_fixed_) +
                       state_point_.pos_);

        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.intensity = pi.intensity;
    }

    void MapIncremental();
    bool LoadParamsFromYAML(const std::string &yaml);
    void MakeKF();
    void ProjectKFs(CloudPtr cloud, int size_limit = 1000);

   private:
    Options options_;

    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;

    /// local map related
    double filter_size_map_min_ = 0;

    /// params
    std::vector<double> extrinT_{3, 0.0};
    std::vector<double> extrinR_{9, 0.0};
    Mat3d offset_R_lidar_fixed_ = Mat3d::Identity();
    Vec3d offset_t_lidar_fixed_ = Vec3d::Zero();
    std::string map_file_path_;

    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;

    /// point clouds data
    CloudPtr scan_down_body_{new PointCloudType()};
    CloudPtr scan_down_world_{new PointCloudType()};
    pcl::VoxelGrid<PointType> voxel_scan_;

    /// 点面相关
    std::vector<PointVector> nearest_points_;
    std::vector<Vec4f> corr_pts_;
    std::vector<Vec4f> corr_norm_;
    std::vector<float> residuals_;
    std::vector<char> point_selected_surf_;
    std::vector<Vec4f> plane_coef_;

    /// 点到点相关
    std::vector<char> point_selected_icp_;

    /// options
    bool keep_first_imu_estimation_ = false;
    bool extrinsic_est_en_ = false;
    double timediff_lidar_wrt_imu_ = 0.0;
    double last_timestamp_lidar_ = 0;
    double first_lidar_time_ = 0.0;

    bool enable_skip_lidar_ = true;
    int skip_lidar_num_ = 5;
    int skip_lidar_cnt_ = 0;

    /// statistics and flags ///
    int scan_count_ = 0;
    int publish_count_ = 0;
    bool flg_first_scan_ = true;
    bool flg_EKF_inited_ = false;
    int effect_feat_surf_ = 0, frame_num_ = 0, effect_feat_icp_ = 0;

    double last_lidar_time_ = 0;

    MeasureGroup measures_;
    ESKF kf_;
    ESKF kf_imu_;
    NavState state_point_;
    bool use_aa_ = false;
    std::list<Keyframe::Ptr> proj_kfs_;
};

}  // namespace lightning

#endif  // FASTER_LIO_LASER_MAPPING_H
