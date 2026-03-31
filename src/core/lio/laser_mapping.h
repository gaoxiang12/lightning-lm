#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "common/options.h"
#include "core/ivox3d/ivox3d.h"
#include "core/lio/eskf.hpp"
#include "core/lio/imu_processing.hpp"
#include "pointcloud_preprocess.h"
#include "common/pole_landmark.h"

#include "livox_ros_driver2/msg/custom_msg.hpp"

namespace lightning {

namespace ui {
class PangolinWindow;
}

/**
 * laser mapping
 * 目前有个问题：点云在缓存之后，实际处理的并不是最新的那个点云（通常是buffer里的前一个），这是因为bag里的点云用的开始时间戳，导致
 * 点云的结束时间要比IMU多0.1s左右。为了同步最近的IMU，就只能处理缓冲队列里的那个点云，而不是最新的点云
 */
class LaserMapping {
   public:
    struct SubmapCache {
        CloudPtr geom_cloud_{new PointCloudType()};
        std::vector<PoleLandmark> poles_;
        SE3 center_pose_;
        int end_kf_id_ = -1;
    };

    struct Options {
        Options() {}

        bool is_in_slam_mode_ = true;  // 是否在slam模式下

        /// rolling submap
        int submap_kf_window_ = 20;
        double submap_radius_ = 25.0;

        /// pole extraction
        bool use_pole_landmark_ = true;
        double pole_radius_ = 0.0375;
        double pole_radius_tol_ = 0.02;
        double pole_length_min_ = 0.2;
        double pole_length_max_ = 1.0;
        double pole_max_tilt_deg_ = 8.0;
        double pole_match_dist_th_ = 1.5;
        double pole_match_angle_deg_ = 10.0;

        /// intensity range-bin adaptive threshold
        float intensity_bin_size_ = 5.0f;
        int intensity_max_bins_ = 40;
        float intensity_quantile_ = 0.99f;
        int intensity_min_bin_points_ = 50;
        bool intensity_bin_smooth_ = true;

        /// pole cluster
        double pole_cluster_tol_ = 0.10;
        int pole_cluster_min_size_ = 8;
        int pole_cluster_max_size_ = 200;

        /// pole GN fit
        int pole_fit_max_iters_ = 5;
        double pole_fit_stop_th_ = 1e-4;
        /// 关键帧阈值
        double kf_dis_th_ = 2.0;
        double kf_angle_th_ = 15 * M_PI / 180.0;
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

    LaserMapping(Options options = Options());
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        LOG(INFO) << "laser mapping deconstruct";
    }

    /// init without ros
    bool Init(const std::string &config_yaml);

    bool Run();

    // callbacks of lidar and imu
    /// 处理ROS2的点云
    void ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr &msg);

    /// 处理livox的点云
    void ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg);

    /// 如果已经做了预处理，也可以直接处理点云
    void ProcessPointCloud2(CloudPtr cloud);

    void ProcessIMU(const lightning::IMUPtr &msg_in);

    /// 保存前端的地图
    void SaveMap();

    void SetUI(std::shared_ptr<ui::PangolinWindow> ui) { ui_ = ui; }

    /// 获取关键帧
    Keyframe::Ptr GetKeyframe() const { return last_kf_; }

    /// 获取激光的状态
    NavState GetState() const { return state_point_; }

    /// 获取IMU状态
    NavState GetIMUState() const {
        if (p_imu_->IsIMUInited()) {
            return kf_imu_.GetX();
        } else {
            NavState s;
            s.pose_is_ok_ = false;
            return s;
        }
    }

    CloudPtr GetScanUndist() const { return scan_undistort_; }

    /// 获取最新的点云
    CloudPtr GetRecentCloud();

    std::vector<Keyframe::Ptr> GetAllKeyframes() { return all_keyframes_; }

    /**
     * 计算全局地图
     * @param use_lio_pose
     * @return
     */
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1);

   private:
    // sync lidar with imu
    bool SyncPackages();

    void ObsModel(NavState &s, ESKF::CustomObservationModel &obs);

    /// 是否需要重建 rolling submap
    bool NeedRebuildSubmap(const SE3& pred_pose) const;

    /// 使用最近关键帧重建 rolling submap
    void RebuildSubmap(const SE3& pred_pose);

    /// 从当前帧提取高强度候选点（距离分桶自适应阈值）
    void ExtractHighIntensityCandidatesByRangeBins(
        const CloudPtr& cloud_in,
        CloudPtr& cloud_out,
        std::vector<float>& adaptive_thresholds) const;

    /// 对高强度候选点做聚类并拟合反光柱轴线
    void ExtractPoleLandmarksFromCloud(
        const CloudPtr& cloud_in,
        std::vector<PoleLandmark>& poles_out) const;

    /// 用柱面模型拟合轴线，要求接近竖直
    bool FitCylinderAxis(
        const CloudPtr& cluster,
        PoleLandmark& pole) const;

    /// 当前帧反光柱与 submap 中反光柱匹配
    void MatchPoleLandmarks(
        const std::vector<PoleLandmark>& cur_poles,
        const std::vector<PoleLandmark>& map_poles,
        std::vector<std::pair<int, int>>& matches) const;

    /// 建立反光柱 landmark 约束
    void BuildPoleResiduals(
        const std::vector<PoleLandmark>& cur_poles,
        const std::vector<PoleLandmark>& map_poles,
        const std::vector<std::pair<int, int>>& matches,
        NavState& s,
        ESKF::CustomObservationModel& obs) const;

    inline void PointBodyToWorld(const PointType &pi, PointType &po) {
        Vec3d p_global(state_point_.rot_ * (state_point_.offset_R_lidar_ * pi.getVector3fMap().cast<double>() +
                                            state_point_.offset_t_lidar_) +
                       state_point_.pos_);

        po.x = p_global(0);
        po.y = p_global(1);
        po.z = p_global(2);
        po.intensity = pi.intensity;
    }

    void MapIncremental();

    bool LoadParamsFromYAML(const std::string &yaml);

    /// 创建关键帧
    void MakeKF();

   private:
    Options options_;

    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

    /// local map related
    double filter_size_map_min_ = 0;

    /// rolling submap related
    SubmapCache submap_cache_;
    bool submap_inited_ = false;
    SE3 last_submap_pose_;

    int submap_kf_window_ = 20;
    double submap_radius_ = 25.0;
    double submap_rebuild_trans_th_ = 2.0;
    double submap_rebuild_rot_th_ = 10.0 * M_PI / 180.0;

    float intensity_bin_size_ = 5.0f;
    int intensity_max_bins_ = 40;
    float intensity_quantile_ = 0.99f;
    int intensity_min_bin_points_ = 50;
    bool intensity_bin_smooth_ = true;

    double pole_cluster_tol_ = 0.10;
    int pole_cluster_min_size_ = 8;
    int pole_cluster_max_size_ = 200;

    int pole_fit_max_iters_ = 5;
    double pole_fit_stop_th_ = 1e-4;
    /// params
    std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation
    std::vector<double> extrinR_{9, 0.0};  // lidar-imu rotation
    std::string map_file_path_;

    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;

    /// point clouds data
    CloudPtr scan_undistort_{new PointCloudType()};   // scan after undistortion
    CloudPtr scan_down_body_{new PointCloudType()};   // downsampled scan in body
    CloudPtr scan_down_world_{new PointCloudType()};  // downsampled scan in world
    std::vector<PointVector> nearest_points_;         // nearest points of current scan
    std::vector<Vec4f> corr_pts_;                     // inlier pts
    std::vector<Vec4f> corr_norm_;                    // inlier plane norms
    pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan

    std::vector<float> residuals_;           // point-to-plane residuals

    /// pole landmark
    std::vector<PoleLandmark> current_frame_poles_;
    std::vector<PoleLandmark> all_pole_landmarks_;

    /// 当前帧高强度候选点（调试和提柱用）
    CloudPtr high_intensity_cloud_{new PointCloudType()};

    /// 距离分桶阈值缓存
    mutable std::vector<float> intensity_bin_thresholds_;

    std::vector<bool> point_selected_surf_;  // selected points
    std::vector<Vec4f> plane_coef_;          // plane coeffs

    std::mutex mtx_buffer_;
    std::deque<double> time_buffer_;

    std::deque<PointCloudType::Ptr> lidar_buffer_;
    std::deque<lightning::IMUPtr> imu_buffer_;

    /// options
    bool keep_first_imu_estimation_ = false;    // 在没有建立地图前，是否要使用前几帧的IMU状态
    double timediff_lidar_wrt_imu_ = 0.0;
    double last_timestamp_lidar_ = 0;
    double lidar_end_time_ = 0;
    double last_timestamp_imu_ = -1.0;
    double first_lidar_time_ = 0.0;
    bool lidar_pushed_ = false;

    bool enable_skip_lidar_ = true;  // 雷达是否需要跳帧
    int skip_lidar_num_ = 5;         // 每隔多少帧跳一个雷达
    int skip_lidar_cnt_ = 0;

    /// statistics and flags ///
    int scan_count_ = 0;
    int publish_count_ = 0;
    bool flg_first_scan_ = true;
    bool flg_EKF_inited_ = false;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;
    int effect_feat_num_ = 0, frame_num_ = 0;

    /// 当前帧退化统计
    int current_nn_fail_ = 0;
    int current_plane_fail_ = 0;
    int current_residual_fail_ = 0;
    double current_valid_ratio_ = 0.0;
    double current_nn_fail_ratio_ = 0.0;
    bool current_frame_degenerate_ = false;

    /// 当前帧 landmark 匹配数量
    int current_pole_match_num_ = 0;

    double last_lidar_time_ = 0;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    MeasureGroup measures_;  // sync IMU and lidar scan

    ESKF kf_;      // 点云时刻的IMU状态
    ESKF kf_imu_;  // imu 最新时刻的eskf状态

    NavState state_point_;  // ekf current state

    Vec3d pos_lidar_;  // lidar position after eskf update
    SO3 euler_cur_;    // rotation in euler angles
    bool extrinsic_est_en_ = true;
    bool use_aa_ = false;  // use anderson acceleration?

    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;
};

}  // namespace lightning

#endif  // FASTER_LIO_LASER_MAPPING_H