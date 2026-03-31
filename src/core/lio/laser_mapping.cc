#include <pcl/common/transforms.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "common/options.h"
#include "core/lightning_math.hpp"
#include "laser_mapping.h"
#include "ui/pangolin_window.h"
#include "wrapper/ros_utils.h"

namespace lightning {

bool LaserMapping::Init(const std::string &config_yaml) {
    LOG(INFO) << "init laser mapping from " << config_yaml;

    LOG(INFO) << "build version, 2026-0326-1201 ... ";    
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    ESKF::Options eskf_options;
    eskf_options.max_iterations_ = fasterlio::NUM_MAX_ITERATIONS;
    eskf_options.epsi_ = 1e-3 * Eigen::Matrix<double, 23, 1>::Ones();
    eskf_options.lidar_obs_func_ = [this](NavState &s, ESKF::CustomObservationModel &obs) { ObsModel(s, obs); };
    eskf_options.use_aa_ = use_aa_;
    kf_.Init(eskf_options);

    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    // get params from yaml
    int lidar_type, ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_scan;
    Vec3d lidar_T_wrt_IMU;
    Mat3d lidar_R_wrt_IMU;

    auto yaml = YAML::LoadFile(yaml_file);
    try {
        fasterlio::NUM_MAX_ITERATIONS = yaml["fasterlio"]["max_iteration"].as<int>();
        fasterlio::ESTI_PLANE_THRESHOLD = yaml["fasterlio"]["esti_plane_threshold"].as<float>();

        filter_size_scan = yaml["fasterlio"]["filter_size_scan"].as<float>();
        filter_size_map_min_ = yaml["fasterlio"]["filter_size_map"].as<float>();
        keep_first_imu_estimation_ = yaml["fasterlio"]["keep_first_imu_estimation"].as<bool>();
        gyr_cov = yaml["fasterlio"]["gyr_cov"].as<float>();
        acc_cov = yaml["fasterlio"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["fasterlio"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["fasterlio"]["b_acc_cov"].as<float>();
        preprocess_->Blind() = yaml["fasterlio"]["blind"].as<double>();
        preprocess_->TimeScale() = yaml["fasterlio"]["time_scale"].as<double>();
        lidar_type = yaml["fasterlio"]["lidar_type"].as<int>();
        preprocess_->NumScans() = yaml["fasterlio"]["scan_line"].as<int>();
        preprocess_->PointFilterNum() = yaml["fasterlio"]["point_filter_num"].as<int>();
        extrinsic_est_en_ = yaml["fasterlio"]["extrinsic_est_en"].as<bool>();
        extrinT_ = yaml["fasterlio"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["fasterlio"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["fasterlio"]["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["fasterlio"]["ivox_nearby_type"].as<int>();
        use_aa_ = yaml["fasterlio"]["use_aa"].as<bool>();

        skip_lidar_num_ = yaml["fasterlio"]["skip_lidar_num"].as<int>();

        options_.submap_kf_window_ = yaml["fasterlio"]["submap_kf_window"].as<int>();
        options_.submap_radius_ = yaml["fasterlio"]["submap_radius"].as<double>();

        options_.use_pole_landmark_ = yaml["fasterlio"]["use_pole_landmark"].as<bool>();
        options_.pole_radius_ = yaml["fasterlio"]["pole_radius"].as<double>();
        options_.pole_radius_tol_ = yaml["fasterlio"]["pole_radius_tol"].as<double>();
        options_.pole_length_min_ = yaml["fasterlio"]["pole_length_min"].as<double>();
        options_.pole_length_max_ = yaml["fasterlio"]["pole_length_max"].as<double>();
        options_.pole_max_tilt_deg_ = yaml["fasterlio"]["pole_max_tilt_deg"].as<double>();
        options_.pole_match_dist_th_ = yaml["fasterlio"]["pole_match_dist_th"].as<double>();
        options_.pole_match_angle_deg_ = yaml["fasterlio"]["pole_match_angle_deg"].as<double>();
        options_.intensity_bin_size_ = yaml["fasterlio"]["intensity_bin_size"].as<float>();
        options_.intensity_max_bins_ = yaml["fasterlio"]["intensity_max_bins"].as<int>();
        options_.intensity_quantile_ = yaml["fasterlio"]["intensity_quantile"].as<float>();
        options_.intensity_min_bin_points_ = yaml["fasterlio"]["intensity_min_bin_points"].as<int>();
        options_.intensity_bin_smooth_ = yaml["fasterlio"]["intensity_bin_smooth"].as<bool>();

        options_.pole_cluster_tol_ = yaml["fasterlio"]["pole_cluster_tol"].as<double>();
        options_.pole_cluster_min_size_ = yaml["fasterlio"]["pole_cluster_min_size"].as<int>();
        options_.pole_cluster_max_size_ = yaml["fasterlio"]["pole_cluster_max_size"].as<int>();

        options_.pole_fit_max_iters_ = yaml["fasterlio"]["pole_fit_max_iters"].as<int>();
        options_.pole_fit_stop_th_ = yaml["fasterlio"]["pole_fit_stop_th"].as<double>();
        enable_skip_lidar_ = skip_lidar_num_ > 0;

    } catch (...) {
        LOG(ERROR) << "bad conversion";
        return false;
    }

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_scan, filter_size_scan, filter_size_scan);

    lidar_T_wrt_IMU = math::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = math::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(Vec3d(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(Vec3d(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(Vec3d(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(Vec3d(b_acc_cov, b_acc_cov, b_acc_cov));

    submap_kf_window_ = options_.submap_kf_window_;
    submap_radius_ = options_.submap_radius_;
    intensity_bin_size_ = options_.intensity_bin_size_;
    intensity_max_bins_ = options_.intensity_max_bins_;
    intensity_quantile_ = options_.intensity_quantile_;
    intensity_min_bin_points_ = options_.intensity_min_bin_points_;
    intensity_bin_smooth_ = options_.intensity_bin_smooth_;

    pole_cluster_tol_ = options_.pole_cluster_tol_;
    pole_cluster_min_size_ = options_.pole_cluster_min_size_;
    pole_cluster_max_size_ = options_.pole_cluster_max_size_;
    pole_fit_max_iters_ = options_.pole_fit_max_iters_;
    pole_fit_stop_th_ = options_.pole_fit_stop_th_;

    return true;
}

LaserMapping::LaserMapping(Options options) : options_(options) {
    preprocess_.reset(new PointCloudPreprocess());
    p_imu_.reset(new ImuProcess());
}

void LaserMapping::ProcessIMU(const lightning::IMUPtr &imu) {
    publish_count_++;

    double timestamp = imu->timestamp;

    UL lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }

    if (p_imu_->IsIMUInited()) {
        /// 更新最新imu状态
        kf_imu_.Predict(timestamp - last_timestamp_imu_, p_imu_->Q_, imu->angular_velocity, imu->linear_acceleration);

        // LOG(INFO) << "newest wrt lidar: " << timestamp - kf_.GetX().timestamp_;

        /// 更新ui
        if (ui_) {
            ui_->UpdateNavState(kf_imu_.GetX());
        }
    }

    last_timestamp_imu_ = timestamp;

    imu_buffer_.emplace_back(imu);
}

bool LaserMapping::Run() {
    if (!SyncPackages()) {
        return false;
    }

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);

    /// 以 IMU 预测位姿为中心维护 rolling submap
    SE3 pred_pose = kf_.GetX().GetPose();
    if (!submap_inited_ || NeedRebuildSubmap(pred_pose)) {
        RebuildSubmap(pred_pose);
    }

    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        LOG(WARNING) << "No point, skip this scan!";
        return false;
    }

    /// the first scan
    if (flg_first_scan_) {
        LOG(INFO) << "first scan pts: " << scan_undistort_->size();

        state_point_ = kf_.GetX();
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            PointBodyToWorld(scan_undistort_->points[i], scan_down_world_->points[i]);
        }
        ivox_->AddPoints(scan_down_world_->points);

        first_lidar_time_ = measures_.lidar_end_time_;
        state_point_.timestamp_ = lidar_end_time_;
        flg_first_scan_ = false;
        return true;
    }

    if (enable_skip_lidar_) {
        skip_lidar_cnt_++;
        skip_lidar_cnt_ = skip_lidar_cnt_ % skip_lidar_num_;

        if (skip_lidar_cnt_ != 0) {
            /// 更新UI中的内容
            if (ui_) {
                ui_->UpdateNavState(kf_.GetX());
                ui_->UpdateScan(scan_undistort_, kf_.GetX().GetPose());
            }

            return false;
        }
    }

    // LOG(INFO) << "LIO get cloud at beg: " << std::setprecision(14) << measures_.lidar_begin_time_
    //           << ", end: " << measures_.lidar_end_time_;

    if (last_lidar_time_ > 0 && (measures_.lidar_begin_time_ - last_lidar_time_) > 0.5) {
        LOG(ERROR) << "检测到雷达断流，时长：" << (measures_.lidar_begin_time_ - last_lidar_time_);
    }

    last_lidar_time_ = measures_.lidar_begin_time_;

    flg_EKF_inited_ = (measures_.lidar_begin_time_ - first_lidar_time_) >= fasterlio::INIT_TIME;

    /// downsample
    voxel_scan_.setInputCloud(scan_undistort_);
    voxel_scan_.filter(*scan_down_body_);

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->size() << ", " << scan_down_body_->size();
        return false;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);

    /// 提取当前帧高强度候选和反光柱 landmark
    current_frame_poles_.clear();
    high_intensity_cloud_->clear();

    if (options_.use_pole_landmark_) {
        ExtractHighIntensityCandidatesByRangeBins(scan_down_body_, high_intensity_cloud_, intensity_bin_thresholds_);
        ExtractPoleLandmarksFromCloud(high_intensity_cloud_, current_frame_poles_);
    }

    Timer::Evaluate(
        [&, this]() {
            // 成员变量预分配
            residuals_.resize(cur_pts, 0);
            point_selected_surf_.resize(cur_pts, true);
            plane_coef_.resize(cur_pts, Vec4f::Zero());

            auto old_state = kf_.GetX();

            kf_.Update(ESKF::ObsType::LIDAR, 1e-3);
            state_point_ = kf_.GetX();

            if (keep_first_imu_estimation_ && all_keyframes_.size() < 5 &&
                (old_state.rot_.inverse() * state_point_.rot_).log().norm() > 0.3 * M_PI / 180) {
                kf_.ChangeX(old_state);
                state_point_ = old_state;

                LOG(INFO) << "set state as prediction";
            }

            // LOG(INFO) << "old yaw: " << old_state.rot_.angleZ() << ", new: " << state_point_.rot_.angleZ();

            state_point_.timestamp_ = measures_.lidar_end_time_;
            euler_cur_ = state_point_.rot_;
            pos_lidar_ = state_point_.pos_ + state_point_.rot_ * state_point_.offset_t_lidar_;
        },
        "IEKF Solve and Update");

    for (auto& pole : current_frame_poles_) {
        pole.axis_point_ = state_point_.rot_ * pole.axis_point_body_ + state_point_.pos_;
        pole.axis_dir_ = state_point_.rot_ * pole.axis_dir_body_;
        pole.axis_dir_.normalize();
    }

    /// 坏帧必须在写图前拦截
    if (current_frame_degenerate_) {
        LOG(WARNING) << "[DEGENERATE][run] skip mapping/keyframe frame=" << scan_count_
                     << " nn_fail=" << current_nn_fail_
                     << " valid_ratio=" << current_valid_ratio_
                     << " pole_matches=" << current_pole_match_num_;
        return false;
    }

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

    LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " down " << cur_pts
              << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;

    /// keyframes
    if (last_kf_ == nullptr) {
        MakeKF();
    } else {
        SE3 last_pose = last_kf_->GetLIOPose();
        SE3 cur_pose = state_point_.GetPose();
        if ((last_pose.translation() - cur_pose.translation()).norm() > options_.kf_dis_th_ ||
            (last_pose.so3().inverse() * cur_pose.so3()).log().norm() > options_.kf_angle_th_) {
            MakeKF();
        } else if (!options_.is_in_slam_mode_ && (state_point_.timestamp_ - last_kf_->GetState().timestamp_) > 2.0) {
            MakeKF();
        }
    }

    /// 更新kf_for_imu
    kf_imu_ = kf_;
    if (!measures_.imu_.empty()) {
        double t = measures_.imu_.back()->timestamp;
        for (auto &imu : imu_buffer_) {
            double dt = imu->timestamp - t;
            kf_imu_.Predict(dt, p_imu_->Q_, imu->angular_velocity, imu->linear_acceleration);
            t = imu->timestamp;
        }
    }

    if (ui_) {
        ui_->UpdateScan(scan_undistort_, state_point_.GetPose());
    }

    return true;
}

void LaserMapping::MakeKF() {
    Keyframe::Ptr kf = std::make_shared<Keyframe>(kf_id_++, scan_undistort_, state_point_);

    kf->SetPoles(current_frame_poles_);

    if (last_kf_) {
        // LOG(INFO) << "last kf lio: " << last_kf_->GetLIOPose().translation().transpose()
        //           << ", opt: " << last_kf_->GetOptPose().translation().transpose();

        /// opt pose 用之前的递推
        SE3 delta = last_kf_->GetLIOPose().inverse() * kf->GetLIOPose();
        kf->SetOptPose(last_kf_->GetOptPose() * delta);
    } else {
        kf->SetOptPose(kf->GetLIOPose());
    }

    kf->SetState(state_point_);

    LOG(INFO) << "LIO: create kf " << kf->GetID() << ", state: " << state_point_.pos_.transpose()
              << ", kf opt pose: " << kf->GetOptPose().translation().transpose()
              << ", lio pose: " << kf->GetLIOPose().translation().transpose() << ", time: " << std::setprecision(14)
              << state_point_.timestamp_;

    if (options_.is_in_slam_mode_) {
        all_keyframes_.emplace_back(kf);
    }

    last_kf_ = kf;
}

void LaserMapping::ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr &msg) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            double timestamp = ToSec(msg->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            LOG(INFO) << "get cloud at " << std::setprecision(14) << timestamp
                      << ", latest imu: " << last_timestamp_imu_;

            CloudPtr cloud(new PointCloudType());
            preprocess_->Process(msg, cloud);

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

void LaserMapping::ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr &msg) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            double timestamp = ToSec(msg->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            // LOG(INFO) << "get cloud at " << std::setprecision(14) << timestamp
            //           << ", latest imu: " << last_timestamp_imu_;

            CloudPtr cloud(new PointCloudType());
            preprocess_->Process(msg, cloud);

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

void LaserMapping::ProcessPointCloud2(CloudPtr cloud) {
    UL lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;

            double timestamp = math::ToSec(cloud->header.stamp);
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
                lidar_buffer_.clear();
            }

            lidar_buffer_.push_back(cloud);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

bool LaserMapping::SyncPackages() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_.scan_ = lidar_buffer_.front();
        measures_.lidar_begin_time_ = time_buffer_.front();

        if (measures_.scan_->points.size() <= 1) {
            LOG(WARNING) << "Too few input point cloud!";
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else if (measures_.scan_->points.back().time / double(1000) < 0.5 * lidar_mean_scantime_) {
            lidar_end_time_ = measures_.lidar_begin_time_ + lidar_mean_scantime_;
        } else {
            scan_num_++;
            lidar_end_time_ = measures_.lidar_begin_time_ + measures_.scan_->points.back().time / double(1000);
            lidar_mean_scantime_ +=
                (measures_.scan_->points.back().time / double(1000) - lidar_mean_scantime_) / scan_num_;
        }

        lo::lidar_time_interval = lidar_mean_scantime_;

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    double imu_time = imu_buffer_.front()->timestamp;
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->timestamp;
        if (imu_time > lidar_end_time_) {
            break;
        }

        measures_.imu_.push_back(imu_buffer_.front());

        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;

    // LOG(INFO) << "sync: " << std::setprecision(14) << measures_.lidar_begin_time_ << ", " <<
    // measures_.lidar_end_time_;

    return true;
}

bool LaserMapping::NeedRebuildSubmap(const SE3& pred_pose) const {
    if (!submap_inited_) {
        return true;
    }

    double dt = (last_submap_pose_.translation() - pred_pose.translation()).norm();
    double dr = (last_submap_pose_.so3().inverse() * pred_pose.so3()).log().norm();

    if (dt > submap_rebuild_trans_th_) {
        return true;
    }
    if (dr > submap_rebuild_rot_th_) {
        return true;
    }

    return false;
}

void LaserMapping::RebuildSubmap(const SE3& pred_pose) {
    submap_cache_.geom_cloud_->clear();
    submap_cache_.poles_.clear();

    if (all_keyframes_.empty()) {
        ivox_.reset(new IVoxType(ivox_options_));
        submap_inited_ = true;
        last_submap_pose_ = pred_pose;
        return;
    }

    std::vector<Keyframe::Ptr> selected_kfs;
    selected_kfs.reserve(submap_kf_window_);

    /// 先按距离筛，再限制窗口大小
    for (int i = static_cast<int>(all_keyframes_.size()) - 1; i >= 0; --i) {
        auto& kf = all_keyframes_[i];
        double dist = (kf->GetLIOPose().translation() - pred_pose.translation()).norm();
        if (dist < submap_radius_) {
            selected_kfs.emplace_back(kf);
        }
        if (selected_kfs.size() >= static_cast<size_t>(submap_kf_window_)) {
            break;
        }
    }

    if (selected_kfs.empty()) {
        selected_kfs.emplace_back(all_keyframes_.back());
    }

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(filter_size_map_min_, filter_size_map_min_, filter_size_map_min_);

    for (auto& kf : selected_kfs) {
        CloudPtr cloud = kf->GetCloud();
        if (cloud == nullptr || cloud->empty()) {
            continue;
        }

        CloudPtr cloud_trans(new PointCloudType);
        pcl::transformPointCloud(*cloud, *cloud_trans, kf->GetLIOPose().matrix());

        *submap_cache_.geom_cloud_ += *cloud_trans;

        for (auto& pole : kf->GetPoles()) {
             submap_cache_.poles_.emplace_back(pole);
        }
    }

    CloudPtr filtered(new PointCloudType);
    voxel.setInputCloud(submap_cache_.geom_cloud_);
    voxel.filter(*filtered);
    submap_cache_.geom_cloud_ = filtered;

    ivox_.reset(new IVoxType(ivox_options_));
    ivox_->AddPoints(submap_cache_.geom_cloud_->points);

    submap_cache_.center_pose_ = pred_pose;
    submap_cache_.end_kf_id_ = selected_kfs.front()->GetID();

    submap_inited_ = true;
    last_submap_pose_ = pred_pose;

    LOG(INFO) << "[submap] rebuild, kfs=" << selected_kfs.size()
              << ", pts=" << submap_cache_.geom_cloud_->size()
              << ", grids=" << ivox_->NumValidGrids();
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    size_t cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    std::for_each(index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(scan_down_body_->points[i], scan_down_world_->points[i]);

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];
        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                point_no_need_downsample.emplace_back(point_world);
                return;
            }

            bool need_add = true;
            float dist = math::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= fasterlio::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < fasterlio::NUM_MATCH_POINTS; readd_i++) {
                    if (math::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }

            if (need_add) {
                points_to_add.emplace_back(point_world);  // FIXME 这并发可能有点问题
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    });

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

void LaserMapping::ExtractHighIntensityCandidatesByRangeBins(
    const CloudPtr& cloud_in,
    CloudPtr& cloud_out,
    std::vector<float>& adaptive_thresholds) const {

    cloud_out->clear();
    adaptive_thresholds.clear();

    if (cloud_in == nullptr || cloud_in->empty()) {
        return;
    }

    const float BIN_SIZE = intensity_bin_size_;
    const int MAX_BINS = intensity_max_bins_;
    const float QUANTILE = intensity_quantile_;
    const int MIN_BIN_POINTS = intensity_min_bin_points_;

    std::vector<std::vector<float>> bins(MAX_BINS);

    for (const auto& pt : cloud_in->points) {
        float range = pt.getVector3fMap().norm();
        int bin_id = std::min(MAX_BINS - 1, std::max(0, int(range / BIN_SIZE)));
        bins[bin_id].push_back(pt.intensity);
    }

    adaptive_thresholds.resize(MAX_BINS, std::numeric_limits<float>::quiet_NaN());

    for (int i = 0; i < MAX_BINS; ++i) {
        auto& vals = bins[i];
        if (static_cast<int>(vals.size()) < MIN_BIN_POINTS) {
            continue;
        }

        std::sort(vals.begin(), vals.end());
        size_t qid = static_cast<size_t>(QUANTILE * (vals.size() - 1));
        adaptive_thresholds[i] = vals[qid];
    }

    /// 空桶回填
    for (int i = 0; i < MAX_BINS; ++i) {
        if (std::isfinite(adaptive_thresholds[i])) continue;

        int l = i - 1, r = i + 1;
        while (l >= 0 && !std::isfinite(adaptive_thresholds[l])) --l;
        while (r < MAX_BINS && !std::isfinite(adaptive_thresholds[r])) ++r;

        if (l >= 0 && r < MAX_BINS) {
            adaptive_thresholds[i] = 0.5f * (adaptive_thresholds[l] + adaptive_thresholds[r]);
        } else if (l >= 0) {
            adaptive_thresholds[i] = adaptive_thresholds[l];
        } else if (r < MAX_BINS) {
            adaptive_thresholds[i] = adaptive_thresholds[r];
        }
    }

    /// 平滑
    if (intensity_bin_smooth_ && MAX_BINS >= 3) {
        std::vector<float> smoothed = adaptive_thresholds;
        for (int i = 1; i < MAX_BINS - 1; ++i) {
            if (std::isfinite(adaptive_thresholds[i - 1]) &&
                std::isfinite(adaptive_thresholds[i]) &&
                std::isfinite(adaptive_thresholds[i + 1])) {
                smoothed[i] = 0.25f * adaptive_thresholds[i - 1] +
                              0.5f  * adaptive_thresholds[i] +
                              0.25f * adaptive_thresholds[i + 1];
            }
        }
        adaptive_thresholds.swap(smoothed);
    }

    for (const auto& pt : cloud_in->points) {
        float range = pt.getVector3fMap().norm();
        int bin_id = std::min(MAX_BINS - 1, std::max(0, int(range / BIN_SIZE)));

        float th = adaptive_thresholds[bin_id];
        if (std::isfinite(th) && pt.intensity >= th) {
            cloud_out->points.push_back(pt);
        }
    }

    cloud_out->width = cloud_out->size();
    cloud_out->height = 1;
    cloud_out->is_dense = false;
}

void LaserMapping::ExtractPoleLandmarksFromCloud(
    const CloudPtr& cloud_in,
    std::vector<PoleLandmark>& poles_out) const {

    poles_out.clear();
    if (cloud_in == nullptr || cloud_in->empty()) {
        return;
    }

    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    tree->setInputCloud(cloud_in);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance(pole_cluster_tol_);
    ec.setMinClusterSize(pole_cluster_min_size_);
    ec.setMaxClusterSize(pole_cluster_max_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_in);
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        CloudPtr cluster(new PointCloudType);
        cluster->reserve(indices.indices.size());
        for (int idx : indices.indices) {
            cluster->push_back(cloud_in->points[idx]);
        }

        PoleLandmark pole;
        if (FitCylinderAxis(cluster, pole)) {
            poles_out.emplace_back(pole);
        }
    }
}

bool LaserMapping::FitCylinderAxis(
    const CloudPtr& cluster,
    PoleLandmark& pole) const {

    if (cluster == nullptr || cluster->size() < 8) {
        return false;
    }

    // ---------- 1) PCA 初始化 ----------
    Vec3d mean = Vec3d::Zero();
    for (const auto& pt : cluster->points) {
        mean += pt.getVector3fMap().cast<double>();
    }
    mean /= double(cluster->size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& pt : cluster->points) {
        Vec3d d = pt.getVector3fMap().cast<double>() - mean;
        cov += d * d.transpose();
    }
    cov /= double(cluster->size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    if (es.info() != Eigen::Success) {
        return false;
    }

    Vec3d c = mean;
    Vec3d u = es.eigenvectors().col(2).normalized();
    if (u.dot(Vec3d::UnitZ()) < 0) {
        u = -u;
    }

    double max_tilt_rad = options_.pole_max_tilt_deg_ * M_PI / 180.0;
    double init_ang = std::acos(std::clamp(std::abs(u.dot(Vec3d::UnitZ())), -1.0, 1.0));
    if (init_ang > max_tilt_rad) {
        return false;
    }

    // ---------- 2) 固定半径柱面 GN 精化 ----------
    const double r = options_.pole_radius_;

    Vec3d ref = (std::abs(u.z()) < 0.9) ? Vec3d::UnitZ() : Vec3d::UnitX();
    Vec3d b1 = (ref - ref.dot(u) * u).normalized();
    Vec3d b2 = u.cross(b1).normalized();

    for (int iter = 0; iter < pole_fit_max_iters_; ++iter) {
        Eigen::Matrix<double, 5, 5> H = Eigen::Matrix<double, 5, 5>::Zero();
        Eigen::Matrix<double, 5, 1> g = Eigen::Matrix<double, 5, 1>::Zero();

        int valid_cnt = 0;

        for (const auto& pt : cluster->points) {
            Vec3d p = pt.getVector3fMap().cast<double>();
            Vec3d d = p - c;
            double proj = d.dot(u);
            Vec3d radial = d - proj * u;
            double nr = radial.norm();
            if (nr < 1e-8) {
                continue;
            }

            double e = nr - r;
            valid_cnt++;
            //
            Eigen::RowVector3d de_dc =
                -(radial.transpose() / nr) * (Mat3d::Identity() - u * u.transpose());

            Eigen::RowVector3d de_du =
                -(proj / nr) * radial.transpose();

            double de_da = de_du.dot(b1);
            double de_db = de_du.dot(b2);

            Eigen::Matrix<double, 1, 5> J;
            J << de_dc(0), de_dc(1), de_dc(2), de_da, de_db;

            H += J.transpose() * J;
            g += J.transpose() * e;
        }

        if (valid_cnt < 6) {
            return false;
        }

        Eigen::Matrix<double, 5, 1> dx = -H.ldlt().solve(g);
        if (!dx.allFinite()) {
            return false;
        }

        c += dx.head<3>();
        Vec3d du = dx(3) * b1 + dx(4) * b2;
        u = (u + du).normalized();
        if (u.dot(Vec3d::UnitZ()) < 0) {
            u = -u;
        }

        ref = (std::abs(u.z()) < 0.9) ? Vec3d::UnitZ() : Vec3d::UnitX();
        b1 = (ref - ref.dot(u) * u).normalized();
        b2 = u.cross(b1).normalized();

        if (dx.norm() < pole_fit_stop_th_) {
            break;
        }
    }

    // ---------- 3) 再做几何一致性检查 ----------
    double ang = std::acos(std::clamp(std::abs(u.dot(Vec3d::UnitZ())), -1.0, 1.0));
    if (ang > max_tilt_rad) {
        return false;
    }

    std::vector<double> radial_dists;
    radial_dists.reserve(cluster->size());

    double min_proj = std::numeric_limits<double>::max();
    double max_proj = -std::numeric_limits<double>::max();
    double mean_intensity = 0.0;

    for (const auto& pt : cluster->points) {
        Vec3d p = pt.getVector3fMap().cast<double>();
        Vec3d d = p - c;
        double proj = d.dot(u);
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);

        Vec3d radial = d - proj * u;
        radial_dists.push_back(radial.norm());

        mean_intensity += pt.intensity;
    }

    mean_intensity /= double(cluster->size());
    std::sort(radial_dists.begin(), radial_dists.end());
    double radius = radial_dists[radial_dists.size() / 2];
    double length = max_proj - min_proj;

    if (std::abs(radius - options_.pole_radius_) > options_.pole_radius_tol_) {
        return false;
    }

    if (length < options_.pole_length_min_ || length > options_.pole_length_max_) {
        return false;
    }

    pole.axis_point_body_ = c;
    pole.axis_dir_body_ = u;

    pole.axis_point_ = c;
    pole.axis_dir_ = u;

    pole.radius_ = radius;
    pole.length_ = length;
    pole.mean_intensity_ = mean_intensity;
    pole.support_ = static_cast<int>(cluster->size());
    pole.timestamp_ = state_point_.timestamp_;

    return true;
}

void LaserMapping::MatchPoleLandmarks(
    const std::vector<PoleLandmark>& cur_poles,
    const std::vector<PoleLandmark>& map_poles,
    std::vector<std::pair<int, int>>& matches) const {

    matches.clear();
    if (cur_poles.empty() || map_poles.empty()) {
        return;
    }

    double angle_th = options_.pole_match_angle_deg_ * M_PI / 180.0;

    for (int i = 0; i < static_cast<int>(cur_poles.size()); ++i) {
        double best_dist = 1e9;
        int best_j = -1;

        for (int j = 0; j < static_cast<int>(map_poles.size()); ++j) {
            double ang = std::acos(std::clamp(
                std::abs(cur_poles[i].axis_dir_.dot(map_poles[j].axis_dir_)),
                -1.0, 1.0));
            if (ang > angle_th) {
                continue;
            }

            double dist = (cur_poles[i].axis_point_ - map_poles[j].axis_point_).head<2>().norm();
            if (dist < options_.pole_match_dist_th_ && dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }

        if (best_j >= 0) {
            matches.emplace_back(i, best_j);
        }
    }
}

void LaserMapping::BuildPoleResiduals(
    const std::vector<PoleLandmark>& cur_poles,
    const std::vector<PoleLandmark>& map_poles,
    const std::vector<std::pair<int, int>>& matches,
    NavState& s,
    ESKF::CustomObservationModel& obs) const {

    if (matches.empty()) {
        return;
    }

    int old_rows = obs.h_x_.rows();
    int add_rows = static_cast<int>(matches.size()) * 2;

    Eigen::MatrixXd h_new = Eigen::MatrixXd::Zero(old_rows + add_rows, 12);
    Eigen::VectorXd r_new = Eigen::VectorXd::Zero(old_rows + add_rows);

    if (old_rows > 0) {
        h_new.topRows(old_rows) = obs.h_x_;
        r_new.head(old_rows) = obs.residual_;
    }

    int row = old_rows;
    Mat3d Rwb = s.rot_.matrix();

    for (const auto& m : matches) {
        const auto& pc = cur_poles[m.first];
        const auto& pm = map_poles[m.second];

        /// body系柱心 -> world系
        Vec3d p_b = pc.axis_point_body_;
        Vec3d p_w = s.rot_ * p_b + s.pos_;

        Vec2d res = (pm.axis_point_ - p_w).head<2>();

        Eigen::Matrix<double, 2, 12> J = Eigen::Matrix<double, 2, 12>::Zero();

        /// 对位置误差增量的导数
        J.block<2, 3>(0, 0) = -Eigen::Matrix<double, 2, 3>::Identity();

        /// 对旋转误差增量的导数
        Mat3d px = math::SKEW_SYM_MATRIX(p_b);
        J.block<2, 3>(0, 3) = -(Rwb * px).topRows<2>();

        /// 这里暂时不对外参建 pole 约束，后 6 维保留为 0

        h_new.block(row, 0, 2, 12) = J;
        r_new.segment<2>(row) = res;
        row += 2;
    }

    obs.h_x_ = h_new;
    obs.residual_ = r_new;
}
/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
void LaserMapping::ObsModel(NavState &s, ESKF::CustomObservationModel &obs) {
    int cnt_pts = scan_down_body_->size();

    current_frame_degenerate_ = false;
    current_nn_fail_ = 0;
    current_plane_fail_ = 0;
    current_residual_fail_ = 0;
    current_valid_ratio_ = 0.0;
    current_nn_fail_ratio_ = 0.0;
    current_pole_match_num_ = 0;

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            Mat3f R_wl = (s.rot_ * s.offset_R_lidar_).matrix().cast<float>();
            Vec3f t_wl = (s.rot_ * s.offset_t_lidar_ + s.pos_).cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                /* transform to world frame */
                Vec3f p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                points_near.clear();

                /** Find the closest surfaces in the map **/
                // if (obs.converge_) {
                ivox_->GetClosestPoint(point_world, points_near, fasterlio::NUM_MATCH_POINTS,20.0);
                point_selected_surf_[i] = points_near.size() >= fasterlio::MIN_NUM_MATCH_POINTS;
                if (point_selected_surf_[i]) {
                    point_selected_surf_[i] =
                        math::esti_plane(plane_coef_[i], points_near, fasterlio::ESTI_PLANE_THRESHOLD);
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    } else {
                        point_selected_surf_[i] = false;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");
    
    int dbg_nn_fail = 0;
    int dbg_plane_fail = 0;
    int dbg_residual_fail = 0;
    int dbg_valid = 0;

    for (int i = 0; i < cnt_pts; ++i) {
        if (nearest_points_[i].size() < fasterlio::MIN_NUM_MATCH_POINTS) {
            dbg_nn_fail++;
            continue;
        }
        if (!point_selected_surf_[i]) {
            /// 这里 point_selected_surf_ 为 false 既可能是 plane fail，也可能是 residual fail
            /// 简单起见先按 residual / plane 粗分
            if (std::abs(residuals_[i]) < 0.5) {
                dbg_plane_fail++;
            } else {
                dbg_residual_fail++;
            }
            continue;
        }
        dbg_valid++;
    }

    current_nn_fail_ = dbg_nn_fail;
    current_plane_fail_ = dbg_plane_fail;
    current_residual_fail_ = dbg_residual_fail;
    if (cnt_pts > 0) {
        current_nn_fail_ratio_ = static_cast<double>(dbg_nn_fail) / static_cast<double>(cnt_pts);
        current_valid_ratio_ = static_cast<double>(dbg_valid) / static_cast<double>(cnt_pts);
    }

    if ((cnt_pts > 3000 && current_nn_fail_ratio_ > 0.20) ||
        (cnt_pts > 3000 && current_valid_ratio_ < 0.80) ||
        (dbg_valid < 500)) {
        current_frame_degenerate_ = true;
        obs.valid_ = false;

        LOG(WARNING) << "[DEGENERATE][obsmodel] reject frame="
                     << scan_count_
                     << " cnt_pts=" << cnt_pts
                     << " nn_fail=" << dbg_nn_fail
                     << " valid=" << dbg_valid
                     << " nn_fail_ratio=" << current_nn_fail_ratio_
                     << " valid_ratio=" << current_valid_ratio_;
        return;
    }

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        obs.valid_ = false;
        LOG(WARNING) << "No Effective Points!";
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
            obs.h_x_ = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
            obs.residual_.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const Mat3f off_R = s.offset_R_lidar_.matrix().cast<float>();
            const Vec3f off_t = s.offset_t_lidar_.cast<float>();
            const Mat3f Rt = s.rot_.matrix().transpose().cast<float>();

            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                Vec3f point_this_be = corr_pts_[i].head<3>();
                Mat3f point_be_crossmat = math::SKEW_SYM_MATRIX(point_this_be);
                Vec3f point_this = off_R * point_this_be + off_t;
                Mat3f point_crossmat = math::SKEW_SYM_MATRIX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                Vec3f norm_vec = corr_norm_[i].head<3>();

                /*** calculate the Measurement Jacobian matrix H ***/
                Vec3f C(Rt * norm_vec);
                Vec3f A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    Vec3f B(point_be_crossmat * off_R.transpose() * C);
                    obs.h_x_.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0], B[1],
                        B[2], C[0], C[1], C[2];
                } else {
                    obs.h_x_.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0;
                }

                /// 增加了cauchy's robust kernel
                float res = -corr_pts_[i][3];
                float rho, drho;

                const float delta = 2.0;
                const float dsqr = delta * delta;
                const float dsqr_inv = 1.0 / dsqr;

                if (res >= 0) {
                    rho = dsqr * std::log(1 + res * dsqr_inv);
                    drho = 1.0 / (1 + res * dsqr_inv);
                } else {
                    rho = -dsqr * std::log(1 - res * dsqr_inv);
                    drho = 1.0 / (1 - res * dsqr_inv);
                }

                obs.residual_(i) = rho;
                obs.h_x_.block<1, 12>(i, 0) = obs.h_x_.block<1, 12>(i, 0).eval() * drho;

                // obs.residual_(i) = res;
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
    
    if (options_.use_pole_landmark_ && !current_frame_poles_.empty() && !submap_cache_.poles_.empty()) {
        std::vector<std::pair<int, int>> pole_matches;
        MatchPoleLandmarks(current_frame_poles_, submap_cache_.poles_, pole_matches);
        current_pole_match_num_ = static_cast<int>(pole_matches.size());

        if (!pole_matches.empty()) {
            //注意：先不用反光柱这个特征点信息了，后续如果要加，可以在这里调用 BuildPoleResiduals 来构建残差和雅可比
            //BuildPoleResiduals(current_frame_poles_, submap_cache_.poles_, pole_matches, s, obs);
        }
    }    

    /// 填入中位数平方误差
    std::vector<double> res_sq2;
    for (size_t i = 0; i < cnt_pts; ++i) {
        if (point_selected_surf_[i]) {
            double r = residuals_[i];
            res_sq2.emplace_back(r * r);
        }
    }

    std::sort(res_sq2.begin(), res_sq2.end());
    obs.lidar_residual_mean_ = res_sq2[res_sq2.size() / 2];
    obs.lidar_residual_max_ = res_sq2[res_sq2.size() - 1];
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////

CloudPtr LaserMapping::GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) {
    CloudPtr global_map(new PointCloudType);

    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(res, res, res);

    for (auto &kf : all_keyframes_) {
        CloudPtr cloud = kf->GetCloud();

        CloudPtr cloud_filter(new PointCloudType);

        if (use_voxel) {
            voxel.setInputCloud(cloud);
            voxel.filter(*cloud_filter);

        } else {
            cloud_filter = cloud;
        }

        CloudPtr cloud_trans(new PointCloudType);

        if (use_lio_pose) {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetLIOPose().matrix());
        } else {
            pcl::transformPointCloud(*cloud_filter, *cloud_trans, kf->GetOptPose().matrix());
        }

        *global_map += *cloud_trans;

        LOG(INFO) << "kf " << kf->GetID() << ", pose: " << kf->GetOptPose().translation().transpose();
    }

    CloudPtr global_map_filtered(new PointCloudType);
    if (use_voxel) {
        voxel.setInputCloud(global_map);
        voxel.filter(*global_map_filtered);
    } else {
        global_map_filtered = global_map;
    }

    global_map_filtered->is_dense = false;
    global_map_filtered->height = 1;
    global_map_filtered->width = global_map_filtered->size();

    LOG(INFO) << "global map: " << global_map_filtered->size();

    return global_map_filtered;
}

void LaserMapping::SaveMap() {
    /// 保存地图
    auto global_map = GetGlobalMap(true);

    pcl::io::savePCDFileBinaryCompressed("./data/lio.pcd", *global_map);

    LOG(INFO) << "lio map is saved to ./data/lio.pcd";
}

CloudPtr LaserMapping::GetRecentCloud() {
    if (lidar_buffer_.empty()) {
        return nullptr;
    }

    return lidar_buffer_.front();
}

}  // namespace lightning