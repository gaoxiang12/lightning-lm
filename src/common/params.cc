//
// Created by xiang on 25-3-26.
//

#include "common/params.h"
#include "core/lightning_math.hpp"
#include "io/yaml_io.h"

namespace lightning {

Params::Params(const std::string& yaml_path) {
    yaml_path_ = yaml_path;
    std::cout << "Loading params from " << yaml_path << std::endl;
    auto yaml = YAML::LoadFile(yaml_path);

    Sophus::SE3d Toi;

    try {
        std::string front_type = yaml["frontend_type"].as<std::string>();
        frontend_ = FrontEndType::FASTER_LIO;

        process_cloud_in_step = yaml["process_cloud_in_step"].as<bool>();
        online_mode = yaml["online_mode"].as<bool>();
        with_ui_ = yaml["with_ui"].as<bool>();
        enable_backend_ = yaml["enable_backend"].as<bool>();
        enable_frontend_log_ = yaml["enable_frontend_log"].as<bool>();
        enable_backend_log_ = yaml["enable_backend_log"].as<bool>();
        is_vis_occupancy_map_ = yaml["is_vis_occupancy_map"].as<bool>();
        enable_balm_ = yaml["enable_balm"].as<bool>();

        point_cloud_topic_ = yaml["pointCloudTopic"].as<std::string>();
        imu_topic_ = yaml["imuTopic"].as<std::string>();
        odom_topic_ = yaml["odomTopic"].as<std::string>();
        save_pcd_dir_ = yaml["savePCDDirectory"].as<std::string>();

        baselink_frame_ = yaml["agi_sam"]["baselinkFrame"].as<std::string>();
        odom_frame_ = yaml["agi_sam"]["odometryFrame"].as<std::string>();
        map_frame_ = yaml["agi_sam"]["mapFrame"].as<std::string>();
        save_map_resolution_ = yaml["agi_sam"]["save_map_resolution"].as<float>();

        std::string sensorStr = yaml["agi_sam"]["sensor"].as<std::string>();

        if (sensorStr == "velodyne") {
            sensor_ = SensorType::VELODYNE;
        } else if (sensorStr == "ouster") {
            sensor_ = SensorType::OUSTER;
        } else if (sensorStr == "livox") {
            sensor_ = SensorType::LIVOX;
        } else if (sensorStr == "mid360") {
            sensor_ = SensorType::MID360;
        } else {
        }

        lidar_min_range_ = yaml["agi_sam"]["lidarMinRange"].as<float>();
        lidar_max_range_ = yaml["agi_sam"]["lidarMaxRange"].as<float>();
        kf_dist_th_ = yaml["agi_sam"]["surroundingkeyframeAddingDistThreshold"].as<float>();
        kf_angle_th_ = yaml["agi_sam"]["surroundingkeyframeAddingAngleThreshold"].as<float>();

        imu_type_ = yaml["agi_sam"]["imuType"].as<int>();

        use_fasterlio_undistort_ = yaml["use_fasterlio_undistort"].as<bool>();
        relative_cloud_pt_time_ = yaml["relative_cloud_pt_time"].as<bool>();

        

        auto extrinT = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
        auto extrinR = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();
        // 变量名应修改为R_imu_lidar、t_imu_lidar，保持和 LIO-SAM 代码一致，避免混淆。Toi 是 LIO-SAM 里 IMU 到 LiDAR 的外参，保持不变。
        t_imu_lidar_  = math::VecFromArray<double>(extrinT);
        R_imu_lidar_  = math::MatFromArray<double>(extrinR);
        ext_qrpy_ = Quatd(R_imu_lidar_.transpose());
        

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error at line " << e.mark.line + 1 << ", column " << e.mark.column + 1 << ": "
                  << e.what() << std::endl;
        // 退出程序
        std::exit(EXIT_FAILURE);
    }
}
}  // namespace lightning