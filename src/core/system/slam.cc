#include "core/system/slam.h"

#include <filesystem>
#include <fstream>
#include <utility>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

#include "core/g2p5/g2p5.h"
#include "core/lio/laser_mapping.h"
#include "core/lio/lio_sam/lio_sam_mapping.h"
#include "core/loop_closing/loop_closing.h"
#include "core/maps/tiled_map.h"
#include "ui/pangolin_window.h"
#include "utils/timer.h"
#include "wrapper/ros_utils.h"

namespace lightning {

namespace {

std::string ReadFrontendName(const YAML::Node& yaml) {
    const YAML::Node system = yaml["system"];
    if (!system) {
        return "fasterlio";
    }
    if (system["frontend"]) {
        return system["frontend"].as<std::string>();
    }
    if (system["frontend_type"]) {
        return system["frontend_type"].as<std::string>();
    }
    if (system["lio_frontend"]) {
        return system["lio_frontend"].as<std::string>();
    }
    return "fasterlio";
}

IMUPtr ConvertRosImu(const sensor_msgs::msg::Imu::SharedPtr& msg) {
    IMUPtr imu = std::make_shared<IMU>();
    imu->timestamp = ToSec(msg->header.stamp);
    imu->linear_acceleration =
        Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    imu->angular_velocity = Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    return imu;
}

}  // namespace

SlamSystem::SlamSystem(lightning::SlamSystem::Options options) : options_(options) {
    signal(SIGINT, lightning::debug::SigHandle);
}

bool SlamSystem::Init(const std::string& yaml_path) {
    auto yaml = YAML::LoadFile(yaml_path);

    const std::string frontend = ReadFrontendName(yaml);
    use_lio_sam_ = frontend == "lio_sam" || frontend == "liosam" || frontend == "LIO-SAM";

    if (use_lio_sam_) {
        LOG(INFO) << "SLAM frontend: LIO-SAM";
        lio_sam_ = std::make_shared<LioSamMapping>();
        if (!lio_sam_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init lio-sam module";
            return false;
        }
    } else {
        LOG(INFO) << "SLAM frontend: Faster-LIO";
        lio_ = std::make_shared<LaserMapping>();
        if (!lio_->Init(yaml_path)) {
            LOG(ERROR) << "failed to init lio module";
            return false;
        }
    }

    options_.with_loop_closing_ = yaml["system"]["with_loop_closing"].as<bool>();
    options_.with_visualization_ = yaml["system"]["with_ui"].as<bool>();
    options_.with_2dvisualization_ = yaml["system"]["with_2dui"].as<bool>();
    options_.with_gridmap_ = yaml["system"]["with_g2p5"].as<bool>();
    options_.step_on_kf_ = yaml["system"]["step_on_kf"].as<bool>();

    if (options_.with_loop_closing_) {
        LOG(INFO) << "slam with loop closing";
        LoopClosing::Options options;
        options.online_mode_ = options_.online_mode_;
        lc_ = std::make_shared<LoopClosing>(options);
        lc_->Init(yaml_path);
    }

    if (options_.with_visualization_) {
        LOG(INFO) << "slam with 3D UI";
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->Init();
        if (use_lio_sam_) {
            lio_sam_->SetUI(ui_);
        } else {
            lio_->SetUI(ui_);
        }
    }

    if (options_.with_gridmap_) {
        g2p5::G2P5::Options opt;
        opt.online_mode_ = options_.online_mode_;

        g2p5_ = std::make_shared<g2p5::G2P5>(opt);
        g2p5_->Init(yaml_path);

        if (options_.with_loop_closing_) {
            lc_->SetLoopClosedCB([this]() { g2p5_->RedrawGlobalMap(); });
        }

        if (options_.with_2dvisualization_) {
            g2p5_->SetMapUpdateCallback([this](g2p5::G2P5MapPtr map) {
                cv::Mat image = map->ToCV();
                cv::imshow("map", image);
                cv::waitKey(options_.step_on_kf_ ? 0 : 10);
            });
        }
    }

    if (options_.online_mode_) {
        LOG(INFO) << "online mode, creating ros2 node ... ";

        /// subscribers
        node_ = std::make_shared<rclcpp::Node>("lightning_slam");

        imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
        cloud_topic_ = yaml["common"]["lidar_topic"].as<std::string>();
        livox_topic_ = yaml["common"]["livox_lidar_topic"].as<std::string>();

        rclcpp::QoS qos(10);
        // qos.best_effort();

        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) { ProcessIMU(msg); });

        cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });

        livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
                Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
            });

        savemap_service_ = node_->create_service<SaveMapService>(
            "lightning/save_map", [this](const SaveMapService::Request::SharedPtr& req,
                                         SaveMapService::Response::SharedPtr res) { SaveMap(req, res); });

        LOG(INFO) << "online slam node has been created.";
    }

    return true;
}

SlamSystem::~SlamSystem() {
    if (ui_) {
        ui_->Quit();
    }
}

void SlamSystem::StartSLAM(std::string map_name) {
    map_name_ = map_name;
    running_ = true;
}

void SlamSystem::SaveMap(const SaveMapService::Request::SharedPtr request,
                         SaveMapService::Response::SharedPtr response) {
    map_name_ = request->map_id;
    std::string save_path = "./data/" + map_name_ + "/";

    SaveMap(save_path);
    response->response = 0;
}

void SlamSystem::SaveMap(const std::string& path) {
    std::string save_path = path;
    if (save_path.empty()) {
        save_path = "./data/" + map_name_ + "/";
    }

    LOG(INFO) << "slam map saving to " << save_path;

    if (!std::filesystem::exists(save_path)) {
        std::filesystem::create_directories(save_path);
    } else {
        std::filesystem::remove_all(save_path);
        std::filesystem::create_directories(save_path);
    }

    std::vector<Keyframe::Ptr> keyframes = use_lio_sam_ ? lio_sam_->GetAllKeyframes() : lio_->GetAllKeyframes();
    if (keyframes.empty()) {
        LOG(WARNING) << "no keyframes, skip map saving";
        return;
    }

    auto global_map =
        use_lio_sam_ ? lio_sam_->GetGlobalMap(!options_.with_loop_closing_) : lio_->GetGlobalMap(!options_.with_loop_closing_);

    TiledMap::Options tm_options;
    tm_options.map_path_ = save_path;

    TiledMap tm(tm_options);
    SE3 start_pose = keyframes.front()->GetOptPose();
    tm.ConvertFromFullPCD(global_map, start_pose, save_path);

    pcl::io::savePCDFileBinaryCompressed(save_path + "/global.pcd", *global_map);

    if (options_.with_gridmap_ && g2p5_) {
        auto map = g2p5_->GetNewestMap()->ToROS();
        const int width = map.info.width;
        const int height = map.info.height;

        cv::Mat nav_image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            const int rowStartIndex = y * width;
            for (int x = 0; x < width; ++x) {
                const int index = rowStartIndex + x;
                int8_t data = map.data[index];
                if (data == 0) {                                   // Free
                    nav_image.at<uchar>(height - 1 - y, x) = 255;  // White
                } else if (data == 100) {                          // Occupied
                    nav_image.at<uchar>(height - 1 - y, x) = 0;    // Black
                } else {                                           // Unknown
                    nav_image.at<uchar>(height - 1 - y, x) = 128;  // Gray
                }
            }
        }

        cv::imwrite(save_path + "/map.pgm", nav_image);

        /// yaml
        std::ofstream yamlFile(save_path + "/map.yaml");
        if (!yamlFile.is_open()) {
            LOG(ERROR) << "failed to write map.yaml";
            return;  // 文件打开失败
        }

        YAML::Emitter emitter;
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "image" << YAML::Value << "map.pgm";
        emitter << YAML::Key << "mode" << YAML::Value << "trinary";
        emitter << YAML::Key << "width" << YAML::Value << map.info.width;
        emitter << YAML::Key << "height" << YAML::Value << map.info.height;
        emitter << YAML::Key << "resolution" << YAML::Value << float(0.05);
        std::vector<double> orig{map.info.origin.position.x, map.info.origin.position.y, 0};
        emitter << YAML::Key << "origin" << YAML::Value << orig;
        emitter << YAML::Key << "negate" << YAML::Value << 0;
        emitter << YAML::Key << "occupied_thresh" << YAML::Value << 0.65;
        emitter << YAML::Key << "free_thresh" << YAML::Value << 0.25;
        emitter << YAML::EndMap;

        yamlFile << emitter.c_str();
    }

    LOG(INFO) << "map saved";
}

void SlamSystem::ProcessIMU(const sensor_msgs::msg::Imu::SharedPtr& imu) {
    if (running_ == false) {
        return;
    }

    if (use_lio_sam_) {
        lio_sam_->ProcessIMU(imu);
    } else {
        lio_->ProcessIMU(ConvertRosImu(imu));
    }
}

void SlamSystem::ProcessIMU(const lightning::IMUPtr& imu) {
    if (running_ == false) {
        return;
    }

    if (use_lio_sam_) {
        static bool warned = false;
        if (!warned) {
            LOG(WARNING) << "LIO-SAM frontend needs raw sensor_msgs::msg::Imu; converted IMUPtr is ignored";
            warned = true;
        }
        return;
    }

    lio_->ProcessIMU(imu);
}

void SlamSystem::PublishKeyframeToBackends(const Keyframe::Ptr& kf) {
    if (kf == cur_kf_) {
        return;
    }
    cur_kf_ = kf;

    if (cur_kf_ == nullptr) {
        return;
    }

    if (options_.with_loop_closing_ && lc_) {
        lc_->AddKF(cur_kf_);
    }

    if (options_.with_gridmap_ && g2p5_) {
        g2p5_->PushKeyframe(cur_kf_);
    }

    if (ui_) {
        ui_->UpdateKF(cur_kf_);
    }
}

void SlamSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }

    if (use_lio_sam_) {
        lio_sam_->ProcessPointCloud2(cloud);
        lio_sam_->Run();
        PublishKeyframeToBackends(lio_sam_->GetKeyframe());
        return;
    }

    lio_->ProcessPointCloud2(cloud);
    lio_->Run();
    PublishKeyframeToBackends(lio_->GetKeyframe());
}

void SlamSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }

    if (use_lio_sam_) {
        lio_sam_->ProcessPointCloud2(cloud);
        lio_sam_->Run();
        PublishKeyframeToBackends(lio_sam_->GetKeyframe());
        return;
    }

    lio_->ProcessPointCloud2(cloud);
    lio_->Run();
    PublishKeyframeToBackends(lio_->GetKeyframe());
}

void SlamSystem::Spin() {
    if (options_.online_mode_ && node_ != nullptr) {
        spin(node_);
    }
}

}  // namespace lightning
