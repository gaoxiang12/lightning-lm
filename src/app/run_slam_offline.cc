//
// Created by xiang on 25-3-18.
//

#include <chrono>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core/system/slam.h"
#include "ui/pangolin_window.h"
#include "utils/timer.h"
#include "wrapper/bag_io.h"
#include "wrapper/ros_utils.h"

#include "io/yaml_io.h"
#include <pangolin/pangolin.h>

DEFINE_string(input_bag, "", "输入数据包");
DEFINE_string(config, "./config/default.yaml", "配置文件");

/// 运行一个LIO前端，带可视化
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = google::INFO;

    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_input_bag.empty()) {
        LOG(ERROR) << "未指定输入数据";
        return -1;
    }

    using namespace lightning;

    RosbagIO rosbag(FLAGS_input_bag);

    SlamSystem::Options options;
    options.online_mode_ = false;

    SlamSystem slam(options);

    /// 实时模式好像掉帧掉的比较厉害？

    if (!slam.Init(FLAGS_config)) {
        LOG(ERROR) << "failed to init slam";
        return -1;
    }

    slam.StartSLAM("new_map");

    lightning::YAML_IO yaml(FLAGS_config);
    std::string lidar_topic = yaml.GetValue<std::string>("common", "lidar_topic");
    std::string imu_topic = yaml.GetValue<std::string>("common", "imu_topic");

    rosbag
        /// IMU 的处理
        .AddImuHandle(imu_topic,
                      [&slam](IMUPtr imu) {
                          slam.ProcessIMU(imu);
                          return true;
                      })

        /// lidar 的处理
        .AddPointCloud2Handle(lidar_topic,
                              [&slam](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                                  slam.ProcessLidar(msg);
                                  return true;
                              })
        /// livox 的处理
        .AddLivoxCloudHandle("/livox/lidar",
                             [&slam](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
                                 slam.ProcessLidar(cloud);
                                 return true;
                             })
        .Go();

    slam.SaveMap("");
    slam.PrintExtrinsic();
    Timer::PrintAll();

    LOG(INFO) << "done. Press ESC or close the window to exit.";

    while (!pangolin::ShouldQuit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}