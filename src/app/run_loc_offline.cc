//
// Created by xiang on 25-3-18.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core/localization/localization.h"
#include "ui/pangolin_window.h"
#include "utils/timer.h"
#include "wrapper/bag_io.h"
#include "wrapper/ros_utils.h"

#include "io/yaml_io.h"

DEFINE_string(input_bag, "", "输入数据包");
DEFINE_string(config, "./config/default.yaml", "配置文件");
DEFINE_string(map_path, "./data/new_map/", "地图路径");

/// 运行定位的测试
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

    loc::Localization::Options options;
    options.online_mode_ = false;

    loc::Localization loc(options);
    loc.Init(FLAGS_config, FLAGS_map_path);

    lightning::YAML_IO yaml(FLAGS_config);
    std::string lidar_topic = yaml.GetValue<std::string>("common", "lidar_topic");
    std::string imu_topic = yaml.GetValue<std::string>("common", "imu_topic");
    std::string livox_topic = yaml.GetValue<std::string>("common", "livox_lidar_topic");
    rosbag
    /*
        .AddImuHandle(imu_topic,
                      [&loc](IMUPtr imu) {
                          loc.ProcessIMUMsg(imu);
                          usleep(1000);
                          return true;
                      })
    */
        .AddImuHandle(imu_topic,
            [&loc](sensor_msgs::msg::Imu::SharedPtr msg) {
                if (!msg) {
                    return true;
                }

                IMUPtr imu = std::make_shared<IMU>();
                imu->timestamp = ToSec(msg->header.stamp);
                imu->angular_velocity =Vec3d(msg->angular_velocity.x,msg->angular_velocity.y,msg->angular_velocity.z);
                    
                imu->linear_acceleration =Vec3d(msg->linear_acceleration.x,msg->linear_acceleration.y,msg->linear_acceleration.z);
                imu->orientation =Quatd(msg->orientation.w,msg->orientation.x,msg->orientation.y,msg->orientation.z);

                loc.ProcessIMUMsg(imu);
                usleep(1000);
                return true;
            })
        .AddPointCloud2Handle(lidar_topic,
            [&loc](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
                loc.ProcessLidarMsg(cloud);
                usleep(1000);
                return true;
            })
        .AddLivoxCloudHandle(livox_topic,
                             [&loc](livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
                                 loc.ProcessLivoxLidarMsg(cloud);
                                 usleep(1000);
                                 return true;
                             })
        .Go();

    Timer::PrintAll();
    loc.Finish();

    LOG(INFO) << "done";

    return 0;
}