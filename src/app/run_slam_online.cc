//
// Created by xiang on 25-3-18.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core/system/slam.h"
#include "utils/timer.h"
#include "wrapper/bag_io.h"
#include "wrapper/ros_utils.h"

DEFINE_string(config, "./config/default.yaml", "配置文件");
DEFINE_string(output_map, "./data/new_map", "退出时保存地图的目录");

/// 运行一个LIO前端，带可视化
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = google::INFO;
    auto args = rclcpp::init_and_remove_ros_arguments(argc, argv);
    std::vector<char*> gflags_argv;
    for (auto& arg : args) {
        gflags_argv.push_back(arg.data());
    }
    int gflags_argc = static_cast<int>(gflags_argv.size());
    char** gflags_argv_data = gflags_argv.data();
    google::ParseCommandLineFlags(&gflags_argc, &gflags_argv_data, true);

    using namespace lightning;

    SlamSystem::Options options;
    options.online_mode_ = true;

    SlamSystem slam(options);
    if (!slam.Init(FLAGS_config)) {
        LOG(ERROR) << "failed to init slam";
        return -1;
    }

    slam.StartSLAM("new_map");
    slam.Spin();

    slam.SaveMap(FLAGS_output_map);

    slam.PrintExtrinsic();
    Timer::PrintAll();

    rclcpp::shutdown();

    LOG(INFO) << "done";

    return 0;
}
