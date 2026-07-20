//
// Created by xiang on 25-3-18.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core/system/loc_system.h"
#include "ui/pangolin_window.h"
#include "wrapper/ros_utils.h"

DEFINE_string(config, "./config/default.yaml", "配置文件");

/// 运行定位的测试
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

    LocSystem::Options opt;
    LocSystem loc(opt);

    if (!loc.Init(FLAGS_config)) {
        LOG(ERROR) << "failed to init loc";
    }

    loc.Spin();

    rclcpp::shutdown();

    return 0;
}
