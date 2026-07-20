#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <limits>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "core/frontend/pointcloud_preprocess.h"
#include "core/localization/lidar_loc/lidar_loc.h"
#include "core/localization/localization_result.h"
#include "core/localization/pose_graph/pgo.h"
#include "io/yaml_io.h"

namespace {

sensor_msgs::msg::PointCloud2::SharedPtr MakeFusedCloud() {
    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    msg->header.frame_id = "base_link";
    msg->header.stamp.sec = 100;
    sensor_msgs::PointCloud2Modifier modifier(*msg);
    modifier.setPointCloud2Fields(7, "x", 1, sensor_msgs::msg::PointField::FLOAT32, "y", 1,
                                  sensor_msgs::msg::PointField::FLOAT32, "z", 1,
                                  sensor_msgs::msg::PointField::FLOAT32, "intensity", 1,
                                  sensor_msgs::msg::PointField::FLOAT32, "tag", 1,
                                  sensor_msgs::msg::PointField::UINT8, "line", 1,
                                  sensor_msgs::msg::PointField::UINT8, "timestamp", 1,
                                  sensor_msgs::msg::PointField::FLOAT64);
    modifier.resize(2);
    sensor_msgs::PointCloud2Iterator<float> x(*msg, "x"), y(*msg, "y"), z(*msg, "z"), intensity(*msg, "intensity");
    sensor_msgs::PointCloud2Iterator<uint8_t> tag(*msg, "tag"), line(*msg, "line");
    sensor_msgs::PointCloud2Iterator<double> timestamp(*msg, "timestamp");
    for (int i = 0; i < 2; ++i, ++x, ++y, ++z, ++intensity, ++tag, ++line, ++timestamp) {
        *x = 1.0F + i;
        *y = 0.0F;
        *z = 0.1F;
        *intensity = 20.0F + i;
        *tag = 0;
        *line = static_cast<uint8_t>(i);
        *timestamp = 100000000000.0 + i * 1000000.0;
    }
    return msg;
}

TEST(PointCloudPreprocess, ReadsFusedMid360Layout) {
    lightning::PointCloudPreprocess preprocess;
    preprocess.Set(lightning::LidarType::ROBOSENSE, 0.1, 1);
    preprocess.SetHeightROI(2.0, -2.0);
    auto output = std::make_shared<lightning::PointCloudType>();
    preprocess.Process(MakeFusedCloud(), output);

    ASSERT_EQ(output->size(), 2U);
    EXPECT_FLOAT_EQ(output->at(1).x, 2.0F);
    EXPECT_FLOAT_EQ(output->at(1).intensity, 21.0F);
    EXPECT_NEAR(output->at(1).time, 1.0, 1e-6);
}

TEST(LidarLocInitialization, EnforcesConvergenceConfidenceFiniteAndDistanceGates) {
    const lightning::SE3 candidate(lightning::SO3(), lightning::Vec3d(0.0, 0.0, 0.0));
    const lightning::SE3 inside(lightning::SO3(), lightning::Vec3d(3.0, 4.0, 0.0));
    const lightning::SE3 outside(lightning::SO3(), lightning::Vec3d(3.01, 4.0, 0.0));
    const lightning::SE3 nonfinite(
        lightning::SO3(),
        lightning::Vec3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));

    EXPECT_TRUE(lightning::loc::IsInitializationResultValid(
        true, candidate, inside, 1.8, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        false, candidate, inside, 2.0, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, inside, 1.79, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, outside, 2.0, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, nonfinite, 2.0, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, inside, std::numeric_limits<double>::infinity(), 1.8, 5.0));
}

TEST(LidarLocMatcher, RejectsNonConvergedAndNonFiniteNdtOutput) {
    const Eigen::Matrix4f valid = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f invalid = valid;
    invalid(0, 3) = std::numeric_limits<float>::quiet_NaN();

    EXPECT_TRUE(lightning::loc::IsNdtResultValid(true, valid, 2.0));
    EXPECT_FALSE(lightning::loc::IsNdtResultValid(false, valid, 2.0));
    EXPECT_FALSE(lightning::loc::IsNdtResultValid(true, invalid, 2.0));
    EXPECT_FALSE(lightning::loc::IsNdtResultValid(
        true, valid, std::numeric_limits<double>::infinity()));
}

TEST(LidarLocTracking, FailedMatcherIsNotAValidFiniteMeasurement) {
    lightning::loc::LocalizationResult result;

    lightning::loc::SetTrackingResultState(
        result, false, std::numeric_limits<double>::infinity(), 1);

    EXPECT_FALSE(result.lidar_loc_valid_);
    EXPECT_TRUE(std::isfinite(result.confidence_));
    EXPECT_DOUBLE_EQ(result.confidence_, 0.0);
    EXPECT_EQ(result.status_, lightning::loc::LocalizationStatus::GOOD);
}

TEST(YamlIo, UsesDefaultOnlyWhenValueIsMissing) {
    const std::string path = "/tmp/lightning_yaml_optional_value_test.yaml";
    {
        std::ofstream out(path);
        out << "lidar_loc:\n  configured_distance: 3.5\n";
    }
    const lightning::YAML_IO yaml(path);

    EXPECT_FLOAT_EQ(yaml.GetValueOr<float>("lidar_loc", "max_init_distance", 5.0F), 5.0F);
    EXPECT_FLOAT_EQ(yaml.GetValueOr<float>("lidar_loc", "configured_distance", 5.0F), 3.5F);

    std::remove(path.c_str());
}

}  // namespace

namespace lightning::loc {

TEST(LidarLocTargetSerialization, BackgroundCannotReplaceTargetBetweenRoughAndFine) {
    LidarLoc loc;
    std::vector<int> events;
    std::promise<void> rough_started;
    std::promise<void> allow_fine;
    auto allow_fine_future = allow_fine.get_future();
    auto candidate = std::async(std::launch::async, [&] {
        loc.WithCandidateTargetLock([&] {
            events.push_back(1);
            rough_started.set_value();
            allow_fine_future.wait();
            events.push_back(2);
        });
    });
    rough_started.get_future().wait();

    std::promise<void> background_attempted;
    auto background = std::async(std::launch::async, [&] {
        background_attempted.set_value();
        loc.WithCandidateTargetLock([&] { events.push_back(3); });
    });
    background_attempted.get_future().wait();

    EXPECT_EQ(background.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    allow_fine.set_value();
    candidate.get();
    background.get();
    EXPECT_EQ(events, (std::vector<int>{1, 2, 3}));
}

}  // namespace lightning::loc

namespace {

TEST(LocalizationResult, ProducesMapOdomBaseChain) {
    lightning::loc::LocalizationResult result;
    result.timestamp_ = 12.5;
    result.pose_ = lightning::SE3(lightning::SO3(), lightning::Vec3d(10.0, 2.0, 0.0));
    result.rel_pose_ = lightning::SE3(lightning::SO3(), lightning::Vec3d(3.0, -1.0, 0.0));
    result.rel_pose_set_ = true;
    result.vel_b_ = lightning::Vec3d(0.2, -0.1, 0.0);

    const auto map_odom = result.ToMapOdomMsg();
    const auto odom_base = result.ToOdomBaseMsg();
    const auto odom = result.ToOdomMsg();
    const auto pose = result.ToPoseMsg();

    EXPECT_EQ(map_odom.header.frame_id, "map");
    EXPECT_EQ(map_odom.child_frame_id, "odom");
    EXPECT_EQ(odom_base.header.frame_id, "odom");
    EXPECT_EQ(odom_base.child_frame_id, "base_link");
    EXPECT_NEAR(map_odom.transform.translation.x + odom_base.transform.translation.x, 10.0, 1e-9);
    EXPECT_NEAR(map_odom.transform.translation.y + odom_base.transform.translation.y, 2.0, 1e-9);
    EXPECT_EQ(odom.header.frame_id, "odom");
    EXPECT_EQ(odom.child_frame_id, "base_link");
    EXPECT_DOUBLE_EQ(odom.twist.twist.linear.x, 0.2);
    EXPECT_DOUBLE_EQ(odom.twist.twist.linear.y, -0.1);
    EXPECT_EQ(pose.header.frame_id, "map");
}

TEST(PGO, PreservesLocalizationStatus) {
    lightning::loc::PGO pgo;
    lightning::loc::LocalizationResult output;
    int publish_count = 0;
    pgo.SetHighFrequencyGlobalOutputHandleFunction(
        [&](const lightning::loc::LocalizationResult& result) {
            output = result;
            ++publish_count;
        });

    lightning::NavState odom;
    odom.timestamp_ = 0.9;
    odom.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(2.9, 0.0, 0.0)));
    pgo.ProcessLidarOdom(odom);
    pgo.ProcessDR(odom);
    odom.timestamp_ = 1.0;
    odom.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(3.0, 0.0, 0.0)));
    pgo.ProcessLidarOdom(odom);
    pgo.ProcessDR(odom);

    lightning::loc::LocalizationResult good;
    good.timestamp_ = 1.0;
    good.pose_ = lightning::SE3(lightning::SO3(), lightning::Vec3d(10.0, 0.0, 0.0));
    good.valid_ = true;
    good.lidar_loc_valid_ = true;
    good.status_ = lightning::loc::LocalizationStatus::GOOD;
    ASSERT_TRUE(pgo.ProcessLidarLoc(good));
    ASSERT_GT(publish_count, 0);
    EXPECT_EQ(output.status_, lightning::loc::LocalizationStatus::GOOD);

    lightning::loc::LocalizationResult failed = good;
    failed.timestamp_ = 1.1;
    failed.lidar_loc_valid_ = false;
    failed.status_ = lightning::loc::LocalizationStatus::FAIL;
    EXPECT_FALSE(pgo.ProcessLidarLoc(failed));
    EXPECT_EQ(output.status_, lightning::loc::LocalizationStatus::FAIL);
}

TEST(PGO, RejectsNonFiniteLocalizationConfidence) {
    lightning::loc::PGO pgo;
    lightning::loc::LocalizationResult output;
    int publish_count = 0;
    pgo.SetHighFrequencyGlobalOutputHandleFunction(
        [&](const lightning::loc::LocalizationResult& result) {
            output = result;
            ++publish_count;
        });
    lightning::NavState odom;
    odom.timestamp_ = 0.9;
    odom.SetPose(lightning::SE3());
    pgo.ProcessLidarOdom(odom);
    pgo.ProcessDR(odom);
    odom.timestamp_ = 1.0;
    pgo.ProcessLidarOdom(odom);
    pgo.ProcessDR(odom);

    lightning::loc::LocalizationResult loc;
    loc.timestamp_ = 1.0;
    loc.lidar_loc_valid_ = true;
    loc.confidence_ = 1.0;
    loc.status_ = lightning::loc::LocalizationStatus::GOOD;
    ASSERT_TRUE(pgo.ProcessLidarLoc(loc));
    ASSERT_TRUE(output.lidar_loc_valid_);

    loc.timestamp_ = 1.1;
    loc.confidence_ = std::numeric_limits<double>::infinity();
    loc.status_ = lightning::loc::LocalizationStatus::FOLLOWING_DR;
    const int count_before_rejection = publish_count;

    EXPECT_FALSE(pgo.ProcessLidarLoc(loc));
    EXPECT_GT(publish_count, count_before_rejection);
    EXPECT_FALSE(output.lidar_loc_valid_);
    EXPECT_TRUE(std::isfinite(output.confidence_));
    EXPECT_DOUBLE_EQ(output.confidence_, 0.0);
    EXPECT_EQ(output.status_, lightning::loc::LocalizationStatus::FOLLOWING_DR);
}

TEST(PGO, KeepsRelativePoseAtPublishedTimestamp) {
    lightning::loc::PGO pgo;
    lightning::loc::LocalizationResult output;
    pgo.SetHighFrequencyGlobalOutputHandleFunction(
        [&](const lightning::loc::LocalizationResult& result) { output = result; });

    lightning::NavState state;
    state.timestamp_ = 0.9;
    state.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(2.9, 0.0, 0.0)));
    pgo.ProcessLidarOdom(state);
    pgo.ProcessDR(state);
    state.timestamp_ = 1.0;
    state.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(3.0, 0.0, 0.0)));
    pgo.ProcessLidarOdom(state);
    pgo.ProcessDR(state);

    lightning::loc::LocalizationResult loc;
    loc.timestamp_ = 1.0;
    loc.pose_ = lightning::SE3(lightning::SO3(), lightning::Vec3d(10.0, 0.0, 0.0));
    loc.valid_ = true;
    loc.lidar_loc_valid_ = true;
    loc.status_ = lightning::loc::LocalizationStatus::GOOD;
    ASSERT_TRUE(pgo.ProcessLidarLoc(loc));

    state.timestamp_ = 2.0;
    state.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(4.0, 0.0, 0.0)));
    state.SetVel(lightning::Vec3d(0.5, 0.0, 0.0));
    ASSERT_TRUE(pgo.ProcessDR(state));

    EXPECT_DOUBLE_EQ(output.timestamp_, 2.0);
    EXPECT_NEAR(output.rel_pose_.translation().x(), 4.0, 1e-9);
    EXPECT_NEAR(output.vel_b_.x(), 0.5, 1e-9);
    EXPECT_NEAR((output.pose_ * output.rel_pose_.inverse() * output.rel_pose_).translation().x(),
                output.pose_.translation().x(), 1e-9);
}

TEST(PGO, ResetDropsPreviousLocalization) {
    lightning::loc::PGO pgo;
    int publish_count = 0;
    pgo.SetHighFrequencyGlobalOutputHandleFunction(
        [&](const lightning::loc::LocalizationResult&) { ++publish_count; });

    lightning::NavState state;
    state.timestamp_ = 0.9;
    state.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(2.9, 0.0, 0.0)));
    pgo.ProcessLidarOdom(state);
    pgo.ProcessDR(state);
    state.timestamp_ = 1.0;
    state.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(3.0, 0.0, 0.0)));
    pgo.ProcessLidarOdom(state);
    pgo.ProcessDR(state);

    lightning::loc::LocalizationResult loc;
    loc.timestamp_ = 1.0;
    loc.pose_ = lightning::SE3(lightning::SO3(), lightning::Vec3d(10.0, 0.0, 0.0));
    loc.valid_ = true;
    loc.lidar_loc_valid_ = true;
    loc.status_ = lightning::loc::LocalizationStatus::GOOD;
    ASSERT_TRUE(pgo.ProcessLidarLoc(loc));
    ASSERT_GT(publish_count, 0);

    ASSERT_TRUE(pgo.Reset());
    const int count_after_reset = publish_count;
    state.timestamp_ = 2.0;
    state.SetPose(lightning::SE3(lightning::SO3(), lightning::Vec3d(4.0, 0.0, 0.0)));
    ASSERT_TRUE(pgo.ProcessDR(state));
    EXPECT_EQ(publish_count, count_after_reset);
}

}  // namespace
