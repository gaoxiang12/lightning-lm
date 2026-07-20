//
// Created by xiang on 24-4-11.
//

#include "localization_result.h"
#include "core/lightning_math.hpp"

namespace lightning::loc {

namespace {

void FillTransform(const SE3& pose, geometry_msgs::msg::Transform& transform) {
    transform.translation.x = pose.translation().x();
    transform.translation.y = pose.translation().y();
    transform.translation.z = pose.translation().z();
    transform.rotation.x = pose.so3().unit_quaternion().x();
    transform.rotation.y = pose.so3().unit_quaternion().y();
    transform.rotation.z = pose.so3().unit_quaternion().z();
    transform.rotation.w = pose.so3().unit_quaternion().w();
}

void FillPose(const SE3& pose, geometry_msgs::msg::Pose& msg) {
    msg.position.x = pose.translation().x();
    msg.position.y = pose.translation().y();
    msg.position.z = pose.translation().z();
    msg.orientation.x = pose.so3().unit_quaternion().x();
    msg.orientation.y = pose.so3().unit_quaternion().y();
    msg.orientation.z = pose.so3().unit_quaternion().z();
    msg.orientation.w = pose.so3().unit_quaternion().w();
}

}  // namespace

geometry_msgs::msg::TransformStamped LocalizationResult::ToMapOdomMsg() const {
    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = "map";
    msg.header.stamp = math::FromSec(timestamp_);
    msg.child_frame_id = "odom";
    FillTransform(pose_ * rel_pose_.inverse(), msg.transform);
    return msg;
}

geometry_msgs::msg::TransformStamped LocalizationResult::ToOdomBaseMsg() const {
    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = "odom";
    msg.header.stamp = math::FromSec(timestamp_);
    msg.child_frame_id = "base_link";
    FillTransform(rel_pose_, msg.transform);
    return msg;
}

nav_msgs::msg::Odometry LocalizationResult::ToOdomMsg() const {
    nav_msgs::msg::Odometry msg;
    msg.header.frame_id = "odom";
    msg.header.stamp = math::FromSec(timestamp_);
    msg.child_frame_id = "base_link";
    FillPose(rel_pose_, msg.pose.pose);
    msg.twist.twist.linear.x = vel_b_.x();
    msg.twist.twist.linear.y = vel_b_.y();
    msg.twist.twist.linear.z = vel_b_.z();
    return msg;
}

geometry_msgs::msg::PoseStamped LocalizationResult::ToPoseMsg() const {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id = "map";
    msg.header.stamp = math::FromSec(timestamp_);
    FillPose(pose_, msg.pose);
    return msg;
}

NavState LocalizationResult::ToNavState() const {
    NavState ret;
    ret.timestamp_ = timestamp_;
    ret.confidence_ = confidence_;
    ret.pos_ = pose_.translation();
    ret.rot_ = pose_.so3();
    ret.pose_is_ok_ = status_ == LocalizationStatus::GOOD;

    ret.vel_ = (pose_.so3() * vel_b_);

    return ret;
}

}  // namespace lightning::loc
