//
// Created by xiang on 24-4-11.
//

#include "localization_result.h"
#include "core/lightning_math.hpp"

namespace lightning::loc {

geometry_msgs::msg::TransformStamped LocalizationResult::ToGeoMsg() const {
    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = "map";
    msg.header.stamp = math::FromSec(timestamp_);
    msg.child_frame_id = "laser_link";

    msg.transform.translation.x = pose_.translation().x();
    msg.transform.translation.y = pose_.translation().y();
    msg.transform.translation.z = pose_.translation().z();

    msg.transform.rotation.x = pose_.so3().unit_quaternion().x();
    msg.transform.rotation.y = pose_.so3().unit_quaternion().y();
    msg.transform.rotation.z = pose_.so3().unit_quaternion().z();
    msg.transform.rotation.w = pose_.so3().unit_quaternion().w();

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
