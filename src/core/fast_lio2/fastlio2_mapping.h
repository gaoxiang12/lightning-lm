// FAST-LIO2 适配器
// 实现 LIOFrontend 接口，封装 FASTLIO2Core
// 处理数据同步、点类型转换、ROS2 消息适配

#ifndef LIGHTNING_FASTLIO2_MAPPING_H
#define LIGHTNING_FASTLIO2_MAPPING_H

#include <memory>
#include <vector>

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"
#include "core/frontend/lio_frontend.h"

namespace lightning {

class FASTLIO2Core;

/**
 * FAST-LIO2 适配器
 * 继承 LIOFrontend 接口，支持多态调用
 * 公共功能（预处理、缓冲区管理、数据同步）由基类 LIOFrontend 实现
 */
class FASTLIO2Mapping : public LIOFrontend {
public:
    struct NativeState {
        Vec3d pos;
        Mat3d rot;
        Mat3d offset_R_L_I;
        Vec3d offset_T_L_I;
        Vec3d vel;
        Vec3d bg;
        Vec3d ba;
        Vec3d grav;
    };

    FASTLIO2Mapping();
    ~FASTLIO2Mapping() override;

    // ====== LIOFrontend 接口实现 ======

    bool Init(const std::string& config_yaml) override;
    bool Run() override;

    SE3 GetPose() const override;
    CloudPtr GetScanUndist() const override;
    CloudPtr GetScanDownWorld() const override;
    Keyframe::Ptr GetKeyframe() const override;
    std::vector<Keyframe::Ptr> GetAllKeyframes() override;

    void SetUI(std::shared_ptr<ui::PangolinWindow> ui) override { ui_ = ui; }
    void SaveMap() override;
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) override;
    void PrintExtrinsic() override;
    bool IsExtrinsicEstEnabled() const override { return true; }  // FAST-LIO2 外参始终在状态中
    Vec3d GetExtrinsicT() const override { return native_state_.offset_T_L_I; }
    Mat3d GetExtrinsicR() const override { return native_state_.offset_R_L_I; }

    NativeState GetNativeState() const;

private:
    void DownSample();
    void MakeKF();

    std::shared_ptr<FASTLIO2Core> core_;

    // 输出
    CloudPtr scan_down_body_{new PointCloudType()};
    CloudPtr scan_down_world_{new PointCloudType()};
    NativeState native_state_;
    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;

    // 关键帧参数
    double kf_dis_th_ = 2.0;
    double kf_angle_th_ = 15.0 * M_PI / 180.0;
};

}  // namespace lightning

#endif  // LIGHTNING_FASTLIO2_MAPPING_H
