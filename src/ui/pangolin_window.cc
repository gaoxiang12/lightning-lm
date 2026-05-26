#include "ui/pangolin_window_impl.h"

namespace lightning::ui {

PangolinWindow::PangolinWindow() { impl_ = std::make_shared<PangolinWindowImpl>(); }
PangolinWindow::~PangolinWindow() { Quit(); }

bool PangolinWindow::Init() {
    impl_->cloud_global_need_update_.store(false);
    impl_->kf_result_need_update_.store(false);
    impl_->lidarloc_need_update_.store(false);
    impl_->current_scan_need_update_.store(false);

    bool inited = impl_->Init();
    // 创建渲染线程
    if (inited) {
        impl_->render_thread_ = std::thread([this]() { impl_->Render(); });
    }
    return inited;
}

void PangolinWindow::Reset(const std::vector<Keyframe::Ptr>& keyframes) { impl_->Reset(keyframes); }

void PangolinWindow::Quit() {
    if (impl_->render_thread_.joinable()) {
        impl_->exit_flag_.store(true);
        impl_->render_thread_.join();
    }
    impl_->DeInit();
}

void PangolinWindow::UpdatePointCloudGlobal(const std::map<int, CloudPtr>& cloud) {
    std::lock_guard<std::mutex> lock(impl_->mtx_map_cloud_);
    impl_->cloud_global_map_ = cloud;
    impl_->cloud_global_need_update_.store(true);
}

void PangolinWindow::UpdatePointCloudDynamic(const std::map<int, CloudPtr>& cloud) {
    std::unique_lock<std::mutex> lock(impl_->mtx_map_cloud_);
    impl_->cloud_dynamic_map_.clear();  // need deep copy

    for (auto& cp : cloud) {
        CloudPtr c(new PointCloudType());
        *c = *cp.second;
        impl_->cloud_dynamic_map_.emplace(cp.first, c);
    }

    for (auto iter = impl_->cloud_dynamic_map_.begin(); iter != impl_->cloud_dynamic_map_.end();) {
        if (cloud.find(iter->first) == cloud.end()) {
            iter = impl_->cloud_dynamic_map_.erase(iter);
        } else {
            iter++;
        }
    }

    impl_->cloud_dynamic_need_update_.store(true);
}

void PangolinWindow::UpdateNavState(const NavState& state) {
    std::unique_lock<std::mutex> lock_lio_res(impl_->mtx_nav_state_);

    impl_->pose_ = state.GetPose();
    impl_->vel_ = state.GetVel();
    impl_->bias_acc_ = state.Getba();
    impl_->bias_gyr_ = state.Getbg();
    impl_->confidence_ = state.confidence_;

    impl_->kf_result_need_update_.store(true);
}

void PangolinWindow::UpdateRecentPose(const SE3& pose) {
    std::lock_guard<std::mutex> lock(impl_->mtx_nav_state_);
    impl_->newest_frontend_pose_ = pose;
}

void PangolinWindow::UpdatePredictPose(const SE3& pose) {
    UL lock(impl_->mtx_nav_state_);
    impl_->predicted_pose_ = pose;
}

void PangolinWindow::UpdateScan(CloudPtr cloud, const SE3& pose) {
    std::lock_guard<std::mutex> lock(impl_->mtx_current_scan_);
    std::lock_guard<std::mutex> lock2(impl_->mtx_nav_state_);

    *impl_->current_scan_ = *cloud;  // need deep copy
    impl_->current_scan_pose_ = pose;
    impl_->current_scan_need_update_.store(true);
}

void PangolinWindow::UpdateKF(std::shared_ptr<Keyframe> kf) {
    UL lock(impl_->mtx_current_scan_);
    impl_->all_keyframes_.emplace_back(kf);
}

void PangolinWindow::SetCurrentScanSize(int current_scan_size) { impl_->max_size_of_current_scan_ = current_scan_size; }

void PangolinWindow::SetTImuLidar(const SE3& T_imu_lidar) { impl_->T_imu_lidar_ = T_imu_lidar; }

bool PangolinWindow::ShouldQuit() { return pangolin::ShouldQuit(); }

}  // namespace lightning::ui
