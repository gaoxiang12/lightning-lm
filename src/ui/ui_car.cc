#include "ui/ui_car.h"
#include <GL/gl.h>

namespace lightning::ui {

std::vector<Vec3f> UiCar::car_vertices_ = {
    // clang-format off
     { 0, 0, 0}, { 3.0, 0, 0},
     { 0, 0, 0}, { 0, 3.0, 0},
     { 0, 0, 0}, { 0, 0, 3.0},
    // clang-format on
};

void UiCar::SetPose(const SE3& pose) {
    pts_.clear();
    for (auto& p : car_vertices_) {
        pts_.emplace_back(p);
    }

    // 转换到世界系
    auto pose_f = pose.cast<float>();
    for (auto& pt : pts_) {
        pt = pose_f * pt;
    }
}

void UiCar::Render() {
    glLineWidth(5.0);
    glBegin(GL_LINES);

    /// x -红, y-绿 z-蓝
    glColor3f(color_[0], color_[1], color_[2]);
    glVertex3f(pts_[0][0], pts_[0][1], pts_[0][2]);
    glVertex3f(pts_[1][0], pts_[1][1], pts_[1][2]);

    // glColor3f(0.0, 1.0, 0.0);
    glVertex3f(pts_[2][0], pts_[2][1], pts_[2][2]);
    glVertex3f(pts_[3][0], pts_[3][1], pts_[3][2]);

    // glColor3f(0.0, 0.0, 1.0);
    glVertex3f(pts_[4][0], pts_[4][1], pts_[4][2]);
    glVertex3f(pts_[5][0], pts_[5][1], pts_[5][2]);
    glEnd();
}

}  // namespace lightning::ui
