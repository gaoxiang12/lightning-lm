#include "ui/ui_cloud.h"
#include "common/options.h"

#include <GL/gl.h>
#include <numeric>

namespace lightning::ui {

std::vector<Vec4f> UiCloud::intensity_color_table_pcl_;

UiCloud::UiCloud(CloudPtr cloud) { SetCloud(cloud, SE3()); }

void UiCloud::SetCustomColor(Vec4f custom_color) { custom_color_ = custom_color; }

// 把输入的点云映射为opengl可以渲染的点云
void UiCloud::SetCloud(CloudPtr cloud, const SE3& pose) {
    if (intensity_color_table_pcl_.empty()) {
        BuildIntensityTable();
    }

    // assert(cloud != nullptr && cloud->empty() == false);
    xyz_data_.resize(cloud->size());
    color_data_pcl_.resize(cloud->size());
    color_data_intensity_.resize(cloud->size());
    color_data_height_.resize(cloud->size());
    color_data_gray_.resize(cloud->size());

    std::vector<int> idx(cloud->size());
    std::iota(idx.begin(), idx.end(), 0);  // 使用从0开始递增的整数填充idx

    SE3f pose_l = (pose).cast<float>();

    // 遍历所有点
    for (auto iter = idx.begin(); iter != idx.end(); iter++) {
        const int& id = *iter;
        const auto& pt = cloud->points[id];
        // 计算点的世界坐标
        auto pt_world = pose_l * cloud->points[id].getVector3fMap();
        xyz_data_[id] = Vec3f(pt_world.x(), pt_world.y(), pt_world.z());
        // 把intensity映射为颜色
        color_data_pcl_[id] = IntensityToRgbPCL(pt.intensity);
        color_data_gray_[id] = Vec4f(0.5, 0.5, 0.5, 1.0);
        // 根据高度映射颜色
        color_data_height_[id] = IntensityToRgbPCL(pt.z * 10);
        color_data_intensity_[id] =
            Vec4f(pt.intensity / 255.0 * 3.0, pt.intensity / 255.0 * 3.0, pt.intensity / 255.0 * 3.0, 1.0);
    }
}

void UiCloud::Render() {
    // glPointSize(2.0);

    glBegin(GL_POINTS);
    glPointSize(point_size_);

    for (int i = 0; i < xyz_data_.size(); ++i) {
        if (use_color_ == UseColor::PCL_COLOR) {
            glColor4f(color_data_pcl_[i][0], color_data_pcl_[i][1], color_data_pcl_[i][2], ui::opacity);
        } else if (use_color_ == UseColor::INTENSITY_COLOR) {
            glColor4f(color_data_intensity_[i][0], color_data_intensity_[i][1], color_data_intensity_[i][2],
                      ui::opacity);
        } else if (use_color_ == UseColor::HEIGHT_COLOR) {
            glColor4f(color_data_height_[i][0], color_data_height_[i][1], color_data_height_[i][2], ui::opacity);
        } else if (use_color_ == UseColor::GRAY_COLOR) {
            glColor4f(color_data_gray_[i][0], color_data_gray_[i][1], color_data_gray_[i][2], ui::opacity);
        } else if (use_color_ = UseColor::CUSTOM_COLOR) {
            glColor4f(custom_color_[0], custom_color_[1], custom_color_[2], ui::opacity);
        }

        glVertex3f(xyz_data_[i][0], xyz_data_[i][1], xyz_data_[i][2]);
    }
    glEnd();
}

void UiCloud::BuildIntensityTable() {
    intensity_color_table_pcl_.reserve(255 * 3);
    // 接受rgb三个值，将它们归一化到范围 [0, 1]，同时设置 alpha 通道为0.2。
    auto make_color = [](int r, int g, int b) -> Vec4f { return Vec4f(r / 255.0f, g / 255.0f, b / 255.0f, 0.2f); };

    for (int i = 0; i < 256; i++) {
        intensity_color_table_pcl_.emplace_back(make_color(255, i, 0));
    }
    for (int i = 0; i < 256; i++) {
        intensity_color_table_pcl_.emplace_back(make_color(i, 0, 255));
    }

    for (int i = 0; i < 256; i++) {
        intensity_color_table_pcl_.emplace_back(make_color(0, 255, i));
    }
}

void UiCloud::SetRenderColor(UiCloud::UseColor use_color) { use_color_ = use_color; }

}  // namespace lightning::ui
