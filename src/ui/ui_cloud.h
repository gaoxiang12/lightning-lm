#pragma once

#include "common/eigen_types.h"
#include "common/point_def.h"

namespace lightning::ui {

/// 在UI中使用的点云
/// 固定不变的点云都可以用这个来渲染
class UiCloud {
   public:
    /// 使用哪种颜色渲染本点云
    enum UseColor {
        PCL_COLOR,        // PCL颜色，偏红
        INTENSITY_COLOR,  // 亮度
        HEIGHT_COLOR,     // 高度
        GRAY_COLOR,       // 显示为灰色
        CUSTOM_COLOR,     // 显示为自定颜色
    };

    UiCloud() {}
    UiCloud(CloudPtr cloud);

    /**
     * 从PCL点云来设置一个UI点云
     * @param cloud             PCL 点云: lP, coordinates are in Lidar frame
     * @param pose              点云位姿: Twi，
     */
    void SetCloud(CloudPtr cloud, const SE3& pose);

    /// 渲染这个点云
    void Render();

    /// 指定内置颜色
    void SetRenderColor(UseColor use_color);

    /// 指定自选颜色，RGBA
    void SetCustomColor(Vec4f custom_color);

    void SetPointSize(float point_size) { point_size_ = point_size; }

   private:
    Vec4f IntensityToRgbPCL(const float& intensity) const {
        int index = int(intensity * 3);
        index = index % intensity_color_table_pcl_.size();
        return intensity_color_table_pcl_[index];
    }

    UseColor use_color_ = UseColor::PCL_COLOR;
    Vec4f custom_color_ = Vec4f::Zero();

    float point_size_ = 1.0f;
    std::vector<Vec3f> xyz_data_;              // 点的世界坐标
    std::vector<Vec4f> color_data_pcl_;        // 根据intensity映射得到的颜色
    std::vector<Vec4f> color_data_intensity_;  // 点的intensity不映射直接做颜色
    std::vector<Vec4f> color_data_height_;     // 根据height映射得到的颜色
    std::vector<Vec4f> color_data_gray_;       // 全部点都为灰色

    /// PCL中intensity table
    void BuildIntensityTable();
    // 颜色映射表
    static std::vector<Vec4f> intensity_color_table_pcl_;
};

}  // namespace lightning::ui
