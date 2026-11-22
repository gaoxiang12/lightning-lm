//
// Created by xiang on 25-6-23.
//

#include "core/g2p5/g2p5_map.h"

#include <malloc.h>
#include <cstdlib>
#include <execution>

namespace lightning::g2p5 {

bool G2P5Map::Init(const float &temp_min_x, const float &temp_min_y, const float &temp_max_x, const float &temp_max_y) {
    ReleaseResources();
    min_x_ = temp_min_x;
    min_y_ = temp_min_y;
    max_x_ = temp_max_x;
    max_y_ = temp_max_y;

    grid_size_x_ = ceil((max_x_ - min_x_) / grid_reso_);
    grid_size_y_ = ceil((max_y_ - min_y_) / grid_reso_);

    if (grid_size_x_ <= 0 || grid_size_y_ <= 0) {
        return false;
    }

    grids_ = new SubGrid *[grid_size_x_];
    for (int xi = 0; xi < grid_size_x_; ++xi) {
        grids_[xi] = new SubGrid[grid_size_y_];
    }
    return true;
}

std::shared_ptr<G2P5Map> G2P5Map::MakeDeepCopy() {
    std::shared_ptr<G2P5Map> ret(new G2P5Map(options_));
    ret->min_x_ = min_x_;
    ret->min_y_ = min_y_;
    ret->max_x_ = max_x_;
    ret->max_y_ = max_y_;
    ret->grid_size_x_ = grid_size_x_;
    ret->grid_size_y_ = grid_size_y_;

    ret->grids_ = new SubGrid *[grid_size_x_];
    for (int xi = 0; xi < grid_size_x_; ++xi) {
        ret->grids_[xi] = new SubGrid[grid_size_y_];
        for (int yi = 0; yi < grid_size_y_; ++yi) {
            ret->grids_[xi][yi] = this->grids_[xi][yi];
        }
    }

    return ret;
}

bool G2P5Map::Resize(const float &temp_min_x, const float &temp_min_y, const float &temp_max_x,
                     const float &temp_max_y) {
    int temp_grid_size_x = ceil((temp_max_x - temp_min_x) / grid_reso_) + 1;
    int temp_grid_size_y = ceil((temp_max_y - temp_min_y) / grid_reso_) + 1;

    auto **new_grids = new SubGrid *[temp_grid_size_x];
    for (int xi = 0; xi < temp_grid_size_x; ++xi) {
        new_grids[xi] = new SubGrid[temp_grid_size_y];
    }

    // 地图左上角点所在的子grid id
    int min_grid_x = (int)floor((temp_min_x - min_x_) / grid_reso_);
    int min_grid_y = (int)floor((temp_min_y - min_y_) / grid_reso_);
    // 地图右下角点所在的子grid id + 1，为了左闭右开循环
    int max_grid_x = (int)floor((temp_max_x - min_x_) / grid_reso_) + 1;
    int max_grid_y = (int)floor((temp_max_y - min_y_) / grid_reso_) + 1;

    int dx = min_grid_x < 0 ? 0 : min_grid_x;
    int dy = min_grid_y < 0 ? 0 : min_grid_y;
    int Dx = max_grid_x < this->grid_size_x_ ? max_grid_x : this->grid_size_x_;
    int Dy = max_grid_y < this->grid_size_y_ ? max_grid_y : this->grid_size_y_;

    for (int x = dx; x < Dx; x++) {
        for (int y = dy; y < Dy; y++) {
            assert((x - min_grid_x) >= 0 && (x - min_grid_x) < temp_grid_size_x);
            assert((y - min_grid_y) >= 0 && (y - min_grid_y) < temp_grid_size_y);

            assert((x) >= 0 && (x) < temp_grid_size_x);
            assert((y) >= 0 && (y) < temp_grid_size_y);

            new_grids[x - min_grid_x][y - min_grid_y] = this->grids_[x][y];
        }
        delete[] this->grids_[x];
    }

    delete[] this->grids_;
    this->grids_ = new_grids;
    this->min_x_ = static_cast<int>(std::floor(temp_min_x / grid_reso_)) * grid_reso_; // 左上角只能是子grid左上角的坐标
    this->min_y_ = static_cast<int>(std::floor(temp_min_y / grid_reso_)) * grid_reso_;
    this->max_x_ = this->min_x_ + temp_grid_size_x * grid_reso_;
    this->max_y_ = this->min_y_ + temp_grid_size_y * grid_reso_;
    this->grid_size_x_ = temp_grid_size_x;
    this->grid_size_y_ = temp_grid_size_y;

    return true;
}

G2P5Map::~G2P5Map() {
    if (grids_ != nullptr) {
        for (int xi = 0; xi < grid_size_x_; ++xi) {
            delete[] grids_[xi];
        }
        delete[] grids_;
        grids_ = nullptr;
    }
}

void G2P5Map::SetHitPoint(const float &px, const float &py, const bool &if_hit, float height) {
    if (grids_ == nullptr) {
        return;
    }

    if (px < min_x_ || px > max_x_ || py < min_y_ || py > max_y_) {
        return;
    }

    int x_index = floor((px - min_x_) / options_.resolution_);
    int y_index = floor((py - min_y_) / options_.resolution_);

    UpdateCell(Vec2i(x_index, y_index), if_hit, height);
}

void G2P5Map::UpdateCell(const Vec2i &point_index, const bool &if_hit, float height) {
    if (grids_ == nullptr) {
        return;
    }

    int x_index = point_index.x();
    int y_index = point_index.y();
    int xi = (x_index >> SUB_GRID_SIZE);
    int yi = (y_index >> SUB_GRID_SIZE);

    if (xi < 0 || xi > (grid_size_x_ - 1) || yi < 0 || yi > (grid_size_y_ - 1)) {
        return;
    }

    int sub_index_i = x_index - (xi << SUB_GRID_SIZE);
    int sub_index_j = y_index - (yi << SUB_GRID_SIZE);

    if (sub_index_i < 0 || sub_index_i > (sub_grid_width_ - 1) || sub_index_j < 0 ||
        sub_index_j > (sub_grid_width_ - 1)) {
        return;
    }

    grids_[xi][yi].SetGridHitPoint(if_hit, sub_index_i, sub_index_j, height);
}

void G2P5Map::SetMissPoint(const float &point_x, const float &point_y, const float &laser_origin_x,
                           const float &laser_origin_y, float height, float lidar_height) {
    if (grids_ == nullptr) {
        return;
    }

    int point_x_index = floor(point_x / options_.resolution_);
    int point_y_index = floor(point_y / options_.resolution_);

    int xi_lidar = floor(laser_origin_x / options_.resolution_);
    int yi_lidar = floor(laser_origin_y / options_.resolution_);

    float k = 0;
    int sign = 1;
    int diff_y = point_y_index - yi_lidar;
    int diff_x = point_x_index - xi_lidar;

    /// 整数的直线填充算法
    if (diff_y == 0 && diff_x == 0) {
        return;
    }

    if (!GetDataIndex(laser_origin_x, laser_origin_y, xi_lidar, yi_lidar)) {
        return;
    }

    std::vector<Vec2i> updated_pts;
    std::vector<float> heights;

    if (std::abs(diff_y) > std::abs(diff_x)) {
        if (diff_y == 0) {
            return;
        }

        k = float(diff_x) / diff_y;
        float dh = (lidar_height - height) / diff_y;

        sign = diff_y > 0 ? 1 : -1;
        for (int j = sign; j != diff_y; j += sign) {
            int i = float(j * k);
            int x_index = xi_lidar + i;
            int y_index = yi_lidar + j;

            updated_pts.emplace_back(Vec2i(x_index, y_index));
            heights.emplace_back(lidar_height - j * dh);
        }
    } else {
        if (diff_x == 0) {
            return;
        }

        k = float(diff_y) / diff_x;
        sign = diff_x > 0 ? 1 : -1;

        float dh = (lidar_height - height) / diff_x;

        for (int i = sign; i != diff_x; i += sign) {
            int j = float(i * k);
            int x_index = xi_lidar + i;
            int y_index = yi_lidar + j;

            updated_pts.emplace_back(Vec2i(x_index, y_index));
            heights.emplace_back(lidar_height - i * dh);
        }
    }

    for (int i = 0; i < updated_pts.size(); ++i) {
        UpdateCell(updated_pts[i], false, heights[i]);
    }
}

bool G2P5Map::GetDataIndex(const float x, const float y, int &x_index, int &y_index) {
    if (x > max_x_ || x < min_x_ || y > max_y_ || y < min_y_) {
        return false;
    }
    x_index = floor((x - min_x_) / options_.resolution_);
    y_index = floor((y - min_y_) / options_.resolution_);

    return true;
}

void G2P5Map::ReleaseResources() {
    if (grids_ != nullptr) {
        for (int xi = 0; xi < grid_size_x_; ++xi) {
            delete[] grids_[xi];
        }
        delete[] grids_;
        grids_ = nullptr;
    }
    min_x_ = min_y_ = 10000;
    max_x_ = max_y_ = -10000;

    grid_size_x_ = grid_size_y_ = 0;
    malloc_trim(0);
}

nav_msgs::msg::OccupancyGrid G2P5Map::ToROS() {
    nav_msgs::msg::OccupancyGrid occu_map;
    int image_width = grid_size_x_ * sub_grid_width_;
    int image_height = grid_size_y_ * sub_grid_width_;
    occu_map.info.resolution = static_cast<nav_msgs::msg::MapMetaData::_resolution_type>(options_.resolution_);
    occu_map.info.width = static_cast<nav_msgs::msg::MapMetaData::_width_type>(image_width);
    occu_map.info.height = static_cast<nav_msgs::msg::MapMetaData::_width_type>(image_height);
    occu_map.info.origin.position.x = min_x_;
    occu_map.info.origin.position.y = min_y_;

    int grid_map_size = occu_map.info.width * occu_map.info.height;
    int grid_map_size_1 = grid_map_size - 1;
    occu_map.data.resize(grid_map_size);

    std::fill(occu_map.data.begin(), occu_map.data.end(), -1);
    int tmp_area = 0;
    int index_min, index_max;

    for (int bxi = 0; bxi < grid_size_x_; ++bxi) {
        for (int byi = 0; byi < grid_size_y_; ++byi) {
            if (grids_[bxi][byi].IsEmpty()) {
                continue;
            }

            for (int sxi = 0; sxi < sub_grid_width_; ++sxi) {
                for (int syi = 0; syi < sub_grid_width_; ++syi) {
                    int x = (bxi << SUB_GRID_SIZE) + sxi;
                    int y = (byi << SUB_GRID_SIZE) + syi;

                    if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
                        unsigned int hit_cnt = 0, visit_cnt = 0;
                        grids_[bxi][byi].GetHitAndVisit(sxi, syi, hit_cnt, visit_cnt);

                        float occ = (visit_cnt > 3) ? (float)hit_cnt / (float)visit_cnt : -1;

                        /// 注意这里有转置符号
                        if (occ < 0) {
                            continue;
                        } else if (occ > options_.occupancy_ratio_) {
                            tmp_area++;
                            occu_map.data[MapIdx(image_width, x, y)] = 100;
                        } else {
                            int index = MapIdx(image_width, x, y);
                            index_min = std::max(0, index - 1);
                            index_max = std::min(grid_map_size_1, index + 1);

                            for (auto extend = index_min; extend <= index_max; extend++) {
                                if (occu_map.data[extend] < 0) {  //-1
                                    tmp_area++;
                                    occu_map.data[extend] = 0;
                                }
                            }
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
    }
    return occu_map;
}

cv::Mat G2P5Map::ToCV() {
    int image_width = grid_size_x_ * sub_grid_width_;
    int image_height = grid_size_y_ * sub_grid_width_;

    // 定义要映射的颜色值
    cv::Vec3b black_color = cv::Vec3b(0, 0, 0);
    cv::Vec3b white_color = cv::Vec3b(255, 255, 255);
    cv::Vec3b other_color = cv::Vec3b(127, 127, 127);

    cv::Mat image(image_height, image_width, CV_8UC3, other_color);

    int image_height_1 = image_height - 1;
    int image_width_1 = image_width - 1;
    int index_y_min, index_y_max, index_x_min, index_x_max;

    for (int bxi = 0; bxi < grid_size_x_; ++bxi) {
        for (int byi = 0; byi < grid_size_y_; ++byi) {
            if (grids_[bxi][byi].IsEmpty()) {
                continue;
            }

            for (int sxi = 0; sxi < sub_grid_width_; ++sxi) {
                for (int syi = 0; syi < sub_grid_width_; ++syi) {
                    int x = (bxi << SUB_GRID_SIZE) + sxi;
                    int y = (byi << SUB_GRID_SIZE) + syi;

                    if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
                        unsigned int hit_cnt = 0, visit_cnt = 0;
                        grids_[bxi][byi].GetHitAndVisit(sxi, syi, hit_cnt, visit_cnt);

                        float occ = visit_cnt ? (hit_cnt == 0 ? 0 : (float)hit_cnt / (float)visit_cnt) : -1;
                        // assert(occ <= 1.0);

                        /// 注意这里有转置符号
                        if (occ < 0) {
                            continue;
                        } else if (occ > options_.occupancy_ratio_) {  // 0.49
                            image.at<cv::Vec3b>(y, x) = black_color;
                        } else {
                            index_y_min = std::max(0, y - 1);
                            index_y_max = std::min(image_height_1, y + 1);
                            index_x_min = std::max(0, x - 1);
                            index_x_max = std::min(image_width_1, x + 1);
                            for (int extend_y = index_y_min; extend_y <= index_y_max; extend_y++) {
                                for (int extend_x = index_x_min; extend_x <= index_x_max; extend_x++) {
                                    if (image.at<cv::Vec3b>(extend_y, extend_x) == other_color) {
                                        image.at<cv::Vec3b>(extend_y, extend_x) = white_color;
                                    }
                                }
                            }
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
    }

    /// cv的需要 flip y 才能转到俯视视角
    cv::Mat image_flip;
    cv::flip(image, image_flip, 1);
    return image_flip;
}

}  // namespace lightning::g2p5