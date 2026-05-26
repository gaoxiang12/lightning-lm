//
// Created by xiang on 25-6-23.
//

#include "core/g2p5/g2p5.h"
#include "common/constant.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <map>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/segmentation/impl/sac_segmentation.hpp>

#include "utils/timer.h"
#include "yaml-cpp/yaml.h"

namespace lightning::g2p5 {

G2P5::~G2P5() { Quit(); }

void G2P5::Quit() {
    quit_flag_ = true;

    if (options_.online_mode_) {
        draw_frontend_map_thread_.Quit();
    }

    if (draw_backend_map_thread_.joinable()) {
        draw_backend_map_thread_.join();
    }
}

void G2P5::PushKeyframe(Keyframe::Ptr kf) {
    UL lock(kf_mutex_);
    all_keyframes_.emplace_back(kf);

    if (options_.online_mode_) {
        draw_frontend_map_thread_.AddMessage(kf);
    } else {
        RenderFront(kf);
    }
}

void G2P5::RenderFront(Keyframe::Ptr kf) {
    {
        UL lock(frontend_mutex_);
        frontend_current_ = kf;
    }

    {
        UL lock{newest_map_mutex_};

        lightning::Timer::Evaluate([&]() { AddKfToMap({kf}, frontend_map_); }, "G2P5 Occupancy Mapping", true);
        newest_map_ = frontend_map_;
    }

    /// 向外回调
    if (map_update_cb_) {
        map_update_cb_(newest_map_);
    }
}

void G2P5::RedrawGlobalMap() { backend_redraw_flag_ = true; }

void G2P5::RenderBack() {
    while (!quit_flag_) {
        while (!backend_redraw_flag_ && !quit_flag_) {
            sleep(1);
        }

        if (quit_flag_) {
            break;
        }

        is_busy_ = true;

        /// 后端重绘被触发
        backend_redraw_flag_ = false;

        /// 重新绘制整张地图，如果中间过程又被重绘了，则退出
        std::vector<Keyframe::Ptr> all_keyframes;
        {
            UL lock(kf_mutex_);
            all_keyframes = all_keyframes_;
        }

        if (all_keyframes.empty()) {
            is_busy_ = false;
            continue;
        }

        G2P5Map::Options opt;
        opt.resolution_ = options_.grid_map_resolution_;
        backend_map_ = std::make_shared<G2P5Map>(opt);
        auto cur_kf = all_keyframes.begin();
        bool abort = false;

        for (; cur_kf != all_keyframes.end(); ++cur_kf) {
            AddKfToMap({*cur_kf}, backend_map_);
            if (backend_redraw_flag_) {
                LOG(INFO) << "backend redraw triggered in process, abort";
                abort = true;
                break;
            }

            if (quit_flag_) {
                abort = true;
                break;
            }
        }

        if (abort) {
            /// 继续重绘
            is_busy_ = false;
            continue;
        }

        /// 绘制过程中前端可能发生了更新，要保证后端绘制和前端的一致性
        int cur_idx = all_keyframes.back()->GetID();
        while (true) {
            Keyframe::Ptr frontend_kf = nullptr;
            {
                UL lock(frontend_mutex_);
                frontend_kf = frontend_current_;
            }

            if (cur_idx == frontend_kf->GetID()) {
                break;
            }

            int frontend_idx = frontend_kf->GetID();
            std::vector<Keyframe::Ptr> kfs;

            {
                UL lock(kf_mutex_);
                for (int i = cur_idx + 1; i <= frontend_idx; ++i) {
                    kfs.emplace_back(all_keyframes_[i]);
                }
            }

            AddKfToMap(kfs, backend_map_);
            cur_idx = frontend_idx;
        }

        {
            /// 同步前后端地图，替换newest map
            UL lock{newest_map_mutex_};
            frontend_map_ = backend_map_;
            newest_map_ = frontend_map_;
        }

        /// 向外回调
        if (map_update_cb_) {
            map_update_cb_(newest_map_);
        }

        is_busy_ = false;
    }

    LOG(INFO) << "backend render quit";
}

bool G2P5::ResizeMap(const std::vector<Keyframe::Ptr> &kfs, G2P5MapPtr &map) {
    /// 重设地图大小
    float init_min_x, init_min_y, init_max_x, init_max_y;
    float min_x, min_y, max_x, max_y;
    map->GetMinAndMax(init_min_x, init_min_y, init_max_x, init_max_y);
    min_x = init_min_x;
    min_y = init_min_y;
    max_x = init_max_x;
    max_y = init_max_y;

    /// 从后往前迭代
    for (auto it = kfs.begin(); it != kfs.end(); ++it) {
        if (quit_flag_) {
            return true;
        }

        auto kf = *it;
        SE3 pose = kf->GetOptPose();
        auto cloud = kf->GetCloud();

        if (pose.translation().x() < min_x) {
            min_x = pose.translation().x();
        }

        if (pose.translation().y() < min_y) {
            min_y = pose.translation().y();
        }

        if (pose.translation().x() > max_x) {
            max_x = pose.translation().x();
        }

        if (pose.translation().y() > max_y) {
            max_y = pose.translation().y();
        }

        if (min_x > max_x || min_y > max_y) {
            return false;
        }

        for (size_t i = 0; i < cloud->points.size(); i += 10) {
            float range = cloud->points[i].getVector3fMap().norm();

            if (range > options_.usable_scan_range_ || range <= 0.01 || std::isnan(range)) {
                continue;
            }

            Vec3d point = pose * cloud->points[i].getVector3fMap().cast<double>();

            if ((point.x() - 1) < min_x) {
                min_x = point.x() - 1;
            }
            if ((point.y() - 1) < min_y) {
                min_y = point.y() - 1;
            }
            if ((point.x() + 1) > max_x) {
                max_x = point.x() + 1;
            }
            if ((point.y() + 1) > max_y) {
                max_y = point.y() + 1;
            }
        }
    }

    if (min_x > max_x || min_y > max_y) {
        return false;
    }

    /// resize map if necessary
    if (min_x < init_min_x || min_y < init_min_y || max_x > init_max_x || max_y > init_max_y) {
        min_x = (min_x > init_min_x) ? init_min_x : min_x;
        min_y = (min_y > init_min_y) ? init_min_y : min_y;
        max_x = (max_x < init_max_x) ? init_max_x : max_x;
        max_y = (max_y < init_max_y) ? init_max_y : max_y;

        float r = map->GetGridResolution();
        min_x = static_cast<int>((floor)(min_x / r)) * r;
        min_y = static_cast<int>((floor)(min_y / r)) * r;
        max_x = static_cast<int>((ceil)(max_x / r)) * r;
        max_y = static_cast<int>((ceil)(max_y / r)) * r;
        map->Resize(min_x, min_y, max_x, max_y);

        // LOG(INFO) << "map resized to " << min_x << ", " << min_y << ", " << max_x << ", " << max_y;
    }

    return true;
}

bool G2P5::AddKfToMap(const std::vector<Keyframe::Ptr> &kfs, G2P5MapPtr &map) {
    /// 如果需要，更新地图大小
    ResizeMap(kfs, map);

    for (const auto &kf : kfs) {
        Convert3DTo2DScan(kf, map);
    }

    return true;
}

G2P5MapPtr G2P5::GetNewestMap() {
    UL lock{newest_map_mutex_};
    LOG(INFO) << "getting newest map";
    if (newest_map_ == nullptr) {
        return nullptr;
    }

    return newest_map_->MakeDeepCopy();
}

void G2P5::Convert3DTo2DScan(Keyframe::Ptr kf, G2P5MapPtr &map) {
    // 3D转2D算法
    if (options_.esti_floor_) {
        if (!DetectPlaneCoeffs(kf)) {
            /// 如果动态检测失败，就用之前的参数
            floor_coeffs_ = Vec4d(0, 0, 1, -options_.default_floor_height_);
        } else {
            if (options_.verbose_) {
                LOG(INFO) << "floor coeffs: " << floor_coeffs_.transpose();
            }
        }
    } else {
        floor_coeffs_ = Vec4d(0, 0, 1, -options_.default_floor_height_);
    }

    // step 1. 计算每个方向上发出射线上的高度分布, // NOTE 转成整形的360度是有精度损失的
    std::vector<std::map<double, double>> rays(360);               // map键值：距离-相对高度（以距离排序）
    std::vector<Vec2d> angle_distance_height(360, Vec2d::Zero());  // 每个角度上的距离-高度值
    std::vector<Vec3d> pts_3d;                                     /// 距离地面0.3 ～ 1.2米之间的点云，激光坐标系下

    SE3 Twb = kf->GetOptPose();

    double min_th = options_.min_th_floor_;
    double max_th = options_.max_th_floor_;

    /// 把激光系下的点云转到当前submap坐标系
    auto cloud = kf->GetCloud();

    CloudPtr obstacle_cloud(new PointCloudType);

    /// 黑色点的处理方式：所有在障碍物范围内的都是黑色点
    int cnt_valid = 0;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto &pt = cloud->points[i];
        if (quit_flag_) {
            return;
        }

        Vec3d pc = Vec3d(pt.x, pt.y, pt.z);
        Vec4d pn = Vec4d(pt.x, pt.y, pt.z, 1);

        // 计算激光点所在角度方向以及高度
        Vec2d p = pc.head<2>();
        double dis = p.norm();

        if (dis > options_.usable_scan_range_) {
            continue;
        }

        double dis_floor = pn.dot(floor_coeffs_);  /// 该点到地面的距离
        double dangle = atan2(p[1], p[0]) * constant::kRAD2DEG;
        int angle = int(round(dangle) + 360) % 360;

        if (dis_floor > min_th) {
            if (dis_floor < max_th) {
                // 特别矮的和特别高的都不计入
                pts_3d.emplace_back(pc);

                rays[angle].insert({dis, dis_floor});

                /// 设置黑点
                Vec3d p_world = Twb * pc;
                map->SetHitPoint(p_world[0], p_world[1], true, dis_floor);

                cnt_valid++;

                obstacle_cloud->points.push_back(pt);
            }
        } else if (dis_floor > -min_th) {
            // 地面附近或者地面以下
            rays[angle].insert({dis, dis_floor});
            cnt_valid++;
        }
    }

    // if (options_.verbose_) {
    //     obstacle_cloud->is_dense = false;
    //     obstacle_cloud->height = 1;
    //     obstacle_cloud->width = obstacle_cloud->size();

    //     pcl::io::savePCDFile("./data/obs.pcd", *obstacle_cloud);
    // }

    // LOG(INFO) << "valid obs: " << cnt_valid << ", total: " << cloud->size();

    std::vector<double> floor_esti_data;  // 地面高度估计值

    // step 2, 考察每个方向上的分布曲线
    // 正常场景中，每个方向由较低高度开始（地面），转到较高的高度（物体）
    // 如若不是，那么可能发生了遮挡或进入盲区
    constexpr double default_ray_distance = -1;
    const double floor_rh = floor_coeffs_[3];

    for (int i = 0; i < 360; ++i) {
        if (quit_flag_) {
            return;
        }

        if (rays[i].size() < 2) {
            // 该方向测量数据很少，在16线中是不太可能出现的（至少地面上应该有线），出现，则说明有严重遮挡或失效，认为该方向取一个默认的最小距离（车宽）
            angle_distance_height[i] = Vec2d(default_ray_distance, floor_rh);
            continue;
        }

        /// 取距离和高度
        for (auto iter = rays[i].rbegin(); iter != rays[i].rend(); ++iter) {
            if (iter->second < options_.min_th_floor_) {
                angle_distance_height[i] = Vec2d(iter->first, iter->second);
                continue;
            }

            auto next_iter = iter;
            next_iter++;

            if (next_iter != rays[i].rend()) {
                if (iter->second > options_.min_th_floor_ && next_iter->second < options_.min_th_floor_) {
                    // 当前点是障碍但下一个点不是
                    angle_distance_height[i] = Vec2d(iter->first, iter->second);
                    break;
                }
            } else {
                angle_distance_height[i] = Vec2d(iter->first, iter->second);
            }
        }
    }

    // 以2D scan方式添加白色点
    SetWhitePoints(angle_distance_height, kf, map);
}

void G2P5::SetWhitePoints(const std::vector<Vec2d> &pt2d, Keyframe::Ptr kf, G2P5MapPtr &map) {
    assert(pt2d.size() == 360);

    SE3 pose = kf->GetOptPose();
    Vec3d orig = pose.translation();

    for (int i = 0; i < 360; ++i) {
        if (quit_flag_) {
            return;
        }

        double angle = float(i) * constant::kDEG2RAD;
        float r = pt2d[i][0];
        float h = pt2d[i][1];

        Vec3d p_local(r * cos(angle), r * sin(angle), h);
        Vec3d p_world = pose * p_local;

        /// 某方向无测量值时，认为无效
        if (r <= 0 || r > options_.usable_scan_range_) {
            /// 比较近时，涂白
            if (r < 0.1) {
                map->SetMissPoint(p_world[0], p_world[1], orig[0], orig[1], h, options_.lidar_height_);
            }
            continue;
        }

        map->SetMissPoint(p_world[0], p_world[1], orig[0], orig[1], h, options_.lidar_height_);
    }
}

bool G2P5::DetectPlaneCoeffs(Keyframe::Ptr kf) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<PointType> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.25);

    CloudPtr cloud(new PointCloudType);
    for (auto &pt : kf->GetCloud()->points) {
        if (pt.z < options_.lidar_height_ + options_.default_floor_height_) {
            cloud->points.push_back(pt);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    if (cloud->size() < 200) {
        // LOG(ERROR) << "not enough points cloud->size(): " << cloud->size();
        return false;
    }

    // if (options_.verbose_) {
    //     pcl::io::savePCDFile("./data/floor_candi.pcd", *cloud);
    // }

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (coefficients->values[2] < 0.99) {
        LOG(ERROR) << "floor is not horizontal. ";
        return false;
    }

    if (inliers->indices.size() < 100) {
        LOG(ERROR) << "cannot get enough points on floor: " << inliers->indices.size();
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        floor_coeffs_[i] = coefficients->values[i];
    }
    cloud->clear();

    return true;
}

void G2P5::Init(std::string yaml_path) {
    auto yaml = YAML::LoadFile(yaml_path);
    options_.esti_floor_ = yaml["g2p5"]["esti_floor"].as<bool>();
    options_.min_th_floor_ = yaml["g2p5"]["min_th_floor"].as<float>();
    options_.max_th_floor_ = yaml["g2p5"]["max_th_floor"].as<float>();
    options_.lidar_height_ = yaml["g2p5"]["lidar_height"].as<float>();
    options_.grid_map_resolution_ = yaml["g2p5"]["grid_map_resolution"].as<float>();
    options_.default_floor_height_ = yaml["g2p5"]["floor_height"].as<float>();

    G2P5Map::Options opt;
    opt.resolution_ = options_.grid_map_resolution_;
    frontend_map_ = std::make_shared<G2P5Map>(opt);

    if (options_.online_mode_) {
        draw_frontend_map_thread_.SetProcFunc([this](Keyframe::Ptr kf) { RenderFront(kf); });
        draw_frontend_map_thread_.Start();
    }

    draw_backend_map_thread_ = std::thread([this]() { RenderBack(); });
}

}  // namespace lightning::g2p5