//
// Created by xiang on 23-2-7.
//

#ifndef LIGHNING_TILED_MAP_H
#define LIGHNING_TILED_MAP_H

#include <opencv2/core/core.hpp>
#include <set>
#include <map>
#include <unordered_map>

#include "common/functional_points.h"
#include "common/point_def.h"
#include "common/std_types.h"
#include "core/lightning_math.hpp"
#include "core/maps/tiled_map_chunk.h"

namespace lightning {

/**
 * 包含动静态的点云
 *
 * 静态地图从文件目录中读取（先切分，切分使用同一个接口）
 * 动态地图实时从lidar odom中构建
 *
 * TODO:
 * - **done** 动态区域也需要分块，不然太大
 * - 动态区域应该有生命周期和最大采样点（目前在option中设置）
 * - 动态图层的保存与读取
 * - thread safety
 */
class TiledMap {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /// 动态区域的处理策略
    enum class DynamicCloudPolicy {
        /// 短时间保留，车辆移出该区域就会清空
        SHORT = 0,

        /// 长时间保留在内存，车辆移出区域后仍然保留，但不会存盘并在下次开机时使用
        LONG,

        /// 更长时间的保留，会存盘，下次运行也会读取，超过一定时间后才会清除
        PERSISTENT,
    };

    struct Options {
        Options() {}
        std::string map_path_ = "./data/maps/";                       // 地图存储路径
        float chunk_size_ = 100.0;                                    // 单个chunk大小
        float inv_chunk_size_ = 1.0 / 100.0;                          // 反向chunk size
        float voxel_size_in_chunk_ = 0.1;                             // chunk内部的分辨率
        const int load_nearby_chunks_ = 8;                            // 载入邻近范围内的点云
        bool save_dynamic_layer_ = false;                             // 是否将动态图层也存入地图文件夹
        int load_map_size_ = 2;                                       // 载入的邻近网格数
        int unload_map_size_ = 3;                                     // 卸载的网格数
        bool enable_dynamic_polygon_ = false;                         // 是否使用动态区域的多边形
        size_t max_pts_in_dyn_chunk_ = 50000;                         // 每个动态区块中最多点数
        DynamicCloudPolicy policy_ = DynamicCloudPolicy::PERSISTENT;  // 动态区域处理策略

        bool delete_when_unload_ = false;   // 卸载区域时是否清空点云
        bool load_dyn_cloud_ = true;        // 初始化时是否加载动态点云
        bool save_dyn_when_quit_ = true;    // 退出时是否保存动态点云
        bool save_dyn_when_unload_ = true;  // 卸载时是否保存动态点云
    };

    /// 动态区域的多边形
    struct DynamicPolygon {
        DynamicPolygon() {}
        DynamicPolygon(int id, const Vec2d& pt) : id_(id) { polygon_.emplace_back(cv::Point2f(pt[0], pt[1])); }
        int id_ = 0;
        std::vector<cv::Point2f> polygon_;
    };

    enum class NearbyType {
        CENTER,
        NEARBY6,
    };
    struct NdtMapOptions {
        NdtMapOptions() {}
        float static_cloud_weight_ = 1.0;              // 静态点云权重
        float dynamic_cloud_weight_ = 0.1;             // 动态点云权重
        NearbyType nearby_type_ = NearbyType::CENTER;  // 邻近类型
        float voxel_size_ = 1.0;                       // voxel大小
        int max_iteration_ = 5;                        // 最大迭代次数
        double eps_ = 1e-2;                            // 收敛判定条件
        double res_outlier_th_ = 50.0;                 // 异常值拒绝阈值
        float inv_voxel_size_ = 1.0;                   // 栅格尺寸倒数
        int min_pts_in_voxel_ = 3;                     // 每个栅格中最小点数
        int min_effective_pts_ = 500;                  // 最小有效点数
        size_t capacity_ = 100000;                     // 缓存的体素数量
        int max_pts_in_voxel_ = 50;                    // 每个栅格中最大点数

        bool verbose_ = false;
    };

    using KeyType = Eigen::Matrix<int, 3, 1>;  // 体素的索引
    struct VoxelData {
        VoxelData() {}
        VoxelData(size_t id) { idx_.emplace_back(id); }
        VoxelData(const Vec3f& pt) {
            pts_.emplace_back(pt);
            num_pts_ = 1;
        }

        void AddPoint(const Vec3f& pt) {
            pts_.emplace_back(pt);
            if (!ndt_estimated_) {
                num_pts_++;
            }
        }

        std::vector<Vec3f> pts_;           // 内部点，多于一定数量之后再估计均值和协方差
        std::vector<size_t> idx_;          // 点云中点的索引
        Vec3f mu_ = Vec3f::Zero();         // 均值
        Mat3f sigma_ = Mat3f::Identity();  // 协方差
        Mat3f info_ = Mat3f::Zero();       // 协方差之逆

        bool ndt_estimated_ = false;  // NDT是否已经估计
        int num_pts_ = 0;             // 总共的点数，用于更新估计
    };

    using GridHashMap = std::unordered_map<KeyType, VoxelData, math::hash_vec<3>>;
    using ChunkHashMap = std::unordered_map<Vec2i, std::shared_ptr<MapChunk>, math::hash_vec<2>>;

    TiledMap(Options options = Options()) : options_(options) {}

    /**
     * 从单个PCD进行转换
     * @param map       PCD地图
     * @param start_pose    地图起点
     * @return
     */
    bool ConvertFromFullPCD(CloudPtr map, const SE3& start_pose, const std::string& map_path = "./data/maps/");

    /// 载入地图索引文件
    bool LoadMapIndex();

    /// 载入某个位置附近的点云
    void LoadOnPose(const SE3& pose);

    // /// 载入某个位置附近的点云
    // void LoadOnPose(const SE3& pose, bool wndt = true);

    /// 获取当前载入的静态点云（分块形式）
    /// 现在会调用深拷贝
    std::map<int, CloudPtr> GetStaticCloud();

    /// 获取当前载入的动态点云（分块形式）
    /// 现在会调用深拷贝
    std::map<int, CloudPtr> GetDynamicCloud();

    /// 获取拼接好的地图
    CloudPtr GetAllMap();

    /**
     * 使用最近配准之后的scan更新动态图层
     * @param cloud_world       世界坐标系下点云
     * @param remove_old_pts    是否会移除历史的数据
     */
    void UpdateDynamicCloud(CloudPtr cloud_world, bool remove_old_pts = true);

    /// 重置动态图层
    void ResetDynamicCloud();

    /// accessors
    /// 获取地图原点
    Vec3d GetOrigin() const { return origin_; }

    /// 添加地图功能点
    void AddFP(const FunctionalPoint& fp) { func_points_.emplace_back(fp); }

    /// 获取所有功能点
    std::vector<FunctionalPoint> GetAllFP() const { return func_points_; }

    int NumActiveChunks() const { return loaded_chunks_.size(); }

    bool MapUpdated() const { return map_updated_; }
    bool DynamicMapUpdated() const { return dynamic_map_updated_; }

    void CleanMapUpdate() {
        map_updated_ = false;
        dynamic_map_updated_ = false;
    }

    /// 更新静态点云地图
    void AddStaticCloud(CloudPtr cloud);

    /// 更新动态点云地图
    void AddDynamicCloud(CloudPtr cloud);

    /// 更新体素内部数据, 根据新加入的pts和历史的估计情况来确定自己的估计
    void UpdateVoxel(VoxelData& v, bool flag_first_scan = true);

    // 获取静态NDT栅格地图
    GridHashMap GetStaticGridMap();

    // 获取动态NDT栅格地图
    GridHashMap GetDynamicGridMap();

    GridHashMap static_grids_;   // 静态地图栅格
    GridHashMap dynamic_grids_;  // 动态地图栅格

    std::map<int, DynamicPolygon> GetDynamicPolygons() const { return dynamic_polygon_; }

    /// 存储到二进制pcd文件
    void SaveToBin(bool only_dynamic = false);

    template <class T>
    void SetNewTargetForNDT(T ndt) {
        bool has_cloud = false;
        {
            UL lock(static_data_mutex_);
            for (auto& idx : loaded_chunks_) {
                if (!static_chunks_[idx]->cloud_->empty()) {
                    has_cloud = true;
                }
                ndt->AddTarget(static_chunks_[idx]->cloud_);
            }
        }

        {
            UL lock(dynamic_data_mutex_);
            for (auto& idx : loaded_chunks_) {
                if (dynamic_chunks_.find(idx) != dynamic_chunks_.end()) {
                    if (dynamic_chunks_[idx]->cloud_ != nullptr) {
                        has_cloud = true;
                        ndt->AddTarget(dynamic_chunks_[idx]->cloud_);
                    }
                }
            }
        }

        ndt->ComputeTargetGrids();
        if (has_cloud) {
            ndt->initCompute();
        }
    }

    /// 清理地图
    void ClearMap();

   private:
    /**
     * 测试某个点是否落在动态区域
     * @param pt 给定点
     * @return 动态区域ID
     */
    int FallsInDynamicArea(const Vec3f& pt);

   private:
    /// 注意这里有个边长
    inline Vec2i Pos2Grid(const Vec2f& pt) const {
        Vec2d p = ((pt.cast<double>() * options_.inv_chunk_size_) + Vec2d(0.5, 0.5));
        return Vec2i(floor(p[0]), floor(p[1]));
    }

    inline Vec2i Pos2Grid(const Vec2d& pt) const {
        Vec2d p = ((pt * options_.inv_chunk_size_) + Vec2d(0.5, 0.5));
        return Vec2i(floor(p[0]), floor(p[1]));
    }

    Options options_;
    NdtMapOptions ndt_map_options_;

    std::mutex static_data_mutex_;   // 静态数据锁
    std::mutex dynamic_data_mutex_;  // 动态数据锁

    Vec3d origin_ = Vec3d::Zero();
    ChunkHashMap static_chunks_;       // 静态地图
    ChunkHashMap dynamic_chunks_;      // 动态地图
    std::map<int, Vec2i> id_to_grid_;  // 从ID找grid
    int chunk_id_ = 0;                 // chunk生成时的ID

    Vec2i last_load_grid_ = Vec2i::Zero();  // 上次加载时的网格
    bool last_load_grid_set_ = false;       // 上次加载的flag

    std::set<Vec2i, math::less_vec<2>> loaded_chunks_;  // 已经载入的区块，以网格ID为索引
    bool map_updated_ = false;
    bool dynamic_map_updated_ = false;

    std::map<int, DynamicPolygon> dynamic_polygon_;  // 动态区域，从txt中读取

    std::vector<FunctionalPoint> func_points_;  // 功能点

    bool flag_first_dynamic_scan_ = true;  // 首帧动态点云特殊处理
    bool flag_first_static_scan_ = true;   // 首帧静态点云特殊处理
};

}  // namespace lightning

#endif