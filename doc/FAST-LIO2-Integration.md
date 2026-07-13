# FAST-LIO2 集成开发文档

## 1. 目标

将 FAST-LIO2 作为可切换的 LIO 前端集成到 Lightning-LM 中：

- **建图模式**: 可选使用 FAST-LIO2 或原生 LIO (AA-FasterLIO)
- **定位模式**: 始终使用原生 LIO
- **统一接口**: 通过多态 `LIOFrontend` 基类实现 `lio_->Run()` 调用

## 2. 架构设计

```
SlamSystem
├── LIOFrontend* lio_  ← 多态指针
│   ├── LaserMapping      (原生 LIO, 12D ESKF, iVox)
│   └── FASTLIO2Mapping   (FAST-LIO2, 23D ESEKF, ikd-Tree)
├── LoopClosing           (不变)
├── G2P5                  (不变)
└── TiledMap              (不变)
```

## 3. 文件清单

### 3.1 新增文件


| 文件                                        | 用途                            |
| ------------------------------------------- | ------------------------------- |
| `src/core/frontend/lio_frontend.h`          | LIO 前端抽象基类                |
| `src/core/fast_lio2/fastlio2_mapping.h`     | FAST-LIO2 适配器头文件          |
| `src/core/fast_lio2/fastlio2_mapping.cc`    | FAST-LIO2 适配器实现            |
| `src/core/fast_lio2/fastlio2_core.h`        | FAST-LIO2 算法核心封装 (头文件) |
| `src/core/fast_lio2/fastlio2_core.cc`       | FAST-LIO2 算法核心封装 (实现)   |
| `src/core/fast_lio2/fastlio2_use_ikfom.hpp` | ESEKF 状态定义                  |
| `src/core/fast_lio2/fastlio2_so3_math.h`    | SO3 数学工具                    |
| `thirdparty/ikd-Tree/`                      | ikd-Tree 库 (从 FAST_LIO 复制)  |
| `thirdparty/IKFoM_toolkit/`                 | ESEKF 工具库 (从 FAST_LIO 复制) |

### 3.2 修改文件


| 文件                            | 修改内容                                                    |
| ------------------------------- | ----------------------------------------------------------- |
| `src/core/lio/laser_mapping.h`  | 继承`LIOFrontend`，添加 `override`                          |
| `src/core/lio/laser_mapping.cc` | 无逻辑变更，仅标记 override                                 |
| `src/core/system/slam.h`        | `LaserMapping*` → `LIOFrontend*`，添加 `GetAllKeyframes()` |
| `src/core/system/slam.cc`       | 根据配置创建前端实例                                        |
| `config/default_linghou.yaml`   | 添加`mapping_frontend` 和 `fast_lio2` 配置段                |
| `CMakeLists.txt`                | 添加新文件和 ikd-Tree 依赖                                  |

## 4. 接口定义

### 4.1 LIOFrontend 基类

```cpp
// src/core/frontend/lio_frontend.h
namespace lightning {

class LIOFrontend {
public:
    virtual ~LIOFrontend() = default;

    virtual bool Init(const std::string& config_yaml) = 0;
    virtual bool Run() = 0;

    // 点云输入 (统一使用 Lightning-LM 的点类型)
    virtual void ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) = 0;
    virtual void ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg) = 0;
    virtual void ProcessPointCloud2(CloudPtr cloud) = 0;
    virtual void ProcessIMU(const IMUPtr& msg_in) = 0;

    // 结果获取
    virtual SE3 GetPose() const = 0;
    virtual CloudPtr GetScanUndist() const = 0;
    virtual CloudPtr GetScanDownWorld() const = 0;
    virtual Keyframe::Ptr GetKeyframe() const { return nullptr; }
    virtual std::vector<Keyframe::Ptr> GetAllKeyframes() { return {}; }

    // 可选功能
    virtual void SetUI(std::shared_ptr<ui::PangolinWindow> ui) {}
    virtual void SaveMap() {}
    virtual CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel = true, float res = 0.1) { return nullptr; }
    virtual void PrintExtrinsic() {}
};

}  // namespace lightning
```

### 4.2 FASTLIO2Mapping 适配器

```cpp
// src/core/fast_lio2/fastlio2_mapping.h
namespace lightning {

class FASTLIO2Mapping : public LIOFrontend {
public:
    struct NativeState {
        Vec3d pos;
        Mat3d rot;
        Mat3d offset_R_L_I;  // LiDAR→IMU 旋转
        Vec3d offset_T_L_I;  // LiDAR→IMU 平移
        Vec3d vel;
        Vec3d bg;
        Vec3d ba;
        Vec3d grav;
    };

    FASTLIO2Mapping();
    ~FASTLIO2Mapping() override;

    bool Init(const std::string& config_yaml) override;
    bool Run() override;

    void ProcessPointCloud2(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) override;
    void ProcessPointCloud2(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg) override;
    void ProcessPointCloud2(CloudPtr cloud) override;
    void ProcessIMU(const IMUPtr& msg_in) override;

    SE3 GetPose() const override;
    CloudPtr GetScanUndist() const override;
    CloudPtr GetScanDownWorld() const override;
    Keyframe::Ptr GetKeyframe() const override;
    std::vector<Keyframe::Ptr> GetAllKeyframes() override;

    void SetUI(std::shared_ptr<ui::PangolinWindow> ui) override;
    void SaveMap() override;
    CloudPtr GetGlobalMap(bool use_lio_pose, bool use_voxel, float res) override;

    NativeState GetNativeState() const;

private:
    bool SyncPackages();
    void DownSample();
    void MakeKF();

    std::shared_ptr<FASTLIO2Core> core_;
    std::shared_ptr<PointCloudPreprocess> preprocess_;

    // 数据缓冲
    std::deque<IMUPtr> imu_buffer_;
    std::deque<CloudPtr> lidar_buffer_;
    std::deque<double> time_buffer_;
    std::mutex mtx_buffer_;

    // 输出
    CloudPtr scan_undistort_{new PointCloudType()};
    CloudPtr scan_down_body_{new PointCloudType()};   // 体坐标系降采样点云
    CloudPtr scan_down_world_{new PointCloudType()};  // 世界坐标系降采样点云
    NativeState native_state_;
    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    int kf_id_ = 0;

    // 同步状态
    bool lidar_pushed_ = false;
    double last_timestamp_imu_ = -1.0;
    double lidar_end_time_ = 0;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;

    // 关键帧参数
    double kf_dis_th_ = 2.0;
    double kf_angle_th_ = 15.0 * M_PI / 180.0;

    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;
};

}  // namespace lightning
```

### 4.3 FASTLIO2Core 算法核心

```cpp
// src/core/fast_lio2/fastlio2_core.h
namespace lightning {

class FASTLIO2Core {
    friend void fastlio2_h_share_model(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data);

public:
    struct Config {
        int lidar_type = 4;          // 1=Livox, 2=Velodyne, 3=Ouster, 4=RoboSense
        int scan_line = 64;
        double blind = 0.5;
        double filter_size_scan = 0.5;
        double filter_size_map = 0.5;
        int max_iteration = 4;
        double acc_cov = 0.1;
        double gyr_cov = 0.1;
        double b_acc_cov = 0.0001;
        double b_gyr_cov = 0.0001;
        Vec3d extrinsic_T = Vec3d::Zero();
        Mat3d extrinsic_R = Mat3d::Identity();
    };

    bool Init(const Config& config);

    /// IMU 传播 + 去畸变
    void IMUProcess(const FASTLIO2MeasureGroup& measures, CloudPtr& scan_out);

    /// 观测更新 (点到面 ICP + ESEKF)
    void Observe(CloudPtr scan_down);

    /// 地图更新 (添加点到 ikd-Tree)
    void UpdateMap(CloudPtr scan_world);

    SE3 GetPose() const;
    NativeState GetNativeState() const;

    KD_TREE<PointType>& GetKDTree() { return ikdtree_; }
    void PointBodyToWorld(const PointType& pi, PointType& po);
    CloudPtr GetScanDownBody() const { return feats_down_body_; }
    int GetFeatsDownSize() const { return feats_down_size_; }
    void SetFeatsDownSize(int n) { feats_down_size_ = n; }

private:
    static FASTLIO2Core* instance_;
    Config config_;

    // ESEKF (IKFoM)
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;
    state_ikfom state_point_;
    Eigen::Matrix<double, 12, 12> Q_;

    // ikd-Tree
    KD_TREE<PointType> ikdtree_;

    // 点云
    CloudPtr feats_undistort_{new PointCloudType()};
    CloudPtr feats_down_body_{new PointCloudType()};
    CloudPtr feats_down_world_{new PointCloudType()};

    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterSurf_;
    pcl::VoxelGrid<PointType> downSizeFilterMap_;

    // IMU 处理
    Pose6D last_imu_pose_;
    Vec3d mean_acc_, mean_gyr_;
    Vec3d cov_acc_, cov_gyr_;
    Vec3d angvel_last_, acc_s_last_;
    double first_lidar_time_ = 0.0;
    double last_lidar_end_time_ = 0.0;  // 注意：初始化为 0.0，不是 -1.0
    bool imu_need_init_ = true;
    bool b_first_frame_ = true;

    // 状态
    bool flg_EKF_inited_ = false;
    int effct_feat_num_ = 0;
    int feats_down_size_ = 0;

    // 最近邻搜索
    std::vector<PointVector> Nearest_Points;
    bool point_selected_surf_[100000] = {false};
    float res_last_[100000] = {0.0};

    // 外参
    Mat3d Lidar_R_wrt_IMU_;
    Vec3d Lidar_T_wrt_IMU_;
};

}  // namespace lightning
```

## 5. SlamSystem 改造

### 5.1 头文件变更

```cpp
// src/core/system/slam.h
// 变更前:
// #include "core/lio/laser_mapping.h"
// std::shared_ptr<LaserMapping> lio_ = nullptr;

// 变更后:
#include "core/frontend/lio_frontend.h"
std::shared_ptr<LIOFrontend> lio_ = nullptr;

// 新增方法
std::vector<Keyframe::Ptr> GetAllKeyframes();
```

### 5.2 Init 变更

```cpp
// src/core/system/slam.cc
bool SlamSystem::Init(const std::string& yaml_path) {
    auto yaml = YAML::LoadFile(yaml_path);

    // 选择前端
    std::string frontend_type = yaml["system"]["mapping_frontend"]
        .as<std::string>("aa_fasterlio");

    if (frontend_type == "aa_fasterlio") {
        lio_ = std::make_shared<LaserMapping>();
    } else if (frontend_type == "fast_lio2") {
        lio_ = std::make_shared<FASTLIO2Mapping>();
    } else {
        LOG(ERROR) << "unknown frontend: " << frontend_type;
        return false;
    }

    lio_->Init(yaml_path);

    // 后端不变...
}
```

## 6. 关键修复 (调试过程中发现的问题)

### 6.1 UI 点云双重变换 (根本原因)

**问题**: UI 中点云没有配准，显示为散乱点云。

**原因**: `ui_cloud.cc` 的 `SetCloud()` 会对点云应用 pose 变换：

```cpp
auto pt_world = pose_l * cloud->points[id].getVector3fMap();
```

但 fast_lio2 传给 UI 的是 `scan_down_world_`（已经是世界坐标），导致双重变换。

**修复**: 传 `scan_down_body_`（体坐标系）给 UI：

```cpp
// fastlio2_mapping.cc
// 修复前:
ui_->UpdateScan(scan_down_world_, GetPose());

// 修复后:
ui_->UpdateScan(scan_down_body_, GetPose());
```

### 6.2 关键帧双重变换

**问题**: 保存的地图点云错位。

**原因**: `MakeKF()` 存储 `scan_down_world_`，但 `GetGlobalMap()` 再次用 pose 变换它。

**修复**: 关键帧存储 `scan_down_body_`：

```cpp
// fastlio2_mapping.cc
// 修复前:
auto kf = std::make_shared<Keyframe>(kf_id_++, scan_down_world_, nav_state);

// 修复后:
auto kf = std::make_shared<Keyframe>(kf_id_++, scan_down_body_, nav_state);
```

### 6.3 高度 ROI 缺失

**问题**: 点云稀疏，缺少天花板点（z≈3-4m）。

**原因**: fast_lio2 预处理器默认高度 ROI `[-1.0, 1.0]`，天花板点被过滤。aa_fasterlio 的 LaserMapping 会设置 `SetHeightROI(-2.0, 5.0)`。

**修复**: 在 `FASTLIO2Mapping::Init()` 中读取配置并设置 ROI：

```cpp
// fastlio2_mapping.cc
if (yaml["roi"]) {
    float height_max = yaml["roi"]["height_max"].as<float>(5.0);
    float height_min = yaml["roi"]["height_min"].as<float>(-2.0);
    preprocess_->SetHeightROI(height_max, height_min);
}
```

### 6.4 IMU-雷达时间同步

**问题**: 点云没有配准，z 轴有 ~0.05m 偏移。

**原因**: `SyncPackages()` 使用固定估计值 `lidar_mean_scantime_` 计算 `lidar_end_time_`，而非从实际点云时间戳计算。

**修复**: 从点云实际时间戳计算 `lidar_end_time_`：

```cpp
// fastlio2_mapping.cc SyncPackages()
if (cloud->points.size() <= 1) {
    lidar_end_time_ = beg_time + lidar_mean_scantime_;
} else if (cloud->points.back().time / 1000.0 < 0.5 * lidar_mean_scantime_) {
    lidar_end_time_ = beg_time + lidar_mean_scantime_;
} else {
    lidar_end_time_ = beg_time + cloud->points.back().time / 1000.0;
    lidar_mean_scantime_ += (cloud->points.back().time / 1000.0 - lidar_mean_scantime_) / scan_num_;
}
```

### 6.5 首帧状态同步

**问题**: 第一帧点云使用初始零/单位阵状态。

**原因**: `state_point_` 仅在 `Observe()` 中更新，但首帧不调用 `Observe()`。

**修复**: IMU 前向传播后立即同步 `state_point_`：

```cpp
// fastlio2_core.cc IMUProcess()
kf_.predict(dt, Q_, in);
imu_state = kf_.get_x();
state_point_ = imu_state;  // 新增：同步状态
```

### 6.6 首帧 IMU 传播时间步长

**问题**: 首帧 IMU 传播 dt=0.1s（过大），导致初始位姿偏差。

**原因**: `last_imu_pose_.offset_time` 初始化为 `measures.lidar_beg_time`，与首个 IMU 样本时间差 ~0.1s。

**修复**: 初始化为首个 IMU 样本的时间戳：

```cpp
// fastlio2_core.cc IMUProcess()
last_imu_pose_.offset_time = measures.imu.front()->timestamp;
```

### 6.7 配置不一致

**问题**: `fast_lio2.scan_line` 为 32，实际 RoboSense 为 128 线。

**修复**: 统一配置：

```yaml
fast_lio2:
  scan_line: 128  # 与 fasterlio.scan_line 一致
```

## 7. 配置文件

```yaml
# config/default_linghou.yaml

system:
  mapping_frontend: "fast_lio2"  # 可选: aa_fasterlio 或 fast_lio2

# FAST-LIO2 专用配置
fast_lio2:
  lidar_type: 4                # 1=Livox, 2=Velodyne, 3=Ouster, 4=RoboSense
  scan_line: 128               # 必须与 fasterlio.scan_line 一致
  blind: 0.5
  filter_size_scan: 0.5
  filter_size_map: 0.5
  max_iteration: 4
  acc_cov: 0.1
  gyr_cov: 0.1
  b_acc_cov: 0.0001
  b_gyr_cov: 0.0001
  extrinsic_T: [0, 0, 0.0]
  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]

# 高度 ROI（两个前端共用）
roi:
  height_max: 5.0
  height_min: -2.0
```

## 8. 数据流

```
PointCloud2 消息
    ↓
Localization::ProcessLidarMsg()
    ↓ PointCloudPreprocess (point_filter_num=4, height ROI)
    ↓
lio_->ProcessPointCloud2(CloudPtr)  ← 已预处理的点云
    ↓ 推入 lidar_buffer_
    ↓
lio_->Run()
    ↓ SyncPackages()         ← 从点云时间戳计算 lidar_end_time_
    ↓ IMUProcess()           ← ESIKF 前向传播 + 去畸变
    ↓   → feats_down_body_   ← 体坐标系降采样点云
    ↓   → feats_down_world_  ← 世界坐标系降采样点云
    ↓ Observe()              ← ikd-Tree 最近邻 + 点到面 ICP + ESEKF 更新
    ↓ UpdateMap()            ← 添加点到 ikd-Tree
    ↓ ui_->UpdateScan(scan_down_body_, pose)  ← 传体坐标系点云
    ↓ MakeKF()               ← 存储 scan_down_body_（体坐标系）
    ↓
Keyframe → LoopClosing / G2P5 / TiledMap
```

## 9. 注意事项

### 9.1 点云坐标系

- `scan_down_body_`: 体坐标系，用于 UI 显示和关键帧存储
- `scan_down_world_`: 世界坐标系，用于 ikd-Tree 地图更新
- **不要**将 `scan_down_world_` 传给 UI 或存储到关键帧（会导致双重变换）

### 9.2 高度 ROI

两个前端都必须设置高度 ROI，否则天花板/地面点会被过滤：

- Localization 预处理器已设置 ROI
- 每个前端的预处理器也需要设置 ROI（用于 PointCloud2/CustomMsg 直接输入路径）

### 9.3 IMU 时间同步

`last_lidar_end_time_` 初始化为 `0.0`（不是 `-1.0`），确保首帧 IMU 传播使用正确的 dt。

### 9.4 ESEKF 状态表示

- 重力使用 S2 流形表示（2D 参数化 3D 单位向量）
- 与 aa_fasterlio 的直接 3D 向量表示略有差异
- 两者轨迹对比平均误差 ~0.04m（51s 灵猴数据集）

## 10. 验证清单

- [X]  原有 `aa_fasterlio` 模式功能正常
- [X]  `fast_lio2` 模式建图功能正常
- [X]  UI 点云正确显示（体坐标系 + pose 变换）
- [X]  关键帧存储正确（体坐标系）
- [X]  高度 ROI 正确（天花板点保留）
- [X]  IMU-雷达时间同步正确
- [X]  灵猴数据集测试通过
- [ ]  NCLT 数据集测试
- [ ]  处理时间 < 100ms/帧
