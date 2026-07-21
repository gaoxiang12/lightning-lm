# BEVPlace2 集成设计文档

## 1. 目标

将 BEVPlace2（基于 BEV 图像的 LiDAR 全局定位方法）集成到 Lightning-LM 中，作为可切换的回环检测/全局定位策略。

- **回环检测**: 可选使用 BEVPlace2 描述子检索 或 空间距离过滤
- **位姿验证**: 可选使用 BEVPlace2 BEV 特征匹配 或 NDT 匹配
- **统一接口**: 通过多态策略接口实现，YAML 参数切换
- **可扩展性**: 后续集成其它方法（NetVLAD、Hloc 等）只需实现接口

## 2. 架构设计

### 2.1 现有架构

```
LoopClosing (单一类，职责耦合)
├── DetectLoopCandidates()    → 空间距离过滤
├── ComputeLoopCandidates()   → NDT 位姿验证
└── PoseOptimization()        → 图优化 (miao)
```

### 2.2 目标架构

```
LoopClosing (编排器，策略解耦)
├── LoopDetector* detector_           ← 多态指针
│   ├── SpatialLoopDetector           (空间距离过滤，默认)
│   └── BEVPlace2LoopDetector         (BEVPlace2 描述子检索)
├── LoopPoseEstimator* pose_estimator_ ← 多态指针
│   ├── NDTPoseEstimator              (NDT 位姿验证，默认)
│   └── BEVPlace2PoseEstimator        (BEVPlace2 特征匹配)
└── PoseOptimization()                → 图优化 (不变)
```

### 2.3 与 LIOFrontend 模式的一致性

本次设计与已有的 `LIOFrontend` 多态模式保持一致：

| 维度 | LIOFrontend (前端) | LoopDetector/PoseEstimator (回环) |
|------|-------------------|----------------------------------|
| 基类 | `LIOFrontend` | `LoopDetector` / `LoopPoseEstimator` |
| 工厂 | `SlamSystem::Init()` if-else | `CreateLoopDetector()` / `CreateLoopPoseEstimator()` |
| 配置 | `system.mapping_frontend` | `loop_closing.detector` / `loop_closing.pose_estimator` |
| 实现 | `LaserMapping`, `FASTLIO2Mapping` | `Spatial`, `NDT`, `BEVPlace2` |

## 3. 接口定义

### 3.1 LoopDetector — 回环候选检测接口

```cpp
// src/core/loop_closing/loop_detector.h
namespace lightning {

struct LoopDetectResult {
    std::vector<LoopCandidate> candidates_;
};

class LoopDetector {
public:
    virtual ~LoopDetector() = default;

    /// 初始化（从 YAML 读取参数）
    virtual void Init(const std::string& yaml_path) = 0;

    /// 添加新关键帧（维护数据库/索引）
    virtual void AddKeyframe(Keyframe::Ptr kf) = 0;

    /// 检测回环候选
    virtual LoopDetectResult Detect(Keyframe::Ptr cur_kf) = 0;
};

}  // namespace lightning
```

### 3.2 LoopPoseEstimator — 回环位姿估计接口

```cpp
// src/core/loop_closing/loop_pose_estimator.h
namespace lightning {

class LoopPoseEstimator {
public:
    virtual ~LoopPoseEstimator() = default;

    virtual void Init(const std::string& yaml_path) = 0;

    /// 估计回环候选的相对位姿
    /// @param c 候选帧对（输入 Tij_ 为初始估计，输出更新后的 Tij_ 和分数）
    /// @param all_kfs 所有关键帧（用于构建子图等）
    virtual void Estimate(LoopCandidate& c,
                          const std::vector<Keyframe::Ptr>& all_kfs) = 0;
};

}  // namespace lightning
```

## 4. 现有实现（从 LoopClosing 提取）

### 4.1 SpatialLoopDetector

从 `LoopClosing::DetectLoopCandidates()` 提取（`loop_closing.cc:96-143`）：

```cpp
class SpatialLoopDetector : public LoopDetector {
public:
    void Init(const std::string& yaml_path) override;
    void AddKeyframe(Keyframe::Ptr kf) override;
    LoopDetectResult Detect(Keyframe::Ptr cur_kf) override;

private:
    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_loop_kf_ = nullptr;

    // 参数
    int loop_kf_gap_ = 20;
    int min_id_interval_ = 20;
    int closest_id_th_ = 50;
    double max_range_ = 30.0;
};
```

### 4.2 NDTPoseEstimator

从 `LoopClosing::ComputeForCandidate()` 提取（`loop_closing.cc:168-251`）：

```cpp
class NDTPoseEstimator : public LoopPoseEstimator {
public:
    void Init(const std::string& yaml_path) override;
    void Estimate(LoopCandidate& c,
                  const std::vector<Keyframe::Ptr>& all_kfs) override;

private:
    double ndt_score_th_ = 1.0;
    int submap_idx_range_ = 40;
};
```

## 5. BEVPlace2 策略实现

### 5.1 BEVPlace2LoopDetector

基于 BEVPlace2 的全局描述子检索替代空间距离过滤：

```cpp
class BEVPlace2LoopDetector : public LoopDetector {
public:
    void Init(const std::string& yaml_path) override;
    void AddKeyframe(Keyframe::Ptr kf) override;
    LoopDetectResult Detect(Keyframe::Ptr cur_kf) override;

private:
    /// 从点云生成 BEV 图像
    /// - 体素降采样 0.4m
    /// - 裁剪到 [-40m, +40m]
    /// - 投影到 200x200 灰度图
    cv::Mat GenerateBEVImage(CloudPtr cloud);

    /// 提取全局描述子（8192-dim L2 归一化）
    std::vector<float> ExtractDescriptor(const cv::Mat& bev_image);

    // 模型和索引
    // void* model_;           // PyTorch/LibTorch 模型
    // void* faiss_index_;     // FAISS 索引

    // 参数
    double descriptor_match_th_ = 0.5;
    double bev_resolution_ = 0.4;
    int bev_size_ = 200;
    double bev_range_ = 40.0;
};
```

**工作流程**：
1. `AddKeyframe()`: 生成 BEV 图像 → 提取描述子 → 加入 FAISS 索引
2. `Detect()`: 生成当前帧 BEV → 提取描述子 → FAISS Top-1 检索 → 距离 < 阈值则为候选

### 5.2 BEVPlace2PoseEstimator

基于 BEV 特征匹配 + RANSAC 的位姿估计：

```cpp
class BEVPlace2PoseEstimator : public LoopPoseEstimator {
public:
    void Init(const std::string& yaml_path) override;
    void Estimate(LoopCandidate& c,
                  const std::vector<Keyframe::Ptr>& all_kfs) override;

private:
    /// 提取局部特征（128-dim, 200x200）
    cv::Mat ExtractLocalFeatures(const cv::Mat& bev_image);

    /// FAST 特征检测 + BFMatcher + RANSAC 位姿估计
    SE3 EstimatePose(const cv::Mat& feat1, const cv::Mat& feat2);
};
```

**工作流程**：
1. 对两帧 BEV 图像提取局部特征
2. 检测 FAST 关键点
3. BFMatcher 匹配特征
4. `rigidRansac()` 估计 2D 刚体变换（x, y, yaw）
5. 转换为 SE3 相对位姿

## 6. 工厂函数

```cpp
// src/core/loop_closing/loop_closing_factory.h
namespace lightning {

std::unique_ptr<LoopDetector> CreateLoopDetector(
    const std::string& type, const std::string& yaml_path);

std::unique_ptr<LoopPoseEstimator> CreateLoopPoseEstimator(
    const std::string& type, const std::string& yaml_path);

}  // namespace lightning
```

```cpp
// src/core/loop_closing/loop_closing_factory.cc
std::unique_ptr<LoopDetector> CreateLoopDetector(
    const std::string& type, const std::string& yaml_path) {
    std::unique_ptr<LoopDetector> d;
    if (type == "spatial") {
        d = std::make_unique<SpatialLoopDetector>();
    } else if (type == "bevplace2") {
        d = std::make_unique<BEVPlace2LoopDetector>();
    } else {
        LOG(ERROR) << "unknown loop detector: " << type;
        return nullptr;
    }
    d->Init(yaml_path);
    return d;
}

std::unique_ptr<LoopPoseEstimator> CreateLoopPoseEstimator(
    const std::string& type, const std::string& yaml_path) {
    std::unique_ptr<LoopPoseEstimator> e;
    if (type == "ndt") {
        e = std::make_unique<NDTPoseEstimator>();
    } else if (type == "bevplace2") {
        e = std::make_unique<BEVPlace2PoseEstimator>();
    } else {
        LOG(ERROR) << "unknown loop pose estimator: " << type;
        return nullptr;
    }
    e->Init(yaml_path);
    return e;
}
```

## 7. YAML 配置

```yaml
# config/default_nclt.yaml（新增字段）
loop_closing:
  # 回环检测策略: spatial | bevplace2
  detector: spatial
  # 位姿估计策略: ndt | bevplace2
  pose_estimator: ndt

  # 通用参数
  loop_kf_gap: 20
  min_id_interval: 20
  closest_id_th: 50
  max_range: 30.0
  with_height: true
  height_noise: 0.1

  # 图优化参数
  motion_trans_noise: 0.1
  motion_rot_noise: 0.05236
  loop_trans_noise: 0.2
  loop_rot_noise: 0.05236
  rk_loop_th: 1.04

# BEVPlace2 专用参数
bevplace2:
  model_path: ./src/BEVPlace2/runs/Aug08_10-17-29/model_best.pth.tar
  bev_resolution: 0.4
  bev_size: 200
  bev_range: 40.0
  descriptor_match_th: 0.5
  use_gpu: true
```

## 8. 改造后的 LoopClosing

```cpp
class LoopClosing {
public:
    struct Options {
        bool verbose_ = true;
        bool online_mode_ = false;
        std::string detector_type_ = "spatial";
        std::string pose_estimator_type_ = "ndt";
        // 图优化参数...
    };

    void Init(const std::string yaml_path);
    void AddKF(Keyframe::Ptr kf);
    void SetLoopClosedCB(LoopClosedCallback cb) { loop_cb_ = cb; }

private:
    void HandleKF(Keyframe::Ptr kf);
    void PoseOptimization();

    Options options_;

    // 策略组件（多态）
    std::unique_ptr<LoopDetector> detector_;
    std::unique_ptr<LoopPoseEstimator> pose_estimator_;

    // 图优化（不变）
    std::shared_ptr<miao::Optimizer> optimizer_;
    std::vector<std::shared_ptr<miao::VertexSE3>> kf_vert_;
    std::vector<std::shared_ptr<miao::EdgeSE3>> edge_loops_;

    // 数据
    std::vector<LoopCandidate> candidates_;
    std::vector<Keyframe::Ptr> all_keyframes_;
    Keyframe::Ptr last_kf_ = nullptr;
    Keyframe::Ptr cur_kf_ = nullptr;
    AsyncMessageProcess<Keyframe::Ptr> kf_thread_;
    LoopClosedCallback loop_cb_;
};
```

改造后的 `LoopClosing::HandleKF()`：

```cpp
void LoopClosing::HandleKF(Keyframe::Ptr kf) {
    if (kf == last_kf_) return;

    cur_kf_ = kf;
    all_keyframes_.emplace_back(kf);

    // 1. 策略化的候选检测
    detector_->AddKeyframe(kf);
    auto result = detector_->Detect(kf);
    candidates_ = result.candidates_;

    // 2. 策略化的位姿估计
    for (auto& c : candidates_) {
        pose_estimator_->Estimate(c, all_keyframes_);
    }

    // 过滤成功的候选
    std::vector<LoopCandidate> succ;
    for (auto& c : candidates_) {
        if (c.ndt_score_ > 0) {  // 分数含义由策略定义
            succ.emplace_back(c);
        }
    }
    candidates_.swap(succ);

    // 3. 图优化（不变）
    PoseOptimization();

    last_kf_ = kf;
}
```

## 9. 文件清单

### 9.1 新增文件

| 文件 | 用途 |
|------|------|
| `src/core/loop_closing/loop_detector.h` | LoopDetector 抽象接口 |
| `src/core/loop_closing/loop_pose_estimator.h` | LoopPoseEstimator 抽象接口 |
| `src/core/loop_closing/spatial_loop_detector.h/.cc` | 空间距离检测（从现有代码提取） |
| `src/core/loop_closing/ndt_pose_estimator.h/.cc` | NDT 位姿验证（从现有代码提取） |
| `src/core/loop_closing/bevplace2_loop_detector.h/.cc` | BEVPlace2 描述子检索 |
| `src/core/loop_closing/bevplace2_pose_estimator.h/.cc` | BEVPlace2 特征匹配位姿估计 |
| `src/core/loop_closing/loop_closing_factory.h/.cc` | 工厂函数 |

### 9.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `src/core/loop_closing/loop_closing.h` | 重构为编排器，持有策略指针 |
| `src/core/loop_closing/loop_closing.cc` | 调用策略接口替代内联逻辑 |
| `src/core/system/slam.cc` | 传递 `loop_closing.detector` / `loop_closing.pose_estimator` 配置 |
| `src/CMakeLists.txt` | 添加新源文件 |

### 9.3 新增配置

| 文件 | 说明 |
|------|------|
| `config/default_nclt_bevplace2.yaml` | 使用 BEVPlace2 策略的 NCLT 配置示例 |

## 10. 实施步骤

### Phase 1: 接口提取（无行为变化）

1. 创建 `loop_detector.h` 和 `loop_pose_estimator.h` 接口头文件
2. 从 `LoopClosing::DetectLoopCandidates()` 提取 `SpatialLoopDetector`
3. 从 `LoopClosing::ComputeForCandidate()` 提取 `NDTPoseEstimator`
4. 创建工厂函数 `loop_closing_factory.h/.cc`
5. 改造 `LoopClosing` 使用策略指针
6. **验证**: 现有 NDT 回环行为不变

### Phase 2: BEVPlace2 策略实现

7. 在 BEVPlace2 中添加 C++ 推理接口（LibTorch 或 pybind11）
8. 实现 `BEVPlace2LoopDetector`（描述子提取 + FAISS 检索）
9. 实现 `BEVPlace2PoseEstimator`（FAST 特征 + BFMatcher + RANSAC）
10. 添加 YAML 配置支持

### Phase 3: 测试与验证

11. 创建 `config/default_nclt_bevplace2.yaml`
12. 运行 NCLT 数据集对比测试
13. 验证图优化结果一致性

## 11. 验证方法

```bash
# 基线测试（spatial + ndt）
colcon build --packages-select lightning
ros2 run lightning run_slam_offline \
  --config ./src/lightning-lm/config/default_nclt.yaml \
  --input_bag <bag>

# BEVPlace2 测试
colcon build --packages-select lightning
ros2 run lightning run_slam_offline \
  --config ./src/lightning-lm/config/default_nclt_bevplace2.yaml \
  --input_bag <bag>

# 对比指标
# - 回环检测数量
# - 图优化后轨迹精度
# - 运行时间
```

## 12. 扩展性

后续集成其它方法（如 NetVLAD、Hloc、AnyLoc 等）只需：

1. 实现 `LoopDetector` 接口（描述子提取 + 检索）
2. 实现 `LoopPoseEstimator` 接口（位姿估计）
3. 在工厂函数中添加分支
4. 在 YAML 配置中添加新选项

**无需修改** `LoopClosing` 编排器或图优化代码。
