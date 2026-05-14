
# lightning-lm LIO-SAM 离线建图前端说明

本文档说明 lightning-lm 中新增的 **LIO-SAM 离线建图前端分支**。

该分支的目标不是完整复刻原版 LIO-SAM 的 ROS topic 节点系统，而是把 LIO-SAM 的核心离线建图链路集成到 lightning-lm 中，作为原 `LaserMapping / FAST-LIO` 前端的一个可选替代分支。

当前新增分支的定位是：

```text
LIO-SAM Offline Mapping Core for lightning-lm
```

也就是说，它保留 LIO-SAM 的核心建图算法：

```text
ImageProjection
FeatureExtraction
mapOptimization
GTSAM / iSAM2
LIO-SAM 内部 loop closure
```

同时去掉原版 LIO-SAM 的 ROS topic handoff、IMUPreintegration、TransformFusion、在线 TF 发布、GPSFactor 等在线系统组件。

---

## 1. 设计目标

新增 LIO-SAM 分支的主要目标是：

```text
1. 在 lightning-lm 中增加一个可参数选择的 LIO-SAM 离线建图前端。
2. 保持原有 LaserMapping / FAST-LIO 分支不受影响。
3. LIO-SAM 分支只用于 offline mapping，不用于在线实时定位。
4. 不使用原版 LIO-SAM 的 IMUPreintegration 和 TransformFusion。
5. 不通过 ROS topic 在 ImageProjection、FeatureExtraction、mapOptimization 之间传递数据。
6. 改为函数调用式数据流。
7. 使用 lightning-lm 的 rosbag 离线读取和保存地图框架。
8. 使用 LIO-SAM 自己的 mapOptimization / GTSAM / loop closure 作为该分支后端。
9. 在 LIO-SAM 分支启用时，自动关闭 lightning-lm 自己的 LoopClosing，避免双后端优化。
10. 增加坏帧检测和保护逻辑，避免发散位姿污染 keyframe 和因子图。
```

当前代码中，`SlamSystem` 已经新增 `use_lio_sam_` 分支，并在 LIO-SAM 模式下创建 `LioSamMapping`，而不是创建原 `LaserMapping`。`SlamSystem` 还会在 LIO-SAM 分支下关闭 lightning-lm 自己的 LoopClosing。

---

## 2. 总体架构

新增后的 lightning-lm 前端架构如下：

```text
frontend = faster_lio / fastlio / 默认
    -> LaserMapping
    -> lightning-lm 后端 / LoopClosing / UI / SaveMap

frontend = lio_sam / liosam
    -> LioSamMapping
        -> ImageProjection
        -> FeatureExtraction
        -> mapOptimization
    -> LIO-SAM 内部 GTSAM / loop closure
    -> lightning-lm UI / SaveMap / G2P5
```

`LioSamMapping` 是一个和 `LaserMapping` 平行的类。它内部持有三个 LIO-SAM 核心模块：

```cpp
std::unique_ptr<::ImageProjection> image_projection_;
std::unique_ptr<::FeatureExtraction> feature_extraction_;
std::unique_ptr<::mapOptimization> map_optimization_;
```

并且维护自己的 LiDAR buffer、IMU buffer、同步数据包、当前 scan、当前状态、keyframe 列表。

---

## 3. 数据流

当前 LIO-SAM 分支的数据流为：

```text
rosbag
  |
  |-- sensor_msgs::msg::Imu
  |       |
  |       v
  |   SlamSystem::ProcessIMU()
  |       |
  |       v
  |   LioSamMapping::ProcessIMU()
  |       |
  |       v
  |   imu_buffer_
  |
  |-- sensor_msgs::msg::PointCloud2 / livox_ros_driver2::msg::CustomMsg
          |
          v
      SlamSystem::ProcessLidar()
          |
          v
      LioSamMapping::ProcessPointCloud2()
          |
          v
      lidar_buffer_
          |
          v
      LioSamMapping::Run()
          |
          v
      SyncPackages()
          |
          v
      ImageProjection::Run()
          |
          v
      FeatureExtraction::Run()
          |
          v
      mapOptimization::Run()
          |
          v
      MakeLightningKeyframeIfNeeded()
          |
          v
      SyncLightningKeyframePoses()
          |
          v
      UI / SaveMap / G2P5
```

离线 rosbag 读取中，LIO-SAM 分支需要原始 ROS IMU 消息，而不是 lightning-lm 内部转换后的 `IMUPtr`，因为 LIO-SAM 的去畸变和姿态初值依赖 `sensor_msgs::msg::Imu` 中的 `orientation`、`angular_velocity` 和 `linear_acceleration`。当前 `RosbagIO` 已经支持原始 `sensor_msgs::msg::Imu` 回调。

---

## 4. 配置方式

在配置文件中选择 LIO-SAM 前端：

```yaml
frontend: lio_sam
```

或者：

```yaml
frontend_type: lio_sam
```

具体字段名取决于当前配置文件读取逻辑。当前代码会兼容 `lio_sam` / `liosam` 这类名称。

LIO-SAM 相关参数建议放在：

```yaml
lio_sam:
  sensor: velodyne
  N_SCAN: 16
  Horizon_SCAN: 1800
  downsampleRate: 1

  lidarMinRange: 0.5
  lidarMaxRange: 200.0

  useImuHeadingInitialization: false
  useImuAccelRollPitchInitialization: true

  edgeThreshold: 1.0
  surfThreshold: 0.1
  edgeFeatureMinValidNum: 10
  surfFeatureMinValidNum: 100

  odometrySurfLeafSize: 0.4
  mappingCornerLeafSize: 0.2
  mappingSurfLeafSize: 0.4

  surroundingkeyframeAddingDistThreshold: 1.0
  surroundingkeyframeAddingAngleThreshold: 0.2
  surroundingKeyframeDensity: 2.0
  surroundingKeyframeSearchRadius: 50.0

  loopClosureEnableFlag: true

  mappingMotionGateEnable: true
  mappingIcpFallbackEnable: false
  mappingMotionMaxSpeed: 3.0
  mappingMotionMaxAngularVelocity: 90.0
  mappingMotionMaxCurvature: 2.0
  mappingMotionMaxRollPitchDeg: 20.0
```

当前版本的 LIO-SAM 分支只支持 offline mapping。如果在 online mode 下选择 `lio_sam`，初始化应直接报错退出。

---

## 5. 运行方式

离线建图命令示例：

```bash
ros2 run lightning run_slam_offline \
  --input_bag /path/to/data.db3 \
  --config /path/to/my_mapping.yaml
```

建图完成后，地图会由 lightning-lm 的 `SaveMap()` 逻辑保存。LIO-SAM 分支下保存地图时，会从 `lio_sam_->GetAllKeyframes()` 和 `lio_sam_->GetGlobalMap()` 获取 keyframe 和全局地图，而不是从原 `LaserMapping` 获取。

默认保存路径通常为：

```text
./data/new_map/
```

其中 `./` 表示运行命令时的当前工作目录。

---

## 6. 与原版 LIO-SAM 的主要区别

### 6.1 原版 LIO-SAM 的结构

原版 LIO-SAM 是 ROS topic 节点式结构：

```text
ImageProjection
  subscribe: point cloud / IMU / incremental odom
  publish: deskewed cloud + CloudInfo

FeatureExtraction
  subscribe: deskewed CloudInfo
  publish: feature CloudInfo

mapOptimization
  subscribe: feature CloudInfo / GPS / loop info
  publish: mapping odometry / path / local map / global map
```

原版 `ImageProjection` 订阅 LiDAR、IMU 和 `odomTopic + "_incremental"`，内部维护 `cloudQueue`、`imuQueue`、`odomQueue`，并在 `cloudHandler()` 中执行点云缓存、去畸变、投影和 CloudInfo 发布。

原版 `FeatureExtraction` 订阅 `lio_sam/deskew/cloud_info`，执行曲率计算、遮挡点标记和角点/面点提取，然后发布 feature cloud info。

原版 `mapOptimization` 订阅 feature cloud info、GPS 和外部 loop info，并负责 scan-to-map 优化、GTSAM/iSAM2、GPSFactor、loop factor、位姿发布、TF 发布、路径发布和 save map service。

---

### 6.2 当前 LIO-SAM offline 分支的结构

当前分支删除了 ROS topic handoff，改成函数调用：

```text
LioSamMapping::Run()
  -> ImageProjection::Run()
  -> FeatureExtraction::Run()
  -> mapOptimization::Run()
```

当前 `ImageProjection` 不再创建 ROS subscriber/publisher，而是通过函数参数接收当前点云、同步好的 IMU 序列、LiDAR scan begin time 和 scan end time。

当前 `FeatureExtraction` 不再订阅和发布 `lio_sam::msg::CloudInfo`，而是直接读写 `LioSamCloudInfo`。

当前 `mapOptimization` 不再作为 ROS topic callback 节点运行，而是由 `LioSamMapping` 逐帧调用 `Run()`。它保留了 GTSAM / iSAM2、scan-to-map、loop closure、keyframe 和 `correctPoses()` 等核心功能。

---

## 7. 保留的原版 LIO-SAM 核心算法

当前 offline 分支保留了以下 LIO-SAM 核心算法：

```text
1. IMU rotation deskew。
2. range image projection。
3. cloudExtraction。
4. calculateSmoothness。
5. markOccludedPoints。
6. extractFeatures。
7. scan-to-map LM optimization。
8. cornerOptimization。
9. surfOptimization。
10. LMOptimization。
11. degeneracy handling。
12. keyframe selection。
13. odom factor。
14. loop factor。
15. GTSAM / iSAM2 incremental optimization。
16. performLoopClosure。
17. correctPoses。
```

因此，这个分支保留的是 **LIO-SAM mapping core**，不是简单地只拿了 LIO-SAM 的前端特征提取。

---

## 8. 删除或替换的原版功能

当前分支删除或替换了以下原版 LIO-SAM 功能：

```text
1. 删除 ImageProjection / FeatureExtraction / mapOptimization 之间的 ROS topic 通信。
2. 删除 lio_sam::msg::CloudInfo 作为模块间通信载体，改用 LioSamCloudInfo。
3. 删除 IMUPreintegration 节点。
4. 删除 TransformFusion 节点。
5. 删除 odomTopic + "_incremental" 依赖。
6. 删除 ImageProjection 中的 odomDeskewInfo。
7. 删除 mapOptimization 中的 ROS odometry / path / TF publisher。
8. 删除 LIO-SAM save_map service，改用 lightning-lm SaveMap。
9. 删除 GPSFactor / addGPSFactor。
10. 删除外部 loopInfo topic 输入。
```

原版 `imuPreintegration.cpp` 中包含 `TransformFusion`、IMU factor、CombinedImuFactor、incremental odom 发布等在线系统逻辑。当前 offline 分支没有接入这一套逻辑。

---

## 9. SyncPackages 离线同步逻辑

原版 LIO-SAM 依赖 ROS callback 队列和内部 queue 做数据同步。当前 offline 分支使用 `LioSamMapping::SyncPackages()` 明确同步每一帧 LiDAR 和 IMU。

同步逻辑为：

```text
1. 如果 LiDAR 或 IMU buffer 为空，返回 false。
2. 从 lidar_buffer_ 取出当前 cloud。
3. 根据点云内最后一个点的相对时间计算 scan duration。
4. lidar_begin_time = cloud.header.stamp。
5. lidar_end_time = lidar_begin_time + scan_duration。
6. 如果 IMU 最新时间小于 lidar_end_time，等待更多 IMU。
7. 如果最早 IMU 晚于 lidar_begin_time，说明缺少 scan start 前 IMU，丢掉该 cloud。
8. 收集覆盖 lidar_begin_time 到 lidar_end_time + margin 的 IMU。
9. 同步成功后弹出当前 LiDAR。
10. 保留 IMU 边界，供下一帧使用。
```

当前使用的时间单位是：

```text
LiDAR point.time: 秒
Ouster point.t: 纳秒，转换为秒
ROS header.stamp: 秒
```

时间转换函数可由 lightning-lm 的 `ToSec()` 工具完成。

注意：

```text
当前实现假设点云的最后一个点时间可以代表 scan duration。
如果驱动输出点云不是按时间排序，建议改成 max(point.time)。
```

---

## 10. ImageProjection offline 说明

当前 `ImageProjection::Run()` 接收：

```cpp
const sensor_msgs::msg::PointCloud2& cloud_msg
const std::vector<sensor_msgs::msg::Imu>& imus
double lidar_begin_time
double lidar_end_time
LioSamCloudInfo& cloudInfoOut
```

处理流程为：

```text
1. 清空内部 imuQueue。
2. 对传入的 IMU 做 imuConverter。
3. cachePointCloud。
4. deskewInfo。
5. projectPointCloud。
6. cloudExtraction。
7. packCloudInfo。
8. resetParameters。
```

和原版相比：

```text
保留：
    点云类型转换
    IMU orientation 获取 scan start 姿态
    IMU angular_velocity 积分
    deskewPoint
    range image projection
    cloudExtraction

删除：
    cloudQueue
    imuHandler callback
    odometryHandler callback
    odomQueue
    odomDeskewInfo
    publishClouds
```

当前去畸变只使用 IMU 旋转，不使用 IMUPreintegration 输出的 incremental odom。

---

## 11. FeatureExtraction offline 说明

当前 `FeatureExtraction::Run()` 接收并修改 `LioSamCloudInfo`：

```text
输入：
    cloud_deskewed
    point_range
    point_col_ind
    start_ring_index
    end_ring_index

处理：
    calculateSmoothness
    markOccludedPoints
    extractFeatures

输出：
    cloud_corner
    cloud_surface
```

它和原版 `FeatureExtraction` 的算法基本等价，只是原版通过 ROS topic 订阅和发布 `lio_sam::msg::CloudInfo`，当前版本改为直接函数调用。

---

## 12. mapOptimization offline 说明

当前 `mapOptimization::Run()` 是 LIO-SAM offline 分支的核心。

处理流程为：

```text
1. 读取当前帧 timestamp。
2. 读取 corner / surface feature cloud。
3. resetFrameQuality。
4. updateInitialGuess。
5. extractSurroundingKeyFrames。
6. downsampleCurrentScan。
7. scan2MapOptimization。
8. performLoopClosure。
9. saveKeyFramesAndFactor。
10. correctPoses。
11. updateOdometryState。
```

和原版相比，它保留了 LIO-SAM 的核心 mapping backend，但移除了 ROS publisher、GPSFactor、外部 loop topic、save map service 等在线逻辑。当前版本还新增了坏帧处理和 motion gate。

---

## 13. 坏帧处理逻辑

当前版本新增了严格的 bad-frame rejection 机制。其目的如下：

```text
1. 防止 LM 发散位姿进入 keyframe。
2. 防止发散位姿进入 GTSAM 因子图。
3. 防止坏帧更新 last accepted tracking pose。
4. 防止连续坏帧后仍然强行建图。
```

核心状态包括：

```text
mappingPoseReliable
mappingPoseSource
mappingTrackingState
mappingFailureCount
lastAcceptedTrackingTransform
lastAcceptedTrackingTransformTime
lastAcceptedImuTransform
lowSpeedGuessPrevTrustedTransform
lowSpeedGuessLastTrustedTransform
frameInitialGuessTransform
```

### 13.1 初始猜测

每帧进入 scan-to-map 之前，`updateInitialGuess()` 会生成当前帧的初始位姿。

如果是第一帧：

```text
roll/pitch/yaw 来自 IMU scan start 姿态；
如果 useImuHeadingInitialization=false，则 yaw 置 0。
```

如果不是第一帧：

```text
初始位姿 = 最近一次 accepted tracking pose
        + 从最近 accepted IMU 姿态到当前 IMU 姿态的旋转增量
        + 低速平移预测
```

这里的关键设计是：**预测始终锚定在最近一次可靠位姿，而不是锚定在坏帧上。**

### 13.2 LM 接受条件

LM 结果必须同时满足：

```text
1. LM 正常运行。
2. LM 收敛。
3. transformTobeMapped 有限。
4. motion gate 通过。
```

如果通过，则：

```text
acceptMappingPose("LM")
```

否则进入 fallback 或 reject。

### 13.3 ICP fallback

当前代码保留 ICP fallback 逻辑，但是否启用由参数控制：

```yaml
mappingIcpFallbackEnable: false
```

当 `mappingIcpFallbackEnable=false` 时，LM 失败后不会运行 ICP fallback，而是直接 reject。

当 `mappingIcpFallbackEnable=true` 时，ICP 结果也必须满足：

```text
1. ICP converged。
2. fitness score 合格。
3. transform finite。
4. motion gate 通过。
```

才会被接受。

### 13.4 rejectMappingPose

坏帧被 reject 后：

```text
1. mappingPoseReliable = false。
2. mappingPoseSource = 失败原因。
3. transformTobeMapped 恢复为最近 accepted tracking pose。
4. mappingFailureCount++。
5. 当前帧不会保存为 keyframe。
6. 当前帧不会添加 odom factor。
7. 连续坏帧超过 kMaxConsecutiveMappingFailures 后终止离线建图。
```

当前最大连续坏帧数在代码中写死为：

```cpp
static constexpr int kMaxConsecutiveMappingFailures = 10;
```

这个策略适合 offline mapping：如果连续多帧都无法可靠匹配，继续强行运行很可能只会污染地图，因此直接停止并打印 fatal 更安全。

---

## 14. Motion Gate 逻辑

当前 motion gate 已经简化为：

```text
candidate pose vs lastAcceptedTrackingTransform
```

也就是说，所有候选位姿都和最近一次可靠 accepted pose 判断连续性。

检查项包括：

```text
1. finite。
2. translation speed。
3. angular velocity。
4. curvature。
5. roll/pitch。
```

当前版本不再使用之前复杂的 recovery mode、predicted reference、position innovation、yaw innovation 等逻辑。

这个设计的原因是：

```text
当前分支采用连续坏帧超过阈值直接退出的策略；
因此不需要复杂的坏帧 recovery 状态机；
所有 frame acceptance 只需要判断和最近可靠位姿是否连续。
```

---

## 15. lastAcceptedTrackingTransform 设计

`lastAcceptedTrackingTransform` 是当前 bad-frame 逻辑中最重要的参考位姿。

它的更新规则是：

```text
如果当前 reliable frame 没有成为 keyframe：
    直接用当前 accepted LM/ICP pose 更新 lastAcceptedTrackingTransform。

如果当前 reliable frame 成为 keyframe：
    先由 GTSAM/iSAM2 优化；
    再用 cloudKeyPoses6D->back() 的优化后 pose 更新 lastAcceptedTrackingTransform。
```

这样设计的原因是：

```text
1. 不是每个可靠 LM 位姿都会成为 keyframe。
2. 如果只用 keyframe pose 作为 tracking reference，中间的可靠非关键帧位姿会丢失。
3. 如果当前帧成为 keyframe，则后端优化后的 pose 比 LM 原始 pose 更可信。
```

因此：

```text
lastAcceptedTrackingTransform 代表“最近一次可信 tracking pose”，
而不是“最近一次 keyframe pose”。
```

---

## 16. Loop Closure 策略

当前 LIO-SAM 分支使用 LIO-SAM 自己的内部回环：

```text
mapOptimization::performLoopClosure()
mapOptimization::addLoopFactor()
mapOptimization::correctPoses()
```

当 LIO-SAM 前端启用时，lightning-lm 自己的 LoopClosing 会自动关闭。

这样设计的原因是：

```text
LIO-SAM 的 scan-to-map、keyframe pose、local map cache、GTSAM graph、loop closure 是一体的。
如果让 lightning-lm 外部 LoopClosing 也优化同一批 keyframe，就会出现两个后端同时管理 OptPose 的问题。
```

因此当前策略是：

```text
LIO-SAM branch:
    LIO-SAM 自己负责 loop closure 和 GTSAM/iSAM2 优化；
    lightning-lm 只负责 UI、地图保存、统一入口和可视化。
```

---

## 17. Keyframe 与地图保存

当前只有通过 `mapOptimization` 保存的 LIO-SAM keyframe 才会进入 lightning-lm 的 `all_keyframes_`。

`MakeLightningKeyframeIfNeeded()` 会检查：

```text
1. mapOptimization 是否创建了新 keyframe。
2. cloudKeyPoses6D 是否增长。
3. rawCloudKeyFrames 是否有对应点云。
```

然后把 LIO-SAM keyframe 转成 lightning-lm 的 `Keyframe`。

保存地图时，LIO-SAM 分支会从：

```cpp
lio_sam_->GetAllKeyframes()
lio_sam_->GetGlobalMap()
```

获取数据。

注意：

```text
UI 当前 scan 显示的是当前处理帧，不一定是 keyframe。
最终保存地图使用的是 accepted keyframes。
```

---

## 18. UI 显示说明

当前 UI 有两类显示：

```text
1. 当前 scan 显示：
    每个成功处理的帧都可能调用 UpdateScan。
    它用于观察当前帧点云和当前 pose。
    它不等于 keyframe，也不等于最终地图。

2. keyframe / map 显示：
    只有 MakeLightningKeyframeIfNeeded 创建出的 keyframe 才会进入地图保存和 keyframe 可视化。
```

因此，如果 UI 中偶尔看到当前 scan 异常，并不一定表示最终地图被污染。是否污染最终地图，要看该帧是否被保存为 keyframe，以及 `mappingPoseReliable` 是否为 true。

---

## 19. 当前限制

当前 LIO-SAM offline 分支有以下限制：

```text
1. 只支持 offline mapping。
2. 不支持在线实时运行。
3. 不使用 IMUPreintegration。
4. 不使用 TransformFusion。
5. 不使用 odom deskew。
6. 不接 GPS/RTK factor。
7. 不使用原版 LIO-SAM 的 save_map service。
8. 不使用原版 LIO-SAM 的 ROS odom/path/tf publisher。
9. 要求点云带有逐点相对时间。
10. 当前 scan duration 使用最后一个点的时间，要求点云按时间排序。
11. 连续坏帧超过 10 帧会终止离线建图。
12. LIO-SAM 分支启用时，lightning-lm 自己的 LoopClosing 会关闭。
```

---

## 20. 与原版 LIO-SAM 的对比表

| 模块                       | 原版 LIO-SAM                            | 当前 lightning-lm LIO-SAM 分支              |
| ------------------------ | ------------------------------------- | --------------------------------------- |
| 数据入口                     | ROS topic callback                    | lightning-lm rosbag offline input       |
| 模块通信                     | ROS topic + `lio_sam::msg::CloudInfo` | 函数调用 + `LioSamCloudInfo`                |
| ImageProjection          | 订阅 LiDAR / IMU / odom                 | 接收同步后的 cloud + IMU                      |
| 点云去畸变                    | IMU rotation + odom deskew            | IMU rotation deskew                     |
| IMUPreintegration        | 使用                                    | 不使用                                     |
| TransformFusion          | 使用                                    | 不使用                                     |
| FeatureExtraction        | topic callback                        | 函数调用                                    |
| mapOptimization          | topic callback                        | 函数调用                                    |
| GTSAM/iSAM2              | 保留                                    | 保留                                      |
| GPSFactor                | 有                                     | 当前无                                     |
| Loop closure             | LIO-SAM 内部                            | LIO-SAM 内部                              |
| lightning-lm LoopClosing | 无                                     | LIO-SAM 分支下自动关闭                         |
| ROS odom/path/tf publish | 有                                     | 删除                                      |
| SaveMap                  | LIO-SAM service                       | lightning-lm SaveMap                    |
| 坏帧保护                     | 原版较弱                                  | 新增 motion gate / reject / failure limit |
| keyframe 输出              | LIO-SAM 内部                            | 转成 lightning `Keyframe`                 |
| 地图保存                     | LIO-SAM save_map service              | lightning-lm 保存流程                       |

---

## 21. 已知风险与后续改进

### 21.1 scan duration 建议改为 max point time

当前通过最后一个点时间计算 scan duration。如果点云不是按时间排序，可能导致同步错误。

后续建议：

```cpp
scan_duration = max(point.time)
```

而不是：

```cpp
scan_duration = cloud.points.back().time
```

---

### 21.2 增加 HasNewKeyframe 接口

当前 `SlamSystem::ProcessLidar()` 中每帧都会调用：

```cpp
PublishKeyframeToBackends(lio_sam_->GetKeyframe());
```

虽然内部有重复 keyframe 过滤，但语义不够清楚。

建议后续增加：

```cpp
bool LioSamMapping::HasNewKeyframe() const;
```

然后改成：

```cpp
if (lio_sam_->Run() && lio_sam_->HasNewKeyframe())
    PublishKeyframeToBackends(lio_sam_->GetKeyframe());
```

---

### 21.3 后续增加 RTK / GPS factor

当前 offline 版本没有接入 GPS/RTK。

建议未来新增：

```text
RTKManager
mapOptimization::addRTKFactor()
```

并在：

```cpp
saveKeyFramesAndFactor()
```

中加入：

```cpp
addRTKFactor();
```

RTK 应作为 keyframe-level position factor，而不是作为前端强先验。

---

### 21.4 header-only 工程组织

当前 offline LIO-SAM 模块以 header-only 方式被 `lio_sam_mapping.cc` 包含。需要确保 CMake 不要同时编译重复的 `.cpp` 实现。

建议长期整理为：

```text
image_projection_offline.h / .cc
feature_extraction_offline.h / .cc
map_optimization_offline.h / .cc
```

---

### 21.5 编译 warning 清理

部分点类型中存在：

```cpp
PCL_ADD_INTENSITY;
```

可能产生 pedantic warning。可以改成：

```cpp
PCL_ADD_INTENSITY
```

这不影响运行，只是清理编译输出。

---

## 22. 推荐调试日志

建议保留以下日志：

```text
[SYNC]
    LiDAR begin / end / IMU count

[POINT_TIME_CHECK]
    point time min / max / last / duration

[IMU_CONVERTER]
    raw RPY / converted RPY / acc

[MOTION_GATE]
    candidate vs last accepted tracking pose

[T2M]
    transformTobeMapped, reliable, state, source, keyposes

[KEYFRAME]
    created keyframe id / pose / raw cloud size

[MAPPING_FAIL]
    failure source / continuous failure count

[MAPPING_FATAL]
    bad frame count exceeds threshold
```

建议日志级别：

```text
正常 tracking:
    INFO 或 DEBUG

motion gate reject:
    WARN

连续坏帧终止:
    FATAL

NaN / Inf / 点云时间错误:
    ERROR 或 FATAL
```

---

## 23. 最终结论

当前新增的 LIO-SAM 分支不是原版 LIO-SAM 的完整 ROS 节点复刻，而是一个面向 lightning-lm 的 **LIO-SAM offline mapping core**。

它保留了原版 LIO-SAM 的核心建图算法：

```text
IMU rotation deskew
feature extraction
scan-to-map optimization
GTSAM/iSAM2
loop closure
correctPoses
```

同时替换了原版的在线 ROS 架构：

```text
ROS topic handoff
IMUPreintegration
TransformFusion
GPSFactor
ROS odom/path/tf publish
```

并新增了适合离线建图的同步和坏帧保护机制：

```text
SyncPackages
lastAcceptedTrackingTransform
motion gate
rejectMappingPose
failure count limit
```



该分支保留了 LIO-SAM 的核心离线建图流程，
将原版 ROS topic 模块通信改为函数调用，
并增加了适用于离线建图的 LiDAR/IMU 同步和坏帧拒绝机制。
```

