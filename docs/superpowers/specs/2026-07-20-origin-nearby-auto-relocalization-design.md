# Lightning-LM 原点附近自动重定位设计

## 目标

实时定位启动后，即使没有收到 RViz `/initialpose`，车辆位于地图 `start` 功能点附近时也能自动完成初次重定位。默认支持候选点平面半径 5 m 内的位置偏差和完整 360° 航向不确定性。

本功能不是全地图地点识别。超过局部 NDT 收敛范围时保持未初始化，不能输出未经验证的外参位姿。

## 当前问题

- 自动初始化虽然会遍历地图功能点，但 `InitWithFP()` 未使用已有的 `YawSearch()`，因此配置的全航向搜索没有生效。
- `Localize()` 当前无论初始化置信度是否达标都会返回成功，错误匹配可能被标记为 `GOOD`。
- 切换功能点后，地图块和 NDT target 通过后台线程更新；本次初始化可能仍在旧 target 上执行。

## 方案

### 初始化候选

- 沿用地图 `index.txt` 中现有功能点，不新增地点数据库。
- 无外部初值时优先尝试 `start`；已有 `recover` 功能点时仍可按现有逻辑继续尝试。
- 每个候选开始匹配前，调用 `LoadOnPose(candidate)`，并同步重建 NDT target，保证匹配地图与候选位置一致。

### 两级 NDT 搜索

- 候选平移使用功能点坐标，不增加 XY 网格枚举。
- 使用现有粗 NDT 和 `grid_search_angle_range/grid_search_angle_step` 遍历完整 360° 航向。
- 选择粗匹配中置信度最高且收敛的候选，再使用正常分辨率 NDT 精配准。
- NDT 自身连续优化平移；最终平面位置必须距当前功能点不超过 `max_init_distance`。

这种方式利用现有 5 m 粗 NDT 的收敛域，避免 XY × yaw 上千次组合搜索。默认上限为 5 m，实际收敛能力由场景结构和点云重叠决定。

### 初始化门限

沿用并严格执行已有参数：

```yaml
lidar_loc:
  min_init_confidence: 1.8
  max_init_distance: 5.0
  grid_search_angle_range: 180.0
  grid_search_angle_step: 60
```

只有同时满足以下条件才允许进入 `GOOD`：

- 粗匹配存在有效候选。
- 精匹配收敛。
- 位姿和平移均为有限值。
- 精匹配置信度不低于 `min_init_confidence`。
- 优化后 XY 位置距候选功能点不超过 `max_init_distance`，单位为米。

任一条件失败时保持 `INITIALIZING`，沿用现有重试节流，不更新 PGO，不发布旧的 `GOOD`。

### 外部初值

`/initialpose` 仍作为独立的局部重定位入口。它同样需要先同步加载对应地图块并执行收敛、有限值和置信度检查；距离限制以外部初值为中心，不限制它必须靠近地图原点。

## 修改范围

- `src/core/localization/lidar_loc/lidar_loc.h`
  - 增加 `max_init_distance_` 配置。
- `src/core/localization/lidar_loc/lidar_loc.cc`
  - 读取参数；恢复两级 yaw 搜索；同步更新候选地图 target；修正初始化成功判定。
- `config/default_linghou.yaml`
  - 增加默认 5 m 初始化距离。
- `../lightning_nav2/config/default_linghou_navigation.yaml`
  - 增加相同导航运行参数。
- `test/test_navigation_interfaces.cc`
  - 增加初始化门限的最小回归检查。

不增加 Scan Context、XY 网格搜索、地图格式或新 ROS 接口。

## 验证

### 单元回归

- 置信度低于门限时拒绝初始化。
- 最终 XY 偏移超过 5 m 时拒绝初始化。
- 非有限置信度或位姿拒绝初始化。
- 合法候选通过门限。

### 数据回归

- 使用当前 `new_map` 和实时融合点云，在原点附近不发布 `/initialpose`。
- 分别以多个朝向启动，验证自动从 `INITIALIZING` 进入 `GOOD`。
- 在距原点超过 5 m 或无地图重叠的位置启动，验证不会输出 `GOOD`。
- 记录首次成功耗时、最终置信度以及相对 `start` 的 XY 距离。

### 构建

只构建 ROS 2 包：

```bash
colcon build --packages-select lightning lightning_nav2
colcon test --packages-select lightning
```

## 风险与边界

- 5 m 是默认安全上限，不保证所有场景都能从 5 m 误差收敛；重复走廊、低结构区域可能需要更准确的位置先验。
- 60 个 yaw 候选会增加首次定位耗时，但只发生在未初始化阶段，并沿用现有失败重试节流。
- 若实测原点附近有效率仍不足，再考虑 XY 粗网格；本次不提前引入。
