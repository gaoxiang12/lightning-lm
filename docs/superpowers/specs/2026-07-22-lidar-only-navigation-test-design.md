# LiDAR-Only Minimal Navigation Test

## Goal

用现有 Lightning 定位和 Nav2 验证最小导航闭环，仅使用融合 LiDAR
`/driver/lidar/point_cloud/Data` 感知障碍，不驱动底盘。

## Changes

- `local_costmap` 只保留 `voxel_layer` 和 `inflation_layer`。
- `voxel_layer.observation_sources` 只包含 `fused_lidar`。
- 移除相机点云 source 和超声波 `range_layer`。
- 启动时使用 `enable_hardware_adapters:=false`，不启动雷达适配器和速度桥。
- 保留 lower-clock、Lightning、map server、Nav2 和 RViz。

不增加点云 relay。只有实测两个上位机 LiDAR 订阅导致跨机数据持续中断时，
才单独设计 relay。

## Acceptance

- `/driver/lidar/point_cloud/Data` 在上位机持续约 10 Hz。
- Lightning 状态进入 `GOOD`，里程计无非物理发散。
- TF 为 `map -> odom -> base_link`。
- Nav2 五个生命周期节点均为 `active`。
- 局部代价地图能标记融合 LiDAR 障碍物。
- 日志不持续出现点云早于 TF 缓存或 observation buffer 超时。
- `/move/ManualMoveCmd` 不新增发布者，底盘不移动。
