# LiDAR-Only Minimal Navigation Test

## Goal

用现有 Lightning 定位和 Nav2 验证最小导航闭环，仅使用融合 LiDAR
`/driver/lidar/point_cloud/Data` 感知障碍，不驱动底盘。

## Changes

- `local_costmap` 只保留 `voxel_layer` 和 `inflation_layer`。
- 上位机只由 `lidar_relay` 跨机订阅一次原始融合点云，再在本机发布
  `/lightning_nav2/fused_lidar` 给 Lightning 和 Nav2。
- `voxel_layer.observation_sources` 只包含本机 `fused_lidar`。
- 移除相机点云 source 和超声波 `range_layer`。
- 启动时使用 `enable_hardware_adapters:=false`，不启动雷达适配器和速度桥。
- 保留 lower-clock、Lightning、map server、Nav2 和 RViz。

实测两个上位机远程订阅会造成 2--4.4 s 数据间断，因此启用单读取者 relay。
导航 launch 同时将 CycloneDDS 固定到直连下位机的 `eth0`，避免 DDS 选择
`10.200.0.25` 所在网卡。voxel 高度范围从 -0.1 m 开始，包含略低于
`base_link` 的传感器原点。

## Acceptance

- `/lightning_nav2/fused_lidar` 在上位机持续约 10 Hz。
- Lightning 状态进入 `GOOD`，里程计无非物理发散。
- TF 为 `map -> odom -> base_link`。
- Nav2 五个生命周期节点均为 `active`。
- 局部代价地图能标记融合 LiDAR 障碍物。
- 日志不持续出现点云早于 TF 缓存或 observation buffer 超时。
- `/move/ManualMoveCmd` 不新增发布者，底盘不移动。
