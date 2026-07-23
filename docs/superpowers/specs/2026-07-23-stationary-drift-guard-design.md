# Lightning 静止漂移抑制设计

## 问题

在线定位把每个 IMU 预测状态作为 DR 输入，但 `Localization::ProcessIMUMsg` 中的停车判定被注释，导致 `NavState::is_parking_` 始终为 `false`。已有的停车位姿冻结路径因此不可达，底盘静止时仍积分 IMU 噪声并发布漂移位姿。

## 设计

- 在 IMU→DR 的统一入口进行静止判定，避免在 TF、Nav2 或发布层掩盖漂移。
- 静止条件同时要求机体系线速度和 IMU 角速度低于阈值，并连续满足指定样本数。
- 达到条件后设置 `is_parking_=true` 并把速度置零；任一条件超限立即清除计数并退出停车状态。
- 阈值通过 `lidar_loc` YAML 配置，默认：线速度 `0.05 m/s`、角速度 `0.05 rad/s`、连续 `5` 个 IMU 样本。
- 复用现有 `LidarLoc`/PGO 停车冻结逻辑，不增加节点、话题或 TF。

## 修改范围

- `src/core/localization/localization.h/.cpp`：保存阈值、连续计数并生成停车状态。
- `config/default_linghou.yaml` 与 `../lightning_nav2/config/default_linghou_navigation.yaml`：加入相同阈值。
- 聚焦测试验证：连续静止后进入停车、检测到运动立即退出、配置参数存在且一致。

## 验收

- 仅构建和测试 ROS 2 `lightning`、`lightning_nav2`。
- 实时静止测试中定位保持 `GOOD`，`/odom` 速度归零，`map→base_link` 不再持续漂移。
- 低速真实运动超过任一阈值时不冻结位姿。
