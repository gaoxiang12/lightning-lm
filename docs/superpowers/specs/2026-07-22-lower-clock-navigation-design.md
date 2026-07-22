# Lower-Clock Navigation Design

## Goal

在不修改下位机系统时间的情况下，让上位机 Lightning、Navigation2 和 RViz 使用下位机传感器时间域，消除 TF 与代价地图因跨机时钟相差约 112.9 天而产生的消息丢弃。

## Architecture

新增 ROS 2 节点 `lower_clock_bridge`：

- 订阅 `/driver/imu/Data`，使用 `sensor_data` QoS。
- 首次收到有限、非零且有效的 IMU `header.stamp` 时，记录该时间和本机 `time.monotonic_ns()`。
- 此后以 100 Hz 发布 `/clock`：

  ```text
  clock = anchor_sensor_stamp + (monotonic_now - anchor_monotonic)
  ```

- 节点自身保持系统时间，不能启用 `use_sim_time`，避免 `/clock` 启动依赖自身。
- IMU 暂时中断时 `/clock` 继续单调前进，使 Nav2 超时检测和安全门仍然有效。
- 运行期间不自动重新锚定；若传感器时间跳变或漂移超过 100 ms，只记录节流告警，不制造 ROS 时间回退。

## Integration

`navigation.launch.py` 启动 `lower_clock_bridge`，并为以下上位机节点设置 `use_sim_time: true`：

- Lightning 定位
- Nav2 map server、controller、planner、behavior server、BT navigator 和 lifecycle manager
- RViz2
- `radar_adapter`
- `cmd_vel_bridge`

原始 LiDAR、IMU 和相机消息不重发、不改时间戳。Lightning 内部仍以原始传感器时间同步、去畸变和估计；其 odometry 与 TF 时间戳自然处于 `/clock` 相同时间域。

增加 launch 参数 `use_lower_clock`，默认 `true`。设为 `false` 时不启动时钟桥，并让所有节点使用系统时间，便于在未来下位机时间恢复同步后关闭兼容层。

## Navigation Compatibility Fix

在 `nav2_params.yaml` 的 `bt_navigator.plugin_lib_names` 中增加：

```yaml
- nav2_remove_passed_goals_action_bt_node
```

该插件已安装，但当前未加载，导致 Humble 默认 `navigate_through_poses` 行为树无法识别 `RemovePassedGoals`，BT navigator 不能进入 active。

## Failure Handling

- 尚未收到有效 IMU 时间前不发布 `/clock`。
- 检测到传感器时间回退、非有限时间或与预测 `/clock` 偏差超过 100 ms 时告警。
- 不因数据中断冻结 `/clock`；已有传感器新鲜度检查负责停车。
- 本方案不掩盖约 3 秒的多传感器同步断流，也不绕过安全门。

## Validation

1. 单元检查：锚定后经过 2 秒，发布时钟增加 2 秒；IMU 中断时仍单调增加。
2. 配置检查：启用 `use_lower_clock` 时所有上位机导航节点均使用模拟时间，时钟桥除外。
3. 实时检查：`/clock` 与 IMU/LiDAR 时间戳差小于 100 ms。
4. 定位检查：Lightning 在 10 秒内进入 `GOOD`，TF 树为 `map -> odom -> base_link`。
5. Nav2 检查：所有 lifecycle 节点 active，不再出现 `RemovePassedGoals` 或旧时间戳消息过滤错误。
6. 安全检查：人为停止传感器输入后，速度桥持续输出零速。

## Deliberate Limits

- 不做运行中漂移校正。每次启动重新锚定；只有长时间运行确认漂移超过 100 ms 时才增加有界渐进校正。
- 不部署整套导航到下位机，不复制大体积 PointCloud2，不修改下位机系统配置。
