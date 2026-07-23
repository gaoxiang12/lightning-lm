# Navigation Command Chain Fix

## Goal

修复 RViz 下发导航点后底盘不移动的问题。第一阶段仅使用融合 LiDAR
`/driver/lidar/point_cloud/Data` 作为导航避障和速度安全门输入。

## Confirmed Failures

- `cmd_vel_bridge` 当前仍强制等待相机和超声波，和仅 LiDAR 导航配置冲突。
- 历史实车运行中，Nav2 因 `map -> odom` TF 落后当前数据约 22.6 秒而中止控制。
- 同期 local costmap 报告融合 LiDAR observation buffer 长时间未更新；这可能是 TF
  失败导致的消息过滤，不能在确认前归因于 relay。

## Design

### LiDAR-only safety gate

- 安全门只要求：Lightning 状态为 `GOOD(2)`、定位更新时间不超过 0.5 秒、融合
  LiDAR 更新时间不超过 0.5 秒、五个 Nav2 生命周期节点均为 active、控制模式设置成功。
- 删除速度桥对相机点云和超声波话题的订阅与接口导入。
- 保留 2 秒稳定恢复、0.2 秒速度命令超时和持续零速输出。
- 安全门状态变化时输出一条原因日志，避免无日志零速。

### TF and relay investigation

- 使用 `enable_hardware_adapters:=false` 启动，不产生新的底盘速度发布者。
- 手动重定位进入 `GOOD` 后发送测试导航目标，检查 `map -> odom -> base_link`、
  `/cmd_vel_nav` 和融合 LiDAR observation buffer。
- 若 `map -> odom` 停更，修复 Lightning 中停止传播的源头；不增加独立 TF 保活节点。
- 只有确认 `/lightning_nav2/fused_lidar` 自身断流时才修改 relay；TF 消息过滤导致的
  observation buffer 告警不通过重写 relay 处理。

## Files

- `src/lightning_nav2/lightning_nav2/safety.py`
- `src/lightning_nav2/lightning_nav2/cmd_vel_bridge.py`
- `src/lightning_nav2/test/test_safety.py`
- Lightning TF 源文件仅在安全回归复现停更后确定。

## Validation

- 测试先证明仅有定位、LiDAR 和 active Nav2 时旧安全门仍关闭，再做最小修改。
- 仅构建并测试 `lightning_nav2`；若修改 Lightning，再单独构建并测试 `lightning`。
- 安全在线测试不得启动速度桥；确认 Nav2 能持续发布非零 `/cmd_vel_nav` 后，再请求
  实车速度转发授权。

## Assumptions

- `mode=1` 是 `/move/ManualMoveCmd` 对应的手动速度控制模式。
- 本阶段不使用相机和超声波作为安全门条件。
- 不通过放宽 Nav2 transform tolerance 掩盖秒级时间错误。
