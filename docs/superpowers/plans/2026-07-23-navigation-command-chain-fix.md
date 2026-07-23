# Navigation Command Chain Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让仅 LiDAR 导航的速度安全门能够放行，并在不连接底盘速度桥时验证 Nav2 的 TF、规划和速度输出链路。

**Architecture:** 删除速度桥对相机和超声波的硬依赖，保留定位、融合 LiDAR、Nav2 生命周期、控制模式和命令超时门控。安全在线回归先隔离底盘输出；只有复现 `map -> odom` 停更时才根据该次日志确定 Lightning 修改点。

**Tech Stack:** ROS 2 Humble、Python 3、rclpy、Navigation2、pytest。

## Global Constraints

- 第一阶段仅使用 `/driver/lidar/point_cloud/Data` 进行避障和传感器安全门控。
- 不放宽 Nav2 transform tolerance 来掩盖秒级时间错误。
- 仅构建 `lightning_nav2`；若后续证据要求修改 Lightning，再单独构建 `lightning`。
- 实时回归使用 `enable_hardware_adapters:=false`，不得发布到底盘。
- 保留用户现有未提交修改。

---

### Task 1: LiDAR-only safety gate with block reason

**Files:**
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/lightning_nav2/safety.py`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/lightning_nav2/cmd_vel_bridge.py`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/test/test_safety.py`

**Interfaces:**
- Consumes: Lightning status, localization receive time, fused LiDAR receive time, Nav2 active state.
- Produces: `SafetyGate.update(...) -> bool` and `SafetyGate.reason: str`.

- [ ] **Step 1: Write failing LiDAR-only and reason tests**

Replace the test helper with only the required inputs and add assertions:

```python
def healthy_times(now):
    return {name: now for name in ("localization", "lidar")}


def test_gate_ignores_unused_camera_and_radar():
    gate = SafetyGate(recovery_sec=0.0)
    assert gate.update(10.0, 2, healthy_times(10.0), True)


def test_gate_reports_stale_lidar():
    gate = SafetyGate(recovery_sec=0.0)
    times = healthy_times(10.0)
    times["lidar"] = 9.4
    assert not gate.update(10.0, 2, times, True)
    assert gate.reason == "lidar stale"
```

- [ ] **Step 2: Run RED test**

Run:

```bash
source /opt/ros/humble/setup.bash
python3 -m pytest -q src/lightning_nav2/test/test_safety.py
```

Expected: failure because the current gate requires `camera` and `radar`, and has no `reason`.

- [ ] **Step 3: Implement the minimum gate**

Use only these limits and set one stable reason:

```python
limits = {"localization": 0.5, "lidar": 0.5}
```

Evaluate status, Nav2 state, each timestamp, and recovery in that order. Set `reason` to one of:
`localization not GOOD`, `Nav2 inactive`, `<name> missing`, `<name> stale`, `recovering`, or an empty string when open.

In `cmd_vel_bridge.py`:

- import only `move.srv.SetControlMode` dynamically;
- delete camera and radar subscriptions;
- log only when the combined blocking reason changes;
- keep LiDAR sensor-data QoS, localization subscription, mode retry, 0.2-second command timeout, and zero output unchanged.

- [ ] **Step 4: Run GREEN tests and package verification**

Run:

```bash
source /opt/ros/humble/setup.bash
python3 -m pytest -q src/lightning_nav2/test/test_safety.py
colcon build --packages-select lightning_nav2 --symlink-install
colcon test --packages-select lightning_nav2 --event-handlers console_direct+
colcon test-result --verbose
```

Expected: all `lightning_nav2` tests pass with zero failures.

### Task 2: Safe live navigation regression

**Files:** None unless the runtime evidence identifies a specific Lightning defect.

**Interfaces:**
- Consumes: live IMU and fused LiDAR, `/initialpose`, `/navigate_to_pose`.
- Produces: diagnostic `/cmd_vel_nav`; no `/move/ManualMoveCmd` publisher from this stack.

- [ ] **Step 1: Launch without hardware adapters**

```bash
source /opt/ros/humble/setup.bash
source /home/tjzn/Workspace/install/setup.bash
ros2 launch lightning_nav2 navigation.launch.py \
  map:=/home/tjzn/Workspace/data/new_map/map.yaml \
  sensor_extrinsics:=/home/tjzn/Workspace/sensors_extrinsic.yaml \
  enable_hardware_adapters:=false use_rviz:=false use_lower_clock:=true
```

- [ ] **Step 2: Initialize and send one bounded test goal**

Publish the previously verified initial pose `(-0.475, -0.03, yaw=-0.066 rad)`, wait for status `GOOD(2)`, then send a goal no more than 0.5 m from the reported pose. Do not start `cmd_vel_bridge`.

- [ ] **Step 3: Verify each boundary**

Check:

```bash
ros2 topic info /move/ManualMoveCmd --verbose
ros2 topic hz /lightning_nav2/fused_lidar
ros2 topic echo --once /lightning/localization_status
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_link
ros2 topic echo --once /cmd_vel_nav
```

Expected: no new upper-computer `/move/ManualMoveCmd` publisher, LiDAR near 10 Hz, status 2, finite current TF, and a nonzero Nav2 velocity for a reachable goal.

- [ ] **Step 4: Inspect TF failure evidence if regression fails**

If Nav2 reports stale TF, capture the current `/clock`, `/odom` stamp, `map -> odom` stamp, localization status, and the latest Lightning callback log from the same run. Stop without changing transform tolerance or adding a TF keeper; the exact producer that stopped determines the next minimal patch.

