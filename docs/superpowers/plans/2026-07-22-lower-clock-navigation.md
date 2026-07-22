# Lower-Clock Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让上位机 Lightning、Navigation2 和 RViz 在不修改下位机系统时间的情况下使用下位机传感器时间域，并修复 BT navigator 的插件加载失败。

**Architecture:** 一个独立 ROS 2 Python 节点从 IMU 首帧建立“下位机时间—本机 monotonic 时间”锚点，以 100 Hz 持续发布 `/clock`。导航 launch 通过一个开关让所有上位机消费者使用模拟时间；原始点云、IMU、Lightning 估计和 TF 时间戳保持不变。

**Tech Stack:** ROS 2 Humble、rclpy、`rosgraph_msgs/msg/Clock`、`sensor_msgs/msg/Imu`、ament_cmake_python、pytest、Navigation2。

## Global Constraints

- 只构建 `lightning_nav2`；若 Lightning 源码未变化，不重建 `lightning`，绝不构建整个 workspace。
- 时钟桥自身必须保持 `use_sim_time=false`，避免 `/clock` 依赖自身启动。
- 不复制 PointCloud2，不修改原始传感器消息，不修改下位机系统时间。
- `/clock` 在 IMU 暂停时继续前进；运行中不得让 ROS 时间回退。
- 初版不做漂移校正；每次启动重新锚定，偏差超过 100 ms 只告警。
- `/home/tjzn/Workspace/src/lightning_nav2` 当前不是 Git 仓库；每个任务以测试结果作为检查点，不创建伪提交。

---

### Task 1: Lower-clock bridge

**Files:**
- Create: `/home/tjzn/Workspace/src/lightning_nav2/lightning_nav2/lower_clock_bridge.py`
- Create: `/home/tjzn/Workspace/src/lightning_nav2/scripts/lower_clock_bridge`
- Create: `/home/tjzn/Workspace/src/lightning_nav2/test/test_lower_clock_bridge.py`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/CMakeLists.txt`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/package.xml`

**Interfaces:**
- Consumes: `sensor_msgs/msg/Imu /driver/imu/Data` with `qos_profile_sensor_data`.
- Produces: `rosgraph_msgs/msg/Clock /clock` at 100 Hz after the first valid IMU stamp.
- Produces: `ClockAnchor.observe(sensor_ns: int, steady_ns: int) -> int | None` and `ClockAnchor.now_ns(steady_ns: int) -> int | None` for deterministic tests.

- [ ] **Step 1: Write the failing clock-model tests**

Create `test/test_lower_clock_bridge.py`:

```python
from lightning_nav2.lower_clock_bridge import ClockAnchor


def test_clock_anchor_advances_during_sensor_gap():
    anchor = ClockAnchor()
    assert anchor.observe(100_000_000_000, 5_000_000_000) == 0
    assert anchor.now_ns(7_000_000_000) == 102_000_000_000


def test_clock_anchor_rejects_invalid_stamp_and_reports_skew():
    anchor = ClockAnchor()
    assert anchor.observe(0, 1_000_000_000) is None
    assert anchor.observe(100_000_000_000, 5_000_000_000) == 0
    assert anchor.observe(102_050_000_000, 7_000_000_000) == 50_000_000
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
python3 -m pytest -q src/lightning_nav2/test/test_lower_clock_bridge.py
```

Expected: collection fails with `ModuleNotFoundError` for `lightning_nav2.lower_clock_bridge`.

- [ ] **Step 3: Implement the minimal clock bridge**

Create `lightning_nav2/lower_clock_bridge.py`:

```python
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu


NSEC_PER_SEC = 1_000_000_000


class ClockAnchor:
    def __init__(self):
        self.sensor_ns = None
        self.steady_ns = None

    def observe(self, sensor_ns, steady_ns):
        if sensor_ns <= 0:
            return None
        if self.sensor_ns is None:
            self.sensor_ns = sensor_ns
            self.steady_ns = steady_ns
            return 0
        return sensor_ns - self.now_ns(steady_ns)

    def now_ns(self, steady_ns):
        if self.sensor_ns is None:
            return None
        return self.sensor_ns + steady_ns - self.steady_ns


class LowerClockBridge(Node):
    def __init__(self):
        super().__init__("lower_clock_bridge")
        self.declare_parameter("source_topic", "/driver/imu/Data")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("max_skew", 0.1)
        self.anchor = ClockAnchor()
        self.max_skew_ns = int(self.get_parameter("max_skew").value * NSEC_PER_SEC)
        self.last_warning_ns = 0
        clock_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.clock_publisher = self.create_publisher(Clock, "/clock", clock_qos)
        self.create_subscription(
            Imu,
            self.get_parameter("source_topic").value,
            self._on_imu,
            qos_profile_sensor_data,
        )
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        self.create_timer(1.0 / rate, self._publish)

    def _on_imu(self, msg):
        sensor_ns = msg.header.stamp.sec * NSEC_PER_SEC + msg.header.stamp.nanosec
        steady_ns = time.monotonic_ns()
        was_uninitialized = self.anchor.sensor_ns is None
        skew_ns = self.anchor.observe(sensor_ns, steady_ns)
        if skew_ns is None:
            return
        if was_uninitialized:
            self.get_logger().info(f"clock anchored at {sensor_ns * 1e-9:.9f}")
        elif abs(skew_ns) > self.max_skew_ns and steady_ns - self.last_warning_ns >= 5 * NSEC_PER_SEC:
            self.last_warning_ns = steady_ns
            self.get_logger().warning(f"sensor clock skew is {skew_ns * 1e-9:.3f}s")

    def _publish(self):
        clock_ns = self.anchor.now_ns(time.monotonic_ns())
        if clock_ns is None:
            return
        msg = Clock()
        msg.clock.sec, msg.clock.nanosec = divmod(clock_ns, NSEC_PER_SEC)
        self.clock_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LowerClockBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()
```

Create executable `scripts/lower_clock_bridge`:

```python
#!/usr/bin/env python3
from lightning_nav2.lower_clock_bridge import main

if __name__ == "__main__":
    main()
```

Run `chmod +x src/lightning_nav2/scripts/lower_clock_bridge`.

- [ ] **Step 4: Register runtime dependencies, executable, and test**

Add `lower_clock_bridge` to the existing `install(PROGRAMS ...)` list and add:

```cmake
ament_add_pytest_test(test_lower_clock_bridge test/test_lower_clock_bridge.py)
```

inside the existing `BUILD_TESTING` block in `CMakeLists.txt`.

Add to `package.xml`:

```xml
<exec_depend>rosgraph_msgs</exec_depend>
```

- [ ] **Step 5: Run the focused tests**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
python3 -m pytest -q src/lightning_nav2/test/test_lower_clock_bridge.py
```

Expected: `2 passed`.

### Task 2: Navigation launch and BT compatibility

**Files:**
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/launch/navigation.launch.py`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/config/nav2_params.yaml`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/test/test_lightning_config.py`

**Interfaces:**
- Consumes: `lower_clock_bridge` executable from Task 1.
- Produces: launch argument `use_lower_clock` with default string value `true`.
- Produces: all upper navigation nodes with `use_sim_time` equal to `use_lower_clock`.

- [ ] **Step 1: Add failing configuration assertions**

Append to `test/test_lightning_config.py`:

```python
def test_navigation_launch_uses_lower_clock_for_all_consumers():
    launch = (Path(__file__).parents[1] / "launch" / "navigation.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'DeclareLaunchArgument("use_lower_clock", default_value="true")' in launch
    assert 'executable="lower_clock_bridge"' in launch
    assert launch.count('{"use_sim_time": use_lower_clock}') == 10


def test_bt_navigator_loads_remove_passed_goals_plugin():
    nav2 = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
    )
    plugins = nav2["bt_navigator"]["ros__parameters"]["plugin_lib_names"]
    assert "nav2_remove_passed_goals_action_bt_node" in plugins
```

The expected count covers Lightning, five Nav2 servers, lifecycle manager, radar adapter, velocity bridge, and RViz. The clock bridge must not receive this parameter.

- [ ] **Step 2: Run the tests and verify both assertions fail**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
python3 -m pytest -q src/lightning_nav2/test/test_lightning_config.py
```

Expected: two failures for the absent launch argument/executable and absent BT plugin.

- [ ] **Step 3: Add the launch clock switch and clock node**

In `generate_launch_description()` add:

```python
use_lower_clock = LaunchConfiguration("use_lower_clock")
```

Add to the returned `LaunchDescription` before Lightning:

```python
DeclareLaunchArgument("use_lower_clock", default_value="true"),
Node(
    package="lightning_nav2",
    executable="lower_clock_bridge",
    name="lower_clock_bridge",
    condition=IfCondition(use_lower_clock),
    output="screen",
),
```

Add `parameters=[{"use_sim_time": use_lower_clock}]` to Lightning and RViz. Append `{"use_sim_time": use_lower_clock}` to every existing parameters list for map server, controller server, planner server, behavior server, BT navigator, lifecycle manager, radar adapter, and velocity bridge. The resulting velocity bridge node must include:

```python
Node(
    package="lightning_nav2",
    executable="cmd_vel_bridge",
    name="cmd_vel_bridge",
    parameters=[{"use_sim_time": use_lower_clock}],
    condition=IfCondition(enable_hardware_adapters),
    output="screen",
),
```

- [ ] **Step 4: Load the installed Humble BT plugin**

Add to `bt_navigator.ros__parameters.plugin_lib_names` in `config/nav2_params.yaml`:

```yaml
- nav2_remove_passed_goals_action_bt_node
```

- [ ] **Step 5: Run all package-level tests**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
source /home/tjzn/Workspace/hardware_interfaces_ws/install/setup.bash
colcon build --packages-select lightning_nav2 --symlink-install
source /home/tjzn/Workspace/install/setup.bash
colcon test --packages-select lightning_nav2
colcon test-result --verbose
```

Expected: build succeeds and all `lightning_nav2` tests report zero failures.

### Task 3: Real-time acceptance

**Files:**
- No source changes.
- Runtime log: `/tmp/lightning_lower_clock_navigation.log`

**Interfaces:**
- Consumes: `/clock`, `/driver/imu/Data`, Lightning localization outputs, Nav2 lifecycle services.
- Verifies: clock skew, localization state, TF tree, lifecycle state, costmap input, clean shutdown.

- [ ] **Step 1: Start the complete stack without RViz or a navigation goal**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
source /home/tjzn/Workspace/hardware_interfaces_ws/install/setup.bash
source /home/tjzn/Workspace/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
timeout --signal=INT 45 ros2 launch lightning_nav2 navigation.launch.py \
  map:=/home/tjzn/Workspace/data/new_map/map.yaml \
  sensor_extrinsics:=/home/tjzn/Workspace/sensors_extrinsic.yaml \
  enable_hardware_adapters:=true \
  use_lower_clock:=true \
  use_rviz:=false > /tmp/lightning_lower_clock_navigation.log 2>&1
```

- [ ] **Step 2: Verify `/clock` uses the sensor time domain**

While the launch is running, execute:

```bash
ros2 topic echo --once --field clock /clock
ros2 topic echo --once --field header.stamp /driver/imu/Data
```

Expected: absolute difference is below `0.1 s`, and successive `/clock` samples are strictly increasing even across a short IMU input pause.

- [ ] **Step 3: Verify localization and TF**

Run:

```bash
timeout 10 ros2 topic echo --once /lightning/localization_status
timeout 10 ros2 run tf2_ros tf2_echo map base_link
```

Expected: status `data: 2` (`GOOD`) and a finite `map -> base_link` transform through `map -> odom -> base_link`.

- [ ] **Step 4: Verify all lifecycle nodes are active**

Run:

```bash
for node in map_server controller_server planner_server behavior_server bt_navigator; do
  ros2 lifecycle get /$node
done
```

Expected: each command prints `active [3]`.

- [ ] **Step 5: Check logs for eliminated blockers**

Run:

```bash
rg -n "RemovePassedGoals|earlier than all the data|Failed to bring up|Traceback|RCLError" \
  /tmp/lightning_lower_clock_navigation.log
```

Expected: no matches. Sensor freshness warnings caused by the existing approximately 3-second transport interruption may remain; those must keep the speed bridge closed and are not a clock-bridge failure.

- [ ] **Step 6: Confirm clean shutdown and report remaining hardware limits**

Expected after timeout: Python adapters finish cleanly and the latest `/move/ManualMoveCmd` is zero. Report separately that all eight ultrasonic sensors currently advertise `is_connected=false`; do not claim obstacle safety acceptance until hardware is connected.
