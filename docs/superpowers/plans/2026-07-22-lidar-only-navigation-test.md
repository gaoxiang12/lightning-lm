# LiDAR-Only Navigation Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Nav2 局部代价地图精简为仅使用融合 LiDAR，并在不输出底盘速度时完成实时导航回归。

**Architecture:** Lightning 继续用融合 LiDAR 和 IMU 定位；Nav2 voxel layer 只订阅融合 LiDAR。关闭硬件适配器，因此不启动超声波适配器和速度桥。

**Tech Stack:** ROS 2 Humble、Navigation2、pytest、YAML。

## Global Constraints

- 只构建 `lightning_nav2`，不构建整个 workspace。
- 不增加点云 relay、新依赖或新节点。
- 不启动速度桥，不向 `/move/ManualMoveCmd` 新增发布者。
- `src/lightning_nav2` 不是 Git 仓库，以测试输出为检查点。

---

### Task 1: LiDAR-only local costmap

**Files:**
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/test/test_lightning_config.py`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/config/nav2_params.yaml`

**Interfaces:**
- Consumes: `/driver/lidar/point_cloud/Data` (`sensor_msgs/msg/PointCloud2`).
- Produces: local costmap with `voxel_layer` and `inflation_layer` only.

- [ ] **Step 1: Add the failing configuration test**

```python
def test_minimal_navigation_uses_only_fused_lidar_for_obstacles():
    nav2 = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
    )
    params = nav2["local_costmap"]["local_costmap"]["ros__parameters"]
    assert params["plugins"] == ["voxel_layer", "inflation_layer"]
    voxel = params["voxel_layer"]
    assert voxel["observation_sources"] == "fused_lidar"
    assert "camera" not in voxel
    assert "range_layer" not in params
```

- [ ] **Step 2: Verify RED**

Run `source /opt/ros/humble/setup.bash && python3 -m pytest -q src/lightning_nav2/test/test_lightning_config.py`.

Expected: the new test fails because `camera` and `range_layer` are present.

- [ ] **Step 3: Apply the minimum YAML change**

Set `plugins: [voxel_layer, inflation_layer]` and `observation_sources: fused_lidar`. Delete the `camera` source and complete `range_layer` block; keep the fused LiDAR limits unchanged.

- [ ] **Step 4: Verify GREEN and build only the affected package**

```bash
source /opt/ros/humble/setup.bash
python3 -m pytest -q src/lightning_nav2/test/test_lightning_config.py
colcon build --packages-select lightning_nav2 --symlink-install
colcon test --packages-select lightning_nav2 --event-handlers console_direct+
colcon test-result --verbose
```

Expected: all `lightning_nav2` tests pass.

### Task 2: Safe live regression

**Files:** None.

**Interfaces:**
- Consumes: live LiDAR/IMU and `/clock`.
- Produces: diagnostic evidence only; no chassis command output.

- [ ] **Step 1: Start the minimal stack**

```bash
source /opt/ros/humble/setup.bash
source /home/tjzn/Workspace/install/setup.bash
ros2 launch lightning_nav2 navigation.launch.py \
  map:=/home/tjzn/Workspace/data/new_map/map.yaml \
  sensor_extrinsics:=/home/tjzn/Workspace/sensors_extrinsic.yaml \
  enable_hardware_adapters:=false \
  use_rviz:=false \
  use_lower_clock:=true
```

- [ ] **Step 2: Verify live signals**

Check LiDAR near 10 Hz, localization status `2`, finite `map -> base_link`, all five lifecycle nodes `active`, `/local_costmap/costmap` updating, and no upper `cmd_vel_bridge` publisher on `/move/ManualMoveCmd`.

- [ ] **Step 3: Stop and inspect output**

Expected: no sustained observation-buffer timeout, no sustained `earlier than all the data in the transform cache`, no `RemovePassedGoals` error, and no Python traceback.
