# Stationary Drift Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop publishing IMU-driven pose drift while the chassis is stationary, without freezing real motion.

**Architecture:** Detect stationary state at the existing IMU-to-DR boundary from body speed and gyro norm. Propagate the existing `NavState::is_parking_` flag so LidarLoc and PGO reuse their current pose-freeze path.

**Tech Stack:** ROS 2 Humble, C++17, Eigen, GoogleTest, YAML.

## Global Constraints

- Build only `lightning` and `lightning_nav2`; never build the whole workspace.
- Defaults are `0.05 m/s`, `0.05 rad/s`, and `5` consecutive IMU samples.
- Motion above either threshold exits parking immediately.
- Do not add nodes, topics, TF frames, or dependencies.

---

### Task 1: Stationary State Detection

**Files:**
- Modify: `src/core/localization/localization.h`
- Modify: `src/core/localization/localization.cpp`
- Modify: `test/test_navigation_interfaces.cc`

**Interfaces:**
- Consumes: `NavState`, body angular velocity, configured thresholds, consecutive sample counter.
- Produces: `UpdateParkingState(...)` updating `NavState::is_parking_`, velocity, and counter.

- [ ] **Step 1: Write failing tests**

Add tests proving five quiet samples enter parking and one moving sample immediately exits it.

- [ ] **Step 2: Verify RED**

Run `colcon test --packages-select lightning --ctest-args -R test_navigation_interfaces` after the existing build. Expected: compile failure because `UpdateParkingState` does not exist.

- [ ] **Step 3: Implement the minimum detector**

Declare and define `UpdateParkingState`; count only samples below both thresholds, clamp at the required count, zero velocity only when parked, and reset immediately on motion. Parse optional YAML values with the documented defaults and call the function once in `ProcessIMUMsg` before forwarding DR.

- [ ] **Step 4: Verify GREEN**

Run `colcon build --packages-select lightning` and `colcon test --packages-select lightning`; expect all tests to pass.

### Task 2: Robot Configuration and Regression

**Files:**
- Modify: `config/default_linghou.yaml`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/config/default_linghou_navigation.yaml`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/test/test_lightning_config.py`

**Interfaces:**
- Consumes: `lidar_loc.parking_speed_threshold`, `parking_gyro_threshold`, `parking_min_samples`.
- Produces: identical values in mapping and navigation configurations.

- [ ] **Step 1: Write failing config test**

Require both configurations to contain identical positive thresholds and sample count.

- [ ] **Step 2: Verify RED**

Run `python3 -m pytest -q src/lightning_nav2/test/test_lightning_config.py`; expected failure for missing keys.

- [ ] **Step 3: Add the three YAML values**

Set `parking_speed_threshold: 0.05`, `parking_gyro_threshold: 0.05`, and `parking_min_samples: 5` under `lidar_loc` in both files.

- [ ] **Step 4: Verify and build**

Run focused Python tests, then build/test only `lightning` and `lightning_nav2`.

- [ ] **Step 5: Safe live regression**

Start navigation with `enable_hardware_adapters:=false`, manually initialize, and observe stationary `/odom` and TF for at least 30 seconds. Confirm localization remains `GOOD`, twist becomes zero, and no upper-machine `/move/ManualMoveCmd` publisher exists.
