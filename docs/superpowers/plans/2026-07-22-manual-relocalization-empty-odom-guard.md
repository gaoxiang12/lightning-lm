# Manual Relocalization Empty Odom Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent manual `/initialpose` from crashing PGO when reset leaves the LiDAR odometry queue temporarily empty.

**Architecture:** Preserve the current reset and retry flow. Guard the diagnostic access to `lidar_odom_pose_queue_.back()` so interpolation failure returns normally until a later odometry frame repopulates the queue.

**Tech Stack:** ROS 2 Humble, C++17, GoogleTest, colcon.

## Global Constraints

- Modify only the Lightning ROS 2 package.
- Do not change localization thresholds, frame conventions, or PGO reset behavior.
- Build only `lightning`.

---

### Task 1: Guard empty LiDAR odometry diagnostics

**Files:**
- Modify: `test/test_navigation_interfaces.cc`
- Modify: `src/core/localization/pose_graph/pgo_impl.cc:176-181`

**Interfaces:**
- Consumes: an interpolation failure while `lidar_odom_pose_queue_` may be empty.
- Produces: `AssignLidarOdomPoseIfNeeded()` returns `false` without dereferencing an empty deque.

- [ ] **Step 1: Add a failing source regression test**

Add a test that reads `pgo_impl.cc` and requires an `empty()` branch before the only diagnostic `.back()` access.

- [ ] **Step 2: Verify RED**

Run `source /opt/ros/humble/setup.bash && colcon test --packages-select lightning --ctest-args -R test_navigation_interfaces`.

Expected: the new assertion fails because the source currently calls `.back()` unconditionally.

- [ ] **Step 3: Implement the minimum guard**

Log the frame timestamp alone when the queue is empty; otherwise include the latest odometry timestamp. Return `false` in both cases.

- [ ] **Step 4: Verify GREEN and build**

Run:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select lightning
colcon test --packages-select lightning --event-handlers console_direct+
colcon test-result --verbose
```

Expected: build succeeds and all Lightning tests pass.

- [ ] **Step 5: Reproduce the real workflow**

Start safe localization with hardware adapters disabled, publish a known `/initialpose`, and verify `run_loc_online` stays alive and localization reaches `GOOD(2)`.
