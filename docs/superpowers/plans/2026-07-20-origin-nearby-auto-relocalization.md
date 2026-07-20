# Lightning-LM Origin-Nearby Auto Relocalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Lightning-LM safely initialize near a map functional point without `/initialpose`, using configurable confidence and distance gates.

**Architecture:** Reuse the existing functional-point map index and two NDT instances: search all configured yaw candidates with the 5 m rough NDT, then refine the best converged candidate with the normal-resolution NDT. Centralize acceptance in a pure validation function so automatic and external initialization share identical convergence, finite-value, confidence, and XY-distance rules.

**Tech Stack:** ROS 2 Humble, C++17, Eigen/Sophus `SE3`, PCL-OMP NDT, yaml-cpp wrapper, GoogleTest, colcon/ament.

## Global Constraints

- Build only ROS 2 packages with `colcon build --packages-select lightning lightning_nav2`; never build the entire mixed ROS 1/ROS 2 workspace.
- Do not add Scan Context, an XY search lattice, a new map format, or a new ROS interface.
- Keep `lidar_loc.min_init_confidence` configurable; its configured value is an inclusive lower bound.
- Add `lidar_loc.max_init_distance` in metres with default value `5.0` in both Linghou runtime configurations.
- Search yaw with the existing `grid_search_angle_range: 180.0` and `grid_search_angle_step: 60` parameters.
- Only a converged, finite fine-NDT result with confidence at least the configured threshold and XY displacement no greater than the configured maximum may enter `GOOD`.
- A rejected result remains `INITIALIZING`, does not update PGO, and does not publish a stale `GOOD` result.
- Preserve unrelated user changes under `doc/` and `docs/`; stage only files named by each task.

---

### Task 1: Add a testable initialization acceptance gate

**Files:**
- Modify: `test/test_navigation_interfaces.cc`
- Modify: `src/core/localization/lidar_loc/lidar_loc.h`
- Modify: `src/core/localization/lidar_loc/lidar_loc.cc`

**Interfaces:**
- Consumes: `lightning::SE3`, NDT convergence status, candidate pose, optimized pose, transformation probability, and configured thresholds.
- Produces: `bool IsInitializationResultValid(bool matcher_converged, const SE3& candidate_pose, const SE3& result_pose, double confidence, double min_confidence, double max_distance)` in namespace `lightning::loc`.

- [ ] **Step 1: Write the failing gate regression test**

Add the localization header and the following test to `test/test_navigation_interfaces.cc`:

```cpp
#include <limits>

#include "core/localization/lidar_loc/lidar_loc.h"

TEST(LidarLocInitialization, EnforcesConvergenceConfidenceFiniteAndDistanceGates) {
    const lightning::SE3 candidate(lightning::SO3(), lightning::Vec3d(0.0, 0.0, 0.0));
    const lightning::SE3 inside(lightning::SO3(), lightning::Vec3d(3.0, 4.0, 0.0));
    const lightning::SE3 outside(lightning::SO3(), lightning::Vec3d(3.01, 4.0, 0.0));
    const lightning::SE3 nonfinite(
        lightning::SO3(),
        lightning::Vec3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));

    EXPECT_TRUE(lightning::loc::IsInitializationResultValid(
        true, candidate, inside, 1.8, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        false, candidate, inside, 2.0, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, inside, 1.79, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, outside, 2.0, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, nonfinite, 2.0, 1.8, 5.0));
    EXPECT_FALSE(lightning::loc::IsInitializationResultValid(
        true, candidate, inside, std::numeric_limits<double>::infinity(), 1.8, 5.0));
}
```

- [ ] **Step 2: Build the focused test target and verify the new test fails**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select lightning --cmake-target test_navigation_interfaces
```

Expected: compilation fails because `lightning::loc::IsInitializationResultValid` is not declared.

- [ ] **Step 3: Declare the gate and add the distance option**

Immediately inside `namespace lightning::loc` and before `class LidarLoc` in `src/core/localization/lidar_loc/lidar_loc.h`, add:

```cpp
bool IsInitializationResultValid(bool matcher_converged, const SE3& candidate_pose, const SE3& result_pose,
                                 double confidence, double min_confidence, double max_distance);
```

Add next to `min_init_confidence_` in `LidarLoc::Options`:

```cpp
float max_init_distance_ = 5.0F;  // 初始化结果距候选功能点的最大 XY 距离，单位 m
```

- [ ] **Step 4: Implement the pure safety gate**

Add `<cmath>` to `src/core/localization/lidar_loc/lidar_loc.cc`, then define the function before `LidarLoc::LidarLoc`:

```cpp
bool IsInitializationResultValid(bool matcher_converged, const SE3& candidate_pose, const SE3& result_pose,
                                 double confidence, double min_confidence, double max_distance) {
    if (!matcher_converged || !std::isfinite(confidence) || !std::isfinite(min_confidence) ||
        !std::isfinite(max_distance) || max_distance < 0.0) {
        return false;
    }
    if (!candidate_pose.translation().allFinite() || !candidate_pose.unit_quaternion().coeffs().allFinite() ||
        !result_pose.translation().allFinite() || !result_pose.unit_quaternion().coeffs().allFinite()) {
        return false;
    }
    const Eigen::Vector2d delta_xy =
        (result_pose.translation() - candidate_pose.translation()).head<2>();
    return confidence >= min_confidence && delta_xy.norm() <= max_distance;
}
```

- [ ] **Step 5: Run the focused test and verify it passes**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select lightning --cmake-target test_navigation_interfaces
./build/lightning/test/test_navigation_interfaces --gtest_filter=LidarLocInitialization.*
```

Expected: one test runs and passes.

- [ ] **Step 6: Commit the independently tested gate**

Run:

```bash
cd /home/tjzn/Workspace/src/lightning-lm
git add src/core/localization/lidar_loc/lidar_loc.h \
        src/core/localization/lidar_loc/lidar_loc.cc \
        test/test_navigation_interfaces.cc
git commit -m "test: add relocalization acceptance gate"
```

Expected: the commit includes only the three listed files.

---

### Task 2: Enforce NDT convergence and restore two-stage yaw search

**Files:**
- Modify: `src/core/localization/lidar_loc/lidar_loc.h:18-22`
- Modify: `src/core/localization/lidar_loc/lidar_loc.cc:226-308`
- Modify: `src/core/localization/lidar_loc/lidar_loc.cc:822-890`
- Modify: `test/test_navigation_interfaces.cc`

**Interfaces:**
- Consumes: `IsInitializationResultValid(...)` from Task 1, `NDTType::hasConverged()`, rough and fine NDT targets, and existing yaw search parameters.
- Produces: `bool IsNdtResultValid(bool converged, const Eigen::Matrix4f& transform, double confidence)`; `YawSearch(...) == true` only after both a valid rough candidate and a converged finite fine alignment; `InitWithFP(...) == true` only after the shared safety gate passes.

- [ ] **Step 1: Add a failing regression for invalid matcher output**

Add this test to `test/test_navigation_interfaces.cc`:

```cpp
TEST(LidarLocMatcher, RejectsNonConvergedAndNonFiniteNdtOutput) {
    const Eigen::Matrix4f valid = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f invalid = valid;
    invalid(0, 3) = std::numeric_limits<float>::quiet_NaN();

    EXPECT_TRUE(lightning::loc::IsNdtResultValid(true, valid, 2.0));
    EXPECT_FALSE(lightning::loc::IsNdtResultValid(false, valid, 2.0));
    EXPECT_FALSE(lightning::loc::IsNdtResultValid(true, invalid, 2.0));
    EXPECT_FALSE(lightning::loc::IsNdtResultValid(
        true, valid, std::numeric_limits<double>::infinity()));
}
```

- [ ] **Step 2: Run the focused test before changing matcher flow**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select lightning --cmake-target test_navigation_interfaces
./build/lightning/test/test_navigation_interfaces --gtest_filter=LidarLocInitialization.*
```

Expected: compilation fails because `lightning::loc::IsNdtResultValid` is not declared.

- [ ] **Step 3: Declare and implement matcher-result validation**

Next to `IsInitializationResultValid` in `lidar_loc.h`, add:

```cpp
bool IsNdtResultValid(bool converged, const Eigen::Matrix4f& transform, double confidence);
```

Next to the initialization gate definition in `lidar_loc.cc`, add:

```cpp
bool IsNdtResultValid(bool converged, const Eigen::Matrix4f& transform, double confidence) {
    return converged && transform.allFinite() && std::isfinite(confidence);
}
```

- [ ] **Step 4: Make `Localize` report actual matcher validity**

Replace the unconditional success block after `ndt->align(...)` with:

```cpp
    trans = ndt->getFinalTransformation();
    confidence = ndt->getTransformationProbability();
    loc_success = IsNdtResultValid(ndt->hasConverged(), trans, confidence);
    if (!loc_success) {
        LOG(WARNING) << "NDT failed: converged=" << ndt->hasConverged()
                     << ", confidence=" << confidence;
        return false;
    }
```

Leave the existing ICP refinement and final `SE3` conversion after this guard unchanged. This corrects both initialization and normal tracking semantics without applying the initialization confidence threshold to already-initialized tracking.

- [ ] **Step 5: Replace `YawSearch` with converged-candidate selection and fine refinement**

Use the existing yaw range and step values, but initialize scores to negative infinity and record only successful rough matches:

```cpp
bool LidarLoc::YawSearch(SE3& pose, double& confidence, CloudPtr input, CloudPtr output) {
    const SE3 init_pose = pose;
    auto rpyxyz = math::SE3ToRollPitchYaw(init_pose);
    const double init_yaw = rpyxyz.yaw;
    const int step = lidar_loc::grid_search_angle_step;
    const double radius = lidar_loc::grid_search_angle_range * constant::kDEG2RAD;
    if (step <= 0 || !std::isfinite(radius)) {
        LOG(ERROR) << "invalid yaw search configuration: step=" << step << ", radius=" << radius;
        return false;
    }

    const double angle_search_step = 2.0 * radius / step;
    double best_score = -std::numeric_limits<double>::infinity();
    SE3 best_pose = init_pose;
    bool found_rough_candidate = false;

    for (int i = 0; i < step; ++i) {
        rpyxyz.yaw = init_yaw + i * angle_search_step - radius;
        SE3 candidate = math::XYZRPYToSE3(rpyxyz);
        double candidate_score = -std::numeric_limits<double>::infinity();
        if (!Localize(candidate, candidate_score, input, output, true)) {
            continue;
        }
        if (!found_rough_candidate || candidate_score > best_score) {
            found_rough_candidate = true;
            best_score = candidate_score;
            best_pose = candidate;
        }
    }

    if (!found_rough_candidate) {
        confidence = best_score;
        LOG(WARNING) << "yaw search found no converged rough NDT candidate";
        return false;
    }

    pose = best_pose;
    confidence = best_score;
    if (!Localize(pose, confidence, input, output, false)) {
        LOG(WARNING) << "fine NDT failed after yaw search";
        return false;
    }

    LOG(INFO) << "yaw search fine result, score=" << confidence
              << ", pose=" << pose.translation().transpose();
    return true;
}
```

Add `#include <limits>` in `lidar_loc.cc`. Remove the unused `<execution>` include if no other code in that file uses it.

- [ ] **Step 6: Route functional-point initialization through yaw search and the gate**

Replace the direct match at the beginning of `InitWithFP` with:

```cpp
    double fitness_score = -std::numeric_limits<double>::infinity();
    SE3 pose_esti = fp_pose;
    CloudPtr output_cloud(new PointCloudType);
    const bool matcher_converged = YawSearch(pose_esti, fitness_score, input, output_cloud);
    loc_inited_ = IsInitializationResultValid(
        matcher_converged, fp_pose, pose_esti, fitness_score,
        options_.min_init_confidence_, options_.max_init_distance_);
```

In the failure log, include all relevant measurements:

```cpp
        const double distance_xy =
            (pose_esti.translation() - fp_pose.translation()).head<2>().norm();
        LOG(WARNING) << "init rejected: converged=" << matcher_converged
                     << ", score=" << fitness_score
                     << ", min_score=" << options_.min_init_confidence_
                     << ", distance_xy=" << distance_xy
                     << ", max_distance=" << options_.max_init_distance_;
```

- [ ] **Step 7: Rebuild and run all Lightning unit tests**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select lightning
colcon test --packages-select lightning --event-handlers console_direct+
colcon test-result --verbose
```

Expected: `lightning` builds; all tests report zero failures and zero errors.

- [ ] **Step 8: Commit the matcher flow correction**

Run:

```bash
cd /home/tjzn/Workspace/src/lightning-lm
git add src/core/localization/lidar_loc/lidar_loc.h \
        src/core/localization/lidar_loc/lidar_loc.cc \
        test/test_navigation_interfaces.cc
git commit -m "fix: validate automatic NDT relocalization"
```

Expected: the commit contains the NDT convergence guard, yaw search restoration, shared gate call, and the added regression only.

---

### Task 3: Synchronize candidate map targets and expose thresholds in YAML

**Files:**
- Modify: `src/core/localization/lidar_loc/lidar_loc.cc:60-90`
- Modify: `src/core/localization/lidar_loc/lidar_loc.cc:478-520`
- Modify: `config/default_linghou.yaml:86-99`
- Modify: `/home/tjzn/Workspace/src/lightning_nav2/config/default_linghou_navigation.yaml:86-100`

**Interfaces:**
- Consumes: YAML keys `lidar_loc.min_init_confidence` and `lidar_loc.max_init_distance`; `TiledMap::LoadOnPose(const SE3&)`; `LidarLoc::UpdateGlobalMap()`.
- Produces: every initialization attempt matches against a synchronously rebuilt target centered on that attempt's candidate; both Linghou entry points use `1.8` minimum confidence and `5.0 m` maximum XY displacement.

- [ ] **Step 1: Add configuration loading for maximum initialization distance**

Immediately after reading `min_init_confidence` in `LidarLoc::Init`, add:

```cpp
    options_.max_init_distance_ = yaml.GetValue<float>("lidar_loc", "max_init_distance");
```

Replace the existing single-value initialization log with:

```cpp
    LOG(INFO) << "init acceptance: min confidence=" << options_.min_init_confidence_
              << ", max XY distance=" << options_.max_init_distance_ << " m";
```

- [ ] **Step 2: Add identical thresholds to both Linghou configurations**

In each `lidar_loc:` block in `config/default_linghou.yaml` and `/home/tjzn/Workspace/src/lightning_nav2/config/default_linghou_navigation.yaml`, keep the existing confidence value and add:

```yaml
  min_init_confidence: 1.8
  max_init_distance: 5.0
```

- [ ] **Step 3: Synchronize the map before an external-pose attempt**

Change the `initial_pose_set_` branch in `LidarLoc::Align` to:

```cpp
        if (initial_pose_set_) {
            map_->LoadOnPose(initial_pose_);
            UpdateGlobalMap();
            if (InitWithFP(input, initial_pose_)) {
                LOG(INFO) << "init with external pose: " << initial_pose_.translation().transpose();
                initial_pose_set_ = false;
                return;
            }
        }
```

This keeps a failed `/initialpose` available for retry under the existing behavior and centers the 5 m acceptance radius on the supplied pose, not on the origin.

- [ ] **Step 4: Try `start` first and synchronously rebuild each automatic candidate target**

After `auto all_fps = map_->GetAllFP();`, stable-partition the candidates so the existing `start` point is tried first without discarding `recover` or other functional points:

```cpp
            std::stable_sort(all_fps.begin(), all_fps.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.name_ == "start" && rhs.name_ != "start";
            });
```

Change the loop body to:

```cpp
            for (const auto& fp : all_fps) {
                map_->LoadOnPose(fp.pose_);
                UpdateGlobalMap();
                if (InitWithFP(input, fp.pose_)) {
                    LOG(INFO) << "init with fp: " << fp.name_;
                    fp_init_success = true;
                    break;
                }
            }
```

- [ ] **Step 5: Verify configuration installation and package builds**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select lightning lightning_nav2
rg -n "min_init_confidence|max_init_distance" \
  src/lightning-lm/config/default_linghou.yaml \
  src/lightning_nav2/config/default_linghou_navigation.yaml \
  install/lightning/share/lightning/config/default_linghou.yaml \
  install/lightning_nav2/share/lightning_nav2/config/default_linghou_navigation.yaml
```

Expected: all four YAML files show `min_init_confidence: 1.8` and `max_init_distance: 5.0`; both selected packages build successfully.

- [ ] **Step 6: Run the full regression suite**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
colcon test --packages-select lightning --event-handlers console_direct+
colcon test-result --verbose
```

Expected: zero failed tests and zero test errors.

- [ ] **Step 7: Commit tracked Lightning changes without staging user-owned files**

Run:

```bash
cd /home/tjzn/Workspace/src/lightning-lm
git add src/core/localization/lidar_loc/lidar_loc.cc config/default_linghou.yaml
git commit -m "feat: configure origin-nearby relocalization"
```

Expected: only the two listed Lightning files are committed. The navigation YAML remains a workspace integration change unless its parent repository has an explicit tracked location.

---

### Task 4: Validate no-initial-pose behavior with live ROS 2 data

**Files:**
- Inspect: `/home/tjzn/Workspace/data/new_map/index.txt`
- Inspect: `/home/tjzn/Workspace/src/lightning_nav2/launch/navigation.launch.py`
- Inspect: `/home/tjzn/Workspace/command.txt`
- No source modifications expected.

**Interfaces:**
- Consumes: live `/driver/lidar/point_cloud/Data`, live `/driver/imu/Data`, map `/home/tjzn/Workspace/data/new_map`, and `/lightning/localization_status` (`INITIALIZING=1`, `GOOD=2`).
- Produces: recorded evidence that near-origin starts reach `GOOD` without `/initialpose`, while out-of-range or non-overlapping starts remain `INITIALIZING`.

- [ ] **Step 1: Confirm the live inputs and functional point before launch**

Run:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic hz /driver/lidar/point_cloud/Data
ros2 topic hz /driver/imu/Data
sed -n '1,80p' data/new_map/index.txt
```

Expected: both topics update continuously; the map index contains a `start` functional point near `(0, 0)`.

- [ ] **Step 2: Launch localization without publishing `/initialpose`**

Run in terminal A:

```bash
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
source install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch lightning_nav2 navigation.launch.py \
  map:=/home/tjzn/Workspace/data/new_map/map.yaml \
  lightning_config:=/home/tjzn/Workspace/src/lightning_nav2/config/default_linghou_navigation.yaml
```

Do not use RViz “2D Pose Estimate” during this check.

- [ ] **Step 3: Observe status, pose, confidence, and TF consistency**

Run in terminal B:

```bash
source /opt/ros/humble/setup.bash
source /home/tjzn/Workspace/install/setup.bash
ros2 topic echo /lightning/localization_status
ros2 topic echo /lightning/localization_pose
ros2 run tf2_ros tf2_echo map odom
```

Expected near the map origin: status changes from `1` to `2`; logs show a finite fine-NDT score at least `1.8` and XY distance from `start` no greater than `5.0 m`; the published pose is finite and `map -> odom -> base_link` remains the only TF chain.

- [ ] **Step 4: Repeat near-origin starts at distinct headings**

Stop the launch, rotate the stationary platform to at least three headings separated by roughly 90°, and repeat Steps 2–3 without `/initialpose`.

Expected: each run can select a different yaw candidate but reaches `GOOD`; record time-to-first-`GOOD`, final confidence, and logged XY distance for each heading.

- [ ] **Step 5: Verify rejection outside the configured local basin**

Start where the current scan has no valid overlap with the `start` map or where a resulting solution would be more than `5.0 m` from every tried functional point, again without `/initialpose`.

Expected: status remains `INITIALIZING`; logs report no converged candidate, low confidence, non-finite output, or excessive XY distance; no false `GOOD` is published.

- [ ] **Step 6: Verify `/initialpose` remains an independent local fallback**

At a mapped location away from the origin, publish a nearby RViz “2D Pose Estimate”.

Expected: Lightning synchronously loads the map around the supplied pose, applies the same confidence and 5 m local-distance gates centered on that pose, and reaches `GOOD` only after a valid fine-NDT result.

- [ ] **Step 7: Record verification evidence and final repository state**

Run:

```bash
cd /home/tjzn/Workspace/src/lightning-lm
git log -4 --oneline
git status --short
cd /home/tjzn/Workspace
colcon test-result --verbose
```

Expected: the relocalization commits are present; test results contain zero failures/errors; unrelated user-owned document moves remain untouched and are explicitly reported rather than committed.
