# Lightning-LM

ROS2 (Humble) C++17 package for 3D LiDAR SLAM and localization. Build system: ament_cmake via `colcon build`.

## Build

```bash
# Install system dependencies (Ubuntu 22.04)
bash ./scripts/install_dep.sh

# Build (run from workspace root containing src/)
colcon build --packages-select lightning

# After build, source the workspace
source install/setup.bash
```

Pangolin is vendored in `thirdparty/`. If not installed system-wide, build it from `thirdparty/Pangolin-0.9.3/` first.

## Runtime

```bash
# Offline SLAM (mapping from bag)
ros2 run lightning run_slam_offline \
  --config ./src/lightning-lm/config/default_nclt.yaml \
  --input_bag <bag_file>

# Offline localization
ros2 run lightning run_loc_offline \
  --config ./src/lightning-lm/config/default_nclt.yaml \
  --input_bag <bag_file>

# Online variants: replace _offline with _online
# Save map via service:
ros2 service call /lightning/save_map lightning/srv/SaveMap "{map_id: new_map}"
```

**Config path note**: `--config` is relative to your current working directory, not the package root. Run from workspace root (e.g. `/home/tjzn/Workspace`) for paths like `./src/lightning-lm/config/...`.

## Runtime Gotchas

- **Pangolin segfault with UI**: Hardware OpenGL can crash. Fix: `export LIBGL_ALWAYS_SOFTWARE=1`
- **Pangolin library path**: If `libpango_plot.so` not found, add: `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
- **ROS2 environment**: Both system and workspace setup must be sourced:
  ```bash
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  ```
- **Config paths**: `--config` is relative to where you run the command, not to the package root.
- **bashrc shortcut**: All env vars are in `~/.bashrc` — new terminals auto-source ROS2 + Pangolin lib path + software rendering. If running in an existing shell without sourcing bashrc, run: `source ~/.bashrc`

## Architecture

```
src/
  app/              Entry points (7 executables)
  core/
    lio/            LIO front-end (ESKF, laser_mapping, IMU processing)
    loop_closing/   Loop closure detection
    g2p5/           3D-to-2D grid map conversion
    maps/           Tiled map for large-scale scenes
    localization/   Lidar localization (NDT-OMP, pose graph, PGO)
    system/         SLAM and localization system orchestration
    miao/           Lightweight optimization library (derived from g2o)
  ui/               Pangolin 3D visualization
  common/           Shared types, keyframe, nav_state, params
  io/               YAML and file I/O
  wrapper/          ROS bag I/O, ROS utilities
  utils/            Timer, pointcloud utilities
thirdparty/         Pangolin, Sophus, livox_ros_driver
config/             Per-dataset YAML configs
srv/                ROS service definitions (SaveMap, LocCmd)
```

## Executables

| Binary | Purpose |
|--------|---------|
| `run_slam_offline` | Offline SLAM from bag (fastest iteration) |
| `run_slam_online` | Online SLAM with live sensors |
| `run_loc_offline` | Offline localization from bag |
| `run_loc_online` | Online localization with live sensors |
| `run_frontend_offline` | LIO front-end only |
| `run_loop_offline` | Loop closure only |
| `test_ui` | UI test executable |

## Config System

YAML files in `config/`. Key sections:
- `common` — LiDAR/IMU topic names, dataset type
- `fasterlio` — LIO params: `lidar_type` (1=Livox, 2=Velodyne, 3=Ouster), `scan_line`, `time_scale`
- `system` — Feature toggles: `with_loop_closing`, `with_ui`, `with_2dui`, `with_g2p5`
- `g2p5` — Grid map params
- `loop_closing` — Loop closure thresholds
- `lidar_loc` — Localization params

LiDAR topic names are dataset-specific. Check bag metadata (`ros2 bag info <bag>`) to find the correct topics.

### Known configs

| Config | Topics | LiDAR type | Notes |
|--------|--------|------------|-------|
| `default_nclt.yaml` | `points_raw`, `imu_raw` | Velodyne 32 (type=2) | NCLT dataset; `with_ui: true` |
| `default_nclt_noui.yaml` | `points_raw`, `imu_raw` | Velodyne 32 (type=2) | Same but headless |
| `default.yaml` | `/livox/lidar`, `/livox/imu` | Livox (type=1) | Default config |

## Code Style

- `.clang-format` present — Google-based style, 120 column limit, 4-space indent
- C++17 required
- OpenMP used throughout for parallelism

## Dependencies

ros2-humble, Eigen3, PCL, OpenCV, Pangolin, glog, gflags, yaml-cpp, pcl_conversions, rosbag2_cpp, Sophus (vendored), TBB (via PCL)

## Common Warnings

- **`Failed to find match for field 'intensity'`**: Benign. The NCLT bag's PointCloud2 lacks an intensity field. LIO works correctly without it. The warning appears thousands of times but does not affect results.
