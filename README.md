# Lightning-LM

English | [中文](./README_CN.md)

Lightning-Speed Lidar Localization and Mapping

Lightning-LM is a complete laser mapping and localization module.

Features of Lightning-LM:

1. [done] Complete 3D Lidar SLAM, fast LIO front-end (AA-FasterLIO), standard
2. [done] 3D to 2D map conversion (g2p5), optional, if selected outputs real-time 2D grid map, can be saved
3. [done] Real-time loop closure detection, standard, performs back-end loop closure detection and correction if
   selected
4. [done] Smooth high-precision 3D Lidar localization, standard
5. [done] Dynamic loading scheme for map partitions, suitable for large-scale scenes
6. [done] Localization with separate dynamic and static layers, adaptable to dynamic scenes, selectable strategies for
   dynamic layer, optional, if selected saves dynamic layer map content, three strategies available (short-term,
   medium-term, permanent), default is permanent
7. [done] High-frequency IMU smooth output, standard, 100Hz
8. GPS geoinformation association, optional (TODO)
9. Vehicle odometry input, optional (TODO)
10. [done] Lightweight optimization library miao and incremental optimization (derived from g2o, but lighter and faster,
    supports incremental optimization, no need to rebuild optimization model), standard, used in both loop closure and
    localization
11. [done] Two verification schemes: offline and online. Offline allows breakpoint debugging with strong consistency.
    Online allows multi-threaded concurrency, fast processing speed, dynamic frame skipping, and low resource usage.
12. [done] High-frequency output based on extrapolator and smoother, adjustable smoothing factor
13. [done] High-performance computing: All the above features can run using less than one CPU core on the pure CPU
    side (online localization 0.8 cores, mapping 1.2 cores, 32-line LiDAR, without UI).

## Updates

### 2026.4.2

- Significantly improved the stability of mapping and localization, now adapted to multi-floor data mentioned in issues
  and data provided by Deep Robotics. Related data is being uploaded to BaiduYun — feel free to try it!
- Adjusted the structure and dimensions of state variables. `ba`, `grav`, `offset_R`, `offset_t` no longer need to be
  estimated online, reducing the state vector to 12 dimensions (previously 23).
- Added a correction scale for laser localization. Localization is now based on LIO prediction to prevent large laser
  jumps.
- Added point-to-point ICP error in the LaserMapping module; point-to-point computations are also accelerated with
  multi-threading.
- Added some practical tricks in the ESKF module.
- Localization now uses LIO keyframes for map-to-map registration.
- Tuned parameters for several Deep Robotics datasets and GitHub issue datasets.
- Adjusted the ESKF interface to accommodate point-to-point ICP (which has different dimensions from point-to-plane ICP).
- Added Kalman filter tricks: symmetrization of the P matrix, protection of minimum values, etc.

### 2026.3.20

- MapIncremental is now called at the keyframe level to improve LIO robustness (no drift on the VBR dataset).
- Fixed a Jacobian issue with the height constraint (issue #100, #110).
- Fixed an out-of-bounds issue in the loop closure detection module (issue #88).
- Adapted to Deep Robotics quadruped robot data (RoboSense lidar type=4).
- Added timestamp data checks (VBR has abnormal timestamp issues).
- Loop closure detection now uses the optimized pose as the initial estimate (useful for large loops).
- If the point cloud after voxelization has too few points, use the pre-voxelization point cloud for LIO (prevents too
  few points after downsampling).
- Fixed an issue with `std::vector<bool>` in parallelization.
- Fixed several issues that could cause segmentation faults.

### 2025.11.27

- Added Cauchy's kernel in the LIO module.
- Added the `try_self_extrap` configuration in the localization module (disabled by default). When disabled, the
  localization module does not use its own extrapolated pose for localization (since the localization interval is large
  and can be inaccurate when the vehicle moves significantly).
- Added a configuration file for Livox, as it is widely used.
- If a fixed height is set during mapping, localization will also use this map height (disabled by default).

### 2025.11.13

- Fix two logic typos in FasterLIO.
- Add global height constraint that can be configured in loop_closing.with_height. If set true, lightning-lm will keep
  the output map into the same height to avoid the Z-axis drifting in large scenes. It should not be set if you are
  using lightning-lm in scenes that have multi-floor or stairs structures.

## Examples

- Mapping on the VBR campus dataset:

  ![](./doc/slam_vbr.gif)

- Localization on VBR

  ![](./doc/lm_loc_vbr_campus.gif)

- Map on VBR
    - Point Cloud

  ![](./doc/campus_vbr.png)
    - Grid Map

  ![](./doc/campus.png)

- Localization on the NCLT dataset

![](./doc/lm_loc1_nclt.gif)

- Data on the Deep Robotics quadruped robot

![](./doc/demo_ysc1.png)
![](./doc/demo_ysc2.png)
![](./doc/demo_ysc3.png)

- Tilted mounting demo

  ![](./doc/demo_github.png)

## Build

### Environment

Ubuntu 22.04 or higher.

Ubuntu 20.04 should also work, but not tested.

### Dependencies

- ros2 humble or above
- Pangolin (for visualization, see thirdparty)
- OpenCV
- PCL
- yaml-cpp
- glog
- gflags
- pcl_conversions

On Ubuntu 22.04, run: ```bash ./scripts/install_dep.sh```.

### Build

Build this package with ```colcon build```.

Then ```source install/setup.bash``` to use it.

### Build Results

After building, you will get the corresponding online/offline mapping and localization programs for this package. The
offline programs are suitable for scenarios with offline data packets to quickly obtain mapping/localization results,
while the online programs are suitable for scenarios with actual sensors to obtain real-time results.

For example, calling the offline mapping program on the NCLT dataset:
```ros2 run lightning run_slam_offline --input_bag ~/data/NCLT/20130110/20130110.db3 --config ./config/default_nclt.yaml```

If you want to call the online version, just change the offline part to online.

## Testing on Datasets

You can directly use our converted datasets. If you need the original datasets, you need to convert them to the ros2 db3
format.

Converted dataset addresses:

- OneDrive: https://1drv.ms/f/c/1a7361d22c554503/EpDSys0bWbxDhNGDYL_O0hUBa2OnhNRvNo2Gey2id7QMQA?e=7Ui0f5
- BaiduYun: https://pan.baidu.com/s/1XmFitUtnkKa2d0YtWquQXw?pwd=xehn 提取码: xehn

Original dataset addresses:

- NCLT dataset: http://robots.engin.umich.edu/nclt/
- UrbanLoco dataset: https://github.com/weisongwen/UrbanLoco
- VBR dataset: https://www.rvp-group.net/slam-dataset.html

### Mapping Test

1. Real-time mapping (real-time bag playback)
    - Start the mapping program:
      ```ros2 run lightning run_slam_online --config ./config/default_nclt.yaml```
    - Play the data bag
    - Save the map ```ros2 service call /lightning/save_map lightning/srv/SaveMap "{map_id: new_map}"```
2. Offline mapping (traverse data, faster)
    - ```ros2 run lightning run_slam_offline --config ./config/default_nclt.yaml --input_bag [bag_file]```
    - It will automatically save to the data/new_map directory after finishing.
3. Viewing the map
    - View the full map: ```pcl_viewer ./data/new_map/global.pcd```
    - The actual map is stored in blocks, global.pcd is only for displaying the result.
    - map.pgm stores the 2D grid map information.
    - Note that during the localization program run or upon exit, results for dynamic layers might also be stored in the
      same directory, so there might be more files.

### Localization Test

1. Real-time localization
    - Write the map path to `system.map_path` in the yaml file, default is `new_map` (consistent with the mapping
      default).
    - Place the vehicle at the mapping starting point.
    - Start the localization program:
      ```ros2 run lightning run_loc_online --config ./config/default_nclt.yaml```
    - Play the bag or input sensor data.
2. Offline localization
    - ```ros2 run lightning run_loc_offline --config ./config/default_nclt.yaml --input_bag [bag_file]```
3. Receiving localization results
    - The localization program outputs TF topics at the same frequency as the IMU (50-100Hz).

### Debugging on Your Own Device

First, you need to know your LiDAR type and set the corresponding `fasterlio.lidar_type`. Set it to 1 for Livox series,
2 for Velodyne, 3 for Ouster.
If it's not one of the above types, you can refer to the Velodyne setup method.

A simpler way is to first record a ros2 bag, get offline mapping and localization working, and then debug the online
situation.

You usually need to modify `common.lidar_topic` and `common.imu_topic` to set the LiDAR and IMU topics.

The IMU and LiDAR extrinsic parameters can default to zero; we are not sensitive to them.

The `fasterlio.time_scale` related to timestamps is sensitive. You should pay attention to whether the LiDAR point cloud
has timestamps for each point and if they are calculated correctly. This code is in `core/lio/pointcloud_preprocess`.

Refer to the next section for other parameter adjustments.

### Deep Robotics Quadruped Robot

Repo: https://github.com/DeepRoboticsLab/lightning-lm-deep-robotics

Video: [Embodied Intelligence Episode 3 | [Lynx M20] [SLAM] M20 RoboSense LiDAR Usage and Secondary Development, Using lightning-lm as an Example] https://www.bilibili.com/video/BV12YQZBqE1b?vd_source=57f46145c37bfb96f7583c9e02081590

### Fine-tuning Lightning-LM

You can fine-tune Lightning by modifying the configuration file, turning some features on or off. Common configuration
items include:

- `system.with_loop_closing` Whether loop closure detection is needed
- `system.with_ui` Whether 3D UI is needed
- `system.with_2dui` Whether 2D UI is needed
- `system.with_g2p5` Whether grid map is needed
- `system.map_path` Storage path for the map
- `fasterlio.point_filter_num` Point sampling number. Increasing this results in fewer points, faster computation, but
  not recommended to set above 10.
- `g2p5.esti_floor` Whether g2p5 needs to dynamically estimate ground parameters. If the LiDAR rotates horizontally and
  the height is constant, you can turn this option off.
- `g2p5.grid_map_resolution` Resolution of the grid map

### TODO

- [done] UI displays trajectory after loop closure
- [done] Grid map saved in ROS-compatible format
- [done] Check if grid map resolution values are normal
- Force 2D output
- Additional convenience features (turn localization on/off, reinitialize, specify location, etc.)

### Test Results

1. Mapping

- NCLT: pass
- VBR: pass
- Livox Multi Floor: pass
- GitHub issues:
    - Tilted 30 degrees https://github.com/gaoxiang12/lightning-lm/issues/75#issuecomment-4131131883 pass (need to
      disable IMU filter)
    - multi_floor multi-floor map: pass (can map but cannot loop close)
    - Outdoor only, elevated bridge
- geely: pass
- Deep Robotics (yunshenchu):
    - building1 multi-floor indoor/outdoor mixed: pass
    - building2: pass
    - building3: pass
    - grass: need to increase minimum height, e.g., above 0.5
    - road1: same as above, pass

2. Localization

## Miscellaneous

1. Converting ros1 data to ros2
   Install ``` pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rosbags```

   Convert: ```rosbags-convert --src [your_ROS1_bag_file.bag] --dst [output_ROS2_bag_directory]```

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gaoxiang12/lightning-lm&type=date&legend=top-left)](https://www.star-history.com/#gaoxiang12/lightning-lm&type=date&legend=top-left)

