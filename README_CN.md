# Lightning-LM

[English](./README.md) | 中文

Lightning-Speed Lidar Localization and Mapping

Lightning-LM 是一个完整的激光建图+定位模块。

Lightning-LM特性：

1. [done] 完整的3D Lidar SLAM，快速的LIO前端（AA-FasterLIO），标配
2. [done] 3D至2D地图转换（g2p5），选配，选上的话会输出实时的2D栅格，可以保存
3. [done] 实时回环检测，标配，选上的话会进行后端回环检测并闭环
4. [done] 流畅的高精3D Lidar 定位，标配
5. [done] 地图分区动态加载方案，适用大场景
6. [done] 动静态图层分离定位，适配动态场景，可选择动态图层的策略，选配，选上的话会保存动态图层的地图内容，有三种策略可以选（短期、中期、永久），默认永久
7. [done] 高频率IMU平滑输出，标配，100Hz
8. GPS地理信息关联，选配 (TODO)
9. 车辆里程计输入，选配 (TODO)
10. [done] 轻量优化库miao以及增量式优化（来自g2o，但更轻更快，支持增量优化，不需要重新构建优化模型），标配，在回环、定位中均有用到
11. [done] 离线与在线两种验证方案。离线可以断点调试，一致性强。在线可以多线程并发，处理速度快，可以设置动态跳帧，占用低。
12. [done] 基于外推器和平滑器的高频率输出，平滑因子可调
13. [done] 高性能计算：以上这些特性在纯CPU端不到一个核心就可以运行（在线定位0.8个核，建图1.2个核，32线雷达，无UI情况）

## 更新

### 2026.4.2

- 对建图和定位的稳定性进行了大量提升，已适配issue里提到的多层数据，云深处提供的多层数据，相关数据正在上传百度云，欢迎大家尝试！
- 调整了状态变量的结构和维度，现在ba, grav，offset_R, offset_t不用在线估计了，状态变量减到了12维（原本是23维）
- 添加定位部分激光定位的修正量比例，现在会基于LIO预测来进行定位，防止激光跳变太大
- LaserMapping部分加上了点到点ICP误差,点到点的部分也使用多线程加速
- ESKF部分加了一些实用的trick
- 定位现在会取LIO的关键帧来进行map to map配准
- 针对云深处的几个数据集和github issues里的数据集做了调参
- 调整了ESKF的接口以适应点到点ICP的情况（因为点到点ICP与点面ICP的维度不同）
- 添加了一些Kalman filter里的tricks: 对P阵做对称化，保护最小值等

### 2026.3.20

- MapIncremental在关键帧层面调用，增强LIO的鲁棒性（VBR数据集上不飘移）
- 修正了高度约束的雅可比问题 issue#100 #110
- 修正回环检测模块的越界问题 issue#88
- 适配云深处四足机器人数据 (RoboSense的lidar type=4)
- 增加了一些关于时间戳的数据检查(VBR存在异常时间戳问题）
- 回环检测现在使用优化后的位姿作为检测的初值(在大回环时有用)
- 如果voxel之后的点云数量太少，使用voxel之前的点云进行LIO（防止降采样之后点太少）
- 修正了std::vector<bool>在并行化时的问题
- 修正了几处可能导致seg fault的问题

### 2025.11.27

- 在LIO模块中添加了Cauchy's kernel
- 在定位模块中增加了配置： try_self_extrap，默认关闭。也就是定位模块不会根据自身外推的位姿做定位（因为定位间隔较大，车辆运动较大时不准）。
- 添加了一个livox的配置文件，因为用livox的人比较多
- 如果建图时设置了固定高度，那么定位也会使用这个地图高度（默认关闭）

### 2025.11.13

- 修复了FasterLIO中的两个逻辑问题
- 增加了高度约束，在loop_closing.with_height中配置。配置高度约束以后，lightning会保障输出地图的水平度（限制Z轴飘移），但这样就不适用于多层室内之类的带有立体结构的场景。

## 案例

- VBR campus数据集上的建图：

![](./doc/slam_vbr.gif)

- VBR上的定位

  ![](./doc/lm_loc_vbr_campus.gif)

- VBR上的地图
    - 点云

  ![](./doc/campus_vbr.png)
    - 栅格

  ![](./doc/campus.png)

- NCLT 数据集上的定位

![](./doc/lm_loc1_nclt.gif)

- 云深处四足机械狗上的数据

![](./doc/demo_ysc1.png)
![](./doc/demo_ysc2.png)
![](./doc/demo_ysc3.png)

- 斜装的DEMO
  ![](./doc/demo_github.png)

## 编译

### 环境

Ubuntu 22.04 或更高版本。

Ubuntu 20.04 应该也可行，未测试。

### 依赖

- ros2 humble 及以上
- Pangolin（用于可视化，见thirdparty）
- OpenCV
- PCL
- yaml-cpp
- glog
- gflags
- pcl_conversions

在Ubuntu 22.04上，执行：```bash ./scripts/install_dep.sh```即可。

### 编译

```colcon build```本包即可。

然后```source install/setup.bash```即可使用。

### 编译结果

编译后，会得到本包对应的在线/离线建图程序与定位程序。离线程序适用于存在离线数据包，快速得到建图/定位结果的方案，在线程序则适用于有实际传感器，得到实时结果的方案。

例如：在NCLT数据集上调用离线建图程序:
```ros2 run lightning run_slam_offline --input_bag ~/data/NCLT/20130110/20130110.db3 --config ./config/default_nclt.yaml```

如果希望调用在线的版本，则将offline部分改成online即可。

## 在数据集上测试

您可以直接使用我们转换完的数据集。如果需要原始的数据集，您需要将它们转换到ros2的db3格式。

转换后的数据集地址：

- OneDrive: https://1drv.ms/f/c/1a7361d22c554503/EpDSys0bWbxDhNGDYL_O0hUBa2OnhNRvNo2Gey2id7QMQA?e=7Ui0f5
- BaiduYun: https://pan.baidu.com/s/1XmFitUtnkKa2d0YtWquQXw?pwd=xehn 提取码: xehn

原始数据集地址：

- NCLT 数据集：http://robots.engin.umich.edu/nclt/
- UrbanLoco 数据集： https://github.com/weisongwen/UrbanLoco
- VBR 数据集：https://www.rvp-group.net/slam-dataset.html

### 建图测试

1. 实时建图（实时播包）
    - 启动建图程序:
      ```ros2 run lightning run_slam_online --config ./config/default_nclt.yaml```
    - 播放数据包
    - 保存地图 ```ros2 service call /lightning/save_map lightning/srv/SaveMap "{map_id: new_map}"```
2. 离线建图（遍历跑数据，更快一些）
    - ```ros2 run lightning run_slam_offline --config ./config/default_nclt.yaml --input_bag 数据包```
    - 结束后会自动保存至data/new_map目录下
3. 查看地图
    - 查看完整地图：```pcl_viewer ./data/new_map/global.pcd```
    - 实际地图是分块存储的，global.pcd仅用于显示结果
    - map.pgm存储了2D栅格地图信息
    - 请注意，在定位程序运行过程中或退出时，也可能在同目录存储动态图层的结果，所以文件可能会有更多。

### 定位测试

1. 实时定位
    - 将地图路径写到yaml中的 system-map_path 下，默认是new_map（和建图默认一致)
    - 将车放在建图起点处
    - 启动定位程序：
      ```ros2 run lightning run_loc_online --config ./config/default_nclt.yaml```
    - 播包或者输入传感器数据即可

2. 离线定位
    - ```ros2 run lightning run_loc_offline --config ./config/default_nclt.yaml --input_bag 数据包```

3. 接收定位结果
    - 定位程序输出与IMU同频的TF话题（50-100Hz）

### 在您自己的设备上调试

首先您需要知道自己的雷达类型，设置对应的fasterlio.lidar_type类型。livox系列的配置成1，Velodyne的设置成2,ouster设置成3.
如果不在以上种类，可以参考velodyne的设置方式。

比较简单的方式是先录一个ros2的数据包，将离线的建图、定位调通后，再去调试在线的情况。

您通常需要修改common.lidar_topic和common.imu_topic来设置雷达与imu的话题。

imu和雷达外参默认为零就好，我们对这个不敏感。

时间戳相关的fasterlio.time_scale是敏感的。您最好关注一下雷达点云是否带有每个点的时间戳，以及它们是否计算正确。这些代码在core/lio/pointcloud_preprocess里.

其他参数调整参考下一节。

### 云深处四足机器人

repo在这里：https://github.com/DeepRoboticsLab/lightning-lm-deep-robotics

视频：【具身智能第三期 | [Lynx M20] [SLAM] M20速腾雷达使用与二次开发，以复现lightning-lm为例】https://www.bilibili.com/video/BV12YQZBqE1b?vd_source=57f46145c37bfb96f7583c9e02081590

### 对lightning-lm进行微调

您可以在配置文件中对lightning进行微调，打开或者关闭一些功能。常见的配置项有：

- system.with_loop_closing 是否需要回环检测
- system.with_ui 是否需要3DUI
- system.with_2dui 是否需要2DUI
- system.with_g2p5 是否需要栅格地图
- system.map_path 地图的存储路径
- fasterlio.point_filter_num 点的采样数。调大后点数会少一些，计算更快，但不建议调到10以上。
- g2p5.esti_floor g2p5是否需要动态估计地面参数。如果雷达水平旋转且高度不变，可以关闭此选项.
- g2p5.grid_map_resolution 栅格地图的分辨率

### TODO

- [done] UI显示闭环后轨迹
- [done] 栅格地图保存为兼容ROS的形式
- [done] 检查栅格地图的分辨率取值是否正常
- 强制2D输出
- 额外便利性功能（打开关闭定位，重新初始化，指定位置等）

### 测试情况

1. 建图

- NCLT: pass
- VBR: pass
- Livox Multi Floor: pass
- github:
    - 斜装30度 https://github.com/gaoxiang12/lightning-lm/issues/75#issuecomment-4131131883 pass 需要关掉IMU filter
    - multi_floor 多层地图 pass 可以建图但是没法闭环
    - 纯室外 高架桥
- geely: pass
- yunshenchu
    - building1 多层室内外混合 pass
    - building2 pass
    - building3 pass
    - grass 需要把最小高度设高一些 例如0.5以上
    - road1 同上 pass

2. 定位

## 其他

1. 将ros1数据转换至ros2
   安装 ``` pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rosbags```

   转换: ```rosbags-convert --src [你的ROS1_bag文件.bag] --dst [输出ROS2bag目录]```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gaoxiang12/lightning-lm&type=date&legend=top-left)](https://www.star-history.com/#gaoxiang12/lightning-lm&type=date&legend=top-left)
