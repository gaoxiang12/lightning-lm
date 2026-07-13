# AA-FasterLIO vs FAST-LIO2 技术对比

> 本文档从数据预处理、数据同步、点云去畸变、KNN、ICP配准、ESKF更新、地图管理七个维度，对比 lightning-lm 中两个 LIO 前端的实现差异。

---

## 1. 数据预处理

两个前端共享同一个 `PointCloudPreprocess` 类，但使用方式有差异。

### AA-FasterLIO (`laser_mapping.cc:139-141`)

```cpp
preprocess_.reset(new PointCloudPreprocess());
// Init 中设置参数
preprocess_->SetLidarType(LidarType::ROBOSENSE);  // lidar_type=4
preprocess_->NumScans() = yaml["fasterlio"]["scan_line"].as<int>();  // 128
preprocess_->PointFilterNum() = yaml["fasterlio"]["point_filter_num"].as<int>();  // 1
preprocess_->Blind() = yaml["fasterlio"]["blind"].as<double>();  // 0.5
preprocess_->SetHeightROI(-2.0, 5.0);  // 高度ROI裁剪
```

- `PointCloudPreprocess::Process()` 直接输出 `PointXYZIT`（time 单位：毫秒）
- 降采样后由 `laser_mapping.cc:201-203` 的 `VoxelGrid` 再做一次 scan-level 降采样
- 降采样失败时自动回退到更小分辨率 (`laser_mapping.cc:211-221`)

### FAST-LIO2 (`fastlio2_mapping.cc:17,64-73`)

```cpp
preprocess_ = std::make_shared<PointCloudPreprocess>();  // 基类成员
preprocess_->Set(static_cast<LidarType>(config.lidar_type), config.blind, 4);  // point_filter_num=4
preprocess_->SetHeightROI(height_max, height_min);  // 与 aa_fasterlio 一致
```

- 共享同一个 `PointCloudPreprocess`，预处理逻辑完全相同
- 降采样在 `DownSample()` 中完成 (`fastlio2_mapping.cc:279-282`)，使用 `filter_size_scan=0.5`
- **无自适应回退机制**：aa_fasterlio 在降采样过激时会用 0.1 分辨率重试，fast_lio2 没有这个逻辑

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| 预处理类 | `PointCloudPreprocess` | `PointCloudPreprocess`（相同） |
| point_filter_num | 可配置（yaml） | 固定为 4 |
| 降采样分辨率 | 可配置（yaml） | 固定 0.5 |
| 自适应降采样 | ✅ 有（<10% 或 < min_pts 时回退 0.1） | ❌ 无 |
| 高度 ROI | ✅ `SetHeightROI(-2.0, 5.0)` | ✅ 相同（修复后） |

---

## 2. 数据同步

### AA-FasterLIO (`laser_mapping.cc:386-407` + `lio_frontend.cc:48-88`)

`SyncPackages()` 调用基类公共同步逻辑后，额外填充 `MeasureGroup`：

```cpp
bool LaserMapping::SyncPackages() {
    if (!LIOFrontend::SyncPackages()) return false;  // 基类：检查缓冲区、计算 lidar_end_time_、收集 IMU
    measures_.scan_ = lidar_buffer_.front();
    measures_.lidar_begin_time_ = time_buffer_.front();
    measures_.lidar_end_time_ = lidar_end_time_;
    measures_.imu_.insert(measures_.imu_.end(), synced_imu_.begin(), synced_imu_.end());
    // 弹出已处理帧...
}
```

基类 `LIOFrontend::SyncPackages()` 的 `lidar_end_time_` 计算逻辑：
- 取 `cloud->points.back().time / 1000.0` 作为扫描结束时间（从点时间戳反推）
- 如果点时间戳异常（< 0.5 × 平均扫描时间），回退到 `lidar_mean_scantime_`
- 如果计算结果 > 5 × 平均扫描时间，也回退到 `lidar_mean_scantime_`

### FAST-LIO2 (`fastlio2_mapping.cc:203-252`)

有自己的 `SyncPackages()` 覆盖基类，计算逻辑完全相同：

```cpp
bool FASTLIO2Mapping::SyncPackages() {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    // ...与基类完全相同的 lidar_end_time_ 计算逻辑
    // （曾经不同，修复后与 aa_fasterlio 一致）
}
```

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| lidar_end_time_ 计算 | `points.back().time / 1000.0` + 安全回退 | 相同（修复后） |
| IMU 收集 | 基类统一收集 `synced_imu_` | 自己覆写了一次，逻辑相同 |
| 数据结构 | `MeasureGroup`（lightning-lm 原生） | `FASTLIO2MeasureGroup`（fast_lio2 适配层） |

---

## 3. IMU 初始化

### AA-FasterLIO (`imu_processing.hpp:125-173` + `275-313`)

```cpp
// 首帧：用第一个 IMU 样本作为初始均值
if (b_first_frame_) {
    mean_acc_ = meas.imu_.front()->linear_acceleration;  // 用真实数据
    mean_gyr_ = meas.imu_.front()->angular_velocity;
}

// 在线均值/方差更新（与 FAST-LIO2 相同的公式）
for (const auto &imu : meas.imu_) {
    mean_acc_ += (cur_acc - mean_acc_) / N;
    cov_acc_ = cov_acc_ * (N-1)/N + (cur_acc - mean_acc_).cwiseProduct(...) * (N-1)/(N*N);
}

// 状态初始化
init_state.grav_ = -mean_acc_ / mean_acc_.norm() * G_m_s2;  // 3D 向量
init_state.bg_ = mean_gyr_;
kf_state.ChangeX(init_state);

// 协方差：P = I，仅 bg 块设 0.0001
init_P.setIdentity();
init_P.block<3,3>(bgIdx, bgIdx) = 0.0001 * Mat3d::Identity();
```

**初始化后处理**（`Process()` 中 `init_iter_num_ > max_init_count_` 时）：
```cpp
// 加速度缩放因子：根据 mean_acc_norm 自动判断
if (mean_acc_norm > 0.5 && mean_acc_norm < 1.5) {
    acc_scale_factor_ = G_m_s2;      // 单位 g → m/s²
} else if (mean_acc_norm > 7.0 && mean_acc_norm < 12.0) {
    acc_scale_factor_ = 1.0;         // 已经是 m/s²
}
```

### FAST-LIO2 (`fastlio2_core.cc:183-232` + `234-268`)

```cpp
// 首帧：使用固定初始值
if (b_first_frame_) {
    mean_acc_ = Vec3d(0, 0, -1.0);  // 固定值，非真实数据
    mean_gyr_ = Vec3d(0, 0, 0);
}

// 在线均值/方差更新（与 AA-FasterLIO 相同的公式）
for (const auto& imu : meas.imu) {
    mean_acc_ += (cur_acc - mean_acc_) / N;
    cov_acc_(i) = cov_acc_(i) * k + cov_acc_delta(i) * (N-1)/(N*N);  // 分量循环
}

// 状态初始化
init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * G_m_s2);  // S2 流形
init_state.bg = mean_gyr_;
init_state.offset_T_L_I = Lidar_T_wrt_IMU_;  // 额外：外参
init_state.offset_R_L_I = Lidar_R_wrt_IMU_;
kf_.change_x(init_state);

// 协方差：更精细的分块设置
init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;      // rot
init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;   // ba
init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;   // offset_R
init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;    // offset_T
init_P(21,21) = init_P(22,22) = 0.00001;                   // grav
```

**初始化后处理**（无 acc_scale_factor，传播时用 `G_m_s2 / mean_acc_.norm()` 缩放）

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| 首帧均值 | 第一个 IMU 样本值 | 固定 `(0, 0, -1.0)` |
| 重力表示 | 3D 向量 `grav_` | S2 流形（2D 参数化） |
| 外参初始化 | 无（固定外参） | `offset_R_L_I`, `offset_T_L_I` 写入状态 |
| 加速度缩放 | 显式判断 `mean_acc_norm` 范围 | 隐式 `G_m_s2 / mean_acc_.norm()` |
| 协方差粒度 | 仅 bg 块调小 | rot/ba/offset_R/offset_T/grav 分别设置 |
| 初始化帧数 | 20 帧 (`max_init_count_`) | 10 帧 (`MAX_INI_COUNT`) |
| 在线均值/方差公式 | **相同** | **相同**（可抽取） |

> 可抽取部分：在线均值/方差更新循环，两者公式完全一致。

---

## 4. 点云去畸变

### AA-FasterLIO (`imu_processing.hpp:174-312`)

**前向传播**：使用 `ESKF::Predict()` 进行 IMU 预积分

```cpp
// imu_processing.hpp:203-251
for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    angvel_avr = 0.5 * (head->angular_velocity + tail->angular_velocity);
    acc_avr = 0.5 * (head->linear_acceleration + tail->linear_acceleration);
    acc_avr = acc_avr * acc_scale_factor_;  // 加速度缩放因子
    kf_state.Predict(dt, Q_, gyro, acc);    // 12D ESKF 预测
    // 保存 IMU pose...
}
```

**后向去畸变**（线性插值，1 阶）：

```cpp
// imu_processing.hpp:272-312
Mat3d R_i(R_imu * math::exp(angvel_avr, dt).matrix());
Vec3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
Vec3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos_);
Vec3d p_compensate = R_lidar_imu_.transpose() *
    (imu_state.rot_.inverse() * (R_i * (R_lidar_imu_ * P_i + t_lidar_mu_) + T_ei) - t_lidar_mu_);
```

### FAST-LIO2 (`fastlio2_core.cc:232-407`)

**前向传播**：使用 `kf_.predict()` 进行 23D ESEKF 预测

```cpp
// fastlio2_core.cc:328-361
for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    angvel_avr = 0.5 * (head->angular_velocity + tail->angular_velocity);
    acc_avr = 0.5 * (head->linear_acceleration + tail->linear_acceleration);
    acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();  // 加速度归一化
    in.acc = acc_avr; in.gyro = angvel_avr;
    kf_.predict(dt, Q_, in);  // 23D ESEKF 预测
    // 保存 IMU pose...
}
```

**后向去畸变**（二次多项式，2 阶）：

```cpp
// fastlio2_core.cc:379-406
Mat3d R_i = R_imu * Exp(angvel_avr, dt);
Vec3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
Vec3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos;
Vec3d P_compensate = imu_state.offset_R_L_I.conjugate() *
    (imu_state.rot.conjugate() *
     (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
     imu_state.offset_T_L_I);
```

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| 去畸变公式 | `R_lidar_imu^T * (R_end^T * (R_i * (R_lid * P + t_lid) + T_ei) - t_lid)` | 相同结构，但外参从 ESEKF 状态中读取 |
| 外参来源 | 固定外参（Init 时设定） | ESEKF 状态变量（在线估计 `offset_R_L_I`, `offset_T_L_I`） |
| 加速度缩放 | `acc_scale_factor_`（根据 mean_acc_norm 自动判断 1.0 或 G_m_s2） | `G_m_s2 / mean_acc_.norm()`（始终缩放） |
| IMU 滤波 | ✅ 可选低通滤波（`use_imu_filter_=true`） | ❌ 无滤波 |
| 时间戳修复 | 依赖基类安全回退 | 检测 garbage timestamp，自动分配线性时间戳 |
| 去畸变阶数 | 2 阶（acc + dt^2 项） | 2 阶（acc + dt^2 项），公式结构相同 |

> 注意：虽然代码注释标注 fast_lio2 为"2 阶"，aa_fasterlio 为"1 阶"，但实际代码中两者都使用了 `0.5 * acc * dt^2` 项，去畸变精度相当。

---

## 5. KNN（最近邻搜索）

### AA-FasterLIO (`laser_mapping.cc:505`)

使用 **iVox**（哈希体素网格）进行近似最近邻搜索：

```cpp
ivox_->GetClosestPoint(point_world, points_near, fasterlio::NUM_MATCH_POINTS);
point_selected_surf_[i] = points_near.size() >= fasterlio::MIN_NUM_MATCH_POINTS;
```

- 数据结构：iVox（`src/core/ivox3d/`），基于哈希表的体素网格
- 搜索半径：0.5m
- 最近邻数量：`NUM_MATCH_POINTS`（3-10 个）
- 近邻类型：`NEARBY6/18/26`（相邻体素数量）

### FAST-LIO2 (`fastlio2_core.cc:42-44`)

使用 **ikd-Tree**（增量 KD 树）进行精确最近邻搜索：

```cpp
core->ikdtree_.Nearest_Search(point_world, FASTLIO2Core::NUM_MATCH_POINTS,
                               points_near, pointSearchSqDis);
core->point_selected_surf_[i] = points_near.size() < FASTLIO2Core::NUM_MATCH_POINTS ? false :
                                pointSearchSqDis[FASTLIO2Core::NUM_MATCH_POINTS - 1] > 5 ? false : true;
```

- 数据结构：ikd-Tree（`thirdparty/ikd-Tree/`），增量 KD 树
- 搜索半径：5m（距离过滤阈值）
- 最近邻数量：`NUM_MATCH_POINTS`（5 个）
- 迭代时复用上一次搜索结果（`ekfom_data.converge` 时才重新搜索）

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| 数据结构 | iVox（哈希体素网格） | ikd-Tree（增量 KD 树） |
| 搜索性质 | 近似最近邻 | 精确最近邻 |
| 距离阈值 | 0.5m | 5m |
| 迭代复用 | 每次重新搜索 | 收敛时跳过搜索 |
| 内存开销 | 较大（哈希表） | 较小（树结构） |
| 插入/删除 | 增量插入，无显式删除 | 增量插入 + ray-casting 删除 |

---

## 6. ICP 配准

### AA-FasterLIO (`laser_mapping.cc:489-631`)

**双模式 ICP**：点到面 + 点到点混合

```cpp
// 点到面 ICP
float pd2 = plane_coef_[i].dot(temp);  // 点到平面距离
if (p_body.norm() > 81 * pd2 * pd2) {  // 距离阈值过滤
    point_selected_surf_[i] = true;
    residuals_[i] = pd2;
}

// 平面估计
point_selected_surf_[i] = math::esti_plane(plane_coef_[i], points_near, ESTI_PLANE_THRESHOLD);

// 雅可比矩阵（6D：位姿 [R|t]）
Vec3f C(Rt * norm_vec);
Vec3f A(point_crossmat * C);
J << norm_p.x, norm_p.y, norm_p.z, A(0), A(1), A(2);

// 混合残差
obs.HTH_ += JTJ[i] * options_.plane_icp_weight_;  // 点面权重
obs.HTr_ += JTr[i] * options_.plane_icp_weight_;
```

### FAST-LIO2 (`fastlio2_core.cc:15-138`)

**纯点到面 ICP**：

```cpp
// 平面估计（QR 分解）
Eigen::Vector3f normvec = A.colPivHouseholderQr().solve(b);

// 距离阈值过滤
float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
            pabcd(2) * point_world.z + pabcd(3);
float s = 1 - 0.9f * std::fabs(pd2) / sqrt(p_body.norm());
if (s > 0.9f) { point_selected_surf_[i] = true; }

// 雅可比矩阵（12D，但只填前 6 列）
ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
    A(0), A(1), A(2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
ekfom_data.h(i) = -norm_p.intensity;
```

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| ICP 类型 | 点到面 + 点到点混合 | 纯点到面 |
| 平面估计 | `esti_plane()` 自定义函数 | QR 分解 |
| 距离过滤 | `p.norm() > 81 * pd2^2` | `s = 1 - 0.9*|pd2|/sqrt(p.norm()) > 0.9` |
| 雅可比维度 | 6D（仅位姿） | 12D（位姿 + 外参，但外参列填 0） |
| 有效特征阈值 | `effect_feat_surf_ < 20` 时无效 | `effct_feat_num_ < 1` 时无效 |
| 收敛加速 | Anderson Acceleration（可选） | 无 |

---

## 7. ESKF 更新

### AA-FasterLIO：12D ESKF (`eskf.hpp`)

状态维度：12D `[pos(3), rot(3), vel(3), bg(3)]`

```cpp
// eskf.hpp:25-30
static constexpr int process_noise_dim_ = 12;
static constexpr int pose_obs_dim_ = 6;  // 激光观测只约束位姿
static constexpr int state_dim_ = NavState::dim;  // 12
```

更新流程：
1. `ObsModel()` 计算 `HTH_` 和 `HTr_`（`laser_mapping.cc:568-588`）
2. `kf_.Update(ESKF::ObsType::LIDAR, 1.0)` 执行 ESKF 更新
3. 支持 Anderson Acceleration 加速收敛
4. 退化检测：`degeneracy_threshold_ratio_` 控制协方差膨胀

### FAST-LIO2：23D ESEKF (`fastlio2_use_ikfom.hpp`)

状态维度：23D `[pos(3), rot(3), offset_R(3), offset_T(3), vel(3), bg(3), ba(3), grav(2)]`

```cpp
// fastlio2_use_ikfom.hpp:15-24
MTK_BUILD_MANIFOLD(state_ikfom,
    ((vect3, pos))
    ((SO3, rot))
    ((SO3, offset_R_L_I))    // 在线外参估计
    ((vect3, offset_T_L_I))  // 在线外参估计
    ((vect3, vel))
    ((vect3, bg))
    ((vect3, ba))
    ((S2, grav))             // 重力方向（球面流形）
);
```

更新流程：
1. `fastlio2_h_share_model()` 计算观测模型（`fastlio2_core.cc:15-138`）
2. `kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time)` 迭代更新
3. 使用 IKFoM 框架的流形运算
4. 重力方向参数化为 S2 流形（2D），更数值稳定

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| 状态维度 | 12D | 23D |
| 状态变量 | pos, rot, vel, bg | pos, rot, offset_R, offset_T, vel, bg, ba, grav |
| 外参 | 固定（Init 设定） | 在线估计 |
| 重力表示 | 3D 向量（`grav_`） | S2 流形（2D 参数化） |
| 加速度 bias | ❌ 无 | ✅ 有（`ba`） |
| EKF 框架 | 自研 ESKF（`eskf.hpp`） | IKFoM（`IKFoM_toolkit`） |
| 迭代收敛 | `max_iterations_=4` | `max_iteration`（配置） |
| Anderson Acceleration | ✅ 可选 | ❌ 无 |
| 退化检测 | ✅ 有 | ❌ 无 |

---

## 8. 地图管理

### AA-FasterLIO：iVox (`laser_mapping.cc:409-459`)

```cpp
void LaserMapping::MapIncremental() {
    // 1. 变换到世界坐标
    PointBodyToWorld(scan_down_body_->points[i], scan_down_world_->points[i]);

    // 2. 去重判断：比较当前点与最近邻到体素中心的距离
    Eigen::Vector3f center = ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;
    float dist = math::calc_dist(point_world.getVector3fMap(), center);

    // 如果最近邻已经在更近的位置，不添加
    for (int readd_i = 0; readd_i < fasterlio::NUM_MATCH_POINTS; readd_i++) {
        if (math::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
            need_add = false; break;
        }
    }

    // 3. 插入 iVox
    ivox_->AddPoints(points_to_add);
    ivox_->AddPoints(point_no_need_downsample);  // 跳过降采样的点
}
```

- 数据结构：iVox（哈希体素网格）
- 去重策略：与最近邻比较距离
- 无显式地图删除机制

### FAST-LIO2：ikd-Tree (`fastlio2_core.cc:445-504`)

```cpp
void FASTLIO2Core::UpdateMap(CloudPtr scan_world) {
    // 1. 变换到世界坐标
    PointBodyToWorld(feats_down_body_->points[i], feats_down_world_->points[i]);

    // 2. 去重判断：与最近邻比较到体素中心的距离
    PointType mid_point;
    mid_point.x = std::floor(feats_down_world_->points[i].x / config_.filter_size_map) *
                   config_.filter_size_map + 0.5 * config_.filter_size_map;
    // ... y, z 类似

    for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        float near_dist = (points_near[readd_i].x - mid_point.x) * ... ;
        if (near_dist < dist) { need_add = false; break; }
    }

    // 3. 插入 ikd-Tree
    ikdtree_.Add_Points(PointToAdd, true);       // 需要降采样的点
    ikdtree_.Add_Points(PointNoNeedDownsample, false);  // 无需降采样的点
}
```

### FAST-LIO2：ikd-Tree (`fastlio2_core.cc:445-504`)

逻辑与 iVox 版本高度相似，但数据结构不同：
- `ikdtree_.Add_Points()` 支持增量更新和平衡
- `ikdtree_.set_downsample_param(0.5)` 控制体素降采样
- **首次帧特殊处理**：`fastlio2_mapping.cc:131-143`，首帧构建 ikd-Tree 并 `return false`（跳过 Observe）

### 差异总结

| 项目 | AA-FasterLIO | FAST-LIO2 |
|------|-------------|-----------|
| 数据结构 | iVox（哈希体素网格） | ikd-Tree（增量 KD 树） |
| 去重逻辑 | 相同（体素中心距离比较） | 相同 |
| 降采样参数 | `filter_size_map_min_`（可配置） | `config_.filter_size_map`（0.5） |
| 首帧处理 | 直接添加到 iVox，正常处理 | 构建 ikd-Tree 后跳过 Observe |
| 增量插入 | ✅ | ✅ |
| 显式删除 | ❌（仅插入） | ✅（ray-casting 删除） |
| 内存管理 | 哈希表自动扩展 | 树结构动态平衡 |

---

## 总结对比表

| 维度 | AA-FasterLIO | FAST-LIO2 | 主要差异 |
|------|-------------|-----------|---------|
| **数据预处理** | `PointCloudPreprocess` + 自适应降采样 | `PointCloudPreprocess` + 固定降采样 | fast_lio2 无自适应回退 |
| **数据同步** | 基类统一收集 IMU | 覆写了 SyncPackages（逻辑相同） | 实质相同 |
| **IMU 初始化** | 首帧用真实数据，显式 acc_scale | 首帧用固定值，隐式归一化 | 均值/方差公式相同，可抽取 |
| **去畸变** | 12D ESKF 前向 + 共享后向公式 | 23D ESEKF 前向 + 共享后向公式 | 前向不同，后向已抽取 |
| **KNN** | iVox（近似，0.5m） | ikd-Tree（精确，5m） | 数据结构完全不同 |
| **ICP** | 点到面 + 点到点混合 | 纯点到面 | aa_fasterlio 更鲁棒 |
| **ESKF** | 12D，固定外参，可选 AA | 23D，在线外参，S2 重力 | fast_lio2 状态更丰富 |
| **地图管理** | iVox，仅插入 | ikd-Tree，插入 + ray-casting 删除 | fast_lio2 支持地图清理 |
| **轨迹精度** | 基准 | avg 0.037m 偏差 | 不同 EKF 公式的正常差异 |

### 设计取舍

- **AA-FasterLIO**：更紧凑的状态空间（12D），更快的 iVox 搜索，双模式 ICP 更鲁棒，退化检测更完善。适合快速部署和资源受限场景。
- **FAST-LIO2**：更丰富的状态估计（23D，含在线外参和加速度 bias），精确最近邻搜索，ikd-Tree 支持地图清理。适合高精度和长时间运行场景。
