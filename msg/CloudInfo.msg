# Cloud Info
std_msgs/Header header

int32[] start_ring_index
int32[] end_ring_index

int32[] point_col_ind
float32[] point_range

int64 imu_available
int64 odom_available

# Attitude for LOAM initialization
float32 imu_pos_delta_x
float32 imu_pos_delta_y
float32 imu_pos_delta_z

float32 imu_roll_init
float32 imu_pitch_init
float32 imu_yaw_init

# Initial guess from imu pre-integration
float32 initial_guess_x
float32 initial_guess_y
float32 initial_guess_z
float32 initial_guess_roll
float32 initial_guess_pitch
float32 initial_guess_yaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed
sensor_msgs/PointCloud2 cloud_corner
sensor_msgs/PointCloud2 cloud_surface
