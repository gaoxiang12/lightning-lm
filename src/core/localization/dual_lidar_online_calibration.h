#pragma once

#include <functional>
#include <limits>
#include <mutex>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace lightning::loc {

struct DualLidarCalibrationResult {
    double timestamp = 0.0;
    Eigen::Isometry3d T_front_rear = Eigen::Isometry3d::Identity();
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();
    double fitness = 0.0;
    int accepted_observations = 0;
};

class DualLidarOnlineCalibration {
   public:
    using ResultCallback = std::function<void(const DualLidarCalibrationResult&)>;

    bool Init(const std::string& yaml_path);

    void ProcessPointCloudPair(const sensor_msgs::msg::PointCloud2::SharedPtr& front_msg,
                               const sensor_msgs::msg::PointCloud2::SharedPtr& rear_msg);

    void SetResultCallback(ResultCallback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        result_callback_ = std::move(callback);
    }

    Eigen::Isometry3d Estimate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return T_front_rear_est_;
    }

    Eigen::Matrix<double, 6, 6> Covariance() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return P_;
    }

   private:
    using IcpPoint = pcl::PointXYZI;
    using IcpCloud = pcl::PointCloud<IcpPoint>;
    using IcpCloudPtr = IcpCloud::Ptr;

    struct RegistrationResult {
        Eigen::Isometry3d T_front_rear = Eigen::Isometry3d::Identity();
        Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Identity();

        double fitness = std::numeric_limits<double>::infinity();
        double residual_sigma = std::numeric_limits<double>::infinity();
        double hessian_condition = std::numeric_limits<double>::infinity();

        int correspondences = 0;
    };

    IcpCloudPtr ConvertAndFilter(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) const;

    void Predict();

    bool RunRegistration(const IcpCloudPtr& front_cloud,
                         const IcpCloudPtr& rear_cloud,
                         RegistrationResult& result) const;

    bool EstimateMeasurementCovariance(const IcpCloudPtr& front_cloud,
                                       const IcpCloudPtr& rear_cloud,
                                       RegistrationResult& result) const;

    bool Update(const RegistrationResult& result);

    void PublishResult(double timestamp, double fitness) const;
    void SaveYaml(double timestamp, double fitness) const;
    void LogEstimate(double timestamp, double fitness) const;

    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double Deg2Rad(double deg) { return deg * kPi / 180.0; }
    static constexpr double Rad2Deg(double rad) { return rad * 180.0 / kPi; }

    static Eigen::Isometry3d MakePose(const Eigen::Vector3d& t, const Eigen::Vector3d& rpy_rad);
    static Eigen::Vector3d RotationToRpy(const Eigen::Matrix3d& R);

    Eigen::Isometry3d T_front_rear_est_ = Eigen::Isometry3d::Identity();
    Eigen::Matrix<double, 6, 6> P_ = Eigen::Matrix<double, 6, 6>::Identity();

    mutable std::mutex mutex_;

    bool use_gicp_ = true;

    double voxel_leaf_size_ = 0.3;
    double blind_ = 0.5;
    int point_filter_num_ = 1;
    int min_points_ = 1000;

    int max_iterations_ = 40;
    double max_correspondence_distance_ = 2.0;
    double transformation_epsilon_ = 1e-6;
    double euclidean_fitness_epsilon_ = 1e-5;
    double fitness_score_max_range_ = 2.0;
    double max_fitness_score_ = 1.0;

    int accepted_observations_ = 0;
    int rejected_observations_ = 0;

    int output_interval_ = 10;
    std::string output_yaml_path_;

    ResultCallback result_callback_;
};

}  // namespace lightning::loc