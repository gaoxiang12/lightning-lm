#include "core/localization/dual_lidar_online_calibration.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

#include "Sophus/se3.hpp"
#include "wrapper/ros_utils.h"

namespace lightning::loc {
namespace {

template <typename T>
T ReadYaml(const YAML::Node& node, const std::string& key, const T& default_value) {
    if (!node || !node[key]) {
        return default_value;
    }
    return node[key].as<T>();
}

Eigen::Isometry3d ReadTransform(const YAML::Node& node) {
    constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    if (node["translation"]) {
        const auto data = node["translation"].as<std::vector<double>>();
        if (data.size() == 3) {
            t = Eigen::Vector3d(data[0], data[1], data[2]);
        }
    }

    if (node["rotation"]) {
        const auto data = node["rotation"].as<std::vector<double>>();
        if (data.size() == 9) {
            R << data[0], data[1], data[2],
                 data[3], data[4], data[5],
                 data[6], data[7], data[8];
        }
    } else if (node["rpy_deg"]) {
        const auto data = node["rpy_deg"].as<std::vector<double>>();
        if (data.size() == 3) {
            const double roll = data[0] * kDeg2Rad;
            const double pitch = data[1] * kDeg2Rad;
            const double yaw = data[2] * kDeg2Rad;
            R = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
        }
    }

    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = R;
    pose.translation() = t;
    return pose;
}

}  // namespace

bool DualLidarOnlineCalibration::Init(const std::string& yaml_path) {
    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(yaml_path);
    } catch (const std::exception& e) {
        LOG(ERROR) << "failed to load dual lidar calibration yaml " << yaml_path << ": " << e.what();
        return false;
    }

    YAML::Node cfg = yaml["dual_lidar_online_calibration"];
    if (!cfg && yaml["localization"] && yaml["localization"]["dual_lidar_online_calibration"]) {
        cfg = yaml["localization"]["dual_lidar_online_calibration"];
    }

    std::string method = ReadYaml<std::string>(cfg, "registration_method", "gicp");
    std::transform(method.begin(), method.end(), method.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    use_gicp_ = method != "icp";

    voxel_leaf_size_ = ReadYaml<double>(cfg, "voxel_leaf_size", voxel_leaf_size_);
    blind_ = ReadYaml<double>(cfg, "blind", blind_);
    point_filter_num_ = std::max(1, ReadYaml<int>(cfg, "point_filter_num", point_filter_num_));
    min_points_ = ReadYaml<int>(cfg, "min_points", min_points_);
    max_iterations_ = ReadYaml<int>(cfg, "max_iterations", max_iterations_);
    max_correspondence_distance_ =
        ReadYaml<double>(cfg, "max_correspondence_distance", max_correspondence_distance_);
    transformation_epsilon_ = ReadYaml<double>(cfg, "transformation_epsilon", transformation_epsilon_);
    euclidean_fitness_epsilon_ = ReadYaml<double>(cfg, "euclidean_fitness_epsilon", euclidean_fitness_epsilon_);
    fitness_score_max_range_ = ReadYaml<double>(cfg, "fitness_score_max_range", fitness_score_max_range_);
    max_fitness_score_ = ReadYaml<double>(cfg, "max_fitness_score", max_fitness_score_);

    const YAML::Node loc_cfg = yaml["dual_lidar_localization"];
    if (!loc_cfg || !loc_cfg["T_front_rear"]) {
        LOG(ERROR) << "dual_lidar_localization.T_front_rear is required as the initial prior for dual lidar calibration";
        return false;
    }
    T_front_rear_est_ = ReadTransform(loc_cfg["T_front_rear"]);

    P_.setZero();
    for (int i = 0; i < 3; ++i) {
        P_(i, i) = 0.2 * 0.2;
        P_(i + 3, i + 3) = Deg2Rad(3.0) * Deg2Rad(3.0);
    }

    output_interval_ = std::max(1, ReadYaml<int>(cfg, "output_interval", output_interval_));
    output_yaml_path_ = ReadYaml<std::string>(cfg, "output_yaml_path", "");

    accepted_observations_ = 0;
    rejected_observations_ = 0;

    LOG(INFO) << "DualLidarOnlineCalibration init. method=" << (use_gicp_ ? "gicp" : "icp")
              << ", convention: p_front = T_front_rear * p_rear"
              << ", init_t=" << T_front_rear_est_.translation().transpose()
              << ", init_rpy_deg="
              << (RotationToRpy(T_front_rear_est_.linear()) * 180.0 / kPi).transpose()
              << ", model: fixed extrinsic, Q=0, R from Hessian, NIS gate";

    return true;
}

void DualLidarOnlineCalibration::ProcessPointCloudPair(
    const sensor_msgs::msg::PointCloud2::SharedPtr& front_msg,
    const sensor_msgs::msg::PointCloud2::SharedPtr& rear_msg) {
    if (!front_msg || !rear_msg) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const IcpCloudPtr front_cloud = ConvertAndFilter(front_msg);
    const IcpCloudPtr rear_cloud = ConvertAndFilter(rear_msg);

    if (!front_cloud || !rear_cloud ||
        static_cast<int>(front_cloud->size()) < min_points_ ||
        static_cast<int>(rear_cloud->size()) < min_points_) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB][SKIP] not enough points. front="
                     << (front_cloud ? front_cloud->size() : 0)
                     << " rear=" << (rear_cloud ? rear_cloud->size() : 0)
                     << " min=" << min_points_;
        return;
    }

    const double timestamp = ToSec(front_msg->header.stamp);

    Predict();

    RegistrationResult result;
    if (!RunRegistration(front_cloud, rear_cloud, result)) {
        rejected_observations_++;
        return;
    }

    if (!Update(result)) {
        rejected_observations_++;
        return;
    }

    accepted_observations_++;
    LogEstimate(timestamp, result.fitness);
    if (accepted_observations_ % output_interval_ == 0) {
        SaveYaml(timestamp, result.fitness);
    }

    PublishResult(timestamp, result.fitness);
}

DualLidarOnlineCalibration::IcpCloudPtr DualLidarOnlineCalibration::ConvertAndFilter(
    const sensor_msgs::msg::PointCloud2::SharedPtr& msg) const {
    IcpCloudPtr raw(new IcpCloud());
    IcpCloudPtr finite(new IcpCloud());
    IcpCloudPtr filtered(new IcpCloud());

    pcl::fromROSMsg(*msg, *raw);

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*raw, *finite, indices);

    const double blind2 = blind_ * blind_;
    filtered->reserve(finite->size());

    for (size_t i = 0; i < finite->size(); i += static_cast<size_t>(point_filter_num_)) {
        const IcpPoint& p = finite->points[i];

        const double range2 = p.x * p.x + p.y * p.y + p.z * p.z;
        if (range2 <= blind2) {
            continue;
        }

        filtered->push_back(p);
    }

    filtered->width = static_cast<uint32_t>(filtered->size());
    filtered->height = 1;
    filtered->is_dense = false;

    if (voxel_leaf_size_ <= 0.0) {
        return filtered;
    }

    IcpCloudPtr downsampled(new IcpCloud());

    pcl::VoxelGrid<IcpPoint> voxel;
    const float leaf = static_cast<float>(voxel_leaf_size_);
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.setInputCloud(filtered);
    voxel.filter(*downsampled);

    return downsampled;
}

void DualLidarOnlineCalibration::Predict() {
    // Fixed extrinsic model:
    //
    //     T_front_rear,k = T_front_rear,k-1
    //     P_k^- = P_k-1^+
    //
    // The true extrinsic is assumed constant, so Q = 0.
    // If mechanical drift needs to be modeled later, add a physically justified small Q here.
}

bool DualLidarOnlineCalibration::RunRegistration(const IcpCloudPtr& front_cloud,
                                                 const IcpCloudPtr& rear_cloud,
                                                 RegistrationResult& result) const {
    const Eigen::Matrix4f initial_guess = T_front_rear_est_.matrix().cast<float>();

    IcpCloud aligned;
    Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();

    bool converged = false;
    double fitness = std::numeric_limits<double>::infinity();

    if (use_gicp_) {
        pcl::GeneralizedIterativeClosestPoint<IcpPoint, IcpPoint> reg;
        reg.setMaxCorrespondenceDistance(max_correspondence_distance_);
        reg.setMaximumIterations(max_iterations_);
        reg.setTransformationEpsilon(transformation_epsilon_);
        reg.setInputSource(rear_cloud);
        reg.setInputTarget(front_cloud);
        reg.align(aligned, initial_guess);

        converged = reg.hasConverged();
        fitness = reg.getFitnessScore(fitness_score_max_range_);
        final_transform = reg.getFinalTransformation();
    } else {
        pcl::IterativeClosestPoint<IcpPoint, IcpPoint> reg;
        reg.setMaxCorrespondenceDistance(max_correspondence_distance_);
        reg.setMaximumIterations(max_iterations_);
        reg.setTransformationEpsilon(transformation_epsilon_);
        reg.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon_);
        reg.setRANSACIterations(0);
        reg.setInputSource(rear_cloud);
        reg.setInputTarget(front_cloud);
        reg.align(aligned, initial_guess);

        converged = reg.hasConverged();
        fitness = reg.getFitnessScore(fitness_score_max_range_);
        final_transform = reg.getFinalTransformation();
    }

    if (!converged || !std::isfinite(fitness)) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB][ICP_FAIL] converged=" << converged << " fitness=" << fitness;
        return false;
    }

    if (max_fitness_score_ > 0.0 && fitness > max_fitness_score_) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB][REJECT_FITNESS] fitness=" << fitness << " > " << max_fitness_score_;
        return false;
    }

    Eigen::Matrix3d R = final_transform.block<3, 3>(0, 0).cast<double>();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();

    if (R.determinant() < 0.0) {
        Eigen::Matrix3d U = svd.matrixU();
        U.col(2) *= -1.0;
        R = U * svd.matrixV().transpose();
    }

    result.T_front_rear.setIdentity();
    result.T_front_rear.linear() = R;
    result.T_front_rear.translation() =
        final_transform.block<3, 1>(0, 3).cast<double>();
    result.fitness = fitness;

    if (!EstimateMeasurementCovariance(front_cloud, rear_cloud, result)) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB][REJECT_COV] failed to estimate measurement covariance";
        return false;
    }

    LOG(INFO) << "[DUAL_LIDAR_CALIB][MEAS] fitness=" << result.fitness
              << " correspondences=" << result.correspondences
              << " sigma=" << result.residual_sigma
              << " hessian_cond=" << result.hessian_condition;

    return true;
}

bool DualLidarOnlineCalibration::EstimateMeasurementCovariance(
    const IcpCloudPtr& front_cloud,
    const IcpCloudPtr& rear_cloud,
    RegistrationResult& result) const {
    if (!front_cloud || !rear_cloud || front_cloud->empty() || rear_cloud->empty()) {
        return false;
    }

    IcpCloudPtr rear_in_front(new IcpCloud());
    pcl::transformPointCloud(*rear_cloud,
                             *rear_in_front,
                             result.T_front_rear.matrix().cast<float>());

    pcl::KdTreeFLANN<IcpPoint> kdtree;
    kdtree.setInputCloud(front_cloud);

    constexpr int kPlaneNeighbors = 10;
    constexpr int kMinCorrespondences = 100;

    const double max_dist2 = max_correspondence_distance_ * max_correspondence_distance_;

    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    double residual_sq_sum = 0.0;
    int used = 0;

    std::vector<int> nn_indices(kPlaneNeighbors);
    std::vector<float> nn_dists2(kPlaneNeighbors);

    const Eigen::Matrix3d R = result.T_front_rear.linear();

    for (size_t i = 0; i < rear_cloud->size(); ++i) {
        const IcpPoint& p_rear_raw = rear_cloud->points[i];
        const IcpPoint& p_front_raw = rear_in_front->points[i];

        if (kdtree.nearestKSearch(p_front_raw, kPlaneNeighbors, nn_indices, nn_dists2) < kPlaneNeighbors) {
            continue;
        }

        if (static_cast<double>(nn_dists2[0]) > max_dist2) {
            continue;
        }

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();

        for (int index : nn_indices) {
            const IcpPoint& q = front_cloud->points[index];
            centroid += Eigen::Vector3d(q.x, q.y, q.z);
}

        centroid /= static_cast<double>(kPlaneNeighbors);

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

        for (int index : nn_indices) {
            const IcpPoint& q = front_cloud->points[index];
            const Eigen::Vector3d dq = Eigen::Vector3d(q.x, q.y, q.z) - centroid;
            cov += dq * dq.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> plane_solver(cov);
        if (plane_solver.info() != Eigen::Success) {
            continue;
        }

        const Eigen::Vector3d normal = plane_solver.eigenvectors().col(0).normalized();
        const Eigen::Vector3d p_rear(p_rear_raw.x, p_rear_raw.y, p_rear_raw.z);
        const Eigen::Vector3d p_front(p_front_raw.x, p_front_raw.y, p_front_raw.z);

        const double residual = normal.dot(p_front - centroid);

        Eigen::Matrix3d skew;
        skew << 0.0, -p_rear.z(), p_rear.y(),
                p_rear.z(), 0.0, -p_rear.x(),
                -p_rear.y(), p_rear.x(), 0.0;

        Eigen::Matrix<double, 1, 6> J;
        J.block<1, 3>(0, 0) = normal.transpose() * R;
        J.block<1, 3>(0, 3) = -normal.transpose() * R * skew;

        H += J.transpose() * J;
        residual_sq_sum += residual * residual;
        used++;
    }

    if (used < kMinCorrespondences) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB][COV_FAIL] too few correspondences: "
                     << used << " < " << kMinCorrespondences;
        return false;
    }

    const double sigma2 =
        std::max(1e-8, residual_sq_sum / static_cast<double>(std::max(1, used - 6)));

    const Eigen::Matrix<double, 6, 6> H_sym = 0.5 * (H + H.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(H_sym);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::Matrix<double, 6, 1> eig = solver.eigenvalues();

    const double lambda_min = std::max(1e-12, eig.minCoeff());
    const double lambda_max = std::max(1e-12, eig.maxCoeff());

    result.hessian_condition = lambda_max / lambda_min;

    for (int i = 0; i < 6; ++i) {
        eig(i) = 1.0 / std::max(eig(i), 1e-6);
    }

    result.covariance =
        sigma2 *
        solver.eigenvectors() *
        eig.asDiagonal() *
        solver.eigenvectors().transpose();

    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    result.correspondences = used;
    result.residual_sigma = std::sqrt(sigma2);

    return result.covariance.allFinite();
}

bool DualLidarOnlineCalibration::Update(const RegistrationResult& result) {
    Sophus::SE3d T_pred(Eigen::Quaterniond(T_front_rear_est_.linear()),
                        T_front_rear_est_.translation());

    Sophus::SE3d T_meas(Eigen::Quaterniond(result.T_front_rear.linear()),
                        result.T_front_rear.translation());

    const Eigen::Matrix<double, 6, 1> residual =
        (T_pred.inverse() * T_meas).log();

    const Eigen::Matrix<double, 6, 6> S =
        P_ + result.covariance;

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB][REJECT_NIS] S decomposition failed";
        return false;
    }

    const Eigen::Matrix<double, 6, 1> S_inv_residual = ldlt.solve(residual);
    const double nis = residual.dot(S_inv_residual);

    constexpr double kChiSquare6Dof99 = 16.81;

    if (!std::isfinite(nis) || nis > kChiSquare6Dof99) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB][REJECT_NIS] nis=" << nis
                     << " threshold=" << kChiSquare6Dof99
                     << " residual_trans=" << residual.head<3>().norm()
                     << " residual_rot_deg=" << Rad2Deg(residual.tail<3>().norm())
                     << " fitness=" << result.fitness
                     << " sigma=" << result.residual_sigma
                     << " hessian_cond=" << result.hessian_condition;
        return false;
    }

    const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();

    const Eigen::Matrix<double, 6, 6> K = P_ * ldlt.solve(I);
    const Eigen::Matrix<double, 6, 1> delta = K * residual;

    Sophus::SE3d T_new = T_pred * Sophus::SE3d::exp(delta);
    T_front_rear_est_.setIdentity();
    T_front_rear_est_.linear() = T_new.rotationMatrix();
    T_front_rear_est_.translation() = T_new.translation();

    const Eigen::Matrix<double, 6, 6> IK = I - K;

    P_ = IK * P_ * IK.transpose() + K * result.covariance * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());

    LOG(INFO) << "[DUAL_LIDAR_CALIB][UPDATE_OK] nis=" << nis
              << " delta_trans=" << delta.head<3>().norm()
              << " delta_rot_deg=" << Rad2Deg(delta.tail<3>().norm())
              << " residual_trans=" << residual.head<3>().norm()
              << " residual_rot_deg=" << Rad2Deg(residual.tail<3>().norm())
              << " P_diag=" << P_.diagonal().transpose();

    return true;
}

void DualLidarOnlineCalibration::PublishResult(double timestamp, double fitness) const {
    if (!result_callback_) {
        return;
    }

    DualLidarCalibrationResult result;
    result.timestamp = timestamp;
    result.T_front_rear = T_front_rear_est_;
    result.covariance = P_;
    result.fitness = fitness;
    result.accepted_observations = accepted_observations_;

    result_callback_(result);
}

void DualLidarOnlineCalibration::SaveYaml(double timestamp, double fitness) const {
    if (output_yaml_path_.empty()) {
        return;
    }

    const Eigen::Quaterniond q(T_front_rear_est_.linear());
    const Eigen::Vector3d rpy = RotationToRpy(T_front_rear_est_.linear());

    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "convention" << YAML::Value << "p_front = T_front_rear * p_rear";
    out << YAML::Key << "timestamp" << YAML::Value << timestamp;
    out << YAML::Key << "accepted_observations" << YAML::Value << accepted_observations_;
    out << YAML::Key << "rejected_observations" << YAML::Value << rejected_observations_;
    out << YAML::Key << "fitness" << YAML::Value << fitness;
    out << YAML::Key << "T_front_rear" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "translation" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << T_front_rear_est_.translation().x() << T_front_rear_est_.translation().y()
        << T_front_rear_est_.translation().z() << YAML::EndSeq;
    out << YAML::Key << "rpy_deg" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << Rad2Deg(rpy.x()) << Rad2Deg(rpy.y()) << Rad2Deg(rpy.z()) << YAML::EndSeq;
    out << YAML::Key << "quaternion_xyzw" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << q.x() << q.y() << q.z() << q.w() << YAML::EndSeq;
    out << YAML::EndMap;
    out << YAML::Key << "covariance_diag" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int i = 0; i < 6; ++i) {
        out << P_(i, i);
    }
    out << YAML::EndSeq;
    out << YAML::EndMap;

    std::ofstream ofs(output_yaml_path_);
    if (!ofs) {
        LOG(WARNING) << "[DUAL_LIDAR_CALIB] failed to open output yaml: " << output_yaml_path_;
        return;
    }
    ofs << out.c_str() << std::endl;

    LOG(INFO) << "[DUAL_LIDAR_CALIB][SAVE_YAML] path=" << output_yaml_path_
          << " timestamp=" << timestamp
          << " accepted=" << accepted_observations_
          << " rejected=" << rejected_observations_
          << " fitness=" << fitness
          << " translation=" << T_front_rear_est_.translation().transpose()
          << " rpy_deg=" << (rpy * 180.0 / kPi).transpose()
          << " quaternion_xyzw=" << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
          << " covariance_diag=" << P_.diagonal().transpose();
}

void DualLidarOnlineCalibration::LogEstimate(double timestamp, double fitness) const {
    const Eigen::Vector3d rpy = RotationToRpy(T_front_rear_est_.linear());

    LOG(INFO) << "[DUAL_LIDAR_CALIB][EST] timestamp=" << timestamp
              << " accepted=" << accepted_observations_
              << " rejected=" << rejected_observations_
              << " fitness=" << fitness
              << " translation=" << T_front_rear_est_.translation().transpose()
              << " rpy_deg=" << (rpy * 180.0 / kPi).transpose()
              << " covariance_diag=" << P_.diagonal().transpose();
}

Eigen::Isometry3d DualLidarOnlineCalibration::MakePose(const Eigen::Vector3d& t,
                                                       const Eigen::Vector3d& rpy_rad) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = (Eigen::AngleAxisd(rpy_rad.z(), Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(rpy_rad.y(), Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(rpy_rad.x(), Eigen::Vector3d::UnitX()))
                        .toRotationMatrix();
    pose.translation() = t;
    return pose;
}

Eigen::Vector3d DualLidarOnlineCalibration::RotationToRpy(const Eigen::Matrix3d& R) {
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double roll = std::atan2(R(2, 1), R(2, 2));
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
}

}  // namespace lightning::loc
