//
// Created by gaoxiang on 2020/8/10.
//

#pragma once

#ifndef LIGHTNING_EIGEN_TYPES_H
#define LIGHTNING_EIGEN_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cfloat>
#include <memory>
#include <mutex>

#include "Sophus/se2.hpp"
#include "Sophus/se3.hpp"

namespace lightning {

/// 数值类型缩写
using IdType = size_t;

// pose represented as sophus structs
using SO2 = Sophus::SO2d;
using SE2 = Sophus::SE2d;
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

/// numerical types
/// alias for eigen struct
using Vec2f = Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;
using Vec4f = Eigen::Matrix<float, 4, 1>;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Vec4d = Eigen::Vector4d;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec7d = Eigen::Matrix<double, 7, 1>;
using Vec12d = Eigen::Matrix<double, 12, 1>;
using Vec15d = Eigen::Matrix<double, 15, 1>;
using VecXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using Mat2f = Eigen::Matrix<float, 2, 2>;
using Mat3f = Eigen::Matrix<float, 3, 3>;
using Mat4f = Eigen::Matrix<float, 4, 4>;
using Mat6f = Eigen::Matrix<float, 6, 6>;
using Mat1d = Eigen::Matrix<double, 1, 1>;
using Mat2d = Eigen::Matrix<double, 2, 2>;
using Mat3d = Eigen::Matrix<double, 3, 3>;
using Mat4d = Eigen::Matrix<double, 4, 4>;
using Mat6d = Eigen::Matrix<double, 6, 6>;

using Mat12d = Eigen::Matrix<double, 12, 12>;
using Mat36f = Eigen::Matrix<float, 3, 6>;
using Mat23f = Eigen::Matrix<float, 2, 3>;
using Mat66f = Eigen::Matrix<float, 6, 6>;
using MatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

template <int N, typename T = double>
using VectorN = Eigen::Matrix<T, N, 1, Eigen::ColMajor>;
using Vector2 = VectorN<2>;
using Vector3 = VectorN<3>;
using Vector4 = VectorN<4>;
using Vector6 = VectorN<6>;
using Vector7 = VectorN<7>;
using VectorX = VectorN<Eigen::Dynamic>;

template <int N, typename T = double>
using MatrixN = Eigen::Matrix<T, N, N, Eigen::ColMajor>;
using Matrix2 = MatrixN<2>;
using Matrix3 = MatrixN<3>;
using Matrix4 = MatrixN<4>;
using MatrixX = MatrixN<Eigen::Dynamic>;

using H6d = Eigen::Matrix<double, 1, 6>;
using H1d = Eigen::Matrix<double, 1, 1>;

using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;

using Aff3d = Eigen::Affine3d;
using Quatd = Eigen::Quaternion<double>;
using Trans3d = Eigen::Translation3d;
using AngAxisd = Eigen::AngleAxis<double>;
using Quat = Quatd;
using Aff3f = Eigen::Affine3f;
using Quatf = Eigen::Quaternion<float>;
using Trans3f = Eigen::Translation3f;
using AngAxisf = Eigen::AngleAxis<float>;
using SE3f = Sophus::SE3f;

}  // namespace lightning

#endif
