// Adapted from FAST_LIO's use-ikfom.hpp for Lightning-LM integration
// Removes ROS dependencies and uses Lightning-LM types

#ifndef LIGHTNING_FASTLIO2_USE_IKFOM_HPP
#define LIGHTNING_FASTLIO2_USE_IKFOM_HPP

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_ikfom,
    ((vect3, pos))
    ((SO3, rot))
    ((SO3, offset_R_L_I))
    ((vect3, offset_T_L_I))
    ((vect3, vel))
    ((vect3, bg))
    ((vect3, ba))
    ((S2, grav))
);

MTK_BUILD_MANIFOLD(input_ikfom,
    ((vect3, acc))
    ((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
    ((vect3, ng))
    ((vect3, na))
    ((vect3, nbg))
    ((vect3, nba))
);

inline MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
    MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
    MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);
    MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001);
    MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001);
    MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001);
    return cov;
}

// Process model: f(x, u)
inline Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    vect3 a_inertial = s.rot * (in.acc - s.ba);
    for (int i = 0; i < 3; i++) {
        res(i) = s.vel[i];
        res(i + 3) = omega[i];
        res(i + 12) = a_inertial[i] + s.grav[i];
    }
    return res;
}

// Jacobian df/dx
inline Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero();
    cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
    vect3 acc_;
    in.acc.boxminus(acc_, s.ba);
    vect3 omega;
    in.gyro.boxminus(omega, s.bg);
    cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix() * MTK::hat(acc_);
    cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
    Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
    Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
    s.S2_Mx(grav_matrix, vec, 21);
    cov.template block<3, 2>(12, 21) = grav_matrix;
    cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
    return cov;
}

// Jacobian df/dw
inline Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
    cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
    cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
    cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
    return cov;
}

#endif  // LIGHTNING_FASTLIO2_USE_IKFOM_HPP
