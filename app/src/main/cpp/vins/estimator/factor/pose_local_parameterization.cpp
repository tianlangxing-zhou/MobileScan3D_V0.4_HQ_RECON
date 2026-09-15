#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Eigen::Vector3d> p(x);
    Eigen::Map<const Eigen::Quaterniond> q(x + 3);
    Eigen::Map<const Eigen::Vector3d> dp(delta);
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p_out(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q_out(x_plus_delta + 3);
    p_out = p + dp;
    q_out = (q * dq).normalized();
    return true;
}

bool PoseLocalParameterization::PlusJacobian(const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();
    return true;
}

bool PoseLocalParameterization::Minus(const double* y, const double* x, double* y_minus_x) const {
    (void)y;
    (void)x;
    (void)y_minus_x;
    return false;
}

bool PoseLocalParameterization::MinusJacobian(const double* x, double* jacobian) const {
    (void)x;
    (void)jacobian;
    return false;
}
