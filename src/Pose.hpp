#ifndef POSE_HPP
#define POSE_HPP

#include <Eigen/Core>
#include "rotation.hpp"

/**
 * @brief Rigid-body transform helpers for the world (n), body (b), camera (c) and landmark (j) frames.
 *
 * Frame conventions (project-wide):
 *   - world n: NED-like (outdoor) or arbitrary fixed world with z down (indoor).
 *   - body b : x forward, y right, z down.
 *   - camera c: x right, y down, z along the optical axis.
 *   - Rbc: fixed extrinsic, columns are the camera axes expressed in body coordinates
 *     (v_b = Rbc v_c). Default [[0,0,1],[1,0,0],[0,1,0]].
 *   - R_nb = rpy2rot(Theta_nb). Camera orientation R_nc = R_nb Rbc.
 *   - rBNn / r_bn^n: origin of body b in world n, expressed in world coordinates.
 *   - rCBb: origin of camera c in body b, expressed in body coordinates (lever arm; zero by default).
 *
 * Camera point (world point p_n):  p_c = Rnc^T (p_n - r_cn) = Rbc^T ( R_nb^T (p_n - r_bn) - rCBb ).
 */
namespace rigid
{

/** @brief Default body -> camera extrinsic Rbc = [[0,0,1],[1,0,0],[0,1,0]] (camera axes expressed in body). */
inline Eigen::Matrix3d defaultRbc()
{
    Eigen::Matrix3d r;
    r << 0.0, 0.0, 1.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;
    return r;
}

/** @brief Rigid transform T = (R, t) mapping x -> R x + t. */
struct Transform
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

/** @brief Transform from ZYX Euler angles and translation. */
inline Transform fromRpy(const Eigen::Vector3d& Theta, const Eigen::Vector3d& t)
{
    return Transform{rpy2rot(Theta), t};
}

/** @brief Composition a * b (apply b first, then a). */
inline Transform compose(const Transform& a, const Transform& b)
{
    return Transform{a.R * b.R, a.R * b.t + a.t};
}

/** @brief Inverse transform: (R^T, -R^T t). */
inline Transform inverse(const Transform& a)
{
    return Transform{a.R.transpose(), -(a.R.transpose() * a.t)};
}

/** @brief Applies the transform to a point. */
inline Eigen::Vector3d apply(const Transform& a, const Eigen::Vector3d& p)
{
    return a.R * p + a.t;
}

/** @brief Body pose in world as a transform (R_nb, r_bn^n). */
inline Transform bodyInWorld(const Eigen::Vector3d& Theta_nb, const Eigen::Vector3d& r_bn_n)
{
    return fromRpy(Theta_nb, r_bn_n);
}

/** @brief Camera orientation in world R_nc = R_nb Rbc. */
inline Eigen::Matrix3d cameraRotationInWorld(const Eigen::Vector3d& Theta_nb, const Eigen::Matrix3d& Rbc)
{
    return rpy2rot(Theta_nb) * Rbc;
}

/** @brief Camera origin in world: r_cn^n = r_bn^n + R_nb rCBb. */
inline Eigen::Vector3d cameraOriginInWorld(const Eigen::Vector3d& Theta_nb, const Eigen::Vector3d& r_bn_n,
                                           const Eigen::Vector3d& rCBb)
{
    return r_bn_n + rpy2rot(Theta_nb) * rCBb;
}

/**
 * @brief World point -> camera coordinates p_c = Rbc^T (R_nb^T (p_n - r_bn) - rCBb).
 */
inline Eigen::Vector3d worldToCamera(const Eigen::Vector3d& p_n, const Eigen::Vector3d& r_bn_n,
                                     const Eigen::Vector3d& Theta_nb, const Eigen::Matrix3d& Rbc,
                                     const Eigen::Vector3d& rCBb)
{
    return Rbc.transpose() * (rpy2rot(Theta_nb).transpose() * (p_n - r_bn_n) - rCBb);
}

/** @brief Camera coordinates -> world point (inverse of worldToCamera). */
inline Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& p_c, const Eigen::Vector3d& r_bn_n,
                                     const Eigen::Vector3d& Theta_nb, const Eigen::Matrix3d& Rbc,
                                     const Eigen::Vector3d& rCBb)
{
    return r_bn_n + rpy2rot(Theta_nb) * (rCBb + Rbc * p_c);
}

/** @brief Camera-point value and Jacobians with respect to world point, body position and attitude. */
struct CameraPointJacobian
{
    Eigen::Vector3d pc;          ///< camera coordinates p_c
    Eigen::Matrix3d dPc_dPn;     ///< d p_c / d p_n = Rnc^T
    Eigen::Matrix3d dPc_drbn;    ///< d p_c / d r_bn^n = -Rnc^T
    Eigen::Matrix3d dPc_dTheta;  ///< d p_c / d Theta_nb, columns for phi, theta, psi
};

/**
 * @brief Camera point p_c and its exact Jacobians for p_c = Rbc^T (R_nb^T (p_n - r_bn) - rCBb).
 *
 * d p_c / d Theta_k = Rbc^T (dR_nb/dTheta_k)^T (p_n - r_bn).
 */
inline CameraPointJacobian cameraPointJacobian(const Eigen::Vector3d& p_n, const Eigen::Vector3d& r_bn_n,
                                               const Eigen::Vector3d& Theta_nb, const Eigen::Matrix3d& Rbc,
                                               const Eigen::Vector3d& rCBb)
{
    const Eigen::Matrix3d Rnb = rpy2rot(Theta_nb);
    const Eigen::Vector3d u = p_n - r_bn_n;
    const Eigen::Matrix3d Rnc_T = Rbc.transpose() * Rnb.transpose();

    CameraPointJacobian out;
    out.pc = Rbc.transpose() * (Rnb.transpose() * u - rCBb);
    out.dPc_dPn = Rnc_T;
    out.dPc_drbn = -Rnc_T;
    out.dPc_dTheta.col(0) = Rbc.transpose() * (drpy2rotDphi(Theta_nb).transpose() * u);
    out.dPc_dTheta.col(1) = Rbc.transpose() * (drpy2rotDtheta(Theta_nb).transpose() * u);
    out.dPc_dTheta.col(2) = Rbc.transpose() * (drpy2rotDpsi(Theta_nb).transpose() * u);
    return out;
}

} // namespace rigid

#endif
