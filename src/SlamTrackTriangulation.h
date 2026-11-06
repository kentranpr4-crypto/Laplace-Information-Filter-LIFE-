#ifndef SLAMTRACKTRIANGULATION_H
#define SLAMTRACKTRIANGULATION_H

/**
 * @file SlamTrackTriangulation.h
 * @brief Parallax-gated (delayed) initialisation of a point landmark from two camera poses, with the correlated
 *        information-form augmentation of Scenario 6 (keyframe pose A and current pose B both in the state).
 *
 * Geometry. Camera i (i = A: keyframe, i = B: current) has body pose eta_i = [r_i ; Theta_i] (r_i = r_BN^n,
 * Theta_i = ZYX Euler angles), camera-to-world rotation R_i = R_nb(Theta_i) Rbc and camera origin
 * c_i = r_i + R_nb(Theta_i) rCBb. A normalised undistorted ray a_i = (x_i, y_i, 1) (Camera::pixelToNormalised)
 * has world direction v_i = R_i a_i.
 *
 * Triangulation (least squares on the perpendicular distances). For the unit-free projectors
 * P_i = I - v_i v_i^T / (v_i^T v_i) the point minimising sum_i ||P_i (L - c_i)||^2 solves
 *   M L = b,   M = sum_i P_i,   b = sum_i P_i c_i.
 * For two rays this is the midpoint of the common perpendicular in the limit of exact rays; it is exact (zero residual)
 * for noise-free rays. M is 3 x 3 and symmetric positive definite exactly when the rays are not parallel; the
 * solve uses an LDLT factorisation (no explicit inverse of any covariance).
 *
 * Jacobians (hand-written, validated by central differences in test/src/SlamTrackTriangulation.cpp):
 *   dL = M^{-1} sum_i [ -dQ_i (c_i - L) + P_i dc_i ],   Q_i = v_i v_i^T / s_i,   s_i = v_i^T v_i,
 *   dQ_i = (dv_i v_i^T + v_i dv_i^T) / s_i - v_i v_i^T (2 v_i^T dv_i) / s_i^2,
 * evaluated for every parameter p of ray i:
 *   body position r_i (dv = 0, dc = e_k),
 *   Euler angle Theta_i,k (dv = dR_k Rbc a_i with dR_k = d R_nb / d Theta_k, dc = dR_k rCBb),
 *   normalised ray coordinate x_i or y_i (dv = R_i e_x or R_i e_y, dc = 0).
 *
 * Noise and augmentation. The pixel noise sigma_px (isotropic, per coordinate) becomes sigma_n = sigma_px / f in the
 * normalised coordinates (f = mean focal length; the distortion Jacobian is neglected, documented approximation).
 * Linearising L about the means: L = G x + c + w with
 *   G = dL/dx (3 x n, nonzero only on the two pose blocks), c = L_hat - G mu,
 *   w ~ N(0, Q),  Q = sigma_n^2 Ju Ju^T,  Ju = dL/d(x_A, y_A, x_B, y_B) (3 x 4).
 * The information-form factor lS of w (lS^T lS = Q^{-1}) comes from a thin QR of sigma_n Ju^T = Qr R, which gives
 * lS = R^{-T} (triangular solve, no inverse of Q). The pair (G, c, lS) is passed to
 * SystemVisualNav::addLandmarkLinear, which keeps the cross-covariance between the new landmark and both poses.
 *
 * Approximation (documented): the pose-noise of the keyframe is carried by the state (it is a state block), so the
 * birth is exact to first order in the state; the pixel noise of the two observations is the only extra noise. The
 * triangulation is the closed-form perpendicular-distance estimator, not the reprojection-error minimiser; for points
 * accepted by the birth gates (angle >= birthMinAngleDeg, reprojection consistency <= birthMaxReprojPx) the two estimators
 * agree to a fraction of the pixel noise.
 */

#include <Eigen/Core>
#include "Camera.h"

namespace slam
{

/** @brief Two-view triangulation of a static point from camera A (keyframe) and camera B (current). */
struct TwoViewTriangulation
{
    bool valid = false;                                 ///< finite point, in front of both cameras, non-degenerate
    Eigen::Vector3d L = Eigen::Vector3d::Constant(0.0); ///< world point r_LN^n [m]
    double angleDeg = 0.0;                              ///< angle between the two rays from the camera centres to L [deg]
    double depthA = 0.0;                                ///< depth of L in camera A [m]
    double depthB = 0.0;                                ///< depth of L in camera B [m]
    double maxReprojPx = 0.0;                           ///< max over both cameras of the normalised-ray misfit of L [px]
    Eigen::Matrix<double, 3, 6> dLdEtaA = Eigen::Matrix<double, 3, 6>::Zero();  ///< dL / d[rA; ThetaA]
    Eigen::Matrix<double, 3, 6> dLdEtaB = Eigen::Matrix<double, 3, 6>::Zero();  ///< dL / d[rB; ThetaB]
    Eigen::Matrix<double, 3, 2> dLdRayA = Eigen::Matrix<double, 3, 2>::Zero();  ///< dL / d[xA, yA]
    Eigen::Matrix<double, 3, 2> dLdRayB = Eigen::Matrix<double, 3, 2>::Zero();  ///< dL / d[xB, yB]
};

/**
 * @brief Triangulates a point from two poses and two normalised rays, with exact first derivatives.
 * @param camera   calibrated camera (Rbc, rCBb, focal length for the reprojection check)
 * @param etaA     keyframe body pose [rA; ThetaA] (world frame, ZYX Euler)
 * @param rayA     undistorted normalised coordinates (x, y) in camera A
 * @param etaB     current body pose [rB; ThetaB]
 * @param rayB     undistorted normalised coordinates (x, y) in camera B
 * @param withJac  fill the derivatives (otherwise they stay zero)
 * @throws std::invalid_argument if the camera is invalid or an input is not finite
 */
TwoViewTriangulation triangulateTwoView(const Camera& camera, const Eigen::Matrix<double, 6, 1>& etaA,
                                        const Eigen::Vector2d& rayA, const Eigen::Matrix<double, 6, 1>& etaB,
                                        const Eigen::Vector2d& rayB, bool withJac);

/** @brief Information-form birth of a point landmark from two keyframe-anchored observations. */
struct TwoViewBirth
{
    bool valid = false;               ///< triangulation valid and the noise factor is finite
    TwoViewTriangulation tri;         ///< triangulation at the state means
    Eigen::MatrixXd g;                ///< G = dL/dx (3 x n): columns of eta_B (offsetB..+5) and eta_A (offsetA..+5)
    Eigen::Vector3d c = Eigen::Vector3d::Zero();   ///< c = L_hat - G mu
    Eigen::Matrix3d lS = Eigen::Matrix3d::Zero();  ///< lS^T lS = Q^{-1}, Q = sigma_n^2 Ju Ju^T
};

/**
 * @brief Builds the correlated birth density (G, c, lS) of a point triangulated from two poses in the state.
 * @param camera   calibrated camera
 * @param mu       state mean (size n)
 * @param offsetA  first index of the keyframe pose [rA; ThetaA] in the state
 * @param offsetB  first index of the current pose [rB; ThetaB] in the state
 * @param pixelA   observation of the point in the keyframe (OpenCV pixel convention)
 * @param pixelB   observation of the point in the current frame
 * @param sigmaPx  pixel standard deviation of each observation coordinate [px]
 * @throws std::invalid_argument for invalid indices, non-positive sigma or a non-finite state
 */
TwoViewBirth twoViewBirth(const Camera& camera, const Eigen::VectorXd& mu, Eigen::Index offsetA, Eigen::Index offsetB,
                          const Eigen::Vector2d& pixelA, const Eigen::Vector2d& pixelB, double sigmaPx);

} // namespace slam

#endif
