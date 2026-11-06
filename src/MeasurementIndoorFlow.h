#ifndef MEASUREMENTINDOORFLOW_H
#define MEASUREMENTINDOORFLOW_H

/**
 * @file MeasurementIndoorFlow.h
 * @brief Depth-free optical-flow odometry likelihood between the delayed pose etaPrev = eta_{k-1} and eta = eta_k,
 *        shared by Scenario 5 and Scenario 6 (indoor, monocular, static scene).
 *
 * Frames and state. Camera A is the camera at time k-1 (state etaPrev), camera B the camera at time k (state eta).
 * Camera-to-world rotations and origins follow from the body pose (Rbc, rCBb from Camera):
 *   R_cA = R_nb(Theta_A) Rbc,  c_A = r_A + R_nb(Theta_A) rCBb,   and the same with B.
 * A static world point P satisfies p_B = R p_A + t with
 *   R = R_cB^T R_cA = Rbc^T R_B^T R_A Rbc,      t = R_cB^T (c_A - c_B).
 * The support is the 12 states of [eta_k ; etaPrev] (indices 6..17 of the SystemVisualNav layout):
 *   column 0..2  r_k (= rBNn at k),   3..5  Theta_k,    6..8  r_{k-1},   9..11  Theta_{k-1}.
 *
 * Correspondences. Each feature i is given as undistorted normalised coordinates x_i = (u, v, 1) in camera A and
 * y_i = (u', v', 1) in camera B (Camera::pixelToNormalised). The flow tracker (ImageFlow) supplies them.
 *
 * Model E (Epipolar, general motion). The Sampson-normalised epipolar residual with E = [t]_x R:
 *   e_i  = y_i^T E x_i,
 *   D_i  = sqrt( (E x_i)_1^2 + (E x_i)_2^2 + (E^T y_i)_1^2 + (E^T y_i)_2^2 ),
 *   r_i  = e_i / D_i       (a distance to the epipolar line, in normalised image units),
 *   r_i^w = r_i / sigma_n, sigma_n = sigma_px / f  (f = mean focal length of the camera).
 * The likelihood is Gaussian in the whitened residuals: log p = -0.5 sum_i (r_i^w)^2 - m log sigma_n - (m/2) log 2 pi.
 *
 * Model R (Rotation, pure-rotation / small-parallax motion). The plane-at-infinity (homography) residual
 *   g_i = pi(R x_i),  pi(q) = (q_1/q_3, q_2/q_3),  r_i = (g_i - y_i) / sigma_n    (2 components per correspondence).
 * This model does not depend on the positions; it is the exact model when the translation is zero.
 *
 * Jacobians (hand-written, validated against central differences in test/src/MeasurementIndoorFlow.cpp):
 *   dR/dTheta_A,m = R_cB^T dR_nb(Theta_A)_m Rbc           dR/dTheta_B,m = (dR_nb(Theta_B)_m Rbc)^T R_cA
 *   dt/dr_A = R_cB^T,  dt/dr_B = -R_cB^T
 *   dt/dTheta_A,m = R_cB^T dR_nb(Theta_A)_m rCBb
 *   dt/dTheta_B,m = (dR_nb(Theta_B)_m Rbc)^T (c_A - c_B) - R_cB^T dR_nb(Theta_B)_m rCBb
 *   dE = [dt]_x R + [t]_x dR,
 *   dr_i/dE = (1/D_i) y_i x_i^T - (e_i / D_i^3) (u' x_i^T + y_i w'^T),  u = E x_i, w = E^T y_i, u' = (u_1,u_2,0), w' = (w_1,w_2,0)
 *   dr_i/dv = sum_ab (dr_i/dE)_ab (dE/dv)_ab,
 *   Model R: dg/dv = D pi(q) (dR/dv) x_i,  D pi(q) = [1/q_3, 0, -q_1/q_3^2; 0, 1/q_3, -q_2/q_3^2], q = R x_i.
 * Gauss-Newton: grad = -J_w^T r^w,  Hess = -J_w^T J_w (J_w = J / sigma_n). The second-order term is omitted; it is zero
 * at zero residual (tested) and vanishes in the noise-free two-view test.
 *
 * Observability and degeneracy (documented, see the Scenario 5/6 reports):
 *  - The epipolar residual is invariant to the scale of t (Sampson normalisation), so the translation MAGNITUDE is not
 *    observable from flow; only the direction of t, and the rotation R, are. Monocular scale comes from the other
 *    measurements (tags with known size in Scenario 5, the depth prior in Scenario 6).
 *  - Near zero translation E -> 0 and the Sampson direction is not determined by the data: the derivative of the
 *    residual with respect to t is ~ 1/|t|, and the direction information is driven by noise, not by parallax. For this
 *    reason the front end selects Model R when the rotation-only misfit explains the matches (small parallax); see
 *    IndoorFlowFrontEnd.h. Model R constrains the rotation and is well conditioned for any |t|.
 *  - Pure rotation makes the epipolar constraint vanish identically (E = 0): rotation then comes from Model R.
 *
 * Gating and robustness are applied outside the likelihood, once per frame, at the predicted state (see
 * IndoorFlowFrontEnd.h): RANSAC for outliers, then a chi2 gate on the prior residuals.
 */

#include <array>
#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "MeasurementBase.h"
#include "SystemEstimator.h"

namespace slam
{

/** @brief Flow model used by the measurement (selected per frame, see the file documentation). */
enum class IndoorFlowModel
{
    Epipolar,   ///< Sampson epipolar residual (1 per correspondence): general motion with parallax
    Rotation    ///< pure-rotation residual pi(R x) - y (2 per correspondence): small parallax
};

/** @brief One correspondence in undistorted normalised coordinates (x = X/Z, y = Y/Z). */
struct IndoorFlowPair
{
    Eigen::Vector2d prevNorm = Eigen::Vector2d::Zero();   ///< feature in camera A (frame k-1)
    Eigen::Vector2d currNorm = Eigen::Vector2d::Zero();   ///< same feature in camera B (frame k)
};

/**
 * @brief Relative pose of camera B (frame k) with respect to camera A (frame k-1) and its derivatives.
 *
 * p_B = R p_A + t. Derivatives are with respect to the 12 support states (column order of the file documentation).
 */
struct IndoorFlowRelativePose
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    std::array<Eigen::Matrix3d, 12> dR{};   ///< dR / d(state 6 + k)
    std::array<Eigen::Vector3d, 12> dt{};   ///< dt / d(state 6 + k)
};

/**
 * @brief Relative pose of the two cameras from the support states (and derivatives when requested).
 * @param x         full state (at least 18 entries: eta_k at 6..11, etaPrev at 12..17)
 * @param Rbc       body-to-camera rotation
 * @param rCBb      camera origin in body coordinates
 * @param withJac   fill dR and dt (otherwise they are zero)
 */
IndoorFlowRelativePose indoorFlowRelativePose(const Eigen::VectorXd& x, const Eigen::Matrix3d& Rbc,
                                              const Eigen::Vector3d& rCBb, bool withJac);

/**
 * @brief Depth-free flow likelihood between etaPrev (camera A) and eta (camera B) (see file documentation).
 *
 * Observation: the vector y is zero of the residual dimension (the residual is not a plain difference, so the stored
 * observation is only a placeholder; MeasurementBase requires one). Support: the 12 states 6..17.
 */
class MeasurementIndoorFlow : public MeasurementBase
{
public:
    static constexpr Eigen::Index kCurrentOffset = 6;    ///< first index of eta_k in the SystemVisualNav layout
    static constexpr Eigen::Index kPreviousOffset = 12;  ///< first index of etaPrev = eta_{k-1}
    static constexpr Eigen::Index kSupportDim = 12;      ///< number of states the likelihood depends on

    /**
     * @param t        measurement time (current frame k, s)
     * @param model    Epipolar or Rotation (see file documentation)
     * @param pairs    at least one correspondence (normalised undistorted coordinates)
     * @param sigmaNorm standard deviation of the residual in normalised units (= sigma_px / f) [-]; must be > 0
     * @param Rbc      body-to-camera rotation
     * @param rCBb     camera origin in body coordinates [m]
     * @throws std::invalid_argument for empty pairs, non-positive sigma or non-finite extrinsics
     */
    MeasurementIndoorFlow(double t, IndoorFlowModel model, std::vector<IndoorFlowPair> pairs, double sigmaNorm,
                          const Eigen::Matrix3d& Rbc, const Eigen::Vector3d& rCBb);

    /** @brief log p(y|x) with the Gauss-Newton gradient and Hessian (full size, zero outside the support). */
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

    /** @brief The 12 states 6..17 (eta_k and etaPrev). */
    std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const override;

    /**
     * @brief Whitened residuals at a full state (size m = N for Epipolar, 2N for Rotation).
     * @param x    full state
     * @param jac  if non-null, receives the whitened Jacobian (m x 12) with respect to states 6..17
     */
    Eigen::VectorXd whitenedResiduals(const Eigen::VectorXd& x, Eigen::MatrixXd* jac) const;

    /** @brief Model in use. */
    IndoorFlowModel model() const { return model_; }
    /** @brief Correspondences. */
    const std::vector<IndoorFlowPair>& pairs() const { return pairs_; }
    /** @brief Residual standard deviation in normalised units. */
    double sigma() const { return sigmaNorm_; }
    /** @brief Body-to-camera rotation. */
    const Eigen::Matrix3d& Rbc() const { return Rbc_; }
    /** @brief Camera origin in body coordinates. */
    const Eigen::Vector3d& rCBb() const { return rCBb_; }
    /** @brief Residual dimension m (N or 2N). */
    Eigen::Index residualDim() const;

private:
    IndoorFlowModel model_;
    std::vector<IndoorFlowPair> pairs_;
    double sigmaNorm_;
    Eigen::Matrix3d Rbc_;
    Eigen::Vector3d rCBb_;
};

} // namespace slam

#endif
