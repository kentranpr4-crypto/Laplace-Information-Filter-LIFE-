#ifndef MEASUREMENTOUTDOORFLOWBUNDLE_H
#define MEASUREMENTOUTDOORFLOWBUNDLE_H

#include <array>
#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "MeasurementBase.h"

/**
 * @brief One feature correspondence between two consecutive frames, in undistorted normalised coordinates
 *        (x = X/Z, y = Y/Z; obtained with Camera::pixelToNormalised).
 */
struct OutdoorFlowPair
{
    Eigen::Vector2d prevNorm = Eigen::Vector2d::Zero();  ///< feature in the previous camera frame A (k-1)
    Eigen::Vector2d currNorm = Eigen::Vector2d::Zero();  ///< same feature in the current camera frame B (k)
};

/**
 * @brief Planar-scene optical-flow bundle: likelihood of matched normalised points between frames k-1 and k.
 *
 * State used (SystemVisualNav layout, support = 12 states):
 *   eta_k  = [rBNn(k) (6..8), Thetanb(k) (9..11)],  etaPrev = eta_{k-1} = [rBNn(k-1) (12..14), Thetanb(k-1) (15..17)].
 *
 * Model. The ground is the plane e3^T p_n = Hg (world z down, Hg = AGL of the initial body position, ASSUMED).
 * For a point on that plane seen from frame A (k-1) and frame B (k), with camera rotations R_nc = R_nb Rbc
 * and camera origins r_c = r_b (lever arm rCBb = 0, verified in camera.xml),
 *
 *   R_BA = Rbc^T R_k^T R_p Rbc,   t_BA = Rbc^T R_k^T (r_p - r_k),
 *   n_A  = Rbc^T R_p^T e3,        d_A  = Hg - z_p,
 *   H    = R_BA + t_BA n_A^T / d_A      (Euclidean homography, p_B proportional to H p_A)
 *
 * and the predicted normalised current observation of feature i is g_i = pi(H u_i), u_i = (x_i, y_i, 1)^T,
 * pi(q) = (q_1/q_3, q_2/q_3). Observations y_i = currNorm_i with white Gaussian noise sigma (normalised units,
 * i.e. sigma_px / f), so
 *
 *   log p(y|x) = -(1/(2 sigma^2)) sum_i ||g_i - y_i||^2 - m (log sigma + (1/2) log 2 pi),   m = 2N,
 *   grad_j     = -(1/sigma^2) sum_i J_i^T (g_i - y_i),
 *   Hess       = -(1/sigma^2) sum_i J_i^T J_i        (Gauss-Newton; see the note below),
 *
 * with J_i = D pi(q_i) (dH/dtheta_j) u_i, D pi(q) = [1/q3, 0, -q1/q3^2; 0, 1/q3, -q2/q3^2].
 * dH/dtheta_j is hand-written for the 12 support parameters (see the .cpp): the translation, the rotation
 * and the plane distance terms are differentiated separately. Validated against central differences in
 * test/src/MeasurementOutdoorFlowBundleTests.cpp.
 *
 * Note: the second-order term sum_i r_i Hess(g_i) is omitted (Gauss-Newton). It vanishes at zero residual, and
 * the tests check the FD Hessian against the GN part at zero-residual states and report the size of the gap
 * at noisy states.
 *
 * Observability (planar flow, with the altimeter fixing the plane scale): relative rotation (all 3 axes),
 * translation direction and magnitude scaled by AGL (|t|/d_A), roll and pitch via the plane normal. Absolute
 * yaw is not observable from flow alone (a constant yaw rotation of the whole world leaves every H unchanged), so
 * heading is carried by the prior.
 */
class MeasurementOutdoorFlowBundle : public MeasurementBase
{
public:
    static constexpr Eigen::Index kCurrentOffset = 6;   ///< first index of eta_k in the SystemVisualNav layout
    static constexpr Eigen::Index kPreviousOffset = 12; ///< first index of etaPrev = eta_{k-1}
    static constexpr Eigen::Index kSupportDim = 12;     ///< number of states the likelihood depends on

    /**
     * @param t           measurement time (current frame, s)
     * @param pairs       correspondences in normalised coordinates (N >= 1)
     * @param sigmaNorm   standard deviation of each normalised coordinate [-]; must be > 0
     * @param groundDepthM Hg, height of the ground plane in the world z-down frame [m] (plane z = Hg)
     * @param Rbc         body-to-camera rotation
     * @throws std::invalid_argument for invalid inputs
     */
    MeasurementOutdoorFlowBundle(double t, std::vector<OutdoorFlowPair> pairs, double sigmaNorm, double groundDepthM,
                                 const Eigen::Matrix3d& Rbc);

    /** @brief log p(y|x) with GN gradient and Hessian (see class documentation). Full-size outputs. */
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

    /** @brief The 12 states of eta_k and etaPrev (indices 6..17). */
    std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const override;

    /**
     * @brief Stacked prediction g(x) (2N) and optional Jacobian with respect to the 12 support states (2N x 12).
     * @throws std::domain_error if a predicted point lies behind the camera
     */
    Eigen::VectorXd predict(const Eigen::VectorXd& x, Eigen::MatrixXd* jac12) const;

    /**
     * @brief Plane homography H(x) from frame A (k-1) to frame B (k) and optionally dH/dtheta_j, j = 0..11,
     * for the support order [rk (3); Thetak (3); rp (3); Thetap (3)].
     */
    static Eigen::Matrix3d homography(const Eigen::VectorXd& x, double groundDepthM, const Eigen::Matrix3d& Rbc,
                                      std::array<Eigen::Matrix3d, kSupportDim>* dH);

    /** @brief Number of correspondences N. */
    std::size_t size() const { return pairs_.size(); }
    /** @brief Correspondences. */
    const std::vector<OutdoorFlowPair>& pairs() const { return pairs_; }
    /** @brief Ground plane height Hg [m]. */
    double groundDepth() const { return groundDepthM_; }
    /** @brief Noise standard deviation in normalised units. */
    double sigma() const { return sigmaNorm_; }

private:
    std::vector<OutdoorFlowPair> pairs_;
    double sigmaNorm_;
    double groundDepthM_;
    Eigen::Matrix3d Rbc_;
};

#endif
