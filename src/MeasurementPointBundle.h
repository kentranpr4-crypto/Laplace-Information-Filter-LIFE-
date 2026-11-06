#ifndef MEASUREMENTPOINTBUNDLE_H
#define MEASUREMENTPOINTBUNDLE_H

#include <limits>
#include <vector>
#include <Eigen/Core>
#include "Camera.h"
#include "MeasurementBase.h"
#include "SystemEstimator.h"

namespace slam
{

/**
 * @brief Pixel-noise model of the indoor point measurements (INFLATED, see Scenario 6 documentation).
 *
 * sigma(u) = sigma0 + sigmaEdge * rho(u),  rho(u) = ||u - c|| / ||(W/2, H/2)||  in [0, 1],
 * i.e. the pixel standard deviation grows from sigma0 at the principal point to sigma0 + sigmaEdge at the
 * image corners. The defaults (1.0 px, +0.5 px) were tuned on the NIS of room.MOV: they are close to the
 * calibration RMS (1.12 px) and not a large inflation, because the NIS with 2.5 px was ~0.2 (over-conservative).
 * Weak indoor intrinsics (fx changes ~5 % across calibration subsets) are handled by the chi2 gate and the
 * descriptor check; the NIS tail (p95 ~1.2) is the evidence for the choice.
 */
struct PixelNoiseModel
{
    double sigma0 = 2.5;      ///< standard deviation at the principal point [px]
    double sigmaEdge = 2.0;   ///< additional standard deviation at the image corners [px]

    /** @brief Pixel standard deviation at pixel u for the given camera. */
    double sigma(const Camera& camera, const Eigen::Vector2d& u) const;
};

/**
 * @brief Pixel projection of a point landmark and its exact first derivatives.
 *
 * Model: p_c = Rbc^T (Rnb^T (r_L^n - r_B^n) - rCBb),  u = pi(p_c) (distorted pinhole, Camera).
 * With the state layout of SystemVisualNav: eta = [r_B^n ; Theta_nb] (6 states), landmark r_L^n (3 states).
 *   dU/deta  = [J_pc dPc/dr_bn , J_pc dPc/dTheta]   (2 x 6)
 *   dU/dL    = J_pc Rnc^T                           (2 x 3)
 * where J_pc = d u / d p_c from Camera::vectorToPixelWithJacobian, and dPc/d(.) from rigid::cameraPointJacobian.
 */
struct PointPixelJacobian
{
    bool valid = false;                              ///< false if the point is not in front of the camera (p_c.z <= minDepth)
    Eigen::Vector2d uv = Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d pc = Eigen::Vector3d::Zero();    ///< camera-frame point
    Eigen::Matrix<double, 2, 6> dUvdEta = Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Matrix<double, 2, 3> dUvdLm = Eigen::Matrix<double, 2, 3>::Zero();
};

/**
 * @brief Projects a world point with its Jacobians.
 * @param camera   calibrated camera (distortion and extrinsics Rbc, rCBb)
 * @param rBNn     body position in world
 * @param thetaNb  ZYX Euler angles of the body
 * @param rLNn     landmark position in world
 * @param minDepth points with p_c.z <= minDepth are reported invalid (uv and Jacobians are NaN/zero)
 */
PointPixelJacobian pointPixelJacobian(const Camera& camera, const Eigen::Vector3d& rBNn, const Eigen::Vector3d& thetaNb,
                                      const Eigen::Vector3d& rLNn, double minDepth = 0.05);

} // namespace slam

/**
 * @brief Laplace/MAP measurement of a bundle of point landmarks observed as pixels.
 *
 * Observation: y = [u_1 ; ... ; u_N] (2N), u_j the measured pixel of landmark j.
 * Likelihood: u_j ~ N(pi(p_c,j(x)), sigma_j^2 I2) independently, so with the whitened residual
 *   r_j = (pi(p_c,j(x)) - y_j) / sigma_j:
 *   log p(y|x) = -0.5 sum_j ||r_j||^2 - sum_j 2 log sigma_j - N log(2 pi)
 *   grad      = -sum_j (1/sigma_j^2) (u_j - y_j)^T dU_j/dx
 *   Hessian   = -sum_j (1/sigma_j^2) (dU_j/dx)^T dU_j/dx          (Gauss-Newton; see below)
 * The exact Hessian adds -sum_j (1/sigma_j) r_j Hess(u_j) (second derivatives of the distorted projection). That
 * term is omitted: it vanishes at zero residual and is O(residual / focal length) relative to the Gauss-Newton
 * term at the MAP. The posterior precision therefore uses the Gauss-Newton Hessian (positive semi-definite,
 * as in bundle adjustment). The supported state is the body pose eta (6) and the landmark blocks (3 each).
 */
class MeasurementPointBundle : public MeasurementBase
{
public:
    /** @brief One pixel observation of a point landmark registered in the estimator. */
    struct Observation
    {
        Eigen::Index landmarkId = -1;      ///< stable id returned by SystemVisualNav::addLandmark
        Eigen::Index landmarkOffset = 0;   ///< first index of the landmark block in the state
        Eigen::Vector2d pixel = Eigen::Vector2d::Zero();  ///< measured pixel
        double sigmaPx = 1.0;              ///< pixel standard deviation (already inflated)
    };

    /**
     * @param t          measurement time (s)
     * @param camera     camera model (copied)
     * @param obs        at least one observation
     * @param poseOffset index of eta = [rBNn; Thetanb] in the state (6 for SystemVisualNav)
     * @throws std::invalid_argument for an empty set or non-positive sigma
     */
    MeasurementPointBundle(double t, const Camera& camera, std::vector<Observation> obs,
                           Eigen::Index poseOffset = 6);

    /** @brief log p(y|x) with the gradient and Gauss-Newton Hessian (see class documentation). */
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

    /** @brief Support: eta (6) followed by the 3 states of every observed landmark. */
    std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const override;

    /** @brief Observations in measurement order. */
    const std::vector<Observation>& observations() const { return obs_; }

    /**
     * @brief Predicted pixels stacked as the observation (2N) at a full state vector.
     * Entries of invalid (behind-camera) points are NaN.
     */
    Eigen::VectorXd predictedPixels(const Eigen::VectorXd& x) const;

    /** @brief Camera used by the measurement. */
    const Camera& camera() const { return camera_; }

    /** @brief Pixel noise standard deviations (2N entries, one per pixel component) for the NIS diagnostics. */
    Eigen::VectorXd pixelSigmas() const;

private:
    Camera camera_;
    std::vector<Observation> obs_;
    Eigen::Index poseOffset_;
};

#endif
