#ifndef MEASUREMENTTAGBUNDLE_H
#define MEASUREMENTTAGBUNDLE_H

/**
 * @file MeasurementTagBundle.h
 * @brief Pose-landmark measurement of identical-ID ArUco tags (Scenario 5): the four corner pixels of every detection.
 *
 * Model. Tag landmark j has state block L_j = [r_jn^n ; Theta_nj] (SystemVisualNav Pose layout), R_nj = rpy2rot(Theta_nj).
 * Its k-th corner in the landmark frame is c_jk (4 x 3 rows, canonical corners rotated by an in-plane Rz(q pi/2), see
 * ArucoTags.h). The world corner is P_jk = r_jn^n + R_nj c_jk, and the pixel is
 *   u_jk = pi( Rbc^T (R_nb^T (P_jk - r_bn^n) - rCBb) ),
 * with pi the distorted pinhole model of Camera. The observation is y_j = [u_j0; ...; u_j3] (8), with independent
 * isotropic noise sigma^2 I2 per corner (the model is equivalent to the point bundle applied to each corner).
 *
 * Derivatives (hand-written, validated against central differences in test/src/MeasurementTagBundle.cpp):
 *   du_k/d eta   = [dU/dr_bn | dU/dTheta_nb] from slam::pointPixelJacobian (at the corner point P_jk),
 *   du_k/d r_j   = dU/dP_jk,
 *   du_k/d Theta_j(m) = dU/dP_jk * (dR_nj/dTheta_m c_jk),   m = phi, theta, psi.
 *
 * Likelihood. Whitened residual per component with w = 1/sigma^2:
 *   log p(y|x) = -0.5 w sum ||u_jk - y_jk||^2 - 8 log sigma - 4 log(2 pi)  (summed over observations),
 *   grad = -w J^T (u - y),   Hessian = -w J^T J  (Gauss-Newton).
 * The exact Hessian adds -w sum_k (u_k - y_k)^T Hess(u_k) which is omitted (documented in the report): it vanishes at a
 * zero residual and is second order in the residual / focal length, the same convention as MeasurementPointBundle.
 */

#include <limits>
#include <vector>
#include <Eigen/Core>
#include "Camera.h"
#include "MeasurementBase.h"
#include "SystemEstimator.h"

namespace slam
{

/** @brief Pixel prediction of the four corners of one tag landmark and its Jacobians. */
struct TagCornerPrediction
{
    bool valid = false;                                   ///< all four corners in front of the camera (depth > minDepth)
    Eigen::Matrix<double, 8, 1> uv =
        Eigen::Matrix<double, 8, 1>::Constant(std::numeric_limits<double>::quiet_NaN());  ///< [u0 v0 ... u3 v3]
    Eigen::Matrix<double, 8, 6> dEta = Eigen::Matrix<double, 8, 6>::Zero();  ///< d uv / d [rBNn; Thetanb]
    Eigen::Matrix<double, 8, 6> dLm = Eigen::Matrix<double, 8, 6>::Zero();   ///< d uv / d [rLNn; ThetaLn]
    Eigen::Matrix<double, 4, 3> worldCorners = Eigen::Matrix<double, 4, 3>::Zero();  ///< P_k (world) [m]
};

/**
 * @brief Predicts the four corner pixels of a tag landmark and their Jacobians.
 * @param camera        calibrated camera (distortion, Rbc, rCBb)
 * @param rBNn          body position in world (eta[0:3])
 * @param thetaNb       body Euler angles (eta[3:6])
 * @param rLNn          tag centre in world (landmark[0:3])
 * @param thetaLn       tag Euler angles (landmark[3:6]), R_nj = rpy2rot(thetaLn)
 * @param cornersLocal  landmark-frame corners c_jk (rows)
 * @param minDepth      corners with camera depth <= minDepth make the prediction invalid
 */
TagCornerPrediction predictTagCorners(const Camera& camera, const Eigen::Vector3d& rBNn,
                                      const Eigen::Vector3d& thetaNb, const Eigen::Vector3d& rLNn,
                                      const Eigen::Vector3d& thetaLn, const Eigen::Matrix<double, 4, 3>& cornersLocal,
                                      double minDepth = 0.05);

} // namespace slam

/** @brief One detected tag registered against a landmark block of the estimator. */
struct TagObservation
{
    Eigen::Index landmarkId = -1;                              ///< stable id returned by SystemVisualNav::addLandmarkLinear
    Eigen::Index landmarkOffset = 0;                           ///< first index of the [r; Theta] block in the state
    Eigen::Matrix<double, 4, 3> cornersLocal = Eigen::Matrix<double, 4, 3>::Zero();  ///< landmark-frame corners
    Eigen::Matrix<double, 8, 1> pixels = Eigen::Matrix<double, 8, 1>::Zero();        ///< measured corners [u0 v0 ... u3 v3]
    double sigmaPx = 2.0;                                      ///< isotropic pixel std per corner component [px]
};

/**
 * @brief Laplace/MAP measurement of a set of tag detections (four corners each), see the file documentation.
 *
 * Observation y = stack of the 8-vectors of all observations. Support: the body pose eta (6) and the 6 states of every
 * observed tag landmark. The class follows MeasurementBase::process for the MAP and the information-form update.
 */
class MeasurementTagBundle : public MeasurementBase
{
public:
    /**
     * @param t          measurement time (s)
     * @param camera     camera model (copied)
     * @param obs        at least one observation
     * @param poseOffset index of eta = [rBNn; Thetanb] in the state (6 for SystemVisualNav)
     * @throws std::invalid_argument for an empty set, a non-positive sigma or a non-finite pixel
     */
    MeasurementTagBundle(double t, const Camera& camera, std::vector<TagObservation> obs,
                         Eigen::Index poseOffset = 6);

    /** @brief log p(y|x) with gradient and Gauss-Newton Hessian; NaN if a predicted corner is behind the camera. */
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

    /** @brief Support: eta (6) followed by the 6 states of every observed tag landmark. */
    std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const override;

    /** @brief Observations in measurement order. */
    const std::vector<TagObservation>& observations() const { return obs_; }

    /** @brief Camera used by the measurement. */
    const Camera& camera() const { return camera_; }

    /** @brief Predicted corner vector of every observation (8m) at a full state; NaN where invalid. */
    Eigen::VectorXd predictedCorners(const Eigen::VectorXd& x) const;

    /** @brief Measurement noise factor R = blockdiag(sigma^2 I8) as a diagonal (8m). */
    Eigen::VectorXd pixelVariances() const;

private:
    Camera camera_;
    std::vector<TagObservation> obs_;
    Eigen::Index poseOffset_;
};

#endif
