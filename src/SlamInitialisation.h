#ifndef SLAMINITIALISATION_H
#define SLAMINITIALISATION_H

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "Camera.h"
#include "PointFeatures.h"
#include "SlamAssociation.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

namespace slam
{

/**
 * @brief Selects new landmark births among the unmatched features of a frame.
 *
 * A feature is a birth candidate if it is unmatched and at least birthSeparationPx away from the predicted pixel of
 * every visible landmark. Candidates are taken in decreasing response order, keeping birthMinSpacingPx between
 * births, up to maxNew of them.
 * @return indices into features
 */
std::vector<std::size_t> selectBirthFeatures(const std::vector<PointFeature>& features,
                                             const std::vector<bool>& featureUsed, const PredictionSet& predictions,
                                             const SlamConfig& cfg, int maxNew);

/**
 * @brief Initialises a point landmark from one bearing observation and a depth prior (single-view initialisation).
 *
 * Model: the camera ray u_c = pixelToVector(pixel) (unit, camera frame) and the depth rho ~ N(rho0, sigmaRho^2)
 * give the world point L = g(eta, rho) = r_BN + Rnb (rCBb + Rbc rho u_c). Writing the (small) bearing error
 * as two orthonormal directions b1, b2 perpendicular to u_c with standard deviation sigmaPx / f, the noise
 * e = [delta rho; delta a1; delta a2] ~ N(0, D^2), D = diag(sigmaRho, sigmaA, sigmaA) enters linearly as
 *   L = G x + c + Jn e,   G = dg/d eta (3 x n, nonzero on r and Theta),  c = g(mu) - G mu,
 *   Jn = Rnb Rbc [u_c, rho0 b1, rho0 b2],
 * so the new block is added with the correlated augmentation (SystemVisualNav::addLandmarkLinear) with
 * lS = D^{-1} Jn^{-1} (lS^T lS = (Jn D^2 Jn^T)^{-1}). The cross-covariance with the pose is therefore kept.
 *
 * The birth pixel is consumed by the initial bearing: the landmark must not be updated with the same pixel in
 * the same frame (the pipeline enforces this by updating only with previously existing landmarks).
 *
 * @return landmark id, or -1 if the pixel does not define a valid ray
 */
Eigen::Index initialiseLandmark(SystemEstimator& est, SystemVisualNav& model, const Camera& camera,
                                const Eigen::Vector2d& pixel, double sigmaPx, double depthPriorMean,
                                double depthPriorSigma);

} // namespace slam

#endif
