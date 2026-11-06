#ifndef S6SPEEDPRIOR_H
#define S6SPEEDPRIOR_H

/**
 * @file S6SpeedPrior.h
 * @brief ASSUMED metric scale for monocular Scenario 6: a speed prior on the body velocity.
 *
 * Why it is needed. The epipolar flow (MeasurementIndoorFlow, Model E) is invariant to the scale of the translation,
 * and the point measurements are invariant to a common scaling of poses and landmarks. Monocular images therefore cannot
 * fix the metric scale. Without an anchor the posterior translation collapses to the prior mode (zero velocity): this
 * was observed as a camera position that stayed at the origin on every frame (synthetic walking sequence, Scenario 6
 * before this rework), which is the original "stays within +/-5 cm" defect.
 *
 * Model. With the body velocity v = vBNb (state nu[0..2]) and a smoothing constant eps,
 *   h(v) = sqrt(v^T v + eps^2),     y = v0,     r = (h(v) - v0) / sigma,
 *   log p(y | x) = -0.5 r^2 - log(sigma) - 0.5 log(2 pi).
 * This is the statement |v| ~ N(v0, sigma^2): the speed is ASSUMED, the direction is observed through the flow.
 *
 * Derivatives (hand-written, validated in test/src/S6SpeedPrior.cpp):
 *   dh/dv = v / h,
 *   grad  = -(r / sigma) (v / h)   on nu[0..2],
 *   Hess  = -(1 / sigma^2) (v / h)(v / h)^T   (Gauss-Newton; the second-order term r/sigma * d2h/dv2 is omitted and is
 *           zero at zero residual, i.e. when |v| = v0).
 *
 * eps (1e-3 m/s) only removes the non-differentiable point v = 0; the bias is h - |v| <= eps^2 / (2 |v|).
 */

#include <vector>
#include <Eigen/Core>
#include "MeasurementBase.h"
#include "SystemEstimator.h"

namespace slam
{

/** @brief Gaussian speed prior |v_BN^b| ~ N(speed, sigma^2) on the body velocity of SystemVisualNav (nu[0..2]). */
class MeasurementSpeedPrior : public MeasurementBase
{
public:
    /**
     * @param t      measurement time (s)
     * @param speed  assumed speed v0 (m/s), > 0
     * @param sigma  standard deviation of the speed (m/s), > 0
     * @throws std::invalid_argument for non-positive or non-finite arguments
     */
    MeasurementSpeedPrior(double t, double speed, double sigma);

    /** @brief log p(y|x) with the gradient and Gauss-Newton Hessian (see the file documentation). */
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

    /** @brief The three body-velocity states nu[0..2]. */
    std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const override;

    /** @brief Assumed speed v0 (m/s). */
    double speed() const { return speed_; }
    /** @brief Speed standard deviation (m/s). */
    double sigma() const { return sigma_; }

private:
    double speed_;
    double sigma_;
};

} // namespace slam

#endif
