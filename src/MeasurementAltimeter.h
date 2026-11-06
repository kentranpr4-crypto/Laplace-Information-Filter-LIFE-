#ifndef MEASUREMENTALTIMETER_H
#define MEASUREMENTALTIMETER_H

#include <vector>
#include <Eigen/Core>
#include "MeasurementBase.h"

/**
 * @brief Relative altimeter measurement (event-driven: one update per change of the SRT altitude value).
 *
 * World z is down and the origin is the initial body position, so the height above the reference is
 * h = -z (relative to the start). The SRT altitude is only used as a relative reference:
 *
 *   y = h_SRT(t) - h_SRT(t0) = -(z(t) - z(t0)) + v,   with z(t0) = 0 by construction,   v ~ N(0, sigma^2).
 *
 * The model is linear: h(x) = -x_z, with x_z = x(8) (eta_k rBNn z component; SystemVisualNav layout).
 * Gradient and Hessian are exact:  g_8 = (h - y)/sigma^2,  H_88 = -1/sigma^2.
 *
 * Only the altitude field of the SRT is used. Latitude and longitude never enter this class.
 */
class MeasurementAltimeter : public MeasurementBase
{
public:
    static constexpr Eigen::Index kZIndex = 8;   ///< index of rBNn_z (eta_k block, offset 6 + 2)

    /**
     * @param t          measurement time (s)
     * @param altitudeRelativeM  y = altitude(t) - altitude(t0) [m]
     * @param sigmaM     standard deviation [m] (> 0)
     */
    MeasurementAltimeter(double t, double altitudeRelativeM, double sigmaM);

    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

    /** @brief Support: the z component of the current position. */
    std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const override;

    /** @brief Predicted relative height h(x) = -x_z. */
    static double predict(const Eigen::VectorXd& x);

    /** @brief Noise standard deviation [m]. */
    double sigma() const { return sigmaM_; }

private:
    double sigmaM_;
};

#endif
