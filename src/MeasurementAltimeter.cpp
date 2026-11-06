#include "MeasurementAltimeter.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

MeasurementAltimeter::MeasurementAltimeter(double t, double altitudeRelativeM, double sigmaM)
    : MeasurementBase(t, Eigen::VectorXd::Constant(1, altitudeRelativeM)), sigmaM_(sigmaM)
{
    if (!(sigmaM_ > 0.0) || !std::isfinite(sigmaM_))
    {
        throw std::invalid_argument("MeasurementAltimeter: sigma must be finite and positive");
    }
}

double MeasurementAltimeter::predict(const Eigen::VectorXd& x)
{
    if (x.size() <= kZIndex)
    {
        throw std::invalid_argument("MeasurementAltimeter::predict: state too small");
    }
    return -x(kZIndex);
}

std::vector<Eigen::Index> MeasurementAltimeter::supportIndices(const SystemEstimator& /*system*/) const
{
    return {kZIndex};
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& /*system*/,
                                           Eigen::VectorXd* g, Eigen::MatrixXd* H) const
{
    const double innovation = predict(x) - y()(0);
    const double inv = 1.0 / (sigmaM_ * sigmaM_);
    const double value = -0.5 * inv * innovation * innovation - std::log(sigmaM_) - 0.5 * std::log(2.0 * std::numbers::pi_v<double>);
    if (g != nullptr)
    {
        g->setZero(x.size());
        // d h / d x_z = -1, so d/dx_z [-(h - y)^2 / (2 s^2)] = (h - y) / s^2.
        (*g)(kZIndex) = innovation * inv;
    }
    if (H != nullptr)
    {
        H->setZero(x.size(), x.size());
        (*H)(kZIndex, kZIndex) = -inv;
    }
    return value;
}
