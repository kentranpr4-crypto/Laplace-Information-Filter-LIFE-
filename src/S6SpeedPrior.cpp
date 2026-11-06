#include "S6SpeedPrior.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace slam
{

namespace
{

/// Smoothing of |v| at v = 0 [m/s]: keeps the speed differentiable (see the header).
constexpr double kSpeedSmoothing = 1e-3;

} // namespace

MeasurementSpeedPrior::MeasurementSpeedPrior(double t, double speed, double sigma)
    : MeasurementBase(t, Eigen::VectorXd::Constant(1, speed)), speed_(speed), sigma_(sigma)
{
    if (!(speed_ > 0.0) || !std::isfinite(speed_) || !(sigma_ > 0.0) || !std::isfinite(sigma_))
    {
        throw std::invalid_argument("MeasurementSpeedPrior: speed and sigma must be finite and positive");
    }
}

std::vector<Eigen::Index> MeasurementSpeedPrior::supportIndices(const SystemEstimator& /*system*/) const
{
    return {0, 1, 2};
}

double MeasurementSpeedPrior::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                                            Eigen::MatrixXd* H) const
{
    if (x.size() != system.dim() || x.size() < 3)
    {
        throw std::invalid_argument("MeasurementSpeedPrior::logLikelihood: state too small or size mismatch");
    }
    const Eigen::Vector3d v = x.head<3>();
    const double h = std::sqrt(v.squaredNorm() + kSpeedSmoothing * kSpeedSmoothing);
    const double r = (h - speed_) / sigma_;
    const double value = -0.5 * r * r - std::log(sigma_) - 0.5 * std::log(2.0 * std::numbers::pi);

    const Eigen::Vector3d dh = v / h;   // dh/dv
    if (g != nullptr)
    {
        *g = Eigen::VectorXd::Zero(x.size());
        g->head<3>() = -(r / sigma_) * dh;
    }
    if (H != nullptr)
    {
        *H = Eigen::MatrixXd::Zero(x.size(), x.size());
        H->topLeftCorner<3, 3>() = -(1.0 / (sigma_ * sigma_)) * dh * dh.transpose();
    }
    return value;
}

} // namespace slam
