#include "MeasurementHorizon.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include "rotation.hpp"

namespace
{

/// Derivative of rpy2rot with respect to component m of Theta (0 = phi, 1 = theta, 2 = psi).
Eigen::Matrix3d rpyDerivative(const Eigen::Vector3d& theta, int m)
{
    switch (m)
    {
    case 0: return drpy2rotDphi(theta);
    case 1: return drpy2rotDtheta(theta);
    default: return drpy2rotDpsi(theta);
    }
}

Eigen::VectorXd zerosLike(std::size_t n)
{
    return Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
}

} // namespace

MeasurementHorizon::MeasurementHorizon(double t, std::vector<Eigen::Vector2d> pointsNorm, double sigmaNorm,
                                       const Eigen::Matrix3d& Rbc)
    : MeasurementBase(t, zerosLike(pointsNorm.size())),
      points_(std::move(pointsNorm)),
      sigmaNorm_(sigmaNorm),
      Rbc_(Rbc)
{
    if (points_.size() < 2)
    {
        throw std::invalid_argument("MeasurementHorizon: at least two boundary points are required");
    }
    for (const Eigen::Vector2d& p : points_)
    {
        if (!p.allFinite())
        {
            throw std::invalid_argument("MeasurementHorizon: boundary points must be finite");
        }
    }
    if (!(sigmaNorm_ > 0.0) || !std::isfinite(sigmaNorm_))
    {
        throw std::invalid_argument("MeasurementHorizon: sigma must be finite and positive");
    }
    if (!Rbc_.allFinite())
    {
        throw std::invalid_argument("MeasurementHorizon: Rbc must be finite");
    }
}

Eigen::Vector3d MeasurementHorizon::horizonNormal(const Eigen::Vector3d& theta, const Eigen::Matrix3d& Rbc)
{
    // R_nb^T e3 is the third row of R_nb; a = Rbc^T (R_nb^T e3).
    return Rbc.transpose() * rpy2rot(theta).row(2).transpose();
}

std::vector<Eigen::Index> MeasurementHorizon::supportIndices(const SystemEstimator& /*system*/) const
{
    return {kThetaOffset, kThetaOffset + 1, kThetaOffset + 2};
}

Eigen::VectorXd MeasurementHorizon::predict(const Eigen::VectorXd& x, Eigen::MatrixXd* jac3) const
{
    if (x.size() < kThetaOffset + kSupportDim)
    {
        throw std::invalid_argument("MeasurementHorizon::predict: state too small");
    }
    const Eigen::Vector3d theta = x.segment<3>(kThetaOffset);
    const Eigen::Vector3d a = horizonNormal(theta, Rbc_);
    const double nrm = std::sqrt(a(0) * a(0) + a(1) * a(1));
    if (!(nrm > 1e-6))
    {
        throw std::domain_error("MeasurementHorizon: horizon undefined (camera axis vertical)");
    }
    const Eigen::Index m = static_cast<Eigen::Index>(points_.size());
    Eigen::VectorXd r(m);
    if (jac3 != nullptr)
    {
        jac3->resize(m, kSupportDim);
    }
    Eigen::Matrix<double, 3, kSupportDim> da;   // da / dTheta
    if (jac3 != nullptr)
    {
        for (int j = 0; j < 3; ++j)
        {
            da.col(j) = Rbc_.transpose() * rpyDerivative(theta, j).row(2).transpose();
        }
    }
    for (Eigen::Index i = 0; i < m; ++i)
    {
        const Eigen::Vector3d p(points_[static_cast<std::size_t>(i)](0), points_[static_cast<std::size_t>(i)](1), 1.0);
        const double s = a.dot(p);
        r(i) = s / nrm;
        if (jac3 != nullptr)
        {
            Eigen::Vector3d drda = p / nrm;
            drda(0) -= s / (nrm * nrm * nrm) * a(0);
            drda(1) -= s / (nrm * nrm * nrm) * a(1);
            jac3->row(i) = (drda.transpose() * da);
        }
    }
    return r;
}

double MeasurementHorizon::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& /*system*/,
                                         Eigen::VectorXd* g, Eigen::MatrixXd* H) const
{
    if (x.size() < kThetaOffset + kSupportDim)
    {
        throw std::invalid_argument("MeasurementHorizon::logLikelihood: state too small");
    }
    const bool needJ = (g != nullptr) || (H != nullptr);
    Eigen::MatrixXd jac;
    const Eigen::VectorXd r = predict(x, needJ ? &jac : nullptr);

    const double inv = 1.0 / (sigmaNorm_ * sigmaNorm_);
    const double m = static_cast<double>(points_.size());
    const double value = -0.5 * inv * r.squaredNorm() - m * (std::log(sigmaNorm_) + 0.5 * std::log(2.0 * std::numbers::pi_v<double>));
    if (g != nullptr)
    {
        g->setZero(x.size());
        g->segment<kSupportDim>(kThetaOffset) = -inv * (jac.transpose() * r);
    }
    if (H != nullptr)
    {
        H->setZero(x.size(), x.size());
        H->block<kSupportDim, kSupportDim>(kThetaOffset, kThetaOffset) = -inv * (jac.transpose() * jac);
    }
    return value;
}
