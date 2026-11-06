#include "MeasurementOutdoorFlowBundle.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include "rotation.hpp"

namespace
{

Eigen::VectorXd stackCurrent(const std::vector<OutdoorFlowPair>& pairs)
{
    Eigen::VectorXd y(2 * static_cast<Eigen::Index>(pairs.size()));
    for (std::size_t i = 0; i < pairs.size(); ++i)
    {
        y.segment<2>(2 * static_cast<Eigen::Index>(i)) = pairs[i].currNorm;
    }
    return y;
}

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

} // namespace

MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(double t, std::vector<OutdoorFlowPair> pairs,
                                                           double sigmaNorm, double groundDepthM,
                                                           const Eigen::Matrix3d& Rbc)
    : MeasurementBase(t, stackCurrent(pairs)),
      pairs_(std::move(pairs)),
      sigmaNorm_(sigmaNorm),
      groundDepthM_(groundDepthM),
      Rbc_(Rbc)
{
    if (pairs_.empty())
    {
        throw std::invalid_argument("MeasurementOutdoorFlowBundle: at least one correspondence is required");
    }
    if (!(sigmaNorm_ > 0.0) || !std::isfinite(sigmaNorm_))
    {
        throw std::invalid_argument("MeasurementOutdoorFlowBundle: sigma must be finite and positive");
    }
    if (!(groundDepthM_ > 0.0) || !std::isfinite(groundDepthM_))
    {
        throw std::invalid_argument("MeasurementOutdoorFlowBundle: ground depth must be finite and positive");
    }
    if (!Rbc_.allFinite())
    {
        throw std::invalid_argument("MeasurementOutdoorFlowBundle: Rbc must be finite");
    }
}

std::vector<Eigen::Index> MeasurementOutdoorFlowBundle::supportIndices(const SystemEstimator& /*system*/) const
{
    std::vector<Eigen::Index> idx;
    idx.reserve(static_cast<std::size_t>(kSupportDim));
    for (Eigen::Index i = 0; i < kSupportDim; ++i)
    {
        idx.push_back(kCurrentOffset + i);
    }
    return idx;
}

Eigen::Matrix3d MeasurementOutdoorFlowBundle::homography(const Eigen::VectorXd& x, double groundDepthM,
                                                         const Eigen::Matrix3d& Rbc,
                                                         std::array<Eigen::Matrix3d, kSupportDim>* dH)
{
    if (x.size() < kPreviousOffset + 6)
    {
        throw std::invalid_argument("MeasurementOutdoorFlowBundle::homography: state must contain eta and etaPrev");
    }
    const Eigen::Vector3d rk = x.segment<3>(kCurrentOffset);
    const Eigen::Vector3d thk = x.segment<3>(kCurrentOffset + 3);
    const Eigen::Vector3d rp = x.segment<3>(kPreviousOffset);
    const Eigen::Vector3d thp = x.segment<3>(kPreviousOffset + 3);

    const Eigen::Matrix3d rk_ = rpy2rot(thk);
    const Eigen::Matrix3d rp_ = rpy2rot(thp);
    const Eigen::Matrix3d RbcT = Rbc.transpose();

    const Eigen::Matrix3d M = RbcT * rk_.transpose();          // world -> camera k, before the lever arm
    const Eigen::Matrix3d RBA = M * rp_ * Rbc;                 // camera rotation from A to B
    const Eigen::Vector3d t = M * (rp - rk);                   // translation from A to B (camera coordinates of B)
    const Eigen::Vector3d nA = RbcT * rp_.row(2).transpose();  // plane normal in camera A (R_np^T e3 = row 2 of R_np)
    const double dA = groundDepthM - rp(2);                    // plane distance from camera A (AGL)
    if (!(dA > 0.0))
    {
        throw std::domain_error("MeasurementOutdoorFlowBundle: camera is not above the ground plane");
    }

    const Eigen::Matrix3d Hm = RBA + t * nA.transpose() / dA;

    if (dH != nullptr)
    {
        // Support order: [rk (0..2), Thetak (3..5), rp (6..8), Thetap (9..11)].
        for (int m = 0; m < 3; ++m)
        {
            // d t / d rk_m = -M e_m ; R_BA and n_A and d_A do not depend on rk.
            (*dH)[static_cast<std::size_t>(m)] = (-M.col(m)) * nA.transpose() / dA;
        }
        for (int m = 0; m < 3; ++m)
        {
            // R_BA = M rp_ Rbc with M = Rbc^T rk_^T, so dM = Rbc^T (dRk)^T ; t = M (rp - rk) ; n_A, d_A fixed.
            const Eigen::Matrix3d dRk = rpyDerivative(thk, m);
            const Eigen::Matrix3d dM = RbcT * dRk.transpose();
            const Eigen::Matrix3d dRBA = dM * rp_ * Rbc;
            const Eigen::Vector3d dt = dM * (rp - rk);
            (*dH)[static_cast<std::size_t>(3 + m)] = dRBA + dt * nA.transpose() / dA;
        }
        for (int m = 0; m < 3; ++m)
        {
            // d t / d rp_m = M e_m ; d(1/d_A)/d rp_m = delta_{m2} / d_A^2 (d_A = Hg - rp_z).
            const Eigen::Vector3d dt = M.col(m);
            const double dInv = (m == 2) ? 1.0 / (dA * dA) : 0.0;
            (*dH)[static_cast<std::size_t>(6 + m)] = dt * nA.transpose() / dA + dInv * t * nA.transpose();
        }
        for (int m = 0; m < 3; ++m)
        {
            // R_BA = M rp_ Rbc ; n_A = Rbc^T rp_^T e3 ; t and d_A do not depend on thetap.
            const Eigen::Matrix3d dRp = rpyDerivative(thp, m);
            const Eigen::Matrix3d dRBA = M * dRp * Rbc;
            const Eigen::Vector3d dnA = RbcT * dRp.row(2).transpose();
            (*dH)[static_cast<std::size_t>(9 + m)] = dRBA + t * dnA.transpose() / dA;
        }
    }
    return Hm;
}

Eigen::VectorXd MeasurementOutdoorFlowBundle::predict(const Eigen::VectorXd& x, Eigen::MatrixXd* jac12) const
{
    std::array<Eigen::Matrix3d, kSupportDim> dH;
    const Eigen::Matrix3d Hm = homography(x, groundDepthM_, Rbc_, jac12 != nullptr ? &dH : nullptr);
    const Eigen::Index m = 2 * static_cast<Eigen::Index>(pairs_.size());
    Eigen::VectorXd g(m);
    if (jac12 != nullptr)
    {
        jac12->resize(m, kSupportDim);
    }
    for (std::size_t i = 0; i < pairs_.size(); ++i)
    {
        const Eigen::Vector3d u(pairs_[i].prevNorm(0), pairs_[i].prevNorm(1), 1.0);
        const Eigen::Vector3d q = Hm * u;
        if (!(q(2) > 0.0))
        {
            throw std::domain_error("MeasurementOutdoorFlowBundle: predicted feature lies behind the camera");
        }
        const Eigen::Index r = 2 * static_cast<Eigen::Index>(i);
        g.segment<2>(r) = q.head<2>() / q(2);
        if (jac12 != nullptr)
        {
            Eigen::Matrix<double, 2, 3> dpi;
            dpi << 1.0 / q(2), 0.0, -q(0) / (q(2) * q(2)),
                   0.0, 1.0 / q(2), -q(1) / (q(2) * q(2));
            for (Eigen::Index j = 0; j < kSupportDim; ++j)
            {
                jac12->block<2, 1>(r, j) = dpi * (dH[static_cast<std::size_t>(j)] * u);
            }
        }
    }
    return g;
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& /*system*/,
                                                   Eigen::VectorXd* g, Eigen::MatrixXd* H) const
{
    const Eigen::Index n = x.size();
    if (n < kPreviousOffset + 6)
    {
        throw std::invalid_argument("MeasurementOutdoorFlowBundle::logLikelihood: state too small");
    }
    const bool needJ = (g != nullptr) || (H != nullptr);
    std::array<Eigen::Matrix3d, kSupportDim> dH;
    const Eigen::Matrix3d Hm = homography(x, groundDepthM_, Rbc_, needJ ? &dH : nullptr);

    const double inv = 1.0 / (sigmaNorm_ * sigmaNorm_);
    double ss = 0.0;
    Eigen::Matrix<double, kSupportDim, 1> gl = Eigen::Matrix<double, kSupportDim, 1>::Zero();
    Eigen::Matrix<double, kSupportDim, kSupportDim> hl = Eigen::Matrix<double, kSupportDim, kSupportDim>::Zero();

    for (std::size_t i = 0; i < pairs_.size(); ++i)
    {
        const Eigen::Vector3d u(pairs_[i].prevNorm(0), pairs_[i].prevNorm(1), 1.0);
        const Eigen::Vector3d q = Hm * u;
        if (!(q(2) > 0.0))
        {
            throw std::domain_error("MeasurementOutdoorFlowBundle: predicted feature lies behind the camera");
        }
        const Eigen::Vector2d e = q.head<2>() / q(2) - pairs_[i].currNorm;
        ss += e.squaredNorm();
        if (needJ)
        {
            Eigen::Matrix<double, 2, 3> dpi;
            dpi << 1.0 / q(2), 0.0, -q(0) / (q(2) * q(2)),
                   0.0, 1.0 / q(2), -q(1) / (q(2) * q(2));
            Eigen::Matrix<double, 2, kSupportDim> jac;
            for (Eigen::Index j = 0; j < kSupportDim; ++j)
            {
                jac.col(j) = dpi * (dH[static_cast<std::size_t>(j)] * u);
            }
            gl += jac.transpose() * e;
            hl += jac.transpose() * jac;
        }
    }

    const double m = 2.0 * static_cast<double>(pairs_.size());
    const double value = -0.5 * inv * ss - m * (std::log(sigmaNorm_) + 0.5 * std::log(2.0 * std::numbers::pi_v<double>));
    if (g != nullptr)
    {
        g->setZero(n);
        g->segment<kSupportDim>(kCurrentOffset) = -inv * gl;
    }
    if (H != nullptr)
    {
        H->setZero(n, n);
        H->block<kSupportDim, kSupportDim>(kCurrentOffset, kCurrentOffset) = -inv * hl;
    }
    return value;
}
