#include "MeasurementIndoorFlow.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include "rotation.hpp"

namespace slam
{

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

/// Residuals below this denominator are treated as "no information" (E = 0 exactly, or a degenerate pair).
constexpr double kDegenerateD = 1e-12;

/// Number of residual rows per correspondence.
Eigen::Index rowsPerPair(IndoorFlowModel model)
{
    return model == IndoorFlowModel::Epipolar ? 1 : 2;
}

Eigen::VectorXd placeholderObservation(IndoorFlowModel model, std::size_t n)
{
    return Eigen::VectorXd::Zero(rowsPerPair(model) * static_cast<Eigen::Index>(n));
}

} // namespace

IndoorFlowRelativePose indoorFlowRelativePose(const Eigen::VectorXd& x, const Eigen::Matrix3d& Rbc,
                                              const Eigen::Vector3d& rCBb, bool withJac)
{
    if (x.size() < MeasurementIndoorFlow::kPreviousOffset + 6)
    {
        throw std::invalid_argument("indoorFlowRelativePose: state must contain eta_k and etaPrev");
    }
    const Eigen::Vector3d rB = x.segment<3>(MeasurementIndoorFlow::kCurrentOffset);
    const Eigen::Vector3d thB = x.segment<3>(MeasurementIndoorFlow::kCurrentOffset + 3);
    const Eigen::Vector3d rA = x.segment<3>(MeasurementIndoorFlow::kPreviousOffset);
    const Eigen::Vector3d thA = x.segment<3>(MeasurementIndoorFlow::kPreviousOffset + 3);

    const Eigen::Matrix3d RnbA = rpy2rot(thA);
    const Eigen::Matrix3d RnbB = rpy2rot(thB);
    const Eigen::Matrix3d RcA = RnbA * Rbc;     // camera A -> world
    const Eigen::Matrix3d RcB = RnbB * Rbc;     // camera B -> world
    const Eigen::Vector3d cA = rA + RnbA * rCBb;
    const Eigen::Vector3d cB = rB + RnbB * rCBb;

    IndoorFlowRelativePose out;
    out.R = RcB.transpose() * RcA;
    out.t = RcB.transpose() * (cA - cB);
    // Every derivative entry is set explicitly: Eigen matrices default-construct uninitialised (NaN in Debug builds).
    for (std::size_t k = 0; k < 12; ++k)
    {
        out.dR[k].setZero();
        out.dt[k].setZero();
    }
    if (!withJac)
    {
        return out;
    }
    const Eigen::Matrix3d RcBT = RcB.transpose();
    // Positions: dR = 0 (already zero), dt/dr_A = R_cB^T, dt/dr_B = -R_cB^T (support columns 6..8 and 0..2).
    for (int m = 0; m < 3; ++m)
    {
        out.dt[static_cast<std::size_t>(6 + m)] = RcBT.col(m);
        out.dt[static_cast<std::size_t>(m)] = -RcBT.col(m);
    }
    // Orientation of B (support columns 3..5): R = (dR_nb(B) Rbc)^T R_cA, t = R_cB^T (c_A - c_B) - ...
    for (int m = 0; m < 3; ++m)
    {
        const Eigen::Matrix3d dRB = rpyDerivative(thB, m);
        const Eigen::Matrix3d dRcBT = (dRB * Rbc).transpose();
        const std::size_t col = static_cast<std::size_t>(3 + m);
        out.dR[col] = dRcBT * RcA;
        out.dt[col] = dRcBT * (cA - cB) - RcBT * (dRB * rCBb);
    }
    // Orientation of A (support columns 9..11): R = R_cB^T (dR_nb(A) Rbc), t = R_cB^T dR_nb(A) rCBb.
    for (int m = 0; m < 3; ++m)
    {
        const Eigen::Matrix3d dRA = rpyDerivative(thA, m);
        const std::size_t col = static_cast<std::size_t>(9 + m);
        out.dR[col] = RcBT * (dRA * Rbc);
        out.dt[col] = RcBT * (dRA * rCBb);
    }
    return out;
}

MeasurementIndoorFlow::MeasurementIndoorFlow(double t, IndoorFlowModel model, std::vector<IndoorFlowPair> pairs,
                                             double sigmaNorm, const Eigen::Matrix3d& Rbc,
                                             const Eigen::Vector3d& rCBb)
    : MeasurementBase(t, placeholderObservation(model, pairs.size())),
      model_(model),
      pairs_(std::move(pairs)),
      sigmaNorm_(sigmaNorm),
      Rbc_(Rbc),
      rCBb_(rCBb)
{
    if (pairs_.empty())
    {
        throw std::invalid_argument("MeasurementIndoorFlow: at least one correspondence is required");
    }
    if (!(sigmaNorm_ > 0.0) || !std::isfinite(sigmaNorm_))
    {
        throw std::invalid_argument("MeasurementIndoorFlow: sigma must be finite and positive");
    }
    if (!Rbc_.allFinite() || !rCBb_.allFinite())
    {
        throw std::invalid_argument("MeasurementIndoorFlow: extrinsics must be finite");
    }
    for (const IndoorFlowPair& p : pairs_)
    {
        if (!p.prevNorm.allFinite() || !p.currNorm.allFinite())
        {
            throw std::invalid_argument("MeasurementIndoorFlow: correspondences must be finite");
        }
    }
}

Eigen::Index MeasurementIndoorFlow::residualDim() const
{
    return rowsPerPair(model_) * static_cast<Eigen::Index>(pairs_.size());
}

std::vector<Eigen::Index> MeasurementIndoorFlow::supportIndices(const SystemEstimator& /*system*/) const
{
    std::vector<Eigen::Index> idx;
    idx.reserve(static_cast<std::size_t>(kSupportDim));
    for (Eigen::Index i = 0; i < kSupportDim; ++i)
    {
        idx.push_back(kCurrentOffset + i);
    }
    return idx;
}

Eigen::VectorXd MeasurementIndoorFlow::whitenedResiduals(const Eigen::VectorXd& x, Eigen::MatrixXd* jac) const
{
    const IndoorFlowRelativePose rp = indoorFlowRelativePose(x, Rbc_, rCBb_, jac != nullptr);
    const Eigen::Matrix3d& R = rp.R;
    const Eigen::Vector3d& t = rp.t;
    const Eigen::Index m = residualDim();
    const Eigen::Index N = static_cast<Eigen::Index>(pairs_.size());
    Eigen::VectorXd r = Eigen::VectorXd::Zero(m);
    if (jac != nullptr)
    {
        *jac = Eigen::MatrixXd::Zero(m, kSupportDim);
    }
    const double invSigma = 1.0 / sigmaNorm_;

    if (model_ == IndoorFlowModel::Rotation)
    {
        for (Eigen::Index i = 0; i < N; ++i)
        {
            const IndoorFlowPair& p = pairs_[static_cast<std::size_t>(i)];
            const Eigen::Vector3d xa(p.prevNorm.x(), p.prevNorm.y(), 1.0);
            const Eigen::Vector3d q = R * xa;
            const double q3 = q.z();
            if (!(std::abs(q3) > kDegenerateD))
            {
                continue;   // R x_A on the plane at infinity: no finite prediction, no information (r = 0)
            }
            const Eigen::Vector2d g(q.x() / q3, q.y() / q3);
            r.segment<2>(2 * i) = (g - p.currNorm) * invSigma;
            if (jac != nullptr)
            {
                // Dpi(q) = [1/q3, 0, -q1/q3^2; 0, 1/q3, -q2/q3^2]
                Eigen::Matrix<double, 2, 3> Dpi;
                Dpi << 1.0 / q3, 0.0, -q.x() / (q3 * q3),
                       0.0, 1.0 / q3, -q.y() / (q3 * q3);
                for (std::size_t k = 0; k < 12; ++k)
                {
                    if (rp.dR[k].isZero(0.0))
                    {
                        continue;
                    }
                    const Eigen::Vector3d dq = rp.dR[k] * xa;
                    const Eigen::Vector2d dg = (Dpi * dq);
                    jac->block<2, 1>(2 * i, static_cast<Eigen::Index>(k)) = dg * invSigma;
                }
            }
        }
        return r;
    }

    // Epipolar (Sampson-normalised) residual.
    Eigen::Matrix3d Tx = Eigen::Matrix3d::Zero();
    {
        Tx << 0.0, -t.z(), t.y(),
              t.z(), 0.0, -t.x(),
              -t.y(), t.x(), 0.0;
    }
    const Eigen::Matrix3d E = Tx * R;
    std::array<Eigen::Matrix3d, 12> dE{};
    if (jac != nullptr)
    {
        for (std::size_t k = 0; k < 12; ++k)
        {
            const Eigen::Vector3d& dt = rp.dt[k];
            Eigen::Matrix3d dTx = Eigen::Matrix3d::Zero();
            dTx << 0.0, -dt.z(), dt.y(),
                   dt.z(), 0.0, -dt.x(),
                   -dt.y(), dt.x(), 0.0;
            dE[k] = dTx * R + Tx * rp.dR[k];
        }
    }
    for (Eigen::Index i = 0; i < N; ++i)
    {
        const IndoorFlowPair& p = pairs_[static_cast<std::size_t>(i)];
        const Eigen::Vector3d xa(p.prevNorm.x(), p.prevNorm.y(), 1.0);
        const Eigen::Vector3d yb(p.currNorm.x(), p.currNorm.y(), 1.0);
        const Eigen::Vector3d u = E * xa;          // epipolar line in B: u . y = 0
        const Eigen::Vector3d w = E.transpose() * yb;  // epipolar line in A
        const double e = yb.dot(u);
        const double q = u.x() * u.x() + u.y() * u.y() + w.x() * w.x() + w.y() * w.y();
        if (!(q > kDegenerateD))
        {
            continue;   // E x and E^T y vanish in the first two components: no information from this pair (r = 0)
        }
        const double D = std::sqrt(q);
        r(i) = e / D * invSigma;
        if (jac != nullptr)
        {
            // dr/dE = (1/D) y x^T - (e / D^3) (u' x^T + y w'^T)
            const Eigen::Vector3d up(u.x(), u.y(), 0.0);
            const Eigen::Vector3d wp(w.x(), w.y(), 0.0);
            const Eigen::Matrix3d G = (1.0 / D) * (yb * xa.transpose())
                                      - (e / (D * D * D)) * (up * xa.transpose() + yb * wp.transpose());
            for (std::size_t k = 0; k < 12; ++k)
            {
                (*jac)(i, static_cast<Eigen::Index>(k)) = G.cwiseProduct(dE[k]).sum() * invSigma;
            }
        }
    }
    return r;
}

double MeasurementIndoorFlow::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                                            Eigen::MatrixXd* H) const
{
    const Eigen::Index n = system.dim();
    if (x.size() != n)
    {
        throw std::invalid_argument("MeasurementIndoorFlow::logLikelihood: state size mismatch");
    }
    if (n < kPreviousOffset + 6)
    {
        throw std::invalid_argument("MeasurementIndoorFlow::logLikelihood: state must contain eta_k and etaPrev");
    }
    Eigen::MatrixXd J;
    const bool needJac = (g != nullptr) || (H != nullptr);
    const Eigen::VectorXd r = whitenedResiduals(x, needJac ? &J : nullptr);
    const double m = static_cast<double>(r.size());
    const double logL = -0.5 * r.squaredNorm() - m * std::log(sigmaNorm_) - 0.5 * m * std::log(2.0 * std::numbers::pi);
    if (g != nullptr)
    {
        *g = Eigen::VectorXd::Zero(n);
        g->segment<kSupportDim>(kCurrentOffset) = -(J.transpose() * r);
    }
    if (H != nullptr)
    {
        *H = Eigen::MatrixXd::Zero(n, n);
        H->block<kSupportDim, kSupportDim>(kCurrentOffset, kCurrentOffset) = -(J.transpose() * J);
    }
    return logL;
}

} // namespace slam
