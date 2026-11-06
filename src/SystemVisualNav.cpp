#include "SystemVisualNav.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <Eigen/LU>
#include "CoreLinalg.h"
#include "rotation.hpp"

namespace
{

constexpr Eigen::Index kDyn = SystemVisualNav::kDynamicDim;

/**
 * @brief Standard deviation (m for position, rad for angles) of the independent jitter in the zero-step delayed shift.
 *
 * An exact copy etaPrev = eta_k with perfectly correlated eta and etaPrev has a singular joint density, which the
 * information form cannot represent (Xi would be singular and the mean undefined). The copy is therefore realised as
 * etaPrev = eta_k + w, w ~ N(0, kDelayedCopyStd^2 I_6), independent of everything else: the mean is exact and the
 * covariance differs from the exact copy by kDelayedCopyStd^2 I_6 (1e-8 for 1e-4). The fixed scale does not involve dt,
 * so the information factor stays well conditioned (L = I / kDelayedCopyStd).
 */
constexpr double kDelayedCopyStd = 1e-4;

} // namespace

SystemVisualNav::SystemVisualNav(const Eigen::Matrix<double, 6, 1>& nuPsd) : nuPsd_(nuPsd)
{
    for (Eigen::Index i = 0; i < 6; ++i)
    {
        if (!(nuPsd_(i) > 0.0) || !std::isfinite(nuPsd_(i)))
        {
            throw std::invalid_argument("SystemVisualNav: nuPsd entries must be finite and strictly positive");
        }
    }
}

Eigen::VectorXd SystemVisualNav::dynamics(double /*t*/, const Eigen::VectorXd& x, Eigen::MatrixXd* A) const
{
    const Eigen::Index n = x.size();
    if (n < kDyn)
    {
        throw std::invalid_argument("SystemVisualNav::dynamics: state too small");
    }
    const Eigen::Vector3d v = x.segment<3>(0);
    const Eigen::Vector3d w = x.segment<3>(3);
    const Eigen::Vector3d theta = x.segment<3>(9);
    requireKinematicPitchSupported(theta);   // body attitude: TK is singular at +/-90 deg (see rotation.hpp)

    const Eigen::Matrix3d r = rpy2rot(theta);
    const Eigen::Matrix3d tk = kinematicMatrixTK(theta);

    Eigen::VectorXd f = Eigen::VectorXd::Zero(n);
    f.segment<3>(6) = r * v;
    f.segment<3>(9) = tk * w;

    if (A != nullptr)
    {
        A->setZero(n, n);
        A->block<3, 3>(6, 0) = r;   // d rdot / d v
        A->block<3, 3>(9, 3) = tk;  // d Thetadot / d omega
        const Eigen::Matrix3d dR[3] = {drpy2rotDphi(theta), drpy2rotDtheta(theta), drpy2rotDpsi(theta)};
        const std::array<Eigen::Matrix3d, 3> dTK = kinematicMatrixTKDerivatives(theta);
        for (int k = 0; k < 3; ++k)
        {
            A->block<3, 1>(6, 9 + k) = dR[k] * v;   // d rdot / d Theta_k
            A->block<3, 1>(9, 9 + k) = dTK[k] * w; // d Thetadot / d Theta_k
        }
    }
    return f;
}

std::vector<Eigen::Index> SystemVisualNav::stochasticIndices() const
{
    std::vector<Eigen::Index> idx;
    idx.reserve(static_cast<std::size_t>(kDyn));
    for (Eigen::Index i = 0; i < kDyn; ++i)
    {
        idx.push_back(i);
    }
    return idx;
}

Eigen::MatrixXd SystemVisualNav::processNoiseSqrtInfo(double dt, const Eigen::VectorXd& x) const
{
    // Qd ~ dt^3 for the pose block, so Qd^{-1} ~ 1/dt^3: a zero-length step has no finite noise factor.
    if (!(dt > kZeroTimeStep) || !std::isfinite(dt))
    {
        throw std::invalid_argument("SystemVisualNav::processNoiseSqrtInfo: dt must exceed kZeroTimeStep");
    }
    if (x.size() < kDyn)
    {
        throw std::invalid_argument("SystemVisualNav::processNoiseSqrtInfo: state too small");
    }
    const Eigen::Vector3d theta = x.segment<3>(9);
    requireKinematicPitchSupported(theta);   // body attitude: T^{-1} needs TK, singular at +/-90 deg

    // T = blockdiag(Rnb, TK): map from body velocity/rate increments to Euler-kinematic increments.
    Eigen::Matrix<double, 6, 6> tInv = Eigen::Matrix<double, 6, 6>::Zero();
    tInv.block<3, 3>(0, 0) = rpy2rot(theta).transpose();     // Rnb^{-1} = Rnb^T
    tInv.block<3, 3>(3, 3) = kinematicMatrixTK(theta).inverse();

    Eigen::MatrixXd ginv = Eigen::MatrixXd::Identity(kDyn, kDyn);
    ginv.block<6, 6>(6, 6) = tInv;

    // Lz = [[D1, D2], [0, D3]] (per axis: W(dt) and I = int W), upper triangular.
    Eigen::MatrixXd lz = Eigen::MatrixXd::Zero(kDyn, kDyn);
    for (Eigen::Index i = 0; i < 6; ++i)
    {
        const double q = nuPsd_(i);
        lz(i, i) = std::sqrt(q * dt);
        lz(i, 6 + i) = 0.5 * dt * dt * std::sqrt(q / dt);
        lz(6 + i, 6 + i) = std::sqrt(q * dt * dt * dt / 12.0);
    }

    // M = Lz^{-T} G^{-1}, so that M^T M = Qd^{-1}.
    const Eigen::MatrixXd m = lz.transpose().template triangularView<Eigen::Lower>().solve(ginv);
    return core::upperFactor(m);
}

Eigen::Index SystemVisualNav::expectedDim() const
{
    Eigen::Index n = landmarkBase();
    for (LandmarkType type : landmarkTypes_)
    {
        n += landmarkDim(type);
    }
    return n;
}

Eigen::Index SystemVisualNav::landmarkDim(LandmarkType type)
{
    return type == LandmarkType::Point ? 3 : 6;
}

std::vector<SystemVisualNav::Landmark> SystemVisualNav::landmarks() const
{
    std::vector<Landmark> out;
    Eigen::Index offset = landmarkBase();
    for (std::size_t i = 0; i < landmarkIds_.size(); ++i)
    {
        Landmark lm;
        lm.id = landmarkIds_[i];
        lm.type = landmarkTypes_[i];
        lm.dim = landmarkDim(lm.type);
        lm.offset = offset;
        offset += lm.dim;
        out.push_back(lm);
    }
    return out;
}

Eigen::Index SystemVisualNav::landmarkOffset(Eigen::Index id) const
{
    for (const Landmark& lm : landmarks())
    {
        if (lm.id == id)
        {
            return lm.offset;
        }
    }
    throw std::out_of_range("SystemVisualNav::landmarkOffset: unknown landmark id " + std::to_string(id));
}

Eigen::Index SystemVisualNav::addLandmark(SystemEstimator& est, LandmarkType type, const GaussianInfo<double>& initial)
{
    if (initial.dim() != landmarkDim(type))
    {
        throw std::invalid_argument("SystemVisualNav::addLandmark: initial density has the wrong dimension");
    }
    if (est.dim() != expectedDim())
    {
        throw std::logic_error("SystemVisualNav::addLandmark: estimator state does not match the model layout");
    }
    est.augment(initial);
    const Eigen::Index id = nextId_++;
    landmarkIds_.push_back(id);
    landmarkTypes_.push_back(type);
    return id;
}

Eigen::Index SystemVisualNav::addLandmarkLinear(SystemEstimator& est, LandmarkType type, const Eigen::MatrixXd& g,
                                                const Eigen::VectorXd& c, const Eigen::MatrixXd& lS)
{
    const Eigen::Index k = landmarkDim(type);
    if (g.rows() != k || g.cols() != est.dim() || c.size() != k || lS.rows() != k || lS.cols() != k)
    {
        throw std::invalid_argument("SystemVisualNav::addLandmarkLinear: inconsistent shapes");
    }
    if (est.dim() != expectedDim())
    {
        throw std::logic_error("SystemVisualNav::addLandmarkLinear: estimator state does not match the model layout");
    }
    const Eigen::Index n = est.dim();
    std::vector<Eigen::Index> keep(static_cast<std::size_t>(n + k));
    for (Eigen::Index i = 0; i < n + k; ++i)
    {
        keep[static_cast<std::size_t>(i)] = i;
    }
    // Joint over [x ; x_L]: the new block is the transition output of the linear map (no old variable is eliminated).
    est.setState(est.state().linearisedTransition(g, c, lS, keep));
    const Eigen::Index id = nextId_++;
    landmarkIds_.push_back(id);
    landmarkTypes_.push_back(type);
    return id;
}

void SystemVisualNav::removeLandmark(SystemEstimator& est, Eigen::Index id)
{
    const std::vector<Landmark> all = landmarks();
    std::size_t pos = all.size();
    for (std::size_t i = 0; i < all.size(); ++i)
    {
        if (all[i].id == id)
        {
            pos = i;
        }
    }
    if (pos == all.size())
    {
        throw std::out_of_range("SystemVisualNav::removeLandmark: unknown landmark id " + std::to_string(id));
    }
    const Landmark target = all[pos];
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = 0; i < est.dim(); ++i)
    {
        if (i < target.offset || i >= target.offset + target.dim)
        {
            keep.push_back(i);
        }
    }
    est.keepOnly(keep);
    landmarkIds_.erase(landmarkIds_.begin() + static_cast<std::ptrdiff_t>(pos));
    landmarkTypes_.erase(landmarkTypes_.begin() + static_cast<std::ptrdiff_t>(pos));
}

void SystemVisualNav::predictDelayed(SystemEstimator& est, double tNew)
{
    if (std::abs(tNew - est.time()) <= kZeroTimeStep)
    {
        // Zero-length step (see kDelayedCopyStd): etaPrev := eta_k with nu and eta carried over unchanged.
        est.predict(tNew);   // zero step: no-op on the density, advances the time label
        const Eigen::Index n = est.dim();
        const Eigen::Index base = landmarkBase();   // landmarks start after the old etaPrev block, if any
        Eigen::MatrixXd e = Eigen::MatrixXd::Zero(kPoseDim, n);
        for (Eigen::Index i = 0; i < kPoseDim; ++i)
        {
            e(i, kEtaOffset + i) = 1.0;
        }
        const Eigen::MatrixXd lS = Eigen::MatrixXd::Identity(kPoseDim, kPoseDim) / kDelayedCopyStd;
        const Eigen::VectorXd c = Eigen::VectorXd::Zero(kPoseDim);
        // Joint over [x_k (n) ; etaPrev_new (6)]; keep [nu_k, eta_k, etaPrev_new, landmarks] in the layout order.
        // The old etaPrev (if any) is integrated out; nu_k and eta_k are the same variables after the shift.
        std::vector<Eigen::Index> keep;
        for (Eigen::Index i = 0; i < kDyn; ++i)
        {
            keep.push_back(i);
        }
        for (Eigen::Index i = 0; i < kPoseDim; ++i)
        {
            keep.push_back(n + i);
        }
        for (Eigen::Index i = base; i < n; ++i)
        {
            keep.push_back(i);
        }
        est.setState(est.state().linearisedTransition(e, c, lS, keep));
        hasDelayedPose_ = true;
        if (est.dim() != expectedDim())
        {
            throw std::logic_error("SystemVisualNav::predictDelayed: layout mismatch after zero-step shift");
        }
        return;
    }

    // Keep eta_k (becomes etaPrev_{k+1}) and all landmark blocks; nu_k and the old etaPrev are integrated out.
    std::vector<Eigen::Index> keepOld;
    for (Eigen::Index i = kEtaOffset; i < kEtaOffset + kPoseDim; ++i)
    {
        keepOld.push_back(i);
    }
    for (Eigen::Index i = landmarkBase(); i < est.dim(); ++i)
    {
        keepOld.push_back(i);
    }
    est.predict(tNew, keepOld);
    hasDelayedPose_ = true;
    if (est.dim() != expectedDim())
    {
        throw std::logic_error("SystemVisualNav::predictDelayed: layout mismatch after prediction");
    }
}

void SystemVisualNav::predictPlain(SystemEstimator& est, double tNew)
{
    std::vector<Eigen::Index> keepOld;
    if (hasDelayedPose_)
    {
        for (Eigen::Index i = kEtaPrevOffset; i < kEtaPrevOffset + kPoseDim; ++i)
        {
            keepOld.push_back(i);
        }
    }
    for (Eigen::Index i = landmarkBase(); i < est.dim(); ++i)
    {
        keepOld.push_back(i);
    }
    est.predict(tNew, keepOld);
    if (est.dim() != expectedDim())
    {
        throw std::logic_error("SystemVisualNav::predictPlain: layout mismatch after prediction");
    }
}

void SystemVisualNav::discardDelayedPose(SystemEstimator& est)
{
    if (!hasDelayedPose_)
    {
        throw std::logic_error("SystemVisualNav::discardDelayedPose: no delayed pose in the state");
    }
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = 0; i < est.dim(); ++i)
    {
        if (i < kEtaPrevOffset || i >= kEtaPrevOffset + kPoseDim)
        {
            keep.push_back(i);
        }
    }
    est.keepOnly(keep);
    hasDelayedPose_ = false;
}
