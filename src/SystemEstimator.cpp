#include "SystemEstimator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{

/// Plain selection: all indices that are not stochastic, ascending.
std::vector<Eigen::Index> plainKeep(Eigen::Index n, const std::vector<Eigen::Index>& stochastic)
{
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = 0; i < n; ++i)
    {
        if (std::find(stochastic.begin(), stochastic.end(), i) == stochastic.end())
        {
            keep.push_back(i);
        }
    }
    return keep;
}

} // namespace

SystemEstimator::SystemEstimator(std::shared_ptr<const SystemBase> model, State initial, double t0)
    : model_(std::move(model)), state_(std::move(initial)), t_(t0)
{
    if (!model_)
    {
        throw std::invalid_argument("SystemEstimator: model must not be null");
    }
    if (!std::isfinite(t0))
    {
        throw std::invalid_argument("SystemEstimator: initial time must be finite");
    }
}

void SystemEstimator::setState(State s)
{
    state_ = std::move(s);
}

void SystemEstimator::predict(double tNew)
{
    const std::vector<Eigen::Index> stoch = model_->stochasticIndices();
    predict(tNew, plainKeep(dim(), stoch));
}

void SystemEstimator::predict(double tNew, const std::vector<Eigen::Index>& keepOld)
{
    // Times that differ by rounding (|dt| <= kZeroTimeStep) are the same instant: see kZeroTimeStep.
    if (!std::isfinite(tNew) || tNew < t_ - kZeroTimeStep)
    {
        throw std::invalid_argument("SystemEstimator::predict: time must be finite and not decrease");
    }
    const Eigen::Index n = dim();
    const std::vector<Eigen::Index> stoch = model_->stochasticIndices();
    const Eigen::Index k = static_cast<Eigen::Index>(stoch.size());
    for (Eigen::Index idx : stoch)
    {
        if (idx < 0 || idx >= n)
        {
            throw std::invalid_argument("SystemEstimator::predict: stochastic index out of range");
        }
    }

    const bool plain = (keepOld == plainKeep(n, stoch));
    const double dt = tNew - t_;
    if (std::abs(dt) <= kZeroTimeStep)
    {
        // Zero step: no flow and no process noise. A plain prediction leaves the density unchanged; only the time
        // label advances. A shifted (delayed) prediction has no zero-step meaning here: the model owns that semantics
        // (SystemVisualNav::predictDelayed), so it is rejected rather than silently approximated.
        if (!plain)
        {
            throw std::invalid_argument(
                "SystemEstimator::predict: a shifted (delayed) prediction needs |dt| > kZeroTimeStep");
        }
        if (dt > 0.0)
        {
            t_ = tNew;
        }
        return;
    }

    // Mean and state-transition matrix by RK4 on the augmented ODE (x, Phi), Phi' = A(x) Phi.
    const Eigen::VectorXd xm = state_.mean();
    Eigen::MatrixXd phi;
    const Eigen::VectorXd x = integrateFlow(*model_, t_, xm, dt, &phi);

    // Linearised stochastic block: x_S' = Phi_S x + c_S + w, with c_S = x_S(t+dt) - Phi_S x_m.
    Eigen::MatrixXd phiS(k, n);
    Eigen::VectorXd cS(k);
    for (Eigen::Index i = 0; i < k; ++i)
    {
        const Eigen::Index r = stoch[static_cast<std::size_t>(i)];
        phiS.row(i) = phi.row(r);
        cS(i) = x(r) - phiS.row(i).dot(xm);
    }
    const Eigen::MatrixXd lS = model_->processNoiseSqrtInfo(dt, xm);
    if (lS.rows() != k || lS.cols() != k)
    {
        throw std::invalid_argument("SystemEstimator::predict: process-noise factor has the wrong size");
    }

    // New variables occupy joint indices n .. n+k-1 in the order of the stochastic block.
    std::vector<Eigen::Index> keep;
    keep.reserve(static_cast<std::size_t>(k) + keepOld.size());
    for (Eigen::Index i = 0; i < k; ++i)
    {
        keep.push_back(n + i);
    }
    for (Eigen::Index idx : keepOld)
    {
        if (idx < 0 || idx >= n)
        {
            throw std::invalid_argument("SystemEstimator::predict: kept index out of range");
        }
        keep.push_back(idx);
    }

    state_ = state_.linearisedTransition(phiS, cS, lS, keep);
    t_ = tNew;
}

namespace
{

/// RK4 on the augmented ODE (x, Phi), Phi' = A(x) Phi, over the whole vector x0 (dense n x n STM).
Eigen::VectorXd rk4WithStm(const SystemBase& model, double t0, const Eigen::VectorXd& x0, double dt,
                           Eigen::MatrixXd* phi)
{
    const Eigen::Index n = x0.size();
    const int nSub = std::max(1, static_cast<int>(std::ceil(dt / model.maxIntegrationStep())));
    const double h = dt / static_cast<double>(nSub);
    Eigen::VectorXd x = x0;
    Eigen::MatrixXd p = Eigen::MatrixXd::Identity(n, n);
    double t = t0;
    auto deriv = [&model](double tt, const Eigen::VectorXd& xx, const Eigen::MatrixXd& pp,
                          Eigen::VectorXd& dx, Eigen::MatrixXd& dp)
    {
        Eigen::MatrixXd a;
        dx = model.dynamics(tt, xx, &a);
        dp = a * pp;
    };
    for (int s = 0; s < nSub; ++s)
    {
        Eigen::VectorXd k1x, k2x, k3x, k4x;
        Eigen::MatrixXd k1p, k2p, k3p, k4p;
        deriv(t, x, p, k1x, k1p);
        deriv(t + 0.5 * h, x + 0.5 * h * k1x, p + 0.5 * h * k1p, k2x, k2p);
        deriv(t + 0.5 * h, x + 0.5 * h * k2x, p + 0.5 * h * k2p, k3x, k3p);
        deriv(t + h, x + h * k3x, p + h * k3p, k4x, k4p);
        x += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        p += (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        t += h;
    }
    if (phi != nullptr)
    {
        *phi = p;
    }
    return x;
}

} // namespace

Eigen::VectorXd SystemEstimator::integrateFlow(const SystemBase& model, double t0, const Eigen::VectorXd& x0,
                                               double dt, Eigen::MatrixXd* phi)
{
    if (!(dt >= 0.0) || !std::isfinite(dt))
    {
        throw std::invalid_argument("integrateFlow: dt must be finite and non-negative");
    }
    const Eigen::Index n = x0.size();
    const Eigen::Index b = model.dynamicBlockDim();
    if (b < 0 || b > n)
    {
        throw std::invalid_argument("integrateFlow: dynamicBlockDim out of range");
    }
    if (b == 0 || b == n)
    {
        return rk4WithStm(model, t0, x0, dt, phi);
    }

    // Block-wise integration (exact for models with the declared structure, see SystemBase::dynamicBlockDim):
    // the leading b states evolve on their own, the trailing states are static, and Phi = blockdiag(Phi_bb, I).
    // Cost O(b^3) per RK4 stage instead of O(n^3).
    Eigen::MatrixXd phiBlock;
    const Eigen::VectorXd xb = rk4WithStm(model, t0, x0.head(b), dt, phi != nullptr ? &phiBlock : nullptr);
    Eigen::VectorXd x = x0;
    x.head(b) = xb;
    if (phi != nullptr)
    {
        *phi = Eigen::MatrixXd::Identity(n, n);
        phi->topLeftCorner(b, b) = phiBlock;
    }
    return x;
}

void SystemEstimator::augment(const State& block)
{
    state_ = State::join(state_, block);
}

void SystemEstimator::keepOnly(const std::vector<Eigen::Index>& keep)
{
    state_ = state_.marginal(keep);
}
