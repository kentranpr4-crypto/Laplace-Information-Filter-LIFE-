#ifndef SYSTEMESTIMATOR_H
#define SYSTEMESTIMATOR_H

#include <memory>
#include <vector>
#include <Eigen/Core>
#include "GaussianInfo.h"
#include "SystemBase.h"

/**
 * @brief Owns the state density (square-root information form), the current time, and a dynamic model.
 *
 * Time update (information form, no covariance is formed):
 *   - integrate xdot = f(t,x) from (mean, I) with RK4, giving x(t+dt) and the state-transition matrix Phi;
 *   - linearise the stochastic block: x_S(t+dt) = Phi_S x(t) + c_S + w, w ~ N(0, Qd), with L^T L = Qd^{-1};
 *   - form the joint over [x(t) ; x_S(t+dt)] by stacking [Xi | d ; -L Phi_S | L c_S] and marginalise
 *     the variables that are not listed in the kept set (GaussianInfo::linearisedTransition).
 *
 * The updated state is ordered as [new stochastic block (model order) ; kept old states (given order)].
 * With stochastic indices {0..k-1} this keeps the layout of the state unchanged for plain predictions.
 * A delayed-state shift is obtained by keeping an old block that is also stochastic (e.g. eta_k),
 * which then appears as a copy in the kept part of the state, with its correlations preserved.
 */
class SystemEstimator
{
public:
    /// State density type used throughout the project.
    using State = GaussianInfo<double>;

    /**
     * @brief Creates an estimator.
     * @param model   dynamic model (shared, read-only)
     * @param initial initial state density
     * @param t0      initial time (s)
     */
    SystemEstimator(std::shared_ptr<const SystemBase> model, State initial, double t0);

    /** @brief Current time (s). */
    double time() const { return t_; }

    /** @brief Current state density. */
    const State& state() const { return state_; }

    /** @brief Replaces the state density (dimension may change); the time is unchanged. */
    void setState(State s);

    /** @brief Dimension of the state. */
    Eigen::Index dim() const { return state_.dim(); }

    /** @brief Dynamic model. */
    const SystemBase& model() const { return *model_; }

    /** @brief Mean of the state (triangular solves). */
    Eigen::VectorXd mean() const { return state_.mean(); }

    /** @brief Covariance of the state (reference / visualisation only). */
    Eigen::MatrixXd covariance() const { return state_.covariance(); }

    /**
     * @brief Plain time update to tNew: the non-stochastic states are carried over unchanged.
     *
     * A zero step (|tNew - time()| <= kZeroTimeStep, see SystemBase.h) is a no-op on the density: no flow and no
     * process noise are applied (the time label is advanced when tNew > time()).
     * @throws std::invalid_argument if tNew < time() - kZeroTimeStep or tNew is not finite
     */
    void predict(double tNew);

    /**
     * @brief Time update to tNew keeping the listed old states (delayed-state form if they include stochastic states).
     *
     * The new state is [stochastic block (model order); keepOld (given order)]; old states not listed are
     * marginalised out. A shifted prediction needs |dt| > kZeroTimeStep; a zero step is only accepted for the
     * plain selection (no-op). The zero-step delayed shift belongs to the model (SystemVisualNav::predictDelayed).
     * @param keepOld distinct indices of the current state
     * @throws std::invalid_argument for a shifted prediction with |dt| <= kZeroTimeStep or a decreasing time
     */
    void predict(double tNew, const std::vector<Eigen::Index>& keepOld);

    /**
     * @brief RK4 integration of the flow of a model from (t0, x0) over dt, with the state-transition matrix.
     *
     * Uses ceil(dt / model.maxIntegrationStep()) equal sub-steps. Phi(t0+dt, t0) satisfies
     * Phi' = A(x(t)) Phi with Phi(t0) = I. If phi is null the STM is not computed.
     * @return x(t0 + dt)
     */
    static Eigen::VectorXd integrateFlow(const SystemBase& model, double t0, const Eigen::VectorXd& x0, double dt,
                                         Eigen::MatrixXd* phi);

    /**
     * @brief Appends an independent density block (state augmentation). The block occupies the last indices.
     */
    void augment(const State& block);

    /**
     * @brief Keeps only the listed states (in the given order), marginalising the others.
     */
    void keepOnly(const std::vector<Eigen::Index>& keep);

private:
    std::shared_ptr<const SystemBase> model_;
    State state_;
    double t_;
};

#endif
