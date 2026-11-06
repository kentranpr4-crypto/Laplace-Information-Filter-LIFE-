#ifndef SYSTEMBASE_H
#define SYSTEMBASE_H

#include <vector>
#include <Eigen/Core>

/**
 * @brief Time steps with |dt| <= kZeroTimeStep (s) are treated as zero.
 *
 * A zero step is a no-op for plain predictions (the state is unchanged, only the time label advances). The
 * process-noise information of a step grows like 1/dt^3 for the pose block (SystemVisualNav), so rounding-level
 * timestamp differences (e.g. 1e-16 s) must never be integrated with process noise: they would turn the
 * information factor ill-conditioned (observed: smallest singular value 0.49 -> 5.8e-15 and a diverging state).
 */
inline constexpr double kZeroTimeStep = 1e-9;

/**
 * @brief Abstract continuous-time dynamic model with a stochastic block (interface only).
 *
 * A model defines
 *   - the deterministic flow xdot = f(t, x) for the full state vector x, with Jacobian A = df/dx,
 *   - the stochastic block: the (ascending) state indices that receive process noise,
 *   - the discrete process-noise square-root information L over that block for a step dt at state x:
 *     the noise increment w ~ N(0, Qd) of the block satisfies L^T L = Qd^{-1}.
 *
 * The estimator (SystemEstimator) integrates the flow with RK4 together with its state-transition
 * matrix and performs the information-form time update.
 */
class SystemBase
{
public:
    virtual ~SystemBase() = default;

    /**
     * @brief Continuous-time flow f(t, x) and optionally its Jacobian A = df/dx.
     * @param t  time (s)
     * @param x  full state (size n)
     * @param A  if non-null, receives the n x n Jacobian
     * @return xdot (size n)
     */
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd& x, Eigen::MatrixXd* A) const = 0;

    /**
     * @brief Indices (ascending) of the states that receive process noise.
     *
     * Each step the estimator returns these states as new variables placed first in the updated state.
     */
    virtual std::vector<Eigen::Index> stochasticIndices() const = 0;

    /**
     * @brief Square-root information L (k x k, upper triangular) of the discrete process noise of the
     * stochastic block over a step of length dt evaluated at state x: L^T L = Qd^{-1}.
     * @throws std::invalid_argument if dt <= 0.
     */
    virtual Eigen::MatrixXd processNoiseSqrtInfo(double dt, const Eigen::VectorXd& x) const = 0;

    /**
     * @brief Dimension b of the leading state block on which the flow depends (0 = unstructured).
     *
     * A model returns b in [1, n) only if, for every state x of dimension n:
     *   - f_i(t, x) = 0 and A_ij = 0 for all i >= b and all j (the trailing states are static), and
     *   - A_ij = 0 for i < b and j >= b (the leading block does not depend on the trailing states).
     * The estimator then integrates only the leading block (RK4 on the b x b STM), which is exact because
     * the full state-transition matrix is blockdiag(Phi_bb, I). In this case dynamics() is called with a
     * vector of size b (the leading entries of x) and must not access entries >= b.
     * Returning 0 (default) selects the unstructured full-state integration.
     */
    virtual Eigen::Index dynamicBlockDim() const { return 0; }

    /** @brief Largest RK4 sub-step (s) used when integrating the flow over one prediction step. */
    double maxIntegrationStep() const { return maxIntegrationStep_; }

    /** @brief Sets the largest RK4 sub-step (s); must be positive. */
    void setMaxIntegrationStep(double h);

protected:
    double maxIntegrationStep_ = 0.01;
};

#endif
