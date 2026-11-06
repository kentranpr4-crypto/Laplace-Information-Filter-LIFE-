#ifndef FUNCMIN_HPP
#define FUNCMIN_HPP

#include <functional>
#include <Eigen/Core>

/**
 * @brief Unconstrained minimisation for the Laplace MAP step (dense, deterministic).
 *
 * Trust-region Newton method (Nocedal & Wright, Alg. 4.1) with the exact trust-region subproblem.
 * When the Hessian is positive definite (Cholesky succeeds) and the Newton step lies inside the radius, that step is
 * returned without an eigen-decomposition. Otherwise the subproblem is solved through the eigen-decomposition of
 * the supplied Hessian (More-Sorensen type secular equation with the hard case). Indefinite Hessians are handled
 * without modification of H.
 * A quasi-Newton fallback is not implemented; the exact Hessian must be supplied.
 */

/** @brief Objective evaluation: returns f(x) and, if non-null, fills gradient g and Hessian H. */
using FuncMinObjective = std::function<double(const Eigen::VectorXd& x, Eigen::VectorXd* g, Eigen::MatrixXd* H)>;

/** @brief Termination status of trustRegionMinimise. */
enum class FuncMinStatus
{
    GradientConverged,   ///< ||g||_inf <= gradientTolerance
    StepConverged,       ///< step below stepTolerance (relative)
    FunctionConverged,   ///< predicted and actual reductions both below functionTolerance
    MaxIterations,       ///< iteration limit reached (result is the best iterate found; see MapAcceptancePolicy)
    NumericalFailure     ///< non-finite or throwing objective at the starting or an accepted point, or the radius collapsed
};

/**
 * @brief Exceptions from the objective (FuncMinObjective) are contained inside trustRegionMinimise.
 *
 * An std::exception thrown while evaluating the objective at the starting point or at an accepted iterate ends the
 * minimisation with NumericalFailure and returns the last accepted iterate (no step is taken from a failed point).
 * An exception at a trial point is treated like a non-finite trial value: the step is rejected and the trust radius
 * is reduced. NumericalFailure is returned if the radius then falls below stepTolerance. No exception escapes.
 */

/** @brief Options of the trust-region minimiser (defaults are deterministic and conservative). */
struct FuncMinOptions
{
    int maxIterations = 200;           ///< iteration limit
    double gradientTolerance = 1e-9;   ///< stop when max|g| is below this
    double stepTolerance = 1e-12;      ///< stop when ||p|| <= stepTolerance (1 + ||x||)
    double functionTolerance = 1e-14;  ///< stop when reductions are below this (relative to 1 + |f|)
    double initialRadius = 1.0;        ///< initial trust-region radius
    double maxRadius = 1e8;            ///< maximum trust-region radius
    double acceptRatio = 1e-4;         ///< minimum rho to accept a step
};

/** @brief Result of trustRegionMinimise. */
struct FuncMinResult
{
    Eigen::VectorXd x;      ///< final iterate
    double f = 0.0;         ///< objective at x
    Eigen::VectorXd g;      ///< gradient at x
    Eigen::MatrixXd H;      ///< Hessian at x
    int iterations = 0;     ///< number of outer iterations performed
    FuncMinStatus status = FuncMinStatus::MaxIterations;
};

/**
 * @brief Minimises f over R^n starting from x0 with a trust-region Newton method.
 * @param f     objective with gradient and Hessian (see FuncMinObjective)
 * @param x0    starting point
 * @param opts  options
 * @return final iterate, its gradient/Hessian and the termination status
 */
FuncMinResult trustRegionMinimise(const FuncMinObjective& f, const Eigen::VectorXd& x0,
                                  const FuncMinOptions& opts = FuncMinOptions());

/**
 * @brief Newton norm lambda = sqrt(g^T H^{-1} g) of a stationary-point test, computed by one triangular solve.
 *
 * With H = L L^T (Cholesky), lambda = ||L^{-1} g||. lambda is the Mahalanobis distance, in the metric H, from the
 * current iterate to the Newton (stationary) point x - H^{-1} g of the local quadratic model; it is dimensionless
 * and invariant to the scaling of the variables. No inverse of H is formed.
 * @return lambda >= 0, or NaN if H is not symmetric positive definite (the quantity is then undefined) or g is non-finite
 */
double newtonNorm(const Eigen::VectorXd& g, const Eigen::MatrixXd& H);

/**
 * @brief Exact solution of the trust-region subproblem min g^T p + 0.5 p^T H p subject to ||p|| <= radius.
 *
 * Uses the eigen-decomposition H = V diag(lambda) V^T and solves the secular equation for the
 * multiplier sigma >= max(0, -lambda_min) by bisection. Handles indefinite H and the hard case.
 */
Eigen::VectorXd solveTrustRegionSubproblem(const Eigen::VectorXd& g, const Eigen::MatrixXd& H, double radius);

#endif
