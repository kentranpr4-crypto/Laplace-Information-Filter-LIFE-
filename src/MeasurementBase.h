#ifndef MEASUREMENTBASE_H
#define MEASUREMENTBASE_H

#include <cstddef>
#include <functional>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include "funcmin.hpp"
#include "SystemEstimator.h"

/**
 * @brief What the Laplace update did with the MAP result (MAP status policy, see MapAcceptancePolicy).
 *
 * Every update ends in exactly one of these outcomes. Only the first two apply the posterior; the three Skipped
 * outcomes leave the state at the prediction and set MeasurementUpdateResult::updateSkipped.
 */
enum class MapOutcome
{
    Converged,                ///< GradientConverged, StepConverged or FunctionConverged: posterior applied
    AcceptedMaxIterations,    ///< MaxIterations, but the Newton norm at the iterate is within the policy: posterior applied
    SkippedMaxIterations,     ///< MaxIterations and the iterate is not usable: update skipped, prediction kept
    SkippedNumericalFailure,  ///< non-finite or throwing objective / curvature: update skipped, prediction kept
    SkippedNoCurvature        ///< the likelihood carries no information above the relative tolerance: update skipped
};

/**
 * @brief MAP status policy (applies to MeasurementBase::process and outdoorLaplaceUpdate).
 *
 * A converged MAP (gradient, step or function criterion) is always applied. A MaxIterations result is a point at which
 * the optimiser stopped without meeting a criterion, so it is not necessarily the mode of the posterior. It is applied
 * only if its Newton norm lambda = sqrt(g^T H^{-1} g) is at most maxIterationsNewtonNorm: lambda is the Mahalanobis
 * distance (in posterior standard deviations) from the iterate to the stationary point of the local quadratic model,
 * so the applied posterior mean lies within that many posterior sigmas of the true MAP. Otherwise the update is
 * skipped and the prediction is kept. NumericalFailure is always skipped.
 */
struct MapAcceptancePolicy
{
    /// Largest accepted Newton norm of a MaxIterations result (posterior sigmas). Default 0.05.
    double maxIterationsNewtonNorm = 0.05;
};

/**
 * @brief Maps a MAP result to its outcome under the policy.
 * @param status      termination status of the MAP optimisation
 * @param newtonNormValue  Newton norm at the returned iterate (newtonNorm(g, H)); NaN when undefined
 * @param policy      acceptance policy
 */
MapOutcome decideMapOutcome(FuncMinStatus status, double newtonNormValue, const MapAcceptancePolicy& policy);

/** @brief True for the outcomes that leave the state at the prediction. */
inline bool mapOutcomeSkipsUpdate(MapOutcome outcome)
{
    return outcome == MapOutcome::SkippedMaxIterations || outcome == MapOutcome::SkippedNumericalFailure ||
           outcome == MapOutcome::SkippedNoCurvature;
}

/** @brief Outcome of MeasurementBase::process. */
struct MeasurementUpdateResult
{
    FuncMinResult map;                 ///< MAP optimisation result (status, iterations, gradient/Hessian at MAP)
    /**
     * True when the posterior is the Laplace posterior computed in square-root form: the Cholesky path (K positive
     * definite) or the eigen path (K positive semi-definite and rank deficient, dropped eigenvalues <= 1e-10 relative).
     * False when K is truly indefinite (exact second-order terms; the posterior precision Lambda_prior + K is then
     * formed explicitly), when that precision had to be floored, or when the update was skipped (updateSkipped).
     */
    bool precisionIsExact = true;
    /// MAP status policy decision (see MapOutcome). Converged/AcceptedMaxIterations apply the posterior.
    MapOutcome outcome = MapOutcome::Converged;
    /**
     * True when the state was left at the prediction because the update was skipped (outcome is one of the Skipped
     * values). A skipped update is never silent: callers and counters (MapStatusCounts) must record it.
     */
    bool updateSkipped = false;
    /// Newton norm lambda = sqrt(g^T H^{-1} g) at the returned MAP iterate (NaN when undefined). See MapAcceptancePolicy.
    double newtonNorm = std::numeric_limits<double>::quiet_NaN();
    /// max|g| at the returned MAP iterate (NaN when the objective was not evaluated at a finite point).
    double gradientNormInf = std::numeric_limits<double>::quiet_NaN();
    /// Number of square-root rows applied to the prior (0 when the update was skipped).
    Eigen::Index curvatureRank = 0;
};

/**
 * @brief Counters of MAP outcomes over a run. Accumulate one entry per Laplace update.
 */
struct MapStatusCounts
{
    std::size_t updates = 0;                    ///< updates recorded
    std::size_t converged = 0;                  ///< MapOutcome::Converged
    std::size_t acceptedMaxIterations = 0;      ///< MapOutcome::AcceptedMaxIterations
    std::size_t skippedMaxIterations = 0;       ///< MapOutcome::SkippedMaxIterations
    std::size_t skippedNumericalFailure = 0;    ///< MapOutcome::SkippedNumericalFailure
    std::size_t skippedNoCurvature = 0;         ///< MapOutcome::SkippedNoCurvature
    std::size_t precisionNotExact = 0;          ///< updates with precisionIsExact == false (including skipped ones)

    /** @brief Records one update. */
    void add(MapOutcome outcome, bool precisionIsExact);
    /** @brief Records one update from its result. */
    void add(const MeasurementUpdateResult& result) { add(result.outcome, result.precisionIsExact); }
    /** @brief Number of updates whose state was left at the prediction. */
    std::size_t skipped() const { return skippedMaxIterations + skippedNumericalFailure + skippedNoCurvature; }
};

/**
 * @brief Abstract measurement at time t with observation y (Laplace / MAP information-form update).
 *
 * process(system) performs
 *   1. system.predict(t)                              (information-form time update),
 *   2. MAP:  x* = argmin V(x),  V(x) = -log p(y|x) - log p(x),      (trust-region Newton, funcmin),
 *   3. posterior: K = -Hess(log p(y|x*)) on the support, kappa = grad(log p(y|x*)) + K x*, and the update
 *      nu_post = nu_prior + kappa, Lambda_post = Lambda_prior + K applied as an information update with a factor of K:
 *        - Cholesky K = U^T U (positive definite):    informationUpdate(U, U^{-T} kappa),
 *        - eigen K = V diag(lambda) V^T (PSD, rank deficient):  rows sqrt(lambda_+) v_+^T with w = v^T kappa / sqrt(lambda),
 *          so the projection of kappa onto range(K) is added (equal to kappa for Gauss-Newton curvature),
 *        - indefinite K: explicit Lambda_prior + K fallback (precisionIsExact = false).
 *      No covariance is inverted and the precision is not formed on the square-root paths.
 *
 * The posterior mean equals x* (stationarity of V), and for a linear-Gaussian measurement the result is
 * exactly the information-filter (Kalman) update (tested).
 *
 * MAP status policy: a converged MAP is applied; a MaxIterations result only if its Newton norm is within
 * MapAcceptancePolicy; any NumericalFailure, a throwing likelihood, or a zero-information measurement is skipped. A skipped
 * update leaves the state at the prediction and sets MeasurementUpdateResult::updateSkipped (never silent).
 */
class MeasurementBase
{
public:
    /**
     * @param t  measurement time (s)
     * @param y  observation vector
     */
    MeasurementBase(double t, Eigen::VectorXd y);
    virtual ~MeasurementBase() = default;

    /** @brief Measurement time (s). */
    double time() const { return t_; }

    /** @brief Observation vector. */
    const Eigen::VectorXd& y() const { return y_; }

    /**
     * @brief log p(y | x) for the full state x, with optional gradient g and Hessian H with respect to x.
     *
     * Implementations must fill g and H (when non-null) exactly (hand-written; validated by finite differences).
     * @param x       full state vector (size = system.dim())
     * @param system  estimator (read-only; used for the layout / indices of the observed states)
     */
    virtual double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                                 Eigen::MatrixXd* H) const = 0;

    /**
     * @brief States the likelihood depends on (the measurement support). Empty (default) means all states.
     *
     * The MAP is then computed over the support only; this is exact (Gaussian marginal/conditional
     * identity) provided logLikelihood does not depend on states outside the returned set.
     */
    virtual std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const;

    /**
     * @brief Predicts the system to time() and applies the Laplace update (see class documentation).
     *
     * MAP status policy (MapAcceptancePolicy): converged MAP applied; MaxIterations applied only if its Newton norm is
     * within policy.maxIterationsNewtonNorm; NumericalFailure, a throwing likelihood, or a zero-information measurement
     * skip the update. A skipped update leaves the state at the prediction and sets updateSkipped.
     * @return the MAP optimisation result, the outcome and the diagnostics of the update
     * @throws std::invalid_argument if time() is before the system time
     */
    MeasurementUpdateResult process(SystemEstimator& system, const FuncMinOptions& opts = FuncMinOptions(),
                                    const MapAcceptancePolicy& policy = MapAcceptancePolicy()) const;

private:
    double t_;
    Eigen::VectorXd y_;
};

/**
 * @brief Reusable helper for residual models y = h(x) + v, v ~ N(0, R).
 *
 * Whitened residual r(x) = Lr (h(x) - y), with Lr upper-triangular sqrt-information of the noise
 * (Lr^T Lr = R^{-1}). Then
 *   log p(y|x) = -0.5 ||r||^2 + sum log|diag Lr| - (m/2) log(2 pi),
 *   grad = -Jr^T r,         Jr = Lr J,   J = dh/dx (supplied by the residual model),
 *   Hess = -(Jr^T Jr + S),  S = sum_k r_k Hess(r_k)  (second-order term).
 * By default S = 0 (Gauss-Newton Hessian). An exact second-order term can be supplied with
 * setSecondOrderTerm(fn) where fn(x, r) returns S for the whitened residual r (the caller
 * forms sum_k r_k * sum_j Lr(k,j) * Hess(h_j)).
 */
class MeasurementGaussianLikelihood
{
public:
    /// Residual model: returns h(x) and, if J != nullptr, fills J = dh/dx (m x n).
    using ResidualFn = std::function<Eigen::VectorXd(const Eigen::VectorXd& x, Eigen::MatrixXd* J)>;
    /// Second-order term S(x, r) = sum_k r_k Hess(r_k)(x) (n x n).
    using SecondOrderFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd& x, const Eigen::VectorXd& r)>;

    /**
     * @param h             residual model
     * @param sqrtInfoNoise upper-triangular Lr (m x m) with Lr^T Lr = R^{-1}
     * @param y             observation (size m)
     * @throws std::invalid_argument on inconsistent sizes or non-positive diagonal of Lr
     */
    MeasurementGaussianLikelihood(ResidualFn h, Eigen::MatrixXd sqrtInfoNoise, Eigen::VectorXd y);

    /** @brief Enables the exact (second-order) Hessian. Without it the Gauss-Newton Hessian is used. */
    void setSecondOrderTerm(SecondOrderFn fn) { secondOrder_ = std::move(fn); }

    /** @brief Whitened residual r(x) = Lr (h(x) - y). */
    Eigen::VectorXd whitenedResidual(const Eigen::VectorXd& x) const;

    /**
     * @brief Log-likelihood with optional gradient and Hessian (see class documentation).
     */
    double logLikelihood(const Eigen::VectorXd& x, Eigen::VectorXd* g, Eigen::MatrixXd* H) const;

    /** @brief Observation dimension m. */
    Eigen::Index dimension() const { return y_.size(); }

private:
    ResidualFn h_;
    Eigen::MatrixXd lr_;
    Eigen::VectorXd y_;
    SecondOrderFn secondOrder_;
    double logDetLr_ = 0.0;
};

/**
 * @brief Concrete linear-Gaussian measurement y = H x + v, v ~ N(0, R). Used by the tests as a reference.
 */
class MeasurementLinearGaussian : public MeasurementBase
{
public:
    /**
     * @param t  measurement time
     * @param h  observation matrix H (m x n)
     * @param y  observation (m)
     * @param r  noise covariance R (m x m, symmetric positive definite)
     */
    MeasurementLinearGaussian(double t, Eigen::MatrixXd h, Eigen::VectorXd y, const Eigen::MatrixXd& r);

    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

private:
    Eigen::MatrixXd h_;
    MeasurementGaussianLikelihood likelihood_;
};

#endif
