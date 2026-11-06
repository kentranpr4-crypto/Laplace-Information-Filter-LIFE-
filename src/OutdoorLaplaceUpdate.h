#ifndef OUTDOORLAPLACEUPDATE_H
#define OUTDOORLAPLACEUPDATE_H

#include <limits>
#include <Eigen/Core>
#include "MeasurementBase.h"
#include "SystemEstimator.h"

/** @brief Diagnostics of one square-root Laplace update. */
struct OutdoorUpdateReport
{
    FuncMinResult map;                    ///< MAP optimisation (status, iterations, objective)
    Eigen::Index rank = 0;                ///< rank r of the support curvature K_S used for the update (rows of the factor)
    double droppedNegativeMass = 0.0;     ///< sum of |negative eigenvalues| of K_S that were not applied (0 for GN)
    double droppedGradientNorm = 0.0;     ///< norm of the part of kappa_S outside range(K_S) (0 when invariance holds)
    /// MAP status policy decision, identical to MeasurementBase::process (see MapAcceptancePolicy).
    MapOutcome outcome = MapOutcome::Converged;
    /// True when the state was left at the prediction (outcome is a Skipped value). Never silent.
    bool updateSkipped = false;
    /// Newton norm lambda = sqrt(g^T H^{-1} g) at the MAP iterate (NaN when undefined).
    double newtonNorm = std::numeric_limits<double>::quiet_NaN();
    /// max|g| at the MAP iterate (NaN when not evaluated at a finite point).
    double gradientNormInf = std::numeric_limits<double>::quiet_NaN();
};

/**
 * @brief Laplace (MAP) measurement update in pure square-root information form.
 *
 * Same mathematics as MeasurementBase::process (predict, MAP over the measurement support, posterior precision
 * Lambda + K with K = -Hess log p(y|x*), information vector nu + kappa, kappa = grad + K x*), but the update is
 * applied without ever forming the posterior information matrix:
 *
 *   K_S = V diag(lambda) V^T,  keep the positive eigenpairs (r of them) and set
 *   F = diag(sqrt(lambda_r)) V_r^T  (r x s),   w = diag(1/sqrt(lambda_r)) V_r^T kappa_S  (so F^T w = projection of kappa_S),
 *   state <- prior.informationUpdate(A = [F in the support columns], b = w),
 *
 * which adds A^T A = K_S and A^T b = kappa_S to Lambda and nu through Q-less QR (GaussianInfo::informationUpdate).
 * This matters for rank-deficient likelihoods (flow and horizon GN curvature is singular): MeasurementBase::process
 * then forms Lambda + K explicitly, squaring the condition number; in the synthetic Scenario 4 test flight this drove
 * the posterior mean to 1e13 m in the unobservable modes (global heading and horizontal offset) at frame 21.
 *
 * MAP status policy: identical to MeasurementBase::process (MapAcceptancePolicy). A converged MAP is applied; a
 * MaxIterations result is applied only within the Newton-norm bound; NumericalFailure, a non-finite or throwing
 * curvature, and a zero-information update are skipped with the state left at the prediction and updateSkipped set.
 *
 * Exact when the likelihood is invariant along the null space of K (then kappa_S has no component there); the
 * dropped component is reported.
 *
 * @throws std::invalid_argument if the measurement is older than the system time or the support is invalid
 */
OutdoorUpdateReport outdoorLaplaceUpdate(const MeasurementBase& measurement, SystemEstimator& system,
                                         const FuncMinOptions& opts = FuncMinOptions(),
                                         const MapAcceptancePolicy& policy = MapAcceptancePolicy());

#endif
