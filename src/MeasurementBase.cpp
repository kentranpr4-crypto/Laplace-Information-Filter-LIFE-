#include "MeasurementBase.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include "GaussianInfo.h"

namespace
{

/**
 * @brief Relative tolerance on the curvature spectrum of the Laplace increment K_S.
 *
 * Eigenvalues lambda <= kRelativeCurvatureTolerance * max(1, max|lambda|) are treated as zero: they carry no
 * information and are dropped from the square-root update (a rank-deficient Gauss-Newton curvature has exactly
 * this structure). Negative eigenvalues below -tol are attributed to rounding and dropped too; more negative ones
 * mark K_S as truly indefinite (exact second-order term) and select the documented explicit fallback. The
 * Cholesky fast path is used only when every pivot^2 exceeds the same relative tolerance times the largest one.
 */
constexpr double kRelativeCurvatureTolerance = 1e-10;

} // namespace

MapOutcome decideMapOutcome(FuncMinStatus status, double newtonNormValue, const MapAcceptancePolicy& policy)
{
    if (!(policy.maxIterationsNewtonNorm >= 0.0))
    {
        throw std::invalid_argument("decideMapOutcome: maxIterationsNewtonNorm must be non-negative");
    }
    switch (status)
    {
    case FuncMinStatus::GradientConverged:
    case FuncMinStatus::StepConverged:
    case FuncMinStatus::FunctionConverged:
        return MapOutcome::Converged;
    case FuncMinStatus::MaxIterations:
        return (std::isfinite(newtonNormValue) && newtonNormValue <= policy.maxIterationsNewtonNorm)
                   ? MapOutcome::AcceptedMaxIterations
                   : MapOutcome::SkippedMaxIterations;
    case FuncMinStatus::NumericalFailure:
    default:
        return MapOutcome::SkippedNumericalFailure;
    }
}

void MapStatusCounts::add(MapOutcome outcome, bool precisionIsExact)
{
    ++updates;
    switch (outcome)
    {
    case MapOutcome::Converged: ++converged; break;
    case MapOutcome::AcceptedMaxIterations: ++acceptedMaxIterations; break;
    case MapOutcome::SkippedMaxIterations: ++skippedMaxIterations; break;
    case MapOutcome::SkippedNumericalFailure: ++skippedNumericalFailure; break;
    case MapOutcome::SkippedNoCurvature: ++skippedNoCurvature; break;
    }
    if (!precisionIsExact)
    {
        ++precisionNotExact;
    }
}

MeasurementBase::MeasurementBase(double t, Eigen::VectorXd y) : t_(t), y_(std::move(y))
{
    if (!std::isfinite(t) || y_.size() == 0 || !y_.allFinite())
    {
        throw std::invalid_argument("MeasurementBase: time and observation must be finite and non-empty");
    }
}

std::vector<Eigen::Index> MeasurementBase::supportIndices(const SystemEstimator& /*system*/) const
{
    return {};
}

MeasurementUpdateResult MeasurementBase::process(SystemEstimator& system, const FuncMinOptions& opts,
                                                const MapAcceptancePolicy& policy) const
{
    if (t_ < system.time() - kZeroTimeStep)
    {
        throw std::invalid_argument("MeasurementBase::process: measurement is older than the system time");
    }
    system.predict(t_);

    const GaussianInfo<double> prior = system.state();
    const Eigen::Index n = prior.dim();

    // Support: the observed states (all states when the measurement does not declare a smaller support).
    std::vector<Eigen::Index> support = supportIndices(system);
    if (support.empty())
    {
        support.resize(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i)
        {
            support[static_cast<std::size_t>(i)] = i;
        }
    }
    {
        std::vector<bool> seen(static_cast<std::size_t>(n), false);
        for (Eigen::Index idx : support)
        {
            if (idx < 0 || idx >= n || seen[static_cast<std::size_t>(idx)])
            {
                throw std::invalid_argument("MeasurementBase::process: invalid measurement support");
            }
            seen[static_cast<std::size_t>(idx)] = true;
        }
    }
    const Eigen::Index s = static_cast<Eigen::Index>(support.size());
    const bool naturalFull = (s == n);
    bool isNaturalOrder = naturalFull;
    for (Eigen::Index i = 0; isNaturalOrder && i < s; ++i)
    {
        isNaturalOrder = (support[static_cast<std::size_t>(i)] == i);
    }
    // Marginal prior of the support (exact: the likelihood does not depend on the other states).
    const GaussianInfo<double> priorS = isNaturalOrder ? prior : prior.marginal(support);
    const Eigen::VectorXd xRef = prior.mean();   // values of the non-support states are irrelevant to the likelihood

    const SystemEstimator& sysRef = system;
    auto embed = [&](const Eigen::VectorXd& xs)
    {
        Eigen::VectorXd x = xRef;
        for (Eigen::Index i = 0; i < s; ++i)
        {
            x(support[static_cast<std::size_t>(i)]) = xs(i);
        }
        return x;
    };
    auto restrictVec = [&](const Eigen::VectorXd& full)
    {
        Eigen::VectorXd out(s);
        for (Eigen::Index i = 0; i < s; ++i)
        {
            out(i) = full(support[static_cast<std::size_t>(i)]);
        }
        return out;
    };
    auto restrictMat = [&](const Eigen::MatrixXd& full)
    {
        Eigen::MatrixXd out(s, s);
        for (Eigen::Index i = 0; i < s; ++i)
        {
            for (Eigen::Index j = 0; j < s; ++j)
            {
                out(i, j) = full(support[static_cast<std::size_t>(i)], support[static_cast<std::size_t>(j)]);
            }
        }
        return out;
    };

    FuncMinObjective objective = [&](const Eigen::VectorXd& xs, Eigen::VectorXd* g, Eigen::MatrixXd* h) -> double
    {
        const Eigen::VectorXd x = embed(xs);
        Eigen::VectorXd gl;
        Eigen::MatrixXd hl;
        const double ll = logLikelihood(x, sysRef, g != nullptr ? &gl : nullptr, h != nullptr ? &hl : nullptr);
        Eigen::VectorXd gp;
        Eigen::MatrixXd hp;
        const double lp = priorS.logPdf(xs, g != nullptr ? &gp : nullptr, h != nullptr ? &hp : nullptr);
        if (g != nullptr)
        {
            *g = -(restrictVec(gl) + gp);
        }
        if (h != nullptr)
        {
            *h = -(restrictMat(hl) + hp);
        }
        return -(ll + lp);
    };

    MeasurementUpdateResult out;
    out.map = trustRegionMinimise(objective, priorS.mean(), opts);
    const Eigen::VectorXd xsStar = out.map.x;
    const Eigen::VectorXd xStar = embed(xsStar);
    out.gradientNormInf = out.map.g.size() > 0 ? out.map.g.cwiseAbs().maxCoeff() : 0.0;

    // MAP status policy (see MapAcceptancePolicy). A failed or unusable MAP leaves the prediction in place.
    out.newtonNorm = (out.map.status == FuncMinStatus::NumericalFailure) ? std::numeric_limits<double>::quiet_NaN()
                                                                         : newtonNorm(out.map.g, out.map.H);
    out.outcome = decideMapOutcome(out.map.status, out.newtonNorm, policy);
    if (mapOutcomeSkipsUpdate(out.outcome))
    {
        out.updateSkipped = true;
        out.precisionIsExact = false;
        if (out.outcome == MapOutcome::SkippedNumericalFailure)
        {
            out.map.status = FuncMinStatus::NumericalFailure;
        }
        return out;
    }

    // Laplace increment about the MAP: K = -Hess log p(y|x*) and kappa = grad log p(y|x*) + K x*, both on the support,
    // so that the posterior mean equals the MAP (stationarity of V).
    Eigen::VectorXd gll;
    Eigen::MatrixXd hll;
    try
    {
        logLikelihood(xStar, system, &gll, &hll);
    }
    catch (const std::exception&)
    {
        // The likelihood throws at the MAP (outside the trust region): the update is skipped, no state change.
        out.outcome = MapOutcome::SkippedNumericalFailure;
        out.map.status = FuncMinStatus::NumericalFailure;
        out.updateSkipped = true;
        out.precisionIsExact = false;
        return out;
    }
    Eigen::MatrixXd kS = -restrictMat(hll);
    kS = 0.5 * (kS + kS.transpose());
    const Eigen::VectorXd kappaS = restrictVec(gll) + kS * xsStar;
    if (!kS.allFinite() || !kappaS.allFinite())
    {
        // Non-finite curvature or gradient at the MAP: reported as a numerical failure; the predicted state is kept
        // and the caller decides (SlamPointPipeline reverts the frame).
        out.outcome = MapOutcome::SkippedNumericalFailure;
        out.map.status = FuncMinStatus::NumericalFailure;
        out.updateSkipped = true;
        out.precisionIsExact = false;
        return out;
    }

    // (a) Cholesky path: K_S = U^T U with well-conditioned pivots. The update adds ||U x - U^{-T} kappa||^2.
    const Eigen::LLT<Eigen::MatrixXd> llt(kS);
    bool choleskyOk = false;
    if (llt.info() == Eigen::Success)
    {
        const Eigen::VectorXd pivot2 = llt.matrixL().toDenseMatrix().diagonal().cwiseAbs2();
        choleskyOk = pivot2.minCoeff() > kRelativeCurvatureTolerance * pivot2.maxCoeff();
    }
    if (choleskyOk)
    {
        const Eigen::MatrixXd u = llt.matrixU();
        Eigen::MatrixXd a = Eigen::MatrixXd::Zero(s, n);
        for (Eigen::Index i = 0; i < s; ++i)
        {
            for (Eigen::Index j = 0; j < s; ++j)
            {
                a(i, support[static_cast<std::size_t>(j)]) = u(i, j);
            }
        }
        const Eigen::VectorXd b = u.transpose().template triangularView<Eigen::Lower>().solve(kappaS);
        system.setState(prior.informationUpdate(a, b));
        out.curvatureRank = s;
        return out;
    }

    // (b) Square-root eigen path for positive semi-definite, rank-deficient K_S (e.g. Gauss-Newton curvature of a
    // measurement that constrains fewer directions than the support). K_S = V diag(lambda) V^T, and with the
    // positive eigenpairs F_k = sqrt(lambda_k) v_k^T (rows) and w_k = v_k^T kappa / sqrt(lambda_k):
    //   F^T F = sum_k lambda_k v_k v_k^T = K_S,     F^T w = sum_k v_k v_k^T kappa = P_range kappa,
    // so informationUpdate(F, w) adds exactly K_S and the projection of kappa. For a GN curvature range(K_S) contains
    // kappa (kappa = -J^T r + K x*), hence P_range kappa = kappa and the posterior mean is the MAP.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(kS);
    if (es.info() != Eigen::Success)
    {
        out.outcome = MapOutcome::SkippedNumericalFailure;
        out.map.status = FuncMinStatus::NumericalFailure;
        out.updateSkipped = true;
        out.precisionIsExact = false;
        return out;
    }
    const Eigen::VectorXd lambda = es.eigenvalues();
    const Eigen::MatrixXd v = es.eigenvectors();
    const double scale = std::max(1.0, lambda.cwiseAbs().maxCoeff());
    const double tol = kRelativeCurvatureTolerance * scale;
    if (lambda.minCoeff() >= -tol)
    {
        std::vector<Eigen::Index> positive;
        for (Eigen::Index i = 0; i < s; ++i)
        {
            if (lambda(i) > tol)
            {
                positive.push_back(i);
            }
        }
        const Eigen::Index r = static_cast<Eigen::Index>(positive.size());
        if (r == 0)
        {
            // No curvature above the relative tolerance: the measurement carries no information at this scale (for
            // example sigma ~ 1e5 in the measurement units). The update is skipped and flagged, never silent.
            out.outcome = MapOutcome::SkippedNoCurvature;
            out.updateSkipped = true;
            out.precisionIsExact = false;
            out.curvatureRank = 0;
            system.setState(prior);
            return out;
        }
        Eigen::MatrixXd a = Eigen::MatrixXd::Zero(r, n);
        Eigen::VectorXd b(r);
        for (Eigen::Index k = 0; k < r; ++k)
        {
            const Eigen::Index i = positive[static_cast<std::size_t>(k)];
            const double sq = std::sqrt(lambda(i));
            for (Eigen::Index j = 0; j < s; ++j)
            {
                a(k, support[static_cast<std::size_t>(j)]) = sq * v(j, i);
            }
            b(k) = v.col(i).dot(kappaS) / sq;
        }
        system.setState(prior.informationUpdate(a, b));
        out.curvatureRank = r;
        return out;
    }

    // (c) Truly indefinite K_S (exact second-order terms): fallback with the explicit posterior precision
    // Lambda_prior + K. This is the only place where the information matrix is formed; it is flagged through
    // precisionIsExact = false. The posterior mean is still the MAP when the precision is positive definite.
    out.precisionIsExact = false;
    Eigen::MatrixXd kFull = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd kappaFull = Eigen::VectorXd::Zero(n);
    for (Eigen::Index i = 0; i < s; ++i)
    {
        kappaFull(support[static_cast<std::size_t>(i)]) = kappaS(i);
        for (Eigen::Index j = 0; j < s; ++j)
        {
            kFull(support[static_cast<std::size_t>(i)], support[static_cast<std::size_t>(j)]) = kS(i, j);
        }
    }
    const Eigen::VectorXd nuPost = prior.infoVec() + kappaFull;
    const Eigen::MatrixXd lambdaPost = prior.information() + kFull;
    Eigen::LLT<Eigen::MatrixXd> llPost(0.5 * (lambdaPost + lambdaPost.transpose()));
    out.curvatureRank = s;
    if (llPost.info() == Eigen::Success)
    {
        system.setState(GaussianInfo<double>::fromSqrtInfo(nuPost, llPost.matrixU()));
        return out;
    }

    // Posterior precision not positive definite: floor the eigenvalues (still flagged through precisionIsExact).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esPost(0.5 * (lambdaPost + lambdaPost.transpose()));
    const double floor = 1e-12 * std::max(1.0, esPost.eigenvalues().cwiseAbs().maxCoeff());
    const Eigen::VectorXd lamClamped = esPost.eigenvalues().cwiseMax(floor);
    const Eigen::MatrixXd lambdaFloored = esPost.eigenvectors() * lamClamped.asDiagonal() * esPost.eigenvectors().transpose();
    system.setState(GaussianInfo<double>::fromInfo(nuPost, lambdaFloored));
    return out;
}

MeasurementGaussianLikelihood::MeasurementGaussianLikelihood(ResidualFn h, Eigen::MatrixXd sqrtInfoNoise,
                                                             Eigen::VectorXd y)
    : h_(std::move(h)), lr_(std::move(sqrtInfoNoise)), y_(std::move(y))
{
    if (!h_)
    {
        throw std::invalid_argument("MeasurementGaussianLikelihood: residual model must be set");
    }
    if (lr_.rows() != lr_.cols() || lr_.rows() != y_.size())
    {
        throw std::invalid_argument("MeasurementGaussianLikelihood: Lr must be m x m with m = size(y)");
    }
    for (Eigen::Index i = 0; i < lr_.rows(); ++i)
    {
        if (!(lr_(i, i) > 0.0))
        {
            throw std::invalid_argument("MeasurementGaussianLikelihood: Lr must have a positive diagonal");
        }
    }
    logDetLr_ = lr_.diagonal().cwiseAbs().array().log().sum();
}

Eigen::VectorXd MeasurementGaussianLikelihood::whitenedResidual(const Eigen::VectorXd& x) const
{
    const Eigen::VectorXd hx = h_(x, nullptr);
    return lr_ * (hx - y_);
}

double MeasurementGaussianLikelihood::logLikelihood(const Eigen::VectorXd& x, Eigen::VectorXd* g,
                                                    Eigen::MatrixXd* H) const
{
    Eigen::MatrixXd j;
    const bool needJ = (g != nullptr) || (H != nullptr);
    const Eigen::VectorXd hx = h_(x, needJ ? &j : nullptr);
    const Eigen::VectorXd r = lr_ * (hx - y_);
    const double m = static_cast<double>(y_.size());
    const double value = -0.5 * r.squaredNorm() + logDetLr_ - 0.5 * m * std::log(2.0 * std::numbers::pi_v<double>);

    if (needJ)
    {
        if (j.rows() != y_.size() || j.cols() != x.size())
        {
            throw std::invalid_argument("MeasurementGaussianLikelihood: Jacobian shape mismatch");
        }
        const Eigen::MatrixXd jr = lr_ * j;
        if (g != nullptr)
        {
            *g = -(jr.transpose() * r);
        }
        if (H != nullptr)
        {
            Eigen::MatrixXd hess = -(jr.transpose() * jr);
            if (secondOrder_)
            {
                hess -= secondOrder_(x, r);
            }
            *H = hess;
        }
    }
    return value;
}

MeasurementLinearGaussian::MeasurementLinearGaussian(double t, Eigen::MatrixXd h, Eigen::VectorXd y,
                                                     const Eigen::MatrixXd& r)
    : MeasurementBase(t, y),
      h_(std::move(h)),
      likelihood_(
          [hm = h_](const Eigen::VectorXd& x, Eigen::MatrixXd* jac) -> Eigen::VectorXd
          {
              if (jac != nullptr)
              {
                  *jac = hm;
              }
              return hm * x;
          },
          GaussianInfo<double>::fromMoment(Eigen::VectorXd::Zero(r.rows()), r).sqrtInfo(), std::move(y))
{
    if (h_.rows() != r.rows())
    {
        throw std::invalid_argument("MeasurementLinearGaussian: H rows must equal R size");
    }
}

double MeasurementLinearGaussian::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& /*system*/,
                                                Eigen::VectorXd* g, Eigen::MatrixXd* H) const
{
    return likelihood_.logLikelihood(x, g, H);
}
