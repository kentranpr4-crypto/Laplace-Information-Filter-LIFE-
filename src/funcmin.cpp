#include "funcmin.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace
{

/// p(sigma) = -(H + sigma I)^{-1} g using the eigen-decomposition; terms with a vanishing denominator are skipped.
Eigen::VectorXd shiftedStep(const Eigen::VectorXd& lam, const Eigen::MatrixXd& V, const Eigen::VectorXd& gh,
                            double sigma, double skipBelow)
{
    Eigen::VectorXd p = Eigen::VectorXd::Zero(V.rows());
    for (Eigen::Index i = 0; i < lam.size(); ++i)
    {
        const double denom = lam(i) + sigma;
        if (denom <= skipBelow)
        {
            continue;
        }
        p -= (gh(i) / denom) * V.col(i);
    }
    return p;
}

/// Second-order necessary condition: H has no (numerically) negative eigenvalue.
/// A successful Cholesky factorisation proves positive definiteness, so the eigen-decomposition is skipped then.
bool isLocallyPositiveSemidefinite(const Eigen::MatrixXd& h)
{
    if (h.rows() == 0)
    {
        return true;
    }
    const Eigen::MatrixXd hs = 0.5 * (h + h.transpose());
    if (Eigen::LLT<Eigen::MatrixXd>(hs).info() == Eigen::Success)
    {
        return true;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(hs);
    if (es.info() != Eigen::Success)
    {
        return false;
    }
    const double scale = std::max(1.0, es.eigenvalues().cwiseAbs().maxCoeff());
    return es.eigenvalues().minCoeff() >= -1e-10 * scale;
}

/// Evaluates the objective; an std::exception is reported through `threw` and does not escape (see funcmin.hpp).
double evaluateContained(const FuncMinObjective& f, const Eigen::VectorXd& x, Eigen::VectorXd* g, Eigen::MatrixXd* h,
                         bool& threw)
{
    threw = false;
    try
    {
        return f(x, g, h);
    }
    catch (const std::exception&)
    {
        threw = true;
        return std::numeric_limits<double>::quiet_NaN();
    }
}

} // namespace

Eigen::VectorXd solveTrustRegionSubproblem(const Eigen::VectorXd& g, const Eigen::MatrixXd& H, double radius)
{
    const Eigen::Index n = g.size();
    if (H.rows() != n || H.cols() != n)
    {
        throw std::invalid_argument("solveTrustRegionSubproblem: H must be n x n");
    }
    if (!(radius > 0.0))
    {
        throw std::invalid_argument("solveTrustRegionSubproblem: radius must be positive");
    }
    if (n == 0)
    {
        return Eigen::VectorXd();
    }

    const Eigen::MatrixXd hs = 0.5 * (H + H.transpose());

    // Fast path: H positive definite (Cholesky succeeds) and the Newton step -H^{-1} g lies inside the trust region.
    // It is then the exact solution of the subproblem, and the eigen-decomposition below is not needed.
    const Eigen::LLT<Eigen::MatrixXd> llt(hs);
    if (llt.info() == Eigen::Success)
    {
        Eigen::VectorXd newton = -llt.solve(g);
        if (newton.allFinite() && newton.norm() <= radius)
        {
            return newton;
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(hs);
    if (es.info() != Eigen::Success)
    {
        throw std::runtime_error("solveTrustRegionSubproblem: eigen-decomposition failed");
    }
    const Eigen::VectorXd lam = es.eigenvalues();     // ascending
    const Eigen::MatrixXd V = es.eigenvectors();
    const Eigen::VectorXd gh = V.transpose() * g;
    const double lamScale = std::max(1.0, lam.cwiseAbs().maxCoeff());
    const double lamMin = lam(0);

    // Interior Newton step when H is positive definite.
    if (lamMin > 0.0)
    {
        Eigen::VectorXd p = shiftedStep(lam, V, gh, 0.0, 0.0);
        if (p.norm() <= radius)
        {
            return p;
        }
    }

    // Boundary solution: ||p(sigma)|| = radius with sigma > sLow.
    const double sLow = (lamMin > 0.0) ? 0.0 : -lamMin;
    const double eps = 1e-14 * lamScale;

    // Hard case: the gradient has (numerically) no component along the most negative eigenvector.
    const double gMinComponent = std::abs(gh(0));
    if (lamMin <= 0.0 && gMinComponent <= 1e-12 * std::max(1.0, g.norm()))
    {
        const Eigen::VectorXd pPartial = shiftedStep(lam, V, gh, sLow + eps, eps);
        if (pPartial.norm() <= radius)
        {
            // Terms of the (near-)degenerate minimal eigenvalue cluster are excluded above; add tau v_min.
            Eigen::VectorXd p = pPartial;
            const double tau = std::sqrt(std::max(0.0, radius * radius - pPartial.squaredNorm()));
            p += tau * V.col(0);
            return p;
        }
    }

    // Bracket the root of ||p(sigma)|| = radius; ||p|| is strictly decreasing on (sLow, inf).
    double lo = sLow + eps;
    double hi = sLow + std::max(g.norm() / radius, 1e-12) + std::abs(lamMin);
    for (int k = 0; k < 200 && shiftedStep(lam, V, gh, hi, 0.0).norm() > radius; ++k)
    {
        hi = sLow + 2.0 * (hi - sLow);
    }
    for (int k = 0; k < 200; ++k)
    {
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi)
        {
            break;
        }
        const double pn = shiftedStep(lam, V, gh, mid, 0.0).norm();
        if (pn > radius)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
        if (std::abs(pn - radius) <= 1e-12 * radius)
        {
            break;
        }
    }
    return shiftedStep(lam, V, gh, hi, 0.0);
}

FuncMinResult trustRegionMinimise(const FuncMinObjective& f, const Eigen::VectorXd& x0, const FuncMinOptions& opts)
{
    FuncMinResult res;
    res.x = x0;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    bool threw = false;
    res.f = evaluateContained(f, res.x, &g, &H, threw);
    if (threw)
    {
        res.status = FuncMinStatus::NumericalFailure;
        return res;
    }
    res.g = g;
    res.H = H;

    if (!std::isfinite(res.f) || !g.allFinite() || !H.allFinite())
    {
        res.status = FuncMinStatus::NumericalFailure;
        return res;
    }

    double radius = opts.initialRadius;
    for (int iter = 0; iter < opts.maxIterations; ++iter)
    {
        res.iterations = iter;
        if (res.g.cwiseAbs().maxCoeff() <= opts.gradientTolerance && isLocallyPositiveSemidefinite(res.H))
        {
            res.status = FuncMinStatus::GradientConverged;
            return res;
        }

        const Eigen::VectorXd p = solveTrustRegionSubproblem(res.g, res.H, radius);
        const double pNorm = p.norm();
        const double predicted = -(res.g.dot(p) + 0.5 * p.dot(res.H * p));

        // Value at the trial point. Any failure here (exception or non-finite value) rejects the step: the iterate is
        // unchanged and the radius shrinks, as for a non-finite trial value.
        const Eigen::VectorXd xTrial = res.x + p;
        bool trialThrew = false;
        const double fTrial = evaluateContained(f, xTrial, nullptr, nullptr, trialThrew);
        if (trialThrew || !std::isfinite(fTrial))
        {
            radius = 0.25 * pNorm;
            if (radius <= opts.stepTolerance)
            {
                res.status = FuncMinStatus::NumericalFailure;
                return res;
            }
            continue;
        }
        const double actual = res.f - fTrial;

        double rho = 0.0;
        if (predicted > 0.0)
        {
            rho = actual / predicted;
        }
        else
        {
            rho = (actual >= 0.0) ? 1.0 : 0.0;
        }

        if (rho < 0.25)
        {
            radius = 0.25 * pNorm;
        }
        else if (rho > 0.75 && pNorm >= 0.99 * radius)
        {
            radius = std::min(2.0 * radius, opts.maxRadius);
        }

        bool accepted = rho > opts.acceptRatio;
        if (accepted)
        {
            // Derivatives at the accepted trial point are computed into temporaries and committed only on success, so a
            // failure here leaves the iterate, its value and its derivatives unchanged.
            Eigen::VectorXd gNew;
            Eigen::MatrixXd hNew;
            bool derivativesThrew = false;
            const double fNew = evaluateContained(f, xTrial, &gNew, &hNew, derivativesThrew);
            if (derivativesThrew || !std::isfinite(fNew) || !gNew.allFinite() || !hNew.allFinite())
            {
                radius = 0.25 * pNorm;
                if (radius <= opts.stepTolerance)
                {
                    res.status = FuncMinStatus::NumericalFailure;
                    return res;
                }
                continue;
            }
            res.x = xTrial;
            res.f = fNew;
            res.g = gNew;
            res.H = hNew;
        }

        if (pNorm <= opts.stepTolerance * (1.0 + res.x.norm()))
        {
            res.status = FuncMinStatus::StepConverged;
            return res;
        }
        if (std::abs(actual) <= opts.functionTolerance * (1.0 + std::abs(res.f))
            && std::abs(predicted) <= opts.functionTolerance * (1.0 + std::abs(res.f)))
        {
            res.status = FuncMinStatus::FunctionConverged;
            return res;
        }
    }

    res.iterations = opts.maxIterations;
    res.status = FuncMinStatus::MaxIterations;
    return res;
}

double newtonNorm(const Eigen::VectorXd& g, const Eigen::MatrixXd& H)
{
    if (g.size() == 0)
    {
        return 0.0;
    }
    if (H.rows() != g.size() || H.cols() != g.size() || !g.allFinite() || !H.allFinite())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::MatrixXd hs = 0.5 * (H + H.transpose());
    const Eigen::LLT<Eigen::MatrixXd> llt(hs);
    if (llt.info() != Eigen::Success)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // lambda^2 = g^T H^{-1} g = ||L^{-1} g||^2 with H = L L^T: one triangular solve, no inverse.
    const Eigen::VectorXd z = llt.matrixL().solve(g);
    return z.norm();
}
