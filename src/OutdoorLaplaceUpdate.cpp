#include "OutdoorLaplaceUpdate.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>
#include <Eigen/Eigenvalues>
#include "GaussianInfo.h"
#include "funcmin.hpp"

OutdoorUpdateReport outdoorLaplaceUpdate(const MeasurementBase& measurement, SystemEstimator& system,
                                         const FuncMinOptions& opts, const MapAcceptancePolicy& policy)
{
    if (measurement.time() < system.time())
    {
        throw std::invalid_argument("outdoorLaplaceUpdate: measurement is older than the system time");
    }
    system.predict(measurement.time());

    const SystemEstimator::State prior = system.state();
    const Eigen::Index n = prior.dim();

    std::vector<Eigen::Index> support = measurement.supportIndices(system);
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
                throw std::invalid_argument("outdoorLaplaceUpdate: invalid measurement support");
            }
            seen[static_cast<std::size_t>(idx)] = true;
        }
    }
    const Eigen::Index s = static_cast<Eigen::Index>(support.size());

    // Exact marginal of the support (the likelihood does not depend on the other states).
    const SystemEstimator::State priorS = prior.marginal(support);
    const Eigen::VectorXd xRef = prior.mean();

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

    const SystemEstimator& sysRef = system;
    FuncMinObjective objective = [&](const Eigen::VectorXd& xs, Eigen::VectorXd* g, Eigen::MatrixXd* h) -> double
    {
        const Eigen::VectorXd x = embed(xs);
        Eigen::VectorXd gl;
        Eigen::MatrixXd hl;
        const double ll = measurement.logLikelihood(x, sysRef, g != nullptr ? &gl : nullptr, h != nullptr ? &hl : nullptr);
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

    OutdoorUpdateReport report;
    report.map = trustRegionMinimise(objective, priorS.mean(), opts);

    // MAP status policy (identical to MeasurementBase::process). A failed MAP leaves the prediction in place.
    report.gradientNormInf = report.map.g.size() > 0 ? report.map.g.cwiseAbs().maxCoeff() : 0.0;
    report.newtonNorm = (report.map.status == FuncMinStatus::NumericalFailure) ? std::numeric_limits<double>::quiet_NaN()
                                                                               : newtonNorm(report.map.g, report.map.H);
    report.outcome = decideMapOutcome(report.map.status, report.newtonNorm, policy);
    if (mapOutcomeSkipsUpdate(report.outcome))
    {
        report.updateSkipped = true;
        if (report.outcome == MapOutcome::SkippedNumericalFailure)
        {
            report.map.status = FuncMinStatus::NumericalFailure;
        }
        return report;
    }

    const Eigen::VectorXd xsStar = report.map.x;
    const Eigen::VectorXd xStar = embed(xsStar);

    // Laplace increment about the MAP on the support: K_S = -Hess, kappa_S = grad + K_S x*_S.
    Eigen::VectorXd gll;
    Eigen::MatrixXd hll;
    try
    {
        measurement.logLikelihood(xStar, sysRef, &gll, &hll);
    }
    catch (const std::exception&)
    {
        report.outcome = MapOutcome::SkippedNumericalFailure;
        report.map.status = FuncMinStatus::NumericalFailure;
        report.updateSkipped = true;
        return report;
    }
    Eigen::MatrixXd kS = -restrictMat(hll);
    kS = 0.5 * (kS + kS.transpose());
    const Eigen::VectorXd kappaS = restrictVec(gll) + kS * xsStar;
    if (!kS.allFinite() || !kappaS.allFinite())
    {
        report.outcome = MapOutcome::SkippedNumericalFailure;
        report.map.status = FuncMinStatus::NumericalFailure;
        report.updateSkipped = true;
        return report;
    }

    // Eigen-decomposition of the (PSD for Gauss-Newton) curvature: K_S = V diag(lambda) V^T.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(kS);
    if (es.info() != Eigen::Success)
    {
        report.outcome = MapOutcome::SkippedNumericalFailure;
        report.map.status = FuncMinStatus::NumericalFailure;
        report.updateSkipped = true;
        return report;
    }
    const Eigen::VectorXd lambda = es.eigenvalues();
    const Eigen::MatrixXd V = es.eigenvectors();
    const double scale = std::max(1.0, lambda.cwiseAbs().maxCoeff());
    const double tol = 1e-12 * scale;

    std::vector<Eigen::Index> pos;
    for (Eigen::Index i = 0; i < s; ++i)
    {
        if (lambda(i) > tol)
        {
            pos.push_back(i);
        }
        else if (lambda(i) < -tol)
        {
            report.droppedNegativeMass += -lambda(i);
        }
    }
    const Eigen::Index r = static_cast<Eigen::Index>(pos.size());
    report.rank = r;

    if (r == 0)
    {
        // No curvature information: the update is skipped and flagged (the posterior would equal the prior).
        report.outcome = MapOutcome::SkippedNoCurvature;
        report.updateSkipped = true;
        report.droppedGradientNorm = kappaS.norm();
        system.setState(prior);
        return report;
    }

    // F (r x s) with F^T F = sum over positive eigenpairs = K_S (projected), and w with F^T w = projection of kappa_S.
    Eigen::MatrixXd f(r, s);
    Eigen::VectorXd w(r);
    Eigen::VectorXd projected = Eigen::VectorXd::Zero(s);
    for (Eigen::Index k = 0; k < r; ++k)
    {
        const Eigen::Index i = pos[static_cast<std::size_t>(k)];
        const double sq = std::sqrt(lambda(i));
        f.row(k) = sq * V.col(i).transpose();
        const double coeff = V.col(i).dot(kappaS) / sq;   // (1/sqrt(lambda)) v^T kappa
        w(k) = coeff;
        projected += V.col(i) * V.col(i).dot(kappaS);
    }
    report.droppedGradientNorm = (kappaS - projected).norm();

    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(r, n);
    for (Eigen::Index j = 0; j < s; ++j)
    {
        a.col(support[static_cast<std::size_t>(j)]) = f.col(j);
    }
    system.setState(prior.informationUpdate(a, w));
    return report;
}
