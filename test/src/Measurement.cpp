#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementBase.h"
#include "../../src/OutdoorLaplaceUpdate.h"
#include "../../src/SystemBase.h"
#include "../../src/SystemEstimator.h"
#include "testUtil.h"

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using testutil::fdGradient;
using testutil::fdJacobian;
using testutil::nearMatrix;
using testutil::randomMatrix;
using testutil::randomSpd;
using testutil::randomVector;

namespace
{

/// Static model: no flow, no process noise. Used to test the measurement update in isolation.
class StaticModel : public SystemBase
{
public:
    VectorXd dynamics(double /*t*/, const VectorXd& x, MatrixXd* a) const override
    {
        if (a != nullptr)
        {
            a->setZero(x.size(), x.size());
        }
        return VectorXd::Zero(x.size());
    }
    std::vector<Index> stochasticIndices() const override { return {}; }
    MatrixXd processNoiseSqrtInfo(double /*dt*/, const VectorXd& /*x*/) const override { return MatrixXd(0, 0); }
};

/// Residual model h(x) = [x0^2 ; x0 x1 + sin x2] with its Jacobian and second derivatives.
VectorXd nonlinearH(const VectorXd& x, MatrixXd* j)
{
    VectorXd h(2);
    h << x(0) * x(0), x(0) * x(1) + std::sin(x(2));
    if (j != nullptr)
    {
        j->resize(2, x.size());
        j->setZero();
        (*j)(0, 0) = 2.0 * x(0);
        (*j)(1, 0) = x(1);
        (*j)(1, 1) = x(0);
        (*j)(1, 2) = std::cos(x(2));
    }
    return h;
}

/// Exact second-order term S = sum_j c_j Hess(h_j) with c = Lr^T r (the whitened residual weighting).
MatrixXd nonlinearSecondOrder(const VectorXd& x, const VectorXd& r, const MatrixXd& lr)
{
    const VectorXd c = lr.transpose() * r;
    MatrixXd s = MatrixXd::Zero(x.size(), x.size());
    s(0, 0) += c(0) * 2.0;
    s(0, 1) += c(1) * 1.0;
    s(1, 0) += c(1) * 1.0;
    s(2, 2) += c(1) * (-std::sin(x(2)));
    return s;
}

/// Nonlinear Gaussian measurement built on MeasurementGaussianLikelihood.
class NonlinearMeasurement : public MeasurementBase
{
public:
    NonlinearMeasurement(double t, VectorXd y, const MatrixXd& lr, bool exactHessian)
        : MeasurementBase(t, y), lr_(lr),
          lik_(nonlinearH, lr, std::move(y))
    {
        if (exactHessian)
        {
            lik_.setSecondOrderTerm([this](const VectorXd& x, const VectorXd& r)
                                    { return nonlinearSecondOrder(x, r, lr_); });
        }
    }
    double logLikelihood(const VectorXd& x, const SystemEstimator& /*system*/, VectorXd* g,
                         MatrixXd* h) const override
    {
        return lik_.logLikelihood(x, g, h);
    }

private:
    MatrixXd lr_;
    MeasurementGaussianLikelihood lik_;
};

/// Same measurement, declaring that it only depends on states 0..2 (used to test the restricted update).
class NonlinearMeasurementSupported : public NonlinearMeasurement
{
public:
    using NonlinearMeasurement::NonlinearMeasurement;
    std::vector<Index> supportIndices(const SystemEstimator& /*system*/) const override { return {0, 1, 2}; }
};

} // namespace

SCENARIO("MeasurementBase: linear-Gaussian Laplace update equals the Kalman information-filter update")
{
    GIVEN("a 3-D prior, a 2-D linear measurement y = H x + v with R, and a static system at the measurement time")
    {
        const MatrixXd p = randomSpd(3, 1101);
        const VectorXd mu = randomVector(3, 1102);
        const MatrixXd h = randomMatrix(2, 3, 1103);
        const MatrixXd r = randomSpd(2, 1104);
        const VectorXd y = randomVector(2, 1105);

        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        const MeasurementLinearGaussian meas(0.0, h, y, r);

        WHEN("the measurement is processed (Laplace update)")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the MAP is found and the posterior matches the Kalman covariance and mean")
            {
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(res.precisionIsExact);
                const MatrixXd pInv = p.inverse();
                const MatrixXd rInv = r.inverse();
                const MatrixXd pPost = (pInv + h.transpose() * rInv * h).inverse();
                const VectorXd muPost = pPost * (pInv * mu + h.transpose() * rInv * y);
                CAPTURE_EIGEN(sys.mean());
                CHECK(nearMatrix(sys.mean(), muPost, 1e-8));
                CHECK(nearMatrix(sys.covariance(), pPost, 1e-8));
                CHECK(nearMatrix(sys.state().infoVec(), pPost.inverse() * muPost, 1e-8));
            }
        }
    }
}

SCENARIO("MeasurementBase: a measurement older than the system time is rejected")
{
    GIVEN("a system at t = 2 and a measurement at t = 1")
    {
        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(VectorXd::Zero(1), MatrixXd::Identity(1, 1)), 2.0);
        const MeasurementLinearGaussian meas(1.0, MatrixXd::Identity(1, 1), VectorXd::Zero(1), MatrixXd::Identity(1, 1));
        THEN("process throws")
        {
            CHECK_THROWS_AS(meas.process(sys), std::invalid_argument);
        }
    }
}

SCENARIO("MeasurementBase: a restricted support gives the same posterior as the full-state Laplace update")
{
    GIVEN("a correlated 5-D prior and a measurement that only observes states 0..2")
    {
        const MatrixXd p = randomSpd(5, 1401);
        const VectorXd mu = randomVector(5, 1402) * 0.3;
        MatrixXd lr(2, 2);
        lr << 1.1, 0.2,
              0.0, 0.7;
        const VectorXd y = randomVector(2, 1403);

        auto model = std::make_shared<StaticModel>();
        SystemEstimator full(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        SystemEstimator reduced(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        NonlinearMeasurement measFull(0.0, y, lr, true);
        NonlinearMeasurementSupported measReduced(0.0, y, lr, true);

        WHEN("both updates are applied")
        {
            const MeasurementUpdateResult rf = measFull.process(full);
            const MeasurementUpdateResult rr = measReduced.process(reduced);
            THEN("the MAP estimates agree and the posterior information and mean are identical")
            {
                CHECK(rr.map.status != FuncMinStatus::NumericalFailure);
                CHECK(nearMatrix(rr.map.x, rf.map.x.head(3), 1e-7));
                CHECK(nearMatrix(reduced.mean(), full.mean(), 1e-7));
                CHECK(nearMatrix(reduced.state().information(), full.state().information(), 1e-8));
                CHECK(nearMatrix(reduced.state().infoVec(), full.state().infoVec(), 1e-8));
            }
        }
    }
}

SCENARIO("MeasurementGaussianLikelihood: gradient and Hessian match finite differences")
{
    GIVEN("a nonlinear residual model with noise sqrt-information Lr and observation y")
    {
        MatrixXd lr(2, 2);
        lr << 2.0, 0.3,
              0.0, 1.5;
        const VectorXd y = randomVector(2, 1201);
        MeasurementGaussianLikelihood gaussGn(nonlinearH, lr, y);
        MeasurementGaussianLikelihood gaussExact(nonlinearH, lr, y);
        gaussExact.setSecondOrderTerm([&lr](const VectorXd& x, const VectorXd& r)
                                      { return nonlinearSecondOrder(x, r, lr); });

        for (unsigned s = 0; s < 4; ++s)
        {
            const VectorXd x = randomVector(3, 1210 + s);
            INFO("sample ", s, " x = ", x.transpose());
            WHEN("the gradient and Hessian are evaluated")
            {
                VectorXd g;
                MatrixXd hGn;
                MatrixXd hEx;
                const double value = gaussGn.logLikelihood(x, &g, &hGn);
                gaussExact.logLikelihood(x, nullptr, &hEx);
                THEN("the value is consistent with the closed form")
                {
                    const VectorXd rr = lr * (nonlinearH(x, nullptr) - y);
                    const double ref = -0.5 * rr.squaredNorm() + std::log(2.0) + std::log(1.5)
                                     - std::log(2.0 * std::numbers::pi);
                    CHECK(value == doctest::Approx(ref).epsilon(1e-12));
                }
                THEN("the gradient matches central finite differences")
                {
                    const VectorXd gFd = fdGradient([&](const VectorXd& xx) { return gaussGn.logLikelihood(xx, nullptr, nullptr); }, x);
                    CAPTURE_EIGEN(g);
                    CAPTURE_EIGEN(gFd);
                    CHECK(nearMatrix(g, gFd, 1e-6));
                }
                THEN("the exact Hessian matches finite differences of the gradient")
                {
                    const MatrixXd hFd = fdJacobian(
                        [&](const VectorXd& xx)
                        {
                            VectorXd gg;
                            gaussExact.logLikelihood(xx, &gg, nullptr);
                            return gg;
                        },
                        x);
                    CAPTURE_EIGEN(hEx);
                    CAPTURE_EIGEN(hFd);
                    CHECK(nearMatrix(hEx, hFd, 1e-5));
                }
                THEN("the Gauss-Newton Hessian is the exact Hessian without the second-order term")
                {
                    const MatrixXd s2 = nonlinearSecondOrder(x, lr * (nonlinearH(x, nullptr) - y), lr);
                    CHECK(nearMatrix(hGn, hEx + s2, 1e-10));
                }
            }
        }
    }
}

SCENARIO("MeasurementBase: Laplace posterior is stationary at the MAP and has the Hessian-based information")
{
    GIVEN("a 3-D Gaussian prior and the nonlinear measurement above")
    {
        const MatrixXd p = randomSpd(3, 1301);
        const VectorXd mu = randomVector(3, 1302) * 0.3;
        MatrixXd lr(2, 2);
        lr << 1.2, 0.1,
              0.0, 0.8;
        const VectorXd y = randomVector(2, 1303);

        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        NonlinearMeasurement meas(0.0, y, lr, true);
        const GaussianInfo<double> prior = sys.state();

        WHEN("the measurement is processed")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the exact second-order curvature is indefinite at the MAP, so the documented explicit fallback is flagged")
            {
                VectorXd gAtMap;
                MatrixXd hAtMap;
                meas.logLikelihood(sys.mean(), sys, &gAtMap, &hAtMap);
                const Eigen::SelfAdjointEigenSolver<MatrixXd> esK(-hAtMap);
                CAPTURE_EIGEN(esK.eigenvalues());
                CHECK(esK.eigenvalues().minCoeff() < -1e-6);
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(!res.precisionIsExact);
            }
            THEN("the posterior mean is the MAP estimate (zero gradient of the negative log posterior)")
            {
                const VectorXd xs = sys.mean();
                VectorXd gll;
                meas.logLikelihood(xs, sys, &gll, nullptr);
                VectorXd gp;
                prior.logPdf(xs, &gp, nullptr);
                CAPTURE_EIGEN(gll);
                CAPTURE_EIGEN(gp);
                CHECK((gll + gp).cwiseAbs().maxCoeff() < 1e-6);
            }
            THEN("the posterior information equals prior information plus the negative Hessian of log p(y|x*)")
            {
                VectorXd gll;
                MatrixXd hll;
                meas.logLikelihood(sys.mean(), sys, &gll, &hll);
                const MatrixXd expected = prior.information() - hll;
                CAPTURE_EIGEN(sys.state().information());
                CAPTURE_EIGEN(expected);
                CHECK(nearMatrix(sys.state().information(), expected, 1e-7));
            }
        }
    }
}

namespace
{

/// Likelihood -0.5 x0^2 that throws std::domain_error outside |x0| <= domain (a domain-limited model, as the flow).
class DomainLimitedMeasurement : public MeasurementBase
{
public:
    DomainLimitedMeasurement(double t, double domain) : MeasurementBase(t, VectorXd::Zero(1)), domain_(domain) {}
    double logLikelihood(const VectorXd& x, const SystemEstimator& /*system*/, VectorXd* g, MatrixXd* h) const override
    {
        if (std::abs(x(0)) > domain_)
        {
            throw std::domain_error("DomainLimitedMeasurement: outside the model domain");
        }
        if (g != nullptr)
        {
            g->setZero(x.size());
            (*g)(0) = -x(0);
        }
        if (h != nullptr)
        {
            h->setZero(x.size(), x.size());
            (*h)(0, 0) = -1.0;
        }
        return -0.5 * x(0) * x(0);
    }
    std::vector<Index> supportIndices(const SystemEstimator& /*system*/) const override { return {0}; }

private:
    double domain_;
};

/// A likelihood that always throws (a broken model): must be contained, never escape process().
class AlwaysThrowingMeasurement : public MeasurementBase
{
public:
    explicit AlwaysThrowingMeasurement(double t) : MeasurementBase(t, VectorXd::Zero(1)) {}
    double logLikelihood(const VectorXd& /*x*/, const SystemEstimator& /*system*/, VectorXd* /*g*/,
                         MatrixXd* /*h*/) const override
    {
        throw std::runtime_error("AlwaysThrowingMeasurement: no likelihood");
    }
};

/// Asserts that two Gaussian states are the same density (information vector and square-root information).
void requireSameState(const GaussianInfo<double>& a, const GaussianInfo<double>& b)
{
    REQUIRE(a.dim() == b.dim());
    CHECK(nearMatrix(a.infoVec(), b.infoVec(), 1e-14));
    CHECK(nearMatrix(a.sqrtInfo(), b.sqrtInfo(), 1e-14));
}

} // namespace

SCENARIO("MAP status policy (R4-01): a converged MAP is applied and classified Converged")
{
    GIVEN("a 3-D prior and a linear-Gaussian measurement of two of the three states")
    {
        const MatrixXd p = randomSpd(3, 4101);
        const VectorXd mu = randomVector(3, 4102);
        const MatrixXd h = randomMatrix(2, 3, 4103);
        const MatrixXd r = randomSpd(2, 4104);
        const VectorXd y = randomVector(2, 4105);
        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        const MeasurementLinearGaussian meas(0.0, h, y, r);
        WHEN("it is processed with the default options")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the outcome is Converged, nothing is skipped and the posterior is the Kalman posterior")
            {
                CHECK(res.outcome == MapOutcome::Converged);
                CHECK(!res.updateSkipped);
                CHECK(res.precisionIsExact);
                CHECK(res.curvatureRank >= 1);
                CHECK(res.newtonNorm <= 1e-6);
                const MatrixXd pInv = p.inverse();
                const MatrixXd pPost = (pInv + h.transpose() * r.inverse() * h).inverse();
                CHECK(nearMatrix(sys.mean(), pPost * (pInv * mu + h.transpose() * r.inverse() * y), 1e-8));
            }
        }
    }
}

SCENARIO("MAP status policy (R4-01): MaxIterations is applied only within the Newton-norm bound")
{
    GIVEN("a linear-Gaussian measurement; one exact Newton step reaches the MAP but the iteration limit is 1")
    {
        const MatrixXd p = randomSpd(3, 4201);
        const VectorXd mu = randomVector(3, 4202);
        const MatrixXd h = randomMatrix(2, 3, 4203);
        const MatrixXd r = randomSpd(2, 4204);
        const VectorXd y = randomVector(2, 4205);
        auto model = std::make_shared<StaticModel>();
        const MeasurementLinearGaussian meas(0.0, h, y, r);
        FuncMinOptions opts;
        opts.maxIterations = 1;
        opts.initialRadius = 1e6;
        WHEN("the update is processed")
        {
            SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
            const MeasurementUpdateResult res = meas.process(sys, opts);
            THEN("MaxIterations is reported at a stationary point and the posterior is applied")
            {
                CHECK(res.map.status == FuncMinStatus::MaxIterations);
                CHECK(res.outcome == MapOutcome::AcceptedMaxIterations);
                CHECK(!res.updateSkipped);
                CHECK(res.newtonNorm <= 0.05);
                const MatrixXd pInv = p.inverse();
                const MatrixXd pPost = (pInv + h.transpose() * r.inverse() * h).inverse();
                CHECK(nearMatrix(sys.mean(), pPost * (pInv * mu + h.transpose() * r.inverse() * y), 1e-8));
            }
        }
        WHEN("the policy bound is set to zero")
        {
            SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
            MapAcceptancePolicy strict;
            strict.maxIterationsNewtonNorm = 0.0;
            const GaussianInfo<double> before = sys.state();
            const MeasurementUpdateResult res = meas.process(sys, opts, strict);
            THEN("the same MaxIterations result is skipped and the prediction is kept")
            {
                CHECK(res.outcome == MapOutcome::SkippedMaxIterations);
                CHECK(res.updateSkipped);
                CHECK(!res.precisionIsExact);
                requireSameState(sys.state(), before);
            }
        }
    }
    GIVEN("a nonlinear measurement with a prior far from the data and an iteration limit of 1 with radius 1")
    {
        const MatrixXd p = MatrixXd::Identity(3, 3);
        const VectorXd mu = VectorXd::Constant(3, 5.0);
        MatrixXd lr(2, 2);
        lr << 1.2, 0.1,
              0.0, 0.8;
        const VectorXd y = randomVector(2, 4301);
        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        const GaussianInfo<double> before = sys.state();
        NonlinearMeasurement meas(0.0, y, lr, false);
        FuncMinOptions opts;
        opts.maxIterations = 1;
        opts.initialRadius = 1.0;
        WHEN("it is processed with the default policy")
        {
            const MeasurementUpdateResult res = meas.process(sys, opts);
            THEN("the non-converged MAP is skipped, the prediction is kept and the skip is flagged")
            {
                CHECK(res.map.status == FuncMinStatus::MaxIterations);
                CHECK(res.outcome == MapOutcome::SkippedMaxIterations);
                CHECK(res.updateSkipped);
                CHECK(!res.precisionIsExact);
                CAPTURE(res.newtonNorm);
                CHECK(res.newtonNorm > 0.05);
                requireSameState(sys.state(), before);
            }
            THEN("the counters record one skipped MaxIterations update")
            {
                MapStatusCounts counts;
                counts.add(res);
                CHECK(counts.updates == 1);
                CHECK(counts.skippedMaxIterations == 1);
                CHECK(counts.skipped() == 1);
                CHECK(counts.precisionNotExact == 1);
            }
        }
        WHEN("the same update is processed with a very permissive bound")
        {
            MapAcceptancePolicy loose;
            loose.maxIterationsNewtonNorm = 1e12;
            const MeasurementUpdateResult res = meas.process(sys, opts, loose);
            THEN("the bound alone decides: the unconverged point is applied and flagged AcceptedMaxIterations")
            {
                CHECK(res.outcome == MapOutcome::AcceptedMaxIterations);
                CHECK(!res.updateSkipped);
            }
        }
    }
}

SCENARIO("MAP status policy (R4-01): decideMapOutcome maps every status and rejects an invalid bound")
{
    MapAcceptancePolicy policy;
    policy.maxIterationsNewtonNorm = 0.05;
    CHECK(decideMapOutcome(FuncMinStatus::GradientConverged, 1e9, policy) == MapOutcome::Converged);
    CHECK(decideMapOutcome(FuncMinStatus::StepConverged, 1e9, policy) == MapOutcome::Converged);
    CHECK(decideMapOutcome(FuncMinStatus::FunctionConverged, 1e9, policy) == MapOutcome::Converged);
    CHECK(decideMapOutcome(FuncMinStatus::MaxIterations, 0.01, policy) == MapOutcome::AcceptedMaxIterations);
    CHECK(decideMapOutcome(FuncMinStatus::MaxIterations, 0.2, policy) == MapOutcome::SkippedMaxIterations);
    CHECK(decideMapOutcome(FuncMinStatus::MaxIterations, std::numeric_limits<double>::quiet_NaN(), policy) ==
          MapOutcome::SkippedMaxIterations);
    CHECK(decideMapOutcome(FuncMinStatus::NumericalFailure, 0.0, policy) == MapOutcome::SkippedNumericalFailure);
    MapAcceptancePolicy bad;
    bad.maxIterationsNewtonNorm = -1.0;
    CHECK_THROWS_AS(decideMapOutcome(FuncMinStatus::MaxIterations, 0.0, bad), std::invalid_argument);
}

SCENARIO("MAP status counters (R4-01) accumulate every outcome and the exactness flag")
{
    MapStatusCounts counts;
    counts.add(MapOutcome::Converged, true);
    counts.add(MapOutcome::AcceptedMaxIterations, true);
    counts.add(MapOutcome::SkippedMaxIterations, false);
    counts.add(MapOutcome::SkippedNumericalFailure, false);
    counts.add(MapOutcome::SkippedNoCurvature, false);
    CHECK(counts.updates == 5);
    CHECK(counts.converged == 1);
    CHECK(counts.acceptedMaxIterations == 1);
    CHECK(counts.skippedMaxIterations == 1);
    CHECK(counts.skippedNumericalFailure == 1);
    CHECK(counts.skippedNoCurvature == 1);
    CHECK(counts.skipped() == 3);
    CHECK(counts.precisionNotExact == 3);
}

SCENARIO("MeasurementBase (R4-02): an exception in the likelihood is contained and leaves the state unchanged")
{
    GIVEN("a measurement whose likelihood always throws")
    {
        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(VectorXd::Constant(1, 0.3), MatrixXd::Identity(1, 1)), 0.0);
        const GaussianInfo<double> before = sys.state();
        const AlwaysThrowingMeasurement meas(0.0);
        WHEN("it is processed")
        {
            MeasurementUpdateResult res;
            CHECK_NOTHROW(res = meas.process(sys));
            THEN("the outcome is SkippedNumericalFailure, the update is flagged and the prior is kept")
            {
                CHECK(res.outcome == MapOutcome::SkippedNumericalFailure);
                CHECK(res.map.status == FuncMinStatus::NumericalFailure);
                CHECK(res.updateSkipped);
                CHECK(!res.precisionIsExact);
                requireSameState(sys.state(), before);
            }
        }
    }
    GIVEN("a measurement that is only defined for |x0| <= 1, with a prior centred at 0.9")
    {
        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(VectorXd::Constant(1, 0.9), MatrixXd::Identity(1, 1)), 0.0);
        const DomainLimitedMeasurement meas(0.0, 1.0);
        WHEN("it is processed")
        {
            MeasurementUpdateResult res;
            CHECK_NOTHROW(res = meas.process(sys));
            THEN("the trust region never commits an out-of-domain point and the posterior mean is the analytic MAP 0.45")
            {
                CHECK(std::isfinite(sys.mean()(0)));
                CHECK(std::abs(sys.mean()(0)) <= 1.0);
                CHECK(sys.mean()(0) == doctest::Approx(0.45).epsilon(1e-6));
                CHECK(res.outcome == MapOutcome::Converged);
            }
        }
    }
}

SCENARIO("MeasurementBase (R4-16): a measurement with no curvature is skipped and flagged, not silent")
{
    GIVEN("an observation with a zero Jacobian (no information about the state)")
    {
        auto model = std::make_shared<StaticModel>();
        const MatrixXd p = randomSpd(3, 4501);
        const VectorXd mu = randomVector(3, 4502);
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        const GaussianInfo<double> before = sys.state();
        const MeasurementLinearGaussian meas(0.0, MatrixXd::Zero(3, 3), randomVector(3, 4503), MatrixXd::Identity(3, 3));
        WHEN("it is processed")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the update is skipped: state unchanged, outcome SkippedNoCurvature, updateSkipped and precisionIsExact false")
            {
                CHECK(res.outcome == MapOutcome::SkippedNoCurvature);
                CHECK(res.updateSkipped);
                CHECK(!res.precisionIsExact);
                CHECK(res.curvatureRank == 0);
                requireSameState(sys.state(), before);
            }
        }
    }
    GIVEN("a uniform but tiny information (sigma = 1e6, information 1e-12 per axis, full rank)")
    {
        auto model = std::make_shared<StaticModel>();
        const MatrixXd p = randomSpd(3, 4511);
        const VectorXd mu = randomVector(3, 4512);
        const VectorXd y = randomVector(3, 4513);
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        const MeasurementLinearGaussian meas(0.0, MatrixXd::Identity(3, 3), y, 1e12 * MatrixXd::Identity(3, 3));
        WHEN("it is processed")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("it is applied exactly (Cholesky path: the test is relative, so it is not dropped) and matches the Kalman posterior")
            {
                CHECK(res.outcome == MapOutcome::Converged);
                CHECK(!res.updateSkipped);
                CHECK(res.precisionIsExact);
                CHECK(res.curvatureRank == 3);
                const MatrixXd pInv = p.inverse();
                const MatrixXd pPost = (pInv + 1e-12 * MatrixXd::Identity(3, 3)).inverse();
                CHECK(nearMatrix(sys.mean(), pPost * (pInv * mu + 1e-12 * y), 1e-6));
            }
        }
    }
}

SCENARIO("outdoorLaplaceUpdate honours the same MAP status policy as MeasurementBase::process")
{
    GIVEN("a linear-Gaussian measurement and a one-iteration limit with an exact first step")
    {
        const MatrixXd p = randomSpd(3, 4601);
        const VectorXd mu = randomVector(3, 4602);
        const MatrixXd h = randomMatrix(2, 3, 4603);
        const MatrixXd r = randomSpd(2, 4604);
        const VectorXd y = randomVector(2, 4605);
        auto model = std::make_shared<StaticModel>();
        const MeasurementLinearGaussian meas(0.0, h, y, r);
        FuncMinOptions opts;
        opts.maxIterations = 1;
        opts.initialRadius = 1e6;
        WHEN("the update is applied")
        {
            SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
            const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, sys, opts);
            THEN("AcceptedMaxIterations is applied and matches the Kalman posterior")
            {
                CHECK(rep.map.status == FuncMinStatus::MaxIterations);
                CHECK(rep.outcome == MapOutcome::AcceptedMaxIterations);
                CHECK(!rep.updateSkipped);
                const MatrixXd pInv = p.inverse();
                const MatrixXd pPost = (pInv + h.transpose() * r.inverse() * h).inverse();
                CHECK(nearMatrix(sys.mean(), pPost * (pInv * mu + h.transpose() * r.inverse() * y), 1e-8));
            }
        }
        WHEN("the policy bound is zero")
        {
            SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
            const GaussianInfo<double> before = sys.state();
            MapAcceptancePolicy strict;
            strict.maxIterationsNewtonNorm = 0.0;
            // The exact step leaves a tiny but positive Newton norm, so the strict bound skips it.
            const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, sys, opts, strict);
            THEN("the update is skipped and the prediction is kept")
            {
                if (rep.newtonNorm > 0.0)
                {
                    CHECK(rep.outcome == MapOutcome::SkippedMaxIterations);
                    CHECK(rep.updateSkipped);
                    requireSameState(sys.state(), before);
                }
                else
                {
                    CHECK(rep.outcome == MapOutcome::AcceptedMaxIterations);
                }
            }
        }
    }
    GIVEN("a far prior and a one-iteration limit with radius 1 (the MAP has not converged)")
    {
        const MatrixXd p = MatrixXd::Identity(3, 3);
        const VectorXd mu = VectorXd::Constant(3, 5.0);
        MatrixXd lr(2, 2);
        lr << 1.2, 0.1,
              0.0, 0.8;
        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(mu, p), 0.0);
        const GaussianInfo<double> before = sys.state();
        NonlinearMeasurement meas(0.0, randomVector(2, 4611), lr, false);
        FuncMinOptions opts;
        opts.maxIterations = 1;
        opts.initialRadius = 1.0;
        WHEN("it is applied")
        {
            const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, sys, opts);
            THEN("the non-converged MAP is skipped and flagged")
            {
                CHECK(rep.outcome == MapOutcome::SkippedMaxIterations);
                CHECK(rep.updateSkipped);
                requireSameState(sys.state(), before);
            }
        }
    }
    GIVEN("a measurement whose likelihood always throws")
    {
        auto model = std::make_shared<StaticModel>();
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(VectorXd::Constant(1, 0.3), MatrixXd::Identity(1, 1)), 0.0);
        const GaussianInfo<double> before = sys.state();
        const AlwaysThrowingMeasurement meas(0.0);
        WHEN("it is applied")
        {
            OutdoorUpdateReport rep;
            CHECK_NOTHROW(rep = outdoorLaplaceUpdate(meas, sys));
            THEN("it is skipped as a numerical failure, with no state change and no exception")
            {
                CHECK(rep.outcome == MapOutcome::SkippedNumericalFailure);
                CHECK(rep.updateSkipped);
                requireSameState(sys.state(), before);
            }
        }
    }
    GIVEN("a measurement with a zero Jacobian")
    {
        auto model = std::make_shared<StaticModel>();
        const MatrixXd p = randomSpd(3, 4621);
        SystemEstimator sys(model, GaussianInfo<double>::fromMoment(randomVector(3, 4622), p), 0.0);
        const GaussianInfo<double> before = sys.state();
        const MeasurementLinearGaussian meas(0.0, MatrixXd::Zero(3, 3), randomVector(3, 4623), MatrixXd::Identity(3, 3));
        WHEN("it is applied")
        {
            const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, sys);
            THEN("the zero-information update is flagged SkippedNoCurvature")
            {
                CHECK(rep.outcome == MapOutcome::SkippedNoCurvature);
                CHECK(rep.updateSkipped);
                requireSameState(sys.state(), before);
            }
        }
    }
}
