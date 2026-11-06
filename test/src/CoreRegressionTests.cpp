// Regression tests for the estimation-core defects found in the Scenario 4 run and the review of T-A2-001/M3-COREFIX:
//   C1  zero-length time steps (|dt| ~ 1e-16 s) were integrated with 1/sqrt(dt) (1/dt^3 in information) process noise;
//   C2  MeasurementBase::process left the square-root form for rank-deficient (Gauss-Newton) curvature.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/funcmin.hpp"
#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementBase.h"
#include "../../src/SystemBase.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using GaussianD = GaussianInfo<double>;
using testutil::fdGradient;
using testutil::nearMatrix;
using testutil::randomMatrix;
using testutil::randomSpd;
using testutil::randomVector;

namespace
{

Eigen::Matrix<double, 6, 1> corePsd()
{
    Eigen::Matrix<double, 6, 1> q;
    q << 0.04, 0.03, 0.05, 0.01, 0.02, 0.015;
    return q;
}

/// Static model (no flow, no process noise): isolates the measurement update.
class CoreStaticModel : public SystemBase
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

/// Scalar residual h(x) = x0 * x1 (rank-one Gauss-Newton curvature on a 3-D support) with optional exact second order.
VectorXd productH(const VectorXd& x, MatrixXd* j)
{
    VectorXd h(1);
    h(0) = x(0) * x(1);
    if (j != nullptr)
    {
        j->setZero(1, x.size());
        (*j)(0, 0) = x(1);
        (*j)(0, 1) = x(0);
    }
    return h;
}

/// Exact second-order term of the scalar residual: S = sum_k r_k Hess(r_k) = r * Hess(x0 x1) (Lr = 1).
MatrixXd productSecondOrder(const VectorXd& x, const VectorXd& r)
{
    MatrixXd s = MatrixXd::Zero(x.size(), x.size());
    s(0, 1) = r(0);
    s(1, 0) = r(0);
    return s;
}

class ProductMeasurement : public MeasurementBase
{
public:
    ProductMeasurement(double t, double y, double sigma, bool exactHessian)
        : MeasurementBase(t, VectorXd::Constant(1, y)),
          lik_(productH, MatrixXd::Constant(1, 1, 1.0 / sigma), VectorXd::Constant(1, y))
    {
        if (exactHessian)
        {
            lik_.setSecondOrderTerm(productSecondOrder);
        }
    }
    double logLikelihood(const VectorXd& x, const SystemEstimator& /*system*/, VectorXd* g, MatrixXd* h) const override
    {
        return lik_.logLikelihood(x, g, h);
    }

private:
    MeasurementGaussianLikelihood lik_;
};

/// Moment-form Kalman reference for y = H x + v (explicit covariance algebra; test-side reference only).
void kalmanReference(const VectorXd& mu, const MatrixXd& p, const MatrixXd& h, const MatrixXd& r, const VectorXd& y,
                     VectorXd& muPost, MatrixXd& pPost)
{
    const MatrixXd s = h * p * h.transpose() + r;
    const MatrixXd k = (s.ldlt().solve(h * p)).transpose();   // K = P H^T S^{-1}
    muPost = mu + k * (y - h * mu);
    pPost = p - k * h * p;
    pPost = 0.5 * (pPost + pPost.transpose());
}

bool isUpperTriangularFinite(const MatrixXd& m)
{
    return m.allFinite() && m.triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0);
}

} // namespace

SCENARIO("C1: a zero-length time step (dt = 1e-16 s) leaves the information factor unchanged to rounding")
{
    GIVEN("a visual-navigation state with a point landmark and a nontrivial correlated density at t = 0")
    {
        auto model = std::make_shared<SystemVisualNav>(corePsd());
        VectorXd mu0(12);
        mu0 << 0.5, 0.1, -0.2, 0.05, -0.03, 0.02, 1.0, 2.0, -3.0, 0.1, -0.2, 0.4;
        SystemEstimator est(model, GaussianD::fromMoment(mu0, 0.05 * randomSpd(12, 7101)), 0.0);
        model->addLandmark(est, SystemVisualNav::LandmarkType::Point,
                           GaussianD::fromMoment(Vector3d(4.0, 1.0, 0.0), 0.5 * Matrix3d::Identity()));
        const MatrixXd xi0 = est.state().sqrtInfo();
        const VectorXd mean0 = est.mean();
        WHEN("a plain prediction to t = 1.1e-16 s is applied (the failure of the Scenario 4 run)")
        {
            model->predictPlain(est, 1.1e-16);
            THEN("the factor and the mean are unchanged to rounding, and the time label advances")
            {
                CHECK(nearMatrix(est.state().sqrtInfo(), xi0, 1e-12));
                CHECK(nearMatrix(est.mean(), mean0, 1e-10));
                CHECK(est.time() == 1.1e-16);
                CHECK(est.state().sqrtInfo().diagonal().cwiseAbs().minCoeff() > 1e-3);
            }
        }
        WHEN("a thousand zero-length steps are applied (no accumulation)")
        {
            for (int k = 1; k <= 1000; ++k)
            {
                model->predictPlain(est, 1.0e-16 * static_cast<double>(k));
            }
            THEN("the factor is still unchanged to rounding and the state stays finite")
            {
                CHECK(nearMatrix(est.state().sqrtInfo(), xi0, 1e-10));
                CHECK(est.mean().allFinite());
                CHECK(est.state().sqrtInfo().allFinite());
            }
        }
    }
}

SCENARIO("C1: the process-noise factor refuses zero-length steps")
{
    GIVEN("the visual-navigation model and a state")
    {
        const SystemVisualNav model(corePsd());
        VectorXd x = VectorXd::Zero(12);
        x.segment<3>(9) = Vector3d(0.1, -0.2, 0.4);
        THEN("dt <= kZeroTimeStep is rejected")
        {
            CHECK_THROWS_AS(model.processNoiseSqrtInfo(1.1e-16, x), std::invalid_argument);
            CHECK_THROWS_AS(model.processNoiseSqrtInfo(kZeroTimeStep, x), std::invalid_argument);
            CHECK_THROWS_AS(model.processNoiseSqrtInfo(-1e-3, x), std::invalid_argument);
        }
        THEN("a step just above the threshold gives a finite factor")
        {
            const MatrixXd l = model.processNoiseSqrtInfo(2.0 * kZeroTimeStep, x);
            CHECK(l.allFinite());
            CHECK(isUpperTriangularFinite(l));
        }
    }
}

SCENARIO("C1: a shifted prediction with a zero-length step is rejected by the generic estimator")
{
    GIVEN("an estimator over the visual-navigation model")
    {
        auto model = std::make_shared<SystemVisualNav>(corePsd());
        SystemEstimator est(model, GaussianD::fromMoment(VectorXd::Zero(12), 0.1 * MatrixXd::Identity(12, 12)), 0.0);
        std::vector<Index> keep;
        for (Index i = 0; i < 12; ++i)
        {
            keep.push_back(i);
        }
        THEN("the non-plain selection at dt = 1e-12 is rejected (the model owns the zero-step shift)")
        {
            CHECK_THROWS_AS(est.predict(1e-12, keep), std::invalid_argument);
        }
        THEN("a time that decreases by more than the tolerance is rejected, one within it is a no-op")
        {
            est.predict(1.0);
            CHECK_THROWS_AS(est.predict(1.0 - 1e-6), std::invalid_argument);
            CHECK_NOTHROW(est.predict(1.0 - 1e-16));
            CHECK(est.time() == doctest::Approx(1.0));
        }
    }
}

SCENARIO("C1: the zero-length delayed shift at dt = 1e-16 s equals the exact dt = 0 shift")
{
    GIVEN("two identical states with a point landmark, each with its own (stateful) model")
    {
        VectorXd mu0(12);
        mu0 << 0.2, -0.1, 0.3, 0.02, 0.01, -0.02, 2.0, -1.0, 0.5, 0.2, 0.1, -0.3;
        const MatrixXd p0 = 0.05 * randomSpd(12, 7201);
        const auto modelA = std::make_shared<SystemVisualNav>(corePsd());
        const auto modelB = std::make_shared<SystemVisualNav>(corePsd());
        SystemEstimator a(modelA, GaussianD::fromMoment(mu0, p0), 0.0);
        SystemEstimator b(modelB, GaussianD::fromMoment(mu0, p0), 0.0);
        modelA->addLandmark(a, SystemVisualNav::LandmarkType::Point, GaussianD::fromMoment(Vector3d(3, 1, 0), 0.4 * Matrix3d::Identity()));
        modelB->addLandmark(b, SystemVisualNav::LandmarkType::Point, GaussianD::fromMoment(Vector3d(3, 1, 0), 0.4 * Matrix3d::Identity()));
        WHEN("the delayed shift is applied with dt = 0 and with dt = 1.1e-16")
        {
            modelA->predictDelayed(a, 0.0);
            modelB->predictDelayed(b, 1.1e-16);
            THEN("both results are identical to rounding and etaPrev is the copy of eta in the mean")
            {
                CHECK(nearMatrix(a.mean(), b.mean(), 1e-12));
                CHECK(nearMatrix(a.state().sqrtInfo(), b.state().sqrtInfo(), 1e-10));
                CHECK(nearMatrix(a.mean().segment(12, 6), a.mean().segment(6, 6), 1e-9));
                CHECK(modelA->hasDelayedPose());
                CHECK(modelB->hasDelayedPose());
                CHECK(a.dim() == 21);
                CHECK(b.dim() == 21);
            }
            THEN("the zero-length shift keeps the landmark at the same offset as a positive-length one")
            {
                CHECK(modelA->landmarkOffset(modelA->landmarks()[0].id) == 18);
                CHECK(modelB->landmarkOffset(modelB->landmarks()[0].id) == 18);
            }
        }
    }
}

SCENARIO("C1: a measurement 1e-16 s older than the system time is accepted as simultaneous")
{
    GIVEN("a system at t = 1e-16 and a linear measurement at t = 0")
    {
        auto model = std::make_shared<CoreStaticModel>();
        SystemEstimator sys(model, GaussianD::fromMoment(VectorXd::Zero(2), MatrixXd::Identity(2, 2)), 1.0e-16);
        const MatrixXd h = MatrixXd::Identity(2, 2);
        const MeasurementLinearGaussian meas(0.0, h, VectorXd::Constant(2, 0.5), MatrixXd::Identity(2, 2));
        WHEN("the measurement is processed")
        {
            THEN("it is accepted and the posterior is the usual Kalman update")
            {
                CHECK_NOTHROW(meas.process(sys));
                CHECK(nearMatrix(sys.mean(), Eigen::Vector2d(0.25, 0.25), 1e-9));
            }
        }
    }
    GIVEN("a system at t = 1e-3 and a measurement at t = 0 (genuinely older)")
    {
        auto model = std::make_shared<CoreStaticModel>();
        SystemEstimator sys(model, GaussianD::fromMoment(VectorXd::Zero(1), MatrixXd::Identity(1, 1)), 1.0e-3);
        const MeasurementLinearGaussian meas(0.0, MatrixXd::Identity(1, 1), VectorXd::Zero(1), MatrixXd::Identity(1, 1));
        THEN("it is still rejected")
        {
            CHECK_THROWS_AS(meas.process(sys), std::invalid_argument);
        }
    }
}

SCENARIO("C2: rank-deficient Gauss-Newton curvature (linear, rank 1) matches the moment-form Kalman update")
{
    GIVEN("a 3-D Gaussian prior and a 2-D measurement H x + v whose H has rank one")
    {
        const MatrixXd p = randomSpd(3, 7301);
        const VectorXd mu = randomVector(3, 7302);
        const VectorXd hRow = randomVector(3, 7303);
        MatrixXd h(2, 3);
        h << hRow.transpose(), 2.0 * hRow.transpose();
        const MatrixXd r = randomSpd(2, 7304);
        const VectorXd y = randomVector(2, 7305);
        auto model = std::make_shared<CoreStaticModel>();
        SystemEstimator sys(model, GaussianD::fromMoment(mu, p), 0.0);
        const MeasurementLinearGaussian meas(0.0, h, y, r);
        WHEN("the Laplace update is applied (eigen square-root path, K = H^T R^{-1} H has rank 1)")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            VectorXd muRef;
            MatrixXd pRef;
            kalmanReference(mu, p, h, r, y, muRef, pRef);
            THEN("the result is exact (precisionIsExact) and equals the moment-form posterior")
            {
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(res.precisionIsExact);
                CAPTURE_EIGEN(sys.mean());
                CHECK(nearMatrix(sys.mean(), muRef, 1e-9));
                CHECK(nearMatrix(sys.covariance(), pRef, 1e-8));
            }
            THEN("the information matrix equals the prior information plus H^T R^{-1} H")
            {
                const MatrixXd expected = p.inverse() + h.transpose() * r.inverse() * h;
                CHECK(nearMatrix(sys.state().information(), expected, 1e-8));
            }
            THEN("the posterior mean is the MAP (zero gradient of the negative log posterior)")
            {
                CHECK(nearMatrix(sys.mean(), res.map.x, 1e-8));
            }
            THEN("the square-root factor is upper triangular with a positive diagonal and finite entries")
            {
                const MatrixXd& xi = sys.state().sqrtInfo();
                CHECK(isUpperTriangularFinite(xi));
                CHECK(xi.diagonal().minCoeff() > 0.0);
            }
        }
    }
}

SCENARIO("C2: a badly scaled prior keeps the weak-direction information accurate (square-root form)")
{
    GIVEN("a prior whose information is 1e8 along one rotated axis and 1e-4 along the other, given by its factor Xi0")
    {
        const double th = 0.4;
        Eigen::Matrix2d r0;
        r0 << std::cos(th), -std::sin(th),
              std::sin(th), std::cos(th);
        // Xi0^T Xi0 = R0 diag(1e8, 1e-4) R0^T: the weak eigenvector is the second column of R0.
        const Eigen::Matrix2d xi0 = Eigen::DiagonalMatrix<double, 2>(1e4, 1e-2).toDenseMatrix() * r0.transpose();
        auto model = std::make_shared<CoreStaticModel>();
        SystemEstimator sys(model, GaussianD::fromSqrtInfo(Eigen::Vector2d::Zero(), xi0), 0.0);
        // Linear observation of the weak direction: K = H^T R^{-1} H = c u u^T with c = 1e-4 (rank one).
        const double c = 1e-4;
        const MatrixXd h = r0.col(1).transpose();
        const MeasurementLinearGaussian meas(0.0, h, VectorXd::Zero(1), MatrixXd::Constant(1, 1, 1.0 / c));
        WHEN("the measurement is processed")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the exact posterior precision is R0 diag(1e8, 2e-4) R0^T, so the weak singular value of Xi is sqrt(2e-4)")
            {
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(res.precisionIsExact);
                const Eigen::JacobiSVD<MatrixXd> svd(sys.state().sqrtInfo());
                const VectorXd s = svd.singularValues();   // descending
                CAPTURE_EIGEN(s);
                const double weakRef = std::sqrt(1e-4 + c);
                CHECK(std::abs(s(1) - weakRef) <= 1e-7 * weakRef);
                CHECK(std::abs(s(0) - 1e4) <= 1e-7 * 1e4);
            }
        }
    }
}

SCENARIO("C2: rank-deficient nonlinear Gauss-Newton curvature: the posterior mean is the MAP")
{
    GIVEN("a 3-D prior and the scalar measurement y = x0 x1 + v (rank-one Jacobian at every state)")
    {
        const MatrixXd p = 0.3 * randomSpd(3, 7401);
        const VectorXd mu = VectorXd(Vector3d(1.0, 1.5, 0.4));
        auto model = std::make_shared<CoreStaticModel>();
        SystemEstimator sys(model, GaussianD::fromMoment(mu, p), 0.0);
        const ProductMeasurement meas(0.0, 2.2, 0.5, false);
        const GaussianD prior = sys.state();
        WHEN("the measurement is processed")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the MAP converges, is exact, and the posterior mean equals it")
            {
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(res.precisionIsExact);
                CAPTURE_EIGEN(sys.mean());
                CAPTURE_EIGEN(res.map.x);
                CHECK(nearMatrix(sys.mean(), res.map.x, 1e-8));
            }
            THEN("the gradient of the negative log posterior vanishes at the posterior mean")
            {
                VectorXd gll;
                meas.logLikelihood(sys.mean(), sys, &gll, nullptr);
                VectorXd gp;
                prior.logPdf(sys.mean(), &gp, nullptr);
                CHECK((gll + gp).cwiseAbs().maxCoeff() < 1e-6);
            }
            THEN("the factor is upper triangular and finite and the information is prior plus the GN curvature")
            {
                CHECK(isUpperTriangularFinite(sys.state().sqrtInfo()));
                VectorXd gll;
                MatrixXd hll;
                meas.logLikelihood(sys.mean(), sys, &gll, &hll);
                CHECK(nearMatrix(sys.state().information(), prior.information() - hll, 1e-7));
            }
        }
    }
}

SCENARIO("C2: a truly indefinite curvature (exact second-order term) takes the flagged fallback")
{
    GIVEN("a prior centred where the residual is large, so that K = J^T J + r Hess(h) is indefinite at the MAP")
    {
        const MatrixXd p = 0.5 * Matrix3d::Identity();
        const VectorXd mu = Vector3d(3.0, 3.0, 0.0);
        auto model = std::make_shared<CoreStaticModel>();
        SystemEstimator sys(model, GaussianD::fromMoment(mu, p), 0.0);
        const ProductMeasurement meas(0.0, 0.5, 1.0, true);
        const GaussianD prior = sys.state();
        WHEN("the measurement is processed")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            VectorXd gll;
            MatrixXd hll;
            meas.logLikelihood(res.map.x, sys, &gll, &hll);
            const Eigen::SelfAdjointEigenSolver<MatrixXd> es(-hll);
            THEN("the curvature at the MAP is truly indefinite and the fallback is flagged")
            {
                CHECK(es.eigenvalues().minCoeff() < -1e-3);
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(!res.precisionIsExact);
            }
            THEN("the posterior mean is still the MAP and the precision is prior plus K")
            {
                CHECK(nearMatrix(sys.mean(), res.map.x, 1e-7));
                CHECK(nearMatrix(sys.state().information(), prior.information() - hll, 1e-7));
                CHECK(isUpperTriangularFinite(sys.state().sqrtInfo()));
            }
        }
    }
}

SCENARIO("C2: a measurement that is flat at the MAP (zero curvature) leaves the prior mode unchanged")
{
    GIVEN("a prior and a measurement with zero Jacobian (x0 x1 at the origin of the support)")
    {
        const MatrixXd p = randomSpd(3, 7501);
        const VectorXd mu = VectorXd::Zero(3);
        auto model = std::make_shared<CoreStaticModel>();
        SystemEstimator sys(model, GaussianD::fromMoment(mu, p), 0.0);
        const ProductMeasurement meas(0.0, 0.0, 1.0, false);
        WHEN("it is processed")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the posterior mean is the prior mode and the factor is finite")
            {
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(nearMatrix(sys.mean(), mu, 1e-9));
                CHECK(sys.state().sqrtInfo().allFinite());
            }
        }
    }
}

SCENARIO("funcmin: a rank-deficient positive semi-definite Hessian takes the eigen path and converges")
{
    GIVEN("f(x) = 0.5 (a^T x - 1)^2, whose Hessian a a^T is rank one (the minimiser set is a hyperplane)")
    {
        const Vector3d a(1.0, 2.0, 0.5);
        auto f = [&a](const VectorXd& x, VectorXd* g, MatrixXd* h) -> double
        {
            const double res = a.dot(x) - 1.0;
            if (g != nullptr)
            {
                *g = res * a;
            }
            if (h != nullptr)
            {
                *h = a * a.transpose();
            }
            return 0.5 * res * res;
        };
        WHEN("it is minimised from the origin")
        {
            const FuncMinResult r = trustRegionMinimise(f, VectorXd::Zero(3));
            THEN("it reaches the minimum value zero with a finite iterate")
            {
                CHECK(r.status != FuncMinStatus::NumericalFailure);
                CHECK(r.f < 1e-14);
                CHECK(r.x.allFinite());
                CHECK(std::abs(a.dot(r.x) - 1.0) < 1e-7);
            }
        }
    }
}

SCENARIO("funcmin: the Cholesky fast path and the boundary solution agree with the reference")
{
    GIVEN("a positive definite Hessian and a gradient")
    {
        const MatrixXd h = randomSpd(6, 7601);
        const VectorXd g = randomVector(6, 7602);
        const VectorXd newton = -(h.ldlt().solve(g)).eval();
        WHEN("the trust radius contains the Newton step (Cholesky fast path)")
        {
            const VectorXd p = solveTrustRegionSubproblem(g, h, 1.5 * newton.norm());
            THEN("the Newton step is returned exactly")
            {
                CHECK(nearMatrix(p, newton, 1e-10));
            }
        }
        WHEN("the trust radius is smaller than the Newton step (boundary solution)")
        {
            const double radius = 0.6 * newton.norm();
            const VectorXd p = solveTrustRegionSubproblem(g, h, radius);
            THEN("the step lies on the boundary and no random feasible step has a lower model value")
            {
                CHECK(p.norm() == doctest::Approx(radius).epsilon(1e-9));
                const double model = g.dot(p) + 0.5 * p.dot(h * p);
                for (unsigned s = 0; s < 100; ++s)
                {
                    VectorXd q = randomVector(6, 7700 + s);
                    q *= radius / q.norm();
                    CHECK(g.dot(q) + 0.5 * q.dot(h * q) >= model - 1e-12);
                }
            }
        }
    }
}
