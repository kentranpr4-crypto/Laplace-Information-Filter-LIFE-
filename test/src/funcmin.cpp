#include <cmath>
#include <limits>
#include <stdexcept>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/funcmin.hpp"
#include "testUtil.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using testutil::fdGradient;
using testutil::nearMatrix;
using testutil::randomVector;

namespace
{

/// Generalised Rosenbrock in two variables with analytic gradient and Hessian.
double rosenbrock(const VectorXd& x, VectorXd* g, MatrixXd* h)
{
    const double a = x(0);
    const double b = x(1);
    const double f = 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
    if (g != nullptr)
    {
        *g = VectorXd(2);
        (*g)(0) = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a);
        (*g)(1) = 200.0 * (b - a * a);
    }
    if (h != nullptr)
    {
        *h = MatrixXd(2, 2);
        (*h)(0, 0) = 1200.0 * a * a - 400.0 * b + 2.0;
        (*h)(0, 1) = -400.0 * a;
        (*h)(1, 0) = -400.0 * a;
        (*h)(1, 1) = 200.0;
    }
    return f;
}

/// f(x) = x0^4/4 - x0^2/2 + x1^2: two minima at (+-1, 0), a maximum at the origin (negative curvature there).
double doubleWell(const VectorXd& x, VectorXd* g, MatrixXd* h)
{
    const double a = x(0);
    const double b = x(1);
    const double f = 0.25 * a * a * a * a - 0.5 * a * a + b * b;
    if (g != nullptr)
    {
        *g = VectorXd(2);
        (*g)(0) = a * a * a - a;
        (*g)(1) = 2.0 * b;
    }
    if (h != nullptr)
    {
        *h = MatrixXd(2, 2);
        (*h)(0, 0) = 3.0 * a * a - 1.0;
        (*h)(0, 1) = 0.0;
        (*h)(1, 0) = 0.0;
        (*h)(1, 1) = 2.0;
    }
    return f;
}

} // namespace

SCENARIO("funcmin: trust-region Newton converges on the Rosenbrock function")
{
    GIVEN("the classic Rosenbrock function started at (-1.2, 1)")
    {
        const VectorXd x0 = (VectorXd(2) << -1.2, 1.0).finished();
        WHEN("it is minimised with the exact gradient and Hessian")
        {
            const FuncMinResult r = trustRegionMinimise(rosenbrock, x0);
            THEN("it converges to the minimiser (1, 1)")
            {
                CAPTURE_EIGEN(r.x);
                CHECK(r.status != FuncMinStatus::MaxIterations);
                CHECK(r.status != FuncMinStatus::NumericalFailure);
                CHECK(nearMatrix(r.x, Eigen::Vector2d(1.0, 1.0), 1e-6));
                CHECK(r.f < 1e-12);
                CHECK(r.g.cwiseAbs().maxCoeff() < 1e-6);
            }
            THEN("the returned Hessian is the exact Hessian at the solution")
            {
                VectorXd g;
                MatrixXd h;
                rosenbrock(r.x, &g, &h);
                CHECK(nearMatrix(r.H, h, 1e-9));
            }
        }
    }
}

SCENARIO("funcmin: indefinite Hessians (negative curvature) are handled without modification")
{
    GIVEN("the double-well function started at the saddle/maximum (0, 0)")
    {
        const VectorXd x0 = VectorXd::Zero(2);
        WHEN("the hard case (zero gradient, negative curvature) is minimised")
        {
            const FuncMinResult r = trustRegionMinimise(doubleWell, x0);
            THEN("the iterate escapes along the negative-curvature direction to a minimum (+-1, 0)")
            {
                CAPTURE_EIGEN(r.x);
                CHECK(std::abs(std::abs(r.x(0)) - 1.0) < 1e-6);
                CHECK(std::abs(r.x(1)) < 1e-8);
                CHECK(r.f == doctest::Approx(-0.25).epsilon(1e-9));
            }
        }
    }
    GIVEN("the double-well function started near the maximum at (0.1, 0.5)")
    {
        const VectorXd x0 = (VectorXd(2) << 0.1, 0.5).finished();
        WHEN("it is minimised")
        {
            const FuncMinResult r = trustRegionMinimise(doubleWell, x0);
            THEN("it converges to a local minimum with a positive definite Hessian")
            {
                CHECK(r.status != FuncMinStatus::MaxIterations);
                CHECK(std::abs(std::abs(r.x(0)) - 1.0) < 1e-6);
                CHECK(std::abs(r.x(1)) < 1e-8);
                CHECK(Eigen::SelfAdjointEigenSolver<MatrixXd>(r.H).eigenvalues().minCoeff() > 0.0);
            }
        }
    }
}

SCENARIO("funcmin: trust-region subproblem is exact for interior, boundary and indefinite cases")
{
    GIVEN("a positive definite H and a gradient")
    {
        const Eigen::Matrix4d h = testutil::randomSpd(4, 1001);
        const Eigen::Vector4d g = randomVector(4, 1002);
        WHEN("the radius is large")
        {
            const VectorXd p = solveTrustRegionSubproblem(g, h, 1e6);
            THEN("the Newton step -H^{-1} g is returned")
            {
                CHECK(nearMatrix(p, -(h.ldlt().solve(g)).eval(), 1e-10));
            }
        }
        WHEN("the radius is small")
        {
            const double radius = 0.05;
            const VectorXd p = solveTrustRegionSubproblem(g, h, radius);
            THEN("the step lies on the boundary and no random feasible step is better")
            {
                CHECK(p.norm() == doctest::Approx(radius).epsilon(1e-9));
                const double model = g.dot(p) + 0.5 * p.dot(h * p);
                for (unsigned s = 0; s < 200; ++s)
                {
                    Eigen::Vector4d q = randomVector(4, 2000 + s);
                    q *= radius / q.norm();
                    CHECK(g.dot(q) + 0.5 * q.dot(h * q) >= model - 1e-12);
                }
            }
        }
    }
    GIVEN("an indefinite H with zero gradient (hard case)")
    {
        MatrixXd h(2, 2);
        h << -1.0, 0.0,
              0.0, 2.0;
        const VectorXd g = VectorXd::Zero(2);
        WHEN("the subproblem is solved with radius 0.5")
        {
            const VectorXd p = solveTrustRegionSubproblem(g, h, 0.5);
            THEN("the step has norm equal to the radius and lies along the negative-curvature direction")
            {
                CHECK(p.norm() == doctest::Approx(0.5).epsilon(1e-12));
                CHECK(std::abs(p(0)) == doctest::Approx(0.5).epsilon(1e-12));
            }
        }
    }
}

SCENARIO("funcmin: a non-finite objective at the start is reported, not propagated")
{
    GIVEN("an objective that returns NaN")
    {
        auto bad = [](const VectorXd&, VectorXd* g, MatrixXd* h) -> double
        {
            if (g != nullptr)
            {
                *g = VectorXd::Zero(2);
            }
            if (h != nullptr)
            {
                *h = MatrixXd::Identity(2, 2);
            }
            return std::numeric_limits<double>::quiet_NaN();
        };
        WHEN("it is minimised")
        {
            const FuncMinResult r = trustRegionMinimise(bad, VectorXd::Zero(2));
            THEN("the status is NumericalFailure")
            {
                CHECK(r.status == FuncMinStatus::NumericalFailure);
            }
        }
    }
}

SCENARIO("funcmin: gradient of the test objectives matches finite differences")
{
    GIVEN("Rosenbrock at a random point")
    {
        const VectorXd x = randomVector(2, 3001);
        VectorXd g;
        rosenbrock(x, &g, nullptr);
        const VectorXd gFd = fdGradient([](const VectorXd& xx) { return rosenbrock(xx, nullptr, nullptr); }, x);
        CHECK(nearMatrix(g, gFd, 1e-6));
    }
}

SCENARIO("funcmin (R4-02): an exception thrown by the objective at a trial point never escapes")
{
    GIVEN("f(x) = (x-3)^2 that throws std::domain_error for x > 0.5, started at x0 = 0.4 with a radius of 10")
    {
        auto f = [](const VectorXd& x, VectorXd* g, MatrixXd* h) -> double
        {
            if (x(0) > 0.5)
            {
                throw std::domain_error("outside the model domain");
            }
            if (g != nullptr)
            {
                *g = VectorXd::Constant(1, 2.0 * (x(0) - 3.0));
            }
            if (h != nullptr)
            {
                *h = MatrixXd::Constant(1, 1, 2.0);
            }
            return (x(0) - 3.0) * (x(0) - 3.0);
        };
        FuncMinOptions opts;
        opts.initialRadius = 10.0;
        WHEN("it is minimised")
        {
            FuncMinResult r;
            CHECK_NOTHROW(r = trustRegionMinimise(f, VectorXd::Constant(1, 0.4), opts));
            THEN("the iterate stays inside the domain, is finite and the objective does not increase")
            {
                CAPTURE(r.x(0));
                CHECK(r.x(0) <= 0.5);
                CHECK(r.x(0) > 0.4);
                CHECK(std::isfinite(r.f));
                CHECK(r.f <= (0.4 - 3.0) * (0.4 - 3.0));
                CHECK(r.g.allFinite());
            }
        }
    }
    GIVEN("an objective that throws only when derivatives are requested beyond x = 0.45 (the value is fine)")
    {
        auto f = [](const VectorXd& x, VectorXd* g, MatrixXd* h) -> double
        {
            if (g != nullptr && x(0) > 0.45)
            {
                throw std::runtime_error("derivatives unavailable");
            }
            if (g != nullptr)
            {
                *g = VectorXd::Constant(1, 2.0 * (x(0) - 3.0));
            }
            if (h != nullptr)
            {
                *h = MatrixXd::Constant(1, 1, 2.0);
            }
            return (x(0) - 3.0) * (x(0) - 3.0);
        };
        FuncMinOptions opts;
        opts.initialRadius = 10.0;
        WHEN("it is minimised")
        {
            FuncMinResult r;
            CHECK_NOTHROW(r = trustRegionMinimise(f, VectorXd::Constant(1, 0.4), opts));
            THEN("no accepted iterate lies where the derivatives failed: the state is never committed there")
            {
                CAPTURE(r.x(0));
                CHECK(r.x(0) <= 0.45 + 1e-12);
                CHECK(r.g.allFinite());
                CHECK(std::isfinite(r.f));
            }
        }
    }
    GIVEN("an objective that throws at the starting point")
    {
        auto f = [](const VectorXd&, VectorXd*, MatrixXd*) -> double { throw std::logic_error("no value"); };
        WHEN("it is minimised")
        {
            const VectorXd x0 = VectorXd::Constant(2, 0.7);
            FuncMinResult r;
            CHECK_NOTHROW(r = trustRegionMinimise(f, x0));
            THEN("the status is NumericalFailure and the starting point is returned unchanged")
            {
                CHECK(r.status == FuncMinStatus::NumericalFailure);
                CHECK(r.x.isApprox(x0));
            }
        }
    }
}

SCENARIO("funcmin: MaxIterations is reported when the iteration limit stops a non-converged minimisation")
{
    GIVEN("Rosenbrock from (-1.2, 1) with a single iteration allowed")
    {
        FuncMinOptions opts;
        opts.maxIterations = 1;
        WHEN("it is minimised")
        {
            VectorXd x0(2);
            x0 << -1.2, 1.0;
            const FuncMinResult r = trustRegionMinimise(rosenbrock, x0, opts);
            THEN("the status is MaxIterations and the Newton norm is finite and positive")
            {
                CHECK(r.status == FuncMinStatus::MaxIterations);
                CHECK(std::isfinite(newtonNorm(r.g, r.H)));
                CHECK(newtonNorm(r.g, r.H) > 0.0);
            }
        }
    }
}

SCENARIO("funcmin: Newton norm equals sqrt(g^T H^{-1} g) and is undefined for an indefinite Hessian")
{
    GIVEN("a random symmetric positive definite H and a random gradient")
    {
        const MatrixXd h = testutil::randomSpd(4, 3101);
        const VectorXd g = randomVector(4, 3102);
        WHEN("the Newton norm is evaluated by one triangular solve")
        {
            const double lambda = newtonNorm(g, h);
            THEN("it equals the reference sqrt(g^T H^{-1} g) (explicit inverse, test only)")
            {
                const double ref = std::sqrt(g.dot(h.inverse() * g));
                CHECK(lambda == doctest::Approx(ref).epsilon(1e-10));
            }
        }
        WHEN("H has a negative eigenvalue")
        {
            MatrixXd hi = h;
            hi(0, 0) = -10.0;
            THEN("the norm is NaN (undefined), so the MAP policy will not accept the iterate")
            {
                CHECK(std::isnan(newtonNorm(g, hi)));
            }
        }
        WHEN("the gradient is zero")
        {
            THEN("the norm is zero")
            {
                CHECK(newtonNorm(VectorXd::Zero(4), h) == doctest::Approx(0.0));
            }
        }
    }
}
