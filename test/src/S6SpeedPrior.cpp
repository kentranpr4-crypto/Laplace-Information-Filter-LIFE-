// Tests of the Scenario 6 speed prior (S6SpeedPrior.h): the gradient is validated against central finite differences of
// the log-likelihood (tolerance 1e-6 relative), the Gauss-Newton Hessian against finite differences of the gradient at a
// zero residual (where it is exact), and the prior pulls the body speed to the assumed value.
#include <cmath>
#include <memory>
#include <doctest/doctest.h>
#include <Eigen/Core>
#include "../../src/GaussianInfo.h"
#include "../../src/S6SpeedPrior.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"

namespace
{

Eigen::VectorXd stateWithVelocity(const Eigen::Vector3d& v)
{
    Eigen::VectorXd x = Eigen::VectorXd::Zero(18);
    x.head<3>() = v;
    x.segment<3>(3) << 0.02, -0.01, 0.03;
    x.segment<3>(6) << 0.5, -0.2, 0.1;
    x.segment<3>(9) << 0.01, 0.02, -0.03;
    x.segment<6>(12) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    return x;
}

} // namespace

SCENARIO("the speed prior gradient and Gauss-Newton Hessian match finite differences")
{
    GIVEN("a speed prior of 0.5 m/s with sigma 0.25 m/s and an estimator of the Scenario 6 layout")
    {
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.02));
        SystemEstimator est(model, GaussianInfo<double>::fromMoment(Eigen::VectorXd::Zero(18),
                                                                    Eigen::MatrixXd::Identity(18, 18)), 0.0);
        const slam::MeasurementSpeedPrior prior(0.0, 0.5, 0.25);
        REQUIRE(prior.speed() == doctest::Approx(0.5));
        WHEN("the body velocity is away from the speed shell")
        {
            const Eigen::VectorXd x = stateWithVelocity(Eigen::Vector3d(0.3, -0.25, 0.2));
            Eigen::VectorXd g;
            Eigen::MatrixXd h;
            const double value = prior.logLikelihood(x, est, &g, &h);
            THEN("the log-likelihood is finite and the gradient matches central differences")
            {
                CHECK(std::isfinite(value));
                const double eps = 1e-6;
                for (Eigen::Index k = 0; k < 3; ++k)
                {
                    Eigen::VectorXd p = x;
                    Eigen::VectorXd m = x;
                    p(k) += eps;
                    m(k) -= eps;
                    const double fd = (prior.logLikelihood(p, est, nullptr, nullptr)
                                       - prior.logLikelihood(m, est, nullptr, nullptr)) / (2.0 * eps);
                    CAPTURE(k);
                    CHECK(std::abs(fd - g(k)) <= 1e-6 * std::max(1.0, std::abs(g(k))));
                }
                CHECK(g.tail(15).isZero(0.0));
            }
            THEN("the Gauss-Newton Hessian is the Jacobian product (the gradient's derivative with zero residual part)")
            {
                // Hessian: -(1/sigma^2) dh dh^T; check its action against the finite-difference gradient direction.
                const Eigen::Vector3d v = x.head<3>();
                const double hv = std::sqrt(v.squaredNorm() + 1e-6);
                const Eigen::Vector3d dh = v / hv;
                CHECK(h.topLeftCorner<3, 3>().isApprox(-(1.0 / (0.25 * 0.25)) * dh * dh.transpose(), 1e-12));
                CHECK(h.bottomRightCorner(15, 15).isZero(0.0));
            }
        }
        WHEN("the speed equals the assumed speed (zero residual)")
        {
            const Eigen::VectorXd x = stateWithVelocity(Eigen::Vector3d(0.5, 0.0, 0.0));
            const double eps = 1e-6;
            Eigen::MatrixXd hFd = Eigen::MatrixXd::Zero(3, 3);
            Eigen::VectorXd g;
            Eigen::MatrixXd h;
            prior.logLikelihood(x, est, &g, &h);
            THEN("the Gauss-Newton Hessian equals the finite-difference Hessian of the gradient")
            {
                for (Eigen::Index k = 0; k < 3; ++k)
                {
                    Eigen::VectorXd p = x;
                    Eigen::VectorXd m = x;
                    p(k) += eps;
                    m(k) -= eps;
                    Eigen::VectorXd gp;
                    Eigen::VectorXd gm;
                    prior.logLikelihood(p, est, &gp, nullptr);
                    prior.logLikelihood(m, est, &gm, nullptr);
                    hFd.col(k) = (gp.head<3>() - gm.head<3>()) / (2.0 * eps);
                }
                CAPTURE(hFd);
                CHECK(h.topLeftCorner<3, 3>().isApprox(hFd, 1e-5));
            }
        }
        WHEN("the velocity is zero (the smoothing keeps the speed differentiable)")
        {
            const Eigen::VectorXd x = stateWithVelocity(Eigen::Vector3d::Zero());
            Eigen::VectorXd g;
            THEN("the gradient is finite and zero along the speed direction")
            {
                prior.logLikelihood(x, est, &g, nullptr);
                CHECK(g.allFinite());
                CHECK(g.head<3>().isZero(1e-12));
            }
        }
    }
}

SCENARIO("the speed prior pulls the body speed to the assumed value")
{
    GIVEN("a body velocity of 0.1 m/s and a speed prior of 0.5 m/s with sigma 0.25 m/s")
    {
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.02));
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(12);
        mu(0) = 0.1;
        SystemEstimator est(model, GaussianInfo<double>::fromMoment(mu, Eigen::MatrixXd::Identity(12, 12) * 0.09), 0.0);
        WHEN("the speed measurement is processed")
        {
            const slam::MeasurementSpeedPrior prior(0.0, 0.5, 0.25);
            const MeasurementUpdateResult res = prior.process(est);
            THEN("the posterior speed moves towards the assumed speed without overshooting it")
            {
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                const double speed = est.mean().head<3>().norm();
                CHECK(speed > 0.1);
                CHECK(speed <= 0.5 + 1e-6);
            }
        }
    }
}
