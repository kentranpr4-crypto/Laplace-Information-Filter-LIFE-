#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <Eigen/Core>

#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementAltimeter.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "testUtil.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

SCENARIO("Altimeter: gradient and Hessian are exact for the linear relative-height model")
{
    GIVEN("a relative altitude measurement y = h - h0 = 2.5 m with sigma 0.02 m")
    {
        MeasurementAltimeter meas(1.0, 2.5, 0.02);
        auto sys = std::make_unique<SystemEstimator>(
            std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1)),
            GaussianInfo<double>::fromMoment(VectorXd::Zero(12), MatrixXd::Identity(12, 12)), 0.0);
        WHEN("the log-likelihood is evaluated at several heights")
        {
            for (double z : {-2.0, -2.5, 0.3})
            {
                VectorXd x = VectorXd::Zero(12);
                x(8) = z;
                VectorXd g;
                MatrixXd h;
                const double ll = meas.logLikelihood(x, *sys, &g, &h);
                const auto fn = [&](const VectorXd& xx) { return meas.logLikelihood(xx, *sys, nullptr, nullptr); };
                const VectorXd fdg = testutil::fdGradient(fn, x, 1e-6);
                THEN("the gradient matches central differences and the Hessian is -1/sigma^2 on z")
                {
                    CAPTURE(z);
                    CHECK(std::isfinite(ll));
                    CHECK(g.isApprox(fdg, 1e-6));
                    CHECK(h(8, 8) == doctest::Approx(-1.0 / (0.02 * 0.02)));
                    CHECK(h.trace() == doctest::Approx(h(8, 8)));
                }
            }
        }
    }
}

SCENARIO("Altimeter: the update pins the z position to the relative-height reference")
{
    GIVEN("a broad prior on z centred at -3 m (height 3 m above the start)")
    {
        Eigen::Matrix<double, 6, 1> psd = Eigen::Matrix<double, 6, 1>::Constant(0.1);
        VectorXd mean = VectorXd::Zero(12);
        mean(8) = -3.0;
        VectorXd sig = VectorXd::Constant(12, 5.0);
        SystemEstimator est(std::make_shared<SystemVisualNav>(psd),
                            GaussianInfo<double>::fromMoment(mean, sig.array().square().matrix().asDiagonal()), 0.0);
        WHEN("an altimeter reading of 0 m (relative to the start) is processed")
        {
            MeasurementAltimeter meas(0.0, 0.0, 0.02);
            const auto result = meas.process(est);
            THEN("the z mean moves to the reading within the measurement precision")
            {
                CHECK(result.precisionIsExact);
                CHECK(std::abs(est.mean()(8)) < 0.02);
            }
            THEN("the posterior variance of z is no larger than the measurement variance")
            {
                const double varZ = est.state().marginal({8}).covariance()(0, 0);
                CHECK(varZ <= 0.02 * 0.02 * (1.0 + 1e-6));
            }
        }
    }
}

SCENARIO("Altimeter: invalid sigma is rejected")
{
    THEN("non-positive or non-finite sigma throws")
    {
        CHECK_THROWS_AS(MeasurementAltimeter(0.0, 1.0, 0.0), std::invalid_argument);
        CHECK_THROWS_AS(MeasurementAltimeter(0.0, 1.0, -0.1), std::invalid_argument);
        CHECK_THROWS_AS(MeasurementAltimeter(0.0, 1.0, NAN), std::invalid_argument);
    }
}
