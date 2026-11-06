#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>

#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementHorizon.h"
#include "../../src/Pose.hpp"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "testUtil.h"

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using testutil::fdGradient;
using testutil::fdHessian;
using testutil::fdJacobian;

namespace
{

/// State of 18 dimensions with the given body pose in eta_k; other blocks zero.
VectorXd stateWithPose(const Vector3d& r, const Vector3d& theta)
{
    VectorXd x = VectorXd::Zero(18);
    x.segment<3>(6) = r;
    x.segment<3>(9) = theta;
    x.segment<3>(12) = r;
    x.segment<3>(15) = theta;
    return x;
}

/// Normalised image points of world-horizontal directions, computed with the rigid-body projection of a far point.
std::vector<Vector2d> horizonPoints(const Vector3d& r, const Vector3d& theta, const Eigen::Matrix3d& Rbc, int n)
{
    std::vector<Vector2d> pts;
    for (int i = 0; i < 4 * n; ++i)   // sample more directions than needed: about 40 percent face the camera
    {
        const double psi = 2.0 * std::numbers::pi_v<double> * i / (4 * n);
        const Vector3d far = r + 1.0e6 * Vector3d(std::cos(psi), std::sin(psi), 0.0);
        const Vector3d pc = rigid::worldToCamera(far, r, theta, Rbc, Vector3d::Zero());
        // Keep directions well in front of the camera (bounded normalised coordinates); grazing directions
        // would give huge normalised values that are not image observations.
        if (pc.z() > 0.3 * pc.norm())
        {
            pts.emplace_back(pc.x() / pc.z(), pc.y() / pc.z());
        }
    }
    return pts;
}

struct Attitude
{
    const char* name;
    Vector3d theta;
};

std::vector<Attitude> attitudes()
{
    return {
        {"level", Vector3d(0.0, 0.0, 0.0)},
        {"rolled", Vector3d(0.2, 0.05, 0.4)},
        {"pitched and banked", Vector3d(-0.15, 0.22, -1.1)},
    };
}

} // namespace

SCENARIO("Visual horizon: predicted residuals vanish for world-horizontal directions")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    for (const Attitude& a : attitudes())
    {
        CAPTURE(a.name);
        GIVEN("points at infinity along world-horizontal directions, projected by the rigid-body model")
        {
            const Vector3d r(5.0, -3.0, 40.0);
            const auto pts = horizonPoints(r, a.theta, Rbc, 24);
            REQUIRE(pts.size() >= 12);
            MeasurementHorizon meas(0.0, pts, 1e-3, Rbc);
            WHEN("the horizon residuals are evaluated at the true attitude")
            {
                const VectorXd rr = meas.predict(stateWithPose(r, a.theta), nullptr);
                THEN("they are zero to the accuracy of the far-point approximation")
                {
                    CHECK(rr.cwiseAbs().maxCoeff() < 1e-9);
                }
            }
        }
    }
}

SCENARIO("Visual horizon: the residual does not depend on yaw")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    GIVEN("boundary points and a roll/pitch attitude")
    {
        const Vector3d r(0.0, 0.0, 30.0);
        const Vector3d th(0.1, -0.12, 0.3);
        const auto pts = horizonPoints(r, th, Rbc, 16);
        MeasurementHorizon meas(0.0, pts, 1e-3, Rbc);
        WHEN("the yaw of the state is changed")
        {
            const VectorXd a = meas.predict(stateWithPose(r, th), nullptr);
            VectorXd thRot = th;
            thRot(2) += 1.7;
            const VectorXd b = meas.predict(stateWithPose(r, thRot), nullptr);
            THEN("the horizon is unchanged (yaw is not observed by the horizon)")
            {
                CHECK((a - b).cwiseAbs().maxCoeff() < 1e-12);
            }
        }
    }
}

SCENARIO("Visual horizon: Jacobian, gradient and Gauss-Newton Hessian match finite differences")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    auto sys = std::make_unique<SystemEstimator>(
        std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1)),
        GaussianInfo<double>::fromMoment(VectorXd::Zero(18), MatrixXd::Identity(18, 18)), 0.0);
    for (const Attitude& a : attitudes())
    {
        CAPTURE(a.name);
        GIVEN("noisy boundary points around the attitude (non-zero residuals)")
        {
            const Vector3d r(1.0, 2.0, 25.0);
            auto pts = horizonPoints(r, a.theta, Rbc, 20);
            std::mt19937 rng(17u);
            std::normal_distribution<double> noise(0.0, 3e-3);
            for (auto& p : pts)
            {
                p += Vector2d(noise(rng), noise(rng));
            }
            MeasurementHorizon meas(0.0, pts, 2e-3, Rbc);
            const VectorXd x = stateWithPose(r, a.theta);
            WHEN("the residual Jacobian is compared with central differences")
            {
                MatrixXd jac;
                meas.predict(x, &jac);
                const MatrixXd fd = fdJacobian([&](const VectorXd& xx) { return meas.predict(xx, nullptr); }, x, 1e-6);
                THEN("the three attitude columns agree") { CHECK(jac.isApprox(fd.middleCols(9, 3), 1e-6)); }
            }
            WHEN("the log-likelihood gradient is compared with central differences")
            {
                VectorXd g;
                meas.logLikelihood(x, *sys, &g, nullptr);
                const VectorXd fd = fdGradient([&](const VectorXd& xx) { return meas.logLikelihood(xx, *sys, nullptr, nullptr); }, x, 1e-6);
                THEN("the attitude gradient agrees and the rest is zero")
                {
                    CHECK(g.segment(9, 3).isApprox(fd.segment(9, 3), 1e-5));
                    CHECK(g.head(9).isZero());
                    CHECK(g.tail(6).isZero());
                }
            }
            WHEN("the Hessian is compared with the finite-difference Hessian of the log-likelihood")
            {
                MatrixXd h;
                meas.logLikelihood(x, *sys, nullptr, &h);
                const MatrixXd fd = fdHessian([&](const VectorXd& xx) { return meas.logLikelihood(xx, *sys, nullptr, nullptr); }, x, 1e-4);
                THEN("the attitude block agrees to the size of the omitted second-order term")
                {
                    const double scale = std::max(1.0, h.block(9, 9, 3, 3).cwiseAbs().maxCoeff());
                    CHECK((h.block(9, 9, 3, 3) - fd.block(9, 9, 3, 3)).cwiseAbs().maxCoeff() < 2e-2 * scale);
                }
            }
        }
    }
}

SCENARIO("Visual horizon: exact-line observations give an exact Gauss-Newton Hessian")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    auto sys = std::make_unique<SystemEstimator>(
        std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1)),
        GaussianInfo<double>::fromMoment(VectorXd::Zero(18), MatrixXd::Identity(18, 18)), 0.0);
    GIVEN("points exactly on the horizon of an attitude with roll and pitch")
    {
        const Vector3d r(0.0, 0.0, 20.0);
        const Vector3d th(0.07, 0.11, 0.9);
        const auto pts = horizonPoints(r, th, Rbc, 20);
        MeasurementHorizon meas(0.0, pts, 1e-3, Rbc);
        const VectorXd x = stateWithPose(r, th);
        WHEN("the Hessian is compared with finite differences")
        {
            MatrixXd h;
            meas.logLikelihood(x, *sys, nullptr, &h);
            const MatrixXd fd = fdHessian([&](const VectorXd& xx) { return meas.logLikelihood(xx, *sys, nullptr, nullptr); }, x, 1e-4);
            THEN("it equals the Gauss-Newton Hessian (second-order term vanishes at zero residual)")
            {
                const double scale = std::max(1.0, h.block(9, 9, 3, 3).cwiseAbs().maxCoeff());
                CHECK((h.block(9, 9, 3, 3) - fd.block(9, 9, 3, 3)).cwiseAbs().maxCoeff() < 1e-4 * scale);
            }
        }
    }
}

SCENARIO("Visual horizon: the update corrects roll and pitch but leaves yaw to its prior")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    GIVEN("a prior with a 0.1 rad roll/pitch error and a horizon observed at the true attitude")
    {
        const Vector3d r(0.0, 0.0, 20.0);
        const Vector3d truth(0.05, -0.08, 0.6);
        const auto pts = horizonPoints(r, truth, Rbc, 40);
        MeasurementHorizon meas(0.0, pts, 1e-3, Rbc);
        Eigen::Matrix<double, 6, 1> psd = Eigen::Matrix<double, 6, 1>::Constant(0.1);
        VectorXd mean = stateWithPose(r, truth);
        mean(9) += 0.1;
        mean(10) -= 0.1;
        VectorXd sig = VectorXd::Constant(18, 1.0);
        sig.segment<3>(9).setConstant(0.5);
        SystemEstimator est(std::make_shared<SystemVisualNav>(psd),
                            GaussianInfo<double>::fromMoment(mean, sig.array().square().matrix().asDiagonal()), 0.0);
        WHEN("the horizon measurement is processed")
        {
            const double yawBefore = est.mean()(11);
            meas.process(est);
            THEN("roll and pitch move onto the observed horizon")
            {
                CHECK(std::abs(est.mean()(9) - truth(0)) < 0.01);
                CHECK(std::abs(est.mean()(10) - truth(1)) < 0.01);
            }
            THEN("yaw is untouched by the horizon (only the prior-driven marginal remains)")
            {
                CHECK(std::abs(est.mean()(11) - yawBefore) < 1e-9);
            }
        }
    }
}

SCENARIO("Visual horizon: invalid inputs are rejected")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    THEN("fewer than two points, non-finite points or non-positive sigma throw")
    {
        CHECK_THROWS_AS(MeasurementHorizon(0.0, {Vector2d(0.1, 0.2)}, 1e-3, Rbc), std::invalid_argument);
        CHECK_THROWS_AS(MeasurementHorizon(0.0, {Vector2d(0.1, 0.2), Vector2d(NAN, 0.0)}, 1e-3, Rbc), std::invalid_argument);
        CHECK_THROWS_AS(MeasurementHorizon(0.0, {Vector2d(0.1, 0.2), Vector2d(0.0, 0.0)}, 0.0, Rbc), std::invalid_argument);
    }
    THEN("a camera looking along the vertical has no defined horizon")
    {
        MeasurementHorizon meas(0.0, {Vector2d(0.1, 0.2), Vector2d(0.0, 0.0)}, 1e-3, Rbc);
        const VectorXd x = stateWithPose(Vector3d::Zero(), Vector3d(0.0, std::numbers::pi / 2.0 - 1e-9, 0.0));
        CHECK_THROWS_AS(meas.predict(x, nullptr), std::domain_error);
    }
}
