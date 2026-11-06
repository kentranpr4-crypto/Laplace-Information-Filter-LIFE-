// Tag corner measurement: corner Jacobians, likelihood gradient and Gauss-Newton Hessian against central finite differences.
#include <cmath>
#include <memory>
#include <numbers>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/Camera.h"
#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementTagBundle.h"
#include "../../src/Pose.hpp"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using slam::predictTagCorners;
using testutil::fdHessian;
using testutil::fdJacobian;
using testutil::nearMatrix;

namespace
{

/// Indoor-like camera with distortion and a non-trivial lever arm (every term of the chain rule is exercised).
Camera tagTestCamera()
{
    Camera cam(1920, 1080, (Matrix3d() << 889.3, 0.0, 959.5, 0.0, 892.3, 539.5, 0.0, 0.0, 1.0).finished(),
               {-0.0246, -0.0061, 0.0005, 0.0001, 0.0013});
    cam.setBodyToCameraExtrinsics(rigid::defaultRbc(), Vector3d(0.05, -0.02, -0.10));
    return cam;
}

/// Canonical corners of a tag of side 0.15 m, rotated in plane by q quarter turns.
Matrix<double, 4, 3> cornersFor(int q)
{
    const double h = 0.075;
    Matrix<double, 4, 3> c;
    c << -h, h, 0.0,
          h, h, 0.0,
          h, -h, 0.0,
         -h, -h, 0.0;
    const Matrix3d r = rpy2rot(Vector3d(0.0, 0.0, 0.5 * std::numbers::pi_v<double> * q));
    return (c * r.transpose()).eval();
}

/// Joint state x = [eta (6); L (6)] (n = 12) used by the Jacobian checks: body pose and one tag pose.
VectorXd tagState(const Vector3d& rB, const Vector3d& thB, const Vector3d& rL, const Vector3d& thL)
{
    VectorXd x = VectorXd::Zero(12);
    x.segment<3>(0) = rB;
    x.segment<3>(3) = thB;
    x.segment<3>(6) = rL;
    x.segment<3>(9) = thL;
    return x;
}

/// Corner vector as a function of the 12-vector [eta; L]; NaN-free for the states used in the checks.
VectorXd cornersOf(const Camera& cam, const VectorXd& x, const Matrix<double, 4, 3>& c)
{
    return predictTagCorners(cam, x.segment<3>(0), x.segment<3>(3), x.segment<3>(6), x.segment<3>(9), c).uv;
}

/// Dummy estimator for the measurement interface (the tag likelihood ignores it).
std::shared_ptr<SystemEstimator> dummyEstimator(Index n)
{
    auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1));
    GaussianInfo<double> g = GaussianInfo<double>::fromMoment(VectorXd::Zero(n), MatrixXd::Identity(n, n));
    return std::make_shared<SystemEstimator>(model, g, 0.0);
}

} // namespace

SCENARIO("tag corner predictions: the Jacobians match central finite differences")
{
    const Camera cam = tagTestCamera();
    const std::vector<VectorXd> states = {
        tagState(Vector3d(0.2, -0.1, 0.05), Vector3d(0.05, -0.08, 0.15), Vector3d(2.5, 0.4, 0.3), Vector3d(0.1, 0.05, 0.6)),
        tagState(Vector3d(-0.3, 0.2, -0.1), Vector3d(-0.02, 0.1, -0.4), Vector3d(3.5, -0.9, -0.2), Vector3d(-0.2, 0.0, 2.0)),
        tagState(Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(1.8, 0.1, 0.0), Vector3d(0.0, 0.0, 0.0)),
    };
    GIVEN("several body and tag poses (including a tag rotated in plane by 1 and 3 quarter turns)")
    {
        for (std::size_t s = 0; s < states.size(); ++s)
        {
            for (int q : {0, 1, 3})
            {
                const VectorXd& x = states[s];
                const Matrix<double, 4, 3> c = cornersFor(q);
                const slam::TagCornerPrediction pred = predictTagCorners(cam, x.segment<3>(0), x.segment<3>(3),
                                                                        x.segment<3>(6), x.segment<3>(9), c);
                INFO("state ", s, " quarter turns ", q);
                REQUIRE(pred.valid);
                const MatrixXd jFd = fdJacobian([&](const VectorXd& y) { return cornersOf(cam, y, c); }, x, 1e-6);
                MatrixXd jAn(8, 12);
                jAn.block<8, 6>(0, 0) = pred.dEta;
                jAn.block<8, 6>(0, 6) = pred.dLm;
                CAPTURE_EIGEN(jAn);
                CAPTURE_EIGEN(jFd);
                THEN("the 8 x 12 analytic Jacobian equals the central-difference Jacobian (relative 1e-6)")
                {
                    CHECK(nearMatrix(jAn, jFd, 1e-6));
                }
            }
        }
    }
    GIVEN("a state with a tag behind the camera")
    {
        const VectorXd x = tagState(Vector3d::Zero(), Vector3d::Zero(), Vector3d(-2.0, 0.0, 0.0), Vector3d::Zero());
        const slam::TagCornerPrediction pred =
            predictTagCorners(cam, x.segment<3>(0), x.segment<3>(3), x.segment<3>(6), x.segment<3>(9), cornersFor(0));
        THEN("the prediction is invalid and its corners are NaN (the trust region rejects it)")
        {
            CHECK_FALSE(pred.valid);
            CHECK(std::isnan(pred.uv(0)));
        }
    }
}

SCENARIO("tag likelihood: gradient and Gauss-Newton Hessian")
{
    const Camera cam = tagTestCamera();
    const Matrix<double, 4, 3> c = cornersFor(0);
    // Two tags, eta at offset 0, landmark blocks at 6 and 12 (n = 18).
    const Index n = 18;
    VectorXd x = VectorXd::Zero(n);
    x.segment<3>(0) = Vector3d(0.2, -0.1, 0.05);
    x.segment<3>(3) = Vector3d(0.05, -0.08, 0.15);
    x.segment<3>(6) = Vector3d(2.5, 0.4, 0.3);
    x.segment<3>(9) = Vector3d(0.1, 0.05, 0.6);
    x.segment<3>(12) = Vector3d(3.0, -0.6, -0.2);
    x.segment<3>(15) = Vector3d(-0.1, 0.2, -0.8);

    std::vector<TagObservation> obs(2);
    obs[0].landmarkId = 0;
    obs[0].landmarkOffset = 6;
    obs[0].cornersLocal = c;
    obs[0].sigmaPx = 2.0;
    obs[1].landmarkId = 1;
    obs[1].landmarkOffset = 12;
    obs[1].cornersLocal = cornersFor(2);
    obs[1].sigmaPx = 2.5;

    const VectorXd predicted = MeasurementTagBundle(0.0, cam, obs, 0).predictedCorners(x);
    // Perturbed observations so that the residuals are non-zero (eta at offset 0 in this test layout).
    for (std::size_t j = 0; j < obs.size(); ++j)
    {
        Matrix<double, 8, 1> shift;
        for (int k = 0; k < 8; ++k)
        {
            shift(k) = (k % 2 == 0 ? 1.5 : -1.0) * (1.0 + 0.3 * static_cast<double>(j));
        }
        obs[j].pixels = predicted.segment<8>(8 * static_cast<Index>(j)) + shift;
    }
    const MeasurementTagBundle meas(0.0, cam, obs, 0);
    const auto est = dummyEstimator(n);

    WHEN("the gradient and the Hessian are evaluated at a generic state")
    {
        VectorXd g;
        MatrixXd h;
        const double value = meas.logLikelihood(x, *est, &g, &h);
        const VectorXd gFd = testutil::fdGradient([&](const VectorXd& y) { return meas.logLikelihood(y, *est, nullptr, nullptr); }, x, 1e-6);
        THEN("the value is finite and the gradient equals the central-difference gradient (relative 1e-5)")
        {
            CHECK(std::isfinite(value));
            CAPTURE_EIGEN(g);
            CAPTURE_EIGEN(gFd);
            CHECK(nearMatrix(g, gFd, 1e-5));
        }
        THEN("the Hessian is symmetric and negative semi-definite (Gauss-Newton term -J^T W J)")
        {
            CHECK((h - h.transpose()).norm() < 1e-9);
            CHECK(Eigen::SelfAdjointEigenSolver<MatrixXd>(0.5 * (h + h.transpose())).eigenvalues().maxCoeff() < 1e-9);
        }
    }

    WHEN("the observations equal the predictions (zero residual)")
    {
        std::vector<TagObservation> exact = obs;
        for (std::size_t j = 0; j < exact.size(); ++j)
        {
            exact[j].pixels = predicted.segment<8>(8 * static_cast<Index>(j));
        }
        const MeasurementTagBundle zeroRes(0.0, cam, exact, 0);
        VectorXd g;
        MatrixXd h;
        zeroRes.logLikelihood(x, *est, &g, &h);
        const MatrixXd hFd = fdHessian([&](const VectorXd& y) { return zeroRes.logLikelihood(y, *est, nullptr, nullptr); }, x, 1e-4);
        THEN("the gradient vanishes and the Gauss-Newton Hessian equals the exact Hessian (second-order term is zero)")
        {
            CHECK(g.norm() < 1e-8);
            CAPTURE_EIGEN(h);
            CAPTURE_EIGEN(hFd);
            CHECK(nearMatrix(h, hFd, 1e-4));
        }
    }

    WHEN("the support and the variance vector are queried")
    {
        const std::vector<Index> support = meas.supportIndices(*est);   // poseOffset 0 in this layout
        THEN("the support is eta (6) followed by the six states of every tag, and the pixel variances match sigma^2")
        {
            CHECK(support.size() == 18);
            CHECK(support.front() == 0);
            CHECK(support.back() == 17);
            const VectorXd v = meas.pixelVariances();
            CHECK(v.size() == 16);
            CHECK(v(0) == doctest::Approx(4.0));
            CHECK(v(15) == doctest::Approx(6.25));
        }
    }

    WHEN("an observation is not a valid tag (non-positive sigma or non-finite pixel)")
    {
        std::vector<TagObservation> bad = obs;
        bad[0].sigmaPx = 0.0;
        THEN("the measurement is rejected")
        {
            CHECK_THROWS_AS(MeasurementTagBundle(0.0, cam, bad), std::invalid_argument);
        }
    }
}
