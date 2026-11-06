// Point-landmark pixel measurement: projection Jacobians, gradient and Hessian against central finite differences.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/Camera.h"
#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementPointBundle.h"
#include "../../src/Pose.hpp"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using GaussianD = GaussianInfo<double>;
using slam::PixelNoiseModel;
using slam::PointPixelJacobian;
using slam::pointPixelJacobian;
using testutil::fdGradient;
using testutil::fdHessian;
using testutil::fdJacobian;
using testutil::nearMatrix;

namespace
{

/// Indoor-like camera with distortion and a non-trivial lever arm (rCBb) so every term of the chain rule is exercised.
Camera testCamera()
{
    Camera cam(1920, 1080, (Matrix3d() << 889.3, 0.0, 959.5, 0.0, 892.3, 539.5, 0.0, 0.0, 1.0).finished(),
               {-0.0246, -0.0061, 0.0005, 0.0001, 0.0013});
    Matrix3d rbc = rigid::defaultRbc();
    cam.setBodyToCameraExtrinsics(rbc, Vector3d(0.05, -0.02, -0.10));
    return cam;
}

/// State with a body pose and two point landmarks in front of the camera (n = 18).
VectorXd poseAndPoints(double yaw = 0.15, double pitch = -0.08, double roll = 0.05)
{
    VectorXd x = VectorXd::Zero(18);
    x.segment<3>(6) = Vector3d(0.2, -0.1, 0.05);
    x.segment<3>(9) = Vector3d(roll, pitch, yaw);
    x.segment<3>(12) = Vector3d(2.5, 0.4, 0.3);
    x.segment<3>(15) = Vector3d(3.4, -0.9, -0.2);
    return x;
}

/// Measurement with observations generated from the given pixels.
std::vector<MeasurementPointBundle::Observation> observationsFrom(const VectorXd& pixels, double sigma)
{
    std::vector<MeasurementPointBundle::Observation> obs(2);
    obs[0].landmarkId = 0;
    obs[0].landmarkOffset = 12;
    obs[0].pixel = pixels.segment<2>(0);
    obs[0].sigmaPx = sigma;
    obs[1].landmarkId = 1;
    obs[1].landmarkOffset = 15;
    obs[1].pixel = pixels.segment<2>(2);
    obs[1].sigmaPx = sigma;
    return obs;
}

} // namespace

SCENARIO("PixelNoiseModel: the pixel sigma grows from the principal point to the image corners")
{
    GIVEN("the indoor-like camera and the default noise model")
    {
        const Camera cam = testCamera();
        const PixelNoiseModel model;
        WHEN("the sigma is evaluated at the principal point and at a corner")
        {
            const double centre = model.sigma(cam, Vector2d(cam.cx(), cam.cy()));
            const double corner = model.sigma(cam, Vector2d(0.0, 0.0));
            THEN("the centre gives sigma0 and the corner gives sigma0 + sigmaEdge (inflated, > calibration RMS)")
            {
                CHECK(centre == doctest::Approx(model.sigma0));
                CHECK(corner > centre);
                CHECK(corner <= model.sigma0 + model.sigmaEdge + 1e-12);
                CHECK(model.sigma0 >= 0.5 * 1.118);   // NIS-tuned: within a factor ~2 of the calibration RMS (1.118 px)
                CHECK(model.sigma0 <= 3.0 * 1.118);
            }
        }
    }
}

SCENARIO("pointPixelJacobian: the projection Jacobians with respect to eta and the landmark match finite differences")
{
    GIVEN("a camera with distortion and lever arm, and several body poses with points in front of it")
    {
        const Camera cam = testCamera();
        for (int s = 0; s < 4; ++s)
        {
            const double k = 0.1 * static_cast<double>(s);
            const Vector3d r(0.2 + k, -0.1 + 0.5 * k, 0.05 - k);
            const Vector3d theta(0.05 - 0.2 * k, -0.08 + 0.1 * k, 0.15 + 0.3 * k);
            const Vector3d lm(2.5 + 0.3 * k, 0.4 - 0.1 * k, 0.3 + 0.05 * k);
            INFO("pose sample ", s);
            WHEN("the analytic Jacobians are compared with central differences")
            {
                const PointPixelJacobian pj = pointPixelJacobian(cam, r, theta, lm);
                REQUIRE(pj.valid);
                VectorXd etaLm(9);
                etaLm << r, theta, lm;
                const MatrixXd jFd = fdJacobian(
                    [&](const VectorXd& v)
                    {
                        return pointPixelJacobian(cam, v.head<3>(), v.segment<3>(3), v.tail<3>()).uv;
                    },
                    etaLm, 1e-6);
                THEN("d u / d eta equals the finite-difference block (2 x 6)")
                {
                    CAPTURE_EIGEN(pj.dUvdEta);
                    CHECK(nearMatrix(pj.dUvdEta, jFd.leftCols(6), 1e-6));
                }
                THEN("d u / d rLNn equals the finite-difference block (2 x 3)")
                {
                    CAPTURE_EIGEN(pj.dUvdLm);
                    CHECK(nearMatrix(pj.dUvdLm, jFd.rightCols(3), 1e-6));
                }
                THEN("the camera-frame point equals rigid::worldToCamera")
                {
                    CHECK(nearMatrix(pj.pc, rigid::worldToCamera(lm, r, theta, cam.Rbc(), cam.rCBb()), 1e-12));
                }
            }
        }
    }
}

SCENARIO("pointPixelJacobian: points behind the camera are reported invalid")
{
    GIVEN("a point behind the camera")
    {
        const Camera cam = testCamera();
        WHEN("the projection is requested")
        {
            const PointPixelJacobian pj = pointPixelJacobian(cam, Vector3d::Zero(), Vector3d::Zero(), Vector3d(-2.0, 0.0, 0.0));
            THEN("it is flagged invalid and the pixel is NaN")
            {
                CHECK(!pj.valid);
                CHECK(!pj.uv.allFinite());
            }
        }
    }
}

SCENARIO("MeasurementPointBundle: the gradient equals the finite-difference gradient of the log-likelihood")
{
    GIVEN("two point observations with arbitrary pixels and the pose-plus-points state")
    {
        const Camera cam = testCamera();
        const VectorXd x0 = poseAndPoints();
        const VectorXd pixels = (VectorXd(4) << 980.0, 500.0, 1100.0, 560.0).finished();
        MeasurementPointBundle meas(0.0, cam, observationsFrom(pixels, 3.0));
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1));
        SystemEstimator sys(model, GaussianD::fromMoment(VectorXd::Zero(18), MatrixXd::Identity(18, 18)), 0.0);
        for (int s = 0; s < 3; ++s)
        {
            VectorXd x = x0;
            x.segment<3>(6) += 0.05 * s * Vector3d::Ones();
            x.segment<3>(12) += 0.02 * s * Vector3d(1, -1, 1);
            INFO("sample ", s);
            WHEN("the analytic gradient is compared with central differences")
            {
                VectorXd g;
                meas.logLikelihood(x, sys, &g, nullptr);
                const VectorXd gFd = fdGradient([&](const VectorXd& xx) { return meas.logLikelihood(xx, sys, nullptr, nullptr); }, x, 1e-6);
                THEN("it matches the finite-difference gradient (landmark and pose blocks)")
                {
                    CAPTURE_EIGEN(g.transpose());
                    CHECK(nearMatrix(g, gFd, 1e-6));
                    CHECK(g.segment<6>(0).isZero(0.0));   // nu is not observed
                }
            }
        }
    }
}

SCENARIO("MeasurementPointBundle: the Gauss-Newton Hessian equals the finite-difference Hessian at zero residual")
{
    GIVEN("observations equal to the predicted pixels (zero residual), where the omitted second-order term vanishes")
    {
        const Camera cam = testCamera();
        const VectorXd x = poseAndPoints();
        const VectorXd pixels = [&]
        {
            const MeasurementPointBundle probe(0.0, cam, observationsFrom(VectorXd::Zero(4), 1.0));
            return probe.predictedPixels(x);
        }();
        MeasurementPointBundle meas(0.0, cam, observationsFrom(pixels, 2.0));
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1));
        SystemEstimator sys(model, GaussianD::fromMoment(VectorXd::Zero(18), MatrixXd::Identity(18, 18)), 0.0);
        WHEN("the analytic Hessian is compared with the central-difference Hessian of the log-likelihood")
        {
            MatrixXd h;
            meas.logLikelihood(x, sys, nullptr, &h);
            const MatrixXd hFd = fdHessian([&](const VectorXd& xx) { return meas.logLikelihood(xx, sys, nullptr, nullptr); }, x, 1e-4);
            THEN("the 9 x 9 pose-landmark blocks agree and the unobserved rows are zero")
            {
                CAPTURE_EIGEN(h.block(6, 6, 9, 9));
                CHECK(nearMatrix(h.block(6, 6, 9, 9), hFd.block(6, 6, 9, 9), 1e-5));
                CHECK(nearMatrix(h.block(12, 12, 3, 3), hFd.block(12, 12, 3, 3), 1e-5));
                CHECK(h.topRows(6).isZero(0.0));
                CHECK(h.block(0, 0, 18, 18).isApprox(h.block(0, 0, 18, 18).transpose()));
            }
        }
    }
    GIVEN("a residual of a few pixels (the omitted second-order term is then present)")
    {
        const Camera cam = testCamera();
        const VectorXd x = poseAndPoints();
        VectorXd pixels = [&]
        {
            const MeasurementPointBundle probe(0.0, cam, observationsFrom(VectorXd::Zero(4), 1.0));
            return probe.predictedPixels(x);
        }();
        pixels += VectorXd::Constant(4, 3.0);
        MeasurementPointBundle meas(0.0, cam, observationsFrom(pixels, 2.0));
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1));
        SystemEstimator sys(model, GaussianD::fromMoment(VectorXd::Zero(18), MatrixXd::Identity(18, 18)), 0.0);
        WHEN("the Gauss-Newton Hessian is compared with the exact finite-difference Hessian")
        {
            MatrixXd h;
            meas.logLikelihood(x, sys, nullptr, &h);
            const MatrixXd hFd = fdHessian([&](const VectorXd& xx) { return meas.logLikelihood(xx, sys, nullptr, nullptr); }, x, 1e-4);
            THEN("the relative difference is small, quantifying the omitted second-order term")
            {
                const double rel = (h - hFd).norm() / hFd.norm();
                CAPTURE(rel);
                CHECK(rel < 0.05);
            }
        }
    }
}

SCENARIO("MeasurementPointBundle: the support is the pose and the observed landmark blocks")
{
    GIVEN("the bundle of two observations")
    {
        const Camera cam = testCamera();
        MeasurementPointBundle meas(0.0, cam, observationsFrom(VectorXd::Constant(4, 500.0), 2.0));
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1));
        SystemEstimator sys(model, GaussianD::fromMoment(VectorXd::Zero(18), MatrixXd::Identity(18, 18)), 0.0);
        WHEN("the support is requested")
        {
            const auto support = meas.supportIndices(sys);
            THEN("it lists eta (6) then the three states of each observed landmark")
            {
                REQUIRE(support.size() == 6 + 3 * 2);
                CHECK(support.front() == 6);
                CHECK(support[5] == 11);
                CHECK(support[6] == 12);
                CHECK(support.back() == 17);
            }
        }
    }
}

SCENARIO("MeasurementPointBundle: the MAP recovers the body pose from exact pixels when the landmarks are known")
{
    GIVEN("six landmarks with tight priors at their true positions, exact pixels, and a wide prior on the pose")
    {
        // A single view cannot fix both the pose and the landmark depths, so the landmark priors are tight
        // (1 mm). With 12 pixel measurements for the 6 pose unknowns the pose is then well determined.
        const Camera cam = testCamera();
        const Index nl = 6;
        const Index n = 12 + 3 * nl;
        VectorXd truth = VectorXd::Zero(n);
        truth.segment<3>(6) = Vector3d(0.2, -0.1, 0.05);
        truth.segment<3>(9) = Vector3d(0.05, -0.08, 0.15);
        for (Index j = 0; j < nl; ++j)
        {
            truth.segment<3>(12 + 3 * j) = Vector3d(2.0 + 0.3 * static_cast<double>(j), 0.6 - 0.25 * static_cast<double>(j),
                                                    0.4 * std::sin(static_cast<double>(j)));
        }
        std::vector<MeasurementPointBundle::Observation> obs;
        for (Index j = 0; j < nl; ++j)
        {
            MeasurementPointBundle::Observation o;
            o.landmarkId = j;
            o.landmarkOffset = 12 + 3 * j;
            o.pixel = pointPixelJacobian(cam, truth.segment<3>(6), truth.segment<3>(9), truth.segment<3>(12 + 3 * j)).uv;
            o.sigmaPx = 2.0;
            obs.push_back(o);
        }
        MeasurementPointBundle meas(0.0, cam, obs);
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.1));
        // Prior mean: pose offset (2 mm, 2 mrad), landmarks exactly at the truth with 1 mm sigma, nu at 0 with unit sigma.
        // The MAP is a stationary point of the objective; the pull of the prior offset is amplified only along the
        // weakly observed (depth-like) pose directions, so it stays ~1e-5 m for this offset.
        VectorXd mu = truth;
        mu.segment<3>(6) += Vector3d(0.002, -0.002, 0.002);
        mu.segment<3>(9) += Vector3d(0.002, 0.002, -0.002);
        MatrixXd p = MatrixXd::Identity(n, n);
        p.block(6, 6, 6, 6) *= 0.25;
        p.bottomRightCorner(3 * nl, 3 * nl) *= 1e-6;
        SystemEstimator sys(model, GaussianD::fromMoment(mu, p), 0.0);
        WHEN("the measurement is processed (Laplace update at the MAP)")
        {
            const MeasurementUpdateResult res = meas.process(sys);
            THEN("the MAP is stationary (|grad| < 1e-6), recovers the pose to 1e-4 and the landmarks to 1e-4")
            {
                CAPTURE(static_cast<int>(res.map.status));
                CAPTURE(res.map.iterations);
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CHECK(res.map.g.cwiseAbs().maxCoeff() < 1e-6);
                CAPTURE_EIGEN((sys.mean() - truth).transpose());
                CHECK(nearMatrix(sys.mean().segment<6>(6), truth.segment<6>(6), 1e-4));
                for (Index j = 0; j < nl; ++j)
                {
                    CHECK(nearMatrix(sys.mean().segment<3>(12 + 3 * j), truth.segment<3>(12 + 3 * j), 1e-4));
                }
            }
            THEN("the posterior precision is positive definite and the pose uncertainty has shrunk")
            {
                CHECK(res.precisionIsExact);
                CHECK(sys.covariance().block(6, 6, 3, 3).trace() < p.block(6, 6, 3, 3).trace());
            }
        }
    }
}

SCENARIO("MeasurementPointBundle: invalid construction is rejected")
{
    const Camera cam = testCamera();
    THEN("an empty observation set is rejected")
    {
        CHECK_THROWS_AS(MeasurementPointBundle(0.0, cam, {}), std::invalid_argument);
    }
    THEN("a non-positive pixel sigma is rejected")
    {
        auto obs = observationsFrom(VectorXd::Constant(4, 100.0), 0.0);
        CHECK_THROWS_AS(MeasurementPointBundle(0.0, cam, obs), std::invalid_argument);
    }
    THEN("a non-finite observation pixel is rejected")
    {
        auto obs = observationsFrom(VectorXd::Constant(4, 100.0), 1.0);
        obs[0].pixel(0) = std::nan("");
        CHECK_THROWS_AS(MeasurementPointBundle(0.0, cam, obs), std::invalid_argument);
    }
}
