#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <Eigen/Core>

#include "../../src/Camera.h"
#include "../../src/GaussianBase.h"
#include "../../src/GaussianInfo.h"
#include "../../src/IndoorFlowFrontEnd.h"
#include "../../src/MeasurementIndoorFlow.h"
#include "../../src/Pose.hpp"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Index;
using slam::IndoorFlowFrame;
using slam::IndoorFlowModel;
using slam::IndoorFlowOptions;
using slam::IndoorFlowPair;
using slam::IndoorFlowPixelMatch;
using slam::MeasurementIndoorFlow;
using slam::buildIndoorFlowFrame;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using testutil::fdGradient;
using testutil::fdHessian;
using testutil::fdJacobian;

namespace
{

constexpr double kF = 500.0;

Camera flowCamera()
{
    return Camera(640, 480, (Matrix3d() << kF, 0.0, 320.0, 0.0, kF, 240.0, 0.0, 0.0, 1.0).finished(), {});
}

/// 18-state vector: eta_k = [rB; ThB] at 6..11, etaPrev = [rA; ThA] at 12..17.
VectorXd makeState(const Vector3d& rB, const Vector3d& thB, const Vector3d& rA, const Vector3d& thA)
{
    VectorXd x = VectorXd::Zero(18);
    x.segment<3>(6) = rB;
    x.segment<3>(9) = thB;
    x.segment<3>(12) = rA;
    x.segment<3>(15) = thA;
    return x;
}

/// Estimator of dimension 18 whose mean is x (only used to pass the dimension and the mean to the measurement).
SystemEstimator dummyEstimator(const VectorXd& x)
{
    const Eigen::Matrix<double, 6, 1> psd = Eigen::Matrix<double, 6, 1>::Constant(0.1);
    auto model = std::make_shared<SystemVisualNav>(psd);
    return SystemEstimator(model, GaussianInfo<double>::fromMoment(x, MatrixXd::Identity(18, 18)), 0.0);
}

/// Noise-free correspondences of a static scene: points in front of camera A at depths in [zMin, zMax],
/// projected into camera B with the rigid-body model (undistorted normalised coordinates).
std::vector<IndoorFlowPair> syntheticPairs(const VectorXd& x, const Matrix3d& Rbc, const Vector3d& rCBb, int n,
                                           unsigned seed, double zMin = 2.0, double zMax = 5.0)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uv(-0.45, 0.45);
    std::uniform_real_distribution<double> z(zMin, zMax);
    const Matrix3d RcA = rigid::cameraRotationInWorld(x.segment<3>(15), Rbc);
    const Matrix3d RcB = rigid::cameraRotationInWorld(x.segment<3>(9), Rbc);
    const Vector3d cA = rigid::cameraOriginInWorld(x.segment<3>(15), x.segment<3>(12), rCBb);
    const Vector3d cB = rigid::cameraOriginInWorld(x.segment<3>(9), x.segment<3>(6), rCBb);
    std::vector<IndoorFlowPair> pairs;
    while (static_cast<int>(pairs.size()) < n)
    {
        const Vector3d pA = Vector3d(uv(rng), uv(rng), 1.0) * z(rng);
        const Vector3d P = RcA * pA + cA;
        const Vector3d pB = RcB.transpose() * (P - cB);
        if (pB.z() <= 0.2)
        {
            continue;
        }
        pairs.push_back({Vector2d(pA.x() / pA.z(), pA.y() / pA.z()), Vector2d(pB.x() / pB.z(), pB.y() / pB.z())});
    }
    return pairs;
}

/// Adds i.i.d. Gaussian noise (std sigmaNorm) to every normalised coordinate.
void addNoise(std::vector<IndoorFlowPair>& pairs, double sigmaNorm, unsigned seed, bool bothImages)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, sigmaNorm);
    for (IndoorFlowPair& p : pairs)
    {
        p.currNorm += Vector2d(g(rng), g(rng));
        if (bothImages)
        {
            p.prevNorm += Vector2d(g(rng), g(rng));
        }
    }
}

struct FlowState
{
    const char* name;
    VectorXd x;
};

/// Representative states: general motion, pure rotation with a tiny translation, large attitude, and a
/// small-parallax pan with a non-zero lever arm.
std::vector<FlowState> representativeStates()
{
    return {
        {"general motion", makeState(Vector3d(0.30, -0.10, 0.05), Vector3d(0.10, -0.05, 0.20),
                                     Vector3d(0.10, 0.05, 0.00), Vector3d(0.02, 0.03, 0.10))},
        {"near pure rotation (|t| = 1 mm)", makeState(Vector3d(1.0e-3, 0.0, 0.0), Vector3d(0.00, 0.02, 0.15),
                                                      Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0))},
        {"near pure rotation, large yaw", makeState(Vector3d(0.20, 0.10, 0.00), Vector3d(0.05, -0.02, 0.90),
                                                    Vector3d(0.20, 0.10, 0.00) + Vector3d(0.0, 2e-4, 0.0),
                                                    Vector3d(0.05, -0.02, 0.85))},
    };
}

} // namespace

SCENARIO("Indoor flow: Jacobians of the epipolar and rotation residuals match central differences")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb(0.02, -0.01, 0.03);
    for (const FlowState& s : representativeStates())
    {
        CAPTURE(s.name);
        // Noisy correspondences: the Jacobian is checked at the generating state, away from zero residual.
        std::vector<IndoorFlowPair> pairs = syntheticPairs(s.x, Rbc, rCBb, 25, 7u);
        addNoise(pairs, 2e-3, 9u, true);
        for (IndoorFlowModel model : {IndoorFlowModel::Epipolar, IndoorFlowModel::Rotation})
        {
            const std::string modelName = model == IndoorFlowModel::Epipolar ? "epipolar" : "rotation";
            CAPTURE(modelName);
            GIVEN("a measurement on this state")
            {
                MeasurementIndoorFlow meas(0.0, model, pairs, 2e-3, Rbc, rCBb);
                const auto f = [&meas](const VectorXd& x) { return meas.whitenedResiduals(x, nullptr); };
                MatrixXd Jan;
                meas.whitenedResiduals(s.x, &Jan);
                const MatrixXd Jfd = fdJacobian(f, s.x, 1e-7);
                WHEN("the analytic Jacobian (12 support states) is compared with central differences")
                {
                    const MatrixXd Jsupp = Jfd.middleCols(6, 12);
                    CAPTURE(testutil::toString(Jan.topLeftCorner(3, 12)));
                    const double scale = std::max(1.0, Jsupp.cwiseAbs().maxCoeff());
                    THEN("they agree to 1e-5 relative to the largest entry")
                    {
                        CHECK((Jan - Jsupp).cwiseAbs().maxCoeff() <= 1e-5 * scale);
                    }
                }
            }
        }
    }
}

SCENARIO("Indoor flow: gradient and Gauss-Newton Hessian of the log-likelihood")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb(0.02, -0.01, 0.03);
    for (const FlowState& s : representativeStates())
    {
        CAPTURE(s.name);
        SystemEstimator est = dummyEstimator(s.x);
        GIVEN("noise-free correspondences at the generating state (zero residual)")
        {
            for (IndoorFlowModel model : {IndoorFlowModel::Epipolar, IndoorFlowModel::Rotation})
            {
                const std::string modelName = model == IndoorFlowModel::Epipolar ? "epipolar" : "rotation";
                CAPTURE(modelName);
                // The rotation model is exact only for zero translation: its zero-residual state is a pure pan.
                VectorXd xs = s.x;
                if (model == IndoorFlowModel::Rotation)
                {
                    xs.segment<3>(6) = xs.segment<3>(12);   // rB := rA: pure pan
                }
                const std::vector<IndoorFlowPair> pairs = syntheticPairs(xs, Rbc, Vector3d::Zero(), 25, 13u);
                MeasurementIndoorFlow meas(0.0, model, pairs, 2e-3, Rbc, Vector3d::Zero());
                SystemEstimator estS = dummyEstimator(xs);
                // Quadratic part of the log-likelihood: the normalising constant (about -m log sigma) would dominate the
                // rounding of the finite differences; its derivatives are zero.
                const auto logL = [&](const VectorXd& x)
                {
                    return -0.5 * meas.whitenedResiduals(x, nullptr).squaredNorm();
                };
                VectorXd g;
                MatrixXd H;
                meas.logLikelihood(xs, estS, &g, &H);
                WHEN("the gradient and the Hessian are compared with central differences")
                {
                    // The step must be much smaller than the baseline |t| (the curvature in t varies on that scale).
                    const VectorXd gfd = fdGradient(logL, xs, 1e-7);
                    const MatrixXd Hfd = fdHessian(logL, xs, 1e-6);
                    THEN("the gradient matches and the Gauss-Newton Hessian is exact at zero residual")
                    {
                        CHECK((g - gfd).cwiseAbs().maxCoeff() <= 1e-4 * std::max(1.0, gfd.cwiseAbs().maxCoeff()));
                        CHECK((H.block(6, 6, 12, 12) - Hfd.block(6, 6, 12, 12)).cwiseAbs().maxCoeff()
                              <= 1e-3 * std::max(1.0, Hfd.cwiseAbs().maxCoeff()));
                    }
                    THEN("the residual is zero at the truth")
                    {
                        CHECK(meas.whitenedResiduals(xs, nullptr).cwiseAbs().maxCoeff() < 1e-8);
                    }
                }
            }
        }
    }
}

SCENARIO("Indoor flow: NIS of noisy two-view data is close to the number of degrees of freedom")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb(0.0, 0.0, 0.0);
    const VectorXd x = makeState(Vector3d(0.12, -0.05, 0.03), Vector3d(0.03, 0.02, 0.12), Vector3d(0.0, 0.0, 0.0),
                                 Vector3d(0.0, 0.0, 0.0));
    const double sigmaNorm = 1.0 / kF;   // 1 px
    GIVEN("300 correspondences with 1 px noise on both images (epipolar) and on the current image (rotation)")
    {
        std::vector<IndoorFlowPair> clean = syntheticPairs(x, Rbc, rCBb, 300, 21u);
        VectorXd xPan = x;
        xPan.segment<3>(12) = xPan.segment<3>(6);   // pure pan for the rotation model (zero translation)
        std::vector<IndoorFlowPair> cleanPan = syntheticPairs(xPan, Rbc, rCBb, 300, 22u);
        WHEN("the mean squared whitened residual is computed")
        {
            std::vector<IndoorFlowPair> epi = clean;
            addNoise(epi, sigmaNorm, 5u, true);
            const MeasurementIndoorFlow me(0.0, IndoorFlowModel::Epipolar, epi, sigmaNorm, Rbc, rCBb);
            const VectorXd re = me.whitenedResiduals(x, nullptr);
            const double nisE = re.squaredNorm() / static_cast<double>(re.size());

            std::vector<IndoorFlowPair> rot = cleanPan;
            addNoise(rot, sigmaNorm, 6u, false);
            const MeasurementIndoorFlow mr(0.0, IndoorFlowModel::Rotation, rot, sigmaNorm, Rbc, rCBb);
            const VectorXd rr = mr.whitenedResiduals(xPan, nullptr);
            const double nisR = rr.squaredNorm() / static_cast<double>(rr.size());
            THEN("both are close to one (the Sampson residual is calibrated to the pixel noise)")
            {
                CHECK(nisE > 0.8);
                CHECK(nisE < 1.25);
                CHECK(nisR > 0.85);
                CHECK(nisR < 1.15);
            }
        }
    }
}

SCENARIO("Indoor flow: the epipolar residual does not depend on the translation magnitude")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb = Vector3d::Zero();   // exact scale invariance requires cA - cB proportional to the baseline
    const VectorXd x1 = makeState(Vector3d(0.10, -0.05, 0.02), Vector3d(0.04, 0.01, 0.12), Vector3d::Zero(),
                                  Vector3d::Zero());
    VectorXd x2 = x1;
    x2.segment<3>(6) = 3.0 * x1.segment<3>(6);   // same rotations, translation x3
    GIVEN("correspondences generated at x1 and the state x2 with a three times longer baseline")
    {
        const std::vector<IndoorFlowPair> pairs = syntheticPairs(x1, Rbc, rCBb, 40, 31u);
        const MeasurementIndoorFlow m(0.0, IndoorFlowModel::Epipolar, pairs, 1e-3, Rbc, rCBb);
        WHEN("the residuals are evaluated at both states")
        {
            const VectorXd r1 = m.whitenedResiduals(x1, nullptr);
            const VectorXd r2 = m.whitenedResiduals(x2, nullptr);
            THEN("they are identical: the scale of t is not observable from the epipolar residual")
            {
                CHECK((r1 - r2).cwiseAbs().maxCoeff() < 1e-9);
                CHECK(r1.cwiseAbs().maxCoeff() < 1e-8);
            }
        }
    }
}

SCENARIO("Indoor flow: pure rotation (t = 0) has no epipolar information and no NaN")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb(0.02, -0.01, 0.03);
    const VectorXd x = makeState(Vector3d(0.3, 0.1, 0.0), Vector3d(0.02, 0.03, 0.2), Vector3d(0.3, 0.1, 0.0),
                                 Vector3d(0.01, -0.02, 0.1));   // same position: t = 0 when rCBb = 0 (see below)
    const Vector3d rCBbZero = Vector3d::Zero();   // zero lever arm: identical positions give exactly t = 0
    GIVEN("correspondences of a pure rotation with identical positions in A and B")
    {
        const std::vector<IndoorFlowPair> pairs = syntheticPairs(x, Rbc, rCBbZero, 20, 41u);
        const MeasurementIndoorFlow m(0.0, IndoorFlowModel::Epipolar, pairs, 1e-3, Rbc, rCBbZero);
        WHEN("the residuals and the Jacobian are evaluated")
        {
            MatrixXd J;
            const VectorXd r = m.whitenedResiduals(x, &J);
            THEN("they are finite and zero: the epipolar constraint degenerates, the rotation model carries the information")
            {
                CHECK(r.allFinite());
                CHECK(J.allFinite());
                CHECK(r.cwiseAbs().maxCoeff() < 1e-6);
            }
        }
    }
}

SCENARIO("Indoor flow: the rotation model explains a pure pan, the epipolar model explains parallax")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb(0.02, -0.01, 0.03);
    const VectorXd panState = makeState(Vector3d(0.2, 0.0, 0.0), Vector3d(0.0, 0.02, 0.10), Vector3d(0.2, 0.0, 0.0),
                                        Vector3d(0.0, 0.02, 0.08));
    const VectorXd translatedState = makeState(Vector3d(0.5, 0.0, 0.0), Vector3d(0.0, 0.02, 0.10),
                                               Vector3d(0.2, 0.0, 0.0), Vector3d(0.0, 0.02, 0.08));
    GIVEN("a pure pan and a 0.3 m lateral translation, both with the prior equal to the truth")
    {
        std::vector<IndoorFlowPixelMatch> panMatches;
        std::vector<IndoorFlowPixelMatch> transMatches;
        const Camera cam = flowCamera();
        const auto toPx = [](const Vector2d& n) { return Vector2d(kF * n.x() + 320.0, kF * n.y() + 240.0); };
        for (const IndoorFlowPair& p : syntheticPairs(panState, Rbc, rCBb, 60, 51u))
        {
            panMatches.push_back({toPx(p.prevNorm), toPx(p.currNorm)});
        }
        for (const IndoorFlowPair& p : syntheticPairs(translatedState, Rbc, rCBb, 60, 52u))
        {
            transMatches.push_back({toPx(p.prevNorm), toPx(p.currNorm)});
        }
        IndoorFlowOptions opt;
        WHEN("the front end builds the frames with the predicted state equal to the truth")
        {
            SystemEstimator estPan = dummyEstimator(panState);
            const IndoorFlowFrame fPan = buildIndoorFlowFrame(cam, estPan, panMatches, opt);
            SystemEstimator estTr = dummyEstimator(translatedState);
            const IndoorFlowFrame fTr = buildIndoorFlowFrame(cam, estTr, transMatches, opt);
            THEN("the pan selects the rotation model and the translation selects the epipolar model")
            {
                CHECK(fPan.valid);
                CHECK(fPan.model == IndoorFlowModel::Rotation);
                CHECK(fTr.valid);
                CHECK(fTr.model == IndoorFlowModel::Epipolar);
            }
        }
    }
}

SCENARIO("Indoor flow: RANSAC and the gate reject outlier correspondences")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb(0.02, -0.01, 0.03);
    const VectorXd x = makeState(Vector3d(0.30, -0.10, 0.05), Vector3d(0.10, -0.05, 0.20), Vector3d(0.10, 0.05, 0.00),
                                 Vector3d(0.02, 0.03, 0.10));
    GIVEN("60 consistent correspondences plus 20 random outliers (moving objects, mismatches)")
    {
        const Camera cam = flowCamera();
        const auto toPx = [](const Vector2d& n) { return Vector2d(kF * n.x() + 320.0, kF * n.y() + 240.0); };
        std::vector<IndoorFlowPixelMatch> matches;
        for (const IndoorFlowPair& p : syntheticPairs(x, Rbc, rCBb, 60, 61u))
        {
            matches.push_back({toPx(p.prevNorm), toPx(p.currNorm)});
        }
        std::mt19937 rng(62u);
        std::uniform_real_distribution<double> uPx(20.0, 620.0);
        std::uniform_real_distribution<double> vPx(20.0, 460.0);
        for (int i = 0; i < 20; ++i)
        {
            matches.push_back({Vector2d(uPx(rng), vPx(rng)), Vector2d(uPx(rng), vPx(rng))});
        }
        WHEN("the front end builds the frame at the truth")
        {
            SystemEstimator est = dummyEstimator(x);
            IndoorFlowOptions opt;
            const IndoorFlowFrame f = buildIndoorFlowFrame(cam, est, matches, opt);
            THEN("the frame is valid, keeps (almost) all inliers and rejects most outliers")
            {
                CHECK(f.valid);
                CHECK(f.input == 80);
                CHECK(f.gated >= 55);
                CHECK(f.gated <= 66);
            }
        }
    }
}

SCENARIO("Indoor flow: invalid inputs are rejected or give no update")
{
    const Matrix3d Rbc = rigid::defaultRbc();
    const Vector3d rCBb(0.0, 0.0, 0.0);
    const VectorXd x = makeState(Vector3d(0.1, 0.0, 0.0), Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero());
    GIVEN("an empty correspondence set and invalid noise")
    {
        THEN("the measurement constructor rejects empty pairs and non-positive sigma")
        {
            CHECK_THROWS_AS(MeasurementIndoorFlow(0.0, IndoorFlowModel::Epipolar, {}, 1e-3, Rbc, rCBb),
                            std::invalid_argument);
            const std::vector<IndoorFlowPair> pairs = syntheticPairs(x, Rbc, rCBb, 5, 71u);
            CHECK_THROWS_AS(MeasurementIndoorFlow(0.0, IndoorFlowModel::Epipolar, pairs, 0.0, Rbc, rCBb),
                            std::invalid_argument);
        }
        THEN("the front end gives no update for too few matches")
        {
            const Camera cam = flowCamera();
            SystemEstimator est = dummyEstimator(x);
            const IndoorFlowFrame f = buildIndoorFlowFrame(cam, est, {}, IndoorFlowOptions{});
            CHECK_FALSE(f.valid);
            CHECK(f.input == 0);
        }
    }
}
