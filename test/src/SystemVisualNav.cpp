#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/GaussianInfo.h"
#include "../../src/rotation.hpp"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using GaussianD = GaussianInfo<double>;
using testutil::fdJacobian;
using testutil::nearMatrix;
using testutil::randomMatrix;
using testutil::randomSpd;
using testutil::randomVector;

namespace
{

/// Process-noise PSD of the nu random walk used in all tests.
Eigen::Matrix<double, 6, 1> testPsd()
{
    Eigen::Matrix<double, 6, 1> q;
    q << 0.04, 0.03, 0.05, 0.01, 0.02, 0.015;
    return q;
}

/// Independent reference for the 12 x 12 process-noise covariance Qd = G Cz G^T (explicit, no factorisation).
MatrixXd referenceQd(const VectorXd& x, const Eigen::Matrix<double, 6, 1>& q, double dt)
{
    const Vector3d theta = x.segment<3>(9);
    MatrixXd cz = MatrixXd::Zero(12, 12);
    for (Index i = 0; i < 6; ++i)
    {
        cz(i, i) = q(i) * dt;
        cz(i, 6 + i) = q(i) * dt * dt / 2.0;
        cz(6 + i, i) = q(i) * dt * dt / 2.0;
        cz(6 + i, 6 + i) = q(i) * dt * dt * dt / 3.0;
    }
    MatrixXd g = MatrixXd::Identity(12, 12);
    g.block<3, 3>(6, 6) = rpy2rot(theta);
    g.block<3, 3>(9, 9) = kinematicMatrixTK(theta);
    return g * cz * g.transpose();
}

/// Full-state process-noise covariance (landmarks and etaPrev are noise free).
MatrixXd referenceQFull(const VectorXd& x, const Eigen::Matrix<double, 6, 1>& q, double dt)
{
    MatrixXd qf = MatrixXd::Zero(x.size(), x.size());
    qf.topLeftCorner(12, 12) = referenceQd(x, q, dt);
    return qf;
}

/// Base state: nu, eta with a moderate attitude; optional point landmark (mean [4,1,0]).
GaussianD baseState(bool withLandmark, unsigned seed)
{
    VectorXd mu(12);
    mu << 0.5, 0.1, -0.2, 0.05, -0.03, 0.02,   // nu
          1.0, 2.0, -3.0, 0.1, -0.2, 0.4;      // eta: rBNn, Thetanb
    MatrixXd p = 0.05 * randomSpd(12, seed);
    if (withLandmark)
    {
        VectorXd m(15);
        m << mu, 4.0, 1.0, 0.0;
        MatrixXd pf = MatrixXd::Zero(15, 15);
        pf.topLeftCorner(12, 12) = p;
        pf.bottomRightCorner(3, 3) = 0.5 * Matrix3d::Identity();
        return GaussianD::fromMoment(m, pf);
    }
    return GaussianD::fromMoment(mu, p);
}

/// Point landmark density (mean [4,1,0], covariance 0.5 I).
GaussianD pointLandmarkDensity()
{
    return GaussianD::fromMoment(Vector3d(4.0, 1.0, 0.0), 0.5 * Matrix3d::Identity());
}

/// Estimator with a registered point landmark (12-D pose state + 3-D landmark, no etaPrev).
SystemEstimator makeSystemWithLandmark(const std::shared_ptr<SystemVisualNav>& model, unsigned seed)
{
    SystemEstimator est(model, baseState(false, seed), 0.0);
    model->addLandmark(est, SystemVisualNav::LandmarkType::Point, pointLandmarkDensity());
    return est;
}

/// Reference RK4-free flow Jacobian of the discrete map (finite differences of the same integrator).
MatrixXd fdTransition(const SystemBase& model, double t0, const VectorXd& x, double dt)
{
    return fdJacobian([&](const VectorXd& xx) { return SystemEstimator::integrateFlow(model, t0, xx, dt, nullptr); }, x,
                      1e-6);
}

} // namespace

SCENARIO("SystemVisualNav: the flow Jacobian matches central finite differences")
{
    GIVEN("the visual-navigation model with one point landmark")
    {
        SystemVisualNav model(testPsd());
        for (unsigned s = 0; s < 4; ++s)
        {
            VectorXd x = VectorXd::Zero(21);
            x.head(12) = randomVector(12, 1400 + s);
            x.segment<3>(9) = Vector3d(0.6 * x(9), 0.5 * x(10), 1.2 * x(11));   // attitude away from gimbal lock
            x.tail(9) = randomVector(9, 1500 + s);
            INFO("sample ", s, " x = ", x.transpose());
            WHEN("the analytic Jacobian is evaluated")
            {
                MatrixXd a;
                const VectorXd f = model.dynamics(0.0, x, &a);
                THEN("it equals the finite-difference Jacobian of the flow")
                {
                    CHECK(f.allFinite());
                    const MatrixXd aFd = fdJacobian([&](const VectorXd& xx) { return model.dynamics(0.0, xx, nullptr); }, x);
                    CAPTURE_EIGEN(a.block(6, 0, 6, 21));
                    CHECK(nearMatrix(a, aFd, 1e-6));
                }
                THEN("the static states have zero flow and zero Jacobian rows")
                {
                    CHECK(f.tail(9).isZero(0.0));
                    CHECK(a.bottomRows(9).isZero(0.0));
                    CHECK(a.topRows(6).isZero(0.0));
                }
            }
        }
    }
}

SCENARIO("SystemVisualNav: the discrete state-transition matrix matches finite differences of the RK4 map")
{
    GIVEN("a model and a moderate time step (5 RK4 sub-steps)")
    {
        SystemVisualNav model(testPsd());
        const double dt = 0.05;
        for (unsigned s = 0; s < 3; ++s)
        {
            VectorXd x = VectorXd::Zero(21);
            x.head(12) = randomVector(12, 1600 + s);
            x.segment<3>(9) = Vector3d(0.5 * x(9), -0.4 * x(10), 0.9 * x(11));
            x.tail(9) = randomVector(9, 1700 + s);
            INFO("sample ", s);
            WHEN("the STM from the integrator is compared with finite differences")
            {
                MatrixXd phi;
                SystemEstimator::integrateFlow(model, 0.0, x, dt, &phi);
                const MatrixXd phiFd = fdTransition(model, 0.0, x, dt);
                THEN("the STM agrees with the finite-difference Jacobian of the discrete map")
                {
                    CAPTURE_EIGEN(phi.topLeftCorner(12, 12));
                    CHECK(nearMatrix(phi, phiFd, 1e-6));
                }
                THEN("static states map to themselves")
                {
                    CHECK(nearMatrix(phi.bottomRightCorner(9, 9), MatrixXd::Identity(9, 9), 1e-12));
                }
            }
        }
    }
}

SCENARIO("SystemVisualNav: the process-noise square-root information factorises the full-rank Qd^{-1}")
{
    GIVEN("the model and a state with a moderate attitude")
    {
        SystemVisualNav model(testPsd());
        VectorXd x = VectorXd::Zero(18);
        x.head(12) << 0.5, 0.1, -0.2, 0.05, -0.03, 0.02, 1.0, 2.0, -3.0, 0.1, -0.2, 0.4;
        const double dt = 0.04;
        WHEN("the square-root information L is requested")
        {
            const MatrixXd l = model.processNoiseSqrtInfo(dt, x);
            const MatrixXd qd = referenceQd(x, testPsd(), dt);
            THEN("L is 12 x 12, upper triangular and L^T L = Qd^{-1}")
            {
                CHECK(l.rows() == 12);
                CHECK(l.cols() == 12);
                CHECK(l.triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
                CHECK(nearMatrix(l.transpose() * l * qd, MatrixXd::Identity(12, 12), 1e-8));
            }
            THEN("the discrete noise is full rank (finite condition number of Qd)")
            {
                const Eigen::SelfAdjointEigenSolver<MatrixXd> es(qd);
                CHECK(es.eigenvalues().minCoeff() > 0.0);
                CHECK(es.eigenvalues().maxCoeff() / es.eigenvalues().minCoeff() < 1e8);
            }
        }
        WHEN("a non-positive time step is requested")
        {
            THEN("the request is rejected")
            {
                CHECK_THROWS_AS(model.processNoiseSqrtInfo(0.0, x), std::invalid_argument);
            }
        }
    }
    GIVEN("a non-positive intensity on one axis")
    {
        Eigen::Matrix<double, 6, 1> bad = testPsd();
        bad(3) = 0.0;
        THEN("construction is rejected because Qd would be singular")
        {
            CHECK_THROWS_AS(SystemVisualNav model(bad), std::invalid_argument);
        }
    }
}

SCENARIO("SystemVisualNav: information-form prediction matches a moment-form EKF over several steps")
{
    GIVEN("a point-landmark state with covariance and the model")
    {
        auto model = std::make_shared<SystemVisualNav>(testPsd());
        SystemEstimator est = makeSystemWithLandmark(model, 1801);
        VectorXd muRef = est.mean();
        MatrixXd pRef = est.covariance();
        const double dt = 0.05;
        WHEN("three plain prediction steps are taken")
        {
            for (int k = 0; k < 3; ++k)
            {
                // EKF reference: mean by the flow, covariance by the linearised transition plus Qd at the mean.
                const MatrixXd phi = fdTransition(*model, est.time(), muRef, dt);
                const MatrixXd qf = referenceQFull(muRef, testPsd(), dt);
                const VectorXd muNext = SystemEstimator::integrateFlow(*model, est.time(), muRef, dt, nullptr);
                pRef = phi * pRef * phi.transpose() + qf;
                muRef = muNext;

                model->predictPlain(est, est.time() + dt);
            }
            THEN("mean and covariance equal the EKF moment-form reference")
            {
                CAPTURE_EIGEN(est.mean());
                CHECK(nearMatrix(est.mean(), muRef, 1e-7));
                CHECK(nearMatrix(est.covariance(), pRef, 1e-6));
            }
            THEN("the layout is unchanged and the time has advanced")
            {
                CHECK(est.dim() == 15);
                CHECK(est.time() == doctest::Approx(3 * dt));
                CHECK(model->landmarks().size() == 1);
            }
        }
    }
}

SCENARIO("SystemVisualNav: a plain prediction leaves landmark marginals unchanged")
{
    GIVEN("a state with a point landmark and its marginal")
    {
        auto model = std::make_shared<SystemVisualNav>(testPsd());
        SystemEstimator est = makeSystemWithLandmark(model, 1901);
        const GaussianD before = est.state().marginal({12, 13, 14});
        WHEN("a plain prediction is applied")
        {
            model->predictPlain(est, 0.1);
            const GaussianD after = est.state().marginal({12, 13, 14});
            THEN("the landmark mean and covariance are unchanged")
            {
                CHECK(nearMatrix(after.mean(), before.mean(), 1e-9));
                CHECK(nearMatrix(after.covariance(), before.covariance(), 1e-9));
            }
        }
    }
}

SCENARIO("SystemVisualNav: delayed-state shift stores eta_k as etaPrev with the correct correlations")
{
    GIVEN("a point-landmark state without a delayed pose")
    {
        auto model = std::make_shared<SystemVisualNav>(testPsd());
        SystemEstimator est = makeSystemWithLandmark(model, 2001);
        const VectorXd mu = est.mean();
        const MatrixXd p = est.covariance();
        const double dt = 0.05;
        WHEN("a delayed prediction to t = dt is taken")
        {
            const MatrixXd phiS = fdTransition(*model, 0.0, mu, dt).topRows(12);
            const VectorXd muNext = SystemEstimator::integrateFlow(*model, 0.0, mu, dt, nullptr);
            const MatrixXd qd = referenceQd(mu, testPsd(), dt);

            // Joint covariance over [x_k (n) ; x_{k+1,S} (12)], then keep [x_{k+1,S}; eta_k; landmark].
            const Index n = est.dim();
            MatrixXd joint = MatrixXd::Zero(n + 12, n + 12);
            joint.topLeftCorner(n, n) = p;
            joint.topRightCorner(n, 12) = p * phiS.transpose();
            joint.bottomLeftCorner(12, n) = phiS * p;
            joint.bottomRightCorner(12, 12) = phiS * p * phiS.transpose() + qd;
            std::vector<Index> sel;
            for (Index i = 0; i < 12; ++i)
            {
                sel.push_back(n + i);
            }
            for (Index i = 6; i < 12; ++i)
            {
                sel.push_back(i);
            }
            for (Index i = 12; i < n; ++i)
            {
                sel.push_back(i);
            }
            MatrixXd pRef(sel.size(), sel.size());
            for (std::size_t i = 0; i < sel.size(); ++i)
            {
                for (std::size_t j = 0; j < sel.size(); ++j)
                {
                    pRef(static_cast<Index>(i), static_cast<Index>(j)) = joint(sel[i], sel[j]);
                }
            }
            VectorXd muRef(sel.size());
            for (std::size_t i = 0; i < sel.size(); ++i)
            {
                const Index s = sel[i];
                muRef(static_cast<Index>(i)) = s >= n ? muNext(s - n) : mu(s);
            }

            model->predictDelayed(est, dt);
            THEN("the state layout gains an etaPrev block of dimension 6")
            {
                CHECK(model->hasDelayedPose());
                CHECK(est.dim() == 21);
                CHECK(model->landmarkOffset(model->landmarks()[0].id) == 18);
            }
            THEN("etaPrev equals the pre-update eta with the joint covariance, including the cross-covariances")
            {
                CAPTURE_EIGEN(est.covariance().block(6, 12, 6, 6));
                CHECK(nearMatrix(est.mean(), muRef, 1e-7));
                CHECK(nearMatrix(est.covariance(), pRef, 1e-6));
                CHECK(est.covariance().block(6, 12, 6, 6).cwiseAbs().maxCoeff() > 1e-6);
            }
            THEN("a second delayed prediction replaces etaPrev and keeps the layout")
            {
                model->predictDelayed(est, 2 * dt);
                CHECK(est.dim() == 21);
                CHECK(model->hasDelayedPose());
            }
            THEN("discarding the delayed pose restores the 12 + landmark layout")
            {
                model->discardDelayedPose(est);
                CHECK(!model->hasDelayedPose());
                CHECK(est.dim() == 15);
                CHECK(model->landmarkOffset(model->landmarks()[0].id) == 12);
            }
        }
    }
}

SCENARIO("SystemVisualNav: a delayed-state shift at zero time step copies eta_k exactly in the mean")
{
    GIVEN("a point-landmark state at t = 0 with a non-trivial correlated covariance")
    {
        auto model = std::make_shared<SystemVisualNav>(testPsd());
        SystemEstimator est = makeSystemWithLandmark(model, 2301);
        const VectorXd mu = est.mean();
        const MatrixXd p = est.covariance();
        WHEN("a delayed prediction is requested with dt = 0 (zero-step semantics, see kDelayedCopyStd)")
        {
            model->predictDelayed(est, 0.0);
            THEN("the layout gains etaPrev, and nu, eta and the landmark keep their means")
            {
                CHECK(model->hasDelayedPose());
                CHECK(est.dim() == 21);
                CHECK(model->landmarkOffset(model->landmarks()[0].id) == 18);
                CHECK(nearMatrix(est.mean().head(12), mu.head(12), 1e-9));
                CHECK(nearMatrix(est.mean().segment(12, 6), mu.segment(6, 6), 1e-9));
                CHECK(nearMatrix(est.mean().tail(3), mu.tail(3), 1e-9));
            }
            THEN("the marginals of nu, eta and the landmark are unchanged")
            {
                CHECK(nearMatrix(est.covariance().topLeftCorner(12, 12), p.topLeftCorner(12, 12), 1e-9));
                CHECK(nearMatrix(est.covariance().block(12, 12, 6, 6), p.block(6, 6, 6, 6) + 1e-8 * MatrixXd::Identity(6, 6),
                                 1e-9));
            }
            THEN("etaPrev is perfectly correlated with eta up to the copy jitter (cross-covariance equals P_eta)")
            {
                CHECK(nearMatrix(est.covariance().block(6, 12, 6, 6), p.block(6, 6, 6, 6), 1e-9));
            }
            THEN("the factor stays finite and upper triangular")
            {
                CHECK(est.state().sqrtInfo().allFinite());
                CHECK(est.state().sqrtInfo().triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
            }
        }
        WHEN("a plain prediction is requested with dt = 0")
        {
            THEN("it is a no-op")
            {
                const VectorXd before = est.mean();
                const MatrixXd xiBefore = est.state().sqrtInfo();
                model->predictPlain(est, 0.0);
                CHECK(nearMatrix(est.mean(), before, 1e-12));
                CHECK(est.dim() == 15);
                CHECK(nearMatrix(est.state().sqrtInfo(), xiBefore, 1e-14));
            }
        }
    }
}

SCENARIO("SystemVisualNav: landmark augmentation and removal preserve the marginals")
{
    GIVEN("a model with an empty landmark set and a 12-D state")
    {
        auto model = std::make_shared<SystemVisualNav>(testPsd());
        SystemEstimator est(model, baseState(false, 2101), 0.0);
        const MatrixXd p = est.covariance();
        WHEN("a point landmark is added")
        {
            const GaussianD lm = GaussianD::fromMoment(Vector3d(1.0, 2.0, 3.0), 0.3 * Matrix3d::Identity());
            const Index id = model->addLandmark(est, SystemVisualNav::LandmarkType::Point, lm);
            THEN("the state dimension grows by 3 and the new block is the initial density")
            {
                CHECK(est.dim() == 15);
                CHECK(model->landmarkOffset(id) == 12);
                CHECK(nearMatrix(est.state().marginal({12, 13, 14}).mean(), Vector3d(1.0, 2.0, 3.0), 1e-9));
                CHECK(nearMatrix(est.state().marginal({12, 13, 14}).covariance(), 0.3 * Matrix3d::Identity(), 1e-9));
                CHECK(nearMatrix(est.covariance().topLeftCorner(12, 12), p, 1e-9));
                CHECK(est.covariance().block(0, 12, 12, 3).isZero(1e-12));
            }
            AND_WHEN("a pose landmark is added and then the point landmark is removed")
            {
                const GaussianD pose = GaussianD::fromMoment(VectorXd::Constant(6, 0.2), 0.1 * MatrixXd::Identity(6, 6));
                const Index poseId = model->addLandmark(est, SystemVisualNav::LandmarkType::Pose, pose);
                const MatrixXd before = est.covariance();
                model->removeLandmark(est, id);
                THEN("the pose block moves to offset 12 and its covariance is preserved")
                {
                    CHECK(est.dim() == 18);
                    CHECK(model->landmarkOffset(poseId) == 12);
                    CHECK(nearMatrix(est.covariance().bottomRightCorner(6, 6), before.bottomRightCorner(6, 6), 1e-9));
                    CHECK(nearMatrix(est.covariance().topLeftCorner(12, 12), p, 1e-9));
                }
                THEN("removing an unknown landmark is rejected")
                {
                    CHECK_THROWS_AS(model->removeLandmark(est, id), std::out_of_range);
                }
            }
        }
        WHEN("a landmark with the wrong dimension is added")
        {
            const GaussianD bad = GaussianD::fromMoment(VectorXd::Zero(2), MatrixXd::Identity(2, 2));
            THEN("it is rejected")
            {
                CHECK_THROWS_AS(model->addLandmark(est, SystemVisualNav::LandmarkType::Point, bad), std::invalid_argument);
            }
        }
    }
}

SCENARIO("SystemVisualNav: long prediction keeps values finite and the information factor well conditioned")
{
    GIVEN("a pose-landmark state")
    {
        auto model = std::make_shared<SystemVisualNav>(testPsd());
        SystemEstimator est(model, baseState(false, 2201), 0.0);
        const GaussianD pose = GaussianD::fromMoment(VectorXd::Zero(6), 0.2 * MatrixXd::Identity(6, 6));
        model->addLandmark(est, SystemVisualNav::LandmarkType::Pose, pose);
        WHEN("50 delayed and plain predictions are alternated at 30 Hz")
        {
            for (int k = 1; k <= 50; ++k)
            {
                const double t = est.time() + 1.0 / 30.0;
                if (k % 2 == 0)
                {
                    model->predictDelayed(est, t);
                }
                else
                {
                    model->predictPlain(est, t);
                }
            }
            THEN("all quantities are finite and the diagonal of Xi spans a moderate range")
            {
                CHECK(est.mean().allFinite());
                CHECK(est.state().sqrtInfo().allFinite());
                const VectorXd d = est.state().sqrtInfo().diagonal().cwiseAbs();
                CHECK(d.minCoeff() > 0.0);
                CHECK(d.maxCoeff() / d.minCoeff() < 1e7);
                CHECK(est.dim() == 18 + 6);
            }
        }
    }
}

SCENARIO("SystemVisualNav (R4-17): the flow and the noise factor refuse a body pitch beyond 85 degrees")
{
    GIVEN("the shared model and a state whose body pitch is 89.9 degrees")
    {
        const SystemVisualNav model(testPsd());
        VectorXd x = VectorXd::Zero(SystemVisualNav::kDynamicDim);
        x(10) = 89.9 * std::numbers::pi / 180.0;
        WHEN("the flow and its Jacobian are evaluated")
        {
            MatrixXd a;
            THEN("a documented domain error is thrown instead of an ill-conditioned Jacobian")
            {
                CHECK_THROWS_AS(model.dynamics(0.0, x, &a), std::domain_error);
            }
        }
        WHEN("the process-noise factor is requested")
        {
            THEN("a documented domain error is thrown")
            {
                CHECK_THROWS_AS(model.processNoiseSqrtInfo(0.1, x), std::domain_error);
            }
        }
    }
    GIVEN("the same model at a pitch of 80 degrees")
    {
        const SystemVisualNav model(testPsd());
        VectorXd x = VectorXd::Zero(SystemVisualNav::kDynamicDim);
        x(10) = 80.0 * std::numbers::pi / 180.0;
        THEN("the flow and the noise factor are finite")
        {
            MatrixXd a;
            CHECK(model.dynamics(0.0, x, &a).allFinite());
            CHECK(a.allFinite());
            CHECK(model.processNoiseSqrtInfo(0.1, x).allFinite());
        }
    }
}
