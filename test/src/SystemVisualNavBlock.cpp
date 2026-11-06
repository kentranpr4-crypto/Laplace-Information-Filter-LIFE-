// Block-wise time update (SystemVisualNav::dynamicBlockDim) against the full n x n state-transition reference.
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/GaussianInfo.h"
#include "../../src/SystemBase.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "testUtil.h"

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using GaussianD = GaussianInfo<double>;
using testutil::fdJacobian;
using testutil::nearMatrix;
using testutil::randomSpd;
using testutil::randomVector;

namespace
{

Eigen::Matrix<double, 6, 1> blockPsd()
{
    Eigen::Matrix<double, 6, 1> q;
    q << 0.04, 0.03, 0.05, 0.01, 0.02, 0.015;
    return q;
}

/// Reference: unstructured RK4 on (x, Phi) with the full n x n Jacobian (the algorithm before the block change).
VectorXd referenceFullRk4(const SystemBase& model, double t0, const VectorXd& x0, double dt, MatrixXd& phi)
{
    const Index n = x0.size();
    const int nSub = std::max(1, static_cast<int>(std::ceil(dt / model.maxIntegrationStep())));
    const double h = dt / static_cast<double>(nSub);
    VectorXd x = x0;
    MatrixXd p = MatrixXd::Identity(n, n);
    double t = t0;
    auto deriv = [&model](double tt, const VectorXd& xx, const MatrixXd& pp, VectorXd& dx, MatrixXd& dp)
    {
        MatrixXd a;
        dx = model.dynamics(tt, xx, &a);
        dp = a * pp;
    };
    for (int s = 0; s < nSub; ++s)
    {
        VectorXd k1x, k2x, k3x, k4x;
        MatrixXd k1p, k2p, k3p, k4p;
        deriv(t, x, p, k1x, k1p);
        deriv(t + 0.5 * h, x + 0.5 * h * k1x, p + 0.5 * h * k1p, k2x, k2p);
        deriv(t + 0.5 * h, x + 0.5 * h * k2x, p + 0.5 * h * k2p, k3x, k3p);
        deriv(t + h, x + h * k3x, p + h * k3p, k4x, k4p);
        x += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        p += (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        t += h;
    }
    phi = p;
    return x;
}

/// Reference information-form time update with the full STM: new = [eta_k-free 12-block ; keepOld].
GaussianD referenceUpdate(const SystemVisualNav& model, const SystemEstimator& est, double dt,
                          const std::vector<Index>& keepOld)
{
    const Index n = est.dim();
    const VectorXd mu = est.mean();
    MatrixXd phi;
    const VectorXd xNext = referenceFullRk4(model, est.time(), mu, dt, phi);
    MatrixXd phiS(12, n);
    VectorXd cS(12);
    for (Index i = 0; i < 12; ++i)
    {
        phiS.row(i) = phi.row(i);
        cS(i) = xNext(i) - phiS.row(i).dot(mu);
    }
    const MatrixXd lS = model.processNoiseSqrtInfo(dt, mu);
    std::vector<Index> keep;
    for (Index i = 0; i < 12; ++i)
    {
        keep.push_back(n + i);
    }
    for (Index idx : keepOld)
    {
        keep.push_back(idx);
    }
    return est.state().linearisedTransition(phiS, cS, lS, keep);
}

/// Estimator with a random information-form density over 12 + 3 m states (pose + m point landmarks).
SystemEstimator randomEstimator(const std::shared_ptr<SystemVisualNav>& model, Index m, unsigned seed)
{
    SystemEstimator est(model, GaussianD::fromMoment(randomVector(12, seed), 0.05 * randomSpd(12, seed + 1)), 0.0);
    for (Index j = 0; j < m; ++j)
    {
        const Vector3d lm = randomVector(3, seed + 10 + static_cast<unsigned>(j)).cwiseProduct(Vector3d(4, 4, 2));
        model->addLandmark(est, SystemVisualNav::LandmarkType::Point,
                           GaussianD::fromMoment(lm, (0.2 + 0.1 * static_cast<double>(j % 3)) * Eigen::Matrix3d::Identity()));
    }
    // Attitude away from gimbal lock, and a full (correlated) density over the complete state.
    VectorXd mu = est.mean();
    mu.segment<3>(9) = Vector3d(0.4 * mu(9), -0.3 * mu(10), 0.8 * mu(11));
    const Index n = est.dim();
    est.setState(GaussianD::fromMoment(mu, 0.02 * randomSpd(n, seed + 500)));
    return est;
}

} // namespace

SCENARIO("SystemVisualNav: the block-wise flow integration equals the full-state RK4 reference")
{
    GIVEN("a visual-navigation model declaring the 12-state dynamic block")
    {
        auto model = std::make_shared<SystemVisualNav>(blockPsd());
        REQUIRE(model->dynamicBlockDim() == 12);
        // n = 51, 96, 150: the block structure is independent of n; the full-state reference costs O(n^3) per RK4 stage.
        for (Index n : {Index(51), Index(96), Index(150)})
        {
            VectorXd x = VectorXd::Zero(n);
            x.head(12) = randomVector(12, 3000 + static_cast<unsigned>(n));
            x.segment<3>(9) = Vector3d(0.4 * x(9), -0.3 * x(10), 0.8 * x(11));
            x.tail(n - 12) = 4.0 * randomVector(n - 12, 3100 + static_cast<unsigned>(n));
            WHEN("the flow and the STM are compared with the full-state RK4 for n = " + std::to_string(n))
            {
                MatrixXd phiRef;
                const VectorXd xRef = referenceFullRk4(*model, 0.0, x, 0.05, phiRef);
                MatrixXd phiBlock;
                const VectorXd xBlock = SystemEstimator::integrateFlow(*model, 0.0, x, 0.05, &phiBlock);
                // One THEN: doctest re-runs the WHEN block for every THEN, and the full-state reference is the costly part.
                THEN("the state agrees to 1e-10 relative accuracy, the trailing states are exactly static, and the STM "
                     "equals blockdiag(Phi_12, I) of the full reference to 1e-10")
                {
                    CHECK(nearMatrix(xBlock, xRef, 1e-10));
                    CHECK((xBlock.tail(n - 12).array() == x.tail(n - 12).array()).all());
                    CAPTURE_EIGEN(phiBlock.topLeftCorner(12, 12));
                    CHECK(nearMatrix(phiBlock, phiRef, 1e-10));
                    CHECK(phiBlock.bottomLeftCorner(n - 12, 12).isZero(0.0));
                    CHECK(phiBlock.topRightCorner(12, n - 12).isZero(0.0));
                    CHECK(phiBlock.bottomRightCorner(n - 12, n - 12).isIdentity(0.0));
                }
            }
        }
    }
}

SCENARIO("SystemVisualNav: block-wise plain and delayed time updates equal the full-STM information-form update")
{
    GIVEN("random correlated densities with 13 and 30 point landmarks (n = 51, 96)")
    {
        for (Index m : {Index(13), Index(30)})
        {
            const double dt = 1.0 / 29.97;
            WHEN("a plain prediction is taken with the block-wise path (m = " + std::to_string(m) + ")")
            {
                auto model = std::make_shared<SystemVisualNav>(blockPsd());
                SystemEstimator est = randomEstimator(model, m, 4000 + static_cast<unsigned>(m));
                const Index n = est.dim();
                std::vector<Index> keepOld;
                for (Index i = 12; i < n; ++i)
                {
                    keepOld.push_back(i);
                }
                const GaussianD ref = referenceUpdate(*model, est, dt, keepOld);
                model->predictPlain(est, dt);
                THEN("mean and covariance equal the full-STM information-form result")
                {
                    CHECK(est.dim() == n);
                    CHECK(nearMatrix(est.mean(), ref.mean(), 1e-9));
                    CHECK(nearMatrix(est.covariance(), ref.covariance(), 1e-8));
                    CHECK(est.state().sqrtInfo().triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
                }
            }
            WHEN("a delayed prediction is taken with the block-wise path (m = " + std::to_string(m) + ")")
            {
                auto model = std::make_shared<SystemVisualNav>(blockPsd());
                SystemEstimator est = randomEstimator(model, m, 4100 + static_cast<unsigned>(m));
                const Index n = est.dim();
                // Delayed shift keeps eta_k (indices 6..11) and all landmarks; nu_k is integrated out.
                std::vector<Index> keepOld;
                for (Index i = 6; i < 12; ++i)
                {
                    keepOld.push_back(i);
                }
                for (Index i = 12; i < n; ++i)
                {
                    keepOld.push_back(i);
                }
                const GaussianD ref = referenceUpdate(*model, est, dt, keepOld);
                model->predictDelayed(est, dt);
                THEN("the delayed pose equals the pre-update eta with the full-STM cross-covariances")
                {
                    CHECK(est.dim() == n + 6);
                    CHECK(model->hasDelayedPose());
                    // Both states are [new 12 ; eta_k (6) ; landmarks] (etaPrev is the copy of eta_k).
                    CHECK(nearMatrix(est.mean(), ref.mean(), 1e-9));
                    CHECK(nearMatrix(est.covariance(), ref.covariance(), 1e-8));
                }
            }
        }
    }
}

SCENARIO("SystemVisualNav: correlated landmark augmentation reproduces the linear-Gaussian joint moments")
{
    GIVEN("a pose state with covariance and a linear map of the pose to a 3-D landmark")
    {
        auto model = std::make_shared<SystemVisualNav>(blockPsd());
        SystemEstimator est(model, GaussianD::fromMoment(randomVector(12, 5001), 0.1 * randomSpd(12, 5002)), 0.0);
        const MatrixXd p = est.covariance();
        const VectorXd mu = est.mean();
        MatrixXd g = MatrixXd::Zero(3, 12);
        g.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
        g(0, 9) = 0.5;
        g(2, 11) = -0.25;
        const Vector3d c(1.0, -2.0, 0.5);
        const Eigen::Matrix3d q = 0.3 * randomSpd(3, 5003);
        // lS^T lS = Q^{-1}  <=>  lS = chol(Q^{-1}) (upper); test-only explicit inverse of a 3x3 reference matrix.
        const Eigen::Matrix3d lS = Eigen::LLT<Eigen::Matrix3d>(q.inverse()).matrixU();
        WHEN("the landmark is added with the correlated augmentation")
        {
            const Index id = model->addLandmarkLinear(est, SystemVisualNav::LandmarkType::Point, g, c, lS);
            THEN("the landmark mean is G mu + c and the marginal covariance is G P G^T + Q")
            {
                const GaussianD lm = est.state().marginal({12, 13, 14});
                CHECK(nearMatrix(lm.mean(), g * mu + c, 1e-9));
                CHECK(nearMatrix(lm.covariance(), g * p * g.transpose() + q, 1e-8));
            }
            THEN("the cross-covariance Cov(x, x_L) equals P G^T")
            {
                const MatrixXd joint = est.covariance();
                CHECK(nearMatrix(joint.block(0, 12, 12, 3), p * g.transpose(), 1e-8));
                CHECK(joint.block(0, 12, 12, 3).cwiseAbs().maxCoeff() > 1e-6);
            }
            THEN("the layout registers the landmark at offset 12 and the id is returned")
            {
                CHECK(est.dim() == 15);
                CHECK(model->landmarkOffset(id) == 12);
            }
        }
        WHEN("a linear map with the wrong shape is given")
        {
            THEN("it is rejected")
            {
                CHECK_THROWS_AS(model->addLandmarkLinear(est, SystemVisualNav::LandmarkType::Point, MatrixXd::Zero(3, 5),
                                                         c, lS),
                                std::invalid_argument);
            }
        }
    }
}

SCENARIO("SystemVisualNav: the flow Jacobian is block-diagonal as declared by dynamicBlockDim")
{
    GIVEN("a model and a random state with trailing landmark entries")
    {
        auto model = std::make_shared<SystemVisualNav>(blockPsd());
        VectorXd x = VectorXd::Zero(21);
        x.head(12) = randomVector(12, 6001);
        x.segment<3>(9) = Vector3d(0.3, -0.2, 0.7);
        x.tail(9) = randomVector(9, 6002);
        WHEN("the analytic Jacobian is compared with finite differences of the flow")
        {
            MatrixXd a;
            model->dynamics(0.0, x, &a);
            THEN("the trailing rows and columns are exactly zero and the leading block matches finite differences")
            {
                CHECK(a.bottomRows(9).isZero(0.0));
                CHECK(a.rightCols(9).isZero(0.0));
                const MatrixXd aFd = fdJacobian([&](const VectorXd& xx) { return model->dynamics(0.0, xx, nullptr); }, x);
                CHECK(nearMatrix(a.topLeftCorner(12, 12), aFd.topLeftCorner(12, 12), 1e-6));
            }
        }
    }
}
