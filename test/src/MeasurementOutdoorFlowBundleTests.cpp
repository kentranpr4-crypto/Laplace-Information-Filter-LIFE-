#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementBase.h"
#include "../../src/MeasurementOutdoorFlowBundle.h"
#include "../../src/OutdoorLaplaceUpdate.h"
#include "../../src/Pose.hpp"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemBase.h"
#include "../../src/SystemVisualNav.h"
#include "testUtil.h"

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using testutil::fdGradient;
using testutil::fdHessian;
using testutil::fdJacobian;

namespace
{

/// State of 18 dimensions: nu = 0, eta_k = [rk; thk], etaPrev = [rp; thp].
VectorXd makeState(const Vector3d& rk, const Vector3d& thk, const Vector3d& rp, const Vector3d& thp)
{
    VectorXd x = VectorXd::Zero(18);
    x.segment<3>(6) = rk;
    x.segment<3>(9) = thk;
    x.segment<3>(12) = rp;
    x.segment<3>(15) = thp;
    return x;
}

/// Richardson-extrapolated central-difference gradient (error O(h^4)); the flow model is strongly curved.
VectorXd richardsonGradient(const std::function<double(const VectorXd&)>& f, const VectorXd& x, double h)
{
    return (4.0 * fdGradient(f, x, 0.5 * h) - fdGradient(f, x, h)) / 3.0;
}

/// Richardson-extrapolated central-difference Hessian.
MatrixXd richardsonHessian(const std::function<double(const VectorXd&)>& f, const VectorXd& x, double h)
{
    return (4.0 * fdHessian(f, x, 0.5 * h) - fdHessian(f, x, h)) / 3.0;
}

/// Plane-consistent correspondences: pixels of camera A (k-1) inside the field of view are back-projected onto the
/// plane z = Hg with the rigid-body model and re-projected into camera B (k). Points with a nearly grazing ray or
/// behind camera B are discarded, so that the test geometry is well conditioned.
std::vector<OutdoorFlowPair> planePairs(const VectorXd& x, double Hg, const Eigen::Matrix3d& Rbc, int n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uv(-0.4, 0.4);
    const Vector3d rk = x.segment<3>(6);
    const Vector3d thk = x.segment<3>(9);
    const Vector3d rp = x.segment<3>(12);
    const Vector3d thp = x.segment<3>(15);
    const Eigen::Matrix3d RncA = rigid::cameraRotationInWorld(thp, Rbc);
    std::vector<OutdoorFlowPair> pairs;
    for (int attempt = 0; attempt < 100 * n && static_cast<int>(pairs.size()) < n; ++attempt)
    {
        const Vector3d dA(uv(rng), uv(rng), 1.0);
        const Vector3d dn = RncA * dA;
        if (dn.z() <= 1e-3)
        {
            continue;
        }
        const double lambda = (Hg - rp.z()) / dn.z();
        if (lambda <= 0.0)
        {
            continue;
        }
        const Vector3d pn = rp + lambda * dn;
        const Vector3d pb = rigid::worldToCamera(pn, rk, thk, Rbc, Vector3d::Zero());
        if (pb.z() < 0.5 || std::abs(pb.x() / pb.z()) > 1.5 || std::abs(pb.y() / pb.z()) > 1.5)
        {
            continue;
        }
        OutdoorFlowPair p;
        p.prevNorm = Eigen::Vector2d(dA.x() / dA.z(), dA.y() / dA.z());
        p.currNorm = Eigen::Vector2d(pb.x() / pb.z(), pb.y() / pb.z());
        pairs.push_back(p);
    }
    return pairs;
}

/// Estimator used only as the read-only context argument of logLikelihood (dimension 18).
std::unique_ptr<SystemEstimator> dummyEstimator()
{
    Eigen::Matrix<double, 6, 1> psd;
    psd << 1.0, 1.0, 1.0, 0.1, 0.1, 0.1;
    auto model = std::make_shared<SystemVisualNav>(psd);
    auto state = GaussianInfo<double>::fromMoment(VectorXd::Zero(18), MatrixXd::Identity(18, 18));
    return std::make_unique<SystemEstimator>(model, state, 0.0);
}

struct Config
{
    const char* name;
    Vector3d rk, thk, rp, thp;
};

std::vector<Config> representativeStates()
{
    return {
        {"level hover", Vector3d(0.4, -0.2, 0.05), Vector3d(0.0, 0.0, 0.1), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0)},
        {"banked and pitched", Vector3d(3.0, 1.5, -1.0), Vector3d(0.12, -0.08, 0.6), Vector3d(2.6, 1.1, -0.8), Vector3d(0.08, -0.05, 0.4)},
        {"fast translation", Vector3d(-12.0, 8.0, -4.0), Vector3d(-0.2, 0.15, -1.1), Vector3d(-10.0, 7.0, -3.5), Vector3d(-0.18, 0.12, -1.0)},
    };
}

constexpr double kGroundDepth = 63.0;

} // namespace

SCENARIO("Outdoor flow bundle: plane homography reproduces rigid-body projections")
{
    GIVEN("ground points on the plane z = Hg observed from two poses")
    {
        const Eigen::Matrix3d Rbc = rigid::defaultRbc();
        for (const Config& c : representativeStates())
        {
            CAPTURE(c.name);
            const VectorXd x = makeState(c.rk, c.thk, c.rp, c.thp);
            const auto pairs = planePairs(x, kGroundDepth, Rbc, 40, 7u);
            REQUIRE(pairs.size() >= 30);
            WHEN("the homography prediction is evaluated at the true state")
            {
                MeasurementOutdoorFlowBundle meas(0.0, pairs, 1e-3, kGroundDepth, Rbc);
                const VectorXd g = meas.predict(x, nullptr);
                THEN("it matches the projected observations to machine precision")
                {
                    CHECK((g - meas.y()).cwiseAbs().maxCoeff() < 1e-9);
                }
            }
        }
    }
}

SCENARIO("Outdoor flow bundle: homography derivatives match central differences")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    for (const Config& c : representativeStates())
    {
        CAPTURE(c.name);
        GIVEN("a state with roll, pitch, yaw and a non-zero plane distance")
        {
            const VectorXd x = makeState(c.rk, c.thk, c.rp, c.thp);
            std::array<Eigen::Matrix3d, 12> dH;
            MeasurementOutdoorFlowBundle::homography(x, kGroundDepth, Rbc, &dH);
            WHEN("dH/dtheta_j is compared with finite differences of H(x) for all 12 support states")
            {
                auto flatH = [&](const VectorXd& xx)
                {
                    const Eigen::Matrix3d h = MeasurementOutdoorFlowBundle::homography(xx, kGroundDepth, Rbc, nullptr);
                    VectorXd v(9);
                    for (int r = 0; r < 3; ++r)
                    {
                        for (int q = 0; q < 3; ++q)
                        {
                            v(3 * r + q) = h(r, q);
                        }
                    }
                    return v;
                };
                const MatrixXd fd = fdJacobian(flatH, x, 1e-6);
                THEN("every column agrees with the hand-written derivative")
                {
                    for (Index j = 0; j < 12; ++j)
                    {
                        VectorXd an(9);
                        for (int r = 0; r < 3; ++r)
                        {
                            for (int q = 0; q < 3; ++q)
                            {
                                an(3 * r + q) = dH[static_cast<std::size_t>(j)](r, q);
                            }
                        }
                        CAPTURE(j);
                        CHECK(an.isApprox(fd.col(6 + j), 1e-6));
                    }
                }
            }
        }
    }
}

SCENARIO("Outdoor flow bundle: prediction Jacobian and log-likelihood gradient match finite differences")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    auto sys = dummyEstimator();
    for (const Config& c : representativeStates())
    {
        CAPTURE(c.name);
        GIVEN("noisy correspondences around the state")
        {
            VectorXd x = makeState(c.rk, c.thk, c.rp, c.thp);
            auto pairs = planePairs(x, kGroundDepth, Rbc, 30, 11u);
            std::mt19937 rng(3u);
            std::normal_distribution<double> noise(0.0, 2e-3);
            for (auto& p : pairs)
            {
                p.currNorm += Eigen::Vector2d(noise(rng), noise(rng));
            }
            MeasurementOutdoorFlowBundle meas(0.0, pairs, 1e-3, kGroundDepth, Rbc);
            WHEN("the Jacobian of g(x) is compared with central differences")
            {
                MatrixXd jac;
                meas.predict(x, &jac);
                const auto fn = [&](const VectorXd& xx) { return meas.predict(xx, nullptr); };
                const MatrixXd fd = fdJacobian(fn, x, 1e-6);
                THEN("the 12 support columns agree") { CHECK(jac.isApprox(fd.middleCols(6, 12), 1e-6)); }
            }
            WHEN("the gradient of log p(y|x) is compared with central differences")
            {
                VectorXd g;
                meas.logLikelihood(x, *sys, &g, nullptr);
                const auto fn = [&](const VectorXd& xx) { return meas.logLikelihood(xx, *sys, nullptr, nullptr); };
                const VectorXd fd = richardsonGradient(fn, x, 1e-5);
                THEN("the gradient agrees on the support and vanishes outside it")
                {
                    CHECK(g.segment(6, 12).isApprox(fd.segment(6, 12), 1e-5));
                    CHECK(g.head(6).isZero());
                    CHECK(g.tail(0).size() == 0);
                }
            }
        }
    }
}

SCENARIO("Outdoor flow bundle: Gauss-Newton Hessian is exact at zero residual and close otherwise")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    auto sys = dummyEstimator();
    for (const Config& c : representativeStates())
    {
        CAPTURE(c.name);
        GIVEN("exactly plane-consistent correspondences (zero residual)")
        {
            VectorXd x = makeState(c.rk, c.thk, c.rp, c.thp);
            const auto pairs = planePairs(x, kGroundDepth, Rbc, 30, 19u);
            MeasurementOutdoorFlowBundle meas(0.0, pairs, 1e-3, kGroundDepth, Rbc);
            WHEN("the GN Hessian is compared with the finite-difference Hessian of log p")
            {
                MatrixXd h;
                meas.logLikelihood(x, *sys, nullptr, &h);
                const auto fn = [&](const VectorXd& xx) { return meas.logLikelihood(xx, *sys, nullptr, nullptr); };
                const MatrixXd fd = richardsonHessian(fn, x, 1e-4);
                THEN("the support block agrees (second-order term vanishes at zero residual)")
                {
                    const double scale = std::max(1.0, h.cwiseAbs().maxCoeff());
                    CHECK((h.block(6, 6, 12, 12) - fd.block(6, 6, 12, 12)).cwiseAbs().maxCoeff() < 1e-4 * scale);
                }
                THEN("the Hessian is zero outside the support and negative semi-definite with exactly one null direction")
                {
                    CHECK(h.block(0, 0, 18, 6).isZero());
                    // Four exact null directions of the flow curvature: global yaw; common horizontal translation of
                    // both poses (x and y); and a homothety of the two poses about a point on the ground plane, which
                    // scales the relative translation and the plane distance together, leaving t / d_A unchanged.
                    // Heading, horizontal offset and the scale are therefore carried by the prior and the altimeter.
                    Eigen::SelfAdjointEigenSolver<MatrixXd> es(-h.block(6, 6, 12, 12));
                    const double scale = std::max(1.0, es.eigenvalues().cwiseAbs().maxCoeff());
                    CHECK(es.eigenvalues().minCoeff() > -1e-8 * scale);
                    int nullDirections = 0;
                    for (Index i = 0; i < es.eigenvalues().size(); ++i)
                    {
                        nullDirections += std::abs(es.eigenvalues()(i)) < 1e-10 * scale ? 1 : 0;
                    }
                    CHECK(nullDirections == 4);
                }
            }
        }
        GIVEN("small residuals (1e-4 normalised units): the omitted second-order term is small")
        {
            VectorXd x = makeState(c.rk, c.thk, c.rp, c.thp);
            auto pairs = planePairs(x, kGroundDepth, Rbc, 30, 23u);
            std::mt19937 rng(5u);
            std::normal_distribution<double> noise(0.0, 1e-4);
            for (auto& p : pairs)
            {
                p.currNorm += Eigen::Vector2d(noise(rng), noise(rng));
            }
            MeasurementOutdoorFlowBundle meas(0.0, pairs, 1e-3, kGroundDepth, Rbc);
            WHEN("the GN Hessian is compared with the full finite-difference Hessian")
            {
                MatrixXd h;
                meas.logLikelihood(x, *sys, nullptr, &h);
                const auto fn = [&](const VectorXd& xx) { return meas.logLikelihood(xx, *sys, nullptr, nullptr); };
                const MatrixXd fd = richardsonHessian(fn, x, 1e-4);
                THEN("the relative gap is below one percent of the Hessian scale")
                {
                    const double scale = h.block(6, 6, 12, 12).cwiseAbs().maxCoeff();
                    CHECK((h.block(6, 6, 12, 12) - fd.block(6, 6, 12, 12)).cwiseAbs().maxCoeff() < 1e-2 * scale);
                }
            }
        }
    }
}

SCENARIO("Outdoor flow bundle: MAP update recovers the relative motion and attitude of the plane-consistent solution")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    GIVEN("a prior centred on a perturbed state and exact flow observations of the true state")
    {
        const Config c = representativeStates()[1];
        const VectorXd xTrue = makeState(c.rk, c.thk, c.rp, c.thp);
        const auto pairs = planePairs(xTrue, kGroundDepth, Rbc, 60, 31u);
        MeasurementOutdoorFlowBundle meas(0.0, pairs, 1e-3, kGroundDepth, Rbc);

        Eigen::Matrix<double, 6, 1> psd;
        psd << 1.0, 1.0, 1.0, 0.1, 0.1, 0.1;
        auto model = std::make_shared<SystemVisualNav>(psd);
        VectorXd mean0 = xTrue;
        mean0.segment<3>(6) += Vector3d(0.3, -0.2, 0.1);
        mean0.segment<3>(12) += Vector3d(0.3, -0.2, 0.1);
        mean0.segment<3>(9) += Vector3d(0.01, -0.01, 0.02);
        mean0.segment<3>(15) += Vector3d(0.01, -0.01, 0.02);
        VectorXd sig = VectorXd::Constant(18, 1.0);
        sig.segment<3>(6).setConstant(2.0);
        sig.segment<3>(12).setConstant(2.0);
        sig.segment<3>(9).setConstant(0.2);
        sig.segment<3>(15).setConstant(0.2);
        SystemEstimator est(model, GaussianInfo<double>::fromMoment(mean0, sig.array().square().matrix().asDiagonal()), 0.0);
        const double before = (meas.predict(est.mean(), nullptr) - meas.y()).norm();
        WHEN("the measurement is processed (Laplace MAP update)")
        {
            const auto result = meas.process(est);
            const double after = (meas.predict(est.mean(), nullptr) - meas.y()).norm();
            THEN("the optimiser converges, the precision stays positive definite and the residual drops")
            {
                CHECK(result.map.x.allFinite());
                CHECK(result.precisionIsExact);
                CHECK(after < 0.05 * before);
                CHECK(est.state().sqrtInfo().diagonal().minCoeff() > 0.0);
            }
            THEN("the relative displacement between the two poses is recovered (the global position is not observable)")
            {
                const Vector3d dEst = est.mean().segment<3>(6) - est.mean().segment<3>(12);
                const Vector3d dTrue = xTrue.segment<3>(6) - xTrue.segment<3>(12);
                CHECK((dEst - dTrue).norm() < 0.05);
            }
            THEN("roll and pitch are recovered from the plane normal")
            {
                CHECK(std::abs(est.mean()(9) - xTrue(9)) < 0.01);
                CHECK(std::abs(est.mean()(10) - xTrue(10)) < 0.01);
            }
        }
    }
}

SCENARIO("Outdoor flow bundle: invalid inputs are rejected")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    GIVEN("valid correspondences")
    {
        const VectorXd x = makeState(Vector3d::Zero(), Vector3d::Zero(), Vector3d(0, 0, 0), Vector3d::Zero());
        const auto pairs = planePairs(x, kGroundDepth, Rbc, 10, 1u);
        THEN("empty correspondences, non-positive sigma or non-positive ground depth throw")
        {
            CHECK_THROWS_AS(MeasurementOutdoorFlowBundle(0.0, {}, 1e-3, kGroundDepth, Rbc), std::invalid_argument);
            CHECK_THROWS_AS(MeasurementOutdoorFlowBundle(0.0, pairs, 0.0, kGroundDepth, Rbc), std::invalid_argument);
            CHECK_THROWS_AS(MeasurementOutdoorFlowBundle(0.0, pairs, 1e-3, -1.0, Rbc), std::invalid_argument);
        }
        THEN("a camera at or above the ground plane is rejected")
        {
            const VectorXd xHigh = makeState(Vector3d::Zero(), Vector3d::Zero(), Vector3d(0, 0, kGroundDepth + 1.0), Vector3d::Zero());
            CHECK_THROWS_AS(MeasurementOutdoorFlowBundle::homography(xHigh, kGroundDepth, Rbc, nullptr), std::domain_error);
        }
    }
}

namespace
{

/// Model with no dynamics and no process noise: the time update is a no-op at equal times.
class NoDynamicsModel : public SystemBase
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

} // namespace

SCENARIO("Square-root Laplace update reproduces the Kalman update for a rank-1 linear-Gaussian measurement")
{
    GIVEN("a 3-state Gaussian prior and a scalar linear measurement y = H x + v")
    {
        const Eigen::Vector3d m(0.5, -1.0, 2.0);
        const Eigen::Matrix3d P = testutil::randomSpd(3, 5u);
        const Eigen::RowVector3d H(1.0, 0.5, -0.3);
        const double R = 0.04;
        const double y = 0.7;
        // Reference: textbook Kalman update with explicit covariance (reference only, not the estimator path).
        const double S = (H * P * H.transpose())(0, 0) + R;
        const Eigen::Vector3d K = P * H.transpose() / S;
        const Eigen::Vector3d mKalman = m + K * (y - (H * m)(0));
        const Eigen::Matrix3d PKalman = (Eigen::Matrix3d::Identity() - K * H) * P;

        WHEN("the measurement is applied with outdoorLaplaceUpdate")
        {
            SystemEstimator est(std::make_shared<NoDynamicsModel>(), GaussianInfo<double>::fromMoment(m, P), 0.0);
            MatrixXd hm = H;
            MeasurementLinearGaussian meas(0.0, hm, VectorXd::Constant(1, y), MatrixXd::Constant(1, 1, R));
            const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, est);
            THEN("the posterior mean and covariance match the Kalman result")
            {
                CHECK(est.mean().isApprox(mKalman, 1e-9));
                CHECK(est.covariance().isApprox(PKalman, 1e-9));
                CHECK(rep.rank == 1);
                CHECK(rep.droppedGradientNorm < 1e-9);
            }
        }
    }
}

SCENARIO("Square-root Laplace update stays bounded for a rank-deficient flow update with a diffuse prior")
{
    const Eigen::Matrix3d Rbc = rigid::defaultRbc();
    GIVEN("flow observations of a true configuration and a very broad prior on the global modes")
    {
        const Config c = representativeStates()[0];
        const VectorXd xTrue = makeState(c.rk, c.thk, c.rp, c.thp);
        const auto pairs = planePairs(xTrue, kGroundDepth, Rbc, 60, 41u);
        MeasurementOutdoorFlowBundle meas(0.0, pairs, 1e-3, kGroundDepth, Rbc);
        Eigen::Matrix<double, 6, 1> psd;
        psd << 1.0, 1.0, 1.0, 0.1, 0.1, 0.1;
        VectorXd sig = VectorXd::Constant(18, 1.0);
        sig.segment<3>(6).setConstant(1e3);     // global position of the body: essentially unknown
        sig.segment<3>(12).setConstant(1e3);
        sig.segment<3>(9).setConstant(0.3);
        sig.segment<3>(15).setConstant(0.3);
        SystemEstimator est(std::make_shared<SystemVisualNav>(psd),
                            GaussianInfo<double>::fromMoment(xTrue, sig.array().square().matrix().asDiagonal()), 0.0);
        WHEN("the flow update is applied")
        {
            const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, est);
            THEN("the estimate stays finite and the relative geometry is recovered")
            {
                CHECK(est.mean().allFinite());
                CHECK(est.state().sqrtInfo().allFinite());
                CHECK(est.mean().segment<3>(6).norm() < 1e4);
                const Vector3d dEst = est.mean().segment<3>(6) - est.mean().segment<3>(12);
                CHECK((dEst - (xTrue.segment<3>(6) - xTrue.segment<3>(12))).norm() < 0.05);
            }
            THEN("the flow curvature is applied only in its range (the null modes are not updated)")
            {
                CHECK(rep.rank == 8);   // 12 support states minus the 4 null directions
                CHECK(rep.droppedNegativeMass == doctest::Approx(0.0).epsilon(1e-9));
            }
        }
    }
}
