// Tests of the Scenario 6 cheirality repair of the flow translation branch (S6KeyframeFlow.h).
//   - cheiralityCensus: the depth-sign convention (p_B = R p_A + t) and the exact t -> -t mirror.
//   - Induced sign flip: a synthetic two-view flow whose posterior starts in the mirrored branch. The plain flow MAP stays
//     in that branch (the epipolar likelihood is invariant to t -> -t); updateFlowWithCheirality recovers the physical
//     branch, with the translation direction matching the truth.
//   - mirrorKeyframeTranslation is an involution on the mean and maps (R, t) to (R, -t).
// No new derivative is introduced (the census and the mirror are not likelihood terms), so no FD check applies here; the
// flow Jacobian itself is tested with the measurement.
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../../src/Camera.h"
#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementIndoorFlow.h"
#include "../../src/S6KeyframeFlow.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"

namespace
{

/// Synthetic camera: 320 x 240, f = 400 px, identity body-to-camera extrinsics (body = camera).
Camera testCamera()
{
    const Eigen::Matrix3d k = (Eigen::Matrix3d() << 400.0, 0.0, 159.5, 0.0, 400.0, 119.5, 0.0, 0.0, 1.0).finished();
    Camera cam(320, 240, k, {});
    cam.setBodyToCameraExtrinsics(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    return cam;
}

/// Correspondences of static points X in front of camera A (A at the origin, identity rotation) and camera B with
/// p_B = R X + t (so that the relative pose of the pair is (R, t)). Gaussian pixel noise of sigmaPx; deterministic seed.
std::vector<slam::IndoorFlowPair> twoViewPairs(const Camera& cam, const Eigen::Matrix3d& r, const Eigen::Vector3d& t,
                                               double sigmaPx, int count)
{
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, sigmaPx / cam.fx());
    std::uniform_real_distribution<double> lateral(-1.0, 1.0);
    std::uniform_real_distribution<double> depth(2.0, 4.0);
    std::vector<slam::IndoorFlowPair> pairs;
    while (static_cast<int>(pairs.size()) < count)
    {
        const double z = depth(rng);
        const Eigen::Vector3d x(lateral(rng) * z, lateral(rng) * z, z);   // in camera A
        const Eigen::Vector3d xb = r * x + t;                               // in camera B
        if (xb.z() <= 0.5)
        {
            continue;   // keep the points in front of both cameras (the generator, not the test under study)
        }
        slam::IndoorFlowPair pr;
        pr.prevNorm = Eigen::Vector2d(x.x() / x.z(), x.y() / x.z()) + Eigen::Vector2d(noise(rng), noise(rng));
        pr.currNorm = Eigen::Vector2d(xb.x() / xb.z(), xb.y() / xb.z()) + Eigen::Vector2d(noise(rng), noise(rng));
        pairs.push_back(pr);
    }
    return pairs;
}

/// Gaussian prior with mean mu and sigma 1 on every state of the 18-state Scenario 6 layout (weak, not informative).
GaussianInfo<double> weakPrior(const Eigen::VectorXd& mu)
{
    return GaussianInfo<double>::fromMoment(mu, Eigen::MatrixXd::Identity(18, 18));
}

/// Pose layout of SystemVisualNav: velocity 0..5, body pose eta_k at 6..11, keyframe pose etaPrev at 12..17.
Eigen::VectorXd stateWithCurrentPosition(const Eigen::Vector3d& rB)
{
    Eigen::VectorXd x = Eigen::VectorXd::Zero(18);
    x.segment<3>(6) = rB;
    return x;
}

} // namespace

SCENARIO("the depth-sign census follows p_B = R p_A + t and the mirror t -> -t flips every sign")
{
    GIVEN("a rotated two-view geometry (R, t) with points in front of both cameras, without pixel noise")
    {
        const Camera cam = testCamera();
        const Eigen::Matrix3d r = Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, 1.0, -0.4).normalized()).toRotationMatrix();
        const Eigen::Vector3d t(0.2, 0.05, -0.03);
        const std::vector<slam::IndoorFlowPair> pairs = twoViewPairs(cam, r, t, 0.0, 40);
        WHEN("the census is evaluated at the true pose (R, t) and at its mirror (R, -t)")
        {
            const slam::CheiralityCensus truth = slam::cheiralityCensus(pairs, r, t);
            const slam::CheiralityCensus mirror = slam::cheiralityCensus(pairs, r, -t);
            THEN("all determined depths are positive at the truth and all negative at the mirror")
            {
                CHECK(truth.total == 40);
                CHECK(truth.positive == truth.total);
                CHECK(truth.fractionPositive() == doctest::Approx(1.0));
                CHECK(mirror.total == 40);
                CHECK(mirror.positive == 0);
                CHECK(mirror.fractionPositive() == doctest::Approx(0.0));
            }
        }
        WHEN("the census is evaluated on a single correspondence with known depths (lambda = 2, mu = 2)")
        {
            slam::IndoorFlowPair one;
            one.prevNorm = Eigen::Vector2d(0.0, 0.0);      // P_A = (0, 0, 2)
            one.currNorm = Eigen::Vector2d(0.5, 0.0);      // P_B = (1, 0, 2) = (0.5, 0, 1) * 2
            const slam::CheiralityCensus c = slam::cheiralityCensus({one}, Eigen::Matrix3d::Identity(),
                                                                    Eigen::Vector3d(1.0, 0.0, 0.0));
            THEN("the point is counted as positive in both cameras")
            {
                CHECK(c.total == 1);
                CHECK(c.positive == 1);
            }
        }
    }
}

SCENARIO("an induced sign flip of the translation branch is recovered by the cheirality repair")
{
    GIVEN("a two-view flow between a keyframe at the origin and a current camera 0.21 m away (0.5 px noise, 60 points)")
    {
        const Camera cam = testCamera();
        const Eigen::Vector3d cB(0.2, 0.05, 0.0);
        // Camera B at cB, identity rotation: p_B = X - cB, so the relative pose is (I, -cB).
        const std::vector<slam::IndoorFlowPair> pairs = twoViewPairs(cam, Eigen::Matrix3d::Identity(), -cB, 0.5, 60);
        const double sigmaNorm = 0.5 / cam.fx();
        const slam::MeasurementIndoorFlow fm(0.0, slam::IndoorFlowModel::Epipolar, pairs, sigmaNorm, cam.Rbc(),
                                             cam.rCBb());
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.02));
        const Eigen::Vector3d tTrue = (-cB).normalized();
        const FuncMinOptions opts;

        // The induced flip: the prior is centred on the MIRRORED current position (2 cA - cB = -cB).
        WHEN("the plain flow MAP is run from the mirrored start")
        {
            SystemEstimator est(model, weakPrior(stateWithCurrentPosition(-cB)), 0.0);
            const MeasurementUpdateResult res = fm.process(est, opts);
            const slam::IndoorFlowRelativePose rp = slam::indoorFlowRelativePose(est.mean(), cam.Rbc(), cam.rCBb(), false);
            const double cosPlain = rp.t.normalized().dot(tTrue);
            THEN("the MAP stays in the mirrored branch: the epipolar likelihood cannot tell t from -t")
            {
                CHECK(res.map.status != FuncMinStatus::NumericalFailure);
                CAPTURE(cosPlain);
                CHECK(cosPlain < -0.99);
                CHECK(slam::cheiralityCensus(pairs, rp.R, rp.t).fractionPositive() < 0.01);
            }
        }

        WHEN("the cheirality-checked flow update is run from the same mirrored start")
        {
            SystemEstimator est(model, weakPrior(stateWithCurrentPosition(-cB)), 0.0);
            const slam::FlowUpdateResult fu = slam::updateFlowWithCheirality(est, cam, fm, opts);
            const slam::IndoorFlowRelativePose rp = slam::indoorFlowRelativePose(est.mean(), cam.Rbc(), cam.rCBb(), false);
            const double cosFixed = rp.t.normalized().dot(tTrue);
            THEN("the branch is mirrored back to the physical one and the translation direction matches the truth")
            {
                CHECK(fu.mirrored);
                CHECK(fu.fractionBefore < 0.01);
                CHECK(fu.fractionAfter > 0.99);
                CAPTURE(cosFixed);
                CHECK(cosFixed > 0.999);
                CHECK(slam::cheiralityCensus(pairs, rp.R, rp.t).fractionPositive() > 0.99);
                CHECK(est.mean().allFinite());
            }
        }

        WHEN("the cheirality-checked flow update is run from the physical start")
        {
            SystemEstimator est(model, weakPrior(stateWithCurrentPosition(cB)), 0.0);
            const slam::FlowUpdateResult fu = slam::updateFlowWithCheirality(est, cam, fm, opts);
            THEN("no repair is applied and the posterior stays physical")
            {
                CHECK_FALSE(fu.mirrored);
                CHECK(fu.fractionBefore > 0.99);
            }
        }
    }
}

SCENARIO("mirroring the keyframe translation is an involution that flips the relative translation")
{
    GIVEN("an estimator with a non-trivial velocity and a physical two-pose state")
    {
        const Camera cam = testCamera();
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.02));
        Eigen::VectorXd x = Eigen::VectorXd::Zero(18);
        x.head<3>() = Eigen::Vector3d(0.3, -0.1, 0.05);
        x.segment<3>(6) = Eigen::Vector3d(0.25, 0.04, -0.02);
        x.segment<3>(9) = Eigen::Vector3d(0.01, -0.02, 0.03);
        x.segment<3>(15) = Eigen::Vector3d(0.02, 0.01, -0.01);
        SystemEstimator est(model, weakPrior(x), 0.0);
        WHEN("the mirror is applied once and then again")
        {
            const Eigen::VectorXd before = est.mean();
            const slam::IndoorFlowRelativePose rp0 = slam::indoorFlowRelativePose(before, cam.Rbc(), cam.rCBb(), false);
            slam::mirrorKeyframeTranslation(est, cam);
            const Eigen::VectorXd once = est.mean();
            const slam::IndoorFlowRelativePose rp1 = slam::indoorFlowRelativePose(once, cam.Rbc(), cam.rCBb(), false);
            slam::mirrorKeyframeTranslation(est, cam);
            THEN("the first application flips the velocity and the relative translation, the second restores the state")
            {
                CHECK(once.head<3>().isApprox(-before.head<3>(), 1e-9));
                CHECK(rp1.t.isApprox(-rp0.t, 1e-9));
                CHECK(rp1.R.isApprox(rp0.R, 1e-12));
                CHECK(est.mean().isApprox(before, 1e-9));
            }
        }
    }
}
