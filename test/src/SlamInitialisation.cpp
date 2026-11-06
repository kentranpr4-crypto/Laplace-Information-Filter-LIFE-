// Landmark birth (single-view depth prior, correlated augmentation) and the landmark lifecycle (culling, removal).
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "../../src/Camera.h"
#include "../../src/GaussianInfo.h"
#include "../../src/Pose.hpp"
#include "../../src/SlamInitialisation.h"
#include "../../src/SlamLandmarkMap.h"
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
using slam::SlamConfig;
using slam::SlamLandmarkMap;
using slam::SlamLandmarkRecord;
using testutil::fdJacobian;
using testutil::nearMatrix;
using testutil::randomSpd;

namespace
{

Camera initCamera()
{
    Camera cam(1920, 1080, (Matrix3d() << 889.3, 0.0, 959.5, 0.0, 892.3, 539.5, 0.0, 0.0, 1.0).finished(),
               {-0.0246, -0.0061, 0.0005, 0.0001, 0.0013});
    cam.setBodyToCameraExtrinsics(rigid::defaultRbc(), Vector3d(0.05, -0.02, -0.10));
    return cam;
}

/// Pose-only estimator (12 states) with attitude and a full covariance.
SystemEstimator poseEstimator(const std::shared_ptr<SystemVisualNav>& model, unsigned seed)
{
    VectorXd mu = VectorXd::Zero(12);
    mu.segment<3>(6) = Vector3d(0.3, -0.2, 0.1);
    mu.segment<3>(9) = Vector3d(0.04, -0.06, 0.25);
    return SystemEstimator(model, GaussianD::fromMoment(mu, 0.02 * randomSpd(12, seed)), 0.0);
}

} // namespace

SCENARIO("initialiseLandmark: the birth pixel is reproduced by the mean and the depth prior is the mean distance")
{
    GIVEN("a camera with distortion and a pose estimator with a full covariance")
    {
        const Camera cam = initCamera();
        const SlamConfig cfg;
        auto model = std::make_shared<SystemVisualNav>(cfg.nuPsd);
        for (const Vector2d& pixel : {Vector2d(960.0, 540.0), Vector2d(300.0, 800.0), Vector2d(1500.0, 200.0)})
        {
            SystemEstimator est = poseEstimator(model, 8001);
            WHEN("a landmark is initialised at the pixel " + std::to_string(pixel.x()) + ", " + std::to_string(pixel.y()))
            {
                const Index id = slam::initialiseLandmark(est, *model, cam, pixel, 2.0, cfg.depthPriorMean, cfg.depthPriorSigma);
                REQUIRE(id >= 0);
                const VectorXd mu = est.mean();
                const Vector3d L = mu.segment<3>(12);
                THEN("the mean projects onto the birth pixel and lies at the prior depth from the camera centre")
                {
                    const Vector3d pc = rigid::worldToCamera(L, mu.segment<3>(6), mu.segment<3>(9), cam.Rbc(), cam.rCBb());
                    CHECK(cam.vectorToPixel(pc).isApprox(pixel, 1e-6));
                    CHECK(pc.norm() == doctest::Approx(cfg.depthPriorMean).epsilon(1e-9));
                }
                THEN("the landmark is registered with its block after the pose")
                {
                    CHECK(est.dim() == 15);
                    CHECK(model->landmarkOffset(id) == 12);
                }
            }
        }
    }
}

SCENARIO("initialiseLandmark: the cross-covariance is G P and the landmark covariance is G P G^T + Jn D^2 Jn^T")
{
    GIVEN("a pose estimator, a birth pixel, and the derivative of the birth mapping with respect to eta by finite differences")
    {
        const Camera cam = initCamera();
        const SlamConfig cfg;
        auto model = std::make_shared<SystemVisualNav>(cfg.nuPsd);
        SystemEstimator est = poseEstimator(model, 8101);
        const MatrixXd p = est.covariance();
        const VectorXd mu = est.mean();
        const Vector2d pixel(700.0, 420.0);
        const Vector3d uc = cam.pixelToVector(pixel);
        const double rho0 = cfg.depthPriorMean;
        // g(eta) = r + Rnb (rCBb + Rbc rho0 u_c): the birth mapping of the landmark with the ray fixed.
        auto gOf = [&](const VectorXd& eta)
        {
            return rigid::cameraToWorld(rho0 * uc, eta.head(3), eta.tail(3), cam.Rbc(), cam.rCBb());
        };
        const MatrixXd gFd = fdJacobian(gOf, mu.segment<6>(6), 1e-6);
        WHEN("the landmark is initialised")
        {
            const Index id = slam::initialiseLandmark(est, *model, cam, pixel, 2.5, rho0, cfg.depthPriorSigma);
            REQUIRE(id >= 0);
            const MatrixXd pNew = est.covariance();
            THEN("the cross-covariance Cov(eta, L) equals P_eta,eta G^T with G the finite-difference derivative")
            {
                CAPTURE_EIGEN(pNew.block(6, 12, 6, 3));
                CHECK(nearMatrix(pNew.block(6, 12, 6, 3), p.block(6, 6, 6, 6) * gFd.transpose(), 1e-6));
            }
            THEN("the landmark marginal covariance is G P G^T + Qw with Qw = Jn D^2 Jn^T (ray-aligned, depth sigma dominant)")
            {
                const MatrixXd lmCov = pNew.block(12, 12, 3, 3);
                const MatrixXd ref = gFd * p.block(6, 6, 6, 6) * gFd.transpose();
                CHECK(lmCov.allFinite());
                CHECK(lmCov.isApprox(lmCov.transpose(), 1e-12));
                // The landmark covariance exceeds G P G^T by the noise term, which is ray-aligned in depth.
                const Eigen::SelfAdjointEigenSolver<Matrix3d> es(lmCov - ref);
                CHECK(es.eigenvalues().minCoeff() > -1e-9);
                const Vector3d ray = (est.mean().segment<3>(12) - est.mean().segment<3>(6)).normalized();
                const Vector3d major = es.eigenvectors().col(2);
                CHECK(std::abs(major.dot(ray)) > 0.99);
                CHECK(es.eigenvalues()(2) == doctest::Approx(cfg.depthPriorSigma * cfg.depthPriorSigma).epsilon(0.05));
            }
        }
    }
}

SCENARIO("initialiseLandmark: a pixel that does not define a ray returns -1 and leaves the estimator unchanged")
{
    GIVEN("a pose estimator and a NaN pixel")
    {
        const Camera cam = initCamera();
        const SlamConfig cfg;
        auto model = std::make_shared<SystemVisualNav>(cfg.nuPsd);
        SystemEstimator est = poseEstimator(model, 8201);
        WHEN("initialisation is attempted")
        {
            const Index id = slam::initialiseLandmark(est, *model, cam, Vector2d(std::numeric_limits<double>::quiet_NaN(), 0.0), 2.0,
                                                      cfg.depthPriorMean, cfg.depthPriorSigma);
            THEN("no landmark is added")
            {
                CHECK(id == -1);
                CHECK(est.dim() == 12);
                CHECK(model->landmarks().empty());
            }
        }
    }
}

SCENARIO("SlamLandmarkMap and selectLandmarksToCull: lifecycle rules and capacity")
{
    GIVEN("records covering each culling rule and two survivors")
    {
        SlamConfig cfg;
        cfg.maxLandmarks = 2;
        SlamLandmarkMap map(cfg);
        const int frame = 100;
        auto make = [](Index id, bool confirmed, int misses, int behind, int lastSeen)
        {
            SlamLandmarkRecord r;
            r.id = id;
            r.confirmed = confirmed;
            r.misses = misses;
            r.behindFrames = behind;
            r.lastSeenFrame = lastSeen;
            return r;
        };
        map.insert(make(0, false, cfg.tentativeMaxMisses, 0, frame));                  // unconfirmed and missed twice -> culled
        map.insert(make(1, true, cfg.confirmedMaxMisses, 0, frame));                   // confirmed, too many misses -> culled
        map.insert(make(2, true, 3, cfg.maxBehindFrames, frame));                      // behind the camera -> culled
        map.insert(make(3, true, 0, 0, frame - cfg.maxUnseenFrames - 1));              // unseen -> culled
        map.insert(make(4, true, 5, 0, frame - 10));                                   // survivor, many misses
        map.insert(make(5, true, 1, 0, frame - 2));                                    // survivor, fewer misses
        map.insert(make(6, true, 2, 0, frame));                                        // third survivor: capacity removes it
        WHEN("the cull set is selected")
        {
            const std::vector<Index> cull = slam::selectLandmarksToCull(map, frame);
            THEN("the four rule-based victims are selected, then the capacity removes the weakest survivor")
            {
                CHECK(std::find(cull.begin(), cull.end(), 0) != cull.end());
                CHECK(std::find(cull.begin(), cull.end(), 1) != cull.end());
                CHECK(std::find(cull.begin(), cull.end(), 2) != cull.end());
                CHECK(std::find(cull.begin(), cull.end(), 3) != cull.end());
                REQUIRE(cull.size() == 5);
                // Survivors 4, 5, 6 with capacity 2: the one with the most misses (id 4) goes first.
                CHECK(std::find(cull.begin(), cull.end(), 4) != cull.end());
            }
        }
    }
    GIVEN("a duplicate and an unknown landmark id")
    {
        SlamConfig cfg;
        SlamLandmarkMap map(cfg);
        SlamLandmarkRecord r;
        r.id = 7;
        map.insert(r);
        THEN("a duplicate insertion is rejected and an unknown erase throws")
        {
            CHECK_THROWS_AS(map.insert(r), std::invalid_argument);
            CHECK_THROWS_AS(map.erase(99), std::out_of_range);
            CHECK_THROWS_AS(map.at(99), std::out_of_range);
        }
    }
}

SCENARIO("landmark removal keeps the remaining marginals and the block layout")
{
    GIVEN("an estimator with three initialised landmarks and their covariance")
    {
        const Camera cam = initCamera();
        const SlamConfig cfg;
        auto model = std::make_shared<SystemVisualNav>(cfg.nuPsd);
        SystemEstimator est = poseEstimator(model, 8301);
        std::vector<Index> ids;
        for (const Vector2d& pixel : {Vector2d(500.0, 400.0), Vector2d(900.0, 600.0), Vector2d(1300.0, 300.0)})
        {
            ids.push_back(slam::initialiseLandmark(est, *model, cam, pixel, 2.0, cfg.depthPriorMean, cfg.depthPriorSigma));
        }
        const MatrixXd before = est.covariance();
        const VectorXd meanBefore = est.mean();
        WHEN("the middle landmark is removed")
        {
            model->removeLandmark(est, ids[1]);
            THEN("the dimension drops by 3, the second remaining block moves to offset 15, and its marginal is unchanged")
            {
                CHECK(est.dim() == 18);
                CHECK(model->landmarkOffset(ids[2]) == 15);
                CHECK(nearMatrix(est.covariance().block(15, 15, 3, 3), before.block(18, 18, 3, 3), 1e-9));
                CHECK(nearMatrix(est.covariance().topLeftCorner(12, 12), before.topLeftCorner(12, 12), 1e-9));
                CHECK(nearMatrix(est.mean().segment<3>(12), meanBefore.segment<3>(12), 1e-9));
            }
        }
    }
}

SCENARIO("selectLandmarksToCull: a landmark behind the camera for maxBehindFrames is culled")
{
    GIVEN("one record with behindFrames equal to the limit and one with one frame less")
    {
        SlamConfig cfg;
        SlamLandmarkMap map(cfg);
        SlamLandmarkRecord a;
        a.id = 0;
        a.confirmed = true;
        a.lastSeenFrame = 10;
        a.behindFrames = cfg.maxBehindFrames;
        SlamLandmarkRecord b = a;
        b.id = 1;
        b.behindFrames = cfg.maxBehindFrames - 1;
        map.insert(a);
        map.insert(b);
        WHEN("the cull set is selected")
        {
            const auto cull = slam::selectLandmarksToCull(map, 10);
            THEN("only the first is culled")
            {
                REQUIRE(cull.size() == 1);
                CHECK(cull[0] == 0);
            }
        }
    }
}
