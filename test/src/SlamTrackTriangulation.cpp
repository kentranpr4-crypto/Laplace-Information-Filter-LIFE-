// Tests of SlamTrackTriangulation (Scenario 6 delayed initialisation):
//  - exact recovery of a noise-free point from two poses,
//  - hand-written triangulation Jacobians (both poses, both rays) against central finite differences (tolerance 1e-6
//    relative to the Jacobian norm, step 1e-6),
//  - the birth map (G, c, lS): G equals the state Jacobian of the triangulated point, lS^T lS = Q^{-1},
//  - the information-form augmentation keeps the joint covariance: Cov(L, x) = G Cov(x), Cov(L) = G P G^T + Q.
#include <algorithm>
#include <cmath>
#include <memory>
#include <doctest/doctest.h>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "testUtil.h"
#include "../../src/Camera.h"
#include "../../src/GaussianInfo.h"
#include "../../src/Pose.hpp"
#include "../../src/SlamTrackTriangulation.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"

namespace
{

using Vector6 = Eigen::Matrix<double, 6, 1>;

Camera triangulationCamera()
{
    const Eigen::Matrix3d K = (Eigen::Matrix3d() << 420.0, 0.0, 319.5, 0.0, 415.0, 239.5, 0.0, 0.0, 1.0).finished();
    Camera cam(640, 480, K, {-0.04, 0.01, 0.0, 0.0, 0.0});
    // Body-to-camera: camera z (optical axis) = body x, camera x = -body y, camera y = -body z (body x forward).
    const Eigen::Matrix3d rbc = (Eigen::Matrix3d() << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0).finished();
    cam.setBodyToCameraExtrinsics(rbc, Eigen::Vector3d(0.05, -0.02, 0.03));
    return cam;
}

/// Normalised undistorted ray of a world point seen from the pose eta (through the pixel model, as in the pipeline).
Eigen::Vector2d rayOf(const Camera& cam, const Vector6& eta, const Eigen::Vector3d& pWorld)
{
    const Eigen::Vector3d pc = rigid::worldToCamera(pWorld, eta.head<3>(), eta.tail<3>(), cam.Rbc(), cam.rCBb());
    return cam.pixelToNormalised(cam.vectorToPixel(pc));
}

Vector6 poseA()
{
    Vector6 e;
    e << 0.10, -0.05, 0.02, 0.05, -0.10, 0.20;
    return e;
}

Vector6 poseB()
{
    // Lateral baseline of 0.42 m (the point is 3 m ahead: about 8 degrees of parallax). A baseline along the optical
    // axis would give a nearly collinear pair of rays.
    Vector6 e;
    e << 0.12, 0.42, -0.04, -0.04, 0.06, 0.28;
    return e;
}

} // namespace

SCENARIO("triangulateTwoView recovers a noise-free point from two poses")
{
    GIVEN("a calibrated camera, a keyframe pose and a current pose 0.5 m away, and a point 3 m ahead")
    {
        const Camera cam = triangulationCamera();
        const Eigen::Vector3d point(3.0, 0.4, -0.25);
        WHEN("the point is observed in both cameras (noise-free rays)")
        {
            const Eigen::Vector2d rA = rayOf(cam, poseA(), point);
            const Eigen::Vector2d rB = rayOf(cam, poseB(), point);
            const slam::TwoViewTriangulation tri = slam::triangulateTwoView(cam, poseA(), rA, poseB(), rB, false);
            THEN("the point is recovered, is valid, and has a healthy triangulation angle")
            {
                CHECK(tri.valid);
                CAPTURE_EIGEN(tri.L);
                CHECK(tri.L.isApprox(point, 1e-6));
                CHECK(tri.angleDeg > 5.0);
                CHECK(tri.depthA > 0.0);
                CHECK(tri.depthB > 0.0);
                CHECK(tri.maxReprojPx < 1e-3);
            }
        }
        WHEN("the two poses are identical (parallel rays)")
        {
            const Eigen::Vector2d r = rayOf(cam, poseA(), point);
            const slam::TwoViewTriangulation tri = slam::triangulateTwoView(cam, poseA(), r, poseA(), r, false);
            THEN("the triangulation is rejected as degenerate")
            {
                CHECK_FALSE(tri.valid);
            }
        }
        WHEN("the two rays meet behind the cameras (normalised rays of a point behind both cameras)")
        {
            // Normalised coordinates without the pixel model: the point is behind both cameras, so its normalised
            // coordinates are finite and the triangulated point is behind them.
            const Eigen::Vector3d behind(-3.0, 0.2, 0.1);
            const Eigen::Vector3d pcA = rigid::worldToCamera(behind, poseA().head<3>(), poseA().tail<3>(), cam.Rbc(), cam.rCBb());
            const Eigen::Vector3d pcB = rigid::worldToCamera(behind, poseB().head<3>(), poseB().tail<3>(), cam.Rbc(), cam.rCBb());
            const Eigen::Vector2d rA(pcA.x() / pcA.z(), pcA.y() / pcA.z());
            const Eigen::Vector2d rB(pcB.x() / pcB.z(), pcB.y() / pcB.z());
            const slam::TwoViewTriangulation tri = slam::triangulateTwoView(cam, poseA(), rA, poseB(), rB, false);
            THEN("it is not valid")
            {
                CHECK_FALSE(tri.valid);
            }
        }
    }
}

SCENARIO("triangulation Jacobians with respect to both poses and both rays match central finite differences")
{
    GIVEN("two poses with translation and rotation, and two non-exact normalised rays (the point is not on the rays)")
    {
        const Camera cam = triangulationCamera();
        const Vector6 eA = poseA();
        const Vector6 eB = poseB();
        // Rays of a point in front of both cameras, perturbed so that the point is not exactly on the rays.
        const Eigen::Vector3d point(3.0, 0.4, -0.25);
        const Eigen::Vector2d rA = rayOf(cam, eA, point) + Eigen::Vector2d(0.004, -0.003);
        const Eigen::Vector2d rB = rayOf(cam, eB, point) + Eigen::Vector2d(-0.002, 0.005);
        const slam::TwoViewTriangulation ana = slam::triangulateTwoView(cam, eA, rA, eB, rB, true);
        REQUIRE(ana.valid);
        const double h = 1e-6;
        WHEN("each pose parameter is perturbed by a central difference")
        {
            THEN("dL/d[rB; ThetaB] matches the analytic columns")
            {
                for (int k = 0; k < 6; ++k)
                {
                    Vector6 p = eB;
                    Vector6 m = eB;
                    p(k) += h;
                    m(k) -= h;
                    const Eigen::Vector3d fd = (slam::triangulateTwoView(cam, eA, rA, p, rB, false).L
                                                - slam::triangulateTwoView(cam, eA, rA, m, rB, false).L) / (2.0 * h);
                    CAPTURE(k);
                    CHECK((fd - ana.dLdEtaB.col(k)).norm() <= 1e-6 * std::max(1.0, ana.dLdEtaB.col(k).norm()));
                }
            }
            THEN("dL/d[rA; ThetaA] matches the analytic columns")
            {
                for (int k = 0; k < 6; ++k)
                {
                    Vector6 p = eA;
                    Vector6 m = eA;
                    p(k) += h;
                    m(k) -= h;
                    const Eigen::Vector3d fd = (slam::triangulateTwoView(cam, p, rA, eB, rB, false).L
                                                - slam::triangulateTwoView(cam, m, rA, eB, rB, false).L) / (2.0 * h);
                    CAPTURE(k);
                    CHECK((fd - ana.dLdEtaA.col(k)).norm() <= 1e-6 * std::max(1.0, ana.dLdEtaA.col(k).norm()));
                }
            }
            THEN("dL/d[xA, yA] and dL/d[xB, yB] match the analytic columns")
            {
                for (int k = 0; k < 2; ++k)
                {
                    Eigen::Vector2d p = rA;
                    Eigen::Vector2d m = rA;
                    p(k) += h;
                    m(k) -= h;
                    const Eigen::Vector3d fdA = (slam::triangulateTwoView(cam, eA, p, eB, rB, false).L
                                                 - slam::triangulateTwoView(cam, eA, m, eB, rB, false).L) / (2.0 * h);
                    CAPTURE(k);
                    CHECK((fdA - ana.dLdRayA.col(k)).norm() <= 1e-6 * std::max(1.0, ana.dLdRayA.col(k).norm()));
                    Eigen::Vector2d q = rB;
                    Eigen::Vector2d n = rB;
                    q(k) += h;
                    n(k) -= h;
                    const Eigen::Vector3d fdB = (slam::triangulateTwoView(cam, eA, rA, eB, q, false).L
                                                 - slam::triangulateTwoView(cam, eA, rA, eB, n, false).L) / (2.0 * h);
                    CHECK((fdB - ana.dLdRayB.col(k)).norm() <= 1e-6 * std::max(1.0, ana.dLdRayB.col(k).norm()));
                }
            }
        }
    }
}

SCENARIO("twoViewBirth: G is the state Jacobian of the point, c is the affine offset, lS gives the pixel-noise information")
{
    GIVEN("a 21-dimensional state [nu(6); eta_B(6); eta_A(6); landmark(3)] with a non-trivial linearisation point")
    {
        const Camera cam = triangulationCamera();
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(21);
        mu.segment<6>(0) << 0.2, -0.1, 0.05, 0.01, 0.02, -0.03;
        mu.segment<6>(6) = poseB();
        mu.segment<6>(12) = poseA();
        mu.segment<3>(18) << 1.0, 2.0, 3.0;
        const Eigen::Vector3d point(3.0, 0.4, -0.25);
        const Eigen::Vector2d pxA = cam.vectorToPixel(rigid::worldToCamera(point, poseA().head<3>(), poseA().tail<3>(),
                                                                          cam.Rbc(), cam.rCBb()));
        const Eigen::Vector2d pxB = cam.vectorToPixel(rigid::worldToCamera(point, poseB().head<3>(), poseB().tail<3>(),
                                                                          cam.Rbc(), cam.rCBb()));
        const double sigmaPx = 1.0;
        WHEN("the birth density is built at the linearisation point")
        {
            const slam::TwoViewBirth b = slam::twoViewBirth(cam, mu, 12, 6, pxA, pxB, sigmaPx);
            REQUIRE(b.valid);
            THEN("G has zero columns on nu and on the landmark block, and matches finite differences on both poses")
            {
                CHECK(b.g.block<3, 6>(0, 0).isZero(0.0));
                CHECK(b.g.block<3, 3>(0, 18).isZero(0.0));
                const double h = 1e-6;
                for (Eigen::Index k = 6; k < 18; ++k)
                {
                    Eigen::VectorXd p = mu;
                    Eigen::VectorXd m = mu;
                    p(k) += h;
                    m(k) -= h;
                    const Eigen::Vector3d lp = slam::twoViewBirth(cam, p, 12, 6, pxA, pxB, sigmaPx).tri.L;
                    const Eigen::Vector3d lm = slam::twoViewBirth(cam, m, 12, 6, pxA, pxB, sigmaPx).tri.L;
                    const Eigen::Vector3d fd = (lp - lm) / (2.0 * h);
                    CAPTURE(k);
                    CHECK((fd - b.g.col(k)).norm() <= 1e-6 * std::max(1.0, b.g.col(k).norm()));
                }
            }
            THEN("the affine offset is c = L - G mu")
            {
                CHECK(b.c.isApprox(b.tri.L - b.g * mu, 1e-12));
            }
            THEN("lS^T lS times the pixel-noise covariance Q = sigma_n^2 Ju Ju^T is the identity")
            {
                const double sigmaN = sigmaPx / (0.5 * (cam.fx() + cam.fy()));
                Eigen::Matrix<double, 3, 4> ju;
                ju << b.tri.dLdRayA, b.tri.dLdRayB;
                const Eigen::Matrix3d q = sigmaN * sigmaN * ju * ju.transpose();
                const Eigen::Matrix3d prod = b.lS.transpose() * b.lS * q;
                CAPTURE(prod);
                CHECK(prod.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
                // lS = R^{-T} is lower triangular (R is the upper factor of the thin QR).
                CHECK(b.lS.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().isZero(0.0));
            }
        }
        WHEN("the birth is requested with a non-positive pixel sigma")
        {
            THEN("it is rejected")
            {
                CHECK_THROWS_AS(slam::twoViewBirth(cam, mu, 12, 6, pxA, pxB, 0.0), std::invalid_argument);
            }
        }
    }
}

SCENARIO("augmenting a birth keeps the joint covariance (correlated with both poses)")
{
    GIVEN("a 12-dimensional state [nu; eta] with a correlated covariance, extended by a delayed keyframe pose")
    {
        const Camera cam = triangulationCamera();
        auto model = std::make_shared<SystemVisualNav>(Eigen::Matrix<double, 6, 1>::Constant(0.02));
        Eigen::VectorXd mu12 = Eigen::VectorXd::Zero(12);
        mu12 << 0.1, 0.0, 0.05, 0.0, 0.01, 0.0, 0.2, -0.1, 0.05, 0.03, -0.02, 0.1;
        Eigen::MatrixXd p12 = Eigen::MatrixXd::Identity(12, 12) * 0.01;
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                if (i != j)
                {
                    p12(i, j) = 0.002 * std::sin(1.0 + i + 2.0 * j);
                }
            }
        }
        p12 = 0.5 * (p12 + p12.transpose());
        SystemEstimator est(model, GaussianInfo<double>::fromMoment(mu12, p12), 0.0);
        model->predictDelayed(est, 0.0);   // keyframe pose := eta (zero step): a degenerate baseline, replaced below
        REQUIRE(est.dim() == 18);
        // A 0.15 m keyframe baseline with a correlated 18-dimensional covariance B B^T + 0.01 I (B is not symmetric).
        Eigen::Matrix<double, 6, 1> baseline;
        baseline << 0.15, 0.02, -0.01, 0.01, -0.02, 0.03;
        Eigen::VectorXd mu18 = Eigen::VectorXd::Zero(18);
        mu18.head<12>() = mu12.head<12>();
        mu18.segment<6>(12) = mu12.segment<6>(6) - baseline;
        Eigen::MatrixXd b18 = Eigen::MatrixXd::Zero(18, 18);
        for (int i = 0; i < 18; ++i)
        {
            for (int j = 0; j < 18; ++j)
            {
                b18(i, j) = 0.05 * std::sin(1.0 + 0.7 * i + 1.3 * j) + (i == j ? 0.1 : 0.0);
            }
        }
        const Eigen::MatrixXd p18Prior = b18 * b18.transpose() + 0.01 * Eigen::MatrixXd::Identity(18, 18);
        est.setState(GaussianInfo<double>::fromMoment(mu18, p18Prior));
        const Eigen::VectorXd mu = est.mean();
        const Eigen::Vector3d point(3.0, 0.4, -0.25);
        const Vector6 eA = mu.segment<6>(12);
        const Vector6 eB = mu.segment<6>(6);
        const Eigen::Vector2d rA = rayOf(cam, eA, point);
        const Eigen::Vector2d rB = rayOf(cam, eB, point);
        const Eigen::Vector2d pxA = cam.vectorToPixel(rigid::worldToCamera(point, eA.head<3>(), eA.tail<3>(), cam.Rbc(), cam.rCBb()));
        const Eigen::Vector2d pxB = cam.vectorToPixel(rigid::worldToCamera(point, eB.head<3>(), eB.tail<3>(), cam.Rbc(), cam.rCBb()));
        REQUIRE(rA.allFinite());
        REQUIRE(rB.allFinite());
        WHEN("the birth is augmented with the information-form factor")
        {
            const slam::TwoViewBirth b = slam::twoViewBirth(cam, mu, 12, 6, pxA, pxB, 1.0);
            REQUIRE(b.valid);
            const Eigen::MatrixXd p18 = est.covariance();   // prior covariance of the 18 existing states
            Eigen::MatrixXd g = Eigen::MatrixXd::Zero(3, est.dim());
            g.leftCols(b.g.cols()) = b.g;
            model->addLandmarkLinear(est, SystemVisualNav::LandmarkType::Point, g, b.c, b.lS);
            REQUIRE(est.dim() == 21);
            const Eigen::MatrixXd p = est.covariance();
            const Eigen::MatrixXd gp = g * p18;
            THEN("the mean of the new landmark is the triangulated point at the linearisation point")
            {
                CHECK(est.mean().segment<3>(18).isApprox(b.tri.L, 1e-9));
            }
            THEN("the cross-covariance with every pre-existing state is G Cov(x)")
            {
                const Eigen::MatrixXd cross = p.block(18, 0, 3, 18);
                CAPTURE(cross);
                CHECK(cross.isApprox(gp.leftCols(18), 1e-6));
            }
            THEN("the landmark covariance is G P G^T + Q with Q = sigma_n^2 Ju Ju^T")
            {
                const double sigmaN = 1.0 / (0.5 * (cam.fx() + cam.fy()));
                Eigen::Matrix<double, 3, 4> ju;
                ju << b.tri.dLdRayA, b.tri.dLdRayB;
                const Eigen::Matrix3d q = sigmaN * sigmaN * ju * ju.transpose();
                const Eigen::Matrix3d expected = g * p18 * g.transpose() + q;
                const Eigen::Matrix3d actual = p.block<3, 3>(18, 18);
                CAPTURE(actual);
                CHECK(actual.isApprox(expected, 1e-6));
            }
            THEN("the augmented joint covariance is finite and symmetric")
            {
                CHECK(p.allFinite());
                CHECK(p.isApprox(p.transpose(), 1e-9));
            }
        }
    }
}
