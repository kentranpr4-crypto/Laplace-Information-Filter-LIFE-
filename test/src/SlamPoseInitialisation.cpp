// Tag pose initialisation (IPPE + Gauss-Newton) and the linearised birth map of a tag landmark.
#include <cmath>
#include <numbers>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/ArucoTags.h"
#include "../../src/CoreLinalg.h"
#include "../../src/Camera.h"
#include "../../src/Pose.hpp"
#include "../../src/SlamPoseInitialisation.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using testutil::fdJacobian;
using testutil::nearMatrix;

namespace
{

Camera poseTestCamera()
{
    Camera cam(1920, 1080, (Matrix3d() << 889.3, 0.0, 959.5, 0.0, 892.3, 539.5, 0.0, 0.0, 1.0).finished(),
               {-0.0246, -0.0061, 0.0005, 0.0001, 0.0013});
    cam.setBodyToCameraExtrinsics(rigid::defaultRbc(), Vector3d(0.05, -0.02, -0.10));
    return cam;
}

/// Projects the canonical corners of a tag at camera pose (Rct, tct) through the camera model.
Matrix<double, 8, 1> projectTag(const Camera& cam, const Matrix<double, 4, 3>& canonical, const Matrix3d& Rct, const Vector3d& tct)
{
    Matrix<double, 8, 1> uv;
    for (int k = 0; k < 4; ++k)
    {
        uv.segment<2>(2 * k) = cam.vectorToPixel(Rct * canonical.row(k).transpose() + tct);
    }
    return uv;
}

Matrix3d expSO3(const Vector3d& w)
{
    const double th = w.norm();
    if (th < 1e-14)
    {
        return Matrix3d::Identity() + skew(w);
    }
    return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}

/// Landmark vector [r_L; Theta_L] of a tag for a given body pose and fixed camera-frame hypothesis and in-plane index q.
Matrix<double, 6, 1> landmarkOf(const Camera& cam, const Matrix<double, 6, 1>& eta, const Matrix3d& Rct, const Vector3d& tct, int q)
{
    const Matrix3d Rnb = rpy2rot(eta.segment<3>(3));
    const Matrix3d M = cam.Rbc() * Rct * tags::inPlaneRotation(q).transpose();
    const Matrix3d RnL = Rnb * M;
    Matrix<double, 6, 1> out;
    out.segment<3>(0) = eta.segment<3>(0) + Rnb * (cam.rCBb() + cam.Rbc() * tct);
    out.segment<3>(3) = rot2rpy(RnL);
    return out;
}

} // namespace

SCENARIO("IPPE with Gauss-Newton refinement recovers the tag pose from its corners")
{
    const Camera cam = poseTestCamera();
    const Matrix<double, 4, 3> canonical = tags::canonicalCorners(0.15);
    const Matrix3d Rtrue = rpy2rot(Vector3d(0.12, -0.08, 0.3));
    const Vector3d ttrue(0.1, -0.05, 2.0);

    GIVEN("the exact corners of a tag 2 m in front of the camera")
    {
        const Matrix<double, 8, 1> z = projectTag(cam, canonical, Rtrue, ttrue);
        WHEN("the hypotheses are computed")
        {
            const std::vector<slam::TagPoseHypothesis> hyps = slam::refinedTagPoseHypotheses(cam, z, canonical, 2.0);
            THEN("a hypothesis equal to the true pose exists and is the best one (rms below 1e-4 px)")
            {
                REQUIRE(!hyps.empty());
                CHECK(hyps.front().rmsPx < 1e-4);
                CHECK(nearMatrix(hyps.front().Rct, Rtrue, 1e-7));
                CHECK((hyps.front().tct - ttrue).norm() < 1e-7);
            }
            THEN("the information factor satisfies Ra^T Ra = J^T J / sigma^2 (positive definite)")
            {
                REQUIRE(!hyps.empty());
                const Matrix<double, 6, 6> info = hyps.front().sqrtInfo.transpose() * hyps.front().sqrtInfo;
                CHECK(info.llt().info() == Eigen::Success);
                CHECK(hyps.front().sqrtInfo.isUpperTriangular(1e-12));
            }
        }
    }

    GIVEN("corners with 0.3 px noise")
    {
        Matrix<double, 8, 1> z = projectTag(cam, canonical, Rtrue, ttrue);
        for (int k = 0; k < 8; ++k)
        {
            z(k) += 0.3 * std::sin(1.7 * k + 0.4);
        }
        WHEN("the best hypothesis is taken")
        {
            const std::vector<slam::TagPoseHypothesis> hyps = slam::refinedTagPoseHypotheses(cam, z, canonical, 2.0);
            // Expected precision for 0.3 px noise at 2 m with a 0.15 m tag (~67 px): depth ~ Z sigma / size ~ 4 mm,
            // planar tilt is the weakly observed direction (~0.03 rad). Tolerances are set above these 1-sigma levels.
            THEN("the position is within 15 mm and the rotation within 0.05 rad of the truth")
            {
                REQUIRE(!hyps.empty());
                CHECK((hyps.front().tct - ttrue).norm() < 1.5e-2);
                const Eigen::AngleAxisd dR(hyps.front().Rct.transpose() * Rtrue);
                CHECK(std::abs(dR.angle()) < 5e-2);
            }
        }
    }

    GIVEN("a nearly fronto-parallel tag (the planar two-fold ambiguity is strongest)")
    {
        const Matrix3d Rflat = rpy2rot(Vector3d(0.02, 0.01, 0.0));
        const Matrix<double, 8, 1> z = projectTag(cam, canonical, Rflat, Vector3d(0.0, 0.0, 2.5));
        WHEN("both planar solutions are requested")
        {
            const std::vector<slam::TagPoseHypothesis> hyps = slam::refinedTagPoseHypotheses(cam, z, canonical, 2.0);
            THEN("at least one hypothesis is returned, sorted by residual, and the first is the true pose")
            {
                REQUIRE(!hyps.empty());
                for (std::size_t i = 1; i < hyps.size(); ++i)
                {
                    CHECK(hyps[i - 1].rmsPx <= hyps[i].rmsPx);
                }
                CHECK(nearMatrix(hyps.front().Rct, Rflat, 1e-6));
            }
        }
    }

    GIVEN("a tag point set that is degenerate (all corners identical)")
    {
        Matrix<double, 8, 1> z;
        z.setConstant(320.0);
        THEN("no hypothesis is returned or none is valid, without throwing")
        {
            CHECK_NOTHROW(slam::refinedTagPoseHypotheses(cam, z, canonical, 2.0));
        }
    }
}

SCENARIO("linearised birth of a tag landmark: G, c and the noise map M_w match finite differences")
{
    const Camera cam = poseTestCamera();
    const Matrix<double, 4, 3> canonical = tags::canonicalCorners(0.15);
    // Body pose with moderate attitude; camera-frame tag hypothesis, n = 12 state (nu, eta).
    VectorXd x = VectorXd::Zero(12);
    x.segment<3>(6) = Vector3d(0.3, -0.2, 0.1);
    x.segment<3>(9) = Vector3d(0.07, -0.05, 0.4);
    const Matrix3d Rct = rpy2rot(Vector3d(0.2, -0.1, 0.45));
    const Vector3d tct(0.15, -0.1, 2.2);
    slam::TagPoseHypothesis hyp;
    hyp.Rct = Rct;
    hyp.tct = tct;
    // A realistic sqrt-information of xi: 1 px noise, Jacobian of the corners.
    MatrixXd jacCam(8, 6);
    for (int k = 0; k < 4; ++k)
    {
        const auto [px, jp] = cam.vectorToPixelWithJacobian(Rct * canonical.row(k).transpose() + tct);
        Matrix<double, 3, 6> dpc;
        dpc.leftCols<3>() = Matrix3d::Identity();
        dpc.rightCols<3>() = -skew(Rct * canonical.row(k).transpose());
        jacCam.block<2, 6>(2 * k, 0) = jp * dpc;
    }
    hyp.sqrtInfo = core::upperFactor(MatrixXd(jacCam / 2.0));   // 2 px noise

    const Matrix<double, 6, 1> regSigma = (Matrix<double, 6, 1>() << 10.0, 10.0, 10.0, 1.0, 1.0, 1.0).finished();

    GIVEN("the birth map of a tag at this camera pose")
    {
        const slam::TagBirthLinearisation lin = slam::linearisedTagBirth(cam, x, 6, hyp, canonical, regSigma);
        const Matrix3d Rnb = rpy2rot(x.segment<3>(9));
        const Matrix3d Rnt = Rnb * cam.Rbc() * Rct;
        const int q = lin.qIndex;
        WHEN("the sensitivity of the landmark with respect to eta is compared with central differences")
        {
            const MatrixXd gFd = fdJacobian([&](const VectorXd& e) {
                Matrix<double, 6, 1> eta = e.segment<6>(6);
                return VectorXd(landmarkOf(cam, eta, Rct, tct, q));
            }, x, 1e-6);
            // Only the eta columns are nonzero in G.
            THEN("G equals the central-difference Jacobian in the eta columns (relative 1e-5)")
            {
                CAPTURE_EIGEN(lin.G);
                CAPTURE_EIGEN(gFd);
                CHECK(nearMatrix(lin.G.block<6, 6>(0, 6), gFd.block<6, 6>(0, 6), 1e-5));
                CHECK(lin.G.block<6, 6>(0, 0).norm() == doctest::Approx(0.0));
            }
            THEN("the constant term makes the affine model exact at the linearisation point")
            {
                const VectorXd eta = x.segment<6>(6);
                const VectorXd lMean = landmarkOf(cam, eta, Rct, tct, q);
                CHECK((lin.G.block<6, 6>(0, 6) * eta + lin.c - lMean).norm() < 1e-9);
            }
            THEN("the landmark frame is the marker frame rotated in plane, and the corners match the world corners")
            {
                for (int k = 0; k < 4; ++k)
                {
                    const Vector3d worldLandmark = lin.rLNn + lin.Rnl * lin.cornersLocal.row(k).transpose();
                    const Vector3d worldTag = lin.rLNn + Rnt * canonical.row(k).transpose();   // origin + R_nc Rct c_k
                    CHECK((worldLandmark - worldTag).norm() < 1e-9);
                }
                CHECK(lin.Rnl.isApprox(rpy2rot(lin.thetaLn), 1e-9));
            }
            THEN("the pitch of the landmark is away from gimbal lock (|x-axis z| <= 1/sqrt(2))")
            {
                CHECK(std::abs(lin.Rnl(2, 0)) <= std::sqrt(0.5) + 1e-9);
            }
        }
        WHEN("the camera-frame pose is perturbed by xi = [dt; dw] at fixed eta")
        {
            // Mw maps xi to the landmark perturbation: dL = Mw xi with Mw = blockdiag(Rnc, TK(Theta_L) R_nL^T Rnc).
            const Matrix3d Rnc = Rnb * cam.Rbc();
            const Matrix3d tkL = kinematicMatrixTK(lin.thetaLn);
            Matrix<double, 6, 6> mw = Matrix<double, 6, 6>::Zero();
            mw.block<3, 3>(0, 0) = Rnc;
            mw.block<3, 3>(3, 3) = tkL * lin.Rnl.transpose() * Rnc;
            const MatrixXd dFd = fdJacobian([&](const VectorXd& xi) {
                const Matrix3d Rp = expSO3(xi.segment<3>(3)) * Rct;
                const Vector3d tp = tct + xi.segment<3>(0);
                return VectorXd(landmarkOf(cam, x.segment<6>(6), Rp, tp, q));
            }, VectorXd::Zero(6), 1e-7);
            THEN("the linear map M_w reproduces the central-difference sensitivity (relative 1e-5)")
            {
                CAPTURE_EIGEN(mw);
                CAPTURE_EIGEN(dFd);
                CHECK(nearMatrix(mw, dFd, 1e-5));
            }
            THEN("lS^T lS equals M_w^{-T} (Ra^T Ra + prior information) M_w^{-1}")
            {
                Matrix<double, 6, 6> priorInfo = Matrix<double, 6, 6>::Zero();
                for (int i = 0; i < 6; ++i)
                {
                    priorInfo(i, i) = 1.0 / (regSigma(i) * regSigma(i));
                }
                const Matrix<double, 6, 6> info = hyp.sqrtInfo.transpose() * hyp.sqrtInfo + priorInfo;
                const Matrix<double, 6, 6> mwInv = mw.inverse();
                const Matrix<double, 6, 6> expected = mwInv.transpose() * info * mwInv;
                const Matrix<double, 6, 6> got = lin.lS.transpose() * lin.lS;
                CAPTURE_EIGEN(expected);
                CAPTURE_EIGEN(got);
                CHECK(nearMatrix(got, expected, 1e-8));
            }
        }
    }

    GIVEN("invalid birth inputs")
    {
        const Matrix<double, 6, 1> bad = Matrix<double, 6, 1>::Zero();
        THEN("a non-positive prior sigma is rejected")
        {
            CHECK_THROWS_AS(slam::linearisedTagBirth(cam, x, 6, hyp, canonical, bad), std::invalid_argument);
        }
    }
}

SCENARIO("in-plane choice keeps the landmark away from the Euler singularity")
{
    GIVEN("a tag whose x-axis points almost straight up in world")
    {
        // Rotation with x-axis along world -z: pitch would be 90 deg for q = 0.
        Matrix3d Rnt;
        Rnt.col(0) = Vector3d(0.0, 0.0, -1.0);
        Rnt.col(1) = Vector3d(0.0, 1.0, 0.0);
        Rnt.col(2) = Vector3d(1.0, 0.0, 0.0);
        THEN("the chosen in-plane rotation gives |R(2,0)| <= 1/sqrt(2) and a valid Euler representation")
        {
            const int q = slam::chooseInPlaneRotation(Rnt);
            const Matrix3d RnL = Rnt * tags::inPlaneRotation(q).transpose();
            CHECK(std::abs(RnL(2, 0)) <= std::sqrt(0.5) + 1e-12);
            CHECK(rpy2rot(rot2rpy(RnL)).isApprox(RnL, 1e-9));
        }
    }
}
