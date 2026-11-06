#include <cmath>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/Pose.hpp"
#include "testUtil.h"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using testutil::fdJacobian;
using testutil::nearMatrix;
using testutil::randomVector;

SCENARIO("Pose: world -> camera transform, its inverse and the Jacobians of the camera point")
{
    GIVEN("a body pose, a fixed extrinsic and a camera lever arm")
    {
        const Vector3d theta(0.2, -0.15, 0.9);
        const Vector3d rbn = randomVector(3, 901);
        const Vector3d pn = randomVector(3, 902) + Vector3d(0.0, 0.0, -5.0);
        const Vector3d rCBb = Vector3d(0.05, -0.02, 0.1);
        const Matrix3d rbc = rigid::defaultRbc();
        INFO("theta = ", theta.transpose(), " rbn = ", rbn.transpose());

        WHEN("a world point is mapped to camera coordinates and back")
        {
            const Vector3d pc = rigid::worldToCamera(pn, rbn, theta, rbc, rCBb);
            const Vector3d back = rigid::cameraToWorld(pc, rbn, theta, rbc, rCBb);
            THEN("the round trip recovers the world point")
            {
                CHECK(nearMatrix(back, pn, 1e-12));
            }
            THEN("the camera coordinates equal Rnc^T (p_n - r_cn)")
            {
                const Matrix3d rnc = rigid::cameraRotationInWorld(theta, rbc);
                const Vector3d rcn = rigid::cameraOriginInWorld(theta, rbn, rCBb);
                CHECK(nearMatrix(pc, rnc.transpose() * (pn - rcn), 1e-12));
            }
        }

        WHEN("the camera-point Jacobians are evaluated")
        {
            const rigid::CameraPointJacobian j = rigid::cameraPointJacobian(pn, rbn, theta, rbc, rCBb);
            THEN("the value equals worldToCamera")
            {
                CHECK(nearMatrix(j.pc, rigid::worldToCamera(pn, rbn, theta, rbc, rCBb), 1e-13));
            }
            THEN("d p_c / d p_n matches central finite differences")
            {
                const Matrix3d fd = fdJacobian(
                    [&](const Vector3d& p) { return rigid::worldToCamera(p, rbn, theta, rbc, rCBb); }, pn);
                CHECK(nearMatrix(j.dPc_dPn, fd, 1e-7));
            }
            THEN("d p_c / d r_bn^n matches central finite differences")
            {
                const Matrix3d fd = fdJacobian(
                    [&](const Vector3d& r) { return rigid::worldToCamera(pn, r, theta, rbc, rCBb); }, rbn);
                CHECK(nearMatrix(j.dPc_drbn, fd, 1e-7));
            }
            THEN("d p_c / d Theta_nb matches central finite differences")
            {
                const Matrix3d fd = fdJacobian(
                    [&](const Vector3d& t) { return rigid::worldToCamera(pn, rbn, t, rbc, rCBb); }, theta);
                CAPTURE_EIGEN(j.dPc_dTheta);
                CAPTURE_EIGEN(fd);
                CHECK(nearMatrix(j.dPc_dTheta, fd, 1e-7));
            }
        }
    }

    GIVEN("two transforms A and B")
    {
        const rigid::Transform a = rigid::fromRpy(Vector3d(0.1, 0.4, -0.3), randomVector(3, 911));
        const rigid::Transform b = rigid::fromRpy(Vector3d(-0.5, 0.2, 1.0), randomVector(3, 912));
        const Vector3d p = randomVector(3, 913);
        WHEN("they are composed and inverted")
        {
            THEN("applying compose(A,B) equals applying A after B")
            {
                CHECK(nearMatrix(rigid::apply(rigid::compose(a, b), p), rigid::apply(a, rigid::apply(b, p)), 1e-13));
            }
            THEN("the inverse undoes the transform and compose(A, A^-1) is the identity")
            {
                CHECK(nearMatrix(rigid::apply(rigid::inverse(a), rigid::apply(a, p)), p, 1e-13));
                const rigid::Transform id = rigid::compose(a, rigid::inverse(a));
                CHECK(nearMatrix(id.R, Matrix3d::Identity(), 1e-13));
                CHECK(nearMatrix(id.t, Vector3d::Zero(), 1e-13));
            }
        }
    }
}
