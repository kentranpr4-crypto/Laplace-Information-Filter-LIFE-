#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using testutil::fdScalarDerivative;
using testutil::nearMatrix;
using testutil::randomVector;

namespace
{

/// Representative attitudes with |theta| well away from gimbal lock.
std::vector<Vector3d> sampleAngles()
{
    std::vector<Vector3d> out;
    out.emplace_back(0.0, 0.0, 0.0);
    out.emplace_back(0.3, -0.2, 1.1);
    out.emplace_back(-1.2, 0.7, -2.5);
    out.emplace_back(2.9, -1.1, 3.0);
    for (unsigned s = 1; s <= 4; ++s)
    {
        Vector3d v = randomVector(3, 700 + s);
        out.emplace_back(3.0 * v(0), 1.2 * v(1), 3.0 * v(2));
    }
    return out;
}

} // namespace

SCENARIO("rotation: elementary rotations and ZYX Euler angles are proper rotations")
{
    GIVEN("a set of representative attitudes")
    {
        for (const Vector3d& theta : sampleAngles())
        {
            INFO("theta = ", theta.transpose());
            WHEN("the rotation matrix R = rpy2rot(theta) is formed")
            {
                const Matrix3d r = rpy2rot(theta);
                THEN("R is orthonormal with determinant +1")
                {
                    CHECK(nearMatrix(r.transpose() * r, Matrix3d::Identity(), 1e-12));
                    CHECK(std::abs(r.determinant() - 1.0) < 1e-12);
                }
                THEN("rpy2rot is the composition Rz(psi) Ry(theta) Rx(phi)")
                {
                    CHECK(nearMatrix(r, rotz(theta(2)) * roty(theta(1)) * rotx(theta(0)), 1e-14));
                }
            }
        }
    }

    GIVEN("a rotation about the x axis by 0.4 rad")
    {
        const Matrix3d r = rotx(0.4);
        THEN("the rotated y axis matches the closed form")
        {
            CHECK(nearMatrix(r.col(1), Vector3d(0.0, std::cos(0.4), std::sin(0.4)), 1e-14));
        }
    }
}

SCENARIO("rotation: rot2rpy inverts rpy2rot away from gimbal lock")
{
    GIVEN("attitudes with |pitch| < 80 degrees")
    {
        for (const Vector3d& theta : sampleAngles())
        {
            if (std::abs(theta(1)) > 1.4)
            {
                continue;
            }
            INFO("theta = ", theta.transpose());
            WHEN("the angles are recovered from the rotation matrix")
            {
                const Vector3d back = rot2rpy(rpy2rot(theta));
                THEN("the original angles are reproduced")
                {
                    CHECK(nearMatrix(back, theta, 1e-12));
                }
            }
        }
    }
}

SCENARIO("rotation: analytic Euler-angle derivatives of R agree with central finite differences")
{
    GIVEN("several attitudes away from gimbal lock")
    {
        for (const Vector3d& theta : sampleAngles())
        {
            INFO("theta = ", theta.transpose());
            WHEN("dR/dphi, dR/dtheta and dR/dpsi are evaluated")
            {
                auto rAt = [&](int k, double a)
                {
                    Vector3d t = theta;
                    t(k) = a;
                    return rpy2rot(t);
                };
                const Matrix3d dphiFd = fdScalarDerivative([&](double a) { return rAt(0, a); }, theta(0));
                const Matrix3d dthetaFd = fdScalarDerivative([&](double a) { return rAt(1, a); }, theta(1));
                const Matrix3d dpsiFd = fdScalarDerivative([&](double a) { return rAt(2, a); }, theta(2));
                THEN("each analytic derivative matches its finite-difference reference")
                {
                    CHECK(nearMatrix(drpy2rotDphi(theta), dphiFd, 1e-7));
                    CHECK(nearMatrix(drpy2rotDtheta(theta), dthetaFd, 1e-7));
                    CHECK(nearMatrix(drpy2rotDpsi(theta), dpsiFd, 1e-7));
                }
            }
        }
    }
}

SCENARIO("rotation: kinematic matrix TK maps body rates to Euler rates and its derivatives are correct")
{
    GIVEN("the inverse map E(theta) with omega_b = E(theta) Thetadot (ZYX)")
    {
        for (const Vector3d& theta : sampleAngles())
        {
            INFO("theta = ", theta.transpose());
            const double sp = std::sin(theta(0));
            const double cp = std::cos(theta(0));
            const double st = std::sin(theta(1));
            const double ct = std::cos(theta(1));
            Matrix3d e;
            e << 1.0, 0.0, -st,
                 0.0, cp, sp * ct,
                 0.0, -sp, cp * ct;
            WHEN("TK is formed")
            {
                const Matrix3d tk = kinematicMatrixTK(theta);
                THEN("TK is the inverse of E (Thetadot = TK omega_b)")
                {
                    CHECK(nearMatrix(e * tk, Matrix3d::Identity(), 1e-12));
                    CHECK(nearMatrix(tk * e, Matrix3d::Identity(), 1e-10));
                }
                THEN("the analytic derivatives match central finite differences")
                {
                    const auto d = kinematicMatrixTKDerivatives(theta);
                    for (int k = 0; k < 3; ++k)
                    {
                        auto tkAt = [&](double a)
                        {
                            Vector3d t = theta;
                            t(k) = a;
                            return kinematicMatrixTK(t);
                        };
                        const Matrix3d fd = fdScalarDerivative(tkAt, theta(k));
                        CHECK(nearMatrix(d[static_cast<std::size_t>(k)], fd, 1e-7));
                    }
                }
            }
        }
    }
}

SCENARIO("rotation: skew and vee are inverse and skew(a) b = a x b")
{
    GIVEN("two random vectors")
    {
        const Vector3d a = randomVector(3, 811);
        const Vector3d b = randomVector(3, 812);
        WHEN("the cross product is formed through skew()")
        {
            THEN("skew(a) b equals a x b and vee(skew(a)) equals a")
            {
                CHECK(nearMatrix(skew(a) * b, a.cross(b), 1e-14));
                CHECK(nearMatrix(vee(skew(a)), a, 1e-14));
            }
        }
    }
}

SCENARIO("rotation: the default body-to-camera extrinsic maps the optical axis to body x")
{
    GIVEN("Rbc = [[0,0,1],[1,0,0],[0,1,0]]")
    {
        Matrix3d rbc;
        rbc << 0.0, 0.0, 1.0,
               1.0, 0.0, 0.0,
               0.0, 1.0, 0.0;
        THEN("camera z (optical axis) is body forward x, camera x (right) is body y, camera y (down) is body z")
        {
            CHECK(nearMatrix(rbc * Vector3d(0.0, 0.0, 1.0), Vector3d(1.0, 0.0, 0.0), 1e-14));
            CHECK(nearMatrix(rbc * Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 1.0, 0.0), 1e-14));
            CHECK(nearMatrix(rbc * Vector3d(0.0, 1.0, 0.0), Vector3d(0.0, 0.0, 1.0), 1e-14));
            CHECK(std::abs(rbc.determinant() - 1.0) < 1e-14);
        }
    }
}

SCENARIO("rotation (R4-17): the estimator attitude guard accepts |pitch| <= 85 deg and rejects anything beyond")
{
    GIVEN("the documented pitch limit")
    {
        THEN("it is 85 degrees")
        {
            CHECK(kMaxKinematicPitchRad == doctest::Approx(85.0 * std::numbers::pi / 180.0));
        }
    }
    GIVEN("attitudes with pitch 0, 84, 89.9 and -89.9 degrees")
    {
        const double d2r = std::numbers::pi / 180.0;
        THEN("84 degrees and level flight pass the guard")
        {
            CHECK_NOTHROW(requireKinematicPitchSupported(Vector3d(0.2, 0.0, 1.0)));
            CHECK_NOTHROW(requireKinematicPitchSupported(Vector3d(0.2, 84.0 * d2r, 1.0)));
        }
        THEN("89.9 degrees, minus 89.9 degrees and a non-finite attitude are rejected with a documented error")
        {
            CHECK_THROWS_AS(requireKinematicPitchSupported(Vector3d(0.0, 89.9 * d2r, 0.0)), std::domain_error);
            CHECK_THROWS_AS(requireKinematicPitchSupported(Vector3d(0.0, -89.9 * d2r, 0.0)), std::domain_error);
            CHECK_THROWS_AS(requireKinematicPitchSupported(Vector3d(0.0, std::numeric_limits<double>::quiet_NaN(), 0.0)),
                            std::domain_error);
        }
    }
    GIVEN("the generic TK helper, which is intentionally unchecked (pose-landmark frames may be at any angle)")
    {
        THEN("it still returns a finite matrix at 84 degrees")
        {
            CHECK(kinematicMatrixTK(Vector3d(0.0, 84.0 * std::numbers::pi / 180.0, 0.0)).allFinite());
        }
    }
}
