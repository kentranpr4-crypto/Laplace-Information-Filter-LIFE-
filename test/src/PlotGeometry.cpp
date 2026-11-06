#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "../../src/GaussianBase.h"
#include "../../src/Plot.h"

using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace
{

/// Upper-triangular S with S^T S = P (from the Cholesky factor of P).
Matrix3d upperSqrt(const Matrix3d & P)
{
    const Eigen::LLT<Matrix3d> llt(P);
    REQUIRE(llt.info() == Eigen::Success);
    return llt.matrixL().transpose();
}

/// A rotated, anisotropic covariance with eigenvalues 4, 1, 0.25.
Matrix3d anisotropicCovariance(Matrix3d & rotationOut)
{
    const Vector3d axis = Vector3d(1.0, 2.0, 3.0).normalized();
    const double angle = 0.7;
    const Eigen::AngleAxisd aa(angle, axis);
    rotationOut = aa.toRotationMatrix();
    Matrix3d diag = Vector3d(4.0, 1.0, 0.25).asDiagonal();
    return rotationOut * diag * rotationOut.transpose();
}

} // namespace

SCENARIO("the 3-sigma threshold for a 3-D ellipsoid is the chi-squared quantile of the 3-sigma mass")
{
    GIVEN("nSigma = 3 in three dimensions")
    {
        const double c = GaussianBase::mahalanobisThresholdSq(3.0, 3.0);
        WHEN("c is compared with the chi-squared quantile of the same probability")
        {
            const double p = GaussianBase::probabilityFromNSigma(3.0);
            THEN("c equals chi2inv(erf(3/sqrt(2)), 3), about 14.16 (the 3-sigma region of a 3-D Gaussian)")
            {
                CHECK(c == doctest::Approx(GaussianBase::chi2inv(p, 3.0)).epsilon(1e-12));
                CHECK(c > 14.1);
                CHECK(c < 14.2);
                CHECK(std::sqrt(c) == doctest::Approx(std::sqrt(14.1565)).epsilon(2e-3));
            }
        }
    }
}

SCENARIO("the ellipsoid linear map is sqrt(c) S^T for an upper-triangular square root S")
{
    GIVEN("an anisotropic 3-D covariance P and its upper-triangular square root S (P = S^T S)")
    {
        Matrix3d R;
        const Matrix3d P = anisotropicCovariance(R);
        const Matrix3d S = upperSqrt(P);
        REQUIRE((S.transpose() * S - P).norm() < 1e-12);
        const Matrix3d strictlyLower = S.triangularView<Eigen::StrictlyLower>();
        REQUIRE(strictlyLower.norm() == 0.0);   // upper triangular

        WHEN("the linear map of the 3-sigma ellipsoid is computed")
        {
            const Matrix3d A = Plot::ellipsoidLinearMap(S, 3.0);
            const double c = GaussianBase::mahalanobisThresholdSq(3.0, 3.0);
            THEN("A equals sqrt(c) S^T, and A A^T = c P")
            {
                CHECK((A - std::sqrt(c) * S.transpose()).norm() < 1e-12);
                CHECK((A * A.transpose() - c * P).norm() < 1e-9 * c * P.norm());
            }
        }
    }
}

SCENARIO("ellipsoid surface points satisfy the 3-sigma Mahalanobis boundary")
{
    GIVEN("an anisotropic, rotated covariance and a non-zero mean")
    {
        Matrix3d R;
        const Matrix3d P = anisotropicCovariance(R);
        const Matrix3d S = upperSqrt(P);
        const Vector3d mu(0.4, -1.2, 2.5);
        const double c = GaussianBase::mahalanobisThresholdSq(3.0, 3.0);
        const Matrix3d Pinv = P.inverse();

        WHEN("the surface is sampled on a 40 x 80 grid")
        {
            const std::vector<Vector3d> pts = Plot::ellipsoidSurfacePoints(mu, S, 3.0, 40, 80);
            THEN("the grid has the expected number of points, all on the boundary (x-mu)^T P^-1 (x-mu) = c")
            {
                CHECK(pts.size() == 40u * 80u);
                double worst = 0.0;
                for (const Vector3d & x : pts)
                {
                    const Vector3d d = x - mu;
                    worst = std::max(worst, std::abs(d.dot(Pinv * d) - c));
                }
                INFO("max |Mahalanobis^2 - c| over the surface = " << worst);
                CHECK(worst < 1e-8 * c);
            }
        }

        WHEN("the extreme points along the principal axes are computed analytically")
        {
            const Eigen::SelfAdjointEigenSolver<Matrix3d> es(P);
            THEN("the extent along principal axis i is sqrt(c * lambda_i), i.e. sqrt(c) sigma_i")
            {
                for (int i = 0; i < 3; ++i)
                {
                    const Vector3d e = es.eigenvectors().col(i);
                    const double lambda = es.eigenvalues()(i);
                    // point of the ellipsoid where the projection on e is maximal: u* = S e / |S e|
                    const Vector3d u = (S * e).normalized();
                    const Vector3d x = mu + Plot::ellipsoidLinearMap(S, 3.0) * u;
                    INFO("axis " << i << " lambda=" << lambda);
                    CHECK((x - mu).dot(e) == doctest::Approx(std::sqrt(c * lambda)).epsilon(1e-10));
                    CHECK((x - mu).dot(Pinv * (x - mu)) == doctest::Approx(c).epsilon(1e-10));

                    // the sampled surface must reach that extent (up to the grid resolution) and never exceed it
                    const std::vector<Vector3d> pts = Plot::ellipsoidSurfacePoints(mu, S, 3.0, 120, 240);
                    double maxProj = -1e300;
                    for (const Vector3d & q : pts)
                        maxProj = std::max(maxProj, (q - mu).dot(e));
                    CHECK(maxProj <= std::sqrt(c * lambda) + 1e-9);
                    CHECK(maxProj >= std::sqrt(c * lambda) * 0.999);
                }
            }
        }
    }
}

SCENARIO("a diagonal square root gives axis-aligned semi-axes sqrt(c) sigma")
{
    GIVEN("S = diag(sigma_x, sigma_y, sigma_z) and the unit-sphere surface")
    {
        const Vector3d sigma(0.5, 0.2, 1.5);
        const Matrix3d S = sigma.asDiagonal();
        const Vector3d mu = Vector3d::Zero();
        const double c = GaussianBase::mahalanobisThresholdSq(3.0, 3.0);
        WHEN("the surface is sampled")
        {
            const std::vector<Vector3d> pts = Plot::ellipsoidSurfacePoints(mu, S, 3.0, 61, 120);
            THEN("the largest coordinate along each axis is sqrt(c) sigma_i (the poles and equator are sampled)")
            {
                for (int i = 0; i < 3; ++i)
                {
                    double maxAbs = 0.0;
                    for (const Vector3d & q : pts)
                        maxAbs = std::max(maxAbs, std::abs(q(i)));
                    CHECK(maxAbs == doctest::Approx(std::sqrt(c) * sigma(i)).epsilon(1e-6));
                }
            }
        }
    }
}

SCENARIO("ellipsoid sampling rejects a degenerate grid")
{
    GIVEN("a grid with a single latitude ring")
    {
        const Matrix3d S = Matrix3d::Identity();
        THEN("ellipsoidSurfacePoints throws std::invalid_argument")
        {
            CHECK_THROWS_AS((void)Plot::ellipsoidSurfacePoints(Vector3d::Zero(), S, 3.0, 1, 8), std::invalid_argument);
            CHECK_THROWS_AS((void)Plot::ellipsoidSurfacePoints(Vector3d::Zero(), S, 3.0, 8, 2), std::invalid_argument);
        }
    }
}

SCENARIO("the display rotation maps the estimator frame (z down) to a proper VTK frame (y up)")
{
    GIVEN("the fixed rotation M returned by Plot::worldToDisplay()")
    {
        const Matrix3d M = Plot::worldToDisplay();
        THEN("M is orthonormal with determinant +1 (no mirroring)")
        {
            CHECK((M.transpose() * M - Matrix3d::Identity()).norm() < 1e-14);
            CHECK(M.determinant() == doctest::Approx(1.0));
        }
        THEN("north (x_n) maps to VTK +X, east (y_n) to VTK +Z, and down (z_n) to VTK -Y (down on screen)")
        {
            CHECK((M * Vector3d(1, 0, 0) - Vector3d(1, 0, 0)).norm() < 1e-14);
            CHECK((M * Vector3d(0, 1, 0) - Vector3d(0, 0, 1)).norm() < 1e-14);
            CHECK((M * Vector3d(0, 0, 1) - Vector3d(0, -1, 0)).norm() < 1e-14);
        }
    }
}
