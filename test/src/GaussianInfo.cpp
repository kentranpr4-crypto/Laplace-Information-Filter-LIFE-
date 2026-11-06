#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/CoreLinalg.h"
#include "../../src/GaussianBase.h"
#include "../../src/GaussianInfo.h"
#include "testUtil.h"

using Gi = GaussianInfo<double>;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using testutil::fdGradient;
using testutil::fdHessian;
using testutil::nearMatrix;
using testutil::randomMatrix;
using testutil::randomSpd;
using testutil::randomVector;

namespace
{

/// Sub-vector / sub-matrix helpers for moment-form references.
VectorXd subVec(const VectorXd& v, const std::vector<Index>& idx)
{
    VectorXd out(static_cast<Index>(idx.size()));
    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        out(static_cast<Index>(i)) = v(idx[i]);
    }
    return out;
}

MatrixXd subMat(const MatrixXd& m, const std::vector<Index>& idx)
{
    const Index k = static_cast<Index>(idx.size());
    MatrixXd out(k, k);
    for (Index i = 0; i < k; ++i)
    {
        for (Index j = 0; j < k; ++j)
        {
            out(i, j) = m(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
        }
    }
    return out;
}

bool isUpperWithNonNegativeDiagonal(const MatrixXd& r)
{
    for (Index j = 0; j < r.cols(); ++j)
    {
        for (Index i = j + 1; i < r.rows(); ++i)
        {
            if (r(i, j) != 0.0)
            {
                return false;
            }
        }
    }
    for (Index i = 0; i < std::min(r.rows(), r.cols()); ++i)
    {
        if (r(i, i) < 0.0)
        {
            return false;
        }
    }
    return true;
}

} // namespace

SCENARIO("GaussianInfo: moment, information and square-root representations round-trip")
{
    GIVEN("a random 4-D Gaussian with mean mu and covariance P")
    {
        const MatrixXd P = randomSpd(4, 11);
        const VectorXd mu = randomVector(4, 12);
        CAPTURE_EIGEN(P);
        CAPTURE_EIGEN(mu);

        WHEN("it is built from the moments")
        {
            const Gi g = Gi::fromMoment(mu, P);
            THEN("dimensions, mean, covariance and information match the references")
            {
                CHECK(g.dim() == 4);
                CAPTURE_EIGEN(g.mean());
                CHECK(nearMatrix(g.mean(), mu, 1e-9));
                CHECK(nearMatrix(g.covariance(), P, 1e-9));
                CHECK(nearMatrix(g.information(), P.inverse(), 1e-8));
                CHECK(nearMatrix(g.infoVec(), P.inverse() * mu, 1e-8));
            }
            THEN("the square-root information is upper triangular with non-negative diagonal")
            {
                CHECK(isUpperWithNonNegativeDiagonal(g.sqrtInfo()));
                CHECK(isUpperWithNonNegativeDiagonal(g.sqrtCov()));
            }
            THEN("Xi^T Xi equals the inverse covariance and S^T S equals P")
            {
                CHECK(nearMatrix(g.sqrtInfo().transpose() * g.sqrtInfo(), P.inverse(), 1e-8));
                CHECK(nearMatrix(g.sqrtCov().transpose() * g.sqrtCov(), P, 1e-9));
            }
            AND_WHEN("converted back through the square-root covariance")
            {
                const Gi h = Gi::fromSqrtMoment(g.mean(), g.sqrtCov());
                THEN("the information form is reproduced")
                {
                    CHECK(nearMatrix(h.information(), g.information(), 1e-8));
                    CHECK(nearMatrix(h.infoVec(), g.infoVec(), 1e-8));
                }
            }
            AND_WHEN("converted back through the information vector and matrix")
            {
                const Gi h = Gi::fromInfo(g.infoVec(), g.information());
                THEN("the mean and covariance are reproduced")
                {
                    CHECK(nearMatrix(h.mean(), mu, 1e-8));
                    CHECK(nearMatrix(h.covariance(), P, 1e-8));
                }
            }
        }
    }
}

SCENARIO("GaussianInfo: Q-less QR gives an upper factor with R^T R = A^T A for all shapes")
{
    GIVEN("tall, square, wide and rank-deficient matrices")
    {
        const std::vector<MatrixXd> mats = {
            randomMatrix(7, 3, 21),                                   // tall
            randomMatrix(4, 4, 22),                                   // square
            randomMatrix(2, 5, 23),                                   // wide (rows < cols)
            randomMatrix(6, 2, 24) * randomMatrix(2, 4, 25),          // rank 2 in 4 columns
            MatrixXd::Zero(3, 3),                                     // all zero
        };
        for (std::size_t c = 0; c < mats.size(); ++c)
        {
            const MatrixXd& a = mats[c];
            INFO("case ", c, " rows=", a.rows(), " cols=", a.cols());
            WHEN("the upper factor is computed")
            {
                const MatrixXd r = core::upperFactor(a);
                THEN("R is cols x cols, upper triangular, non-negative diagonal and R^T R = A^T A")
                {
                    CHECK(r.rows() == a.cols());
                    CHECK(r.cols() == a.cols());
                    CHECK(isUpperWithNonNegativeDiagonal(r));
                    CHECK(nearMatrix(r.transpose() * r, a.transpose() * a, 1e-11));
                }
            }
        }
    }

    GIVEN("a tall augmented system [A | b]")
    {
        const MatrixXd a = randomMatrix(6, 3, 31);
        const VectorXd b = randomVector(6, 32);
        MatrixXd aug(6, 4);
        aug << a, b;
        WHEN("it is triangularised with the first three columns as variables")
        {
            const MatrixXd rw = core::triangularise(aug, 3);
            THEN("R^T R = A^T A and R^T w = A^T b (orthogonal invariance of the right-hand side)")
            {
                const MatrixXd r = rw.leftCols(3);
                const VectorXd w = rw.col(3);
                CHECK(nearMatrix(r.transpose() * r, a.transpose() * a, 1e-11));
                CHECK(nearMatrix(r.transpose() * w, a.transpose() * b, 1e-11));
            }
        }
    }
}

SCENARIO("GaussianInfo: marginal density equals the moment-form sub-block")
{
    GIVEN("a random 5-D Gaussian")
    {
        const MatrixXd P = randomSpd(5, 41);
        const VectorXd mu = randomVector(5, 42);
        const Gi g = Gi::fromMoment(mu, P);

        WHEN("the marginal over {1,3} is computed")
        {
            const std::vector<Index> keep = {1, 3};
            const Gi m = g.marginal(keep);
            THEN("mean and covariance match the sub-block of the moments")
            {
                CHECK(m.dim() == 2);
                CHECK(nearMatrix(m.mean(), subVec(mu, keep), 1e-9));
                CHECK(nearMatrix(m.covariance(), subMat(P, keep), 1e-9));
            }
        }
        WHEN("the marginal keeps the indices in reversed order {3,1}")
        {
            const std::vector<Index> keep = {3, 1};
            const Gi m = g.marginal(keep);
            THEN("the order of the output follows the request")
            {
                CHECK(nearMatrix(m.mean(), subVec(mu, keep), 1e-9));
                CHECK(nearMatrix(m.covariance(), subMat(P, keep), 1e-9));
            }
        }
        WHEN("the marginal keeps every variable")
        {
            const Gi m = g.marginal({0, 1, 2, 3, 4});
            THEN("the density is unchanged")
            {
                CHECK(nearMatrix(m.information(), g.information(), 1e-9));
                CHECK(nearMatrix(m.infoVec(), g.infoVec(), 1e-9));
            }
        }
    }
}

SCENARIO("GaussianInfo: conditional density equals the Gaussian conditioning formulas")
{
    GIVEN("a random 5-D Gaussian and observed values of x_{1,3}")
    {
        const MatrixXd P = randomSpd(5, 51);
        const VectorXd mu = randomVector(5, 52);
        const Gi g = Gi::fromMoment(mu, P);
        const std::vector<Index> given = {1, 3};
        const VectorXd v = randomVector(2, 53);
        const std::vector<Index> freeIdx = {0, 2, 4};
        const MatrixXd pbb = subMat(P, given);
        const MatrixXd pab = [&]()
        {
            MatrixXd m(3, 2);
            for (Index i = 0; i < 3; ++i)
            {
                for (Index j = 0; j < 2; ++j)
                {
                    m(i, j) = P(freeIdx[static_cast<std::size_t>(i)], given[static_cast<std::size_t>(j)]);
                }
            }
            return m;
        }();
        const MatrixXd gain = pab * pbb.inverse();
        const VectorXd muRef = subVec(mu, freeIdx) + gain * (v - subVec(mu, given));
        const MatrixXd pRef = subMat(P, freeIdx) - gain * pab.transpose();

        WHEN("the conditional is computed")
        {
            const Gi c = g.conditional(given, v);
            THEN("the conditional mean and covariance match the Gaussian conditioning formulas")
            {
                CHECK(c.dim() == 3);
                CAPTURE_EIGEN(c.mean());
                CHECK(nearMatrix(c.mean(), muRef, 1e-8));
                CHECK(nearMatrix(c.covariance(), pRef, 1e-8));
            }
        }
    }
}

SCENARIO("GaussianInfo: join and product of densities")
{
    GIVEN("two independent Gaussians of dimension 2 and 3")
    {
        const Gi g1 = Gi::fromMoment(randomVector(2, 61), randomSpd(2, 62));
        const Gi g2 = Gi::fromMoment(randomVector(3, 63), randomSpd(3, 64));
        WHEN("they are joined")
        {
            const Gi j = Gi::join(g1, g2);
            THEN("the result is block diagonal in mean and covariance")
            {
                CHECK(j.dim() == 5);
                CHECK(nearMatrix(j.mean().head(2), g1.mean(), 1e-9));
                CHECK(nearMatrix(j.mean().tail(3), g2.mean(), 1e-9));
                MatrixXd blk = MatrixXd::Zero(5, 5);
                blk.topLeftCorner(2, 2) = g1.covariance();
                blk.bottomRightCorner(3, 3) = g2.covariance();
                CHECK(nearMatrix(j.covariance(), blk, 1e-9));
            }
        }
    }

    GIVEN("two Gaussians over the same 3 variables")
    {
        const Gi g1 = Gi::fromMoment(randomVector(3, 71), randomSpd(3, 72));
        const Gi g2 = Gi::fromMoment(randomVector(3, 73), randomSpd(3, 74));
        WHEN("their product is formed (Bayesian fusion)")
        {
            const Gi p = g1.multiply(g2);
            const MatrixXd lam = g1.information() + g2.information();
            const VectorXd nu = g1.infoVec() + g2.infoVec();
            THEN("information adds and the information vector adds")
            {
                CHECK(nearMatrix(p.information(), lam, 1e-8));
                CHECK(nearMatrix(p.infoVec(), nu, 1e-8));
                CHECK(nearMatrix(p.mean(), lam.inverse() * nu, 1e-8));
            }
        }
    }
}

SCENARIO("GaussianInfo: information update equals the information-filter addition")
{
    GIVEN("a 4-D prior and a linear factor ||A x - b||^2")
    {
        const Gi prior = Gi::fromMoment(randomVector(4, 81), randomSpd(4, 82));
        const MatrixXd a = randomMatrix(2, 4, 83);
        const VectorXd b = randomVector(2, 84);
        WHEN("the factor is added to the prior")
        {
            const Gi post = prior.informationUpdate(a, b);
            THEN("Lambda += A^T A and nu += A^T b")
            {
                CHECK(nearMatrix(post.information(), prior.information() + a.transpose() * a, 1e-8));
                CHECK(nearMatrix(post.infoVec(), prior.infoVec() + a.transpose() * b, 1e-8));
            }
        }
    }
}

SCENARIO("GaussianInfo: affine transforms (with and without noise, invertible and dimension-changing)")
{
    GIVEN("a 4-D Gaussian and a 2 x 4 matrix A with a 2-D noise covariance Q")
    {
        const MatrixXd p = randomSpd(4, 91);
        const VectorXd mu = randomVector(4, 92);
        const Gi g = Gi::fromMoment(mu, p);
        const MatrixXd a = randomMatrix(2, 4, 93);
        const VectorXd b = randomVector(2, 94);
        const MatrixXd q = randomSpd(2, 95);
        const MatrixXd sq = Eigen::LLT<MatrixXd>(q).matrixU();

        WHEN("y = A x + b is computed without noise")
        {
            const Gi y = g.affineTransform(a, b);
            THEN("mean = A mu + b and covariance = A P A^T")
            {
                CHECK(nearMatrix(y.mean(), a * mu + b, 1e-9));
                CHECK(nearMatrix(y.covariance(), a * p * a.transpose(), 1e-9));
            }
        }
        WHEN("y = A x + b + v with v ~ N(0, Q) is computed")
        {
            const Gi y = g.affineTransform(a, b, sq);
            THEN("covariance = A P A^T + Q")
            {
                CHECK(nearMatrix(y.mean(), a * mu + b, 1e-9));
                CHECK(nearMatrix(y.covariance(), a * p * a.transpose() + q, 1e-9));
            }
        }
    }

    GIVEN("an invertible 4 x 4 matrix A")
    {
        const MatrixXd p = randomSpd(4, 101);
        const VectorXd mu = randomVector(4, 102);
        const Gi g = Gi::fromMoment(mu, p);
        const MatrixXd a = randomMatrix(4, 4, 103) + 3.0 * MatrixXd::Identity(4, 4);
        const VectorXd b = randomVector(4, 104);
        WHEN("the transform is applied")
        {
            const Gi y = g.affineTransform(a, b);
            THEN("the result matches the moment-form transform and the inverse transform recovers x")
            {
                CHECK(nearMatrix(y.mean(), a * mu + b, 1e-9));
                CHECK(nearMatrix(y.covariance(), a * p * a.transpose(), 1e-8));
                const Gi back = y.affineTransform(a.inverse(), -(a.inverse() * b));
                CHECK(nearMatrix(back.mean(), mu, 1e-8));
                CHECK(nearMatrix(back.covariance(), p, 1e-8));
            }
        }
    }
}

SCENARIO("GaussianInfo: nonlinear transform is the first-order (linearised) moment transform")
{
    GIVEN("f(x) = [x0^2 + x1 ; sin(x2) + x3 x0] with its analytic Jacobian")
    {
        auto f = [](const VectorXd& x, MatrixXd* jac) -> VectorXd
        {
            VectorXd y(2);
            y << x(0) * x(0) + x(1), std::sin(x(2)) + x(3) * x(0);
            if (jac != nullptr)
            {
                jac->resize(2, 4);
                *jac << 2.0 * x(0), 1.0, 0.0, 0.0,
                        x(3), 0.0, std::cos(x(2)), x(0);
            }
            return y;
        };
        const VectorXd x0 = randomVector(4, 111);
        MatrixXd jacFd;
        jacFd = testutil::fdJacobian([&](const VectorXd& x) { return f(x, nullptr); }, x0);
        MatrixXd jacAn;
        f(x0, &jacAn);
        CHECK(nearMatrix(jacAn, jacFd, 1e-7));

        const MatrixXd p = randomSpd(4, 112);
        const Gi g = Gi::fromMoment(x0, p);
        const MatrixXd q = randomSpd(2, 113);
        const MatrixXd sq = Eigen::LLT<MatrixXd>(q).matrixU();
        WHEN("the nonlinear transform is applied with additive noise")
        {
            const Gi y = g.nonlinearTransform(f, sq);
            THEN("mean = f(mu) and covariance = J P J^T + Q (J evaluated at the mean)")
            {
                CHECK(nearMatrix(y.mean(), f(x0, nullptr), 1e-9));
                CHECK(nearMatrix(y.covariance(), jacAn * p * jacAn.transpose() + q, 1e-9));
            }
        }
    }
}

SCENARIO("GaussianInfo: log density value, gradient and Hessian")
{
    GIVEN("a random 3-D Gaussian")
    {
        const MatrixXd p = randomSpd(3, 121);
        const VectorXd mu = randomVector(3, 122);
        const Gi g = Gi::fromMoment(mu, p);
        const VectorXd x = randomVector(3, 123);
        const double logDetP = std::log(p.determinant());
        const double refValue = -1.5 * std::log(2.0 * std::numbers::pi) - 0.5 * logDetP
                              - 0.5 * (x - mu).dot(p.inverse() * (x - mu));
        WHEN("the log density is evaluated at x with gradient and Hessian")
        {
            VectorXd gr;
            MatrixXd hs;
            const double value = g.logPdf(x, &gr, &hs);
            THEN("value matches the closed form")
            {
                CHECK(value == doctest::Approx(refValue).epsilon(1e-10));
            }
            THEN("gradient matches central finite differences")
            {
                const VectorXd gFd = fdGradient([&](const VectorXd& xx) { return g.logPdf(xx); }, x);
                CAPTURE_EIGEN(gr);
                CAPTURE_EIGEN(gFd);
                CHECK(nearMatrix(gr, gFd, 1e-7));
            }
            THEN("Hessian is -Lambda and matches central finite differences of the gradient")
            {
                CHECK(nearMatrix(hs, -g.information(), 1e-9));
                const MatrixXd hFd = testutil::fdJacobian(
                    [&](const VectorXd& xx)
                    {
                        VectorXd gg;
                        g.logPdf(xx, &gg, nullptr);
                        return gg;
                    },
                    x);
                CHECK(nearMatrix(hs, hFd, 1e-6));
            }
        }
    }
}

SCENARIO("GaussianBase: chi-squared quantiles and n-sigma gates")
{
    GIVEN("reference chi-squared quantiles (scipy.stats.chi2.ppf)")
    {
        THEN("chi2inv(0.997, 3) matches 13.931422665512084")
        {
            CHECK(GaussianBase::chi2inv(0.997, 3.0) == doctest::Approx(13.931422665512084).epsilon(1e-9));
        }
        THEN("the 3-sigma threshold of a scalar Gaussian is exactly 9 (= 3^2)")
        {
            CHECK(GaussianBase::mahalanobisThresholdSq(3.0, 1.0) == doctest::Approx(9.0).epsilon(1e-9));
        }
        THEN("the 3-sigma thresholds in 2-D and 3-D are 11.829158 and 14.156414")
        {
            CHECK(GaussianBase::mahalanobisThresholdSq(3.0, 2.0) == doctest::Approx(11.829158081900795).epsilon(1e-9));
            CHECK(GaussianBase::mahalanobisThresholdSq(3.0, 3.0) == doctest::Approx(14.156413609126675).epsilon(1e-9));
        }
        THEN("invalid probabilities and degrees of freedom are rejected")
        {
            CHECK_THROWS_AS(GaussianBase::chi2inv(1.0, 3.0), std::domain_error);
            CHECK_THROWS_AS(GaussianBase::chi2inv(0.5, 0.0), std::domain_error);
            CHECK_THROWS_AS(GaussianBase::probabilityFromNSigma(0.0), std::domain_error);
        }
    }

    GIVEN("a scalar Gaussian with variance 4 (sigma = 2)")
    {
        const Gi g = Gi::fromMoment(VectorXd::Zero(1), MatrixXd::Constant(1, 1, 4.0));
        THEN("points within 3 sigma (|x| <= 6) are inside and points beyond are outside")
        {
            CHECK(g.isWithinConfidenceRegion(VectorXd::Constant(1, 5.9), 3.0));
            CHECK(!g.isWithinConfidenceRegion(VectorXd::Constant(1, 6.1), 3.0));
            CHECK(g.mahalanobisSquared(VectorXd::Constant(1, 4.0)) == doctest::Approx(4.0).epsilon(1e-9));
        }
    }
}

SCENARIO("GaussianInfo: visualisation helpers (confidence ellipse, ellipsoid and quadric)")
{
    GIVEN("a random 4-D Gaussian")
    {
        const MatrixXd p = randomSpd(4, 131);
        const VectorXd mu = randomVector(4, 132);
        const Gi g = Gi::fromMoment(mu, p);

        WHEN("the 2-D ellipse of states {1,3} is computed at nSigma = 3")
        {
            const std::vector<Index> idx = {1, 3};
            const MatrixXd pts = g.confidenceEllipse(idx, 3.0, 64);
            const MatrixXd p2 = subMat(p, idx);
            const VectorXd mu2 = subVec(mu, idx);
            const double r2 = GaussianBase::mahalanobisThresholdSq(3.0, 2.0);
            THEN("every point lies on the boundary (x - mu)^T P2^{-1} (x - mu) = chi2 threshold")
            {
                CHECK(pts.cols() == 64);
                for (Index k = 0; k < pts.cols(); ++k)
                {
                    const VectorXd d = pts.col(k) - mu2;
                    CHECK(d.dot(p2.inverse() * d) == doctest::Approx(r2).epsilon(1e-8));
                }
            }
        }

        WHEN("the 3-D ellipsoid of states {0,2,3} and its quadric are computed at nSigma = 3")
        {
            const std::vector<Index> idx = {0, 2, 3};
            const MatrixXd pts = g.confidenceEllipsoid(idx, 3.0, 9, 12);
            const MatrixXd p3 = subMat(p, idx);
            const VectorXd mu3 = subVec(mu, idx);
            const double r2 = GaussianBase::mahalanobisThresholdSq(3.0, 3.0);
            const MatrixXd quad = g.confidenceQuadric(idx, 3.0);
            THEN("the ellipsoid surface satisfies the 3-D chi2 boundary and the quadric vanishes on it")
            {
                CHECK(pts.cols() == 9 * 12);
                CHECK(quad.rows() == 4);
                CHECK(nearMatrix(quad.topLeftCorner(3, 3), p3.inverse(), 1e-7));
                for (Index k = 0; k < pts.cols(); ++k)
                {
                    const VectorXd d = pts.col(k) - mu3;
                    CHECK(d.dot(p3.inverse() * d) == doctest::Approx(r2).epsilon(1e-8));
                    VectorXd xh(4);
                    xh << pts.col(k), 1.0;
                    CHECK(std::abs(xh.dot(quad * xh)) < 1e-6 * (1.0 + quad.cwiseAbs().maxCoeff()));
                }
            }
        }
    }
}

SCENARIO("GaussianInfo: permutation, factor normalisation and degenerate inputs")
{
    GIVEN("a random 4-D Gaussian")
    {
        const MatrixXd p = randomSpd(4, 141);
        const VectorXd mu = randomVector(4, 142);
        const Gi g = Gi::fromMoment(mu, p);
        WHEN("the variables are permuted")
        {
            const std::vector<Index> perm = {2, 0, 3, 1};
            const Gi q = g.permuted(perm);
            THEN("the mean and covariance are permuted accordingly")
            {
                CHECK(nearMatrix(q.mean(), subVec(mu, perm), 1e-9));
                CHECK(nearMatrix(q.covariance(), subMat(p, perm), 1e-9));
            }
        }
        WHEN("the density is built from a tall, non-triangular square-root information")
        {
            const MatrixXd xiTall = randomMatrix(7, 4, 143);
            const VectorXd nu = xiTall.transpose() * (xiTall * randomVector(4, 144));
            const Gi h = Gi::fromSqrtInfo(nu, xiTall);
            THEN("the information matrix is Xi^T Xi and the factor is upper triangular")
            {
                CHECK(nearMatrix(h.information(), xiTall.transpose() * xiTall, 1e-10));
                CHECK(isUpperWithNonNegativeDiagonal(h.sqrtInfo()));
            }
        }
        WHEN("a singular covariance is supplied")
        {
            MatrixXd s = MatrixXd::Identity(2, 2);
            s(1, 1) = 0.0;
            THEN("the conversion is rejected instead of producing an infinite information matrix")
            {
                CHECK_THROWS_AS(Gi::fromSqrtMoment(VectorXd::Zero(2), s), std::invalid_argument);
            }
        }
        WHEN("an index list has repeats or is out of range")
        {
            THEN("the operation is rejected")
            {
                CHECK_THROWS_AS(g.marginal({0, 0}), std::invalid_argument);
                CHECK_THROWS_AS(g.marginal({0, 4}), std::invalid_argument);
            }
        }
    }
    GIVEN("a default-constructed (zero-dimensional) density")
    {
        Gi empty;
        THEN("it has dimension zero and an empty mean")
        {
            CHECK(empty.dim() == 0);
            CHECK(empty.mean().size() == 0);
        }
    }
}
