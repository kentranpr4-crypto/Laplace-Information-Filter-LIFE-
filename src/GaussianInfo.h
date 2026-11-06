#ifndef GAUSSIANINFO_H
#define GAUSSIANINFO_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "CoreLinalg.h"
#include "GaussianBase.h"

/**
 * @brief Multivariate Gaussian in square-root information form (LIF / Laplace information filter).
 *
 * Convention (fixed for the whole project):
 *   - information matrix     Lambda = P^{-1} = Xi^T Xi,
 *   - square-root information Xi, upper triangular with non-negative diagonal (stored),
 *   - information vector     nu = Lambda mu,
 *   - auxiliary vector       d  = Xi^{-T} nu  (so that Xi mu = d),
 *   - log density            log p(x) = -0.5 ||Xi x - d||^2 + sum(log diag Xi) - (n/2) log(2 pi).
 *
 * The filter path (marginal, conditional, product, affine/nonlinear transforms, information updates,
 * log-density) works only with Xi and nu through Q-less Householder QR and triangular solves. Covariance
 * formation (covariance(), sqrtCov()) is provided for references, tests and visualisation only.
 *
 * Variables are indexed 0..n-1 in the order of the state vector.
 *
 * @tparam Scalar floating-point scalar type (double is used throughout the project).
 */
template<typename Scalar = double>
class GaussianInfo : public GaussianBase
{
public:
    using Index = Eigen::Index;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    /**
     * @brief Nonlinear transform callback: returns y = f(x) and, if J != nullptr, fills J = df/dx.
     */
    using NonlinearFn = std::function<Vector(const Vector& x, Matrix* J)>;

    /** @brief Zero-dimensional density (placeholder). */
    GaussianInfo() = default;

    /**
     * @brief Density from an information vector and any factor Xi with Xi^T Xi = Lambda.
     *
     * Xi may be tall, square or wide; it is normalised to an n x n upper-triangular factor (Q-less QR),
     * which leaves Lambda and nu unchanged.
     * @param nu  information vector, size n
     * @param xi  square-root information matrix with xi.cols() == n
     */
    static GaussianInfo fromSqrtInfo(const Vector& nu, const Matrix& xi)
    {
        requireSize(nu.size() == xi.cols(), "fromSqrtInfo: nu size must equal number of columns of Xi");
        GaussianInfo g;
        g.nu_ = nu;
        g.xi_ = core::upperFactor(xi);
        return g;
    }

    /**
     * @brief Density from mean and square-root covariance S, where P = S^T S.
     *
     * S may be any n x n factor with S^T S = P (e.g. upper-triangular Cholesky). Computed as
     * R = qr(S), Xi = qr(R^{-T}), nu = Xi^T (Xi mu); no covariance inversion is formed explicitly.
     * @throws std::invalid_argument if P is singular (zero diagonal in R).
     */
    static GaussianInfo fromSqrtMoment(const Vector& mu, const Matrix& S)
    {
        requireSize(mu.size() == S.cols(), "fromSqrtMoment: mean size must equal number of columns of S");
        const Index n = S.cols();
        if (n == 0)
        {
            return GaussianInfo();
        }
        Matrix R = core::upperFactor(S);
        requireNonsingular(R, "fromSqrtMoment");
        const Matrix RinvT = R.transpose().template triangularView<Eigen::Lower>().solve(Matrix::Identity(n, n));
        Matrix xi = core::upperFactor(RinvT);
        Vector nu = xi.transpose() * (xi * mu);
        return GaussianInfo(std::move(nu), std::move(xi));
    }

    /**
     * @brief Density from mean and covariance P (convenience: Cholesky of P, then fromSqrtMoment).
     * @throws std::invalid_argument if P is not symmetric positive definite.
     */
    static GaussianInfo fromMoment(const Vector& mu, const Matrix& P)
    {
        Eigen::LLT<Matrix> llt(P);
        if (llt.info() != Eigen::Success)
        {
            throw std::invalid_argument("fromMoment: covariance is not positive definite");
        }
        return fromSqrtMoment(mu, llt.matrixU());
    }

    /**
     * @brief Density from information vector nu and information matrix Lambda (Cholesky, Lambda = U^T U).
     * @throws std::invalid_argument if Lambda is not symmetric positive definite.
     */
    static GaussianInfo fromInfo(const Vector& nu, const Matrix& lambda)
    {
        Eigen::LLT<Matrix> llt(lambda);
        if (llt.info() != Eigen::Success)
        {
            throw std::invalid_argument("fromInfo: information matrix is not positive definite");
        }
        return fromSqrtInfo(nu, llt.matrixU());
    }

    /** @brief Dimension n of the density. */
    Index dim() const { return nu_.size(); }

    /** @brief Information vector nu = Lambda mu. */
    const Vector& infoVec() const { return nu_; }

    /** @brief Upper-triangular square-root information Xi (Lambda = Xi^T Xi). */
    const Matrix& sqrtInfo() const { return xi_; }

    /**
     * @brief Mean mu = Lambda^{-1} nu by two triangular solves (Xi^T w = nu, Xi mu = w).
     */
    Vector mean() const
    {
        if (dim() == 0)
        {
            return Vector();
        }
        const Vector w = xi_.transpose().template triangularView<Eigen::Lower>().solve(nu_);
        return xi_.template triangularView<Eigen::Upper>().solve(w);
    }

    /**
     * @brief Square-root covariance S (upper triangular) with P = S^T S, via S = qr(Xi^{-T}).
     *
     * Reference / visualisation helper; the estimator path does not require it.
     */
    Matrix sqrtCov() const
    {
        const Index n = dim();
        if (n == 0)
        {
            return Matrix();
        }
        const Matrix xiInvT = xi_.transpose().template triangularView<Eigen::Lower>().solve(Matrix::Identity(n, n));
        return core::upperFactor(xiInvT);
    }

    /** @brief Covariance P = S^T S (reference only; not used by the filter path). */
    Matrix covariance() const
    {
        const Matrix s = sqrtCov();
        return s.transpose() * s;
    }

    /** @brief Information matrix Lambda = Xi^T Xi (reference only). */
    Matrix information() const { return xi_.transpose() * xi_; }

    /**
     * @brief Auxiliary vector d = Xi^{-T} nu (satisfies Xi mu = d and nu = Xi^T d).
     */
    Vector auxiliaryVec() const
    {
        if (dim() == 0)
        {
            return Vector();
        }
        return xi_.transpose().template triangularView<Eigen::Lower>().solve(nu_);
    }

    /**
     * @brief Squared Mahalanobis distance (x - mu)^T P^{-1} (x - mu) = ||Xi x - d||^2.
     */
    Scalar mahalanobisSquared(const Vector& x) const
    {
        requireSize(x.size() == dim(), "mahalanobisSquared: wrong vector size");
        return (xi_ * x - auxiliaryVec()).squaredNorm();
    }

    /**
     * @brief Chi-squared gate: true iff the Mahalanobis distance is within the n-sigma region.
     *
     * Threshold: GaussianBase::mahalanobisThresholdSq(nSigma, n) = chi2inv(erf(nSigma/sqrt 2), n).
     */
    bool isWithinConfidenceRegion(const Vector& x, double nSigma = 3.0) const
    {
        return static_cast<double>(mahalanobisSquared(x)) <= mahalanobisThresholdSq(nSigma, static_cast<double>(dim()));
    }

    /**
     * @brief Log density, optionally with gradient and Hessian with respect to x.
     *
     * log p(x) = -0.5 ||Xi x - d||^2 + sum log|diag Xi| - (n/2) log(2 pi)
     * grad     = -Lambda x + nu = -Xi^T (Xi x - d)
     * Hessian  = -Lambda (exact; the density is exactly quadratic).
     * @param x  evaluation point (size n)
     * @param g  optional output gradient (size n)
     * @param H  optional output Hessian (n x n)
     */
    Scalar logPdf(const Vector& x, Vector* g = nullptr, Matrix* H = nullptr) const
    {
        requireSize(x.size() == dim(), "logPdf: wrong vector size");
        const Index n = dim();
        const Vector r = xi_ * x - auxiliaryVec();
        const Scalar logdet = xi_.diagonal().cwiseAbs().array().log().sum();
        const Scalar value = -Scalar(0.5) * static_cast<Scalar>(n) * std::log(2.0 * std::numbers::pi_v<double>)
                           + logdet - Scalar(0.5) * r.squaredNorm();
        if (g != nullptr)
        {
            *g = -(xi_.transpose() * r);
        }
        if (H != nullptr)
        {
            *H = -(xi_.transpose() * xi_);
        }
        return value;
    }

    /**
     * @brief Marginal density over the variables in keep (in the order given), others integrated out.
     *
     * Information-form marginalisation: permute eliminated variables to the front, triangularise
     * [Xi | d] by Q-less QR, and take the trailing block; the trailing rows give the Schur complement
     * (marginal information) and its information vector.
     * @param keep  distinct indices in [0, n)
     */
    GaussianInfo marginal(const std::vector<Index>& keep) const
    {
        validateSubset(keep);
        const std::vector<Index> elim = complementOf(keep);
        std::vector<Index> perm = elim;
        perm.insert(perm.end(), keep.begin(), keep.end());
        Matrix a(dim(), dim() + 1);
        a.leftCols(dim()) = permuteColumns(xi_, perm);
        a.col(dim()) = auxiliaryVec();
        return condenseAugmented(a, dim(), static_cast<Index>(elim.size()));
    }

    /**
     * @brief Conditional density p(x_free | x_given = value), free variables in ascending index order.
     *
     * With the ordering [free | given], the factor blocks give Xi_free|given = R_aa and
     * d_free|given = w_a - R_ab x_given, where [R | w] is the triangularised joint factor.
     * @param given  distinct indices conditioned upon
     * @param value  values of x_given (size == given.size())
     */
    GaussianInfo conditional(const std::vector<Index>& given, const Vector& value) const
    {
        validateSubset(given);
        requireSize(value.size() == static_cast<Index>(given.size()), "conditional: value size mismatch");
        const Index n = dim();
        const std::vector<Index> freeIdx = complementOf(given);
        std::vector<Index> perm = freeIdx;
        perm.insert(perm.end(), given.begin(), given.end());
        Matrix a(n, n + 1);
        a.leftCols(n) = permuteColumns(xi_, perm);
        a.col(n) = auxiliaryVec();
        const Matrix r = core::triangularise(a, n);
        const Index na = static_cast<Index>(freeIdx.size());
        const Index nb = static_cast<Index>(given.size());
        const Matrix xa = r.topLeftCorner(na, na);
        const Matrix rab = r.block(0, na, na, nb);
        const Vector da = r.col(n).head(na) - rab * value;
        Vector nua = xa.transpose() * da;
        return GaussianInfo(std::move(nua), xa);
    }

    /**
     * @brief Re-orders the variables: new variable i is old variable perm[i] (perm must be a permutation).
     */
    GaussianInfo permuted(const std::vector<Index>& perm) const
    {
        requireSize(static_cast<Index>(perm.size()) == dim(), "permuted: permutation size mismatch");
        validateSubset(perm);
        Vector nup(dim());
        for (Index i = 0; i < dim(); ++i)
        {
            nup(i) = nu_(perm[static_cast<std::size_t>(i)]);
        }
        return fromSqrtInfo(nup, permuteColumns(xi_, perm));
    }

    /**
     * @brief Independent join (block-diagonal): p(x_a, x_b) = p(x_a) p(x_b).
     */
    static GaussianInfo join(const GaussianInfo& a, const GaussianInfo& b)
    {
        const Index na = a.dim();
        const Index nb = b.dim();
        Vector nu(na + nb);
        nu << a.nu_, b.nu_;
        Matrix xi = Matrix::Zero(na + nb, na + nb);
        xi.topLeftCorner(na, na) = a.xi_;
        xi.bottomRightCorner(nb, nb) = b.xi_;
        return GaussianInfo(std::move(nu), std::move(xi));
    }

    /**
     * @brief Product of two densities over the same variables (Bayesian fusion of two Gaussian factors).
     */
    GaussianInfo multiply(const GaussianInfo& other) const
    {
        requireSize(other.dim() == dim(), "multiply: dimension mismatch");
        return informationUpdate(other.xi_, other.auxiliaryVec());
    }

    /**
     * @brief Information update by a linear-Gaussian factor: adds ||A x - b||^2 to -2 log p.
     *
     * Stacks [Xi | d ; A | b] and triangularises (Q-less QR). For a measurement y = H x + v with
     * v ~ N(0, R) and R^{-1} = Lr^T Lr, call with A = Lr H and b = Lr y. The result is exactly the
     * information-filter update Lambda += A^T A, nu += A^T b.
     */
    GaussianInfo informationUpdate(const Matrix& a, const Vector& b) const
    {
        const Index n = dim();
        requireSize(a.cols() == n, "informationUpdate: A column count must equal dim()");
        requireSize(b.size() == a.rows(), "informationUpdate: b size must equal rows of A");
        Matrix m(n + a.rows(), n + 1);
        m.topLeftCorner(n, n) = xi_;
        m.col(n).head(n) = auxiliaryVec();
        m.bottomLeftCorner(a.rows(), n) = a;
        m.col(n).tail(a.rows()) = b;
        const Matrix r = core::triangularise(m, n);
        const Matrix xin = r.leftCols(n);
        return GaussianInfo(xin.transpose() * r.col(n), xin);
    }

    /**
     * @brief Affine transform y = A x + b (no additive noise). Uses sqrt-covariance QR: P_y = qr(S A^T).
     * @throws std::invalid_argument if the output covariance is singular (e.g. rows(A) > rank).
     */
    GaussianInfo affineTransform(const Matrix& a, const Vector& b) const
    {
        requireSize(a.cols() == dim(), "affineTransform: A column count must equal dim()");
        requireSize(b.size() == a.rows(), "affineTransform: b size must equal rows of A");
        const Matrix s = sqrtCov();
        const Matrix stacked = s * a.transpose();
        const Matrix rOut = core::upperFactor(stacked);
        return fromSqrtMoment(a * mean() + b, rOut);
    }

    /**
     * @brief Affine transform with additive noise y = A x + b + v, v ~ N(0, Q), Q = Sq^T Sq.
     *
     * P_y = A P A^T + Q is factored as qr([S A^T ; Sq]). The noise factor Sq may be singular.
     * @param sqrtCovNoise  Sq, rows(A) x rows(A) square-root covariance of v
     */
    GaussianInfo affineTransform(const Matrix& a, const Vector& b, const Matrix& sqrtCovNoise) const
    {
        requireSize(a.cols() == dim(), "affineTransform: A column count must equal dim()");
        requireSize(b.size() == a.rows(), "affineTransform: b size must equal rows of A");
        requireSize(sqrtCovNoise.rows() == a.rows() && sqrtCovNoise.cols() == a.rows(),
                    "affineTransform: noise factor must be rows(A) x rows(A)");
        const Matrix s = sqrtCov();
        Matrix stacked(dim() + a.rows(), a.rows());
        stacked.topRows(dim()) = s * a.transpose();
        stacked.bottomRows(a.rows()) = sqrtCovNoise;
        const Matrix rOut = core::upperFactor(stacked);
        return fromSqrtMoment(a * mean() + b, rOut);
    }

    /**
     * @brief Nonlinear transform y = f(x) with additive noise, first-order (linearised at the mean).
     *
     * J = df/dx at mu, y ~ N(f(mu) + J (x - mu), Sq^T Sq). This is the EKF-style moment transform,
     * implemented in square-root form.
     */
    GaussianInfo nonlinearTransform(const NonlinearFn& f, const Matrix& sqrtCovNoise) const
    {
        Matrix jac;
        const Vector mu = mean();
        const Vector y0 = f(mu, &jac);
        requireSize(jac.rows() == y0.size() && jac.cols() == dim(), "nonlinearTransform: Jacobian shape mismatch");
        return affineTransform(jac, y0 - jac * mu, sqrtCovNoise);
    }

    /**
     * @brief Nonlinear transform without additive noise (linearised at the mean; output rank must be full).
     */
    GaussianInfo nonlinearTransform(const NonlinearFn& f) const
    {
        Matrix jac;
        const Vector mu = mean();
        const Vector y0 = f(mu, &jac);
        requireSize(jac.rows() == y0.size() && jac.cols() == dim(), "nonlinearTransform: Jacobian shape mismatch");
        return affineTransform(jac, y0 - jac * mu);
    }

    /**
     * @brief Linearised Markov transition in information form (time update with an extra output copy).
     *
     * Models a new variable block x' (size k) with x'_S = Phi_S x + c_S + w, w ~ N(0, Qd), Qd^{-1} = L^T L,
     * where the k transition outputs are appended after the current variables. The joint of
     * (x, x') is formed by stacking [Xi | d ; -L Phi_S | L (c) ] and the variables are then marginalised
     * so that the result is the density of the variables listed in keep (indices into the joint: old
     * variables 0..n-1, new variables n..n+k-1). Old variables not listed in keep are integrated out.
     *
     * @param phiS  Phi_S, k x n Jacobian of the transition with respect to the current variables
     * @param cS    constant term c_S = phi(mu) - Phi_S mu (size k)
     * @param lS    k x k sqrt-information of the transition noise (L^T L = Qd^{-1}); must be invertible
     * @param keep  joint indices to keep, in output order
     */
    GaussianInfo linearisedTransition(const Matrix& phiS, const Vector& cS, const Matrix& lS,
                                      const std::vector<Index>& keep) const
    {
        const Index n = dim();
        const Index k = phiS.rows();
        requireSize(phiS.cols() == n && cS.size() == k && lS.rows() == k && lS.cols() == k,
                    "linearisedTransition: inconsistent shapes");
        const Index total = n + k;
        std::vector<bool> seen(static_cast<std::size_t>(total), false);
        for (Index idx : keep)
        {
            if (idx < 0 || idx >= total)
            {
                throw std::invalid_argument("linearisedTransition: keep index out of range");
            }
            if (seen[static_cast<std::size_t>(idx)])
            {
                throw std::invalid_argument("linearisedTransition: repeated keep index");
            }
            seen[static_cast<std::size_t>(idx)] = true;
        }
        std::vector<Index> elim;
        for (Index i = 0; i < total; ++i)
        {
            if (std::find(keep.begin(), keep.end(), i) == keep.end())
            {
                elim.push_back(i);
            }
        }

        // Joint factor over [old (n) | new (k)] with rhs.
        Matrix m = Matrix::Zero(n + k, total + 1);
        m.topLeftCorner(n, n) = xi_;
        m.col(total).head(n) = auxiliaryVec();
        m.block(n, 0, k, n) = -(lS * phiS);
        m.block(n, n, k, k) = lS;
        m.col(total).tail(k) = lS * cS;

        // Permute columns to [eliminated | kept].
        std::vector<Index> perm = elim;
        perm.insert(perm.end(), keep.begin(), keep.end());
        Matrix a(n + k, total + 1);
        a.leftCols(total) = permuteColumns(m.leftCols(total), perm);
        a.col(total) = m.col(total);
        return condenseAugmented(a, total, static_cast<Index>(elim.size()));
    }

    /**
     * @brief Confidence ellipse of a 2-D marginal: points x = mu + S^T (r [cos t; sin t]).
     *
     * r^2 = mahalanobisThresholdSq(nSigma, 2) (probability-consistent n-sigma region).
     * @param idx     two indices of the state (order defines the ellipse axes)
     * @param nSigma  confidence level (default 3.0)
     * @param nPoints number of boundary samples (uniformly spaced in angle, not repeated)
     * @return 2 x nPoints matrix of boundary points
     */
    Matrix confidenceEllipse(const std::vector<Index>& idx, double nSigma = 3.0, Index nPoints = 100) const
    {
        requireSize(idx.size() == 2, "confidenceEllipse: need exactly two indices");
        requireSize(nPoints >= 3, "confidenceEllipse: need at least three points");
        const GaussianInfo m = marginal(idx);
        const Vector mu = m.mean();
        const Matrix s = m.sqrtCov();
        const double r = std::sqrt(mahalanobisThresholdSq(nSigma, 2.0));
        Matrix pts(2, nPoints);
        for (Index k = 0; k < nPoints; ++k)
        {
            const double t = 2.0 * std::numbers::pi_v<double> * static_cast<double>(k) / static_cast<double>(nPoints);
            Vector u(2);
            u << r * std::cos(t), r * std::sin(t);
            pts.col(k) = mu + s.transpose() * u;
        }
        return pts;
    }

    /**
     * @brief Confidence ellipsoid surface of a 3-D marginal: points x = mu + S^T (r u), u on the unit sphere.
     * @return 3 x (nLat * nLon) matrix; latitude samples include both poles.
     */
    Matrix confidenceEllipsoid(const std::vector<Index>& idx, double nSigma = 3.0, Index nLat = 20, Index nLon = 40) const
    {
        requireSize(idx.size() == 3, "confidenceEllipsoid: need exactly three indices");
        requireSize(nLat >= 2 && nLon >= 3, "confidenceEllipsoid: grid too small");
        const GaussianInfo m = marginal(idx);
        const Vector mu = m.mean();
        const Matrix s = m.sqrtCov();
        const double r = std::sqrt(mahalanobisThresholdSq(nSigma, 3.0));
        Matrix pts(3, nLat * nLon);
        Index col = 0;
        for (Index i = 0; i < nLat; ++i)
        {
            const double th = std::numbers::pi_v<double> * static_cast<double>(i) / static_cast<double>(nLat - 1);
            for (Index j = 0; j < nLon; ++j)
            {
                const double ph = 2.0 * std::numbers::pi_v<double> * static_cast<double>(j) / static_cast<double>(nLon);
                Vector u(3);
                u << std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th);
                pts.col(col++) = mu + s.transpose() * (r * u);
            }
        }
        return pts;
    }

    /**
     * @brief Homogeneous quadric Q (4 x 4) of the n-sigma ellipsoid of a 3-D marginal.
     *
     * [x; 1]^T Q [x; 1] = (x - mu)^T Lambda (x - mu) - c, with c = mahalanobisThresholdSq(nSigma, 3).
     * The boundary is the zero set; Q is built from Lambda and nu = Lambda mu (no inversion).
     */
    Matrix confidenceQuadric(const std::vector<Index>& idx, double nSigma = 3.0) const
    {
        requireSize(idx.size() == 3, "confidenceQuadric: need exactly three indices");
        const GaussianInfo m = marginal(idx);
        const Matrix lambda = m.information();
        const Vector nu = m.infoVec();
        const Scalar c = static_cast<Scalar>(mahalanobisThresholdSq(nSigma, 3.0));
        const Scalar muLmu = nu.dot(m.mean());
        Matrix q(4, 4);
        q.topLeftCorner(3, 3) = lambda;
        q.topRightCorner(3, 1) = -nu;
        q.bottomLeftCorner(1, 3) = -nu.transpose();
        q(3, 3) = muLmu - c;
        return q;
    }

private:
    GaussianInfo(Vector nu, Matrix xi) : nu_(std::move(nu)), xi_(std::move(xi)) {}

    static void requireSize(bool ok, const char* message)
    {
        if (!ok)
        {
            throw std::invalid_argument(std::string("GaussianInfo: ") + message);
        }
    }

    static void requireNonsingular(const Matrix& r, const char* where)
    {
        if (r.rows() > 0 && r.diagonal().cwiseAbs().minCoeff() == Scalar(0))
        {
            throw std::invalid_argument(std::string("GaussianInfo::") + where + ": singular covariance");
        }
    }

    static Matrix permuteColumns(const Matrix& a, const std::vector<Index>& perm)
    {
        Matrix out(a.rows(), static_cast<Index>(perm.size()));
        for (Index j = 0; j < static_cast<Index>(perm.size()); ++j)
        {
            out.col(j) = a.col(perm[static_cast<std::size_t>(j)]);
        }
        return out;
    }

    void validateSubset(const std::vector<Index>& subset) const
    {
        std::vector<bool> seen(static_cast<std::size_t>(dim()), false);
        for (Index idx : subset)
        {
            if (idx < 0 || idx >= dim())
            {
                throw std::invalid_argument("GaussianInfo: index out of range");
            }
            if (seen[static_cast<std::size_t>(idx)])
            {
                throw std::invalid_argument("GaussianInfo: repeated index");
            }
            seen[static_cast<std::size_t>(idx)] = true;
        }
    }

    /// Complement of subset within [0, dim()), ascending.
    std::vector<Index> complementOf(const std::vector<Index>& subset) const
    {
        std::vector<bool> in(static_cast<std::size_t>(dim()), false);
        for (Index idx : subset)
        {
            in[static_cast<std::size_t>(idx)] = true;
        }
        std::vector<Index> out;
        for (Index i = 0; i < dim(); ++i)
        {
            if (!in[static_cast<std::size_t>(i)])
            {
                out.push_back(i);
            }
        }
        return out;
    }

    /**
     * @brief Eliminates the first nElim variables of the augmented factor A = [X | b] (X has nVars columns).
     * @return density over the remaining variables (order preserved)
     */
    static GaussianInfo condenseAugmented(const Matrix& a, Index nVars, Index nElim)
    {
        const Matrix r = core::triangularise(a, nVars);
        const Index nk = nVars - nElim;
        const Matrix xk = r.block(nElim, nElim, nk, nk);
        const Vector wk = r.col(nVars).segment(nElim, nk);
        Vector nuk = xk.transpose() * wk;
        return GaussianInfo(std::move(nuk), xk);
    }

    Vector nu_;   ///< information vector
    Matrix xi_;   ///< upper-triangular square-root information
};

#endif
