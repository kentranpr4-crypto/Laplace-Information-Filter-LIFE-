#ifndef CORELINALG_H
#define CORELINALG_H

#include <algorithm>
#include <stdexcept>
#include <Eigen/Core>

/**
 * @brief Dense linear-algebra primitives for the square-root information filter (core only).
 *
 * Everything in the estimation path is expressed through Q-less Householder triangularisation,
 * never through explicit covariance formation or inversion.
 */
namespace core
{

/**
 * @brief Householder triangularisation (Q-less QR) of a dense matrix with optional right-hand sides.
 *
 * Applies an implicit orthogonal Q^T to A = [X | B] (X has nVars columns, B the remaining columns)
 * and returns the first nVars rows of Q^T A, i.e. [R | W] with
 *   - R upper triangular (nVars x nVars) with non-negative diagonal,
 *   - R^T R = X^T X,  R^T W = X^T B   (orthogonal invariance),
 *   - W = top rows of the transformed right-hand sides.
 * The discarded rows carry only residual (constant) terms of the least-squares factor.
 *
 * Rank-deficient or short inputs are handled: when A has fewer rows than nVars the result is
 * zero-padded at the bottom, so R^T R = X^T X still holds exactly.
 *
 * @param A      input matrix (copied by value; consumed)
 * @param nVars  number of leading "variable" columns to triangularise
 * @return nVars x A.cols() matrix [R | W]
 * @throws std::invalid_argument if nVars is outside [0, A.cols()]
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
triangularise(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A, Eigen::Index nVars)
{
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    const Index m = A.rows();
    const Index c = A.cols();
    if (nVars < 0 || nVars > c)
    {
        throw std::invalid_argument("core::triangularise: nVars out of range");
    }

    const Index nRef = std::min(m, nVars);
    for (Index j = 0; j < nRef; ++j)
    {
        const Index len = m - j;
        const Scalar normx = A.col(j).tail(len).norm();
        if (normx == Scalar(0))
        {
            continue;
        }
        // Householder reflector H = I - 2 v v^T / (v^T v) mapping x = A(j:m, j) to alpha e1.
        const Scalar alpha = (A(j, j) >= Scalar(0) ? Scalar(-1) : Scalar(1)) * normx;
        Vector v = A.col(j).tail(len);
        v(0) -= alpha;
        const Scalar vtv = v.squaredNorm();
        if (vtv == Scalar(0))
        {
            continue;
        }
        auto block = A.block(j, j, len, c - j);
        block -= (Scalar(2) / vtv) * v * (v.transpose() * block);
        A(j, j) = alpha;
        A.col(j).segment(j + 1, len - 1).setZero();
    }

    Matrix out = Matrix::Zero(nVars, c);
    out.topRows(nRef) = A.topRows(nRef);
    for (Index j = 0; j < nRef; ++j)
    {
        // Enforce exact triangularity (eliminated entries are zero by construction).
        out.col(j).segment(j + 1, nVars - j - 1).setZero();
    }

    // Consistent sign convention: non-negative pivots (flip the whole row, rhs included).
    for (Index i = 0; i < nRef; ++i)
    {
        if (out(i, i) < Scalar(0))
        {
            out.row(i) *= Scalar(-1);
        }
    }
    return out;
}

/**
 * @brief Upper-triangular factor R with R^T R = A^T A, where R is A.cols() x A.cols().
 *
 * Works for tall, square, wide and rank-deficient A (rows padded with zeros when rows < cols).
 */
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
upperFactor(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A)
{
    return triangularise(A, A.cols());
}

} // namespace core

#endif
