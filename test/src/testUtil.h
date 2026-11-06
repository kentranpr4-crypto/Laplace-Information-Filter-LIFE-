#ifndef TESTUTIL_H
#define TESTUTIL_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <Eigen/Core>
#include <doctest/doctest.h>

/**
 * @brief Shared helpers for the unit tests (formatting and central finite differences).
 *
 * The finite-difference helpers are the independent numerical reference for every hand-written
 * Jacobian, gradient and Hessian in the core library (AGENTS.md / USER decision: no autodiff).
 */
namespace testutil
{

/** @brief Formats any dense Eigen expression for doctest messages. */
template<typename Derived>
std::string toString(const Eigen::MatrixBase<Derived>& m)
{
    std::ostringstream os;
    os << m.derived().format(Eigen::IOFormat(8, 0, " ", "\n", "[", "]"));
    return os.str();
}

/**
 * @brief Central-difference gradient of a scalar function of a vector.
 * @param f  callable double(const Eigen::VectorXd&)
 * @param x  evaluation point
 * @param h  step (default 1e-6, error O(h^2))
 */
template<typename F>
Eigen::VectorXd fdGradient(const F& f, const Eigen::VectorXd& x, double h = 1e-6)
{
    Eigen::VectorXd g(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i)
    {
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xm = x;
        xp(i) += h;
        xm(i) -= h;
        g(i) = (f(xp) - f(xm)) / (2.0 * h);
    }
    return g;
}

/**
 * @brief Central-difference Hessian of a scalar function of a vector (symmetrised).
 */
template<typename F>
Eigen::MatrixXd fdHessian(const F& f, const Eigen::VectorXd& x, double h = 1e-4)
{
    const Eigen::Index n = x.size();
    Eigen::MatrixXd H(n, n);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            Eigen::VectorXd pp = x, pm = x, mp = x, mm = x;
            pp(i) += h; pp(j) += h;
            pm(i) += h; pm(j) -= h;
            mp(i) -= h; mp(j) += h;
            mm(i) -= h; mm(j) -= h;
            H(i, j) = (f(pp) - f(pm) - f(mp) + f(mm)) / (4.0 * h * h);
        }
    }
    return 0.5 * (H + H.transpose());
}

/**
 * @brief Central-difference Jacobian of a vector-valued function of a vector.
 * @return m x n matrix J with J(i,j) = d f_i / d x_j.
 */
template<typename F>
Eigen::MatrixXd fdJacobian(const F& f, const Eigen::VectorXd& x, double h = 1e-6)
{
    const Eigen::Index n = x.size();
    Eigen::VectorXd f0 = Eigen::VectorXd(f(x));
    Eigen::MatrixXd J(f0.size(), n);
    for (Eigen::Index j = 0; j < n; ++j)
    {
        Eigen::VectorXd xp = x;
        Eigen::VectorXd xm = x;
        xp(j) += h;
        xm(j) -= h;
        Eigen::VectorXd fp = Eigen::VectorXd(f(xp));
        Eigen::VectorXd fm = Eigen::VectorXd(f(xm));
        J.col(j) = (fp - fm) / (2.0 * h);
    }
    return J;
}

/**
 * @brief Central-difference derivative of a matrix-valued function of a scalar.
 */
template<typename F>
Eigen::MatrixXd fdScalarDerivative(const F& f, double t, double h = 1e-6)
{
    Eigen::MatrixXd fp = Eigen::MatrixXd(f(t + h));
    Eigen::MatrixXd fm = Eigen::MatrixXd(f(t - h));
    return (fp - fm) / (2.0 * h);
}

/**
 * @brief Relative-to-scale matrix comparison used where isApprox is unreliable near zero.
 * @return true if max|a-b| <= tol * max(1, max|b|).
 */
template<typename DA, typename DB>
bool nearMatrix(const Eigen::MatrixBase<DA>& a, const Eigen::MatrixBase<DB>& b, double tol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return false;
    }
    const double scale = std::max(1.0, b.derived().cwiseAbs().maxCoeff());
    return (a.derived() - b.derived()).cwiseAbs().maxCoeff() <= tol * scale;
}

/** @brief Deterministic pseudo-random matrix with entries uniform in [-1, 1] (fixed seed for reproducibility). */
inline Eigen::MatrixXd randomMatrix(Eigen::Index rows, Eigen::Index cols, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::MatrixXd m(rows, cols);
    for (Eigen::Index j = 0; j < cols; ++j)
    {
        for (Eigen::Index i = 0; i < rows; ++i)
        {
            m(i, j) = dist(rng);
        }
    }
    return m;
}

/** @brief Deterministic random vector with entries uniform in [-1, 1]. */
inline Eigen::VectorXd randomVector(Eigen::Index n, unsigned seed)
{
    return randomMatrix(n, 1, seed).col(0);
}

/** @brief Deterministic random symmetric positive definite matrix (well conditioned). */
inline Eigen::MatrixXd randomSpd(Eigen::Index n, unsigned seed)
{
    const Eigen::MatrixXd a = randomMatrix(n, n, seed);
    return (a * a.transpose()) / static_cast<double>(n) + 0.5 * Eigen::MatrixXd::Identity(n, n);
}

} // namespace testutil

/**
 * @brief Attaches the name and value of an Eigen expression to the current doctest context.
 * Use inside a SCENARIO/SUBCASE; the message is shown when a following CHECK/REQUIRE fails.
 */
#define CAPTURE_EIGEN(x) INFO(#x, " =\n", ::testutil::toString(x))

#endif
