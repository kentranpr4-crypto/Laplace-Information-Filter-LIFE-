#include "SlamTrackTriangulation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include "rotation.hpp"

namespace slam
{

namespace
{

/// Geometry of one camera ray at a pose: world direction v = Rnc a and origin c, with the Euler derivatives.
struct RayGeometry
{
    Eigen::Matrix3d Rnc = Eigen::Matrix3d::Zero();   ///< camera-to-world rotation R_nb(Theta) Rbc
    Eigen::Vector3d c = Eigen::Vector3d::Zero();     ///< camera origin in world r + R_nb rCBb
    Eigen::Vector3d a = Eigen::Vector3d::Zero();     ///< normalised ray (x, y, 1)
    Eigen::Vector3d v = Eigen::Vector3d::Zero();     ///< world ray direction Rnc a
    std::array<Eigen::Matrix3d, 3> dRnb{};           ///< d R_nb / d Theta_k (set to zero when not needed)
};

RayGeometry rayGeometry(const Camera& camera, const Eigen::Matrix<double, 6, 1>& eta, const Eigen::Vector2d& ray,
                        bool withJac)
{
    RayGeometry g;
    const Eigen::Vector3d r = eta.head<3>();
    const Eigen::Vector3d theta = eta.tail<3>();
    const Eigen::Matrix3d rnb = rpy2rot(theta);
    g.Rnc = rnb * camera.Rbc();
    g.c = r + rnb * camera.rCBb();
    g.a = Eigen::Vector3d(ray.x(), ray.y(), 1.0);
    g.v = g.Rnc * g.a;
    for (Eigen::Matrix3d& d : g.dRnb)
    {
        d.setZero();   // Eigen matrices are NaN-initialised in Debug builds: every entry is set explicitly
    }
    if (withJac)
    {
        g.dRnb[0] = drpy2rotDphi(theta);
        g.dRnb[1] = drpy2rotDtheta(theta);
        g.dRnb[2] = drpy2rotDpsi(theta);
    }
    return g;
}

/// Right-hand-side contribution of one ray to dL: -dQ (c - L) + P dc, with P = I - v v^T / s and
/// dQ = (dv v^T + v dv^T) / s - v v^T (2 v.dv) / s^2 (see the header for the derivation).
Eigen::Vector3d rayTerm(const Eigen::Vector3d& v, const Eigen::Vector3d& dv, const Eigen::Vector3d& dc,
                        const Eigen::Vector3d& w)
{
    const double s = v.squaredNorm();
    const double vw = v.dot(w);
    const double vdv = v.dot(dv);
    const Eigen::Vector3d dQw = (dv * vw + v * dv.dot(w)) / s - v * (2.0 * vw * vdv / (s * s));
    const Eigen::Vector3d pdc = dc - v * (v.dot(dc) / s);
    return pdc - dQw;
}

} // namespace

TwoViewTriangulation triangulateTwoView(const Camera& camera, const Eigen::Matrix<double, 6, 1>& etaA,
                                        const Eigen::Vector2d& rayA, const Eigen::Matrix<double, 6, 1>& etaB,
                                        const Eigen::Vector2d& rayB, bool withJac)
{
    if (!camera.isValid())
    {
        throw std::invalid_argument("triangulateTwoView: camera is not valid");
    }
    if (!etaA.allFinite() || !etaB.allFinite() || !rayA.allFinite() || !rayB.allFinite())
    {
        throw std::invalid_argument("triangulateTwoView: poses and rays must be finite");
    }

    TwoViewTriangulation out;
    const RayGeometry gA = rayGeometry(camera, etaA, rayA, withJac);
    const RayGeometry gB = rayGeometry(camera, etaB, rayB, withJac);
    const double sA = gA.v.squaredNorm();
    const double sB = gB.v.squaredNorm();
    if (!(sA > 0.0) || !(sB > 0.0) || !std::isfinite(sA) || !std::isfinite(sB))
    {
        return out;
    }

    // Normal equations of the perpendicular-distance cost sum_i ||P_i (L - c_i)||^2.
    const Eigen::Matrix3d pA = Eigen::Matrix3d::Identity() - gA.v * gA.v.transpose() / sA;
    const Eigen::Matrix3d pB = Eigen::Matrix3d::Identity() - gB.v * gB.v.transpose() / sB;
    const Eigen::Matrix3d m = pA + pB;
    const Eigen::Vector3d b = pA * gA.c + pB * gB.c;

    // Non-degenerate geometry: the smallest eigenvalue of M must be well above the rounding level (non-parallel rays).
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(0.5 * (m + m.transpose()));
    if (es.info() != Eigen::Success || !(es.eigenvalues()(0) > 1e-9 * es.eigenvalues()(2)))
    {
        return out;
    }
    const Eigen::LDLT<Eigen::Matrix3d> ldlt(m);
    const Eigen::Vector3d L = ldlt.solve(b);
    out.L = L;
    if (!L.allFinite())
    {
        return out;
    }

    // Consistency of the point with both cameras: depth, triangulation angle, normalised-ray misfit.
    const Eigen::Vector3d pcA = gA.Rnc.transpose() * (L - gA.c);
    const Eigen::Vector3d pcB = gB.Rnc.transpose() * (L - gB.c);
    out.depthA = pcA.z();
    out.depthB = pcB.z();
    const Eigen::Vector3d uA = L - gA.c;
    const Eigen::Vector3d uB = L - gB.c;
    const double cosAngle = std::clamp(uA.dot(uB) / (uA.norm() * uB.norm()), -1.0, 1.0);
    out.angleDeg = std::acos(cosAngle) * 180.0 / std::numbers::pi;
    const double f = 0.5 * (camera.fx() + camera.fy());
    if (pcA.z() > 0.0 && pcB.z() > 0.0)
    {
        const double eA = f * (Eigen::Vector2d(pcA.x() / pcA.z(), pcA.y() / pcA.z()) - rayA).norm();
        const double eB = f * (Eigen::Vector2d(pcB.x() / pcB.z(), pcB.y() / pcB.z()) - rayB).norm();
        out.maxReprojPx = std::max(eA, eB);
    }
    else
    {
        out.maxReprojPx = std::numeric_limits<double>::infinity();
    }
    out.valid = pcA.z() > 0.0 && pcB.z() > 0.0 && std::isfinite(out.maxReprojPx);

    if (withJac)
    {
        const Eigen::Vector3d wA = gA.c - L;
        const Eigen::Vector3d wB = gB.c - L;
        const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

        // Ray A (keyframe): body position r_A (dv = 0, dc = e_k), Euler angles (dv = dR_k Rbc a, dc = dR_k rCBb).
        for (int k = 0; k < 3; ++k)
        {
            out.dLdEtaA.col(k) = ldlt.solve(rayTerm(gA.v, zero, Eigen::Vector3d::Unit(k), wA));
            const Eigen::Matrix3d dRk = gA.dRnb[static_cast<std::size_t>(k)];
            const Eigen::Vector3d dv = dRk * camera.Rbc() * gA.a;
            const Eigen::Vector3d dc = dRk * camera.rCBb();
            out.dLdEtaA.col(3 + k) = ldlt.solve(rayTerm(gA.v, dv, dc, wA));
        }
        // Ray B (current): same parameterisation.
        for (int k = 0; k < 3; ++k)
        {
            out.dLdEtaB.col(k) = ldlt.solve(rayTerm(gB.v, zero, Eigen::Vector3d::Unit(k), wB));
            const Eigen::Matrix3d dRk = gB.dRnb[static_cast<std::size_t>(k)];
            const Eigen::Vector3d dv = dRk * camera.Rbc() * gB.a;
            const Eigen::Vector3d dc = dRk * camera.rCBb();
            out.dLdEtaB.col(3 + k) = ldlt.solve(rayTerm(gB.v, dv, dc, wB));
        }
        // Normalised ray coordinates: dv = R_nc e_x or e_y (the ray direction is R_nc (x, y, 1)), dc = 0.
        out.dLdRayA.col(0) = ldlt.solve(rayTerm(gA.v, gA.Rnc.col(0), zero, wA));
        out.dLdRayA.col(1) = ldlt.solve(rayTerm(gA.v, gA.Rnc.col(1), zero, wA));
        out.dLdRayB.col(0) = ldlt.solve(rayTerm(gB.v, gB.Rnc.col(0), zero, wB));
        out.dLdRayB.col(1) = ldlt.solve(rayTerm(gB.v, gB.Rnc.col(1), zero, wB));
    }
    return out;
}

TwoViewBirth twoViewBirth(const Camera& camera, const Eigen::VectorXd& mu, Eigen::Index offsetA, Eigen::Index offsetB,
                          const Eigen::Vector2d& pixelA, const Eigen::Vector2d& pixelB, double sigmaPx)
{
    if (!camera.isValid())
    {
        throw std::invalid_argument("twoViewBirth: camera is not valid");
    }
    if (!(sigmaPx > 0.0) || !std::isfinite(sigmaPx))
    {
        throw std::invalid_argument("twoViewBirth: pixel sigma must be finite and positive");
    }
    const Eigen::Index n = mu.size();
    if (!mu.allFinite() || offsetA < 0 || offsetB < 0 || offsetA + 6 > n || offsetB + 6 > n)
    {
        throw std::invalid_argument("twoViewBirth: state must be finite and contain both pose blocks");
    }

    TwoViewBirth out;
    const Eigen::Vector2d rayA = camera.pixelToNormalised(pixelA);
    const Eigen::Vector2d rayB = camera.pixelToNormalised(pixelB);
    if (!rayA.allFinite() || !rayB.allFinite())
    {
        return out;
    }
    out.tri = triangulateTwoView(camera, mu.segment<6>(offsetA), rayA, mu.segment<6>(offsetB), rayB, true);
    if (!out.tri.valid)
    {
        return out;
    }

    // Linear map of the augmentation: L = G x + c + w.
    out.g = Eigen::MatrixXd::Zero(3, n);
    out.g.block<3, 6>(0, offsetB) = out.tri.dLdEtaB;
    out.g.block<3, 6>(0, offsetA) = out.tri.dLdEtaA;
    out.c = out.tri.L - out.g * mu;

    // Noise: the four normalised coordinates are independent with sd sigmaPx / f, so Q = sigma_n^2 Ju Ju^T (3 x 3).
    const double f = 0.5 * (camera.fx() + camera.fy());
    const double sigmaN = sigmaPx / f;
    Eigen::Matrix<double, 3, 4> ju;
    ju << out.tri.dLdRayA, out.tri.dLdRayB;
    // Thin QR of A^T with A = sigma_n Ju (4 x 3 transposed): A^T = Qr R gives R^T R = A A^T = Q (no Q is formed).
    const Eigen::Matrix<double, 4, 3> at = sigmaN * ju.transpose();
    const Eigen::HouseholderQR<Eigen::Matrix<double, 4, 3>> qr(at);
    const Eigen::Matrix3d r = qr.matrixQR().topLeftCorner<3, 3>().triangularView<Eigen::Upper>();
    const double diagMin = r.diagonal().cwiseAbs().minCoeff();
    if (!(diagMin > 1e-12 * r.cwiseAbs().maxCoeff()) || !r.allFinite())
    {
        return out;   // the birth noise is degenerate (rank-deficient two-view Jacobian)
    }
    // lS = R^{-T}: lS^T lS = R^{-1} R^{-T} = (R^T R)^{-1} = Q^{-1}. A triangular solve, not an inverse of Q.
    out.lS = r.transpose().triangularView<Eigen::Lower>().solve(Eigen::Matrix3d::Identity());
    out.valid = out.lS.allFinite() && out.c.allFinite() && out.g.allFinite();
    return out;
}

} // namespace slam
