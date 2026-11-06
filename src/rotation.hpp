#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <Eigen/Core>

/**
 * @brief Rotation utilities: elementary rotations, ZYX Euler angles (RPY), kinematics and derivatives.
 *
 * Conventions:
 *   - Theta = [phi; theta; psi] = [roll; pitch; yaw] (ZYX Euler angles, radians).
 *   - rpy2rot(Theta) = Rz(psi) Ry(theta) Rx(phi), i.e. R_nb (body -> world) for NED-like frames.
 *   - TK(Theta) maps body angular rates to Euler-angle rates: Thetadot = TK(Theta) * omega_b.
 *   - skew(v) v2 = v x v2; vee(skew(v)) = v.
 *   - Derivatives are hand-written and validated against central finite differences in the tests.
 *   - TK and rot2rpy are singular at pitch = +/-90 deg (gimbal lock); callers must avoid it.
 */

/** @brief Skew-symmetric matrix such that skew(a) b = a x b. */
inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d s;
    s << 0.0, -v(2), v(1),
         v(2), 0.0, -v(0),
         -v(1), v(0), 0.0;
    return s;
}

/** @brief Inverse of skew(): extracts the vector from a skew-symmetric matrix (average of both triangles). */
inline Eigen::Vector3d vee(const Eigen::Matrix3d& s)
{
    return 0.5 * Eigen::Vector3d(s(2, 1) - s(1, 2), s(0, 2) - s(2, 0), s(1, 0) - s(0, 1));
}

/** @brief Rotation about the x axis by angle a (right-handed). */
inline Eigen::Matrix3d rotx(double a)
{
    const double c = std::cos(a);
    const double s = std::sin(a);
    Eigen::Matrix3d r;
    r << 1.0, 0.0, 0.0,
         0.0, c, -s,
         0.0, s, c;
    return r;
}

/** @brief Rotation about the y axis by angle a (right-handed). */
inline Eigen::Matrix3d roty(double a)
{
    const double c = std::cos(a);
    const double s = std::sin(a);
    Eigen::Matrix3d r;
    r << c, 0.0, s,
         0.0, 1.0, 0.0,
         -s, 0.0, c;
    return r;
}

/** @brief Rotation about the z axis by angle a (right-handed). */
inline Eigen::Matrix3d rotz(double a)
{
    const double c = std::cos(a);
    const double s = std::sin(a);
    Eigen::Matrix3d r;
    r << c, -s, 0.0,
         s, c, 0.0,
         0.0, 0.0, 1.0;
    return r;
}

/**
 * @brief ZYX Euler angles to rotation matrix: R = Rz(psi) Ry(theta) Rx(phi).
 * @param Theta  [phi; theta; psi] (roll, pitch, yaw)
 */
inline Eigen::Matrix3d rpy2rot(const Eigen::Vector3d& Theta)
{
    return rotz(Theta(2)) * roty(Theta(1)) * rotx(Theta(0));
}

/**
 * @brief Rotation matrix to ZYX Euler angles [phi; theta; psi], valid for |theta| < 90 deg.
 *
 * theta = -asin(R(2,0)), phi = atan2(R(2,1), R(2,2)), psi = atan2(R(1,0), R(0,0)).
 */
inline Eigen::Vector3d rot2rpy(const Eigen::Matrix3d& R)
{
    const double sTheta = -R(2, 0);
    const double theta = std::asin(sTheta > 1.0 ? 1.0 : (sTheta < -1.0 ? -1.0 : sTheta));
    const double phi = std::atan2(R(2, 1), R(2, 2));
    const double psi = std::atan2(R(1, 0), R(0, 0));
    return Eigen::Vector3d(phi, theta, psi);
}

/** @brief dR/dphi of rpy2rot(Theta) (analytic product rule: Rz Ry dRx). */
inline Eigen::Matrix3d drpy2rotDphi(const Eigen::Vector3d& Theta)
{
    const double c = std::cos(Theta(0));
    const double s = std::sin(Theta(0));
    Eigen::Matrix3d dRx;
    dRx << 0.0, 0.0, 0.0,
           0.0, -s, -c,
           0.0, c, -s;
    return rotz(Theta(2)) * roty(Theta(1)) * dRx;
}

/** @brief dR/dtheta of rpy2rot(Theta) (analytic product rule: Rz dRy Rx). */
inline Eigen::Matrix3d drpy2rotDtheta(const Eigen::Vector3d& Theta)
{
    const double c = std::cos(Theta(1));
    const double s = std::sin(Theta(1));
    Eigen::Matrix3d dRy;
    dRy << -s, 0.0, c,
           0.0, 0.0, 0.0,
           -c, 0.0, -s;
    return rotz(Theta(2)) * dRy * rotx(Theta(0));
}

/** @brief dR/dpsi of rpy2rot(Theta) (analytic product rule: dRz Ry Rx). */
inline Eigen::Matrix3d drpy2rotDpsi(const Eigen::Vector3d& Theta)
{
    const double c = std::cos(Theta(2));
    const double s = std::sin(Theta(2));
    Eigen::Matrix3d dRz;
    dRz << -s, -c, 0.0,
           c, -s, 0.0,
           0.0, 0.0, 0.0;
    return dRz * roty(Theta(1)) * rotx(Theta(0));
}

/**
 * @brief Largest |pitch| (theta) for which the body kinematics are accepted by the estimator: 85 deg.
 *
 * TK has 1/cos(theta) and tan(theta) terms, so its norm grows without bound at +/-90 deg. Near that limit the
 * process-noise factor and the flow Jacobian lose all accuracy. Estimator code (SystemVisualNav) calls
 * requireKinematicPitchSupported() before using TK; the generic helpers below do not check, because callers such as
 * pose-landmark frames are not body attitudes and may legitimately be evaluated at any angle.
 */
inline constexpr double kMaxKinematicPitchRad = 85.0 * std::numbers::pi_v<double> / 180.0;

/**
 * @brief Documented guard for body attitudes used by the estimator.
 * @param Theta  [phi; theta; psi] (radians)
 * @throws std::domain_error if any component is non-finite or |theta| > kMaxKinematicPitchRad (85 deg)
 */
inline void requireKinematicPitchSupported(const Eigen::Vector3d& Theta)
{
    if (!Theta.allFinite())
    {
        throw std::domain_error("requireKinematicPitchSupported: attitude must be finite");
    }
    if (std::abs(Theta(1)) > kMaxKinematicPitchRad)
    {
        throw std::domain_error("requireKinematicPitchSupported: |pitch| exceeds 85 deg; the ZYX kinematic matrix is "
                                "singular at 90 deg and is not accurate near it");
    }
}

/**
 * @brief Kinematic matrix TK(Theta) of ZYX Euler angles: Thetadot = TK(Theta) omega_b.
 *
 * TK = [1, sphi*tth, cphi*tth; 0, cphi, -sphi; 0, sphi/cth, cphi/cth].
 * Singular at theta = +/-90 deg. No check is made here (see kMaxKinematicPitchRad and
 * requireKinematicPitchSupported, which the body-state estimator path calls).
 */
inline Eigen::Matrix3d kinematicMatrixTK(const Eigen::Vector3d& Theta)
{
    const double sp = std::sin(Theta(0));
    const double cp = std::cos(Theta(0));
    const double st = std::sin(Theta(1));
    const double ct = std::cos(Theta(1));
    const double tt = st / ct;
    Eigen::Matrix3d tk;
    tk << 1.0, sp * tt, cp * tt,
          0.0, cp, -sp,
          0.0, sp / ct, cp / ct;
    return tk;
}

/**
 * @brief Partial derivatives of TK(Theta) with respect to Theta = [phi; theta; psi].
 * @return array [dTK/dphi, dTK/dtheta, dTK/dpsi]; the psi derivative is identically zero.
 */
inline std::array<Eigen::Matrix3d, 3> kinematicMatrixTKDerivatives(const Eigen::Vector3d& Theta)
{
    const double sp = std::sin(Theta(0));
    const double cp = std::cos(Theta(0));
    const double st = std::sin(Theta(1));
    const double ct = std::cos(Theta(1));
    const double tt = st / ct;
    const double sec2 = 1.0 / (ct * ct);       // d tan(theta)/dtheta = sec^2(theta)
    const double dsecT = st / (ct * ct);       // d(1/cos(theta))/dtheta = sin(theta)/cos^2(theta)

    std::array<Eigen::Matrix3d, 3> d;
    // dTK/dphi
    d[0] << 0.0, cp * tt, -sp * tt,
            0.0, -sp, -cp,
            0.0, cp / ct, -sp / ct;
    // dTK/dtheta
    d[1] << 0.0, sp * sec2, cp * sec2,
            0.0, 0.0, 0.0,
            0.0, sp * dsecT, cp * dsecT;
    // dTK/dpsi
    d[2].setZero();
    return d;
}

#endif
