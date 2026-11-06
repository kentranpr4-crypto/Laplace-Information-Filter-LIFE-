#ifndef MEASUREMENTHORIZON_H
#define MEASUREMENTHORIZON_H

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "MeasurementBase.h"

/**
 * @brief Visual-horizon measurement: boundary points on the sky/ground horizon of the current frame.
 *
 * Geometry. The horizon is the image of the set of directions that are horizontal in the world. A camera
 * direction v_c is horizontal iff e3^T R_nc v_c = 0, i.e. a^T v_c = 0 with
 *
 *   a(Theta) = R_nc^T e3 = Rbc^T R_nb^T e3 = Rbc^T (-sin th, cos th sin phi, cos th cos phi)^T   (R_nb = rpy2rot).
 *
 * In normalised image coordinates p = (x, y, 1) the horizon is the line a^T p = 0. Its signed distance from an
 * observed boundary point p_i (in normalised units) is
 *
 *   r_i(Theta) = a^T p_i / N(a),   N(a) = sqrt(a_x^2 + a_y^2).
 *
 * Observation y_i = 0 with Gaussian noise sigma (normalised units = sigma_px / f). Then
 *
 *   log p(y|x) = -(1/(2 sigma^2)) sum_i r_i^2 - m (log sigma + (1/2) log 2 pi),
 *   grad_j     = -(1/sigma^2) sum_i r_i dr_i/dTheta_j,
 *   Hess       = -(1/sigma^2) sum_i (dr_i/dTheta)(dr_i/dTheta)^T      (Gauss-Newton),
 *
 * with dr/da = p/N - (a^T p)/N^3 (a_x, a_y, 0)^T and da/dTheta_j = Rbc^T (dR_nb/dTheta_j)^T e3 (hand-written).
 *
 * Observability. The horizon depends only on roll and pitch of the body (eta_k Thetanb components 0 and 1, state
 * indices 9 and 10). Yaw (state index 11) does not change a(Theta) and is unobserved by this measurement. The
 * model neglects the dip of the horizon (~ sqrt(2 h / R_earth) = 0.26 deg at 63 m), and the sky/ground boundary
 * is the visible silhouette, which can lie a few tenths of a degree above the geometric horizon (distant ridges).
 * That bias is absorbed by a conservative sigma (see architecture notes).
 */
class MeasurementHorizon : public MeasurementBase
{
public:
    static constexpr Eigen::Index kThetaOffset = 9;  ///< index of Thetanb (roll, pitch, yaw) in the eta_k block
    static constexpr Eigen::Index kSupportDim = 3;   ///< support: roll, pitch, yaw (yaw has zero derivative)

    /**
     * @param t             measurement time (s)
     * @param pointsNorm    undistorted normalised boundary points (x, y); at least 2
     * @param sigmaNorm     standard deviation of the signed point-to-line distance [normalised units]; > 0
     * @param Rbc           body-to-camera rotation
     * @throws std::invalid_argument for invalid inputs
     */
    MeasurementHorizon(double t, std::vector<Eigen::Vector2d> pointsNorm, double sigmaNorm, const Eigen::Matrix3d& Rbc);

    /** @brief log p(y|x) with GN gradient and Hessian. Full-size outputs. */
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd* g,
                         Eigen::MatrixXd* H) const override;

    /** @brief State indices 9..11 (Thetanb of eta_k). */
    std::vector<Eigen::Index> supportIndices(const SystemEstimator& system) const override;

    /**
     * @brief Signed distances r(x) (N) and optional Jacobian dr/d(Thetanb) (N x 3).
     * @throws std::domain_error if the horizon is degenerate (camera looking along the vertical, N(a) ~ 0)
     */
    Eigen::VectorXd predict(const Eigen::VectorXd& x, Eigen::MatrixXd* jac3) const;

    /** @brief Horizon normal a(Theta) = Rbc^T R_nb^T e3 (camera coordinates). */
    static Eigen::Vector3d horizonNormal(const Eigen::Vector3d& theta, const Eigen::Matrix3d& Rbc);

    /** @brief Number of boundary points. */
    std::size_t size() const { return points_.size(); }
    /** @brief Boundary points (normalised). */
    const std::vector<Eigen::Vector2d>& points() const { return points_; }
    /** @brief Noise standard deviation (normalised units). */
    double sigma() const { return sigmaNorm_; }

private:
    std::vector<Eigen::Vector2d> points_;
    double sigmaNorm_;
    Eigen::Matrix3d Rbc_;
};

#endif
