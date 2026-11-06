#ifndef SLAMPOSEINITIALISATION_H
#define SLAMPOSEINITIALISATION_H

/**
 * @file SlamPoseInitialisation.h
 * @brief Single-frame tag pose estimation (IPPE + Gauss-Newton) and the correlated birth of a tag landmark.
 *
 * Camera-frame tag pose: p_c = R_ct p_t + t_ct for a canonical tag point p_t. Perturbations are left-invariant,
 * R_ct <- exp([dw]x) R_ct, t_ct <- t_ct + dt, so the pose information matrix is taken over xi = [dt; dw] (6 x 6).
 *
 * Birth. A detection gives the camera-frame pose hypotheses (IPPE_SQUARE returns the two planar solutions). The tag
 * landmark L = [r_jn^n; Theta_nj] is then the affine-in-eta-linearised map of the world tag pose,
 *   R_nj = R_nb Rbc R_ct Q^T,   r_jn^n = r_bn^n + R_nb (rCBb + Rbc t_ct),        (Q = Rz(q pi/2), see ArucoTags.h)
 * with Theta_nj = rot2rpy(R_nj). Linearising in eta (G = dL/deta, exact by hand-written derivatives) and treating the
 * tag-in-camera noise w = [dt; dw] ~ N(0, A^{-1}) (A = Gauss-Newton information) as independent of the state gives
 *   L = G x + c + w_L,   w_L = M_w w,   M_w = blockdiag(R_nc, TK(Theta_nj) R_nj^T R_nc),   R_nc = R_nb Rbc,
 * and the square-root information of w_L is lS = qr(Ra M_w^{-1}) with Ra^T Ra = A. The cross-covariance with eta is
 * therefore kept (SystemVisualNav::addLandmarkLinear).
 *
 * Orientation parameterisation. The landmark frame is the marker frame rotated by Q about its normal. Q is chosen at
 * birth so that the landmark x-axis is as close to horizontal as possible (|R_nj(2,0)| minimal): the ZYX Euler
 * representation is singular at pitch = +/-90 deg, and a wall tag whose pattern is turned by 90 deg would otherwise
 * sit at the singularity. Q is kept per landmark (4 values) and the corners in the landmark frame become Q c_k.
 */

#include <vector>
#include <Eigen/Core>
#include "Camera.h"

namespace slam
{

/** @brief One refined camera-frame pose hypothesis of a tag. */
struct TagPoseHypothesis
{
    Eigen::Matrix3d Rct = Eigen::Matrix3d::Identity();   ///< tag -> camera rotation (p_c = Rct p_t + tct)
    Eigen::Vector3d tct = Eigen::Vector3d::Zero();       ///< tag origin in camera coordinates [m]
    double rmsPx = 0.0;                                  ///< RMS pixel residual per component after refinement [px]
    Eigen::Matrix<double, 6, 6> sqrtInfo = Eigen::Matrix<double, 6, 6>::Zero();  ///< upper Ra, Ra^T Ra = A (xi = [dt; dw])
};

/**
 * @brief Both planar (IPPE_SQUARE) solutions of one tag detection, each refined by Gauss-Newton with the distorted camera.
 * @param camera      calibrated camera (intrinsics, distortion)
 * @param pixels      detected corners [u0 v0 ... u3 v3] in canonical order
 * @param canonical   canonical corners of the tag (rows, tag frame)
 * @param sigmaPx    pixel std per component (for the information matrix)
 * @return hypotheses sorted by rmsPx (best first); empty if no solution is valid (points behind the camera)
 */
std::vector<TagPoseHypothesis> refinedTagPoseHypotheses(const Camera& camera, const Eigen::Matrix<double, 8, 1>& pixels,
                                                        const Eigen::Matrix<double, 4, 3>& canonical, double sigmaPx);

/** @brief World pose of the tag: R_nt = R_nc R_ct, r_tn^n = r_cn^n + R_nc t_ct, from the camera pose in the state. */
void worldTagPose(const Camera& camera, const Eigen::VectorXd& x, Eigen::Index poseOffset, const Eigen::Matrix3d& Rct,
                  const Eigen::Vector3d& tct, Eigen::Matrix3d& Rnt, Eigen::Vector3d& rnt);

/** @brief Chooses the in-plane rotation q in {0,1,2,3} that makes the landmark x-axis closest to horizontal. */
int chooseInPlaneRotation(const Eigen::Matrix3d& Rnt);

/** @brief Linearised birth of one tag landmark: x_L = G x + c + w_L (see file documentation). */
struct TagBirthLinearisation
{
    Eigen::MatrixXd G;        ///< 6 x n, nonzero only in the eta columns
    Eigen::VectorXd c;        ///< 6 constant term
    Eigen::MatrixXd lS;       ///< 6 x 6 upper, lS^T lS = Qw^{-1}
    Eigen::Vector3d rLNn = Eigen::Vector3d::Zero();      ///< tag centre (world) at the linearisation point
    Eigen::Vector3d thetaLn = Eigen::Vector3d::Zero();   ///< landmark Euler angles at the linearisation point
    Eigen::Matrix3d Rnl = Eigen::Matrix3d::Identity();   ///< landmark rotation rpy2rot(thetaLn)
    int qIndex = 0;                                       ///< in-plane rotation index of the landmark frame
    Eigen::Matrix<double, 4, 3> cornersLocal = Eigen::Matrix<double, 4, 3>::Zero();  ///< landmark-frame corners Q c_k
};

/**
 * @param camera     calibrated camera
 * @param x          state mean (n); the body pose eta is read at poseOffset
 * @param poseOffset index of eta in x (6)
 * @param hyp        refined camera-frame hypothesis (Rct, tct, sqrtInfo)
 * @param canonical  canonical corners of the tag (rows)
 * @param regulariser  prior standard deviations of xi = [dt (m); dw (rad)] (ASSUMED, weak, e.g. 10 m and 1 rad). Stacked
 *                    with the refinement factor so that lS stays invertible for a near-degenerate (fronto-parallel) tag;
 *                    its influence is negligible when the refinement information is well conditioned
 * @return the affine birth map for SystemVisualNav::addLandmarkLinear
 */
TagBirthLinearisation linearisedTagBirth(const Camera& camera, const Eigen::VectorXd& x, Eigen::Index poseOffset,
                                         const TagPoseHypothesis& hyp, const Eigen::Matrix<double, 4, 3>& canonical,
                                         const Eigen::Matrix<double, 6, 1>& regulariser);

} // namespace slam

#endif
