#include "SlamPoseInitialisation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/QR>
#include <opencv2/calib3d.hpp>
#include "ArucoTags.h"
#include "CoreLinalg.h"
#include "Pose.hpp"
#include "rotation.hpp"

namespace slam
{

namespace
{

/// Exponential map of SO(3): exp([w]x), exact for any w (identity for w = 0).
Eigen::Matrix3d expSO3(const Eigen::Vector3d& w)
{
    const double th = w.norm();
    if (th < 1e-12)
    {
        return Eigen::Matrix3d::Identity() + skew(w);
    }
    return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}

/// Camera-frame corner pixels, their 8 x 6 Jacobian with respect to xi = [dt; dw] (left perturbation) and validity.
bool cornerResidualAndJacobian(const Camera& camera, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                               const Eigen::Matrix<double, 4, 3>& canonical, Eigen::Matrix<double, 8, 1>& uv,
                               Eigen::Matrix<double, 8, 6>& jac)
{
    for (int k = 0; k < 4; ++k)
    {
        const Eigen::Vector3d pt = canonical.row(k).transpose();
        const Eigen::Vector3d pc = R * pt + t;
        if (!(pc.z() > 1e-6) || !pc.allFinite())
        {
            return false;
        }
        const auto [px, jp] = camera.vectorToPixelWithJacobian(pc);
        if (!px.allFinite() || !jp.allFinite())
        {
            return false;
        }
        uv.segment<2>(2 * k) = px;
        // d p_c / d[dt; dw] = [I, -[R p]x]   (left perturbation R <- exp([dw]x) R).
        Eigen::Matrix<double, 3, 6> dpc;
        dpc.leftCols<3>() = Eigen::Matrix3d::Identity();
        dpc.rightCols<3>() = -skew(R * pt);
        jac.block<2, 6>(2 * k, 0) = jp * dpc;
    }
    return true;
}

/// Gauss-Newton refinement of one planar pose hypothesis with the distorted camera model.
bool refineHypothesis(const Camera& camera, const Eigen::Matrix<double, 8, 1>& z,
                      const Eigen::Matrix<double, 4, 3>& canonical, Eigen::Matrix3d& R, Eigen::Vector3d& t,
                      Eigen::Matrix<double, 8, 1>& residual, Eigen::Matrix<double, 8, 6>& jac)
{
    for (int it = 0; it < 30; ++it)
    {
        Eigen::Matrix<double, 8, 1> uv;
        Eigen::Matrix<double, 8, 6> j;
        if (!cornerResidualAndJacobian(camera, R, t, canonical, uv, j))
        {
            return false;
        }
        const Eigen::Matrix<double, 8, 1> e = uv - z;
        // Gauss-Newton step = least-squares solution of J delta = -e. Solved by a column-pivoted QR of J itself;
        // the normal matrix J^T J (an information matrix, squared condition number) is never formed.
        const Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 8, 6>> qr(j);
        if (qr.rank() < 6)
        {
            return false;   // degenerate view: the planar pose is not determined by the four corners
        }
        const Eigen::Matrix<double, 6, 1> delta = qr.solve(-e);
        if (!delta.allFinite())
        {
            return false;
        }
        t += delta.head<3>();
        R = expSO3(delta.tail<3>()) * R;
        if (delta.norm() < 1e-10)
        {
            break;
        }
    }
    Eigen::Matrix<double, 8, 1> uv;
    if (!cornerResidualAndJacobian(camera, R, t, canonical, uv, jac))
    {
        return false;
    }
    residual = uv - z;
    return true;
}

} // namespace

std::vector<TagPoseHypothesis> refinedTagPoseHypotheses(const Camera& camera, const Eigen::Matrix<double, 8, 1>& pixels,
                                                        const Eigen::Matrix<double, 4, 3>& canonical, double sigmaPx)
{
    if (!(sigmaPx > 0.0) || !std::isfinite(sigmaPx))
    {
        throw std::invalid_argument("refinedTagPoseHypotheses: sigma must be finite and positive");
    }
    std::vector<TagPoseHypothesis> out;
    if (!camera.isValid() || !pixels.allFinite())
    {
        return out;
    }

    std::vector<cv::Point3d> objectPoints(4);
    std::vector<cv::Point2d> imagePoints(4);
    for (int k = 0; k < 4; ++k)
    {
        objectPoints[static_cast<std::size_t>(k)] = cv::Point3d(canonical(k, 0), canonical(k, 1), canonical(k, 2));
        imagePoints[static_cast<std::size_t>(k)] = cv::Point2d(pixels(2 * k), pixels(2 * k + 1));
    }
    cv::Mat kMat(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            kMat.at<double>(r, c) = camera.cameraMatrix()(r, c);
        }
    }
    const std::vector<double>& dc = camera.distortionCoefficients();
    cv::Mat distMat = dc.empty() ? cv::Mat() : cv::Mat(1, static_cast<int>(dc.size()), CV_64F);
    for (std::size_t i = 0; i < dc.size(); ++i)
    {
        distMat.at<double>(0, static_cast<int>(i)) = dc[i];
    }

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    try
    {
        cv::solvePnPGeneric(objectPoints, imagePoints, kMat, distMat, rvecs, tvecs, false, cv::SOLVEPNP_IPPE_SQUARE);
    }
    catch (const cv::Exception&)
    {
        return out;   // degenerate configuration: no hypothesis
    }

    for (std::size_t s = 0; s < rvecs.size() && s < tvecs.size(); ++s)
    {
        cv::Mat rot;
        cv::Rodrigues(rvecs[s], rot);
        Eigen::Matrix3d R;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                R(r, c) = rot.at<double>(r, c);
            }
        }
        Eigen::Vector3d t(tvecs[s].at<double>(0), tvecs[s].at<double>(1), tvecs[s].at<double>(2));

        Eigen::Matrix<double, 8, 1> residual;
        Eigen::Matrix<double, 8, 6> jac;
        if (!refineHypothesis(camera, pixels, canonical, R, t, residual, jac))
        {
            continue;
        }
        TagPoseHypothesis h;
        h.Rct = R;
        h.tct = t;
        h.rmsPx = std::sqrt(residual.squaredNorm() / 8.0);
        h.sqrtInfo = core::upperFactor(Eigen::MatrixXd(jac / sigmaPx));
        out.push_back(h);
    }
    std::sort(out.begin(), out.end(),
              [](const TagPoseHypothesis& a, const TagPoseHypothesis& b) { return a.rmsPx < b.rmsPx; });
    return out;
}

void worldTagPose(const Camera& camera, const Eigen::VectorXd& x, Eigen::Index poseOffset, const Eigen::Matrix3d& Rct,
                  const Eigen::Vector3d& tct, Eigen::Matrix3d& Rnt, Eigen::Vector3d& rnt)
{
    const Eigen::Vector3d rBNn = x.segment<3>(poseOffset);
    const Eigen::Vector3d thetaNb = x.segment<3>(poseOffset + 3);
    const Eigen::Matrix3d Rnb = rpy2rot(thetaNb);
    const Eigen::Matrix3d Rnc = Rnb * camera.Rbc();
    const Eigen::Vector3d rCNn = rBNn + Rnb * camera.rCBb();
    Rnt = Rnc * Rct;
    rnt = rCNn + Rnc * tct;
}

int chooseInPlaneRotation(const Eigen::Matrix3d& Rnt)
{
    int best = 0;
    double bestValue = std::numeric_limits<double>::infinity();
    for (int q = 0; q < 4; ++q)
    {
        const Eigen::Matrix3d rL = Rnt * tags::inPlaneRotation(q).transpose();
        const double v = std::abs(rL(2, 0));   // z-component of the landmark x-axis = -sin(pitch)
        if (v < bestValue)
        {
            bestValue = v;
            best = q;
        }
    }
    return best;
}

TagBirthLinearisation linearisedTagBirth(const Camera& camera, const Eigen::VectorXd& x, Eigen::Index poseOffset,
                                         const TagPoseHypothesis& hyp, const Eigen::Matrix<double, 4, 3>& canonical,
                                         const Eigen::Matrix<double, 6, 1>& regulariser)
{
    if (x.size() < poseOffset + 6)
    {
        throw std::invalid_argument("linearisedTagBirth: state too small for the body pose");
    }
    if ((regulariser.array() <= 0.0).any())
    {
        throw std::invalid_argument("linearisedTagBirth: regulariser standard deviations must be positive");
    }
    const Eigen::Index n = x.size();
    const Eigen::Vector3d rBNn = x.segment<3>(poseOffset);
    const Eigen::Vector3d thetaNb = x.segment<3>(poseOffset + 3);
    const Eigen::Matrix3d Rnb = rpy2rot(thetaNb);
    const Eigen::Matrix3d Rbc = camera.Rbc();
    const Eigen::Vector3d rCBb = camera.rCBb();
    const Eigen::Matrix3d Rnc = Rnb * Rbc;

    Eigen::Matrix3d Rnt;
    Eigen::Vector3d rnt;
    worldTagPose(camera, x, poseOffset, hyp.Rct, hyp.tct, Rnt, rnt);
    const int q = chooseInPlaneRotation(Rnt);
    const Eigen::Matrix3d Q = tags::inPlaneRotation(q);

    // Landmark frame: R_nL = Rnb M with M = Rbc Rct Q^T (constant in eta).
    const Eigen::Matrix3d M = Rbc * hyp.Rct * Q.transpose();
    const Eigen::Matrix3d RnL = Rnb * M;
    const Eigen::Vector3d thetaL = rot2rpy(RnL);
    const Eigen::Vector3d v = rCBb + Rbc * hyp.tct;
    const Eigen::Vector3d rL = rBNn + Rnb * v;

    // Mean landmark vector and its Jacobian with respect to eta = [rBNn; Thetanb] (hand-written):
    //   dr_L/drBNn = I,                  dr_L/dTheta_k = (dRnb/dTheta_k) v,
    //   dTheta_L/d eta_pos = 0,          dTheta_L/dTheta_nb = TK(Theta_L) M^T TK(Theta_nb)^{-1}.
    // (Derivation: a body perturbation dTheta_nb moves R_nb by exp([dw_n]x), dw_n = Rnb TK^{-1} dTheta_nb,
    //  and R_nL = Rnb M changes by the same left perturbation, so dTheta_L = TK(Theta_L) R_nL^T dw_n.)
    Eigen::Matrix<double, 6, 6> gEta = Eigen::Matrix<double, 6, 6>::Zero();
    gEta.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    gEta.block<3, 1>(0, 3) = drpy2rotDphi(thetaNb) * v;
    gEta.block<3, 1>(0, 4) = drpy2rotDtheta(thetaNb) * v;
    gEta.block<3, 1>(0, 5) = drpy2rotDpsi(thetaNb) * v;
    const Eigen::Matrix3d tkL = kinematicMatrixTK(thetaL);
    const Eigen::Matrix3d tkB = kinematicMatrixTK(thetaNb);
    gEta.block<3, 3>(3, 3) = tkL * M.transpose() * tkB.inverse();

    TagBirthLinearisation out;
    out.G = Eigen::MatrixXd::Zero(6, n);
    out.G.block<6, 6>(0, poseOffset) = gEta;
    Eigen::VectorXd lMean(6);
    lMean << rL, thetaL;
    out.c = lMean - gEta * x.segment<6>(poseOffset);

    // Noise map w_L = Mw xi with xi = [dt; dw_c] (camera-frame tag perturbation), Mw = blockdiag(Rnc, TKL RnL^T Rnc).
    const Eigen::Matrix3d nBlock = tkL * RnL.transpose() * Rnc;
    Eigen::Matrix<double, 6, 6> mwInv = Eigen::Matrix<double, 6, 6>::Zero();
    mwInv.block<3, 3>(0, 0) = Rnc.transpose();
    mwInv.block<3, 3>(3, 3) = nBlock.inverse();

    // Information of xi: stacked Ra (from the refinement) and a weak prior (prior std per component) for numerical safety.
    Eigen::Matrix<double, 12, 6> stacked;
    stacked.topRows<6>() = hyp.sqrtInfo;
    stacked.bottomRows<6>() = regulariser.cwiseInverse().asDiagonal();
    const Eigen::MatrixXd raReg = core::upperFactor(Eigen::MatrixXd(stacked));
    out.lS = core::upperFactor(Eigen::MatrixXd(raReg * mwInv));

    out.rLNn = rL;
    out.thetaLn = thetaL;
    out.Rnl = RnL;
    out.qIndex = q;
    out.cornersLocal = canonical * Q.transpose();   // rows: (Q c_k)^T = c_k^T Q^T
    return out;
}

} // namespace slam
