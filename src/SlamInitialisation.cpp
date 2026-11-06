#include "SlamInitialisation.h"

#include <cmath>
#include <stdexcept>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include "Pose.hpp"
#include "rotation.hpp"

namespace slam
{

std::vector<std::size_t> selectBirthFeatures(const std::vector<PointFeature>& features,
                                             const std::vector<bool>& featureUsed, const PredictionSet& predictions,
                                             const SlamConfig& cfg, int maxNew)
{
    std::vector<std::size_t> births;
    if (maxNew <= 0)
    {
        return births;
    }
    if (featureUsed.size() != features.size())
    {
        throw std::invalid_argument("selectBirthFeatures: featureUsed size mismatch");
    }
    for (std::size_t f = 0; f < features.size() && static_cast<int>(births.size()) < maxNew; ++f)
    {
        if (featureUsed[f])
        {
            continue;
        }
        const Eigen::Vector2d& p = features[f].pixel;
        bool nearPrediction = false;
        for (const LandmarkPrediction& pred : predictions.predictions)
        {
            if (pred.visible && (p - pred.pixel).norm() < cfg.birthSeparationPx)
            {
                nearPrediction = true;
                break;
            }
        }
        if (nearPrediction)
        {
            continue;
        }
        bool nearBirth = false;
        for (std::size_t b : births)
        {
            if ((p - features[b].pixel).norm() < cfg.birthMinSpacingPx)
            {
                nearBirth = true;
                break;
            }
        }
        if (!nearBirth)
        {
            births.push_back(f);
        }
    }
    return births;
}

Eigen::Index initialiseLandmark(SystemEstimator& est, SystemVisualNav& model, const Camera& camera,
                                const Eigen::Vector2d& pixel, double sigmaPx, double depthPriorMean,
                                double depthPriorSigma)
{
    if (!(sigmaPx > 0.0) || !(depthPriorMean > 0.0) || !(depthPriorSigma > 0.0) || !std::isfinite(sigmaPx))
    {
        throw std::invalid_argument("initialiseLandmark: sigmas and the depth prior must be positive");
    }
    const Eigen::Vector3d uc = camera.pixelToVector(pixel);
    if (!uc.allFinite())
    {
        return -1;
    }
    const Eigen::VectorXd mu = est.mean();
    const Eigen::Index n = est.dim();
    const Eigen::Vector3d rBNn = mu.segment<3>(SystemVisualNav::kEtaOffset);
    const Eigen::Vector3d thetaNb = mu.segment<3>(SystemVisualNav::kEtaOffset + 3);
    const Eigen::Matrix3d rbc = camera.Rbc();
    const Eigen::Vector3d rCBb = camera.rCBb();
    const double rho0 = depthPriorMean;
    const Eigen::Vector3d pc = rho0 * uc;

    // Mean and Jacobian with respect to eta = [rBNn; Thetanb].
    const Eigen::Vector3d lMean = rigid::cameraToWorld(pc, rBNn, thetaNb, rbc, rCBb);
    const Eigen::Vector3d t = rCBb + rbc * pc;
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(3, n);
    g.block<3, 3>(0, SystemVisualNav::kEtaOffset) = Eigen::Matrix3d::Identity();
    g.col(SystemVisualNav::kEtaOffset + 3) = drpy2rotDphi(thetaNb) * t;
    g.col(SystemVisualNav::kEtaOffset + 4) = drpy2rotDtheta(thetaNb) * t;
    g.col(SystemVisualNav::kEtaOffset + 5) = drpy2rotDpsi(thetaNb) * t;
    const Eigen::Vector3d c = lMean - g * mu;

    // Two directions perpendicular to the ray (camera frame) for the bearing noise.
    const Eigen::Vector3d helper = std::abs(uc.z()) < 0.9 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
    const Eigen::Vector3d b1 = (helper - helper.dot(uc) * uc).normalized();
    const Eigen::Vector3d b2 = uc.cross(b1);
    const Eigen::Matrix3d rnc = rpy2rot(thetaNb) * rbc;
    Eigen::Matrix3d jn;
    jn.col(0) = rnc * uc;
    jn.col(1) = rnc * (rho0 * b1);
    jn.col(2) = rnc * (rho0 * b2);

    const double focal = 0.5 * (camera.fx() + camera.fy());
    const double sigmaAngle = sigmaPx / focal;
    Eigen::Matrix3d dInv = Eigen::Matrix3d::Zero();
    dInv(0, 0) = 1.0 / depthPriorSigma;
    dInv(1, 1) = 1.0 / sigmaAngle;
    dInv(2, 2) = 1.0 / sigmaAngle;
    // lS = D^{-1} Jn^{-1}: lS^T lS = Jn^{-T} D^{-2} Jn^{-1} = (Jn D^2 Jn^T)^{-1}. Jn is a 3 x 3 geometric map
    // (not a covariance), so this explicit small inverse is the noise factor, not the filter information.
    const Eigen::Matrix3d lS = dInv * jn.inverse();
    if (!lS.allFinite())
    {
        return -1;
    }
    return model.addLandmarkLinear(est, SystemVisualNav::LandmarkType::Point, g, c, lS);
}

} // namespace slam
