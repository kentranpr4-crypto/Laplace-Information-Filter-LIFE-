#include "MeasurementTagBundle.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include "MeasurementPointBundle.h"
#include "Pose.hpp"
#include "rotation.hpp"

namespace slam
{

TagCornerPrediction predictTagCorners(const Camera& camera, const Eigen::Vector3d& rBNn, const Eigen::Vector3d& thetaNb,
                                      const Eigen::Vector3d& rLNn, const Eigen::Vector3d& thetaLn,
                                      const Eigen::Matrix<double, 4, 3>& cornersLocal, double minDepth)
{
    TagCornerPrediction out;
    const Eigen::Matrix3d rLn = rpy2rot(thetaLn);
    const Eigen::Matrix3d dR[3] = {drpy2rotDphi(thetaLn), drpy2rotDtheta(thetaLn), drpy2rotDpsi(thetaLn)};

    for (int k = 0; k < 4; ++k)
    {
        const Eigen::Vector3d ck = cornersLocal.row(k).transpose();
        const Eigen::Vector3d pk = rLNn + rLn * ck;
        out.worldCorners.row(k) = pk.transpose();
        const PointPixelJacobian pj = pointPixelJacobian(camera, rBNn, thetaNb, pk, minDepth);
        if (!pj.valid)
        {
            out.valid = false;
            return out;   // uv stays NaN, Jacobians are not meaningful
        }
        out.uv.segment<2>(2 * k) = pj.uv;
        out.dEta.block<2, 6>(2 * k, 0) = pj.dUvdEta;
        // Landmark position: d P / d r = I, so d uv / d r = dU/dP.
        out.dLm.block<2, 3>(2 * k, 0) = pj.dUvdLm;
        // Landmark orientation: d P / d Theta_m = dR/dTheta_m c_k.
        for (int m = 0; m < 3; ++m)
        {
            out.dLm.block<2, 1>(2 * k, 3 + m) = pj.dUvdLm * (dR[m] * ck);
        }
    }
    out.valid = true;
    return out;
}

} // namespace slam

namespace
{

Eigen::VectorXd stackCorners(const std::vector<TagObservation>& obs)
{
    Eigen::VectorXd y(8 * static_cast<Eigen::Index>(obs.size()));
    for (std::size_t j = 0; j < obs.size(); ++j)
    {
        y.segment<8>(8 * static_cast<Eigen::Index>(j)) = obs[j].pixels;
    }
    return y;
}

} // namespace

MeasurementTagBundle::MeasurementTagBundle(double t, const Camera& camera, std::vector<TagObservation> obs,
                                           Eigen::Index poseOffset)
    : MeasurementBase(t, stackCorners(obs)), camera_(camera), obs_(std::move(obs)), poseOffset_(poseOffset)
{
    if (!camera_.isValid())
    {
        throw std::invalid_argument("MeasurementTagBundle: camera is not valid");
    }
    if (poseOffset_ < 0)
    {
        throw std::invalid_argument("MeasurementTagBundle: negative pose offset");
    }
    for (const TagObservation& o : obs_)
    {
        if (!(o.sigmaPx > 0.0) || !std::isfinite(o.sigmaPx))
        {
            throw std::invalid_argument("MeasurementTagBundle: pixel sigma must be finite and positive");
        }
        if (o.landmarkOffset < 0 || !o.pixels.allFinite() || !o.cornersLocal.allFinite())
        {
            throw std::invalid_argument("MeasurementTagBundle: invalid landmark offset, pixels or corners");
        }
    }
}

std::vector<Eigen::Index> MeasurementTagBundle::supportIndices(const SystemEstimator& /*system*/) const
{
    std::vector<Eigen::Index> idx;
    idx.reserve(6 + 6 * obs_.size());
    for (Eigen::Index i = 0; i < 6; ++i)
    {
        idx.push_back(poseOffset_ + i);
    }
    for (const TagObservation& o : obs_)
    {
        for (Eigen::Index i = 0; i < 6; ++i)
        {
            idx.push_back(o.landmarkOffset + i);
        }
    }
    return idx;
}

double MeasurementTagBundle::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& /*system*/,
                                           Eigen::VectorXd* g, Eigen::MatrixXd* H) const
{
    const Eigen::Index n = x.size();
    if (poseOffset_ + 6 > n)
    {
        throw std::invalid_argument("MeasurementTagBundle::logLikelihood: state too small for the pose block");
    }
    const Eigen::Vector3d rBNn = x.segment<3>(poseOffset_);
    const Eigen::Vector3d thetaNb = x.segment<3>(poseOffset_ + 3);
    if (g != nullptr)
    {
        g->setZero(n);
    }
    if (H != nullptr)
    {
        H->setZero(n, n);
    }

    const double log2Pi = std::log(2.0 * std::numbers::pi_v<double>);
    double value = 0.0;
    for (const TagObservation& o : obs_)
    {
        if (o.landmarkOffset + 6 > n)
        {
            throw std::invalid_argument("MeasurementTagBundle::logLikelihood: landmark block outside the state");
        }
        const slam::TagCornerPrediction pred = slam::predictTagCorners(
            camera_, rBNn, thetaNb, x.segment<3>(o.landmarkOffset), x.segment<3>(o.landmarkOffset + 3), o.cornersLocal);
        if (!pred.valid)
        {
            // A predicted corner is not in front of the camera: the trust-region driver rejects non-finite objectives.
            return std::numeric_limits<double>::quiet_NaN();
        }
        const Eigen::Matrix<double, 8, 1> e = pred.uv - o.pixels;
        const double s = o.sigmaPx;
        const double w = 1.0 / (s * s);
        value += -0.5 * w * e.squaredNorm() - 8.0 * std::log(s) - 4.0 * log2Pi;

        if (g != nullptr || H != nullptr)
        {
            // Local 8 x 12 Jacobian over (eta, landmark), scattered into the full vectors / matrices.
            Eigen::Matrix<double, 8, 12> j12;
            j12.block<8, 6>(0, 0) = pred.dEta;
            j12.block<8, 6>(0, 6) = pred.dLm;
            Eigen::Index idx[12];
            for (Eigen::Index i = 0; i < 6; ++i)
            {
                idx[i] = poseOffset_ + i;
                idx[6 + i] = o.landmarkOffset + i;
            }
            if (g != nullptr)
            {
                const Eigen::Matrix<double, 12, 1> gl = -w * (j12.transpose() * e);
                for (int a = 0; a < 12; ++a)
                {
                    (*g)(idx[a]) += gl(a);
                }
            }
            if (H != nullptr)
            {
                const Eigen::Matrix<double, 12, 12> hl = -w * (j12.transpose() * j12);
                for (int a = 0; a < 12; ++a)
                {
                    for (int b = 0; b < 12; ++b)
                    {
                        (*H)(idx[a], idx[b]) += hl(a, b);
                    }
                }
            }
        }
    }
    return value;
}

Eigen::VectorXd MeasurementTagBundle::predictedCorners(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd out(8 * static_cast<Eigen::Index>(obs_.size()));
    const Eigen::Vector3d rBNn = x.segment<3>(poseOffset_);
    const Eigen::Vector3d thetaNb = x.segment<3>(poseOffset_ + 3);
    for (std::size_t j = 0; j < obs_.size(); ++j)
    {
        const slam::TagCornerPrediction pred = slam::predictTagCorners(
            camera_, rBNn, thetaNb, x.segment<3>(obs_[j].landmarkOffset), x.segment<3>(obs_[j].landmarkOffset + 3),
            obs_[j].cornersLocal);
        out.segment<8>(8 * static_cast<Eigen::Index>(j)) = pred.uv;
    }
    return out;
}

Eigen::VectorXd MeasurementTagBundle::pixelVariances() const
{
    Eigen::VectorXd v(8 * static_cast<Eigen::Index>(obs_.size()));
    for (std::size_t j = 0; j < obs_.size(); ++j)
    {
        v.segment<8>(8 * static_cast<Eigen::Index>(j)).setConstant(obs_[j].sigmaPx * obs_[j].sigmaPx);
    }
    return v;
}
