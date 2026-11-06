#include "MeasurementPointBundle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include "Pose.hpp"
#include "rotation.hpp"

using slam::PointPixelJacobian;
using slam::pointPixelJacobian;

namespace slam
{

double PixelNoiseModel::sigma(const Camera& camera, const Eigen::Vector2d& u) const
{
    const Eigen::Vector2d c(camera.cx(), camera.cy());
    const double halfDiagonal = std::hypot(0.5 * camera.imageWidth(), 0.5 * camera.imageHeight());
    const double rho = std::min(1.0, (u - c).norm() / halfDiagonal);
    return sigma0 + sigmaEdge * rho;
}

PointPixelJacobian pointPixelJacobian(const Camera& camera, const Eigen::Vector3d& rBNn, const Eigen::Vector3d& thetaNb,
                                      const Eigen::Vector3d& rLNn, double minDepth)
{
    PointPixelJacobian out;
    const rigid::CameraPointJacobian cp =
        rigid::cameraPointJacobian(rLNn, rBNn, thetaNb, camera.Rbc(), camera.rCBb());
    out.pc = cp.pc;
    if (!(cp.pc.z() > minDepth) || !cp.pc.allFinite())
    {
        return out;
    }
    const auto [uv, jUv] = camera.vectorToPixelWithJacobian(cp.pc);
    if (!uv.allFinite() || !jUv.allFinite())
    {
        return out;
    }
    out.valid = true;
    out.uv = uv;
    // Chain rule: d u / d(.) = (d u / d p_c) (d p_c / d(.)).
    out.dUvdLm = jUv * cp.dPc_dPn;
    out.dUvdEta.block<2, 3>(0, 0) = jUv * cp.dPc_drbn;
    out.dUvdEta.block<2, 3>(0, 3) = jUv * cp.dPc_dTheta;
    return out;
}

} // namespace slam

namespace
{

Eigen::VectorXd stackPixels(const std::vector<MeasurementPointBundle::Observation>& obs)
{
    Eigen::VectorXd y(2 * static_cast<Eigen::Index>(obs.size()));
    for (std::size_t j = 0; j < obs.size(); ++j)
    {
        y.segment<2>(2 * static_cast<Eigen::Index>(j)) = obs[j].pixel;
    }
    return y;
}

} // namespace

MeasurementPointBundle::MeasurementPointBundle(double t, const Camera& camera, std::vector<Observation> obs,
                                               Eigen::Index poseOffset)
    : MeasurementBase(t, stackPixels(obs)), camera_(camera), obs_(std::move(obs)), poseOffset_(poseOffset)
{
    if (!camera_.isValid())
    {
        throw std::invalid_argument("MeasurementPointBundle: camera is not valid");
    }
    if (poseOffset_ < 0)
    {
        throw std::invalid_argument("MeasurementPointBundle: negative pose offset");
    }
    for (const Observation& o : obs_)
    {
        if (!(o.sigmaPx > 0.0) || !std::isfinite(o.sigmaPx))
        {
            throw std::invalid_argument("MeasurementPointBundle: pixel sigma must be finite and positive");
        }
        if (o.landmarkOffset < 0 || !o.pixel.allFinite())
        {
            throw std::invalid_argument("MeasurementPointBundle: invalid landmark offset or pixel");
        }
    }
}

std::vector<Eigen::Index> MeasurementPointBundle::supportIndices(const SystemEstimator& /*system*/) const
{
    std::vector<Eigen::Index> idx;
    idx.reserve(6 + 3 * obs_.size());
    for (Eigen::Index i = 0; i < 6; ++i)
    {
        idx.push_back(poseOffset_ + i);
    }
    for (const Observation& o : obs_)
    {
        for (Eigen::Index i = 0; i < 3; ++i)
        {
            idx.push_back(o.landmarkOffset + i);
        }
    }
    return idx;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& /*system*/,
                                             Eigen::VectorXd* g, Eigen::MatrixXd* H) const
{
    const Eigen::Index n = x.size();
    if (poseOffset_ + 6 > n)
    {
        throw std::invalid_argument("MeasurementPointBundle::logLikelihood: state too small for the pose block");
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
    for (const Observation& o : obs_)
    {
        if (o.landmarkOffset + 3 > n)
        {
            throw std::invalid_argument("MeasurementPointBundle::logLikelihood: landmark block outside the state");
        }
        const Eigen::Vector3d rLNn = x.segment<3>(o.landmarkOffset);
        const PointPixelJacobian pj = pointPixelJacobian(camera_, rBNn, thetaNb, rLNn);
        if (!pj.valid)
        {
            // Point not in front of the camera: the trust-region driver rejects non-finite trial objectives.
            return std::numeric_limits<double>::quiet_NaN();
        }
        const Eigen::Vector2d e = pj.uv - o.pixel;
        const double s = o.sigmaPx;
        const double w = 1.0 / (s * s);
        value += -0.5 * w * e.squaredNorm() - 2.0 * std::log(s) - log2Pi;

        if (g != nullptr || H != nullptr)
        {
            // Local 9 x 9 block: eta (6) then landmark (3); scattered into the full vectors/matrices.
            Eigen::Matrix<double, 2, 9> j9;
            j9.block<2, 6>(0, 0) = pj.dUvdEta;
            j9.block<2, 3>(0, 6) = pj.dUvdLm;
            Eigen::Index idx[9];
            for (Eigen::Index i = 0; i < 6; ++i)
            {
                idx[i] = poseOffset_ + i;
            }
            for (Eigen::Index i = 0; i < 3; ++i)
            {
                idx[6 + i] = o.landmarkOffset + i;
            }
            if (g != nullptr)
            {
                const Eigen::Matrix<double, 9, 1> gl = -w * (j9.transpose() * e);
                for (int a = 0; a < 9; ++a)
                {
                    (*g)(idx[a]) += gl(a);
                }
            }
            if (H != nullptr)
            {
                const Eigen::Matrix<double, 9, 9> hl = -w * (j9.transpose() * j9);
                for (int a = 0; a < 9; ++a)
                {
                    for (int b = 0; b < 9; ++b)
                    {
                        (*H)(idx[a], idx[b]) += hl(a, b);
                    }
                }
            }
        }
    }
    return value;
}

Eigen::VectorXd MeasurementPointBundle::predictedPixels(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd out(2 * static_cast<Eigen::Index>(obs_.size()));
    const Eigen::Vector3d rBNn = x.segment<3>(poseOffset_);
    const Eigen::Vector3d thetaNb = x.segment<3>(poseOffset_ + 3);
    for (std::size_t j = 0; j < obs_.size(); ++j)
    {
        const PointPixelJacobian pj =
            pointPixelJacobian(camera_, rBNn, thetaNb, x.segment<3>(obs_[j].landmarkOffset));
        out.segment<2>(2 * static_cast<Eigen::Index>(j)) = pj.uv;
    }
    return out;
}

Eigen::VectorXd MeasurementPointBundle::pixelSigmas() const
{
    Eigen::VectorXd s(2 * static_cast<Eigen::Index>(obs_.size()));
    for (std::size_t j = 0; j < obs_.size(); ++j)
    {
        s.segment<2>(2 * static_cast<Eigen::Index>(j)).setConstant(obs_[j].sigmaPx);
    }
    return s;
}
