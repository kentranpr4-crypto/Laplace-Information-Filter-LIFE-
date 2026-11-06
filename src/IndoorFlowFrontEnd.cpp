#include "IndoorFlowFrontEnd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <opencv2/geometry.hpp>
#include <opencv2/core.hpp>
#include "GaussianBase.h"

namespace slam
{

namespace
{

/// Misfit [px] of one pair under a rotation-only prediction pi(R x_A).
double rotationMisfitPx(const Eigen::Matrix3d& R, const IndoorFlowPair& p, double f)
{
    const Eigen::Vector3d q = R * Eigen::Vector3d(p.prevNorm.x(), p.prevNorm.y(), 1.0);
    if (!(std::abs(q.z()) > 1e-12))
    {
        return std::numeric_limits<double>::infinity();
    }
    return f * (Eigen::Vector2d(q.x() / q.z(), q.y() / q.z()) - p.currNorm).norm();
}

double median(std::vector<double> v)
{
    if (v.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    double m = v[mid];
    if (v.size() % 2 == 0)
    {
        const double lower = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid));
        m = 0.5 * (m + lower);
    }
    return m;
}

} // namespace

IndoorFlowFrame buildIndoorFlowFrame(const Camera& camera, const SystemEstimator& est,
                                     const std::vector<IndoorFlowPixelMatch>& matches, const IndoorFlowOptions& opt)
{
    if (!camera.isValid())
    {
        throw std::invalid_argument("buildIndoorFlowFrame: camera is not valid");
    }
    if (!(opt.sigmaPx > 0.0) || !(opt.ransacThresholdPx > 0.0) || !(opt.rotationMisfitPx > 0.0)
        || !(opt.gateNSigma > 0.0) || opt.minCorrespondences < 1)
    {
        throw std::invalid_argument("buildIndoorFlowFrame: invalid options");
    }
    if (est.dim() < MeasurementIndoorFlow::kPreviousOffset + 6)
    {
        throw std::invalid_argument("buildIndoorFlowFrame: estimator state does not contain etaPrev");
    }

    IndoorFlowFrame out;
    out.input = static_cast<int>(matches.size());
    const double f = 0.5 * (camera.fx() + camera.fy());
    out.sigmaNorm = opt.sigmaPx / f;

    // 1. Undistorted normalised correspondences.
    std::vector<IndoorFlowPair> pairs;
    pairs.reserve(matches.size());
    for (const IndoorFlowPixelMatch& m : matches)
    {
        const Eigen::Vector2d a = camera.pixelToNormalised(m.prevPx);
        const Eigen::Vector2d b = camera.pixelToNormalised(m.currPx);
        if (a.allFinite() && b.allFinite())
        {
            pairs.push_back({a, b});
        }
    }
    out.normalised = static_cast<int>(pairs.size());
    if (out.normalised < opt.minCorrespondences)
    {
        return out;
    }

    // 2. Model selection from the prior rotation (predicted state).
    const Eigen::VectorXd x = est.mean();
    const IndoorFlowRelativePose prior = indoorFlowRelativePose(x, camera.Rbc(), camera.rCBb(), false);
    std::vector<double> misfit;
    misfit.reserve(pairs.size());
    for (const IndoorFlowPair& p : pairs)
    {
        misfit.push_back(rotationMisfitPx(prior.R, p, f));
    }
    out.medianRotationMisfitPx = median(misfit);
    out.model = (out.medianRotationMisfitPx < opt.rotationMisfitPx) ? IndoorFlowModel::Rotation : IndoorFlowModel::Epipolar;

    std::vector<IndoorFlowPair> kept;
    if (out.model == IndoorFlowModel::Rotation)
    {
        // 3a. Consensus with the prior rotation (no translation model needed).
        for (std::size_t i = 0; i < pairs.size(); ++i)
        {
            if (misfit[i] <= opt.ransacThresholdPx)
            {
                kept.push_back(pairs[i]);
            }
        }
        out.consensus = static_cast<int>(kept.size());
    }
    else
    {
        // 3b. RANSAC with the 5-point essential matrix on normalised points (threshold converted to normalised units).
        std::vector<cv::Point2d> p1;
        std::vector<cv::Point2d> p2;
        p1.reserve(pairs.size());
        p2.reserve(pairs.size());
        for (const IndoorFlowPair& p : pairs)
        {
            p1.emplace_back(p.prevNorm.x(), p.prevNorm.y());
            p2.emplace_back(p.currNorm.x(), p.currNorm.y());
        }
        cv::Mat mask;
        bool ransacOk = false;
        if (pairs.size() >= 5)
        {
            const cv::Mat E = cv::findEssentialMat(p1, p2, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC, 0.999,
                                                   opt.ransacThresholdPx / f, 1000, mask);
            ransacOk = !E.empty() && !mask.empty();
        }
        if (ransacOk)
        {
            for (std::size_t i = 0; i < pairs.size(); ++i)
            {
                if (mask.at<std::uint8_t>(static_cast<int>(i)) != 0)
                {
                    kept.push_back(pairs[i]);
                }
            }
        }
        else
        {
            kept = pairs;   // RANSAC unavailable (degenerate set): the prior gate below is the only outlier test
        }
        out.consensus = static_cast<int>(kept.size());

        // 4b. Chi2 gate at the predicted state (dof 1 on the whitened Sampson residual).
        if (kept.size() >= 1)
        {
            const MeasurementIndoorFlow probe(0.0, IndoorFlowModel::Epipolar, kept, out.sigmaNorm, camera.Rbc(),
                                              camera.rCBb());
            const Eigen::VectorXd r = probe.whitenedResiduals(x, nullptr);
            const double thr = std::sqrt(GaussianBase::mahalanobisThresholdSq(opt.gateNSigma, 1.0));
            std::vector<IndoorFlowPair> gated;
            gated.reserve(kept.size());
            for (std::size_t i = 0; i < kept.size(); ++i)
            {
                if (std::isfinite(r(static_cast<Eigen::Index>(i))) && std::abs(r(static_cast<Eigen::Index>(i))) <= thr)
                {
                    gated.push_back(kept[i]);
                }
            }
            kept = std::move(gated);
        }
    }

    out.gated = static_cast<int>(kept.size());
    if (out.gated < opt.minCorrespondences)
    {
        return out;
    }
    out.pairs = std::move(kept);
    out.valid = true;
    return out;
}

} // namespace slam
