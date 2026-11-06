#include "S6KeyframeFlow.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <Eigen/Geometry>
#include <opencv2/calib.hpp>
#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include "GaussianInfo.h"
#include "MeasurementIndoorFlow.h"
#include "rotation.hpp"

namespace slam
{

namespace
{

/// Misfit [px] of one pair under a rotation-only prediction pi(R x_A) (as IndoorFlowFrontEnd).
double rotationMisfitPx(const Eigen::Matrix3d& r, const IndoorFlowPair& p, double f)
{
    const Eigen::Vector3d q = r * Eigen::Vector3d(p.prevNorm.x(), p.prevNorm.y(), 1.0);
    if (!(std::abs(q.z()) > 1e-12))
    {
        return std::numeric_limits<double>::infinity();
    }
    return f * (Eigen::Vector2d(q.x() / q.z(), q.y() / q.z()) - p.currNorm).norm();
}

double medianOf(std::vector<double> v)
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

double CheiralityCensus::fractionPositive() const
{
    return total > 0 ? static_cast<double>(positive) / static_cast<double>(total)
                     : std::numeric_limits<double>::quiet_NaN();
}

CheiralityCensus cheiralityCensus(const std::vector<IndoorFlowPair>& pairs, const Eigen::Matrix3d& r,
                                  const Eigen::Vector3d& t)
{
    CheiralityCensus out;
    constexpr double kParallel = 1e-10;   // |Rx x y|^2 below this: the two rays are (nearly) parallel
    for (const IndoorFlowPair& p : pairs)
    {
        const Eigen::Vector3d x(p.prevNorm.x(), p.prevNorm.y(), 1.0);
        const Eigen::Vector3d y(p.currNorm.x(), p.currNorm.y(), 1.0);
        const Eigen::Vector3d rx = r * x;
        const Eigen::Vector3d rxy = rx.cross(y);
        const double den = rxy.squaredNorm();
        if (!(den > kParallel) || !std::isfinite(den))
        {
            ++out.undetermined;
            continue;
        }
        const double lambda = -(t.cross(y)).dot(rxy) / den;   // depth in camera A
        const double mu = (t.cross(rx)).dot(y.cross(rx)) / den;  // depth in camera B
        ++out.total;
        if (lambda > 0.0 && mu > 0.0)
        {
            ++out.positive;
        }
    }
    return out;
}

void mirrorKeyframeTranslation(SystemEstimator& est, const Camera& camera)
{
    const Eigen::Index n = est.dim();
    if (n < MeasurementIndoorFlow::kPreviousOffset + 6 || n < 6)
    {
        throw std::invalid_argument("mirrorKeyframeTranslation: state must contain the velocity and both poses");
    }
    const Eigen::VectorXd mu = est.mean();
    const Eigen::Index cur = MeasurementIndoorFlow::kCurrentOffset;
    const Eigen::Index prev = MeasurementIndoorFlow::kPreviousOffset;
    const Eigen::Vector3d cA = mu.segment<3>(prev) + rpy2rot(Eigen::Vector3d(mu.segment<3>(prev + 3))) * camera.rCBb();
    const Eigen::Vector3d cB = mu.segment<3>(cur) + rpy2rot(Eigen::Vector3d(mu.segment<3>(cur + 3))) * camera.rCBb();

    // delta = [-2 v; 0; 2 (cA - cB) on r_B; 0]: the posterior mean moves to the mirrored state (mean += Xi^T (Xi delta)).
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(n);
    delta.head<3>() = -2.0 * mu.head<3>();
    delta.segment<3>(cur) = 2.0 * (cA - cB);
    const Eigen::MatrixXd& xi = est.state().sqrtInfo();
    const Eigen::VectorXd shifted = est.state().infoVec() + xi.transpose() * (xi * delta);
    if (!shifted.allFinite())
    {
        throw std::invalid_argument("mirrorKeyframeTranslation: non-finite mean shift");
    }
    est.setState(GaussianInfo<double>::fromSqrtInfo(shifted, xi));
}

FlowUpdateResult updateFlowWithCheirality(SystemEstimator& est, const Camera& camera, const MeasurementIndoorFlow& fm,
                                          const FuncMinOptions& opts)
{
    FlowUpdateResult out;
    // The time update is done here, so that the saved prior is the predicted one (process() then makes a zero step).
    // Otherwise restoring the prior after the first update would drop the time update of the repeated update.
    if (fm.time() > est.time())
    {
        est.predict(fm.time());
    }
    const GaussianInfo<double> prior = est.state();
    out.update = fm.process(est, opts);
    if (out.update.map.status == FuncMinStatus::NumericalFailure || !est.mean().allFinite()
        || !est.state().sqrtInfo().allFinite())
    {
        return out;   // the caller reverts to its own pre-update state
    }
    const GaussianInfo<double> first = est.state();
    const IndoorFlowRelativePose rpFirst = indoorFlowRelativePose(est.mean(), camera.Rbc(), camera.rCBb(), false);
    const CheiralityCensus censusFirst = cheiralityCensus(fm.pairs(), rpFirst.R, rpFirst.t);
    out.fractionBefore = censusFirst.fractionPositive();
    out.fractionAfter = out.fractionBefore;
    if (censusFirst.total < kMinCensusPairs || !(out.fractionBefore < 0.5))
    {
        return out;   // physical branch (or too few determined depths to decide)
    }

    // Majority behind the cameras: repeat the update from the prior with the mirrored translation (one fresh update).
    est.setState(prior);
    mirrorKeyframeTranslation(est, camera);
    const MeasurementUpdateResult mirrored = fm.process(est, opts);
    if (mirrored.map.status != FuncMinStatus::NumericalFailure && est.mean().allFinite()
        && est.state().sqrtInfo().allFinite())
    {
        const IndoorFlowRelativePose rpMirror = indoorFlowRelativePose(est.mean(), camera.Rbc(), camera.rCBb(), false);
        const double fracMirror = cheiralityCensus(fm.pairs(), rpMirror.R, rpMirror.t).fractionPositive();
        if (fracMirror > out.fractionBefore)
        {
            out.update = mirrored;
            out.mirrored = true;
            out.fractionAfter = fracMirror;
            return out;
        }
    }
    est.setState(first);   // the mirrored branch is not better: keep the first posterior
    return out;
}

S6FlowFrame buildKeyframeFlowFrame(const Camera& camera, const SystemEstimator& est,
                                   const std::vector<IndoorFlowPixelMatch>& matches, const IndoorFlowOptions& opt)
{
    if (!camera.isValid())
    {
        throw std::invalid_argument("buildKeyframeFlowFrame: camera is not valid");
    }
    if (!(opt.sigmaPx > 0.0) || !(opt.ransacThresholdPx > 0.0) || !(opt.rotationMisfitPx > 0.0)
        || !(opt.gateNSigma > 0.0) || opt.minCorrespondences < 1)
    {
        throw std::invalid_argument("buildKeyframeFlowFrame: invalid options");
    }
    if (est.dim() < MeasurementIndoorFlow::kPreviousOffset + 6)
    {
        throw std::invalid_argument("buildKeyframeFlowFrame: estimator state does not contain etaPrev");
    }

    S6FlowFrame out;
    IndoorFlowFrame& fr = out.flow;
    fr.input = static_cast<int>(matches.size());
    const double f = 0.5 * (camera.fx() + camera.fy());
    fr.sigmaNorm = opt.sigmaPx / f;

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
    fr.normalised = static_cast<int>(pairs.size());
    if (fr.normalised < opt.minCorrespondences)
    {
        return out;
    }

    // 2. Model choice from the predicted relative rotation (as IndoorFlowFrontEnd).
    const Eigen::VectorXd x = est.mean();
    const IndoorFlowRelativePose prior = indoorFlowRelativePose(x, camera.Rbc(), camera.rCBb(), false);
    std::vector<double> misfit;
    misfit.reserve(pairs.size());
    for (const IndoorFlowPair& p : pairs)
    {
        misfit.push_back(rotationMisfitPx(prior.R, p, f));
    }
    fr.medianRotationMisfitPx = medianOf(misfit);
    fr.model = (fr.medianRotationMisfitPx < opt.rotationMisfitPx) ? IndoorFlowModel::Rotation : IndoorFlowModel::Epipolar;

    std::vector<IndoorFlowPair> kept;
    if (fr.model == IndoorFlowModel::Rotation)
    {
        // 3a. Consensus with the predicted rotation (no translation model).
        for (std::size_t i = 0; i < pairs.size(); ++i)
        {
            if (misfit[i] <= opt.ransacThresholdPx)
            {
                kept.push_back(pairs[i]);
            }
        }
        fr.consensus = static_cast<int>(kept.size());
    }
    else
    {
        // 3b. Five-point essential matrix with RANSAC on the normalised points (prior-free consensus).
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
        cv::Mat e;
        if (pairs.size() >= 8)
        {
            e = cv::findEssentialMat(p1, p2, 1.0, cv::Point2d(0.0, 0.0), cv::RANSAC, 0.999, opt.ransacThresholdPx / f,
                                     1000, mask);
        }
        if (!e.empty() && e.rows == 3 && e.cols == 3 && mask.rows == static_cast<int>(pairs.size()))
        {
            for (std::size_t i = 0; i < pairs.size(); ++i)
            {
                if (mask.at<std::uint8_t>(static_cast<int>(i)) != 0)
                {
                    kept.push_back(pairs[i]);
                }
            }
            // Relative pose of the inliers (cheirality-resolved; the translation is a unit direction).
            std::vector<cv::Point2d> q1;
            std::vector<cv::Point2d> q2;
            for (const IndoorFlowPair& p : kept)
            {
                q1.emplace_back(p.prevNorm.x(), p.prevNorm.y());
                q2.emplace_back(p.currNorm.x(), p.currNorm.y());
            }
            cv::Mat rCv;
            cv::Mat tCv;
            const int inl = cv::recoverPose(e, q1, q2, rCv, tCv, 1.0, cv::Point2d(0.0, 0.0));
            if (inl >= 8 && rCv.rows == 3 && tCv.rows == 3)
            {
                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        out.rRel(r, c) = rCv.at<double>(r, c);
                    }
                    out.tRel(r) = tCv.at<double>(r, 0);
                }
                out.hasRelative = out.rRel.allFinite() && out.tRel.allFinite() && out.tRel.norm() > 0.5;
                if (out.hasRelative)
                {
                    out.tRel.normalize();
                }
            }
        }
        else
        {
            kept.clear();   // RANSAC unavailable (degenerate set): no epipolar consensus, no update this frame
        }
        fr.consensus = static_cast<int>(kept.size());
    }

    fr.gated = static_cast<int>(kept.size());
    if (fr.gated < opt.minCorrespondences)
    {
        return out;
    }
    fr.pairs = std::move(kept);
    fr.valid = true;
    return out;
}

} // namespace slam
