#include "SlamPoseAssociation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "CoreLinalg.h"
#include "Pose.hpp"
#include "rotation.hpp"

namespace slam
{

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

/**
 * @brief Upper R with R^T R = X^T X + sigma^2 I, from the Q-less QR of the stacked matrix [X; sigma I].
 *
 * X is (rows x m), the factor-space image of the innovation Jacobian. The innovation covariance is therefore never
 * formed as a product and never factorised; the gate uses R^{-T} e with a triangular solve.
 * @return the m x m factor, or an empty matrix if the factor is singular or non-finite
 */
Eigen::MatrixXd innovationSqrt(const Eigen::MatrixXd& x, double sigma)
{
    const Eigen::Index m = x.cols();
    Eigen::MatrixXd stacked(x.rows() + m, m);
    stacked.topRows(x.rows()) = x;
    stacked.bottomRows(m) = sigma * Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd r = core::upperFactor(stacked);
    if (!r.allFinite() || !(r.diagonal().cwiseAbs().minCoeff() > 0.0))
    {
        return Eigen::MatrixXd();
    }
    return r;
}

/// Mean edge length of a predicted 8-vector of corners [px].
double meanEdge(const Eigen::Matrix<double, 8, 1>& uv)
{
    double s = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        const int k1 = (k + 1) % 4;
        s += (uv.segment<2>(2 * k) - uv.segment<2>(2 * k1)).norm();
    }
    return 0.25 * s;
}

} // namespace

bool pointInTagQuad(const Eigen::Vector2d& p, const Eigen::Matrix<double, 8, 1>& uv)
{
    // Convex polygon: the point is inside when it lies on the same side of all four directed edges.
    bool hasPos = false;
    bool hasNeg = false;
    for (int k = 0; k < 4; ++k)
    {
        const Eigen::Vector2d a = uv.segment<2>(2 * k);
        const Eigen::Vector2d b = uv.segment<2>(2 * ((k + 1) % 4));
        const Eigen::Vector2d e = b - a;
        const double cross = e.x() * (p.y() - a.y()) - e.y() * (p.x() - a.x());
        if (cross > 0.0) hasPos = true;
        if (cross < 0.0) hasNeg = true;
    }
    return !(hasPos && hasNeg);
}

namespace
{

bool allInside(const Eigen::Matrix<double, 8, 1>& uv, double w, double h, double margin)
{
    for (int k = 0; k < 4; ++k)
    {
        const double u = uv(2 * k);
        const double v = uv(2 * k + 1);
        if (!(u >= -margin && u <= w - 1.0 + margin && v >= -margin && v <= h - 1.0 + margin))
        {
            return false;
        }
    }
    return true;
}

} // namespace

TagPredictionSet predictTagCandidates(const SystemEstimator& est, const SystemVisualNav& model, const Camera& camera,
                                      const std::vector<TagPredictionInput>& inputs, const TagPredictionOptions& opts)
{
    if (!(opts.sigmaPx > 0.0) || !std::isfinite(opts.sigmaPx))
    {
        throw std::invalid_argument("predictTagCandidates: sigma must be finite and positive");
    }
    TagPredictionSet set;
    if (inputs.empty())
    {
        return set;
    }

    const Eigen::VectorXd mu = est.mean();
    const Eigen::Vector3d rBNn = mu.segment<3>(SystemVisualNav::kEtaOffset);
    const Eigen::Vector3d thetaNb = mu.segment<3>(SystemVisualNav::kEtaOffset + 3);
    const Eigen::Matrix3d Rnb = rpy2rot(thetaNb);
    const Eigen::Vector3d rCNn = rBNn + Rnb * camera.rCBb();
    const int w = camera.imageWidth();
    const int h = camera.imageHeight();

    for (const TagPredictionInput& in : inputs)
    {
        const Eigen::Index offset = model.landmarkOffset(in.id);
        const Eigen::Vector3d rLNn = mu.segment<3>(offset);
        const Eigen::Vector3d thetaLn = mu.segment<3>(offset + 3);
        const TagCornerPrediction pred = predictTagCorners(camera, rBNn, thetaNb, rLNn, thetaLn, in.cornersLocal, opts.minDepth);
        if (!pred.valid)
        {
            continue;
        }
        const double depth = rigid::worldToCamera(rLNn, rBNn, thetaNb, camera.Rbc(), camera.rCBb()).z();
        if (!(depth <= opts.maxDepth))
        {
            continue;
        }
        if (!allInside(pred.uv, w, h, opts.candidateMarginPx))
        {
            continue;
        }

        TagCandidate c;
        c.record = in.record;
        c.id = in.id;
        c.offset = offset;
        c.cornersLocal = in.cornersLocal;
        c.pred = pred;
        c.depth = depth;
        c.sidePx = meanEdge(pred.uv);
        c.centre = Eigen::Vector2d(pred.uv(0) + pred.uv(2) + pred.uv(4) + pred.uv(6),
                                   pred.uv(1) + pred.uv(3) + pred.uv(5) + pred.uv(7)) * 0.25;

        // Strict visibility: inside the image (shrunk), large enough, and the tag front faces the camera.
        const Eigen::Matrix3d Rtag = rpy2rot(thetaLn);
        const Eigen::Vector3d normal = Rtag.col(2);
        const Eigen::Vector3d toCamera = rCNn - rLNn;
        const double facing = normal.dot(toCamera) / std::max(toCamera.norm(), 1e-12);
        c.visible = allInside(pred.uv, w, h, -opts.visibleMarginPx) && c.sidePx >= opts.minVisibleSidePx
                    && facing > opts.minFacingCos;
        set.candidates.push_back(c);
    }
    if (set.candidates.empty())
    {
        return set;
    }

    // Occlusion (predicted, depth-ordered): a visible candidate whose centre lies inside the quad of a candidate that is
    // nearer by occlusionDepthMarginM is not expected to be detected. Such a candidate is neither visible nor a miss.
    for (std::size_t j = 0; j < set.candidates.size(); ++j)
    {
        TagCandidate& cj = set.candidates[j];
        if (!cj.visible)
        {
            continue;
        }
        for (std::size_t i = 0; i < set.candidates.size(); ++i)
        {
            if (i == j)
            {
                continue;
            }
            const TagCandidate& ci = set.candidates[i];
            if (ci.depth + opts.occlusionDepthMarginM < cj.depth && pointInTagQuad(cj.centre, ci.pred.uv))
            {
                cj.occluded = true;
                cj.visible = false;
                break;
            }
        }
    }

    // Union marginal over [eta; L_1; ...; L_C] (eta is state indices 6..11; the landmark blocks follow in candidate order).
    set.unionIndices.clear();
    for (Eigen::Index i = 0; i < 6; ++i)
    {
        set.unionIndices.push_back(SystemVisualNav::kEtaOffset + i);
    }
    for (TagCandidate& c : set.candidates)
    {
        c.unionOffset = static_cast<Eigen::Index>(set.unionIndices.size());
        for (Eigen::Index i = 0; i < 6; ++i)
        {
            set.unionIndices.push_back(c.offset + i);
        }
    }
    const GaussianInfo<double> marg = est.state().marginal(set.unionIndices);
    set.xiUnion = marg.sqrtInfo();
    const Eigen::Index nu = static_cast<Eigen::Index>(set.unionIndices.size());

    for (TagCandidate& c : set.candidates)
    {
        Eigen::MatrixXd hRow = Eigen::MatrixXd::Zero(8, nu);
        hRow.block<8, 6>(0, 0) = c.pred.dEta;
        hRow.block<8, 6>(0, c.unionOffset) = c.pred.dLm;
        // X = Xi^{-T} H^T by a lower-triangular solve; H P H^T = X^T X (P = Xi^{-1} Xi^{-T}).
        const Eigen::MatrixXd x = set.xiUnion.transpose().triangularView<Eigen::Lower>().solve(hRow.transpose());
        c.S = x.transpose() * x + opts.sigmaPx * opts.sigmaPx * Eigen::Matrix<double, 8, 8>::Identity();   // display/tests
        // The gate uses the factor only: R^T R = [X; sigma I]^T [X; sigma I] = S, from a Q-less QR (S is never factorised).
        const Eigen::MatrixXd r = innovationSqrt(x, opts.sigmaPx);
        if (r.size() == 0)
        {
            throw std::runtime_error("predictTagCandidates: innovation covariance is not positive definite");
        }
        c.sqrtS = r;
    }
    return set;
}

TagAssociationResult associateTagDetections(const TagPredictionSet& predictions,
                                            const std::vector<tags::TagDetection>& detections,
                                            const TagAssociationOptions& opts)
{
    if (!(opts.gateThresholdSq > 0.0))
    {
        throw std::invalid_argument("associateTagDetections: gate threshold must be set (chi2 quantile)");
    }
    const std::size_t nd = detections.size();
    const std::size_t nc = predictions.candidates.size();
    TagAssociationResult out;
    out.detectionCandidate.assign(nd, -1);
    out.candidateDetection.assign(nc, -1);
    out.candidateGated.assign(nc, false);
    out.detectionD2.assign(nd, kInf);
    out.ambiguous.assign(nd, false);
    out.conflict.assign(nd, false);
    out.birthCandidate.assign(nd, false);

    // Squared Mahalanobis distances d2[i][j] for gated pairs (only the gated ones are kept).
    std::vector<std::vector<std::pair<double, std::size_t>>> gated(nd);
    for (std::size_t i = 0; i < nd; ++i)
    {
        const Eigen::Matrix<double, 8, 1> z = detections[i].cornerVector();
        for (std::size_t j = 0; j < nc; ++j)
        {
            const TagCandidate& c = predictions.candidates[j];
            const Eigen::Matrix<double, 8, 1> e = z - c.pred.uv;
            // d^2 = e^T S^{-1} e = ||R^{-T} e||^2 (triangular solve on the stored factor, S = R^T R).
            const Eigen::Matrix<double, 8, 1> q = c.sqrtS.transpose().triangularView<Eigen::Lower>().solve(e);
            const double d2 = q.squaredNorm();
            if (d2 <= opts.gateThresholdSq)
            {
                gated[i].emplace_back(d2, j);
                out.candidateGated[j] = true;
                out.numGatedPairs++;
            }
        }
        std::sort(gated[i].begin(), gated[i].end());
        if (!gated[i].empty())
        {
            out.detectionD2[i] = gated[i].front().first;
        }
    }

    // Ambiguity: a detection gating two candidates with similar cost is not associated.
    std::vector<int> proposal(nd, -1);
    for (std::size_t i = 0; i < nd; ++i)
    {
        if (gated[i].empty())
        {
            continue;
        }
        if (gated[i].size() >= 2 && gated[i][1].first - gated[i][0].first < opts.ambiguityMarginD2)
        {
            out.ambiguous[i] = true;
            continue;
        }
        proposal[i] = static_cast<int>(gated[i].front().second);
    }

    // Conflicts: a candidate proposed by several detections keeps the one with the smallest d^2.
    for (std::size_t j = 0; j < nc; ++j)
    {
        int best = -1;
        double bestD2 = kInf;
        for (std::size_t i = 0; i < nd; ++i)
        {
            if (proposal[i] == static_cast<int>(j) && out.detectionD2[i] < bestD2)
            {
                bestD2 = out.detectionD2[i];
                best = static_cast<int>(i);
            }
        }
        if (best < 0)
        {
            continue;
        }
        out.candidateDetection[j] = best;
        out.detectionCandidate[static_cast<std::size_t>(best)] = static_cast<int>(j);
        for (std::size_t i = 0; i < nd; ++i)
        {
            if (proposal[i] == static_cast<int>(j) && static_cast<int>(i) != best)
            {
                out.conflict[i] = true;
            }
        }
    }

    // Birth candidates: not gated by any candidate and not near a predicted tag centre (a missed association would
    // otherwise create a duplicate landmark for the same physical tag).
    for (std::size_t i = 0; i < nd; ++i)
    {
        if (!gated[i].empty() || out.ambiguous[i])
        {
            continue;
        }
        const Eigen::Vector2d ci = detections[i].centre();
        bool near = false;
        for (const TagCandidate& c : predictions.candidates)
        {
            const double scale = std::max(detections[i].sidePx, c.sidePx);
            if ((ci - c.centre).norm() < opts.nearCentreFactor * scale)
            {
                near = true;
                break;
            }
        }
        out.birthCandidate[i] = !near;
    }
    return out;
}

double jointNis(const TagPredictionSet& predictions, const std::vector<std::size_t>& matched,
                const Eigen::VectorXd& innovation, double sigmaPx)
{
    const Eigen::Index m = static_cast<Eigen::Index>(matched.size());
    if (m == 0)
    {
        return 0.0;
    }
    if (innovation.size() != 8 * m)
    {
        throw std::invalid_argument("jointNis: innovation size must be 8 x matched");
    }
    const Eigen::Index nu = static_cast<Eigen::Index>(predictions.unionIndices.size());
    Eigen::MatrixXd hStack = Eigen::MatrixXd::Zero(8 * m, nu);
    for (Eigen::Index k = 0; k < m; ++k)
    {
        const TagCandidate& c = predictions.candidates[matched[static_cast<std::size_t>(k)]];
        hStack.block<8, 6>(8 * k, 0) = c.pred.dEta;
        hStack.block<8, 6>(8 * k, c.unionOffset) = c.pred.dLm;
    }
    const Eigen::MatrixXd x = predictions.xiUnion.transpose().triangularView<Eigen::Lower>().solve(hStack.transpose());
    const Eigen::MatrixXd r = innovationSqrt(x, sigmaPx);   // R^T R = S (joint innovation covariance), never factorised
    if (r.size() == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::VectorXd q = r.transpose().triangularView<Eigen::Lower>().solve(innovation);
    return q.squaredNorm();
}

} // namespace slam
