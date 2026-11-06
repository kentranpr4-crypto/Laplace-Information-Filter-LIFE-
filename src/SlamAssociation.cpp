#include "SlamAssociation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include "CoreLinalg.h"
#include "GaussianBase.h"
#include "Pose.hpp"

namespace slam
{

double gateDistanceSquared(const LandmarkPrediction& pred, const Eigen::Vector2d& e)
{
    Eigen::Matrix2d rp = pred.pixelSqrtCov;
    if (rp.isZero(0.0) && !pred.pixelCov.isZero(0.0))
    {
        // Factorise (never invert) the 2 x 2 state part; LDLT-free Cholesky, pixelCov is PSD by construction.
        Eigen::LLT<Eigen::Matrix2d> llt(pred.pixelCov);
        if (llt.info() == Eigen::Success)
        {
            rp = llt.matrixU();
        }
        else
        {
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(pred.pixelCov);
            rp = eig.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() * eig.eigenvectors().transpose();
        }
    }
    Eigen::MatrixXd stacked(4, 2);
    stacked.topRows(2) = rp;
    stacked.bottomRows(2) = pred.sigmaPx * Eigen::Matrix2d::Identity();
    const Eigen::MatrixXd rs = core::upperFactor<double>(stacked);   // 2 x 2, rs^T rs = S
    const Eigen::Vector2d w = rs.topLeftCorner(2, 2).transpose().triangularView<Eigen::Lower>().solve(e);
    return w.squaredNorm();
}

PredictionSet predictLandmarks(const SystemEstimator& est, const SystemVisualNav& model, const Camera& camera,
                               const SlamConfig& cfg)
{
    if (est.dim() != model.expectedDim())
    {
        throw std::invalid_argument("predictLandmarks: estimator state does not match the model layout");
    }
    const Eigen::VectorXd mu = est.mean();
    const Eigen::Index n = est.dim();
    const std::vector<SystemVisualNav::Landmark> lms = model.landmarks();
    const Eigen::Index nl = static_cast<Eigen::Index>(lms.size());
    const Eigen::Vector3d rBNn = mu.segment<3>(SystemVisualNav::kEtaOffset);
    const Eigen::Vector3d thetaNb = mu.segment<3>(SystemVisualNav::kEtaOffset + 3);

    PredictionSet out;
    out.predictions.resize(static_cast<std::size_t>(nl));
    Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(2 * nl, n);
    for (Eigen::Index j = 0; j < nl; ++j)
    {
        const SystemVisualNav::Landmark& lm = lms[static_cast<std::size_t>(j)];
        const PointPixelJacobian pj =
            pointPixelJacobian(camera, rBNn, thetaNb, mu.segment<3>(lm.offset), cfg.minDepth);
        LandmarkPrediction& p = out.predictions[static_cast<std::size_t>(j)];
        p.id = lm.id;
        p.offset = lm.offset;
        p.depth = pj.pc.z();
        if (pj.valid)
        {
            p.pixel = pj.uv;
            p.sigmaPx = cfg.pixelNoise.sigma(camera, pj.uv);
            p.visible = p.depth <= cfg.maxDepth && camera.isVectorWithinFOV(pj.pc, cfg.visibilityMarginPx);
            jac.block<2, 6>(2 * j, SystemVisualNav::kEtaOffset) = pj.dUvdEta;
            jac.block<2, 3>(2 * j, lm.offset) = pj.dUvdLm;
        }
    }

    // Z = Xi^{-T} J^T, so that Z^T Z = J Sigma J^T with Sigma = Xi^{-1} Xi^{-T} = (Xi^T Xi)^{-1}.
    if (nl > 0)
    {
        out.whitenedProjection = est.state().sqrtInfo().transpose().triangularView<Eigen::Lower>().solve(jac.transpose());
    }
    else
    {
        out.whitenedProjection = Eigen::MatrixXd::Zero(n, 0);
    }
    for (Eigen::Index j = 0; j < nl; ++j)
    {
        const Eigen::MatrixXd zj = out.whitenedProjection.middleCols(2 * j, 2);
        out.predictions[static_cast<std::size_t>(j)].pixelCov = zj.transpose() * zj;
        out.predictions[static_cast<std::size_t>(j)].pixelSqrtCov = core::upperFactor<double>(zj);
    }
    return out;
}

AssociationResult associateFeatures(const PredictionSet& predictions, const std::vector<PointFeature>& features,
                                    const SlamLandmarkMap& map, const Camera& camera, const SlamConfig& cfg)
{
    const double chi2Gate = GaussianBase::mahalanobisThresholdSq(cfg.gateNSigma, 2.0);
    const std::size_t nPred = predictions.predictions.size();
    const std::size_t nFeat = features.size();

    AssociationResult out;
    out.featureUsed.assign(nFeat, false);
    out.predictionGated.assign(nPred, false);

    struct Candidate
    {
        int feature;
        double d2;
        int hamming;
    };
    std::vector<int> choice(nPred, -1);
    std::vector<std::vector<std::size_t>> claimants(nFeat);

    for (std::size_t p = 0; p < nPred; ++p)
    {
        const LandmarkPrediction& pred = predictions.predictions[p];
        if (!pred.visible)
        {
            continue;
        }
        ++out.numVisible;
        const SlamLandmarkRecord& rec = map.at(pred.id);

        std::vector<Candidate> cands;
        for (std::size_t f = 0; f < nFeat; ++f)
        {
            const Eigen::Vector2d e = features[f].pixel - pred.pixel;
            const double d2 = gateDistanceSquared(pred, e);
            if (!(d2 <= chi2Gate))
            {
                continue;
            }
            const int ham = hammingDistance(rec.descriptor, features[f].descriptor);
            if (ham > cfg.maxHamming)
            {
                continue;
            }
            cands.push_back({static_cast<int>(f), d2, ham});
        }
        if (cands.empty())
        {
            ++out.numNoCandidate;
            continue;
        }
        out.predictionGated[p] = true;
        std::sort(cands.begin(), cands.end(), [](const Candidate& a, const Candidate& b)
                  { return a.hamming != b.hamming ? a.hamming < b.hamming : a.d2 < b.d2; });
        if (cands.size() >= 2 && cands[1].hamming - cands[0].hamming < cfg.ambiguityMargin)
        {
            ++out.numAmbiguous;
            continue;
        }
        choice[p] = cands[0].feature;
        claimants[static_cast<std::size_t>(cands[0].feature)].push_back(p);
    }

    // A feature claimed by several landmarks is ambiguous for all of them.
    for (std::size_t f = 0; f < nFeat; ++f)
    {
        if (claimants[f].size() > 1)
        {
            for (std::size_t p : claimants[f])
            {
                choice[p] = -1;
                ++out.numConflict;
            }
        }
    }

    // Assemble the accepted pairs and the innovation of the matched set.
    std::vector<std::size_t> matchedCols;
    for (std::size_t p = 0; p < nPred; ++p)
    {
        if (choice[p] < 0)
        {
            continue;
        }
        const LandmarkPrediction& pred = predictions.predictions[p];
        const PointFeature& feat = features[static_cast<std::size_t>(choice[p])];
        MeasurementPointBundle::Observation o;
        o.landmarkId = pred.id;
        o.landmarkOffset = pred.offset;
        o.pixel = feat.pixel;
        o.sigmaPx = cfg.pixelNoise.sigma(camera, feat.pixel);
        out.observations.push_back(o);
        out.matchedIds.push_back(pred.id);
        out.matchedPrediction.push_back(p);
        out.matchedFeature.push_back(choice[p]);
        out.featureUsed[static_cast<std::size_t>(choice[p])] = true;
        matchedCols.push_back(p);
    }

    const Eigen::Index m = static_cast<Eigen::Index>(out.observations.size());
    out.innovation.resize(2 * m);
    out.innovationCov = Eigen::MatrixXd::Zero(2 * m, 2 * m);
    if (m > 0)
    {
        const Eigen::Index n = predictions.whitenedProjection.rows();
        Eigen::MatrixXd zm(n, 2 * m);
        for (Eigen::Index k = 0; k < m; ++k)
        {
            const std::size_t p = matchedCols[static_cast<std::size_t>(k)];
            zm.middleCols(2 * k, 2) = predictions.whitenedProjection.middleCols(2 * static_cast<Eigen::Index>(p), 2);
            const LandmarkPrediction& pred = predictions.predictions[p];
            const PointFeature& feat = features[static_cast<std::size_t>(out.matchedFeature[static_cast<std::size_t>(k)])];
            out.innovation.segment<2>(2 * k) = feat.pixel - pred.pixel;
            out.innovationCov.block<2, 2>(2 * k, 2 * k) = Eigen::Matrix2d::Identity() * (out.observations[static_cast<std::size_t>(k)].sigmaPx * out.observations[static_cast<std::size_t>(k)].sigmaPx);
        }
        out.innovationCov += zm.transpose() * zm;
    }
    return out;
}

} // namespace slam
