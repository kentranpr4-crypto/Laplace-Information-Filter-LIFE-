#include "SlamTrackAssociation.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <Eigen/LU>
#include "GaussianBase.h"
#include "MeasurementPointBundle.h"

namespace slam
{

AssociationResult associateTracks(const PredictionSet& predictions, const std::vector<PointFeature>& tracks,
                                  const std::vector<int>& trackIds, const std::vector<bool>& described,
                                  const SlamLandmarkMap& map, const Camera& camera, const SlamConfig& cfg)
{
    if (trackIds.size() != tracks.size() || described.size() != tracks.size())
    {
        throw std::invalid_argument("associateTracks: track, id and descriptor lists must have the same size");
    }
    const double chi2Gate = GaussianBase::mahalanobisThresholdSq(cfg.gateNSigma, 2.0);
    const std::size_t nPred = predictions.predictions.size();
    const std::size_t nTrack = tracks.size();

    AssociationResult out;
    out.featureUsed.assign(nTrack, false);
    out.predictionGated.assign(nPred, false);

    // Track id -> index in this frame, and track id -> owning landmark (from the records).
    std::unordered_map<int, std::size_t> indexOfTrack;
    indexOfTrack.reserve(nTrack);
    for (std::size_t i = 0; i < nTrack; ++i)
    {
        indexOfTrack[trackIds[i]] = i;
    }
    std::unordered_map<int, Eigen::Index> ownerOfTrack;
    for (const SlamLandmarkRecord& r : map.records())
    {
        if (r.trackId >= 0)
        {
            ownerOfTrack[r.trackId] = r.id;
        }
    }

    struct Candidate
    {
        std::size_t track;
        double d2;
        int hamming;
    };
    std::vector<int> choice(nPred, -1);
    std::vector<std::vector<std::size_t>> claimants(nTrack);

    for (std::size_t p = 0; p < nPred; ++p)
    {
        const LandmarkPrediction& pred = predictions.predictions[p];
        if (!pred.visible)
        {
            continue;
        }
        ++out.numVisible;
        const SlamLandmarkRecord& rec = map.at(pred.id);
        auto gateD2 = [&](std::size_t f) -> double
        {
            const Eigen::Vector2d e = tracks[f].pixel - pred.pixel;
            return gateDistanceSquared(pred, e);   // square-root form, no covariance inversion
        };

        // 1. Identity: the landmark's own track, if alive and inside the gate.
        if (rec.trackId >= 0)
        {
            const auto it = indexOfTrack.find(rec.trackId);
            if (it != indexOfTrack.end() && gateD2(it->second) <= chi2Gate)
            {
                out.predictionGated[p] = true;
                choice[p] = static_cast<int>(it->second);
                claimants[it->second].push_back(p);
                continue;
            }
        }

        // 2. Re-acquisition: free tracks (not owned by another landmark) inside the gate with a close descriptor.
        std::vector<Candidate> cands;
        for (std::size_t f = 0; f < nTrack; ++f)
        {
            if (!described[f])
            {
                continue;
            }
            const auto owner = ownerOfTrack.find(trackIds[f]);
            if (owner != ownerOfTrack.end() && owner->second != rec.id)
            {
                continue;
            }
            const double d2 = gateD2(f);
            if (!(d2 <= chi2Gate))
            {
                continue;
            }
            const int ham = hammingDistance(rec.descriptor, tracks[f].descriptor);
            if (ham > cfg.maxHamming)
            {
                continue;
            }
            cands.push_back({f, d2, ham});
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
        choice[p] = static_cast<int>(cands[0].track);
        claimants[cands[0].track].push_back(p);
    }

    // A track claimed by several landmarks is ambiguous for all of them.
    for (std::size_t f = 0; f < nTrack; ++f)
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

    // Accepted pairs (prediction order) and the innovation of the matched set.
    std::vector<std::size_t> matchedCols;
    for (std::size_t p = 0; p < nPred; ++p)
    {
        if (choice[p] < 0)
        {
            continue;
        }
        const LandmarkPrediction& pred = predictions.predictions[p];
        const PointFeature& track = tracks[static_cast<std::size_t>(choice[p])];
        MeasurementPointBundle::Observation o;
        o.landmarkId = pred.id;
        o.landmarkOffset = pred.offset;
        o.pixel = track.pixel;
        o.sigmaPx = cfg.pixelNoise.sigma(camera, track.pixel);
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
            const PointFeature& track = tracks[static_cast<std::size_t>(out.matchedFeature[static_cast<std::size_t>(k)])];
            out.innovation.segment<2>(2 * k) = track.pixel - pred.pixel;
            const double sig = out.observations[static_cast<std::size_t>(k)].sigmaPx;
            out.innovationCov.block<2, 2>(2 * k, 2 * k) = Eigen::Matrix2d::Identity() * (sig * sig);
        }
        out.innovationCov += zm.transpose() * zm;
    }
    return out;
}

} // namespace slam
