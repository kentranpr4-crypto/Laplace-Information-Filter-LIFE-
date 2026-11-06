#include "SlamPoseTracker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <Eigen/Geometry>
#include "GaussianBase.h"
#include "CoreLinalg.h"
#include "GaussianInfo.h"
#include "MeasurementIndoorFlow.h"
#include "MeasurementTagBundle.h"
#include "Pose.hpp"
#include "funcmin.hpp"
#include "rotation.hpp"

namespace slam
{

namespace
{

using Clock = std::chrono::steady_clock;

double millisecondsSince(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Initial density: zero mean; velocity and rate sigmas from the configuration; start pose fixed to a small sigma.
GaussianInfo<double> initialDensity(const SlamPoseConfig& cfg)
{
    Eigen::VectorXd sigmas(SystemVisualNav::kDynamicDim);
    sigmas << cfg.initialVelocitySigma, cfg.initialVelocitySigma, cfg.initialVelocitySigma,
        cfg.initialAngularRateSigma, cfg.initialAngularRateSigma, cfg.initialAngularRateSigma,
        cfg.initialPoseSigma, cfg.initialPoseSigma, cfg.initialPoseSigma,
        cfg.initialPoseSigma, cfg.initialPoseSigma, cfg.initialPoseSigma;
    const Eigen::MatrixXd p = sigmas.cwiseProduct(sigmas).asDiagonal();
    return GaussianInfo<double>::fromMoment(Eigen::VectorXd::Zero(SystemVisualNav::kDynamicDim), p);
}

void validateConfig(const SlamPoseConfig& cfg)
{
    if (!(cfg.tagSideM > 0.0) || !std::isfinite(cfg.tagSideM))
    {
        throw std::invalid_argument("SlamPoseTracker: tag side length must be finite and positive");
    }
    if (!(cfg.pixelSigmaPx > 0.0) || !(cfg.gateNSigma > 0.0) || !(cfg.ambiguityMarginD2 >= 0.0))
    {
        throw std::invalid_argument("SlamPoseTracker: noise, gate and ambiguity options must be positive");
    }
    if (cfg.minObservationsForBirth < 1 || cfg.maxTags < 1 || cfg.minMatchesForUpdate < 1 || cfg.maxPendingAgeFrames < 0)
    {
        throw std::invalid_argument("SlamPoseTracker: invalid birth, capacity or update limits");
    }
    if ((cfg.birthPriorSigma.array() <= 0.0).any() || (cfg.nuPsd.array() <= 0.0).any())
    {
        throw std::invalid_argument("SlamPoseTracker: prior sigmas and process PSDs must be positive");
    }
    if (!(cfg.reidFloorPositionM > 0.0) || !(cfg.reidFloorAngleRad > 0.0) || cfg.archiveMaxEntries < 0
        || cfg.archiveMaxAgeFrames < 0 || !(cfg.duplicateCentreFactor >= 0.0) || !(cfg.duplicateMaxAngleRad >= 0.0))
    {
        throw std::invalid_argument("SlamPoseTracker: invalid re-identification or duplicate options");
    }
}

/// A MAP result that is not usable: a numerical failure, or a MaxIterations result the policy did not accept. A
/// no-information update (SkippedNoCurvature) is not a failure of the optimiser and is counted separately.
bool isMapFailure(const MeasurementUpdateResult& res)
{
    return res.map.status == FuncMinStatus::NumericalFailure || res.outcome == MapOutcome::SkippedNumericalFailure
           || res.outcome == MapOutcome::SkippedMaxIterations;
}

/// Distance between two world poses of a tag: centre distance plus side-scaled rotation angle [m].
double poseDistance(const Eigen::Matrix3d& R1, const Eigen::Vector3d& r1, const Eigen::Matrix3d& R2,
                    const Eigen::Vector3d& r2, double side)
{
    const Eigen::AngleAxisd aa(R1.transpose() * R2);
    return (r1 - r2).norm() + side * std::abs(aa.angle());
}

} // namespace

namespace
{

/// Pose difference used by both re-identification gates: [r_c - r_m ; rotation vector of R_m^T R_c].
Eigen::Matrix<double, 6, 1> archivePoseDelta(const Eigen::Matrix<double, 6, 1>& meanPose,
                                             const Eigen::Matrix<double, 6, 1>& candPose)
{
    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = candPose.head<3>() - meanPose.head<3>();
    const Eigen::AngleAxisd aa(rpy2rot(meanPose.tail<3>()).transpose() * rpy2rot(candPose.tail<3>()));
    delta.tail<3>() = aa.angle() * aa.axis();
    return delta;
}

/// diag(floorPos x3, floorAngle x3): the square-root of the inflation F added to the archived covariance.
Eigen::Matrix<double, 6, 1> floorDiagonal(double floorPositionM, double floorAngleRad, const char* who)
{
    if (!(floorPositionM > 0.0) || !(floorAngleRad > 0.0) || !std::isfinite(floorPositionM)
        || !std::isfinite(floorAngleRad))
    {
        throw std::invalid_argument(std::string(who) + ": floors must be finite and positive");
    }
    Eigen::Matrix<double, 6, 1> f;
    f << floorPositionM, floorPositionM, floorPositionM, floorAngleRad, floorAngleRad, floorAngleRad;
    return f;
}

/// Solves R^T y = v for the upper factor R of a Q-less QR. Returns false if R is singular or non-finite.
bool solveUpperTransposed(const Eigen::MatrixXd& r, const Eigen::VectorXd& v, Eigen::VectorXd& y)
{
    if (!r.allFinite() || !(r.diagonal().cwiseAbs().minCoeff() > 0.0))
    {
        return false;
    }
    y = r.transpose().triangularView<Eigen::Lower>().solve(v);
    return y.allFinite();
}

} // namespace

double tagArchiveDistanceSq(const Eigen::Matrix<double, 6, 1>& meanPose, const Eigen::Matrix<double, 6, 6>& sqrtCov,
                            const Eigen::Matrix<double, 6, 1>& candPose, double floorPositionM, double floorAngleRad)
{
    const Eigen::Matrix<double, 6, 1> floors = floorDiagonal(floorPositionM, floorAngleRad, "tagArchiveDistanceSq");
    const Eigen::Matrix<double, 6, 1> delta = archivePoseDelta(meanPose, candPose);
    // A = [S; diag(floors)] has A^T A = S^T S + F = P + F. Its Q-less QR gives R with R^T R = P + F, so
    // d^2 = delta^T (R^T R)^{-1} delta = ||R^{-T} delta||^2: no product P is formed and no factorisation of P.
    Eigen::MatrixXd stacked(12, 6);
    stacked.topRows<6>() = sqrtCov;
    stacked.bottomRows<6>() = floors.asDiagonal();
    const Eigen::MatrixXd r = core::upperFactor(stacked);
    Eigen::VectorXd y;
    if (!solveUpperTransposed(r, delta, y))
    {
        return std::numeric_limits<double>::infinity();
    }
    return y.squaredNorm();
}

double tagArchiveDistanceSqInfo(const Eigen::Matrix<double, 6, 1>& meanPose, const Eigen::Matrix<double, 6, 6>& sqrtInfo,
                                const Eigen::Matrix<double, 6, 1>& candPose, double floorPositionM, double floorAngleRad)
{
    const Eigen::Matrix<double, 6, 1> floors = floorDiagonal(floorPositionM, floorAngleRad, "tagArchiveDistanceSqInfo");
    const Eigen::Matrix<double, 6, 1> delta = archivePoseDelta(meanPose, candPose);
    // P = Xi^{-1} Xi^{-T} (Lambda = Xi^T Xi), so P + F = Xi^{-1} (I + Xi F Xi^T) Xi^{-T} and
    //   d^2 = v^T (B^T B)^{-1} v,  v = Xi delta,  B = [I; diag(floors) Xi^T]  (B^T B = I + Xi F Xi^T >= I).
    // With the Q-less QR B = Q_B R_B: d^2 = ||R_B^{-T} v||^2. Only products and triangular solves are used.
    const Eigen::VectorXd v = sqrtInfo * delta;
    Eigen::MatrixXd b(12, 6);
    b.topRows<6>() = Eigen::Matrix<double, 6, 6>::Identity();
    b.bottomRows<6>() = floors.asDiagonal() * sqrtInfo.transpose();
    const Eigen::MatrixXd r = core::upperFactor(b);
    Eigen::VectorXd y;
    if (!solveUpperTransposed(r, v, y))
    {
        return std::numeric_limits<double>::infinity();
    }
    return y.squaredNorm();
}

SlamPoseTracker::SlamPoseTracker(const Camera& camera, const SlamPoseConfig& cfg)
    : camera_(camera),
      cfg_(cfg),
      model_(std::make_shared<SystemVisualNav>(cfg.nuPsd)),
      detector_(cfg.detector)
{
    if (!camera_.isValid())
    {
        throw std::invalid_argument("SlamPoseTracker: camera is not valid");
    }
    validateConfig(cfg_);
    gate_ = GaussianBase::mahalanobisThresholdSq(cfg_.gateNSigma, 8.0);
    reidGate_ = GaussianBase::mahalanobisThresholdSq(cfg_.gateNSigma, 6.0);
    est_ = std::make_unique<SystemEstimator>(model_, initialDensity(cfg_), 0.0);
}

std::vector<Eigen::Matrix3d> SlamPoseTracker::blockSqrtCovs(const std::vector<Eigen::Index>& offsets) const
{
    std::vector<Eigen::Matrix3d> out;
    if (offsets.empty())
    {
        return out;
    }
    std::vector<Eigen::Index> idx;
    idx.reserve(3 * offsets.size());
    for (Eigen::Index o : offsets)
    {
        for (Eigen::Index i = 0; i < 3; ++i)
        {
            idx.push_back(o + i);
        }
    }
    const GaussianInfo<double> m = est_->state().marginal(idx);
    const Eigen::Index n = static_cast<Eigen::Index>(idx.size());
    // Xi^{-T} by a triangular solve (no covariance is formed as an inverse of the information matrix).
    const Eigen::MatrixXd xiInvT = m.sqrtInfo().transpose().triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(n, n));
    out.reserve(offsets.size());
    for (std::size_t k = 0; k < offsets.size(); ++k)
    {
        const Eigen::MatrixXd x = xiInvT.middleCols(3 * static_cast<Eigen::Index>(k), 3);
        const Eigen::MatrixXd r = core::upperFactor(x);   // R^T R = X^T X = P_kk
        out.push_back(r.topLeftCorner<3, 3>());
    }
    return out;
}

Eigen::Matrix3d SlamPoseTracker::positionSqrtCov() const
{
    return blockSqrtCovs({SystemVisualNav::kEtaOffset}).front();
}

Eigen::Matrix<double, 6, 6> SlamPoseTracker::landmarkSqrtInfo6(Eigen::Index offset) const
{
    std::vector<Eigen::Index> idx;
    idx.reserve(6);
    for (Eigen::Index i = 0; i < 6; ++i)
    {
        idx.push_back(offset + i);
    }
    // The marginal is formed by QR on the information factor; the archive keeps Xi itself (no covariance is inverted).
    const GaussianInfo<double> m = est_->state().marginal(idx);
    return Eigen::Matrix<double, 6, 6>(m.sqrtInfo());
}

void SlamPoseTracker::archiveCulled(const SlamPoseRecord& rec, int frameIndex)
{
    if (cfg_.archiveMaxEntries <= 0)
    {
        return;
    }
    SlamPoseArchiveEntry e;
    e.identity = rec.identity;
    e.qIndex = rec.qIndex;
    e.cornersLocal = rec.cornersLocal;
    const Eigen::Index off = model_->landmarkOffset(rec.id);
    e.mean = est_->mean().segment<6>(off);
    e.sqrtInfo = landmarkSqrtInfo6(off);
    e.lastSeenFrame = rec.lastSeenFrame;
    e.culledFrame = frameIndex;
    archive_.push_back(e);
    archive_.erase(std::remove_if(archive_.begin(), archive_.end(),
                                  [frameIndex, this](const SlamPoseArchiveEntry& a)
                                  { return frameIndex - a.culledFrame > cfg_.archiveMaxAgeFrames; }),
                   archive_.end());
    while (static_cast<int>(archive_.size()) > cfg_.archiveMaxEntries)
    {
        archive_.erase(archive_.begin());
    }
}

std::vector<Eigen::Index> SlamPoseTracker::selectCulls(const std::vector<SlamPoseRecord>& records, int frame,
                                                       const SlamPoseConfig& cfg)
{
    std::vector<Eigen::Index> ids;
    std::vector<bool> culled(records.size(), false);
    for (std::size_t i = 0; i < records.size(); ++i)
    {
        const SlamPoseRecord& r = records[i];
        const bool unconfirmedDead = !r.confirmed && r.misses >= cfg.unconfirmedMaxMisses;
        const bool confirmedDead = r.confirmed && r.misses >= cfg.confirmedMaxMisses;
        const bool behind = r.behindFrames >= cfg.maxBehindFrames;
        const bool unseen = frame - r.lastSeenFrame > cfg.maxUnseenFrames;
        if (unconfirmedDead || confirmedDead || behind || unseen)
        {
            ids.push_back(r.id);
            culled[i] = true;
        }
    }
    // Capacity: remove the records with the most misses (then the oldest sighting) until the map fits.
    const std::size_t remaining = records.size() - ids.size();
    if (remaining > static_cast<std::size_t>(cfg.maxTags))
    {
        std::vector<std::size_t> order;
        for (std::size_t i = 0; i < records.size(); ++i)
        {
            if (!culled[i])
            {
                order.push_back(i);
            }
        }
        std::sort(order.begin(), order.end(), [&records](std::size_t a, std::size_t b) {
            if (records[a].misses != records[b].misses)
            {
                return records[a].misses > records[b].misses;
            }
            return records[a].lastSeenFrame < records[b].lastSeenFrame;
        });
        const std::size_t excess = remaining - static_cast<std::size_t>(cfg.maxTags);
        for (std::size_t k = 0; k < excess && k < order.size(); ++k)
        {
            ids.push_back(records[order[k]].id);
        }
    }
    return ids;
}

SlamPoseFrameStats SlamPoseTracker::step(const cv::Mat& gray, int frameIndex, double t)
{
    if (!std::isfinite(t) || (started_ && t < est_->time()))
    {
        throw std::invalid_argument("SlamPoseTracker::step: time must be finite and non-decreasing");
    }
    started_ = true;
    const Clock::time_point tStart = Clock::now();
    SlamPoseFrameStats st;
    st.frame = frameIndex;
    st.time = t;
    const double sigma = cfg_.pixelSigmaPx;

    // 1. Detection and optical-flow tracks (ImageFlow must see every frame: it tracks between consecutive calls).
    Clock::time_point t0 = Clock::now();
    detections_ = detector_.detect(gray);
    st.detections = static_cast<int>(detections_.size());
    st.msDetect = millisecondsSince(t0);
    t0 = Clock::now();
    const std::vector<ImageFlowMatch> tracks = flow_.update(gray);
    const double msFlowTracking = millisecondsSince(t0);

    // 2. Delayed time update (etaPrev <- eta_{k-1}, eta <- propagated).
    t0 = Clock::now();
    if (t > est_->time())
    {
        model_->predictDelayed(*est_, t);
    }
    st.msPredict = millisecondsSince(t0);

    // 3. Flow measurement linking etaPrev (frame k-1) and eta (frame k): gated at the predicted state, then a MAP update.
    t0 = Clock::now();
    if (cfg_.useFlow && model_->hasDelayedPose() && !tracks.empty())
    {
        std::vector<IndoorFlowPixelMatch> pixels;
        pixels.reserve(tracks.size());
        for (const ImageFlowMatch& m : tracks)
        {
            IndoorFlowPixelMatch p;
            p.prevPx = Eigen::Vector2d(m.prev.x, m.prev.y);
            p.currPx = Eigen::Vector2d(m.curr.x, m.curr.y);
            pixels.push_back(p);
        }
        const IndoorFlowFrame ff = buildIndoorFlowFrame(camera_, *est_, pixels, cfg_.flow);
        st.flowInput = ff.input;
        st.flowValid = ff.valid;
        st.flowRotationModel = ff.model == IndoorFlowModel::Rotation;
        st.flowPairs = ff.gated;
        if (ff.valid)
        {
            const MeasurementIndoorFlow fm(t, ff.model, ff.pairs, ff.sigmaNorm, camera_.Rbc(), camera_.rCBb());
            const Eigen::VectorXd r = fm.whitenedResiduals(est_->mean(), nullptr);
            st.flowNis = r.squaredNorm();
            st.flowDof = static_cast<int>(r.size());
            FuncMinOptions fo;
            fo.gradientTolerance = cfg_.mapGradientTolerance;
            fo.functionTolerance = 1e-10;
            const GaussianInfo<double> before = est_->state();
            const MeasurementUpdateResult res = fm.process(*est_, fo);
            flowMapCounts_.add(res);
            st.flowMapStatus = static_cast<int>(res.map.status);
            st.flowMapOutcome = static_cast<int>(res.outcome);
            st.flowUpdateSkipped = res.updateSkipped;
            // The policy has already left the prediction in place for a skipped update; the revert below is the
            // safety net for any non-finite result. A no-information skip (SkippedNoCurvature) is not a MAP failure.
            const bool flowFailed = isMapFailure(res) || !est_->mean().allFinite() || !est_->state().sqrtInfo().allFinite();
            if (res.updateSkipped || flowFailed)
            {
                est_->setState(before);
                if (flowFailed)
                {
                    ++mapFailures_;
                }
            }
            else
            {
                st.flowUpdated = true;
            }
        }
    }
    st.msFlow = msFlowTracking + millisecondsSince(t0);

    // 4. Candidate prediction (occlusion-aware visibility) and geometric association.
    t0 = Clock::now();
    std::vector<TagPredictionInput> inputs;
    inputs.reserve(records_.size());
    for (std::size_t r = 0; r < records_.size(); ++r)
    {
        TagPredictionInput in;
        in.record = r;
        in.id = records_[r].id;
        in.cornersLocal = records_[r].cornersLocal;
        inputs.push_back(in);
    }
    TagPredictionOptions popt;
    popt.sigmaPx = sigma;
    popt.minDepth = cfg_.minDepth;
    popt.maxDepth = cfg_.maxDepth;
    popt.candidateMarginPx = cfg_.candidateMarginPx;
    popt.visibleMarginPx = cfg_.visibleMarginPx;
    popt.minVisibleSidePx = cfg_.minVisibleSidePx;
    popt.minFacingCos = cfg_.minFacingCos;
    predictions_ = predictTagCandidates(*est_, *model_, camera_, inputs, popt);
    st.msPredict += millisecondsSince(t0);
    st.candidates = static_cast<int>(predictions_.candidates.size());
    for (const TagCandidate& c : predictions_.candidates)
    {
        st.visibleCandidates += c.visible ? 1 : 0;
        st.occluded += c.occluded ? 1 : 0;
    }

    t0 = Clock::now();
    TagAssociationOptions aopt;
    aopt.gateThresholdSq = gate_;
    aopt.ambiguityMarginD2 = cfg_.ambiguityMarginD2;
    aopt.nearCentreFactor = cfg_.nearCentreFactor;
    association_ = associateTagDetections(predictions_, detections_, aopt);
    st.msAssociate = millisecondsSince(t0);
    for (std::size_t i = 0; i < detections_.size(); ++i)
    {
        st.ambiguous += association_.ambiguous[i] ? 1 : 0;
        st.conflicts += association_.conflict[i] ? 1 : 0;
    }

    // 5. Measurement update with the associated tags (eta and the matched landmarks only).
    t0 = Clock::now();
    std::vector<std::size_t> matchedCand;
    std::vector<TagObservation> obs;
    Eigen::VectorXd innov;
    for (std::size_t i = 0; i < detections_.size(); ++i)
    {
        const int j = association_.detectionCandidate[i];
        if (j < 0)
        {
            continue;
        }
        const TagCandidate& c = predictions_.candidates[static_cast<std::size_t>(j)];
        matchedCand.push_back(static_cast<std::size_t>(j));
        TagObservation o;
        o.landmarkId = c.id;
        o.landmarkOffset = c.offset;
        o.cornersLocal = c.cornersLocal;
        o.pixels = detections_[i].cornerVector();
        o.sigmaPx = sigma;
        obs.push_back(o);
    }
    st.matched = static_cast<int>(obs.size());
    if (st.matched >= cfg_.minMatchesForUpdate)
    {
        const Eigen::Index m8 = 8 * static_cast<Eigen::Index>(matchedCand.size());
        innov.resize(m8);
        for (std::size_t k = 0; k < matchedCand.size(); ++k)
        {
            const TagCandidate& c = predictions_.candidates[matchedCand[k]];
            innov.segment<8>(8 * static_cast<Eigen::Index>(k)) = obs[k].pixels - c.pred.uv;
        }
        st.nis = jointNis(predictions_, matchedCand, innov, sigma);
        st.nisDof = static_cast<int>(m8);

        const MeasurementTagBundle meas(t, camera_, obs, SystemVisualNav::kEtaOffset);
        FuncMinOptions mopt;
        mopt.gradientTolerance = cfg_.mapGradientTolerance;
        mopt.functionTolerance = 1e-10;
        const GaussianInfo<double> before = est_->state();
        const MeasurementUpdateResult res = meas.process(*est_, mopt);
        st.mapIterations = res.map.iterations;
        st.mapStatus = static_cast<int>(res.map.status);
        st.mapOutcome = static_cast<int>(res.outcome);
        st.updateSkipped = res.updateSkipped;
        st.newtonNorm = res.newtonNorm;
        st.precisionExact = res.precisionIsExact;
        tagMapCounts_.add(res);
        // A skipped update (MAP policy) is kept at the prediction; a failed MAP is counted in mapFailures. Either way the
        // frame is a prediction-only frame. The revert is also the safety net for any non-finite result.
        const bool tagFailed = isMapFailure(res) || !est_->mean().allFinite() || !est_->state().sqrtInfo().allFinite();
        if (res.updateSkipped || tagFailed)
        {
            est_->setState(before);
            if (tagFailed)
            {
                ++mapFailures_;
            }
        }
        else
        {
            st.updated = true;
        }
    }
    st.msUpdate = millisecondsSince(t0);

    // 6. Records from the pre-update predictions: hits, visible misses (ambiguous or conflicting gates are neither),
    //    and the behind-camera counters.
    t0 = Clock::now();
    for (SlamPoseRecord& r : records_)
    {
        r.associatedThisFrame = false;
    }
    auto recordOf = [this](Eigen::Index id) -> SlamPoseRecord*
    {
        for (SlamPoseRecord& r : records_)
        {
            if (r.id == id)
            {
                return &r;
            }
        }
        return nullptr;
    };
    for (std::size_t j = 0; j < predictions_.candidates.size(); ++j)
    {
        const TagCandidate& c = predictions_.candidates[j];
        SlamPoseRecord* r = recordOf(c.id);
        if (r == nullptr)
        {
            continue;
        }
        const int di = association_.candidateDetection[j];
        if (di >= 0)
        {
            r->associatedThisFrame = true;
            r->hits += 1;
            r->misses = 0;
            r->lastSeenFrame = frameIndex;
            r->lastPixel = detections_[static_cast<std::size_t>(di)].centre();
            r->confirmed = r->confirmed || r->hits >= cfg_.confirmHits;
        }
        else if (c.visible && !association_.candidateGated[j])
        {
            r->misses += 1;
        }
    }
    {
        const Eigen::VectorXd mu = est_->mean();
        const Eigen::Vector3d rBNn = mu.segment<3>(SystemVisualNav::kEtaOffset);
        const Eigen::Vector3d thetaNb = mu.segment<3>(SystemVisualNav::kEtaOffset + 3);
        for (SlamPoseRecord& r : records_)
        {
            const Eigen::Index off = model_->landmarkOffset(r.id);
            const Eigen::Vector3d rL = mu.segment<3>(off);
            const double depth = rigid::worldToCamera(rL, rBNn, thetaNb, camera_.Rbc(), camera_.rCBb()).z();
            r.behindFrames = (depth < cfg_.minDepth) ? r.behindFrames + 1 : 0;
        }
    }

    // 7. Culling. A culled tag is archived (identity and last density) before it is removed from the state.
    const std::vector<Eigen::Index> cull = selectCulls(records_, frameIndex, cfg_);
    for (Eigen::Index id : cull)
    {
        const auto it = std::find_if(records_.begin(), records_.end(), [id](const SlamPoseRecord& r) { return r.id == id; });
        if (it != records_.end() && cfg_.reidentify)
        {
            archiveCulled(*it, frameIndex);
            st.archived += 1;
        }
        model_->removeLandmark(*est_, id);
        records_.erase(std::remove_if(records_.begin(), records_.end(),
                                      [id](const SlamPoseRecord& r) { return r.id == id; }),
                       records_.end());
    }
    st.culled = static_cast<int>(cull.size());

    // 8. Births: detections that no tag gates, kept pending until seen in minObservationsForBirth frames.
    {
        const Eigen::VectorXd mu = est_->mean();
        for (std::size_t i = 0; i < detections_.size(); ++i)
        {
            if (!association_.birthCandidate[i])
            {
                continue;
            }
            const tags::TagDetection& det = detections_[i];
            const std::vector<TagPoseHypothesis> hyps =
                refinedTagPoseHypotheses(camera_, det.cornerVector(), tags::canonicalCorners(cfg_.tagSideM), sigma);
            if (hyps.empty())
            {
                continue;
            }
            SlamPosePending::Observation ob;
            ob.frame = frameIndex;
            ob.pixels = det.cornerVector();
            ob.hyps = hyps;
            for (const TagPoseHypothesis& h : hyps)
            {
                Eigen::Matrix3d Rnt;
                Eigen::Vector3d rnt;
                worldTagPose(camera_, mu, SystemVisualNav::kEtaOffset, h.Rct, h.tct, Rnt, rnt);
                ob.Rnt.push_back(Rnt);
                ob.rnt.push_back(rnt);
            }
            // Match to a pending tag that has not been updated in this frame.
            SlamPosePending* match = nullptr;
            for (SlamPosePending& p : pending_)
            {
                const double gap = frameIndex - p.lastFrame;
                const double thr = std::max(cfg_.pendingMatchPx, 1.2 * det.sidePx);
                if (p.lastFrame < frameIndex && gap <= cfg_.maxPendingAgeFrames && (p.lastCentre - det.centre()).norm() <= thr)
                {
                    match = &p;
                    break;
                }
            }
            if (match == nullptr)
            {
                pending_.emplace_back();
                match = &pending_.back();
            }
            match->obs.push_back(ob);
            match->lastFrame = frameIndex;
            match->lastCentre = det.centre();
            match->lastSidePx = det.sidePx;
        }

        // Birth of ready pending tags (only those updated in this frame).
        for (auto it = pending_.begin(); it != pending_.end();)
        {
            BirthOutcome outcome = BirthOutcome::Wait;
            if (it->lastFrame == frameIndex && static_cast<int>(it->obs.size()) >= cfg_.minObservationsForBirth
                && static_cast<int>(records_.size()) < cfg_.maxTags)
            {
                outcome = tryBirth(*it, st);
            }
            if (outcome == BirthOutcome::Born)
            {
                st.births += 1;
                it = pending_.erase(it);
            }
            else if (outcome == BirthOutcome::Duplicate)
            {
                st.duplicatesSuppressed += 1;
                it = pending_.erase(it);
            }
            else
            {
                ++it;
            }
        }
        // Age out pending tags that have not been seen recently.
        pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                      [frameIndex, this](const SlamPosePending& p) {
                                          return frameIndex - p.lastFrame > cfg_.maxPendingAgeFrames;
                                      }),
                       pending_.end());
    }
    st.pending = static_cast<int>(pending_.size());
    st.msBirthCull = millisecondsSince(t0);

    // 9. Trajectory: camera position r_CN^n = r_BN^n + R_NB r_CB^B.
    {
        const Eigen::VectorXd mu = est_->mean();
        trajectory_.push_back(rigid::cameraOriginInWorld(mu.segment<3>(SystemVisualNav::kEtaOffset + 3),
                                                         mu.segment<3>(SystemVisualNav::kEtaOffset), camera_.rCBb()));
    }
    st.tags = static_cast<int>(records_.size());
    st.msTotal = millisecondsSince(tStart);
    return st;
}

SlamPoseTracker::BirthOutcome SlamPoseTracker::tryBirth(SlamPosePending& p, SlamPoseFrameStats& st)
{
    const SlamPosePending::Observation& last = p.obs.back();
    const double side = cfg_.tagSideM;
    // The IPPE two-fold ambiguity is resolved by consistency across frames: each hypothesis of the latest observation
    // is scored by its distance to the best-matching hypothesis of every earlier observation (sum over those).
    std::vector<double> score(last.hyps.size(), 0.0);
    for (std::size_t h = 0; h < last.hyps.size(); ++h)
    {
        for (std::size_t e = 0; e + 1 < p.obs.size(); ++e)
        {
            double best = std::numeric_limits<double>::infinity();
            for (std::size_t h2 = 0; h2 < p.obs[e].hyps.size(); ++h2)
            {
                best = std::min(best, poseDistance(last.Rnt[h], last.rnt[h], p.obs[e].Rnt[h2], p.obs[e].rnt[h2], side));
            }
            score[h] += best;
        }
    }
    std::size_t bestH = 0;
    for (std::size_t h = 1; h < score.size(); ++h)
    {
        if (score[h] < score[bestH])
        {
            bestH = h;
        }
    }
    // Ambiguity needs camera motion: birth waits for minBaselineFrames, unless the choice is already decisive.
    double second = std::numeric_limits<double>::infinity();
    for (std::size_t h = 0; h < score.size(); ++h)
    {
        if (h != bestH)
        {
            second = std::min(second, score[h]);
        }
    }
    const bool decisive = last.hyps.size() == 1 || (second - score[bestH]) > cfg_.decisiveMarginM;
    const int span = last.frame - p.obs.front().frame;
    if (!((span >= cfg_.minBaselineFrames) || (decisive && span >= 1)))
    {
        return BirthOutcome::Wait;
    }

    const TagPoseHypothesis& hyp = last.hyps[bestH];
    const Eigen::VectorXd mu = est_->mean();
    Eigen::Matrix3d Rnt;
    Eigen::Vector3d rnt;
    worldTagPose(camera_, mu, SystemVisualNav::kEtaOffset, hyp.Rct, hyp.tct, Rnt, rnt);
    const double depth = rigid::worldToCamera(rnt, mu.segment<3>(SystemVisualNav::kEtaOffset),
                                              mu.segment<3>(SystemVisualNav::kEtaOffset + 3), camera_.Rbc(),
                                              camera_.rCBb()).z();
    if (!(depth > cfg_.minDepth))
    {
        return BirthOutcome::Wait;
    }

    // (a) Duplicate test: an active tag with a similar normal within duplicateCentreFactor * side of the new centre is the
    //     same physical tag. The pending observation is dropped (no second landmark for one tag).
    for (const SlamPoseRecord& r : records_)
    {
        const Eigen::Index off = model_->landmarkOffset(r.id);
        const Eigen::Vector3d rL = mu.segment<3>(off);
        const Eigen::Matrix3d RL = rpy2rot(mu.segment<3>(off + 3));
        const double angle = std::acos(std::clamp(std::abs(RL.col(2).dot(Rnt.col(2))), 0.0, 1.0));
        if ((rL - rnt).norm() < cfg_.duplicateCentreFactor * side && angle < cfg_.duplicateMaxAngleRad)
        {
            return BirthOutcome::Duplicate;
        }
    }

    const Eigen::Matrix<double, 4, 3> canonical = tags::canonicalCorners(side);
    const TagBirthLinearisation lin = linearisedTagBirth(camera_, mu, SystemVisualNav::kEtaOffset, hyp, canonical,
                                                         cfg_.birthPriorSigma);
    if (!lin.G.allFinite() || !lin.c.allFinite() || !lin.lS.allFinite())
    {
        return BirthOutcome::Wait;   // numerically unusable birth (non-finite linearisation): candidate dropped, state unchanged
    }

    // (b) Re-identification: the best of the four in-plane orientations against the archived tags (chi2 gate, dof 6).
    int reIdx = -1;
    double bestD2 = reidGate_;
    if (cfg_.reidentify)
    {
        for (std::size_t a = 0; a < archive_.size(); ++a)
        {
            for (int q = 0; q < 4; ++q)
            {
                const Eigen::Matrix3d Rl = Rnt * tags::inPlaneRotation(q).transpose();
                Eigen::Matrix<double, 6, 1> cand;
                cand.head<3>() = rnt;
                cand.tail<3>() = rot2rpy(Rl);
                const double d2 = tagArchiveDistanceSqInfo(archive_[a].mean, archive_[a].sqrtInfo, cand,
                                                           cfg_.reidFloorPositionM, cfg_.reidFloorAngleRad);
                if (d2 < bestD2)
                {
                    bestD2 = d2;
                    reIdx = static_cast<int>(a);
                }
            }
        }
    }
    int identity = nextIdentity_;
    bool reattached = false;
    if (reIdx >= 0)
    {
        identity = archive_[static_cast<std::size_t>(reIdx)].identity;
        archive_.erase(archive_.begin() + reIdx);
        reattached = true;
        st.reidentified += 1;
    }
    else
    {
        nextIdentity_ += 1;
        st.newIdentities += 1;
    }

    const Eigen::Index id = model_->addLandmarkLinear(*est_, SystemVisualNav::LandmarkType::Pose, lin.G, lin.c, lin.lS);

    SlamPoseRecord rec;
    rec.id = id;
    rec.identity = identity;
    rec.reidentifications = reattached ? 1 : 0;
    rec.qIndex = lin.qIndex;
    rec.cornersLocal = lin.cornersLocal;
    rec.birthFrame = last.frame;
    rec.lastSeenFrame = last.frame;
    rec.hits = static_cast<int>(p.obs.size());
    rec.misses = 0;
    rec.confirmed = rec.hits >= cfg_.confirmHits;
    rec.lastPixel = Eigen::Vector2d(0.25 * (last.pixels(0) + last.pixels(2) + last.pixels(4) + last.pixels(6)),
                                    0.25 * (last.pixels(1) + last.pixels(3) + last.pixels(5) + last.pixels(7)));
    records_.push_back(rec);
    return BirthOutcome::Born;
}

} // namespace slam
