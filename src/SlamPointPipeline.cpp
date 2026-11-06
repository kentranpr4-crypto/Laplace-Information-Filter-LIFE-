#include "SlamPointPipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include "CoreLinalg.h"
#include "GaussianInfo.h"
#include "MeasurementPointBundle.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include "MeasurementIndoorFlow.h"
#include "SlamInitialisation.h"
#include "S6KeyframeFlow.h"
#include "SlamTrackAssociation.h"
#include "S6SpeedPrior.h"
#include "SlamTrackTriangulation.h"
#include "funcmin.hpp"

namespace slam
{

namespace
{

using Clock = std::chrono::steady_clock;

double millisecondsSince(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Initial density: zero mean, velocity and rate sigmas from the configuration, start pose fixed to a small sigma.
/// The velocity direction is seeded from the data at the first epipolar frame (SlamPointPipeline::seedVelocityFromEssential):
/// a fixed heading would bias the MAP into a wrong basin (observed: forward seed + yaw absorbed the lateral walk).
GaussianInfo<double> initialDensity(const SlamConfig& cfg)
{
    Eigen::VectorXd sigmas(SystemVisualNav::kDynamicDim);
    sigmas << cfg.initialVelocitySigma, cfg.initialVelocitySigma, cfg.initialVelocitySigma,
        cfg.initialAngularRateSigma, cfg.initialAngularRateSigma, cfg.initialAngularRateSigma,
        cfg.initialPoseSigma, cfg.initialPoseSigma, cfg.initialPoseSigma,
        cfg.initialPoseSigma, cfg.initialPoseSigma, cfg.initialPoseSigma;
    return GaussianInfo<double>::fromMoment(Eigen::VectorXd::Zero(SystemVisualNav::kDynamicDim),
                                            sigmas.cwiseProduct(sigmas).asDiagonal());
}

void validateConfig(const SlamConfig& cfg)
{
    if (!(cfg.depthPriorMean > 0.0) || !(cfg.depthPriorSigma > 0.0))
    {
        throw std::invalid_argument("SlamPointPipeline: depth prior must be positive");
    }
    if (!(cfg.gateNSigma > 0.0) || cfg.maxHamming < 0 || cfg.maxHamming > 256 || cfg.ambiguityMargin < 0)
    {
        throw std::invalid_argument("SlamPointPipeline: invalid association options");
    }
    if (!(cfg.initialVelocitySigma > 0.0) || !(cfg.initialAngularRateSigma > 0.0) || !(cfg.initialPoseSigma > 0.0))
    {
        throw std::invalid_argument("SlamPointPipeline: initial sigmas must be positive");
    }
    if (cfg.minMatchesForUpdate < 1 || cfg.maxBirthsPerFrame < 0)
    {
        throw std::invalid_argument("SlamPointPipeline: invalid update or birth limits");
    }
    if (!(cfg.keyframeParallaxPx >= 0.0) || !std::isfinite(cfg.keyframeParallaxPx) || cfg.keyframeMinTracks < 1
        || cfg.keyframeMaxAge < 1)
    {
        throw std::invalid_argument("SlamPointPipeline: invalid keyframe options");
    }
    if (cfg.useSpeedPrior && (!(cfg.speedPriorMps > 0.0) || !(cfg.speedPriorSigmaMps > 0.0)))
    {
        throw std::invalid_argument("SlamPointPipeline: speed prior must be positive");
    }
    if (!(cfg.birthMinAngleDeg > 0.0) || !(cfg.birthMaxReprojPx > 0.0) || !(cfg.birthMinDepth > 0.0)
        || !(cfg.birthMaxDepth > cfg.birthMinDepth))
    {
        throw std::invalid_argument("SlamPointPipeline: invalid delayed-birth gates");
    }
}

/// Reasons of a rejected delayed birth (first failing gate).
enum class BirthGate
{
    Accepted,
    Invalid,     ///< degenerate geometry (singular triangulation or noise factor)
    Behind,      ///< triangulated point behind a camera (depth <= 0 in either camera)
    Angle,       ///< triangulation angle below birthMinAngleDeg
    Depth,       ///< depth outside [birthMinDepth, birthMaxDepth] in either camera
    Misfit       ///< normalised-ray misfit above birthMaxReprojPx
};

/// Delayed-birth acceptance: a finite triangulation in front of both cameras, enough parallax and a consistent misfit.
BirthGate birthGate(const TwoViewBirth& b, const SlamConfig& cfg)
{
    if (!b.valid || !b.tri.valid)
    {
        if (b.tri.L.isZero(0.0))
        {
            return BirthGate::Invalid;   // no triangulation was computed (non-finite rays)
        }
        return (b.tri.depthA < 0.0 || b.tri.depthB < 0.0) ? BirthGate::Behind : BirthGate::Invalid;
    }
    if (b.tri.angleDeg < cfg.birthMinAngleDeg)
    {
        return BirthGate::Angle;
    }
    if (b.tri.depthA < cfg.birthMinDepth || b.tri.depthA > cfg.birthMaxDepth || b.tri.depthB < cfg.birthMinDepth
        || b.tri.depthB > cfg.birthMaxDepth)
    {
        return BirthGate::Depth;
    }
    if (b.tri.maxReprojPx > cfg.birthMaxReprojPx)
    {
        return BirthGate::Misfit;
    }
    return BirthGate::Accepted;
}

double medianOf(std::vector<double> v)
{
    if (v.empty())
    {
        return 0.0;
    }
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    return v[mid];
}

} // namespace

SlamPointPipeline::SlamPointPipeline(const Camera& camera, const SlamConfig& cfg)
    : camera_(camera),
      cfg_(cfg),
      model_(std::make_shared<SystemVisualNav>(cfg.nuPsd)),
      map_(cfg),
      extractor_(900)
{
    if (!camera_.isValid())
    {
        throw std::invalid_argument("SlamPointPipeline: camera is not valid");
    }
    validateConfig(cfg_);
    est_ = std::make_unique<SystemEstimator>(model_, initialDensity(cfg_), 0.0);
}

bool SlamPointPipeline::seedVelocityFromEssential(const Eigen::Matrix3d& rRel, const Eigen::Vector3d& tRel)
{
    // Relative pose from the keyframe essential matrix (S6KeyframeFlow): p_B = R p_A + t, unit t (scale not observable).
    // Only the direction of the camera motion is used; the speed is the ASSUMED speed prior.
    // Camera B origin relative to camera A (world): c_B - c_A = -R_cB t with R_cB = R_cA R^T (since R = R_cB^T R_cA).
    const Eigen::VectorXd mu = est_->mean();
    const Eigen::Matrix3d rcA = rigid::cameraRotationInWorld(mu.segment<3>(SystemVisualNav::kEtaPrevOffset + 3), camera_.Rbc());
    const Eigen::Matrix3d rcB = rcA * rRel.transpose();
    Eigen::Vector3d dirWorld = -(rcB * tRel);
    if (!(dirWorld.norm() > 0.5))
    {
        return false;
    }
    dirWorld.normalize();
    const double speed = cfg_.useSpeedPrior ? cfg_.speedPriorMps : 1.0;
    const Eigen::Vector3d vBody = speed * rpy2rot(mu.segment<3>(SystemVisualNav::kEtaOffset + 3)).transpose() * dirWorld;

    // Mean shift in information form: nu' = nu + Lambda delta = nu + Xi^T (Xi delta), delta = [vBody - nu_v; 0].
    const Eigen::Index n = est_->dim();
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(n);
    delta.head<3>() = vBody - mu.head<3>();
    const Eigen::MatrixXd& xi = est_->state().sqrtInfo();
    const Eigen::VectorXd nuShifted = est_->state().infoVec() + xi.transpose() * (xi * delta);
    if (!nuShifted.allFinite())
    {
        return false;
    }
    est_->setState(GaussianInfo<double>::fromSqrtInfo(nuShifted, xi));
    return true;
}

Eigen::Matrix3d SlamPointPipeline::sqrtCovBlock(Eigen::Index offset) const
{
    const Eigen::MatrixXd& xi = est_->state().sqrtInfo();
    const Eigen::Index n = est_->dim();
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(n, 3);
    e.block<3, 3>(offset, 0) = Eigen::Matrix3d::Identity();
    // Y = Xi^{-T} E:  Y^T Y = E^T Sigma E with Sigma = Xi^{-1} Xi^{-T} (triangular solve, no inverse of Lambda).
    const Eigen::MatrixXd y = xi.transpose().triangularView<Eigen::Lower>().solve(e);
    const Eigen::MatrixXd r = core::upperFactor(y);
    return r.topLeftCorner<3, 3>();
}

Eigen::Matrix3d SlamPointPipeline::positionSqrtCov() const
{
    return sqrtCovBlock(SystemVisualNav::kEtaOffset);
}

Eigen::Matrix3d SlamPointPipeline::landmarkSqrtCov(Eigen::Index offset) const
{
    return sqrtCovBlock(offset);
}

void SlamPointPipeline::recordBirth(Eigen::Index id, const Descriptor& descriptor, int frameIndex,
                                    const Eigen::Vector2d& pixel)
{
    SlamLandmarkRecord rec;
    rec.id = id;
    rec.descriptor = descriptor;
    rec.birthFrame = frameIndex;
    rec.lastSeenFrame = frameIndex;
    rec.lastPixel = pixel;
    map_.insert(rec);
}

void SlamPointPipeline::delayedBirths(int frameIndex, const std::vector<KeyframeTrack>& kfTracks, SlamFrameStats& st)
{
    births_.clear();
    birthPixels_.clear();
    const int capacity = std::min(cfg_.maxBirthsPerFrame, cfg_.maxLandmarks - static_cast<int>(map_.size()));
    if (capacity <= 0 || kfTracks.empty())
    {
        return;
    }
    // Posterior of this frame (after the MAP update and the culling): both pose blocks are in the state.
    const Eigen::VectorXd mu = est_->mean();
    const Eigen::Index kfOffset = SystemVisualNav::kEtaPrevOffset;   // keyframe pose
    const Eigen::Index curOffset = SystemVisualNav::kEtaOffset;      // current pose

    // Tracks of this frame by id, and the tracks that already observe a landmark.
    std::unordered_map<int, std::size_t> indexOfTrack;
    indexOfTrack.reserve(trackIds_.size());
    for (std::size_t i = 0; i < trackIds_.size(); ++i)
    {
        indexOfTrack[trackIds_[i]] = i;
    }
    std::unordered_set<int> owned;
    for (const SlamLandmarkRecord& r : map_.records())
    {
        if (r.trackId >= 0)
        {
            owned.insert(r.trackId);
        }
    }

    struct Candidate
    {
        std::size_t track;       // index into kfTracks
        TwoViewBirth birth;      // correlated birth density at the state means
    };
    std::vector<Candidate> cands;
    for (std::size_t i = 0; i < kfTracks.size(); ++i)
    {
        const KeyframeTrack& k = kfTracks[i];
        if (keyframes_.isBorn(k.id) || owned.count(k.id) > 0)
        {
            continue;
        }
        ++st.birthCandidates;
        // A track on a visible predicted landmark is an existing point, not a new one.
        const bool onLandmark = std::any_of(predictions_.predictions.begin(), predictions_.predictions.end(),
                                            [&](const LandmarkPrediction& p)
                                            { return p.visible && (p.pixel - k.pixel).norm() < cfg_.birthSeparationPx; });
        if (onLandmark)
        {
            continue;
        }
        TwoViewBirth b = twoViewBirth(camera_, mu, kfOffset, curOffset, k.kfPixel, k.pixel,
                                      cfg_.pixelNoise.sigma(camera_, k.pixel));
        switch (birthGate(b, cfg_))
        {
        case BirthGate::Accepted:
            cands.push_back({i, std::move(b)});
            break;
        case BirthGate::Invalid:
            ++st.birthRejectedInvalid;
            break;
        case BirthGate::Behind:
            ++st.birthRejectedBehind;
            break;
        case BirthGate::Angle:
            ++st.birthRejectedAngle;
            break;
        case BirthGate::Depth:
            ++st.birthRejectedDepth;
            break;
        case BirthGate::Misfit:
            ++st.birthRejectedMisfit;
            break;
        }
    }

    // Largest parallax first (best-conditioned depths), with a minimum spacing between the births of this frame.
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Candidate& a, const Candidate& b) { return a.birth.tri.angleDeg > b.birth.tri.angleDeg; });
    std::vector<double> angles;
    int placed = 0;
    for (std::size_t c = 0; c < cands.size() && placed < capacity; ++c)
    {
        const KeyframeTrack& k = kfTracks[cands[c].track];
        const Eigen::Vector2d& px = k.pixel;
        const bool spaced = std::none_of(birthPixels_.begin(), birthPixels_.end(),
                                         [&](const Eigen::Vector2d& q) { return (q - px).norm() < cfg_.birthMinSpacingPx; });
        if (!spaced)
        {
            continue;
        }
        // The descriptor of the landmark is the track's descriptor (same computation as in the association). A track
        // without a descriptor (image border) is not born.
        const auto it = indexOfTrack.find(k.id);
        if (it == indexOfTrack.end() || !described_[it->second])
        {
            continue;
        }
        // The candidate matrices were built before the births of this frame: the new blocks are independent of the
        // earlier ones (their pixels and the poses they depend on are the same), so the G row is zero-padded.
        const TwoViewBirth& b = cands[c].birth;
        Eigen::MatrixXd g = Eigen::MatrixXd::Zero(3, est_->dim());
        g.leftCols(b.g.cols()) = b.g;
        const Eigen::Index id = model_->addLandmarkLinear(*est_, SystemVisualNav::LandmarkType::Point, g, b.c, b.lS);
        keyframes_.markBorn(k.id);
        recordBirth(id, features_[it->second].descriptor, frameIndex, px);
        map_.at(id).trackId = k.id;
        birthPixels_.push_back(px);
        angles.push_back(b.tri.angleDeg);
        st.births += 1;
        ++placed;
    }
    st.birthMedianAngleDeg = medianOf(angles);
}

void SlamPointPipeline::legacyBirths(int frameIndex, SlamFrameStats& st)
{
    // Births from unmatched features with a single-view depth prior (the earlier Scenario 6 behaviour, kept for ablation).
    const int capacity = cfg_.maxLandmarks - static_cast<int>(map_.size());
    std::vector<bool> birthBlocked = association_.featureUsed;
    if (cfg_.birthPersistence)
    {
        for (std::size_t f = 0; f < features_.size(); ++f)
        {
            if (birthBlocked[f])
            {
                continue;
            }
            bool persistent = false;
            for (const PendingBirth& pb : pendingBirth_)
            {
                if (pb.frame == frameIndex - 1 && (pb.pixel - features_[f].pixel).norm() <= cfg_.birthPersistencePx)
                {
                    persistent = true;
                    break;
                }
            }
            if (persistent)
            {
                st.persistentCandidates += 1;
            }
            else
            {
                birthBlocked[f] = true;
            }
        }
    }
    births_ = selectBirthFeatures(features_, birthBlocked, predictions_, cfg_,
                                  std::min(cfg_.maxBirthsPerFrame, std::max(0, capacity)));
    birthPixels_.clear();
    pendingBirth_.clear();
    for (std::size_t f = 0; f < features_.size(); ++f)
    {
        if (!association_.featureUsed[f])
        {
            pendingBirth_.push_back({features_[f].pixel, frameIndex});
        }
    }
    for (std::size_t f : births_)
    {
        const PointFeature& feat = features_[f];
        const Eigen::Index id = initialiseLandmark(*est_, *model_, camera_, feat.pixel,
                                                   cfg_.pixelNoise.sigma(camera_, feat.pixel), cfg_.depthPriorMean,
                                                   cfg_.depthPriorSigma);
        if (id < 0)
        {
            continue;
        }
        recordBirth(id, feat.descriptor, frameIndex, feat.pixel);
        birthPixels_.push_back(feat.pixel);
        st.births += 1;
    }
}

SlamFrameStats SlamPointPipeline::step(const cv::Mat& gray, int frameIndex, double t)
{
    if (!std::isfinite(t) || (started_ && t < est_->time()))
    {
        throw std::invalid_argument("SlamPointPipeline::step: time must be finite and non-decreasing");
    }
    started_ = true;
    const Clock::time_point tStart = Clock::now();
    SlamFrameStats st;
    st.frame = frameIndex;
    st.time = t;
    st.landmarksBefore = static_cast<int>(map_.size());
    st.keyframeParallaxPx = std::numeric_limits<double>::quiet_NaN();
    st.speedMps = std::numeric_limits<double>::quiet_NaN();
    st.cheiralityPosPre = std::numeric_limits<double>::quiet_NaN();
    st.cheiralityPosPost = std::numeric_limits<double>::quiet_NaN();
    st.cheiralityPosRecover = std::numeric_limits<double>::quiet_NaN();
    st.transCosPre = std::numeric_limits<double>::quiet_NaN();
    st.transCosPost = std::numeric_limits<double>::quiet_NaN();

    // 1. Persistent LK tracks (ImageFlow must see every frame). The tracks are the features of this frame: each has
    //    a pixel and a descriptor computed at its pixel (PointFeatureExtractor::describeAt), see SlamTrackAssociation.h.
    Clock::time_point t0 = Clock::now();
    flow_.update(gray);
    const std::vector<ImageFlowMatch>& active = flow_.activeTracks();
    features_.clear();
    trackIds_.clear();
    std::vector<Eigen::Vector2d> trackPixels;
    trackPixels.reserve(active.size());
    for (const ImageFlowMatch& m : active)
    {
        PointFeature f;
        f.pixel = Eigen::Vector2d(m.curr.x, m.curr.y);
        features_.push_back(f);
        trackIds_.push_back(m.id);
        trackPixels.push_back(f.pixel);
    }
    std::vector<Descriptor> descriptors;
    extractor_.describeAt(gray, trackPixels, described_, descriptors);
    for (std::size_t i = 0; i < features_.size(); ++i)
    {
        features_[i].descriptor = descriptors[i];
    }
    st.features = static_cast<int>(features_.size());
    st.msDetect = millisecondsSince(t0);
    t0 = Clock::now();

    // 2. Time update keeps the keyframe pose (etaPrev) and the landmarks; the first frame creates the keyframe.
    if (t > est_->time())
    {
        model_->predictPlain(*est_, t);
    }
    if (!model_->hasDelayedPose())
    {
        model_->predictDelayed(*est_, t);   // zero step: keyframe pose := current pose (copy with jitter, see SystemVisualNav)
        keyframes_.start(frameIndex, active);
    }
    // The keyframe started at this frame has no baseline yet: the flow and the births wait for the next frame.
    const bool keyframeAgeZero = keyframes_.keyframeFrame() == frameIndex;
    const std::vector<KeyframeTrack> kfTracks = keyframes_.alive(active);
    st.keyframeTracks = static_cast<int>(kfTracks.size());

    // 3. Flow measurement between the keyframe pose (etaPrev) and the current pose (eta), from the keyframe tracks.
    bool flowValid = false;
    if (cfg_.useFlow && !keyframeAgeZero && !kfTracks.empty())
    {
        std::vector<IndoorFlowPixelMatch> pixels;
        pixels.reserve(kfTracks.size());
        for (const KeyframeTrack& k : kfTracks)
        {
            IndoorFlowPixelMatch p;
            p.prevPx = k.kfPixel;
            p.currPx = k.pixel;
            pixels.push_back(p);
        }
        const S6FlowFrame sf = buildKeyframeFlowFrame(camera_, *est_, pixels, cfg_.flow);
        const IndoorFlowFrame& ff = sf.flow;
        st.flowInput = ff.input;
        st.flowValid = ff.valid;
        st.flowRotationModel = ff.model == IndoorFlowModel::Rotation;
        st.flowPairs = ff.gated;
        st.keyframeParallaxPx = ff.medianRotationMisfitPx;
        flowValid = ff.valid;
        if (ff.valid)
        {
            const MeasurementIndoorFlow fm(t, ff.model, ff.pairs, ff.sigmaNorm, camera_.Rbc(), camera_.rCBb());
            const Eigen::VectorXd r = fm.whitenedResiduals(est_->mean(), nullptr);
            st.flowNis = r.squaredNorm();
            st.flowDof = static_cast<int>(r.size());
            // Cheirality diagnostics (no effect on the estimate): the epipolar residual is invariant to t -> -t, so
            // the sign of the posterior translation is checked against the depth signs and against recoverPose.
            const IndoorFlowRelativePose preRel = indoorFlowRelativePose(est_->mean(), camera_.Rbc(), camera_.rCBb(), false);
            const CheiralityCensus preCensus = cheiralityCensus(ff.pairs, preRel.R, preRel.t);
            st.cheiralityPosPre = preCensus.fractionPositive();
            st.cheiralityTotal = preCensus.total;
            if (sf.hasRelative)
            {
                st.cheiralityPosRecover = cheiralityCensus(ff.pairs, sf.rRel, sf.tRel).fractionPositive();
                const double nPre = preRel.t.norm();
                st.transCosPre = nPre > 0.0 ? preRel.t.dot(sf.tRel) / nPre : std::numeric_limits<double>::quiet_NaN();
            }
            FuncMinOptions fo;
            fo.gradientTolerance = cfg_.mapGradientTolerance;
            fo.functionTolerance = 1e-10;
            const GaussianInfo<double> before = est_->state();
            // Flow MAP with the cheirality check of the translation branch (the epipolar likelihood cannot tell t from -t).
            const FlowUpdateResult fu = updateFlowWithCheirality(*est_, camera_, fm, fo);
            const MeasurementUpdateResult& res = fu.update;
            if (res.map.status == FuncMinStatus::NumericalFailure || !est_->mean().allFinite()
                || !est_->state().sqrtInfo().allFinite())
            {
                est_->setState(before);
                ++mapFailures_;
            }
            else
            {
                st.flowUpdated = true;
                st.flowBranchMirrored = fu.mirrored;
            }
            {
                const IndoorFlowRelativePose postRel = indoorFlowRelativePose(est_->mean(), camera_.Rbc(), camera_.rCBb(), false);
                const CheiralityCensus postCensus = cheiralityCensus(ff.pairs, postRel.R, postRel.t);
                st.cheiralityPosPost = postCensus.fractionPositive();
                const double nPost = postRel.t.norm();
                if (sf.hasRelative)
                {
                    st.transCosPost = nPost > 0.0 ? postRel.t.dot(sf.tRel) / nPost : std::numeric_limits<double>::quiet_NaN();
                    st.flowTPost = nPost > 0.0 ? Eigen::Vector3d(postRel.t / nPost) : Eigen::Vector3d::Zero();
                    st.flowTRec = sf.tRel;
                    const Eigen::Matrix3d dr = postRel.R.transpose() * sf.rRel;
                    const double cosA = std::clamp(0.5 * (dr.trace() - 1.0), -1.0, 1.0);
                    st.flowRotDiffDeg = std::acos(cosA) * 180.0 / std::numbers::pi;
                }
            }
            // First epipolar frame: seed the velocity direction from the essential matrix (once, see initialDensity).
            if (!velocitySeeded_ && sf.hasRelative && ff.model == IndoorFlowModel::Epipolar && st.flowUpdated
                && st.keyframeParallaxPx > cfg_.velocitySeedParallaxPx)
            {
                velocitySeeded_ = seedVelocityFromEssential(sf.rRel, sf.tRel);
            }
        }
    }
    st.msFlow = millisecondsSince(t0);

    // 3b. Metric scale: speed prior on the body velocity (ASSUMED, see S6SpeedPrior.h), every frame.
    if (cfg_.useSpeedPrior)
    {
        const MeasurementSpeedPrior speedPrior(t, cfg_.speedPriorMps, cfg_.speedPriorSigmaMps);
        FuncMinOptions so;
        so.gradientTolerance = cfg_.mapGradientTolerance;
        so.functionTolerance = 1e-10;
        const GaussianInfo<double> before = est_->state();
        const MeasurementUpdateResult res = speedPrior.process(*est_, so);
        if (res.map.status == FuncMinStatus::NumericalFailure || !est_->mean().allFinite()
            || !est_->state().sqrtInfo().allFinite())
        {
            est_->setState(before);
            ++mapFailures_;
        }
    }
    st.speedMps = est_->mean().head<3>().norm();

    // 4. Landmark prediction.
    t0 = Clock::now();
    predictions_ = predictLandmarks(*est_, *model_, camera_, cfg_);
    st.msPredict = millisecondsSince(t0);

    // 5. Association.
    t0 = Clock::now();
    association_ = associateTracks(predictions_, features_, trackIds_, described_, map_, camera_, cfg_);
    st.msAssociate = millisecondsSince(t0);
    st.visible = association_.numVisible;
    st.associated = static_cast<int>(association_.observations.size());
    st.noCandidate = association_.numNoCandidate;
    st.ambiguous = association_.numAmbiguous;
    st.conflicts = association_.numConflict;

    // 6. Measurement update with the associated landmarks (existing landmarks only).
    t0 = Clock::now();
    if (st.associated >= cfg_.minMatchesForUpdate)
    {
        const Eigen::Index m = 2 * static_cast<Eigen::Index>(st.associated);
        const Eigen::VectorXd& innov = association_.innovation;
        Eigen::LLT<Eigen::MatrixXd> llt(association_.innovationCov);
        if (llt.info() == Eigen::Success)
        {
            st.nis = llt.matrixL().solve(innov).squaredNorm();   // diagnostic only: innovation covariance factor
        }
        else
        {
            st.nis = std::numeric_limits<double>::quiet_NaN();
        }
        st.nisDof = static_cast<int>(m);

        MeasurementPointBundle meas(t, camera_, association_.observations, SystemVisualNav::kEtaOffset);
        FuncMinOptions mapOptions;
        mapOptions.gradientTolerance = cfg_.mapGradientTolerance;
        mapOptions.functionTolerance = 1e-10;
        const GaussianInfo<double> before = est_->state();
        const MeasurementUpdateResult res = meas.process(*est_, mapOptions);
        st.mapIterations = res.map.iterations;
        st.mapStatus = static_cast<int>(res.map.status);
        st.precisionExact = res.precisionIsExact;
        // Robustness: a failed MAP (non-finite objective or posterior) is reverted, so the frame is predicted only.
        if (res.map.status == FuncMinStatus::NumericalFailure || !est_->mean().allFinite()
            || !est_->state().sqrtInfo().allFinite())
        {
            est_->setState(before);
            ++mapFailures_;
        }
        else
        {
            st.updated = true;
        }
    }
    st.msUpdate = millisecondsSince(t0);

    // 7. Records: hits / misses / behind-camera counters (from the pre-update predictions).
    for (SlamLandmarkRecord& r : map_.records())
    {
        r.associatedThisFrame = false;
    }
    for (std::size_t k = 0; k < association_.matchedPrediction.size(); ++k)
    {
        const std::size_t p = association_.matchedPrediction[k];
        SlamLandmarkRecord& r = map_.at(predictions_.predictions[p].id);
        r.associatedThisFrame = true;
        r.hits += 1;
        r.misses = 0;
        r.behindFrames = 0;
        r.lastSeenFrame = frameIndex;
        const std::size_t tr = static_cast<std::size_t>(association_.matchedFeature[k]);
        r.lastPixel = features_[tr].pixel;
        r.trackId = trackIds_[tr];
        if (described_[tr])
        {
            r.descriptor = features_[tr].descriptor;
        }
        r.confirmed = r.confirmed || r.hits >= cfg_.confirmHits;
    }
    for (std::size_t k = 0; k < predictions_.predictions.size(); ++k)
    {
        const LandmarkPrediction& p = predictions_.predictions[k];
        SlamLandmarkRecord& r = map_.at(p.id);
        if (r.associatedThisFrame)
        {
            continue;
        }
        // A miss needs a visible landmark with no feature in its gate: an ambiguous or conflicting candidate is not a miss.
        if (p.visible && !association_.predictionGated[k])
        {
            r.misses += 1;
        }
        r.behindFrames = (p.depth < cfg_.minDepth) ? r.behindFrames + 1 : 0;
    }

    // 8. Culling.
    t0 = Clock::now();
    const std::vector<Eigen::Index> cull = selectLandmarksToCull(map_, frameIndex);
    for (Eigen::Index id : cull)
    {
        model_->removeLandmark(*est_, id);
        map_.erase(id);
    }
    st.culled = static_cast<int>(cull.size());

    // 9. Births after the update (the pose is the posterior one).
    if (cfg_.delayedInitialisation)
    {
        if (!keyframeAgeZero)
        {
            delayedBirths(frameIndex, kfTracks, st);
        }
        else
        {
            births_.clear();
            birthPixels_.clear();
        }
    }
    else
    {
        legacyBirths(frameIndex, st);
    }
    st.msBirthCull = millisecondsSince(t0);

    // 10. Keyframe decision (the births above used the keyframe that is current for this frame).
    const int framesSince = keyframes_.hasKeyframe() ? frameIndex - keyframes_.keyframeFrame() : 0;
    if (keyframeShouldStart(cfg_, flowValid, st.keyframeParallaxPx, st.keyframeTracks, framesSince))
    {
        model_->predictDelayed(*est_, t);   // zero step at the same time: keyframe pose := eta_k
        keyframes_.start(frameIndex, active);
        st.keyframeStarted = true;
    }
    st.keyframeAge = keyframes_.hasKeyframe() ? frameIndex - keyframes_.keyframeFrame() : 0;

    {
        // ScenarioRunner contract: camera positions in world, r_CN^n = r_BN^n + R_NB r_CB^B.
        const Eigen::VectorXd mu = est_->mean();
        trajectory_.push_back(rigid::cameraOriginInWorld(mu.segment<3>(SystemVisualNav::kEtaOffset + 3),
                                                         mu.segment<3>(SystemVisualNav::kEtaOffset), camera_.rCBb()));
    }
    st.landmarks = static_cast<int>(map_.size());
    st.msTotal = millisecondsSince(tStart);
    return st;
}

} // namespace slam
