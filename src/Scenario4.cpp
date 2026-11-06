// Scenario 4 (outdoor flight): optical-flow plane bundle + visual horizon + altimeter in a square-root information
// filter. Implementation of Scenario4Runner and the factory makeScenario4Runner().
#include "Scenario4Runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <unordered_map>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include "CoreLinalg.h"
#include "GaussianInfo.h"
#include "MeasurementAltimeter.h"
#include "MeasurementHorizon.h"
#include "MeasurementOutdoorFlowBundle.h"
#include "OutdoorLaplaceUpdate.h"
#include "rotation.hpp"

namespace
{

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Name of a MAP outcome for the per-frame notes.
const char* outcomeName(MapOutcome outcome)
{
    switch (outcome)
    {
    case MapOutcome::Converged: return "converged";
    case MapOutcome::AcceptedMaxIterations: return "max-iterations accepted";
    case MapOutcome::SkippedMaxIterations: return "max-iterations rejected";
    case MapOutcome::SkippedNumericalFailure: return "numerical failure";
    case MapOutcome::SkippedNoCurvature: return "no curvature";
    }
    return "unknown";
}

/// Median of a non-empty sample (average of the two middle values for an even count).
double medianOf(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/**
 * @brief NIS record of one update, from the prior innovation nu and its Jacobian J with respect to the support.
 *
 * Xi is the square-root information of the support prior. B^T = Xi^{-T} J^T is one triangular solve (B = J Xi^{-1}),
 * R_B = qr(B) (k x k) and c = B^T nu. No covariance or information matrix is inverted or factorised as an information
 * matrix (see Scenario4NisRecord).
 */
Scenario4NisRecord makeNisRecord(Scenario4NisKind kind, int frameIndex, const Eigen::VectorXd& nu,
                                 const Eigen::MatrixXd& jac, const SystemEstimator& est,
                                 const std::vector<Eigen::Index>& support)
{
    const GaussianInfo<double> marginal = est.state().marginal(support);
    const Eigen::MatrixXd& xi = marginal.sqrtInfo();
    const Eigen::MatrixXd bT = xi.transpose().triangularView<Eigen::Lower>().solve(jac.transpose());
    Scenario4NisRecord rec;
    rec.kind = kind;
    rec.frameIndex = frameIndex;
    rec.dof = static_cast<int>(nu.size());
    rec.nu2 = nu.squaredNorm();
    const Eigen::MatrixXd b = bT.transpose();
    rec.rB = core::upperFactor(b);
    rec.c = bT * nu;
    return rec;
}

/// Ground-truth-free check: every matrix entry is finite.
bool finite(const Eigen::MatrixXd& m)
{
    return m.allFinite();
}

constexpr double kEarthRadiusM = 6371000.0;

} // namespace

double scenario4NisAt(const Scenario4NisRecord& record, double sigma)
{
    if (!(sigma > 0.0) || !std::isfinite(sigma))
    {
        throw std::invalid_argument("scenario4NisAt: sigma must be positive and finite");
    }
    const Eigen::Index k = record.rB.cols();
    if (k == 0 || record.rB.rows() != k || record.c.size() != k)
    {
        throw std::invalid_argument("scenario4NisAt: inconsistent NIS record");
    }
    // R_sigma = qr([R_B; sigma I]) satisfies R_sigma^T R_sigma = B^T B + sigma^2 I.
    Eigen::MatrixXd stacked = Eigen::MatrixXd::Zero(2 * k, k);
    stacked.topRows(k) = record.rB;
    stacked.bottomRows(k) = sigma * Eigen::MatrixXd::Identity(k, k);
    const Eigen::MatrixXd rSigma = core::upperFactor(stacked);
    // NIS = (||nu||^2 - c^T (B^T B + sigma^2 I)^{-1} c) / sigma^2 = (||nu||^2 - ||R_sigma^{-T} c||^2) / sigma^2.
    const Eigen::VectorXd z = rSigma.transpose().triangularView<Eigen::Lower>().solve(record.c);
    // Rounding can make the difference of two nearly equal terms slightly negative; the NIS is non-negative.
    return std::max(0.0, record.nu2 - z.squaredNorm()) / (sigma * sigma);
}

Scenario4Calibration scenario4CalibrateSigma(const std::vector<Scenario4NisRecord>& records, Scenario4NisKind kind,
                                             int frameBegin, int frameEnd, double sigmaLo, double sigmaHi)
{
    if (!(sigmaLo > 0.0) || !(sigmaHi > sigmaLo) || !std::isfinite(sigmaHi))
    {
        throw std::invalid_argument("scenario4CalibrateSigma: need 0 < sigmaLo < sigmaHi");
    }
    std::vector<const Scenario4NisRecord*> sel;
    for (const Scenario4NisRecord& r : records)
    {
        if (r.kind == kind && r.frameIndex >= frameBegin && r.frameIndex < frameEnd)
        {
            sel.push_back(&r);
        }
    }
    Scenario4Calibration out;
    out.records = sel.size();
    if (sel.empty())
    {
        return out;
    }

    // Median over records of the probability-integral transform P(chi2_dof <= NIS_i(sigma)). It is 0.5 when the median
    // NIS equals the median of its own chi-squared reference. Each record uses its own dof.
    auto pitAt = [&sel](double sigma)
    {
        std::vector<double> pit;
        pit.reserve(sel.size());
        for (const Scenario4NisRecord* r : sel)
        {
            const boost::math::chi_squared_distribution<double> dist(static_cast<double>(r->dof));
            pit.push_back(boost::math::cdf(dist, scenario4NisAt(*r, sigma)));
        }
        return medianOf(std::move(pit));
    };

    // The statistic is non-increasing in sigma (NIS decreases as S grows), so bisect on log(sigma).
    const double fLo = pitAt(sigmaLo) - 0.5;
    const double fHi = pitAt(sigmaHi) - 0.5;
    if (!(fLo >= 0.0 && fHi <= 0.0))
    {
        out.sigma = (fLo < 0.0) ? sigmaLo : sigmaHi;   // not bracketed: report the nearest bound, converged = false
        out.medianPit = pitAt(out.sigma);
        return out;
    }
    double lo = std::log(sigmaLo);
    double hi = std::log(sigmaHi);
    for (int it = 0; it < 200 && hi - lo > 1e-10; ++it)
    {
        const double mid = 0.5 * (lo + hi);
        if (pitAt(std::exp(mid)) - 0.5 > 0.0)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    out.sigma = std::exp(0.5 * (lo + hi));
    out.converged = true;
    out.medianPit = pitAt(out.sigma);
    std::vector<double> ratio;
    ratio.reserve(sel.size());
    for (const Scenario4NisRecord* r : sel)
    {
        ratio.push_back(scenario4NisAt(*r, out.sigma) / static_cast<double>(r->dof));
    }
    out.medianNisPerDof = medianOf(std::move(ratio));
    return out;
}

struct Scenario4Runner::Impl
{
    Camera camera;
    double fps = 0.0;
    cv::Size frameSize;
    std::string videoPath;
    DJIVideoCaption caption;
    std::vector<AltitudeSample> altitudeEvents;
    std::size_t nextAltitude = 0;
    double altitudeReference = 0.0;     ///< h0: first SRT altitude [m]
    std::vector<Eigen::Vector3d> validationLocal;   ///< per SRT record, NED metres relative to the first fix (validation only)
    std::vector<bool> validationValid;

    Eigen::Matrix3d Rbc = Eigen::Matrix3d::Identity();
    double focal = 1.0;

    std::shared_ptr<SystemVisualNav> model;
    std::unique_ptr<SystemEstimator> est;
    ImageFlow flow;
    SkyGroundClassifier classifier;
    bool classifierReady = false;

    bool initialised = false;
    bool havePrevious = false;
    int lastFrameIndex = -1;

    // Validation-only yaw alignment sums (GPS track vs estimate, origin-anchored).
    double sumDot = 0.0;
    double sumCross = 0.0;

    // Per-frame visual products (for overlay and status).
    cv::Mat overlay;
    std::vector<std::string> statusLines;
    std::vector<ImageFlowMatch> lastMatches;   ///< baseline correspondences (keyframe -> current) used by the flow update
    std::vector<std::uint8_t> lastInlier;
    int keyframe = -1;                          ///< frame index of the flow keyframe (etaPrev = eta at this frame)
    std::unordered_map<int, cv::Point2d> keyPositions;  ///< track id -> position at the keyframe (full resolution)
    HorizonEstimate lastHorizon;
    bool lastHorizonUsed = false;
};

Scenario4Runner::Scenario4Runner(Scenario4Options options) : impl_(std::make_unique<Impl>()), options_(std::move(options))
{
    impl_->flow = ImageFlow(options_.flow);
}

Scenario4Runner::~Scenario4Runner() = default;

void Scenario4Runner::initialise(const Camera& camera, double fps, cv::Size frameSize, const std::string& videoPath)
{
    if (!camera.isValid())
    {
        throw std::invalid_argument("Scenario4Runner::initialise: camera is not valid");
    }
    if (!(fps > 0.0) || !std::isfinite(fps))
    {
        throw std::invalid_argument("Scenario4Runner::initialise: fps must be positive");
    }
    if (frameSize.width != camera.imageWidth() || frameSize.height != camera.imageHeight())
    {
        throw std::invalid_argument("Scenario4Runner::initialise: frame size does not match the camera");
    }
    if (camera.rCBb().norm() > 1e-12)
    {
        throw std::invalid_argument("Scenario4Runner::initialise: non-zero camera lever arm is not supported");
    }
    if (options_.useFlow && !(options_.flowSigmaPx > 0.0))
    {
        throw std::invalid_argument("Scenario4Runner::initialise: flow sigma must be positive");
    }
    impl_ = std::make_unique<Impl>();
    Impl& s = *impl_;
    s.camera = camera;
    s.fps = fps;
    s.frameSize = frameSize;
    s.videoPath = videoPath;
    s.Rbc = camera.Rbc();
    s.focal = 0.5 * (camera.fx() + camera.fy());
    s.flow = ImageFlow(options_.flow);

    // SRT: estimator-facing records and validation-only geodetic samples.
    const std::filesystem::path video(videoPath);
    const std::filesystem::path srt = options_.srtPath.empty()
        ? video.parent_path() / (video.stem().string() + ".SRT")
        : options_.srtPath;
    s.caption = DJIVideoCaption::load(srt, DJIVideoCaption::ParseMode::Lenient);
    s.altitudeEvents = s.caption.altitudeUpdates();
    if (s.altitudeEvents.empty())
    {
        throw std::runtime_error("Scenario4Runner: the SRT contains no altitude values: " + srt.string());
    }
    s.altitudeReference = s.altitudeEvents.front().altitudeM;
    groundDepth_ = options_.groundDepthM > 0.0 ? options_.groundDepthM : s.altitudeReference;

    // Validation-only local track (NED metres about the first valid fix). Never used by the estimator.
    const auto geodetic = s.caption.validationOnlyGeodeticSamples();
    const auto records = s.caption.records();
    s.validationLocal.assign(records.size(), Eigen::Vector3d::Zero());
    s.validationValid.assign(records.size(), false);
    std::optional<std::pair<double, double>> origin;
    for (std::size_t i = 0; i < records.size() && i < geodetic.size(); ++i)
    {
        if (!geodetic[i].latitudeDeg || !geodetic[i].longitudeDeg)
        {
            continue;
        }
        if (!origin)
        {
            origin = std::make_pair(*geodetic[i].latitudeDeg, *geodetic[i].longitudeDeg);
        }
        const double d2r = std::numbers::pi_v<double> / 180.0;
        const double north = kEarthRadiusM * (*geodetic[i].latitudeDeg - origin->first) * d2r;
        const double east = kEarthRadiusM * std::cos(origin->first * d2r) * (*geodetic[i].longitudeDeg - origin->second) * d2r;
        const double alt = records[i].altitudeM.value_or(s.altitudeReference);
        s.validationLocal[i] = Eigen::Vector3d(north, east, -(alt - s.altitudeReference));
        s.validationValid[i] = true;
    }

    // Shared process model and prior (world origin = initial body position, heading prior 0).
    Eigen::Matrix<double, 6, 1> nuPsd;
    nuPsd << options_.nuPsdVelocity, options_.nuPsdVelocity, options_.nuPsdVelocity,
             options_.nuPsdAngularRate, options_.nuPsdAngularRate, options_.nuPsdAngularRate;
    s.model = std::make_shared<SystemVisualNav>(nuPsd);

    Eigen::VectorXd mean0 = Eigen::VectorXd::Zero(12);
    Eigen::VectorXd sigma0(12);
    sigma0 << options_.priorVelocitySigma, options_.priorVelocitySigma, options_.priorVelocitySigma,
              options_.priorAngularRateSigma, options_.priorAngularRateSigma, options_.priorAngularRateSigma,
              options_.priorPositionSigmaXY, options_.priorPositionSigmaXY, options_.priorPositionSigmaZ,
              options_.priorRollPitchSigma, options_.priorRollPitchSigma, options_.priorYawSigma;
    const Eigen::MatrixXd p0 = sigma0.array().square().matrix().asDiagonal();
    SystemEstimator::State init = SystemEstimator::State::fromMoment(mean0, p0);
    s.est = std::make_unique<SystemEstimator>(s.model, std::move(init), 0.0);

    // Horizon classifier: trained from the sky*/ground* patches next to the video (or the configured directory).
    s.classifierReady = false;
    if (options_.useHorizon)
    {
        const std::filesystem::path dir = options_.trainingDirectory.empty() ? video.parent_path() : options_.trainingDirectory;
        std::vector<std::filesystem::path> skyFiles;
        std::vector<std::filesystem::path> groundFiles;
        if (std::filesystem::is_directory(dir))
        {
            for (const auto& entry : std::filesystem::directory_iterator(dir))
            {
                const std::string name = entry.path().filename().string();
                if (!entry.is_regular_file() || entry.path().extension() != ".png")
                {
                    continue;
                }
                if (name.rfind("sky", 0) == 0)
                {
                    skyFiles.push_back(entry.path());
                }
                else if (name.rfind("ground", 0) == 0)
                {
                    groundFiles.push_back(entry.path());
                }
            }
        }
        std::sort(skyFiles.begin(), skyFiles.end());
        std::sort(groundFiles.begin(), groundFiles.end());
        std::vector<cv::Mat> sky = options_.trainingSky;
        std::vector<cv::Mat> ground = options_.trainingGround;
        if (sky.empty() && ground.empty() && !skyFiles.empty() && !groundFiles.empty())
        {
            for (const auto& f : skyFiles)
            {
                cv::Mat im = cv::imread(f.string(), cv::IMREAD_COLOR);
                if (!im.empty())
                {
                    sky.push_back(im);
                }
            }
            for (const auto& f : groundFiles)
            {
                cv::Mat im = cv::imread(f.string(), cv::IMREAD_COLOR);
                if (!im.empty())
                {
                    ground.push_back(im);
                }
            }
        }
        if (!sky.empty() && !ground.empty())
        {
            // The classifier is applied at the analysis resolution, so the training patches must be taken at the
            // same scale: local texture (std of L over a window) depends on the resolution.
            for (std::vector<cv::Mat>* set : {&sky, &ground})
            {
                for (cv::Mat& im : *set)
                {
                    cv::resize(im, im, cv::Size(), options_.horizonAnalysisScale, options_.horizonAnalysisScale, cv::INTER_AREA);
                }
            }
            s.classifier = SkyGroundClassifier::train(sky, ground);
            s.classifierReady = true;
        }
        if (!s.classifierReady)
        {
            std::println(stderr, "Scenario 4: no sky*/ground* training patches in {}; horizon updates disabled", dir.string());
        }
    }

    s.initialised = true;
    s.havePrevious = false;
    s.lastFrameIndex = -1;
    diagnostics_.clear();
    trajectory_.clear();
    validationTrack_.clear();
    validationYaw_ = 0.0;
    nisRecords_.clear();
    mapCounts_ = Scenario4MapCounts{};
}

bool Scenario4Runner::horizonAvailable() const
{
    return impl_->classifierReady && options_.useHorizon;
}

const SystemEstimator& Scenario4Runner::estimator() const
{
    if (!impl_->initialised)
    {
        throw std::logic_error("Scenario4Runner::estimator: not initialised");
    }
    return *impl_->est;
}

void Scenario4Runner::processFrame(const cv::Mat& frame, int frameIndex, double t)
{
    Impl& s = *impl_;
    if (!s.initialised)
    {
        throw std::logic_error("Scenario4Runner::processFrame: call initialise() first");
    }
    if (frame.empty() || frame.type() != CV_8UC3 || frame.size() != s.frameSize)
    {
        throw std::invalid_argument("Scenario4Runner::processFrame: frame must be BGR8 with the initialised size");
    }
    if (frameIndex <= s.lastFrameIndex)
    {
        throw std::invalid_argument("Scenario4Runner::processFrame: frames must be processed in increasing order");
    }
    if (!std::isfinite(t))
    {
        throw std::invalid_argument("Scenario4Runner::processFrame: time must be finite");
    }
    const auto tFrame = Clock::now();
    const bool nisOn = options_.computeNis || options_.recordNisRecords;
    Scenario4FrameDiagnostics d;
    d.frameIndex = frameIndex;
    d.time = t;
    std::string notes;
    const bool first = !s.havePrevious;

    // Full-resolution gray image (horizon edge refinement and feature tracking).
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // (1) Horizon front-end: classification at analysis resolution and boundary extraction.
    auto tHz = Clock::now();
    const double scale = options_.horizonAnalysisScale;
    cv::Mat logOdds;
    if (s.classifierReady)
    {
        cv::Mat small;
        cv::resize(frame, small, cv::Size(), scale, scale, cv::INTER_AREA);
        logOdds = s.classifier.logOdds(small);
        s.lastHorizon = extractHorizon(logOdds, scale, s.frameSize.width, options_.horizon, gray);
        d.horizonValid = s.lastHorizon.valid;
        d.horizonCandidates = static_cast<int>(s.lastHorizon.candidates.size());
        d.horizonInliers = static_cast<int>(s.lastHorizon.inliers.size());
        d.horizonSkyFraction = s.lastHorizon.skyFraction;
    }
    else
    {
        s.lastHorizon = HorizonEstimate{};
    }
    d.msHorizon += msSince(tHz);

    // (2) Feature tracking from frame k-1 to k, detection restricted to the ground (log-odds < 0).
    auto tFl = Clock::now();
    cv::Mat detectMask;
    if (s.classifierReady)
    {
        const cv::Mat ground = (logOdds < 0.0f);   // CV_8UC1, 255 = ground
        cv::resize(ground, detectMask, s.frameSize, 0.0, 0.0, cv::INTER_NEAREST);
    }
    s.flow.update(gray, detectMask);
    // Baseline correspondences: tracks alive at the keyframe, with their keyframe and current positions.
    s.lastMatches.clear();
    for (const ImageFlowMatch& tr : s.flow.activeTracks())
    {
        const auto it = s.keyPositions.find(tr.id);
        if (it != s.keyPositions.end())
        {
            ImageFlowMatch m = tr;
            m.prev = it->second;
            m.age = frameIndex - s.keyframe;
            s.lastMatches.push_back(m);
        }
    }
    d.flowMatches = static_cast<int>(s.lastMatches.size());
    d.trackerActive = s.flow.stats().active;
    d.msFlow += msSince(tFl);

    // (3) Time update. etaPrev must equal the pose at the flow keyframe: it is re-anchored (delayed-state shift) on the
    // frame after each keyframe and carried unchanged in between.
    auto tFi = Clock::now();
    if (!first && frameIndex == s.keyframe + 1)
    {
        s.model->predictDelayed(*s.est, t);
    }
    else if (!first)
    {
        s.model->predictPlain(*s.est, t);
    }
    else
    {
        s.est->predict(t);   // zero-length: keeps the layout, no-op
    }
    d.msFilter += msSince(tFi);

    // (4) Optical-flow plane bundle.
    s.lastInlier.assign(s.lastMatches.size(), 0);
    s.lastHorizonUsed = false;
    if (options_.useFlow && !first)
    {
        tFl = Clock::now();
        std::vector<std::uint8_t> inlier;
        cv::Mat hom;
        const int nIn = ImageFlow::robustHomographyInliers(s.lastMatches, options_.flowRansacPx, inlier, hom);
        s.lastInlier = inlier;
        d.flowInliers = nIn;
        if (nIn < options_.minFlowPairs)
        {
            notes += "flow: too few plane inliers; ";
        }
        else
        {
            std::vector<OutdoorFlowPair> pairs;
            pairs.reserve(static_cast<std::size_t>(nIn));
            for (std::size_t i = 0; i < s.lastMatches.size(); ++i)
            {
                if (inlier[i] == 0)
                {
                    continue;
                }
                const ImageFlowMatch& m = s.lastMatches[i];
                const Eigen::Vector2d pa = s.camera.pixelToNormalised(Eigen::Vector2d(m.prev.x, m.prev.y));
                const Eigen::Vector2d pb = s.camera.pixelToNormalised(Eigen::Vector2d(m.curr.x, m.curr.y));
                if (pa.allFinite() && pb.allFinite())
                {
                    pairs.push_back(OutdoorFlowPair{pa, pb});
                }
            }
            const Eigen::VectorXd x = s.est->mean();
            const double agl = groundDepth_ - x(MeasurementOutdoorFlowBundle::kPreviousOffset + 2);
            if (static_cast<int>(pairs.size()) < options_.minFlowPairs)
            {
                notes += "flow: too few valid normalised pairs; ";
            }
            else if (!(agl > options_.minAglForFlowM))
            {
                notes += "flow: predicted height above the plane too small; ";
            }
            else
            {
                try
                {
                    const double sigmaNorm = options_.flowSigmaPx / s.focal;
                    MeasurementOutdoorFlowBundle meas(t, pairs, sigmaNorm, groundDepth_, s.Rbc);
                    // NIS at the prior (before the update), from the support square-root information.
                    Eigen::MatrixXd jac;
                    const Eigen::VectorXd g = meas.predict(x, &jac);
                    const Eigen::VectorXd nu = meas.y() - g;
                    const std::vector<Eigen::Index> support = meas.supportIndices(*s.est);
                    d.flowDof = static_cast<int>(nu.size());
                    if (nisOn)
                    {
                        Scenario4NisRecord rec = makeNisRecord(Scenario4NisKind::Flow, frameIndex, nu, jac, *s.est, support);
                        d.flowNis = scenario4NisAt(rec, sigmaNorm);
                        if (options_.recordNisRecords)
                        {
                            nisRecords_.push_back(std::move(rec));
                        }
                    }
                    const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, *s.est, FuncMinOptions(), options_.mapPolicy);
                    d.flowOutcome = rep.outcome;
                    d.flowNewtonNorm = rep.newtonNorm;
                    mapCounts_.flow.add(rep.outcome, !rep.updateSkipped);
                    d.flowUpdated = !rep.updateSkipped;
                    if (rep.updateSkipped)
                    {
                        notes += std::string("flow: MAP update skipped (") + outcomeName(rep.outcome) + "); ";
                    }
                    const Eigen::VectorXd gPost = meas.predict(s.est->mean(), nullptr);
                    d.flowResidualRmsPx = std::sqrt((meas.y() - gPost).squaredNorm() / static_cast<double>(gPost.size())) * s.focal;
                }
                catch (const std::exception& e)
                {
                    if (!d.flowOutcome)
                    {
                        // The update itself threw before a MAP outcome was produced: counted as a numerical skip.
                        d.flowOutcome = MapOutcome::SkippedNumericalFailure;
                        mapCounts_.flow.add(MapOutcome::SkippedNumericalFailure, false);
                    }
                    notes += std::string("flow: update skipped (") + e.what() + "); ";
                }
            }
        }
        d.msFlow += msSince(tFl);
    }

    // (5) Visual horizon update (roll and pitch).
    if (options_.useHorizon && s.classifierReady && s.lastHorizon.valid && !first)
    {
        tHz = Clock::now();
        std::vector<Eigen::Vector2d> pts;
        pts.reserve(s.lastHorizon.inliers.size());
        for (const cv::Point2d& p : s.lastHorizon.inliers)
        {
            const Eigen::Vector2d n = s.camera.pixelToNormalised(Eigen::Vector2d(p.x, p.y));
            if (n.allFinite())
            {
                pts.push_back(n);
            }
        }
        if (pts.size() >= 2)
        {
            try
            {
                const double sigmaNorm = options_.horizonSigmaPx / s.focal;
                MeasurementHorizon meas(t, pts, sigmaNorm, s.Rbc);
                Eigen::MatrixXd jac;
                const Eigen::VectorXd r = meas.predict(s.est->mean(), &jac);
                const Eigen::VectorXd nu = -r;   // y = 0 for every boundary point
                const std::vector<Eigen::Index> support = meas.supportIndices(*s.est);
                d.horizonDof = static_cast<int>(nu.size());
                if (nisOn)
                {
                    Scenario4NisRecord rec = makeNisRecord(Scenario4NisKind::Horizon, frameIndex, nu, jac, *s.est, support);
                    d.horizonNis = scenario4NisAt(rec, sigmaNorm);
                    if (options_.recordNisRecords)
                    {
                        nisRecords_.push_back(std::move(rec));
                    }
                }
                const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, *s.est, FuncMinOptions(), options_.mapPolicy);
                d.horizonOutcome = rep.outcome;
                d.horizonNewtonNorm = rep.newtonNorm;
                mapCounts_.horizon.add(rep.outcome, !rep.updateSkipped);
                d.horizonUpdated = !rep.updateSkipped;
                if (rep.updateSkipped)
                {
                    notes += std::string("horizon: MAP update skipped (") + outcomeName(rep.outcome) + "); ";
                }
                const Eigen::VectorXd rPost = meas.predict(s.est->mean(), nullptr);
                d.horizonResidualRmsPx = std::sqrt(rPost.squaredNorm() / static_cast<double>(rPost.size())) * s.focal;
                s.lastHorizonUsed = !rep.updateSkipped;
            }
            catch (const std::exception& e)
            {
                if (!d.horizonOutcome)
                {
                    d.horizonOutcome = MapOutcome::SkippedNumericalFailure;
                    mapCounts_.horizon.add(MapOutcome::SkippedNumericalFailure, false);
                }
                notes += std::string("horizon: update skipped (") + e.what() + "); ";
            }
        }
        d.msHorizon += msSince(tHz);
    }
    else if (options_.useHorizon && s.classifierReady && !s.lastHorizon.valid)
    {
        notes += "horizon: not visible; ";
    }

    // (6) Altimeter: every SRT altitude value that has started by this frame (relative to the first sample).
    auto tAl = Clock::now();
    while (options_.useAltimeter && s.nextAltitude < s.altitudeEvents.size() &&
           s.altitudeEvents[s.nextAltitude].timeSeconds <= t + 0.5e-3)
    {
        const AltitudeSample& ev = s.altitudeEvents[s.nextAltitude];
        ++s.nextAltitude;
        try
        {
            // Applied at the estimator clock (the frame time). The SRT block start can differ from the frame time by up to
            // 0.5 ms (and by ~1e-16 s after parsing), which would otherwise be a tiny positive prediction step: the
            // process-noise factor scales as 1/sqrt(dt) and destroys the information factor (found in testing).
            // At 0.2 m/s the 0.5 ms offset is 0.1 mm, far below the altimeter sigma.
            const double tm = s.est->time();
            const double y = ev.altitudeM - s.altitudeReference;
            MeasurementAltimeter meas(tm, y, options_.altitudeSigmaM);
            const Eigen::VectorXd x = s.est->mean();
            const double innov = y - MeasurementAltimeter::predict(x);
            d.altitudeInnovation = innov;
            Eigen::MatrixXd jac = Eigen::MatrixXd::Constant(1, 1, -1.0);
            const Eigen::VectorXd nu = Eigen::VectorXd::Constant(1, innov);
            const std::vector<Eigen::Index> support = meas.supportIndices(*s.est);
            if (nisOn)
            {
                Scenario4NisRecord rec = makeNisRecord(Scenario4NisKind::Altitude, frameIndex, nu, jac, *s.est, support);
                d.altitudeNis = scenario4NisAt(rec, options_.altitudeSigmaM);
                if (options_.recordNisRecords)
                {
                    nisRecords_.push_back(std::move(rec));
                }
            }
            const OutdoorUpdateReport rep = outdoorLaplaceUpdate(meas, *s.est, FuncMinOptions(), options_.mapPolicy);
            d.altitudeOutcome = rep.outcome;
            d.altitudeNewtonNorm = rep.newtonNorm;
            mapCounts_.altitude.add(rep.outcome, !rep.updateSkipped);
            d.altitudeUpdated = d.altitudeUpdated || !rep.updateSkipped;
            if (rep.updateSkipped)
            {
                notes += std::string("altimeter: MAP update skipped (") + outcomeName(rep.outcome) + "); ";
            }
        }
        catch (const std::exception& e)
        {
            if (!d.altitudeOutcome)
            {
                d.altitudeOutcome = MapOutcome::SkippedNumericalFailure;
                mapCounts_.altitude.add(MapOutcome::SkippedNumericalFailure, false);
            }
            notes += std::string("altimeter: update skipped (") + e.what() + "); ";
        }
    }
    d.msAltitude = msSince(tAl);

    // Flow keyframe: after every flowBaselineFrames frames, the current tracks become the new baseline.
    if (s.keyframe < 0 || frameIndex - s.keyframe >= std::max(1, options_.flowBaselineFrames))
    {
        s.keyframe = frameIndex;
        s.keyPositions.clear();
        for (const ImageFlowMatch& tr : s.flow.activeTracks())
        {
            s.keyPositions[tr.id] = tr.curr;
        }
    }

    // Bookkeeping: estimate, validation-only track and yaw alignment.
    const Eigen::VectorXd xNow = s.est->mean();
    const Eigen::Vector3d rNow = xNow.segment<3>(MeasurementOutdoorFlowBundle::kCurrentOffset);
    trajectory_.push_back(rNow);
    if (static_cast<std::size_t>(frameIndex) < s.validationLocal.size() && s.validationValid[static_cast<std::size_t>(frameIndex)])
    {
        const Eigen::Vector3d g = s.validationLocal[static_cast<std::size_t>(frameIndex)];
        s.sumDot += rNow(0) * g(0) + rNow(1) * g(1);
        s.sumCross += rNow(1) * g(0) - rNow(0) * g(1);
        validationYaw_ = std::atan2(s.sumCross, s.sumDot);
        const double c = std::cos(validationYaw_);
        const double sn = std::sin(validationYaw_);
        Eigen::Vector3d v(c * g(0) - sn * g(1), sn * g(0) + c * g(1), g(2));
        validationTrack_.push_back(v);
    }
    else if (!validationTrack_.empty())
    {
        validationTrack_.push_back(validationTrack_.back());
    }

    // Overlay (input frame annotated; order: sky tint -> measured horizon -> predicted horizon -> flow -> text).
    auto tOv = Clock::now();
    if (options_.buildOverlay)
    {
        cv::Mat ov = frame.clone();
        if (s.classifierReady)
        {
            const cv::Mat skySmall = (logOdds > 0.0f);   // CV_8UC1, 255 = sky
            cv::Mat skyFull;
            cv::resize(skySmall, skyFull, s.frameSize, 0.0, 0.0, cv::INTER_NEAREST);
            cv::Mat tint(ov.size(), CV_8UC3, cv::Scalar(200, 120, 40));   // ASSUMED sky tint (BGR)
            cv::Mat blended;
            cv::addWeighted(ov, 0.7, tint, 0.3, 0.0, blended);
            blended.copyTo(ov, skyFull);
        }
        const int thick = std::max(1, s.frameSize.height / 500);
        const double rad = std::max(2.0, s.frameSize.height / 300.0);
        // measured horizon: inliers red, rejected candidates orange (ASSUMED)
        for (const cv::Point2d& p : s.lastHorizon.candidates)
        {
            cv::circle(ov, p, static_cast<int>(rad), cv::Scalar(0, 140, 255), cv::FILLED);
        }
        for (const cv::Point2d& p : s.lastHorizon.inliers)
        {
            cv::circle(ov, p, static_cast<int>(rad * 1.5), cv::Scalar(0, 0, 255), cv::FILLED);
        }
        // predicted horizon from the posterior attitude: yellow (ASSUMED)
        {
            const Eigen::Vector3d theta = xNow.segment<3>(MeasurementHorizon::kThetaOffset);
            const Eigen::Vector3d a = MeasurementHorizon::horizonNormal(theta, s.Rbc);
            std::vector<cv::Point> poly;
            const double fx = s.camera.fx();
            const double fy = s.camera.fy();
            for (int u = 0; u < s.frameSize.width; u += 16)
            {
                if (std::abs(a(1)) < 1e-9)
                {
                    break;
                }
                const double xn = (u - s.camera.cx()) / fx;
                const double vDouble = s.camera.cy() - fy / a(1) * (a(0) * xn + a(2));
                const double vClamped = std::clamp(vDouble, -4.0 * s.frameSize.height, 5.0 * s.frameSize.height);
                poly.emplace_back(u, static_cast<int>(std::lround(vClamped)));
            }
            if (poly.size() >= 2)
            {
                cv::polylines(ov, poly, false, cv::Scalar(0, 255, 255), thick + 1, cv::LINE_AA);
            }
        }
        // flow tracks: plane inliers green, rejected magenta (ASSUMED)
        for (std::size_t i = 0; i < s.lastMatches.size(); ++i)
        {
            const ImageFlowMatch& m = s.lastMatches[i];
            const bool in = s.lastInlier.size() == s.lastMatches.size() && s.lastInlier[i] != 0;
            const cv::Scalar colour = in ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 255);
            cv::line(ov, m.prev, m.curr, colour, thick, cv::LINE_AA);
            cv::circle(ov, m.curr, static_cast<int>(rad), colour, cv::FILLED);
        }
        // text (white with black outline, ASSUMED)
        std::vector<std::string> lines;
        lines.push_back(std::format("frame {:5d}  t={:8.3f} s", frameIndex, t));
        lines.push_back(std::format("h_rel {:7.2f} m (est)  Hg {:6.2f} m", -xNow(MeasurementAltimeter::kZIndex), groundDepth_));
        lines.push_back(std::format("flow {}/{} pairs  horizon {}  tracks {}", d.flowInliers, d.flowMatches,
                                    s.lastHorizonUsed ? "on" : "off", d.trackerActive));
        // Status panel: the top-left block is darkened (ASSUMED style), then white text on it.
        const double fontScale = std::max(0.6, s.frameSize.height / 900.0);
        const int lineStep = static_cast<int>(std::lround(48.0 * fontScale));
        const int panelW = std::min(ov.cols, static_cast<int>(std::lround(900.0 * fontScale)));
        const int panelH = std::min(ov.rows, lineStep * static_cast<int>(lines.size()) + 12);
        cv::Mat panel = ov(cv::Rect(0, 0, panelW, panelH));
        panel.convertTo(panel, -1, 0.4, 0.0);
        for (std::size_t i = 0; i < lines.size(); ++i)
        {
            const cv::Point org(16, lineStep * static_cast<int>(i + 1));
            cv::putText(ov, lines[i], org, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        s.overlay = std::move(ov);
        s.statusLines = std::move(lines);
    }
    d.msOverlay = msSince(tOv);

    s.havePrevious = true;
    s.lastFrameIndex = frameIndex;
    d.notes = notes;
    d.msTotal = msSince(tFrame);
    diagnostics_.push_back(d);

    // Non-finite guard: a NaN in the state is a hard failure (estimator divergence).
    if (!s.est->mean().allFinite() || !finite(s.est->state().infoVec()))
    {
        throw std::runtime_error("Scenario4Runner: non-finite estimator state at frame " + std::to_string(frameIndex));
    }
}

PlotFrameData Scenario4Runner::plotData() const
{
    const Impl& s = *impl_;
    PlotFrameData out;
    if (!s.initialised)
    {
        return out;
    }
    out.overlay = s.overlay;
    const Eigen::VectorXd x = s.est->mean();
    const Eigen::Vector3d rBN = x.segment<3>(MeasurementOutdoorFlowBundle::kCurrentOffset);
    const Eigen::Vector3d theta = x.segment<3>(MeasurementOutdoorFlowBundle::kCurrentOffset + 3);
    out.rCNn = rBN;
    out.Rnc = rpy2rot(theta) * s.Rbc;
    const GaussianInfo<double> pos = s.est->state().marginal({6, 7, 8});
    out.positionSqrtCov = pos.sqrtCov();
    out.trajectory = trajectory_;
    out.validationTrack = validationTrack_;
    out.statusLines = s.statusLines;
    return out;
}

std::unique_ptr<ScenarioRunner> makeScenario4Runner()
{
    return std::make_unique<Scenario4Runner>();
}
