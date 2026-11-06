#ifndef SCENARIO4RUNNER_H
#define SCENARIO4RUNNER_H

#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "ImageFlow.h"
#include "MeasurementBase.h"
#include "ScenarioRunner.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"
#include "VisualHorizon.h"

/**
 * @brief Tunable parameters of the outdoor (Scenario 4) estimator.
 *
 * Units: velocities m/s, angular rates rad/s, angles rad, lengths m, image noise px (full resolution).
 * Values marked "tuned" were chosen from the NIS statistics of the full flight (see engineering notes);
 * all others are design choices.
 */
struct Scenario4Options
{
    /// Directory with the horizon training patches sky*.png / ground*.png. Empty = directory of the video.
    std::filesystem::path trainingDirectory;
    /// SRT file. Empty = <video stem>.SRT next to the video (as in visualNavigation.cpp).
    std::filesystem::path srtPath;
    /// In-memory horizon training patches (BGR8). Used instead of the sky*/ground* files when both lists are non-empty.
    std::vector<cv::Mat> trainingSky;
    std::vector<cv::Mat> trainingGround;

    bool useFlow = true;        ///< enable the optical-flow bundle update
    bool useHorizon = true;     ///< enable the visual-horizon update
    bool useAltimeter = true;   ///< enable the altimeter update

    // Process noise: continuous-time PSD of the random walk on [vBNb; omegaBNb].
    double nuPsdVelocity = 1.0;     ///< (m/s)^2/s, tuned
    double nuPsdAngularRate = 0.05; ///< (rad/s)^2/s, tuned

    // Prior at the first frame (world origin = initial body position, heading prior = 0).
    double priorVelocitySigma = 5.0;      ///< m/s
    double priorAngularRateSigma = 0.5;   ///< rad/s
    double priorPositionSigmaXY = 1.0;    ///< m
    double priorPositionSigmaZ = 0.1;     ///< m (altimeter reference)
    double priorRollPitchSigma = 0.1;     ///< rad
    double priorYawSigma = 0.5;           ///< rad (heading is unobservable apart from this prior)

    // Measurement noise (R5-FIXA). Both sigmas were calibrated by NIS-median matching on the FIRST HALF of the full flight
    // (frames 0..4579) of the reference run (previous values 1.0 px and 0.02 m, NIS records on): flow 0.08261 px, altitude
    // 0.06556 m (procedure: choose sigma such that the median over updates of P(chi2_dof <= NIS_i) equals 0.5). Neither
    // calibrated value is adopted, because the out-of-sample (second-half) check of the closed-loop run rejects them:
    //   - altitude 0.06556 m: out-of-sample NIS/dof median 2.44 (0.02 m: 1.75; chi2(1) median 0.455), and the altitude
    //     error at SRT change instants rises from 0.023 m to 0.121 m RMSE (SRT is the measurement, so this is a fit metric);
    //   - flow 0.0826 px: the synthetic flight of the Scenario 4 regression test misses its 1.5 m horizontal tolerance
    //     (error 2.7 m), because the Lucas-Kanade errors of ~400 features are correlated, not independent.
    // Both values therefore stay at the previous ones, which are NOT NIS-consistent (flow NIS/dof median ~0.008,
    // altitude NIS mean ~4). Evidence: out/scenario4_r5_refA_flight_run_summary.txt (calibration lines),
    // out/scenario4_r5_altcal_flight_run_summary.txt (altitude 0.06556 m run). The flow 0.0826 px rejection evidence is the
    // Scenario 4 synthetic-flight regression test (no full-flight summary file was kept).
    double flowSigmaPx = 1.0;          ///< NOT NIS-calibrated (see above; calibrated candidate 0.0826 px rejected)
    double horizonSigmaPx = 5.0;       ///< PROVISIONAL: refined boundary-point residual RMS on 5 real frames was 2.3-5.1 px
                                       ///< (median ~4.4 px; the real horizon is not a straight line at pixel level).
                                       ///< On the synthetic ideal line the residual is ~0.05 px. Was 8 px before the
                                       ///< P2 fixes (the classifier boundary was 2-5 px biased).
    double altitudeSigmaM = 0.02;      ///< ASSUMED (S2-CAM); NIS-calibrated candidate 0.06556 m rejected (see above)

    // Flow front-end.
    ImageFlowOptions flow;             ///< tracker options (scale 1/2 by default)
    double flowRansacPx = 2.0;         ///< plane (homography) RANSAC threshold on full-resolution pixels
    /// Flow baseline in frames. The flow is measured between a keyframe and the current frame (etaPrev is the pose
    /// at the keyframe), because consecutive-frame translation flow at 60 fps is a few tenths of a pixel, where
    /// Lucas-Kanade shrinks the displacement toward zero (measured: velocity underestimated by ~2.5x).
    int flowBaselineFrames = 5;
    int minFlowPairs = 12;             ///< minimum inlier correspondences for a flow update
    double minAglForFlowM = 5.0;       ///< skip flow if the predicted height above the plane is lower

    // Horizon front-end.
    double horizonAnalysisScale = 0.25; ///< classification resolution (1/4 of the input)
    HorizonExtractionOptions horizon;

    /// Ground plane height Hg in the world z-down frame [m]. Negative = use the first SRT altitude (ASSUMED).
    double groundDepthM = -1.0;

    /// Render the overlay (input frame annotation). Disable for timing-only runs.
    bool buildOverlay = true;

    /// MAP status policy applied to every Laplace update (flow, horizon, altimeter). See MapAcceptancePolicy.
    MapAcceptancePolicy mapPolicy;

    /**
     * @brief Compute the per-update NIS diagnostics (default on).
     *
     * The NIS of an update is evaluated with a QR factorisation of the support square-root information and triangular
     * solves only (no covariance or information inversion). It costs one small QR per update and can be switched off
     * for timing-only runs; the estimator itself never depends on it.
     */
    bool computeNis = true;

    /// Keep one Scenario4NisRecord per NIS evaluation (for offline sigma calibration and out-of-sample checks).
    bool recordNisRecords = false;
};

/** @brief Measurement family of a NIS statistic. */
enum class Scenario4NisKind
{
    Flow,      ///< optical-flow bundle (dof = 2 N)
    Horizon,   ///< visual horizon (dof = number of boundary points)
    Altitude   ///< altimeter (dof = 1)
};

/**
 * @brief Prior-predictive innovation statistic of one update, stored so that the NIS can be evaluated at any sigma.
 *
 * With the support prior covariance P_S = Xi^{-1} Xi^{-T} (Xi = square-root information of the support marginal),
 * the predictive innovation covariance is S(sigma) = sigma^2 I_m + B B^T with B = J Xi^{-1} (m x k). Only the
 * k x k factor R_B (R_B^T R_B = B^T B), c = B^T nu (k) and ||nu||^2 are needed:
 *   NIS(sigma) = (||nu||^2 - ||R_sigma^{-T} c||^2) / sigma^2,   R_sigma = qr([R_B; sigma I_k]).
 * Innovations and B do not depend on sigma (sigma only enters through the update), so scenario4NisAt() reproduces the
 * in-run NIS exactly and allows the offline calibration in scenario4CalibrateSigma().
 */
struct Scenario4NisRecord
{
    Scenario4NisKind kind = Scenario4NisKind::Flow;
    int frameIndex = -1;
    int dof = 0;             ///< measurement dimension m
    double nu2 = 0.0;        ///< ||nu||^2 of the prior innovation (normalised units for flow and horizon)
    Eigen::MatrixXd rB;      ///< k x k upper-triangular factor of B
    Eigen::VectorXd c;       ///< B^T nu (k)
};

/**
 * @brief NIS of a record at noise standard deviation sigma (same units as the record), by QR and triangular solve.
 * @throws std::invalid_argument if sigma is not positive and finite
 */
double scenario4NisAt(const Scenario4NisRecord& record, double sigma);

/** @brief Result of scenario4CalibrateSigma. */
struct Scenario4Calibration
{
    double sigma = std::numeric_limits<double>::quiet_NaN();   ///< calibrated sigma (units of the records)
    std::size_t records = 0;                                    ///< records used
    double medianNisPerDof = std::numeric_limits<double>::quiet_NaN();   ///< median of NIS/dof at sigma
    double medianPit = std::numeric_limits<double>::quiet_NaN();         ///< median of P(chi2_dof <= NIS) at sigma (target 0.5)
    bool converged = false;                                     ///< the target was bracketed and reached
};

/**
 * @brief Robust NIS-matching calibration of a noise sigma on recorded statistics.
 *
 * Chooses sigma such that the median over the selected records of the probability-integral transform
 * P(chi2_dof <= NIS_i(sigma)) equals 0.5, i.e. the median NIS equals the median of its chi-squared reference
 * distribution (dof-aware: each record uses its own dof). The median makes the choice insensitive to outlier frames;
 * the statistic is monotone in sigma, so the root is found by bisection on log(sigma) in [sigmaLo, sigmaHi].
 * @param records     recorded statistics (records of other kinds are ignored)
 * @param kind        measurement family to calibrate
 * @param frameBegin  first frame index included (inclusive)
 * @param frameEnd    end frame index (exclusive); use the first half of the flight for calibration
 * @param sigmaLo     lower bracket (> 0), sigmaHi upper bracket (> sigmaLo)
 */
Scenario4Calibration scenario4CalibrateSigma(const std::vector<Scenario4NisRecord>& records, Scenario4NisKind kind,
                                             int frameBegin, int frameEnd, double sigmaLo, double sigmaHi);

/** @brief Counts of MAP outcomes per measurement family over a run. */
struct Scenario4MapCounts
{
    MapStatusCounts flow;
    MapStatusCounts horizon;
    MapStatusCounts altitude;
};

/** @brief Per-frame diagnostics (for logs and tests). NaN means "not computed". */
struct Scenario4FrameDiagnostics
{
    int frameIndex = -1;
    double time = 0.0;
    bool flowUpdated = false;
    int flowMatches = 0;        ///< tracked correspondences between frames k-1 and k
    int flowInliers = 0;        ///< plane-consistent correspondences used by the update
    double flowNis = std::numeric_limits<double>::quiet_NaN();
    int flowDof = 0;
    bool horizonValid = false;
    bool horizonUpdated = false;
    int horizonCandidates = 0;
    int horizonInliers = 0;
    double horizonSkyFraction = 0.0;
    double horizonNis = std::numeric_limits<double>::quiet_NaN();
    int horizonDof = 0;
    double flowResidualRmsPx = std::numeric_limits<double>::quiet_NaN();    ///< post-update RMS of the plane-flow residuals [px]
    double horizonResidualRmsPx = std::numeric_limits<double>::quiet_NaN(); ///< post-update RMS of the horizon residuals [px]
    bool altitudeUpdated = false;
    double altitudeInnovation = std::numeric_limits<double>::quiet_NaN(); ///< y - h(prior) [m]
    double altitudeNis = std::numeric_limits<double>::quiet_NaN();
    int trackerActive = 0;
    /// MAP outcome of each update attempted on this frame (empty = that update was not attempted).
    std::optional<MapOutcome> flowOutcome;
    std::optional<MapOutcome> horizonOutcome;
    std::optional<MapOutcome> altitudeOutcome;
    double flowNewtonNorm = std::numeric_limits<double>::quiet_NaN();     ///< Newton norm at the flow MAP iterate
    double horizonNewtonNorm = std::numeric_limits<double>::quiet_NaN();  ///< Newton norm at the horizon MAP iterate
    double altitudeNewtonNorm = std::numeric_limits<double>::quiet_NaN(); ///< Newton norm at the altitude MAP iterate
    double msFlow = 0.0;        ///< tracking + flow update + robust fit
    double msHorizon = 0.0;     ///< classification + extraction + horizon update
    double msAltitude = 0.0;    ///< altimeter updates
    double msFilter = 0.0;      ///< predictDelayed + estimator bookkeeping (not inside the other stages)
    double msTotal = 0.0;       ///< processFrame wall time (excluding plotData/overlay if buildOverlay is off)
    double msOverlay = 0.0;     ///< overlay rendering
    std::string notes;          ///< skipped stages and reasons
};

/**
 * @brief Scenario 4 (outdoor flight) runner: optical-flow bundle + visual horizon + altimeter in one
 *        square-root information filter (Laplace information filter, see architecture notes).
 *
 * Frame loop (per processFrame call, frame k at time t_k = k / fps):
 *   1. horizon classification and boundary extraction on the 1/4-scale frame (ground mask for the tracker),
 *   2. feature tracking (ImageFlow, scale 1/2) from frame k-1 to k,
 *   3. time update with the delayed-pose shift (etaPrev <- eta_{k-1}),
 *   4. flow bundle update (plane homography, RANSAC-selected inliers),
 *   5. horizon update (roll and pitch),
 *   6. altimeter updates for every SRT altitude change with block time <= t_k (relative altitude).
 *
 * Latitude/longitude are never used by steps 1-6. They are read only for PlotFrameData::validationTrack.
 */
class Scenario4Runner : public ScenarioRunner
{
public:
    /** @brief Creates a runner with the given options (no I/O until initialise()). */
    explicit Scenario4Runner(Scenario4Options options = {});
    ~Scenario4Runner() override;

    Scenario4Runner(const Scenario4Runner&) = delete;
    Scenario4Runner& operator=(const Scenario4Runner&) = delete;

    void initialise(const Camera& camera, double fps, cv::Size frameSize, const std::string& videoPath) override;
    void processFrame(const cv::Mat& frame, int frameIndex, double t) override;
    PlotFrameData plotData() const override;
    std::string name() const override { return "Scenario 4: flight"; }

    /** @brief Diagnostics of every processed frame, oldest first. */
    const std::vector<Scenario4FrameDiagnostics>& diagnostics() const { return diagnostics_; }

    /** @brief Current estimator. @throws std::logic_error before initialise(). */
    const SystemEstimator& estimator() const;

    /** @brief Options in use. */
    const Scenario4Options& options() const { return options_; }

    /** @brief True when the horizon classifier was trained (sky/ground patches found) and horizon updates are enabled. */
    bool horizonAvailable() const;

    /** @brief Ground plane height Hg actually used [m] (after the default was resolved). */
    double groundDepth() const { return groundDepth_; }

    /** @brief Estimated camera position per processed frame (world n). */
    const std::vector<Eigen::Vector3d>& trajectory() const { return trajectory_; }

    /** @brief Validation-only: GPS-derived local track (aligned in yaw to the estimate so far), per frame. */
    const std::vector<Eigen::Vector3d>& validationTrack() const { return validationTrack_; }

    /** @brief Validation-only: current yaw alignment angle applied to the GPS track [rad]. */
    double validationYaw() const { return validationYaw_; }

    /** @brief NIS records (empty unless options().recordNisRecords), in processing order. */
    const std::vector<Scenario4NisRecord>& nisRecords() const { return nisRecords_; }

    /** @brief MAP outcome counts of the run so far (flow, horizon, altitude). */
    const Scenario4MapCounts& mapCounts() const { return mapCounts_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Scenario4Options options_;
    std::vector<Scenario4FrameDiagnostics> diagnostics_;
    std::vector<Eigen::Vector3d> trajectory_;
    std::vector<Eigen::Vector3d> validationTrack_;
    double validationYaw_ = 0.0;
    double groundDepth_ = 0.0;
    std::vector<Scenario4NisRecord> nisRecords_;
    Scenario4MapCounts mapCounts_;
};

#endif
