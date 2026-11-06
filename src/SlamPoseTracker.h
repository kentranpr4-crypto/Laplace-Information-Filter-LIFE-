#ifndef SLAMPOSETRACKER_H
#define SLAMPOSETRACKER_H

/**
 * @file SlamPoseTracker.h
 * @brief Frame-to-frame SLAM with identical-ID ArUco tags (Scenario 5): information-form estimator, delayed previous
 *        pose, geometric association, correlated birth, culling.
 *
 * Estimator state (SystemVisualNav, pose-landmark layout):
 *   [nu (6) | eta = [rBNn; Thetanb] (6) | etaPrev (6) | L_1 ... L_M (6 each, L = [r_jn^n; Theta_nj])]
 * The world frame is the body frame of the first processed frame (gauge fixed by a 1 mm / 1 mrad prior, as in S6).
 *
 * Per frame k (time t_k):
 *   1. Detect ArUco markers, keep id 5 (ArucoTags.h). Track ImageFlow features from frame k-1 to k.
 *   2. Delayed time update: etaPrev <- eta_{k-1} and eta <- propagated (SystemVisualNav::predictDelayed).
 *   3. Depth-free flow measurement linking etaPrev and eta (MeasurementIndoorFlow, gated by IndoorFlowFrontEnd).
 *   4. Predict every tag landmark (corners, innovation covariance from one union marginal) (SlamPoseAssociation.h).
 *      Visibility is occlusion-aware: a candidate inside the quad of a nearer candidate is not expected to be detected.
 *   5. Geometric association, ambiguity rejection, conflicts, birth candidates.
 *   6. Laplace/MAP update with all associated detections (MeasurementTagBundle, support eta + associated tags).
 *   7. Record bookkeeping (hits, misses, confirmation) from the pre-update predictions.
 *   8. Culling (SlamPoseTracker::selectCulls). Culled tags are archived with their last density.
 *   9. Births: candidates are kept pending until seen in minObservationsForBirth frames, the IPPE hypothesis
 *      consistent across those frames is chosen. Before a landmark is added, (a) a birth at an active tag is a
 *      duplicate and is dropped; (b) a birth compatible with an archived tag re-attaches that tag's identity
 *      (re-identification, chi2 gate with dof 6); otherwise a new identity is created. The landmark is added with
 *      addLandmarkLinear.
 *
 * Identity. Every landmark block of the estimator has a record with an identity (lineage) id. Re-births of a tag that
 * was culled keep its identity, so the number of identities is the number of physical tags seen.
 *
 * Tunable values are in SlamPoseConfig (ASSUMED or TUNED, see the field comments and the Scenario 5 report).
 */

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#include "ArucoTags.h"
#include "Camera.h"
#include "ImageFlow.h"
#include "IndoorFlowFrontEnd.h"
#include "MeasurementBase.h"
#include "SlamPoseAssociation.h"
#include "SlamPoseInitialisation.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

namespace slam
{

/** @brief Configuration of the Scenario 5 SLAM. */
struct SlamPoseConfig
{
    // ---- tag geometry ----
    double tagSideM = 0.15;                       ///< ASSUMED outer side of the black marker border [m]; metric scale is proportional to it
    tags::ArucoTagDetector::Options detector{};   ///< DICT_6X6_50, id 5 (verified empirically)

    // ---- measurement noise and gating ----
    double pixelSigmaPx = 0.629427;               ///< CALIBRATED (R4-10, procedure below): 1 std per corner component [px].
                                                  ///< Calibration: tracker run with sigma0 = 1.0 over the FIRST HALF of tag_identical.MOV (frames 0-536),
                                                  ///< moment estimate sigma^2 = mean(e_k^2) - mean(predicted state variance of corner k) over every matched
                                                  ///< corner (pre-update innovations, 13192 components): sigma^2 = 0.396178 px^2. Raw RMS 0.857 px and a MAD-based
                                                  ///< value 0.729 px are reported for comparison and not adopted. Validation on the SECOND half (out-of-sample):
                                                  ///< NIS/dof mean 1.063, median 0.977, 82.9 % inside the 95 % band (sigma 1.0: 59.0 %).
    double gateNSigma = 3.0;                      ///< chi2 gate with GaussianBase::mahalanobisThresholdSq(nSigma, 8)
    double ambiguityMarginD2 = 4.0;               ///< ASSUMED: second gated candidate within this d^2 makes a detection ambiguous
    double nearCentreFactor = 1.0;                ///< ASSUMED: unmatched detection closer than this * side to a predicted centre is not a birth
    int minMatchesForUpdate = 1;                  ///< ASSUMED: minimum associated tags for a measurement update

    // ---- prediction visibility (see TagPredictionOptions) ----
    double candidateMarginPx = 100.0;             ///< ASSUMED
    double visibleMarginPx = 5.0;                 ///< ASSUMED
    double minVisibleSidePx = 20.0;               ///< ASSUMED: predicted tags smaller than this are not expected to be detected
    double minFacingCos = 0.3;                    ///< ASSUMED: tags viewed at more than ~73 deg are not expected to be detected
    double minDepth = 0.2;                        ///< ASSUMED [m]
    double maxDepth = 30.0;                       ///< ASSUMED [m]

    // ---- births (pending until seen in several frames) ----
    int minObservationsForBirth = 2;              ///< ASSUMED: frames of consistent detection before a tag is born
    int minBaselineFrames = 5;                    ///< ASSUMED: the pending observations must span this many frames so the
                                                  ///< camera has moved enough to separate the two IPPE hypotheses
    double decisiveMarginM = 0.02;                ///< ASSUMED: the chosen IPPE hypothesis must beat the other by this consistency margin [m]
                                                  ///< to be born before minBaselineFrames (a one-solution IPPE result is always decisive)
    int maxPendingAgeFrames = 3;                  ///< ASSUMED: a pending tag not seen for this many frames is dropped
    double pendingMatchPx = 40.0;                 ///< ASSUMED: centre distance to match a detection to a pending tag [px] (or 1.2 side)
    int maxTags = 60;                             ///< ASSUMED capacity (dimension 6 x 60)
    Eigen::Matrix<double, 6, 1> birthPriorSigma =
        (Eigen::Matrix<double, 6, 1>() << 10.0, 10.0, 10.0, 1.0, 1.0, 1.0).finished();  ///< ASSUMED weak prior on the tag-in-camera xi

    // ---- lifecycle ----
    int confirmHits = 3;                          ///< ASSUMED: associated frames before a tag is confirmed
    int unconfirmedMaxMisses = 3;                 ///< ASSUMED: unconfirmed tag removed after this many visible misses
    int confirmedMaxMisses = 60;                  ///< ASSUMED (2 s): confirmed tag removed after this many consecutive visible misses (occluded candidates are not counted)
    int maxUnseenFrames = 300;                    ///< ASSUMED (10 s): any tag not associated for this long is removed
    int maxBehindFrames = 3;                      ///< ASSUMED: tag behind the camera for this long is removed
    bool reidentify = true;                       ///< culled tags are archived; a birth compatible with an archived tag keeps its identity
    int archiveMaxEntries = 200;                  ///< ASSUMED capacity of the archive (oldest entries are dropped)
    int archiveMaxAgeFrames = 1000;               ///< ASSUMED (~33 s at 30 fps): archived tags older than this are forgotten
    double reidFloorPositionM = 0.05;             ///< ASSUMED: position std floor added to the archived covariance in the re-identification gate [m]
    double reidFloorAngleRad = 0.10;              ///< ASSUMED: orientation std floor added to the archived covariance [rad]
    double duplicateCentreFactor = 0.5;           ///< ASSUMED: a birth closer than this * side to an active tag, with a similar normal, is a duplicate
    double duplicateMaxAngleRad = 0.35;           ///< ASSUMED: normal-angle limit of the duplicate test [rad] (20 deg)

    // ---- optical flow (depth-free odometry between etaPrev and eta; MeasurementIndoorFlow, IndoorFlowFrontEnd) ----
    bool useFlow = true;                          ///< add the flow measurement (ablation switch)
    IndoorFlowOptions flow{};                     ///< flow front end options (ASSUMED values, see the header)

    // ---- process model and initial density (inherited from Scenario 6, TUNED on room.MOV, not re-tuned here yet) ----
    Eigen::Matrix<double, 6, 1> nuPsd =
        (Eigen::Matrix<double, 6, 1>() << 0.02, 0.02, 0.02, 0.2, 0.2, 0.2).finished();  ///< random-walk PSD of [v; omega]
    double initialVelocitySigma = 0.3;            ///< ASSUMED 1 std [m/s]
    double initialAngularRateSigma = 0.3;         ///< ASSUMED 1 std [rad/s]
    double initialPoseSigma = 1e-3;               ///< gauge: start pose fixed to ~1 mm / 1 mrad

    // ---- MAP ----
    double mapGradientTolerance = 1e-6;           ///< MAP stop criterion (information units)
};

/** @brief Per-tag record (the geometry lives in the estimator). */
struct SlamPoseRecord
{
    Eigen::Index id = -1;                         ///< SystemVisualNav landmark id
    int qIndex = 0;                               ///< in-plane rotation of the landmark frame (see SlamPoseInitialisation.h)
    Eigen::Matrix<double, 4, 3> cornersLocal = Eigen::Matrix<double, 4, 3>::Zero();
    int birthFrame = 0;
    int identity = -1;                            ///< lineage id of the physical tag (kept across re-births)
    int reidentifications = 0;                    ///< number of times this record was re-attached to an archived identity
    int lastSeenFrame = -1;
    int hits = 0;                                 ///< associated frames so far
    int misses = 0;                               ///< consecutive visible frames without an associated detection
    int behindFrames = 0;                         ///< consecutive frames with the tag centre behind the camera
    bool confirmed = false;
    bool associatedThisFrame = false;
    Eigen::Vector2d lastPixel = Eigen::Vector2d::Zero();  ///< centre of the last associated detection [px]
};

/** @brief A culled tag kept for re-identification: identity, landmark frame and the density at the time of the cull. */
struct SlamPoseArchiveEntry
{
    int identity = -1;                            ///< lineage id of the physical tag
    int qIndex = 0;                               ///< in-plane rotation index of the landmark frame
    Eigen::Matrix<double, 4, 3> cornersLocal = Eigen::Matrix<double, 4, 3>::Zero();
    Eigen::Matrix<double, 6, 1> mean = Eigen::Matrix<double, 6, 1>::Zero();        ///< [r_jn^n; Theta_nj] at the cull
    Eigen::Matrix<double, 6, 6> sqrtInfo = Eigen::Matrix<double, 6, 6>::Zero();    ///< upper Xi of the marginal, Lambda = Xi^T Xi (no covariance formed)
    int lastSeenFrame = -1;
    int culledFrame = -1;
};

/**
 * @brief Squared Mahalanobis distance of a candidate tag pose from an archived pose (re-identification gate).
 *
 * delta = [r_c - r_m ; rotation vector of R_m^T R_c], P = S^T S + diag(floorPos^2 (3), floorAngle^2 (3)),
 * d^2 = delta^T P^{-1} delta. Evaluated as d^2 = ||R^{-T} delta||^2 where R is the upper factor of the stacked
 * matrix [S; diag(floors)] (Q-less Householder QR, R^T R = P) followed by one triangular solve. No covariance is
 * formed by a product or inverted, and no Cholesky factor of P is computed.
 * @param meanPose   archived [r; Theta] (ZYX Euler of the landmark frame)
 * @param sqrtCov    archived upper square-root covariance S (6 x 6), P = S^T S
 * @param candPose   candidate [r; Theta]
 * @return d^2, or +infinity if the factor is singular
 */
double tagArchiveDistanceSq(const Eigen::Matrix<double, 6, 1>& meanPose, const Eigen::Matrix<double, 6, 6>& sqrtCov,
                            const Eigen::Matrix<double, 6, 1>& candPose, double floorPositionM, double floorAngleRad);

/**
 * @brief Same gate as tagArchiveDistanceSq, from the archived square-root INFORMATION factor Xi (Lambda = Xi^T Xi).
 *
 * With v = Xi delta and B = [I; diag(floors) Xi^T] = Q_B R_B, the identity (P + F)^{-1} = Xi^T (I + Xi F Xi^T)^{-1} Xi
 * gives d^2 = v^T (B^T B)^{-1} v = ||R_B^{-T} v||^2. B^T B = I + Xi F Xi^T is at least the identity, so the factor is
 * always well conditioned. This is the production path: the archive stores Xi, not a covariance.
 * @param sqrtInfo   archived upper square-root information (6 x 6)
 */
double tagArchiveDistanceSqInfo(const Eigen::Matrix<double, 6, 1>& meanPose, const Eigen::Matrix<double, 6, 6>& sqrtInfo,
                                const Eigen::Matrix<double, 6, 1>& candPose, double floorPositionM, double floorAngleRad);

/** @brief A tag seen in consecutive frames but not yet born (ambiguity resolution across frames). */
struct SlamPosePending
{
    struct Observation
    {
        int frame = 0;
        Eigen::Matrix<double, 8, 1> pixels = Eigen::Matrix<double, 8, 1>::Zero();
        std::vector<TagPoseHypothesis> hyps;      ///< camera-frame hypotheses (best first)
        std::vector<Eigen::Matrix3d> Rnt;         ///< world rotation of each hypothesis at this frame
        std::vector<Eigen::Vector3d> rnt;         ///< world tag centre of each hypothesis at this frame
    };
    std::vector<Observation> obs;
    int lastFrame = -1;
    Eigen::Vector2d lastCentre = Eigen::Vector2d::Zero();
    double lastSidePx = 0.0;
};

/** @brief Per-frame diagnostics. NaN where not applicable. */
struct SlamPoseFrameStats
{
    int frame = 0;
    double time = 0.0;
    int detections = 0;               ///< id-5 detections
    int candidates = 0;               ///< predicted tags in the candidate set
    int visibleCandidates = 0;        ///< strictly visible predicted tags
    int matched = 0;                  ///< associated detection-tag pairs
    int ambiguous = 0;                ///< detections rejected as ambiguous
    int conflicts = 0;                ///< detections that lost a tag to a better detection
    int births = 0;                   ///< tags born this frame
    int pending = 0;                  ///< tags pending (seen, not yet born)
    int culled = 0;                   ///< tags removed this frame
    int tags = 0;                     ///< tags in the state after the frame
    double nis = 0.0;                 ///< joint NIS of the matched set, before the update
    int nisDof = 0;                   ///< 8 x matched
    int mapStatus = -1;               ///< FuncMinStatus of the tag MAP step (-1 if no update)
    int mapIterations = 0;
    int mapOutcome = -1;              ///< MapOutcome (MeasurementBase.h) of the tag update: policy decision (-1 if no update)
    bool updateSkipped = false;       ///< tag update skipped by the MAP policy: the state was kept at the prediction
    double newtonNorm = std::numeric_limits<double>::quiet_NaN();   ///< Newton norm of the tag MAP at its returned iterate
    int flowMapStatus = -1;           ///< FuncMinStatus of the flow MAP step (-1 if no flow update)
    int flowMapOutcome = -1;          ///< MapOutcome of the flow update (-1 if no flow update)
    bool flowUpdateSkipped = false;   ///< flow update skipped by the MAP policy
    bool updated = false;
    bool precisionExact = true;
    double msDetect = 0.0;            ///< ArUco detection [ms]
    int flowInput = 0;                ///< pixel correspondences from ImageFlow
    bool flowValid = false;           ///< flow measurement used this frame
    bool flowRotationModel = false;   ///< Model R (pure rotation) selected; false = Model E (epipolar)
    int flowPairs = 0;                ///< correspondences in the measurement (after RANSAC and the prior gate)
    double flowNis = 0.0;             ///< whitened flow residual squared at the prior (before the update)
    int flowDof = 0;                  ///< residual dimension of the flow measurement
    bool flowUpdated = false;         ///< flow MAP update applied
    int occluded = 0;                 ///< predicted tags not expected to be detected (occluded by a nearer tag)
    int reidentified = 0;             ///< births that re-attached an archived identity
    int newIdentities = 0;            ///< births that created a new identity
    int duplicatesSuppressed = 0;     ///< births dropped because an active tag is at the same place
    int archived = 0;                 ///< culled tags stored in the archive
    double msFlow = 0.0;              ///< flow tracking and measurement [ms]
    double msPredict = 0.0;           ///< delayed time update and candidate prediction [ms]
    double msAssociate = 0.0;         ///< gating and assignment [ms]
    double msUpdate = 0.0;            ///< MAP and Laplace update [ms]
    double msBirthCull = 0.0;         ///< records, culling and births [ms]
    double msTotal = 0.0;             ///< whole step [ms]
};

/**
 * @brief The Scenario 5 SLAM (see the file documentation).
 */
class SlamPoseTracker
{
public:
    /**
     * @param camera  calibrated camera (copied)
     * @param cfg     configuration (copied)
     * @throws std::invalid_argument for an invalid camera or inconsistent configuration
     */
    SlamPoseTracker(const Camera& camera, const SlamPoseConfig& cfg);

    /**
     * @brief Processes one grey frame at time t (s, non-decreasing).
     * @throws std::invalid_argument for an invalid image or decreasing time
     */
    SlamPoseFrameStats step(const cv::Mat& gray, int frameIndex, double t);

    /** @brief Estimator (state density and time). */
    const SystemEstimator& estimator() const { return *est_; }
    /** @brief Model with the landmark layout. */
    const SystemVisualNav& model() const { return *model_; }
    /** @brief Camera. */
    const Camera& camera() const { return camera_; }
    /** @brief Configuration. */
    const SlamPoseConfig& config() const { return cfg_; }
    /** @brief Records of the tags in the state. */
    const std::vector<SlamPoseRecord>& records() const { return records_; }
    /** @brief Tags seen but not yet born. */
    const std::vector<SlamPosePending>& pending() const { return pending_; }
    /** @brief Camera positions of every processed frame (estimate), oldest first. */
    const std::vector<Eigen::Vector3d>& trajectory() const { return trajectory_; }
    /**
     * @brief Number of frames whose MAP update (tag or flow) was not usable: a numerical failure, a non-finite result,
     * or a MaxIterations result the policy did not accept. The prediction is kept for these frames. A no-information
     * update (SkippedNoCurvature) is not counted here; see tagMapStatusCounts() / flowMapStatusCounts().
     */
    int mapFailures() const { return mapFailures_; }
    /** @brief MAP outcome counters of the tag (pose-landmark) updates over the run (R4-01). */
    const MapStatusCounts& tagMapStatusCounts() const { return tagMapCounts_; }
    /** @brief MAP outcome counters of the flow updates over the run (R4-01). */
    const MapStatusCounts& flowMapStatusCounts() const { return flowMapCounts_; }
    /** @brief Archived (culled, not yet re-attached) tags. */
    const std::vector<SlamPoseArchiveEntry>& archive() const { return archive_; }
    /** @brief Number of identities created so far (lineage ids 0 .. n-1). */
    int identitiesCreated() const { return nextIdentity_; }

    /** @brief Upper sqrt factor (P = S^T S) of the body position covariance (rCBb = 0 for the indoor camera). */
    Eigen::Matrix3d positionSqrtCov() const;
    /**
     * @brief Upper sqrt factors (P = S^T S) of several 3-D state blocks, from ONE marginal over all of them.
     * P_kk = X_k^T X_k with X_k = Xi^{-T} E_k (triangular solve), so cost is one QR regardless of the block count.
     * VISUALISATION ONLY: the ellipsoids drawn by the overlay and the 3-D plot need covariance factors. No gate,
     * association, update or archive uses this function (those use the information factor, see tagArchiveDistanceSqInfo
     * and SlamPoseAssociation.h). A per-vertex triangular solve would avoid the explicit factor if this ever enters an
     * estimator path.
     * @param offsets first index of each 3-state block
     */
    std::vector<Eigen::Matrix3d> blockSqrtCovs(const std::vector<Eigen::Index>& offsets) const;

    /** @brief Detections of the last frame. */
    const std::vector<tags::TagDetection>& lastDetections() const { return detections_; }
    /** @brief Predictions of the last frame (pre-update). */
    const TagPredictionSet& lastPredictions() const { return predictions_; }
    /** @brief Association of the last frame. */
    const TagAssociationResult& lastAssociation() const { return association_; }

    /**
     * @brief Culling rule: ids of the records to remove given their counters (pure function, tested).
     * Rules (any one): unconfirmed with misses >= unconfirmedMaxMisses; confirmed with misses >= confirmedMaxMisses;
     * behind the camera for >= maxBehindFrames; not associated for > maxUnseenFrames. Above maxTags the records with the
     * most misses are added until the map fits.
     */
    static std::vector<Eigen::Index> selectCulls(const std::vector<SlamPoseRecord>& records, int frame,
                                                 const SlamPoseConfig& cfg);

private:
    /// Outcome of a birth attempt for one pending tag.
    enum class BirthOutcome
    {
        Wait,        ///< not enough consistent observations yet
        Born,        ///< landmark added (new identity or re-identified)
        Duplicate    ///< an active tag is at the same place: the pending tag is dropped
    };

    /// Tries to birth one pending tag (consistent hypothesis, duplicate test, re-identification, correlated augmentation).
    BirthOutcome tryBirth(SlamPosePending& p, SlamPoseFrameStats& st);

    /// Upper square-root INFORMATION factor of the 6 x 6 marginal of one landmark block (marginal by QR; no inversion).
    Eigen::Matrix<double, 6, 6> landmarkSqrtInfo6(Eigen::Index offset) const;

    /// Stores a culled tag in the archive (lineage kept) and drops entries that are too old or exceed the capacity.
    void archiveCulled(const SlamPoseRecord& rec, int frameIndex);

    Camera camera_;
    SlamPoseConfig cfg_;
    std::shared_ptr<SystemVisualNav> model_;
    std::unique_ptr<SystemEstimator> est_;
    tags::ArucoTagDetector detector_;
    ImageFlow flow_;
    std::vector<SlamPoseRecord> records_;
    std::vector<SlamPosePending> pending_;
    std::vector<SlamPoseArchiveEntry> archive_;
    std::vector<Eigen::Vector3d> trajectory_;
    std::vector<tags::TagDetection> detections_;
    TagPredictionSet predictions_;
    TagAssociationResult association_;
    bool started_ = false;
    int mapFailures_ = 0;
    MapStatusCounts tagMapCounts_;
    MapStatusCounts flowMapCounts_;
    int nextIdentity_ = 0;
    double gate_ = 0.0;
    double reidGate_ = 0.0;
};

} // namespace slam

#endif
