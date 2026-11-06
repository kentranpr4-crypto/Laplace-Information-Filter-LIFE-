#ifndef SLAMLANDMARKMAP_H
#define SLAMLANDMARKMAP_H

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "IndoorFlowFrontEnd.h"
#include "MeasurementPointBundle.h"
#include "PointFeatures.h"

namespace slam
{

/**
 * @brief Tunable parameters of the point-landmark SLAM layer (shared by Scenario 5 / 6 infrastructure).
 *
 * Values marked TUNED were chosen on the single available sequence room.MOV (in-sample, no independent validation);
 * ASSUMED values are engineering choices without a reference in the assignment data.
 */
struct SlamConfig
{
    // ---- measurement noise (INFLATED for the weak indoor intrinsics; see PixelNoiseModel) ----
    /// 1.0 px at the principal point, +0.5 px at the corners. The struct default of PixelNoiseModel is 2.5 / 2.0 px, which
    /// the earlier comment did not match (NIS/dof ~0.2 at 2.5 px: over-conservative); the value is now set explicitly.
    PixelNoiseModel pixelNoise{1.0, 0.5};  ///< TUNED on room.MOV (NIS over 300 frames): 1.0 px at centre, +0.5 px at corners

    // ---- landmark birth (single-view depth prior; see SlamInitialisation) ----
    double depthPriorMean = 2.5;           ///< ASSUMED: prior depth of a new point [m]
    double depthPriorSigma = 1.5;          ///< ASSUMED: 1-sigma of the prior depth [m]

    // ---- data association (see SlamAssociation) ----
    double gateNSigma = 3.0;               ///< chi2 individual-compatibility gate (nSigma = 3, dof 2)
    int maxHamming = 64;                   ///< ASSUMED: max ORB Hamming distance of an accepted match (of 256 bits)
    int ambiguityMargin = 12;              ///< ASSUMED: a second gated candidate within this many bits makes the match ambiguous
    double birthSeparationPx = 15.0;       ///< ASSUMED: unmatched features closer than this to a predicted landmark are not births
    double birthMinSpacingPx = 25.0;       ///< ASSUMED: minimum spacing between birth features of the same frame
    int maxBirthsPerFrame = 8;             ///< ASSUMED
    int minMatchesForUpdate = 4;           ///< ASSUMED: fewer associations -> prediction only (no measurement update)
    double mapGradientTolerance = 1e-6;    ///< MAP stop: max |gradient| of the log posterior (information units)
    // ---- births with persistence (a feature must be unmatched in two consecutive frames at the same place) ----
    bool birthPersistence = true;          ///< ASSUMED: transient features (one-frame detections) do not become landmarks
    double birthPersistencePx = 3.0;       ///< ASSUMED: maximum displacement of a birth feature between consecutive frames [px]
    // ---- optical flow (depth-free odometry between etaPrev and eta; MeasurementIndoorFlow, IndoorFlowFrontEnd) ----
    bool useFlow = true;                   ///< add the flow measurement (ablation switch)
    IndoorFlowOptions flow{};              ///< flow front end options (ASSUMED values, see the header)

    // ---- keyframes and parallax-gated (delayed) initialisation (Scenario 6 rework, see SlamTrackTriangulation) ----
    bool delayedInitialisation = true;     ///< births are triangulated from keyframe-anchored tracks (false: legacy single-view prior)
    bool keyframeFlow = true;              ///< flow between the keyframe pose and the current pose (false: every frame is a keyframe)
    double keyframeParallaxPx = 40.0;      ///< ASSUMED: rotation-compensated median displacement that starts a new keyframe [px]
    int keyframeMinTracks = 30;            ///< ASSUMED: fewer live keyframe-anchored tracks start a new keyframe
    int keyframeMaxAge = 90;               ///< ASSUMED: keyframe age limit [frames]
    double birthMinAngleDeg = 2.0;         ///< ASSUMED: minimum triangulation angle of a delayed birth [deg]
    double birthMaxReprojPx = 2.0;         ///< ASSUMED: maximum normalised-ray misfit of the triangulated point [px]
    double birthMinDepth = 0.3;            ///< ASSUMED: minimum depth of a delayed birth in either camera [m]
    double birthMaxDepth = 15.0;           ///< ASSUMED: maximum depth of a delayed birth [m]

    // ---- landmark bookkeeping and culling ----
    int maxLandmarks = 60;                 ///< ASSUMED capacity of the active map (dimension 3 x 60 = 180)
    int confirmHits = 3;                   ///< associated frames before a landmark is confirmed
    int tentativeMaxMisses = 2;            ///< unconfirmed landmark removed after this many visible misses
    int confirmedMaxMisses = 12;           ///< confirmed landmark removed after this many consecutive visible misses
    int maxUnseenFrames = 90;              ///< any landmark not associated for this many frames is removed (~3 s)
    int maxBehindFrames = 3;               ///< landmark predicted behind / too close to the camera for this long is removed
    double minDepth = 0.2;                 ///< [m] predicted camera-frame depth below this is "behind the camera"
    double maxDepth = 30.0;                ///< [m] predicted depth above this is not visible (too far to trust)
    double visibilityMarginPx = 4.0;       ///< predicted pixel must be this far inside the image

    // ---- process model prior and process noise (SystemVisualNav) ----
    Eigen::Matrix<double, 6, 1> nuPsd =
        (Eigen::Matrix<double, 6, 1>() << 0.02, 0.02, 0.02, 0.2, 0.2, 0.2).finished();  ///< TUNED on room.MOV (tracking through fast pans, 4 full-video variants): random-walk PSD
    // ---- metric scale (ASSUMED, see S6SpeedPrior.h): monocular images do not fix the scale ----
    bool useSpeedPrior = true;             ///< speed prior |v_BN^b| ~ N(speedPriorMps, speedPriorSigmaMps^2) at every frame
    double speedPriorMps = 0.5;            ///< ASSUMED walking speed [m/s]; also the seed of the initial velocity (body x)
    double speedPriorSigmaMps = 0.25;      ///< ASSUMED 1-sigma of the speed [m/s]
    double velocitySeedParallaxPx = 5.0;   ///< ASSUMED: keyframe parallax needed before the essential-matrix seed is trusted [px]
    double initialVelocitySigma = 0.3;     ///< ASSUMED 1-sigma of the initial body velocity [m/s]
    double initialAngularRateSigma = 0.3;  ///< ASSUMED 1-sigma of the initial body rate [rad/s]
    double initialPoseSigma = 1e-3;        ///< world gauge: the start pose is fixed to ~1 mm / 1 mrad
};

/**
 * @brief Bookkeeping of one landmark (the geometry itself lives in the estimator state).
 */
struct SlamLandmarkRecord
{
    Eigen::Index id = -1;                  ///< SystemVisualNav landmark id
    Descriptor descriptor{};               ///< last accepted descriptor (PointFeatureExtractor::describeAt at the track pixel)
    int trackId = -1;                      ///< ImageFlow track the landmark observes (-1: none, e.g. after a loss)
    int birthFrame = 0;
    int lastSeenFrame = -1;                ///< last frame in which the landmark was associated
    int hits = 0;                          ///< associated frames so far
    int misses = 0;                        ///< consecutive frames predicted visible without an association
    int behindFrames = 0;                  ///< consecutive frames predicted behind / too close to the camera
    bool confirmed = false;
    bool associatedThisFrame = false;
    Eigen::Vector2d lastPixel = Eigen::Vector2d::Zero();
};

/**
 * @brief Records of the active landmarks, in birth order (one record per SystemVisualNav landmark).
 */
class SlamLandmarkMap
{
public:
    /** @brief Creates an empty map with the given configuration. */
    explicit SlamLandmarkMap(const SlamConfig& config);

    /** @brief Configuration in use. */
    const SlamConfig& config() const { return config_; }

    /** @brief Number of active landmarks. */
    std::size_t size() const { return records_.size(); }

    /** @brief True if a record with this id exists. */
    bool contains(Eigen::Index id) const;

    /** @brief Record with the given id. @throws std::out_of_range if unknown. */
    SlamLandmarkRecord& at(Eigen::Index id);
    /** @brief Record with the given id. @throws std::out_of_range if unknown. */
    const SlamLandmarkRecord& at(Eigen::Index id) const;

    /** @brief Inserts a record. @throws std::invalid_argument if the id already exists. */
    void insert(const SlamLandmarkRecord& record);

    /** @brief Removes a record. @throws std::out_of_range if unknown. */
    void erase(Eigen::Index id);

    /** @brief All records in birth order. */
    const std::vector<SlamLandmarkRecord>& records() const { return records_; }

    /** @brief All records in birth order (mutable: used by the frame update). */
    std::vector<SlamLandmarkRecord>& records() { return records_; }

private:
    SlamConfig config_;
    std::vector<SlamLandmarkRecord> records_;
};

/**
 * @brief Landmark ids to delete after the frame update, from the lifecycle rules of the configuration.
 *
 * Rules (any one): unconfirmed with >= tentativeMaxMisses misses; confirmed with >= confirmedMaxMisses misses;
 * behind the camera for >= maxBehindFrames frames; unseen for > maxUnseenFrames frames. If the map still exceeds
 * maxLandmarks, the records with the most misses (then the oldest lastSeenFrame) are added until it fits.
 */
std::vector<Eigen::Index> selectLandmarksToCull(const SlamLandmarkMap& map, int frame);

} // namespace slam

#endif
