#ifndef SLAMPOINTPIPELINE_H
#define SLAMPOINTPIPELINE_H

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <Eigen/Core>
#include "Camera.h"
#include "ImageFlow.h"
#include "IndoorFlowFrontEnd.h"
#include "PointFeatures.h"
#include "SlamAssociation.h"
#include "SlamLandmarkMap.h"
#include "SlamTrackKeyframes.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

namespace slam
{

/** @brief Per-frame diagnostics of SlamPointPipeline (NaN where not applicable). */
struct SlamFrameStats
{
    int frame = 0;
    double time = 0.0;
    int features = 0;               ///< detected features
    int landmarksBefore = 0;        ///< active landmarks at the start of the frame
    int visible = 0;                ///< predicted visible landmarks
    int associated = 0;             ///< accepted feature-landmark pairs
    int noCandidate = 0;            ///< visible landmarks without a gated, descriptor-compatible feature
    int ambiguous = 0;              ///< visible landmarks rejected as ambiguous
    int conflicts = 0;              ///< landmarks rejected because the feature was claimed twice
    bool updated = false;           ///< measurement update performed
    int births = 0;                 ///< landmarks initialised this frame
    int persistentCandidates = 0;   ///< unmatched features that passed the two-frame persistence test
    int culled = 0;                 ///< landmarks removed this frame
    int landmarks = 0;              ///< active landmarks at the end of the frame
    int flowInput = 0;              ///< pixel correspondences from ImageFlow
    bool flowValid = false;         ///< flow measurement used this frame
    bool flowRotationModel = false; ///< Model R (pure rotation) selected; false = Model E (epipolar)
    int flowPairs = 0;              ///< correspondences in the flow measurement (after RANSAC and the prior gate)
    double flowNis = 0.0;           ///< whitened flow residual squared at the prior (before the update)
    int flowDof = 0;                ///< residual dimension of the flow measurement
    bool flowUpdated = false;       ///< flow MAP update applied
    bool flowBranchMirrored = false; ///< translation branch mirrored by the cheirality check (see updateFlowWithCheirality)
    int cheiralityTotal = 0;        ///< flow inlier correspondences with a determined depth sign (posterior pose)
    double cheiralityPosPre = std::numeric_limits<double>::quiet_NaN();  ///< fraction with positive depth in both cameras, prior pose (before the flow MAP; NaN: n/a)
    double cheiralityPosPost = std::numeric_limits<double>::quiet_NaN(); ///< same, posterior pose (after the flow MAP; NaN: n/a)
    double cheiralityPosRecover = std::numeric_limits<double>::quiet_NaN(); ///< same, recoverPose solution of the essential matrix (NaN: no relative pose)
    double transCosPre = std::numeric_limits<double>::quiet_NaN();       ///< cos(angle) between the prior unit translation and the recoverPose one (NaN: n/a)
    double transCosPost = std::numeric_limits<double>::quiet_NaN();      ///< cos(angle) between the posterior unit translation and the recoverPose one (NaN: n/a)
    double flowRotDiffDeg = std::numeric_limits<double>::quiet_NaN();   ///< rotation angle between the posterior and recoverPose relative rotations [deg] (NaN: n/a)
    Eigen::Vector3d flowTPost = Eigen::Vector3d::Zero();   ///< posterior unit translation (camera A frame; zero: n/a)
    Eigen::Vector3d flowTRec = Eigen::Vector3d::Zero();    ///< recoverPose unit translation (camera A frame; zero: n/a)
    int keyframeTracks = 0;         ///< keyframe-anchored tracks alive this frame (flow correspondences)
    double keyframeParallaxPx = 0.0; ///< rotation-compensated median displacement of the keyframe tracks [px] (NaN: unknown)
    bool keyframeStarted = false;   ///< a new keyframe was started at the end of this frame
    int keyframeAge = 0;            ///< frames since the keyframe was started (at the end of the frame)
    int birthCandidates = 0;        ///< keyframe tracks evaluated for a delayed birth
    int birthRejectedInvalid = 0;   ///< candidates rejected: degenerate triangulation or noise factor
    int birthRejectedBehind = 0;    ///< candidates rejected: triangulated point behind a camera
    int birthRejectedAngle = 0;     ///< candidates rejected: triangulation angle below birthMinAngleDeg
    int birthRejectedDepth = 0;     ///< candidates rejected: depth outside the birth range
    int birthRejectedMisfit = 0;    ///< candidates rejected: normalised-ray misfit above birthMaxReprojPx
    double birthMedianAngleDeg = 0.0; ///< median triangulation angle of the accepted births [deg] (0 if none)
    double speedMps = 0.0;          ///< body speed |v_BN^b| after the speed-prior update [m/s] (NaN: unknown)
    double msFlow = 0.0;            ///< flow tracking and measurement [ms]
    double nis = 0.0;              ///< normalised innovation squared of the matched set (before the update)
    int nisDof = 0;                 ///< 2 x associated
    int mapIterations = 0;          ///< trust-region iterations of the MAP step
    int mapStatus = -1;             ///< FuncMinStatus of the MAP step (-1 if no update)
    bool precisionExact = true;     ///< posterior precision positive definite without flooring
    double msDetect = 0.0;          ///< feature extraction [ms]
    double msPredict = 0.0;         ///< time update and landmark prediction [ms]
    double msAssociate = 0.0;       ///< association [ms]
    double msUpdate = 0.0;          ///< MAP + Laplace update [ms]
    double msBirthCull = 0.0;       ///< culling and initialisation [ms]
    double msTotal = 0.0;           ///< whole step [ms]
};

/**
 * @brief Frame-to-frame point-landmark SLAM with the information-form estimator.
 *
 * Per frame: persistent LK tracks (ImageFlow) with a descriptor at each track pixel; keyframe-anchored flow measurement
 * between the keyframe pose (etaPrev) and the current pose (MeasurementIndoorFlow, Model E / Model R, S6KeyframeFlow);
 * ASSUMED speed prior on the body velocity (S6SpeedPrior); landmark prediction; track-identity association with
 * descriptor re-acquisition (SlamTrackAssociation); Laplace/MAP update with the associated pixels; culling; delayed,
 * parallax-gated births from keyframe tracks triangulated from both poses (SlamTrackTriangulation, correlated
 * augmentation); keyframe switching (SlamTrackKeyframes). The legacy single-view births are kept behind
 * SlamConfig::delayedInitialisation = false.
 *
 * The estimator state is [nu(6); eta(6); landmarks(3 each)]; the world frame is the start camera body frame.
 * Monocular scale is not observable: it is set by the depth prior of the first landmarks and by the random-walk
 * PSD of the velocity (both ASSUMED, see SlamConfig).
 */
class SlamPointPipeline
{
public:
    /**
     * @param camera  calibrated camera (copied)
     * @param cfg     configuration (copied)
     * @throws std::invalid_argument if the camera is invalid or the configuration is inconsistent
     */
    SlamPointPipeline(const Camera& camera, const SlamConfig& cfg);

    /**
     * @brief Processes one grey frame at time t (s, non-decreasing).
     * @throws std::invalid_argument for an invalid image or a decreasing time
     */
    SlamFrameStats step(const cv::Mat& gray, int frameIndex, double t);

    /** @brief Current estimator (state density and time). */
    const SystemEstimator& estimator() const { return *est_; }
    /** @brief Dynamic model with the landmark layout. */
    const SystemVisualNav& model() const { return *model_; }
    /** @brief Landmark records. */
    const SlamLandmarkMap& map() const { return map_; }
    /** @brief Camera. */
    const Camera& camera() const { return camera_; }
    /** @brief Configuration. */
    const SlamConfig& config() const { return cfg_; }

    /** @brief Number of frames whose MAP update failed and was reverted (numerical robustness counter). */
    int mapFailures() const { return mapFailures_; }

    /** @brief Camera position r_CN^n of every processed frame (estimate), oldest first. */
    const std::vector<Eigen::Vector3d>& trajectory() const { return trajectory_; }

    /** @brief Body/camera position covariance sqrt factor (upper, P = S^T S) at the current state. */
    Eigen::Matrix3d positionSqrtCov() const;
    /** @brief Sqrt factor (upper, P = S^T S) of the 3-D landmark block at the given state offset. */
    Eigen::Matrix3d landmarkSqrtCov(Eigen::Index offset) const;

    /** @brief Last-frame data for the overlay. */
    const std::vector<PointFeature>& lastFeatures() const { return features_; }
    const PredictionSet& lastPredictions() const { return predictions_; }
    const AssociationResult& lastAssociation() const { return association_; }
    const std::vector<std::size_t>& lastBirthFeatures() const { return births_; }
    /** @brief Pixels (current frame) of the landmarks born in the last frame (delayed births: the track pixel). */
    const std::vector<Eigen::Vector2d>& lastBirthPixels() const { return birthPixels_; }
    /** @brief Keyframe state (live keyframe-anchored tracks and births per keyframe). */
    const SlamTrackKeyframes& keyframes() const { return keyframes_; }

private:
    /// Unmatched feature of the previous frame (a birth needs the same feature in two consecutive frames).
    struct PendingBirth
    {
        Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
        int frame = -1;
    };

    /** @brief Upper sqrt factor (P = S^T S) of the 3 x 3 covariance block at offset (triangular solves only). */
    Eigen::Matrix3d sqrtCovBlock(Eigen::Index offset) const;

    /** @brief Delayed (parallax-gated) births from keyframe-anchored tracks; fills births_, birthPixels_ and stats. */
    void delayedBirths(int frameIndex, const std::vector<KeyframeTrack>& kfTracks, SlamFrameStats& st);

    /**
     * @brief Seeds the body velocity direction from the keyframe relative pose of the essential matrix (once).
     * The velocity mean becomes speedPriorMps along the recovered camera motion (information-form mean shift, no
     * covariance formed). @return true if the seed was applied
     */
    bool seedVelocityFromEssential(const Eigen::Matrix3d& rRel, const Eigen::Vector3d& tRel);

    /** @brief Legacy single-view births from unmatched features with a depth prior (cfg.delayedInitialisation = false). */
    void legacyBirths(int frameIndex, SlamFrameStats& st);

    /** @brief Registers a new landmark (id from the model) in the bookkeeping map. */
    void recordBirth(Eigen::Index id, const Descriptor& descriptor, int frameIndex, const Eigen::Vector2d& pixel);

    Camera camera_;
    SlamConfig cfg_;
    std::shared_ptr<SystemVisualNav> model_;
    std::unique_ptr<SystemEstimator> est_;
    SlamLandmarkMap map_;
    PointFeatureExtractor extractor_;
    ImageFlow flow_;
    SlamTrackKeyframes keyframes_;
    std::vector<PendingBirth> pendingBirth_;
    std::vector<Eigen::Vector3d> trajectory_;
    std::vector<PointFeature> features_;   ///< active LK tracks of the frame (pixel, descriptor at the track pixel)
    std::vector<int> trackIds_;            ///< ImageFlow id of each entry of features_
    std::vector<bool> described_;          ///< entry has a valid descriptor (not too close to the border)
    PredictionSet predictions_;
    AssociationResult association_;
    std::vector<std::size_t> births_;
    std::vector<Eigen::Vector2d> birthPixels_;
    bool started_ = false;
    bool velocitySeeded_ = false;
    int mapFailures_ = 0;
};

} // namespace slam

#endif
