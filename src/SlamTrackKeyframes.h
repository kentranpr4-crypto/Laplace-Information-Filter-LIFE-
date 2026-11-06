#ifndef SLAMTRACKKEYFRAMES_H
#define SLAMTRACKKEYFRAMES_H

/**
 * @file SlamTrackKeyframes.h
 * @brief Keyframe-anchored feature tracks for the Scenario 6 flow and delayed landmark initialisation.
 *
 * A keyframe is a frame whose body pose is copied into the estimator state (the etaPrev block, see
 * SystemVisualNav::predictDelayed). The features of the keyframe are the live LK tracks (ImageFlow, persistent track ids)
 * at that frame. Every later frame gives, for each track still alive, a correspondence (pixel in the keyframe,
 * pixel now). These correspondences drive
 *   - the epipolar flow measurement between the keyframe pose and the current pose (MeasurementIndoorFlow), and
 *   - the delayed triangulation of new point landmarks (SlamTrackTriangulation), which needs a baseline.
 *
 * Keyframe policy (SlamConfig, all ASSUMED values): a new keyframe starts when
 *   - the rotation-compensated median displacement of the keyframe tracks exceeds keyframeParallaxPx (the translation
 *     parallax the epipolar flow needs; the rotation is removed with the predicted rotation, as in IndoorFlowFrontEnd),
 *   - fewer than keyframeMinTracks tracks survive, the flow front end fails, or the keyframe is older than keyframeMaxAge.
 */

#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <Eigen/Core>
#include "ImageFlow.h"
#include "SlamLandmarkMap.h"

namespace slam
{

/** @brief A live track anchored at the keyframe: its pixel in the keyframe and in the current frame. */
struct KeyframeTrack
{
    int id = -1;                                   ///< ImageFlow track id (stable while the track lives)
    Eigen::Vector2d kfPixel = Eigen::Vector2d::Zero();   ///< pixel at the keyframe [px]
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();     ///< pixel in the current frame [px]
    int age = 0;                                   ///< frames the track has survived (ImageFlow)
};

/** @brief Keyframe anchoring of the live tracks (no geometry: the pipeline owns the estimator). */
class SlamTrackKeyframes
{
public:
    /** @brief True once a keyframe has been started. */
    bool hasKeyframe() const { return keyframeFrame_ >= 0; }

    /** @brief Frame index of the keyframe (-1 if none). */
    int keyframeFrame() const { return keyframeFrame_; }

    /**
     * @brief Starts a keyframe at frameIndex: every active track is anchored at its current pixel and the births already
     *        made from the previous keyframe are forgotten.
     */
    void start(int frameIndex, const std::vector<ImageFlowMatch>& active);

    /**
     * @brief Keyframe-anchored tracks that are still active, in the order of the active list.
     * Tracks that started after the keyframe are not included.
     */
    std::vector<KeyframeTrack> alive(const std::vector<ImageFlowMatch>& active) const;

    /** @brief Marks a track as the source of a landmark (it is no longer a birth candidate for this keyframe). */
    void markBorn(int trackId);

    /** @brief True if the track has already produced a landmark for this keyframe. */
    bool isBorn(int trackId) const;

    /** @brief Number of tracks anchored at the keyframe (alive or not). */
    std::size_t anchoredCount() const { return kfPixel_.size(); }

private:
    std::unordered_map<int, Eigen::Vector2d> kfPixel_;   ///< keyframe pixel per track id
    std::unordered_set<int> born_;                       ///< tracks that already produced a landmark
    int keyframeFrame_ = -1;
};

/**
 * @brief Keyframe switching rule, evaluated once per frame after the update (pure function, unit-tested).
 * @param cfg                   configuration (keyframeFlow, keyframeParallaxPx, keyframeMinTracks, keyframeMaxAge)
 * @param flowValid             the flow front end produced a valid measurement for this frame
 * @param medianParallaxPx      rotation-compensated median displacement of the keyframe tracks [px] (NaN if unknown)
 * @param aliveTracks           keyframe-anchored tracks still alive
 * @param framesSinceKeyframe   frames since the keyframe was started (0 = started at this frame: never restarted)
 * @return true if the current frame should become the new keyframe
 */
bool keyframeShouldStart(const SlamConfig& cfg, bool flowValid, double medianParallaxPx, int aliveTracks,
                         int framesSinceKeyframe);

} // namespace slam

#endif
