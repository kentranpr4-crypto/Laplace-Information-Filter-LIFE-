#ifndef IMAGEFLOW_H
#define IMAGEFLOW_H

#include <cstdint>
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief Options of the generic feature tracker (ImageFlow).
 *
 * All pixel quantities that refer to the working (down-scaled) image are documented as "working px";
 * everything returned by ImageFlow is in full-resolution pixels (OpenCV convention, see Camera.h).
 */
struct ImageFlowOptions
{
    double scale = 0.5;               ///< working resolution relative to the input (0 < scale <= 1)
    int maxFeatures = 400;            ///< maximum number of simultaneously active tracks
    double qualityLevel = 0.01;       ///< goodFeaturesToTrack quality level (relative to the best corner)
    double minDistanceWorkPx = 6.0;   ///< minimum spacing between features [working px]
    int blockSize = 7;                ///< goodFeaturesToTrack block size [working px]
    int lkWindow = 21;                ///< Lucas-Kanade window side [working px]
    int lkLevels = 3;                 ///< pyramid levels for pyramidal Lucas-Kanade (maxLevel)
    int lkMaxIterations = 30;         ///< termination: iterations per level
    double lkEpsilon = 0.01;          ///< termination: window update threshold per level [working px]
    double forwardBackwardPx = 1.0;   ///< forward-backward consistency tolerance [full-resolution px]
};

/** @brief One surviving track between the previous and the current frame (full-resolution pixels). */
struct ImageFlowMatch
{
    int id = -1;                 ///< stable track identifier (unique within one ImageFlow instance)
    int age = 0;                 ///< number of frames the track has survived, >= 1 for a match
    cv::Point2d prev;            ///< position in the previous frame [px]
    cv::Point2d curr;            ///< position in the current frame [px]
};

/** @brief Counters of the most recent ImageFlow::update call. */
struct ImageFlowStats
{
    int tracked = 0;             ///< tracks passed to Lucas-Kanade
    int lostLucasKanade = 0;     ///< tracks rejected by the LK status flag
    int lostForwardBackward = 0; ///< tracks rejected by the forward-backward check
    int detected = 0;            ///< new features detected in this frame
    int active = 0;              ///< active tracks after the update
};

/**
 * @brief Generic sparse feature tracker: Shi-Tomasi detection + pyramidal Lucas-Kanade + forward-backward check.
 *
 * Usage (one instance per video stream):
 * @code
 *   ImageFlow flow;
 *   for each frame: matches = flow.update(gray, detectMask);   // matches: previous -> current positions
 * @endcode
 * The first call only detects features (no matches). Tracks that fail the LK status flag or the
 * forward-backward check (|x_back - x_prev| > forwardBackwardPx, full-resolution) are removed.
 * New features are detected in the detection mask (non-zero = allowed; empty = whole image) and kept
 * at least minDistanceWorkPx away from existing tracks until maxFeatures is reached.
 *
 * Robust model fitting is not part of the tracker: see robustHomographyInliers() for a plane-induced
 * (homography) RANSAC used by the outdoor flow scenario.
 */
class ImageFlow
{
public:
    /** @brief Creates a tracker with default options. */
    ImageFlow();

    /** @brief Creates a tracker with the given options. @throws std::invalid_argument on invalid options. */
    explicit ImageFlow(ImageFlowOptions options);

    /** @brief Forgets all tracks and the previous frame. Track ids continue to increase. */
    void reset();

    /**
     * @brief Tracks the active features into gray and detects new ones.
     * @param gray        CV_8UC1 image, full resolution (the size must not change between calls)
     * @param detectMask  optional CV_8UC1 mask at full resolution: non-zero pixels may host new features
     * @return matches between the previous call and this call (empty on the first call)
     * @throws std::invalid_argument if gray is empty or not CV_8UC1, or if the mask size differs
     */
    std::vector<ImageFlowMatch> update(const cv::Mat& gray, const cv::Mat& detectMask = cv::Mat());

    /** @brief Active tracks after the most recent update (full-resolution current positions). */
    const std::vector<ImageFlowMatch>& activeTracks() const { return active_; }

    /** @brief Counters of the most recent update. */
    const ImageFlowStats& stats() const { return stats_; }

    /** @brief Options in use. */
    const ImageFlowOptions& options() const { return options_; }

    /**
     * @brief RANSAC fit of a single planar homography to the matches (pixel domain, full resolution).
     *
     * A pinhole camera viewing a plane induces exactly a homography between two images, so inliers of this
     * fit are the features consistent with one dominant plane (e.g. the ground). Uses cv::findHomography.
     * @param matches         input matches
     * @param thresholdPx     reprojection threshold [px]
     * @param[out] inlierMask one entry per match (1 = inlier); resized to matches.size()
     * @param[out] homography 3x3 CV_64F homography from prev to curr pixels (empty if the fit failed)
     * @return number of inliers (0 if the fit failed or fewer than 4 matches)
     */
    static int robustHomographyInliers(const std::vector<ImageFlowMatch>& matches, double thresholdPx,
                                       std::vector<std::uint8_t>& inlierMask, cv::Mat& homography);

private:
    ImageFlowOptions options_;
    ImageFlowStats stats_;
    cv::Mat prevWork_;                          ///< previous working-resolution image
    std::vector<ImageFlowMatch> active_;        ///< active tracks, full-resolution current positions
    std::vector<cv::Point2f> activeWorkPts_;    ///< active tracks, working px (same order as active_)
    int nextId_ = 0;
    cv::Size fullSize_;
    bool hasPrevious_ = false;
};

#endif
