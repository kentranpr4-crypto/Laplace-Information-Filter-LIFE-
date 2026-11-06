#ifndef ARUCOTAGS_H
#define ARUCOTAGS_H

/**
 * @file ArucoTags.h
 * @brief ArUco detection of the identical-ID tags of Scenario 5 and the canonical tag geometry.
 *
 * Empirical findings on data/indoor/tag_identical.MOV (see engineering-reports/ and the Scenario 5 report):
 *  - The printed markers belong to the 6X6 family; every detection decodes to ID 5 (DICT_6X6_50, DICT_6X6_100 and
 *    DICT_6X6_250 give identical detections, DICT_6X6_1000 detects slightly fewer). No 4X4, 5X5, 7X7 or AprilTag
 *    dictionary detects anything.
 *  - ID 5 is not invariant under 90 degree rotations (checked for the 6X6 dictionaries), so the detector resolves the
 *    in-plane rotation and returns the four corners in the canonical order of the printed pattern for every tag.
 *    The identical ID therefore does not cause a corner-order ambiguity; identity must come from geometry.
 *
 * Canonical tag frame t (right-handed): x to the right of the printed pattern, y up, z = x cross y towards the viewer.
 * The corners c_0..c_3 are TL, TR, BR, BL: (-s/2, s/2, 0), (s/2, s/2, 0), (s/2, -s/2, 0), (-s/2, -s/2, 0), where s is
 * the outer side length of the black marker border (the ArUco corner definition). The same ordering is used by
 * cv::aruco::estimatePoseSingleMarkers and by cv::solvePnP with SOLVEPNP_IPPE_SQUARE.
 */

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

namespace tags
{

/** @brief One detected marker. Corners are in the canonical order (TL, TR, BR, BL) of the printed pattern. */
struct TagDetection
{
    int id = -1;                                          ///< dictionary id (ID 5 for the tags of this video)
    Eigen::Matrix<double, 4, 2> corners = Eigen::Matrix<double, 4, 2>::Zero();  ///< rows = corners, (u, v) pixels
    double sidePx = 0.0;                                  ///< mean edge length [px]

    /** @brief Mean of the four corners [px]. */
    Eigen::Vector2d centre() const { return corners.colwise().mean().transpose(); }
    /** @brief Corners as the 8-vector [u0 v0 u1 v1 u2 v2 u3 v3] (corner-major), the measurement convention. */
    Eigen::Matrix<double, 8, 1> cornerVector() const;
};

/** @brief Canonical corners of a tag of outer side length s, rows = c_0..c_3, in the tag frame [m]. */
Eigen::Matrix<double, 4, 3> canonicalCorners(double side);

/** @brief In-plane rotation Rz(q * pi / 2) about the tag normal, q in {0, 1, 2, 3}. Keeps the normal (z) axis. */
Eigen::Matrix3d inPlaneRotation(int q);

/** @brief Converts a corner matrix (rows = corners) to the 8-vector [u0 v0 ... u3 v3]. */
Eigen::Matrix<double, 8, 1> cornersToVector(const Eigen::Matrix<double, 4, 2>& corners);

/** @brief Converts the 8-vector [u0 v0 ... u3 v3] to a corner matrix (rows = corners). */
Eigen::Matrix<double, 4, 2> vectorToCorners(const Eigen::Matrix<double, 8, 1>& v);

/**
 * @brief ArUco detector restricted to one dictionary id.
 *
 * Wraps cv::aruco::ArucoDetector (OpenCV 5, objdetect module) with sub-pixel corner refinement. Detections of any other
 * id, and detections whose mean edge is below minSidePx, are dropped. The detector is not thread-safe.
 */
class ArucoTagDetector
{
public:
    /** @brief Detector options. */
    struct Options
    {
        cv::aruco::PredefinedDictionaryType dictionary = cv::aruco::DICT_6X6_50;  ///< ID 5 decodes identically in 50/100/250
        int targetId = 5;                 ///< only this id is returned
        double minSidePx = 12.0;          ///< reject detections smaller than this (mean edge) [px]
    };

    /** @brief Constructs a detector with the default options (6X6_50, id 5). */
    ArucoTagDetector();
    /** @brief Constructs a detector with explicit options. */
    explicit ArucoTagDetector(const Options& options);

    /**
     * @brief Detects the target-id markers in a grey or BGR image.
     * @return detections in the order reported by OpenCV; corners are in canonical order.
     */
    std::vector<TagDetection> detect(const cv::Mat& image);

    /** @brief Options in use. */
    const Options& options() const noexcept { return options_; }

private:
    Options options_;
    cv::aruco::ArucoDetector detector_;
};

} // namespace tags

#endif
