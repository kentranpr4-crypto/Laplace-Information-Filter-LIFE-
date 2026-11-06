#ifndef CALIBRATE_H
#define CALIBRATE_H

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Camera.h"

/**
 * @brief Chessboard calibration configuration (config.xml) after validation.
 *
 * XML layout (see data/<scene>/config.xml):
 * @code
 *   <opencv_storage><chessboard_data>
 *     <file_regex>"calibration.MOV"</file_regex>
 *     <chessboard><grid_width>10</grid_width><grid_height>7</grid_height><square_size>0.022</square_size></chessboard>
 *   </chessboard_data></opencv_storage>
 * @endcode
 * grid_width / grid_height are numbers of INNER corners per row / column. The surrounding quotes in
 * file_regex are stripped. The regex is matched (std::regex_match, i.e. whole file name) against the
 * entries of the directory containing config.xml; every matching regular file is used as a source video.
 */
struct ChessboardConfig
{
    std::filesystem::path configPath;          ///< Path of the config.xml that was read
    std::string fileRegex;                     ///< file_regex with surrounding quotes removed
    std::vector<std::filesystem::path> videos; ///< Matching video files, sorted by name (non-empty)
    int innerCornersPerRow = 0;                ///< grid_width  (OpenCV pattern columns)
    int innerCornersPerColumn = 0;             ///< grid_height (OpenCV pattern rows)
    double squareSizeM = 0.0;                  ///< square_size [m]
};

/** @brief Tunable parameters of the calibration pipeline. Defaults are the values used for the project data. */
struct CalibrationOptions
{
    int frameStride = 8;                    ///< Inspect every Nth frame of each video
    int maxViews = 60;                      ///< Upper bound on views used for the final fit (evenly subsampled)
    int minViews = 12;                      ///< Minimum accepted views; below this calibration fails
    double minBoardAreaFraction = 0.02;     ///< Corner bounding-box area / image area lower bound
    double maxGridCurvatureFraction = 0.25; ///< Max second difference of corner rows/cols / median spacing
    double minSharpnessFraction = 0.4;      ///< Min Laplacian-variance sharpness relative to the median detection
    double minSeparationFraction = 0.02;    ///< Min RMS corner displacement to every accepted view / image width
    double outlierViewFactor = 3.0;         ///< Drop views whose RMS > factor * median view RMS (iterated)
    double grossCornerErrorPx = 8.0;        ///< Drop views with any single corner farther than this from the model [px]
    int crossValidationFolds = 5;           ///< View-level folds for distortion-model selection
    /**
     * @brief Allows the 8-coefficient rational model to be selected automatically (default false).
     *
     * Deterministic selection (decision: 5-coefficient distortion, opencv-standard-5): the rational model is always
     * cross-validated and reported in modelScores, but it is used for the camera only when this flag is true AND it
     * beats the standard model by modelSwitchMargin. With the default the standard model is always used, and a
     * warning is recorded when the rational model would have been selected.
     */
    bool allowRationalModel = false;
    double modelSwitchMargin = 0.02;        ///< With allowRationalModel: rational CV RMS must be this relative amount lower
    bool fixPrincipalPointAtCentre = true;  ///< Keep (cx, cy) at the image centre; see calibrate.cpp for the evidence
    std::filesystem::path outputDirectory;  ///< If non-empty: write out/calibration_<scene>_* visualisations and report
    bool verbose = true;                    ///< Print progress and the final report to stdout
};

/** @brief One chessboard detection (or an accepted view after selection). */
struct CalibrationView
{
    int videoIndex = 0;                 ///< Index into ChessboardConfig::videos
    int frameIndex = 0;                 ///< 0-based frame index in that video
    double timeSeconds = 0.0;           ///< frameIndex / fps of that video
    std::vector<cv::Point2f> corners;   ///< Inner corners, row-major (rows * cols), canonical orientation
    double sharpness = 0.0;             ///< Variance of the Laplacian over the board region (grey levels^2)
    double boardAreaFraction = 0.0;     ///< Corner bounding-box area / image area
    double gridCurvatureFraction = 0.0; ///< Max corner second difference / median spacing (quality metric)
    cv::Vec3d rvec{0.0, 0.0, 0.0};      ///< Board pose in the camera frame from the final fit (Rodrigues)
    cv::Vec3d tvec{0.0, 0.0, 0.0};      ///< Board translation in the camera frame from the final fit [m]
    double rmsPx = std::numeric_limits<double>::quiet_NaN(); ///< In-sample RMS reprojection error of the final fit
};

/** @brief Cross-validated score of one distortion model. */
struct ModelScore
{
    std::string name;                 ///< "opencv-standard-5" or "opencv-rational-8"
    bool converged = false;           ///< False if any fold failed to produce a usable camera
    double cvRmsPx = std::numeric_limits<double>::quiet_NaN(); ///< Pooled held-out RMS [px]
    std::string failure;              ///< Exception text if converged is false
};

/** @brief Output of the calibration pipeline. */
struct CalibrationResult
{
    Camera camera;                              ///< Final camera (intrinsics, distortion, metadata)
    std::vector<CalibrationView> views;         ///< Views used in the final fit
    std::vector<std::string> rejectionLog;      ///< Human-readable reasons for rejected detections / views
    std::vector<ModelScore> modelScores;        ///< Cross-validation results per model
    std::string selectedModel;                  ///< Model used for the final camera
    std::vector<double> stdIntrinsics;          ///< OpenCV 1-sigma of (fx, fy, cx, cy, distortion...)
    double rmsPx = std::numeric_limits<double>::quiet_NaN();   ///< In-sample RMS recomputed with Camera::vectorToPixel
    double cvRmsPx = std::numeric_limits<double>::quiet_NaN(); ///< RMS returned by cv::calibrateCamera (cross-check)
    int numFramesInspected = 0;                 ///< Frames decoded (all videos)
    int numBoardsDetected = 0;                  ///< Frames where findChessboardCornersSB succeeded
    int numCandidateViews = 0;                  ///< Detections passing quality gates
    int numPoints = 0;                          ///< 2D-3D correspondences in the final fit
    int numOrientationFlips = 0;                ///< Detections whose corner order was reversed to canonical orientation
    double elapsedSeconds = 0.0;                ///< Wall-clock time of runCalibration
    std::vector<std::string> checks;            ///< "PASS"/"WARN" sanity-check lines
    std::vector<std::string> warnings;          ///< Non-fatal warnings
};

/**
 * @brief Reads and validates config.xml and resolves the calibration video(s).
 * @throws std::runtime_error on missing file, missing/invalid fields, no matching video or invalid regex.
 */
ChessboardConfig readChessboardConfig(const std::filesystem::path & configPath);

/**
 * @brief Full pipeline: read config, decode the video(s), detect chessboards, select views, calibrate,
 * cross-validate the distortion models and compute the reprojection statistics.
 * Does NOT write camera.xml (see calibrateCamera). Writes visualisations/report only if
 * options.outputDirectory is non-empty.
 * @throws std::runtime_error if a video cannot be opened, resolutions differ, or too few views are accepted.
 */
CalibrationResult runCalibration(const std::filesystem::path & configPath, const CalibrationOptions & options = {});

/**
 * @brief Calibrates from already-detected corner sets (no video access). Used by runCalibration and by tests.
 * @param imageSize Image size shared by all views.
 * @param innerCorners Pattern size (inner corners: columns x rows).
 * @param squareSizeM Square side length [m].
 * @param views Detections with corners in canonical row-major order.
 * @param options Model-selection / outlier parameters (the video-related fields are ignored).
 * @throws std::runtime_error if fewer than options.minViews views remain or the fit fails.
 */
CalibrationResult calibrateFromViews(cv::Size imageSize, cv::Size innerCorners, double squareSizeM,
                                     const std::vector<CalibrationView> & views, const CalibrationOptions & options = {});

/**
 * @brief CLI entry point (`a2 <config.xml> -c`). Runs runCalibration with visualisation into
 * <config dir>/../../out (if that directory exists or can be created), then writes camera.xml beside config.xml.
 * @throws std::runtime_error on failure (the CLI reports the message).
 */
void calibrateCamera(const std::filesystem::path & configPath);

#endif
