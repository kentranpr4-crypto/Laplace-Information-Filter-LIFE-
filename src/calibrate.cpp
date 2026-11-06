#include "calibrate.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <map>
#include <print>
#include <regex>
#include <stdexcept>
#include <string>
#include <system_error>

#include <Eigen/Core>
#include <opencv2/calib.hpp>
#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/** @brief Detection plus the data needed for orientation normalisation (not part of the public result). */
struct DetectionRecord
{
    CalibrationView view;
    int cellSign = 0; ///< +1 / -1: brightness of the first interior cell relative to the board mean
};

/** @brief Output of one calibration fit (cv::calibrateCamera with the extended outputs). */
struct Fit
{
    cv::Mat K;
    cv::Mat dist;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    cv::Mat stdIntrinsics;
    double cvRms = kNaN;
};

/** @brief Euclidean length of a 2D point. */
float pointNorm(cv::Point2f p)
{
    return std::sqrt(p.x * p.x + p.y * p.y);
}

/** @brief Strips whitespace and one pair of surrounding quotes. */
std::string stripQuotesAndSpaces(std::string s)
{
    auto isSpace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    while (!s.empty() && isSpace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && isSpace(static_cast<unsigned char>(s.back()))) s.pop_back();
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'') && s.back() == s.front())
    {
        s = s.substr(1, s.size() - 2);
    }
    while (!s.empty() && isSpace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && isSpace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return kNaN;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double m = values[mid];
    if (values.size() % 2 == 0)
    {
        const double lower = *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
        m = 0.5 * (m + lower);
    }
    return m;
}

/** @brief Object points of the inner-corner grid in row-major order, z = 0, scaled by the square size. */
std::vector<cv::Point3f> boardObjectPoints(cv::Size inner, double squareSizeM)
{
    std::vector<cv::Point3f> pts;
    pts.reserve(static_cast<std::size_t>(inner.area()));
    for (int r = 0; r < inner.height; ++r)
    {
        for (int c = 0; c < inner.width; ++c)
        {
            pts.emplace_back(static_cast<float>(c * squareSizeM), static_cast<float>(r * squareSizeM), 0.0f);
        }
    }
    return pts;
}

cv::Vec3d toVec3(const cv::Mat & m)
{
    cv::Mat d;
    m.convertTo(d, CV_64F);
    return cv::Vec3d(d.at<double>(0), d.at<double>(1), d.at<double>(2));
}

/**
 * @brief Converts OpenCV distortion output to the Camera coefficient order (k1, k2, p1, p2, k3, k4, k5, k6).
 * OpenCV 5 returns 14 values when CALIB_RATIONAL_MODEL is set (thin-prism s1..s4 and tilt tauX, tauY follow).
 * Those extra terms are not enabled in this project, so they must be zero; otherwise the fit is not representable.
 */
std::vector<double> toCoefficients(const cv::Mat & dist)
{
    cv::Mat d;
    dist.convertTo(d, CV_64F);
    d = d.reshape(1, 1);
    std::vector<double> all(static_cast<std::size_t>(d.cols));
    for (int i = 0; i < d.cols; ++i)
    {
        all[static_cast<std::size_t>(i)] = d.at<double>(0, i);
    }
    constexpr std::size_t kCameraCoefficients = 8;
    if (all.size() <= kCameraCoefficients)
    {
        return all;
    }
    for (std::size_t i = kCameraCoefficients; i < all.size(); ++i)
    {
        if (std::abs(all[i]) > 1e-12)
        {
            throw std::runtime_error(std::format("unsupported non-zero distortion term at index {} (thin prism / tilt model)", i));
        }
    }
    return std::vector<double>(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(kCameraCoefficients));
}

Eigen::Matrix3d toEigenK(const cv::Mat & K)
{
    cv::Mat d;
    K.convertTo(d, CV_64F);
    Eigen::Matrix3d out;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            out(r, c) = d.at<double>(r, c);
        }
    }
    return out;
}

Camera cameraFromFit(const cv::Mat & K, const cv::Mat & dist, cv::Size imageSize)
{
    return Camera(imageSize.width, imageSize.height, toEigenK(K), toCoefficients(dist));
}

/**
 * @brief Runs cv::calibrateCamera (extended outputs) on the given views.
 * @param flags 0 for the standard 5-coefficient model, cv::CALIB_RATIONAL_MODEL for 8 coefficients.
 */
Fit fitIntrinsics(const std::vector<std::vector<cv::Point3f>> & objectPoints,
                  const std::vector<std::vector<cv::Point2f>> & imagePoints,
                  cv::Size imageSize, int flags)
{
    Fit fit;
    fit.K = cv::Mat::eye(3, 3, CV_64F);
    fit.dist = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat perViewErrors;
    cv::Mat stdExt;
    // Levenberg-Marquardt stopping rule. A 1e-12 tolerance never triggers on noisy data, so every fit ran the
    // full iteration cap (measured: ~7 s per fit on synthetic views). 100 iterations / 1e-9 is ample for RMS to 1e-6 px.
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-9);
    fit.cvRms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, fit.K, fit.dist, fit.rvecs, fit.tvecs,
                                    fit.stdIntrinsics, stdExt, perViewErrors, flags, criteria);
    return fit;
}

/** @brief Sum of squared pixel errors (2D norm squared per point) of one view under a camera model. */
double viewSquaredError(const Camera & camera, const std::vector<cv::Point3f> & objectPoints,
                        const std::vector<cv::Point2f> & imagePoints, const cv::Vec3d & rvec, const cv::Vec3d & tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d Re;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            Re(r, c) = R.at<double>(r, c);
        }
    }
    const Eigen::Vector3d t(tvec[0], tvec[1], tvec[2]);
    double sse = 0.0;
    for (std::size_t i = 0; i < objectPoints.size(); ++i)
    {
        const Eigen::Vector3d P(objectPoints[i].x, objectPoints[i].y, objectPoints[i].z);
        const Eigen::Vector2d pixel = camera.vectorToPixel(Re * P + t);
        const Eigen::Vector2d measured(imagePoints[i].x, imagePoints[i].y);
        sse += (pixel - measured).squaredNorm();
    }
    return sse;
}

/** @brief Maximum normalised second difference of the corner grid (distortion-tolerant grid quality). */
double gridCurvatureFraction(const std::vector<cv::Point2f> & c, cv::Size inner)
{
    const int cols = inner.width;
    const int rows = inner.height;
    double maxCurvature = 0.0;
    std::vector<double> spacings;
    auto at = [&](int r, int col) { return c[static_cast<std::size_t>(r * cols + col)]; };
    for (int r = 0; r < rows; ++r)
    {
        for (int col = 0; col + 1 < cols; ++col)
        {
            spacings.push_back(pointNorm(at(r, col + 1) - at(r, col)));
        }
        for (int col = 1; col + 1 < cols; ++col)
        {
            const cv::Point2f d = at(r, col - 1) - 2.0f * at(r, col) + at(r, col + 1);
            maxCurvature = std::max(maxCurvature, static_cast<double>(pointNorm(d)));
        }
    }
    for (int col = 0; col < cols; ++col)
    {
        for (int r = 0; r + 1 < rows; ++r)
        {
            spacings.push_back(pointNorm(at(r + 1, col) - at(r, col)));
        }
        for (int r = 1; r + 1 < rows; ++r)
        {
            const cv::Point2f d = at(r - 1, col) - 2.0f * at(r, col) + at(r + 1, col);
            maxCurvature = std::max(maxCurvature, static_cast<double>(pointNorm(d)));
        }
    }
    const double spacing = median(spacings);
    return spacing > 0.0 ? maxCurvature / spacing : kNaN;
}

double boardAreaFraction(const std::vector<cv::Point2f> & c, cv::Size imageSize)
{
    const cv::Rect box = cv::boundingRect(c);
    return static_cast<double>(box.area()) / (static_cast<double>(imageSize.width) * imageSize.height);
}

/** @brief Variance of the Laplacian over the board region (higher = sharper). */
double boardSharpness(const cv::Mat & gray, const std::vector<cv::Point2f> & c, cv::Size inner)
{
    const cv::Rect box = cv::boundingRect(c);
    const double margin = std::max(box.width, box.height) / std::max(inner.width, inner.height);
    cv::Rect roi(static_cast<int>(std::floor(box.x - margin)), static_cast<int>(std::floor(box.y - margin)),
                 static_cast<int>(std::ceil(box.width + 2.0 * margin)), static_cast<int>(std::ceil(box.height + 2.0 * margin)));
    roi &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (roi.area() <= 0)
    {
        return 0.0;
    }
    cv::Mat lap;
    cv::Laplacian(gray(roi), lap, CV_32F, 3);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

/** @brief Mean grey level in a 5x5 window around p (clipped to the image). */
double sampleMean(const cv::Mat & gray, cv::Point2f p)
{
    cv::Rect roi(static_cast<int>(std::lround(p.x)) - 2, static_cast<int>(std::lround(p.y)) - 2, 5, 5);
    roi &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (roi.area() <= 0)
    {
        return kNaN;
    }
    return cv::mean(gray(roi))[0];
}

/**
 * @brief Brightness of the first interior cell (between corners (0,0),(0,1),(1,0),(1,1)) relative to the mean
 * of all interior cells. Reversing the corner order flips this sign when (cols + rows) is odd, which lets us
 * identify and correct 180-degree relabelling of the detected grid.
 * @return +1 / -1, or 0 if undefined.
 */
int firstCellSign(const cv::Mat & gray, const std::vector<cv::Point2f> & c, cv::Size inner)
{
    const int cols = inner.width;
    const int rows = inner.height;
    auto cellCentre = [&](int r, int col)
    {
        const std::size_t i00 = static_cast<std::size_t>(r * cols + col);
        const std::size_t i01 = i00 + 1;
        const std::size_t i10 = static_cast<std::size_t>((r + 1) * cols + col);
        const std::size_t i11 = i10 + 1;
        return 0.25f * (c[i00] + c[i01] + c[i10] + c[i11]);
    };
    double sum = 0.0;
    int count = 0;
    for (int r = 0; r + 1 < rows; ++r)
    {
        for (int col = 0; col + 1 < cols; ++col)
        {
            const double v = sampleMean(gray, cellCentre(r, col));
            if (std::isfinite(v))
            {
                sum += v;
                ++count;
            }
        }
    }
    const double first = sampleMean(gray, cellCentre(0, 0));
    if (count == 0 || !std::isfinite(first))
    {
        return 0;
    }
    const double boardMean = sum / count;
    if (first == boardMean)
    {
        return 0;
    }
    return first > boardMean ? 1 : -1;
}

/** @brief Detects the chessboard in a greyscale image. Returns row-major corners on success. */
bool detectBoard(const cv::Mat & gray, cv::Size inner, std::vector<cv::Point2f> & corners)
{
    cv::Mat found;
    // EXHAUSTIVE: measured on outdoor frames (stride 24) to detect the board in 61% of frames vs 10% with
    // CALIB_CB_NORMALIZE_IMAGE; costs ~0.33 s per 2688x1512 frame.
    const bool ok = cv::findChessboardCornersSB(gray, inner, found, cv::CALIB_CB_EXHAUSTIVE);
    if (!ok || found.total() != static_cast<std::size_t>(inner.area()))
    {
        return false;
    }
    corners.assign(found.ptr<cv::Point2f>(), found.ptr<cv::Point2f>() + found.total());
    return true;
}

/** @brief Mean squared corner displacement (RMS over corners) between two corner sets. */
double cornerRms(const std::vector<cv::Point2f> & a, const std::vector<cv::Point2f> & b)
{
    double sse = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const double d = pointNorm(a[i] - b[i]);
        sse += d * d;
    }
    return std::sqrt(sse / static_cast<double>(a.size()));
}

void requireFiniteOrThrow(double value, const char * what)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error(std::format("calibration produced a non-finite {}", what));
    }
}

std::string modelName(bool rational)
{
    return rational ? "opencv-rational-8" : "opencv-standard-5";
}

/**
 * @brief Cross-validates one distortion model over view-level folds.
 * Held-out views are scored with a PnP pose estimated from the fold's intrinsics.
 */
ModelScore crossValidate(const std::vector<std::vector<cv::Point3f>> & allObj,
                         const std::vector<std::vector<cv::Point2f>> & allImg,
                         const std::vector<int> & active, cv::Size imageSize, int flags, bool rational, int folds)
{
    ModelScore score;
    score.name = modelName(rational);
    double sse = 0.0;
    std::size_t points = 0;
    try
    {
        for (int f = 0; f < folds; ++f)
        {
            std::vector<std::vector<cv::Point3f>> trainObj;
            std::vector<std::vector<cv::Point2f>> trainImg;
            std::vector<int> test;
            for (std::size_t k = 0; k < active.size(); ++k)
            {
                const int idx = active[k];
                if (static_cast<int>(k) % folds == f)
                {
                    test.push_back(idx);
                }
                else
                {
                    trainObj.push_back(allObj[static_cast<std::size_t>(idx)]);
                    trainImg.push_back(allImg[static_cast<std::size_t>(idx)]);
                }
            }
            if (test.empty())
            {
                continue;
            }
            const Fit fit = fitIntrinsics(trainObj, trainImg, imageSize, flags);
            const Camera camera = cameraFromFit(fit.K, fit.dist, imageSize);
            for (int idx : test)
            {
                cv::Mat rvec;
                cv::Mat tvec;
                const bool ok = cv::solvePnP(allObj[static_cast<std::size_t>(idx)], allImg[static_cast<std::size_t>(idx)],
                                             fit.K, fit.dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                if (!ok)
                {
                    throw std::runtime_error("solvePnP failed for a held-out view");
                }
                sse += viewSquaredError(camera, allObj[static_cast<std::size_t>(idx)], allImg[static_cast<std::size_t>(idx)],
                                        toVec3(rvec), toVec3(tvec));
                points += allObj[static_cast<std::size_t>(idx)].size();
            }
        }
        score.cvRmsPx = std::sqrt(sse / static_cast<double>(points));
        score.converged = std::isfinite(score.cvRmsPx);
    }
    catch (const std::exception & e)
    {
        score.converged = false;
        score.cvRmsPx = kNaN;
        score.failure = e.what();
    }
    return score;
}

/** @brief Opens the video at its path, validating that it is readable. */
cv::VideoCapture openVideo(const std::filesystem::path & path)
{
    cv::VideoCapture cap(path.string());
    if (!cap.isOpened())
    {
        throw std::runtime_error(std::format("cannot open calibration video {}", path.string()));
    }
    return cap;
}

/** @brief Reads a single frame by index (seek; verified by the caller through re-detection). */
cv::Mat readFrameAt(cv::VideoCapture & cap, int frameIndex)
{
    cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    cv::Mat frame;
    cap.read(frame);
    return frame;
}

std::string timestampLabel(double seconds)
{
    return std::format("{:.2f}s", seconds);
}

/** @brief Writes an image, creating the directory first. Failure is a warning, not an error. */
bool writeImage(const std::filesystem::path & path, const cv::Mat & image, std::vector<std::string> & warnings)
{
    if (image.empty())
    {
        warnings.push_back(std::format("skipped {}: empty image", path.string()));
        return false;
    }
    if (!cv::imwrite(path.string(), image))
    {
        warnings.push_back(std::format("failed to write {}", path.string()));
        return false;
    }
    return true;
}

/** @brief Draws detected corners (green) and reprojected corners (red crosses) on a copy of a frame, then scales it. */
cv::Mat annotatedTile(const cv::Mat & frame, const CalibrationView & view, const Camera & camera,
                      const std::vector<cv::Point3f> & objectPoints, int tileWidth)
{
    cv::Mat img = frame.clone();
    const double scale = static_cast<double>(img.cols) / tileWidth;
    cv::Mat R;
    cv::Rodrigues(view.rvec, R);
    Eigen::Matrix3d Re;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            Re(r, c) = R.at<double>(r, c);
        }
    }
    const Eigen::Vector3d t(view.tvec[0], view.tvec[1], view.tvec[2]);
    const int thickness = std::max(1, static_cast<int>(std::lround(2.0 * scale / 2.0)));
    for (std::size_t i = 0; i < view.corners.size(); ++i)
    {
        cv::circle(img, view.corners[i], static_cast<int>(std::lround(8.0 * scale / 2.0)), cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
        const Eigen::Vector3d P(objectPoints[i].x, objectPoints[i].y, objectPoints[i].z);
        const Eigen::Vector2d p = camera.vectorToPixel(Re * P + t);
        if (p.allFinite())
        {
            const cv::Point2f q(static_cast<float>(p.x()), static_cast<float>(p.y()));
            const int s = static_cast<int>(std::lround(6.0 * scale / 2.0));
            cv::line(img, q - cv::Point2f(static_cast<float>(s), 0.f), q + cv::Point2f(static_cast<float>(s), 0.f), cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
            cv::line(img, q - cv::Point2f(0.f, static_cast<float>(s)), q + cv::Point2f(0.f, static_cast<float>(s)), cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);
        }
    }
    // Mark the first corner so that the grid orientation is visible.
    if (!view.corners.empty())
    {
        cv::circle(img, view.corners.front(), static_cast<int>(std::lround(14.0 * scale / 2.0)), cv::Scalar(255, 0, 255), thickness, cv::LINE_AA);
    }
    const double rmsValue = view.rmsPx;
    const std::string label = std::format("frame {} t={} rms={:.3f}px", view.frameIndex, timestampLabel(view.timeSeconds), rmsValue);
    cv::Mat tile;
    cv::resize(img, tile, cv::Size(tileWidth, static_cast<int>(std::lround(img.rows / scale))), 0, 0, cv::INTER_AREA);
    cv::putText(tile, label, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(tile, label, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return tile;
}

/** @brief Bar chart of per-view RMS errors with the median marked. */
cv::Mat errorBarChart(const std::vector<CalibrationView> & views, double medianRms, double overallRms)
{
    const int width = 1200;
    const int height = 480;
    const int left = 70;
    const int bottom = 420;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    double maxVal = 0.5;
    for (const auto & v : views)
    {
        maxVal = std::max(maxVal, v.rmsPx);
    }
    maxVal *= 1.1;
    const int plotW = width - left - 20;
    const int plotH = bottom - 40;
    const double barW = static_cast<double>(plotW) / std::max<std::size_t>(1, views.size());
    for (std::size_t i = 0; i < views.size(); ++i)
    {
        const double h = plotH * views[i].rmsPx / maxVal;
        const int x0 = left + static_cast<int>(std::lround(i * barW + 1));
        const int x1 = left + static_cast<int>(std::lround((i + 1) * barW - 1));
        const int y0 = bottom - static_cast<int>(std::lround(h));
        cv::rectangle(plot, cv::Point(x0, y0), cv::Point(std::max(x0 + 1, x1), bottom), cv::Scalar(180, 120, 40), cv::FILLED);
    }
    cv::line(plot, cv::Point(left, bottom), cv::Point(width - 20, bottom), cv::Scalar(0, 0, 0), 2);
    cv::line(plot, cv::Point(left, 40), cv::Point(left, bottom), cv::Scalar(0, 0, 0), 2);
    const int yMed = bottom - static_cast<int>(std::lround(plotH * medianRms / maxVal));
    const int yAll = bottom - static_cast<int>(std::lround(plotH * overallRms / maxVal));
    cv::line(plot, cv::Point(left, yMed), cv::Point(width - 20, yMed), cv::Scalar(0, 150, 0), 2);
    cv::line(plot, cv::Point(left, yAll), cv::Point(width - 20, yAll), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    cv::putText(plot, std::format("per-view RMS [px] (n={}); green = median {:.3f}, red = overall {:.3f}", views.size(), medianRms, overallRms),
                cv::Point(left, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(plot, std::format("{:.2f}", maxVal), cv::Point(8, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::putText(plot, "0", cv::Point(left - 20, bottom), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    return plot;
}

/** @brief Writes the human-readable calibration report. */
void writeReport(const std::filesystem::path & path, const CalibrationResult & r, const ChessboardConfig & cfg)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error(std::format("cannot write report {}", path.string()));
    }
    const Camera & cam = r.camera;
    out << std::format("Camera calibration report\n");
    out << std::format("config: {}\n", cfg.configPath.string());
    out << std::format("videos: {}\n", cfg.videos.size());
    for (const auto & v : cfg.videos)
    {
        out << std::format("  {}\n", v.string());
    }
    out << std::format("image size: {}x{}\n", cam.imageWidth(), cam.imageHeight());
    out << std::format("board inner corners: {} x {} (cols x rows), square size {} m\n", cfg.innerCornersPerRow, cfg.innerCornersPerColumn, cfg.squareSizeM);
    out << std::format("frames inspected: {}, boards detected: {}, quality candidates: {}, final views: {}, points: {}\n",
                       r.numFramesInspected, r.numBoardsDetected, r.numCandidateViews, r.views.size(), r.numPoints);
    out << std::format("orientation flips corrected: {}\n", r.numOrientationFlips);
    out << std::format("elapsed: {:.2f} s\n\n", r.elapsedSeconds);
    out << std::format("selected model: {}\n", r.selectedModel);
    out << std::format("RMS reprojection error (in-sample, Camera model): {:.4f} px\n", r.rmsPx);
    out << std::format("RMS reported by cv::calibrateCamera:             {:.4f} px\n\n", r.cvRmsPx);
    out << std::format("fx = {:.4f}\nfy = {:.4f}\ncx = {:.4f}\ncy = {:.4f}\nskew = {:.6f}\n", cam.fx(), cam.fy(), cam.cx(), cam.cy(), cam.skew());
    const auto & d = cam.distortionCoefficients();
    static const char * names[] = {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"};
    for (std::size_t i = 0; i < d.size(); ++i)
    {
        out << std::format("{} = {:.6g}\n", names[i], d[i]);
    }
    out << "\n1-sigma (OpenCV, intrinsics then distortion):\n";
    for (std::size_t i = 0; i < r.stdIntrinsics.size(); ++i)
    {
        out << std::format("  {:.6g}", r.stdIntrinsics[i]);
    }
    out << "\n\nmodel scores (5-fold view-level cross-validation, pooled held-out RMS):\n";
    for (const auto & s : r.modelScores)
    {
        out << std::format("  {:<18} converged={} cvRms={:.4f} px", s.name, s.converged, s.cvRmsPx);
        if (!s.failure.empty())
        {
            out << "  failure: " << s.failure;
        }
        out << "\n";
    }
    out << "\nsanity checks:\n";
    for (const auto & c : r.checks)
    {
        out << "  " << c << "\n";
    }
    if (!r.warnings.empty())
    {
        out << "\nwarnings:\n";
        for (const auto & w : r.warnings)
        {
            out << "  " << w << "\n";
        }
    }
    if (!r.rejectionLog.empty())
    {
        out << "\nrejections (first 200):\n";
        for (std::size_t i = 0; i < r.rejectionLog.size() && i < 200; ++i)
        {
            out << "  " << r.rejectionLog[i] << "\n";
        }
    }
    out << "\nviews used in the final fit:\n";
    out << "  idx video frame   time_s   corners_rms_px  sharpness  area   curvature\n";
    for (std::size_t i = 0; i < r.views.size(); ++i)
    {
        const auto & v = r.views[i];
        out << std::format("  {:3d} {:5d} {:7d} {:9.2f} {:14.4f} {:10.1f} {:6.3f} {:10.4f}\n", i, v.videoIndex, v.frameIndex, v.timeSeconds,
                           v.rmsPx, v.sharpness, v.boardAreaFraction, v.gridCurvatureFraction);
    }
}

/** @brief Writes the accepted views (corners, pose, errors) as OpenCV XML: reproducible fixture for the calibration. */
void writeViewsXml(const std::filesystem::path & path, const CalibrationResult & r, const ChessboardConfig & cfg)
{
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        throw std::runtime_error(std::format("cannot write {}", path.string()));
    }
    fs.writeComment("Accepted chessboard views used for the final camera fit (corners in row-major order)");
    fs << "image_width" << r.camera.imageWidth();
    fs << "image_height" << r.camera.imageHeight();
    fs << "board_inner_corners_cols" << cfg.innerCornersPerRow;
    fs << "board_inner_corners_rows" << cfg.innerCornersPerColumn;
    fs << "square_size_m" << cfg.squareSizeM;
    fs << "selected_model" << r.selectedModel;
    fs << "views" << "[";
    for (const auto & v : r.views)
    {
        cv::Mat corners(static_cast<int>(v.corners.size()), 2, CV_64F);
        for (std::size_t i = 0; i < v.corners.size(); ++i)
        {
            corners.at<double>(static_cast<int>(i), 0) = v.corners[i].x;
            corners.at<double>(static_cast<int>(i), 1) = v.corners[i].y;
        }
        cv::Mat rvec = cv::Mat(cv::Vec3d(v.rvec[0], v.rvec[1], v.rvec[2])).reshape(1, 3).clone();
        cv::Mat tvec = cv::Mat(cv::Vec3d(v.tvec[0], v.tvec[1], v.tvec[2])).reshape(1, 3).clone();
        fs << "{" << "video_index" << v.videoIndex << "frame_index" << v.frameIndex << "time_s" << v.timeSeconds
           << "rms_px" << v.rmsPx << "rvec" << rvec << "tvec" << tvec << "corners" << corners << "}";
    }
    fs << "]";
    fs.release();
}

/** @brief Builds the visualisations and report under outputDirectory. Errors here are warnings. */
void writeVisualisation(const CalibrationResult & r, const ChessboardConfig & cfg, const CalibrationOptions & options)
{
    std::error_code ec;
    std::filesystem::create_directories(options.outputDirectory, ec);
    if (ec)
    {
        throw std::runtime_error(std::format("cannot create output directory {}: {}", options.outputDirectory.string(), ec.message()));
    }
    const std::string scene = cfg.configPath.parent_path().filename().string();
    const cv::Size inner(cfg.innerCornersPerRow, cfg.innerCornersPerColumn);
    const std::vector<cv::Point3f> objectPoints = boardObjectPoints(inner, cfg.squareSizeM);
    std::vector<std::string> warnings;
    const std::string prefix = "calibration_" + scene;

    // Open each source video once; views refer to them by index.
    std::map<int, cv::VideoCapture> captures;
    auto captureFor = [&](int videoIndex) -> cv::VideoCapture &
    {
        auto it = captures.find(videoIndex);
        if (it == captures.end())
        {
            it = captures.emplace(videoIndex, openVideo(cfg.videos[static_cast<std::size_t>(videoIndex)])).first;
        }
        return it->second;
    };

    // (1) Four evenly spaced views with detected corners (green) and reprojected corners (red).
    std::vector<cv::Mat> tiles;
    const int tileWidth = 960;
    std::vector<std::size_t> chosen;
    if (!r.views.empty())
    {
        for (int k = 0; k < 4; ++k)
        {
            chosen.push_back(static_cast<std::size_t>(std::llround(static_cast<double>(k) * (r.views.size() - 1) / 3.0)));
        }
    }
    int seekMismatches = 0;
    double worstSeekDiff = 0.0;
    for (std::size_t idx : chosen)
    {
        const CalibrationView & v = r.views[idx];
        cv::Mat frame = readFrameAt(captureFor(v.videoIndex), v.frameIndex);
        if (frame.empty())
        {
            warnings.push_back(std::format("could not seek to frame {} for visualisation", v.frameIndex));
            continue;
        }
        // Verify that the seeked frame reproduces the stored detection.
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> redetected;
        if (detectBoard(gray, inner, redetected))
        {
            const double diff = cornerRms(redetected, v.corners);
            worstSeekDiff = std::max(worstSeekDiff, diff);
            if (diff > 0.5)
            {
                ++seekMismatches;
            }
        }
        else
        {
            ++seekMismatches;
        }
        tiles.push_back(annotatedTile(frame, v, r.camera, objectPoints, tileWidth));
    }
    if (tiles.size() == 4)
    {
        cv::Mat montage(tiles[0].rows * 2, tileWidth * 2, CV_8UC3, cv::Scalar(0, 0, 0));
        tiles[0].copyTo(montage(cv::Rect(0, 0, tileWidth, tiles[0].rows)));
        tiles[1].copyTo(montage(cv::Rect(tileWidth, 0, tileWidth, tiles[1].rows)));
        tiles[2].copyTo(montage(cv::Rect(0, tiles[0].rows, tileWidth, tiles[2].rows)));
        tiles[3].copyTo(montage(cv::Rect(tileWidth, tiles[0].rows, tileWidth, tiles[3].rows)));
        writeImage(options.outputDirectory / (prefix + "_corners.png"), montage, warnings);
    }
    else if (!tiles.empty())
    {
        cv::Mat montage(tiles[0].rows * 2, tileWidth * 2, CV_8UC3, cv::Scalar(0, 0, 0));
        for (std::size_t i = 0; i < tiles.size() && i < 4; ++i)
        {
            tiles[i].copyTo(montage(cv::Rect(static_cast<int>(i % 2) * tileWidth, static_cast<int>(i / 2) * tiles[0].rows, tileWidth, tiles[i].rows)));
        }
        writeImage(options.outputDirectory / (prefix + "_corners.png"), montage, warnings);
    }

    // (2) Undistorted frame next to the original, from the middle view.
    if (!r.views.empty())
    {
        const CalibrationView & mid = r.views[r.views.size() / 2];
        cv::Mat frame = readFrameAt(captureFor(mid.videoIndex), mid.frameIndex);
        if (!frame.empty())
        {
            cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
            K.at<double>(0, 0) = r.camera.fx();
            K.at<double>(0, 2) = r.camera.cx();
            K.at<double>(1, 1) = r.camera.fy();
            K.at<double>(1, 2) = r.camera.cy();
            const auto & dc = r.camera.distortionCoefficients();
            cv::Mat dist(1, static_cast<int>(dc.size()), CV_64F);
            for (std::size_t i = 0; i < dc.size(); ++i)
            {
                dist.at<double>(0, static_cast<int>(i)) = dc[i];
            }
            cv::Mat und;
            cv::undistort(frame, und, K, dist);
            cv::Mat a;
            cv::Mat b;
            cv::resize(frame, a, cv::Size(tileWidth, static_cast<int>(std::lround(frame.rows * double(tileWidth) / frame.cols))), 0, 0, cv::INTER_AREA);
            cv::resize(und, b, a.size(), 0, 0, cv::INTER_AREA);
            cv::Mat pair(a.rows, a.cols * 2 + 10, CV_8UC3, cv::Scalar(255, 255, 255));
            a.copyTo(pair(cv::Rect(0, 0, a.cols, a.rows)));
            b.copyTo(pair(cv::Rect(a.cols + 10, 0, b.cols, b.rows)));
            cv::putText(pair, "original (distorted)", cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(pair, "undistorted with camera.xml", cv::Point(a.cols + 22, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            writeImage(options.outputDirectory / (prefix + "_undistorted.png"), pair, warnings);
        }
    }

    // (3) Per-view reprojection error chart.
    std::vector<double> rmsValues;
    for (const auto & v : r.views)
    {
        rmsValues.push_back(v.rmsPx);
    }
    writeImage(options.outputDirectory / (prefix + "_per_view_error.png"),
               errorBarChart(r.views, median(rmsValues), r.rmsPx), warnings);

    // (4) Text report and the accepted views (corner fixture for reproducing the RMS).
    try
    {
        writeReport(options.outputDirectory / (prefix + "_report.txt"), r, cfg);
    }
    catch (const std::exception & e)
    {
        warnings.push_back(e.what());
    }
    try
    {
        writeViewsXml(options.outputDirectory / (prefix + "_views.xml"), r, cfg);
    }
    catch (const std::exception & e)
    {
        warnings.push_back(e.what());
    }
    std::println("Visualisations written to {}", options.outputDirectory.string());
    if (seekMismatches > 0)
    {
        std::println("WARNING: {} visualisation frame(s) did not reproduce the stored corners (worst {:.3f} px)", seekMismatches, worstSeekDiff);
    }
    else
    {
        std::println("Seek check: visualisation frames reproduce stored corners (worst {:.3f} px)", worstSeekDiff);
    }
    for (const auto & w : warnings)
    {
        std::println("WARNING: {}", w);
    }
}

void printSummary(const CalibrationResult & r)
{
    const Camera & c = r.camera;
    std::println("");
    std::println("=== Calibration summary ===");
    std::println("views used: {} (points {}), model: {}", r.views.size(), r.numPoints, r.selectedModel);
    std::println("RMS reprojection error: {:.4f} px (cv::calibrateCamera: {:.4f} px)", r.rmsPx, r.cvRmsPx);
    std::println("image size: {}x{}", c.imageWidth(), c.imageHeight());
    std::println("fx={:.3f} fy={:.3f} cx={:.3f} cy={:.3f} skew={:.4g}", c.fx(), c.fy(), c.cx(), c.cy(), c.skew());
    std::print("distortion:");
    for (double k : c.distortionCoefficients())
    {
        std::print(" {:.6g}", k);
    }
    std::println("");
    for (const auto & s : r.modelScores)
    {
        std::println("CV {:<18} converged={} held-out RMS={:.4f} px{}", s.name, s.converged, s.cvRmsPx,
                     s.failure.empty() ? std::string() : std::format(" failure: {}", s.failure));
    }
    for (const auto & line : r.checks)
    {
        std::println("{}", line);
    }
    for (const auto & w : r.warnings)
    {
        std::println("WARNING: {}", w);
    }
    std::println("runtime: {:.2f} s", r.elapsedSeconds);
}
} // namespace

ChessboardConfig readChessboardConfig(const std::filesystem::path & configPath)
{
    if (!std::filesystem::is_regular_file(configPath))
    {
        throw std::runtime_error(std::format("calibration configuration not found: {}", configPath.string()));
    }
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw std::runtime_error(std::format("cannot parse calibration configuration {}", configPath.string()));
    }
    const std::string where = configPath.string();
    cv::FileNode data = fs["chessboard_data"];
    if (data.isNone())
    {
        throw std::runtime_error(std::format("{}: missing <chessboard_data> element", where));
    }
    cv::FileNode regexNode = data["file_regex"];
    if (regexNode.isNone())
    {
        throw std::runtime_error(std::format("{}: missing <file_regex>", where));
    }
    ChessboardConfig cfg;
    cfg.configPath = configPath;
    cfg.fileRegex = stripQuotesAndSpaces(regexNode.string());
    if (cfg.fileRegex.empty())
    {
        throw std::runtime_error(std::format("{}: <file_regex> is empty", where));
    }

    cv::FileNode board = data["chessboard"];
    if (board.isNone())
    {
        throw std::runtime_error(std::format("{}: missing <chessboard> element", where));
    }
    if (board["grid_width"].isNone() || board["grid_height"].isNone() || board["square_size"].isNone())
    {
        throw std::runtime_error(std::format("{}: <chessboard> needs grid_width, grid_height and square_size", where));
    }
    int gw = 0;
    int gh = 0;
    double square = 0.0;
    board["grid_width"] >> gw;
    board["grid_height"] >> gh;
    board["square_size"] >> square;
    if (gw < 2 || gh < 2 || gw > 64 || gh > 64)
    {
        throw std::runtime_error(std::format("{}: grid_width/grid_height must be inner-corner counts in [2, 64], got {}x{}", where, gw, gh));
    }
    if (!(square > 0.0) || !std::isfinite(square) || square > 1.0)
    {
        throw std::runtime_error(std::format("{}: square_size must be in (0, 1] metres, got {}", where, square));
    }
    cfg.innerCornersPerRow = gw;
    cfg.innerCornersPerColumn = gh;
    cfg.squareSizeM = square;

    // Resolve file_regex against the directory entries (whole file name match).
    std::regex pattern;
    try
    {
        pattern = std::regex(cfg.fileRegex, std::regex::ECMAScript);
    }
    catch (const std::regex_error & e)
    {
        throw std::runtime_error(std::format("{}: invalid file_regex '{}': {}", where, cfg.fileRegex, e.what()));
    }
    const std::filesystem::path dir = configPath.parent_path().empty() ? std::filesystem::path(".") : configPath.parent_path();
    for (const auto & entry : std::filesystem::directory_iterator(dir))
    {
        if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), pattern))
        {
            cfg.videos.push_back(entry.path());
        }
    }
    std::sort(cfg.videos.begin(), cfg.videos.end());
    if (cfg.videos.empty())
    {
        throw std::runtime_error(std::format("{}: no file in {} matches file_regex '{}'", where, dir.string(), cfg.fileRegex));
    }
    return cfg;
}

CalibrationResult calibrateFromViews(cv::Size imageSize, cv::Size innerCorners, double squareSizeM,
                                     const std::vector<CalibrationView> & inputViews, const CalibrationOptions & options)
{
    if (imageSize.width <= 0 || imageSize.height <= 0)
    {
        throw std::invalid_argument("calibrateFromViews: image size must be positive");
    }
    if (innerCorners.width < 2 || innerCorners.height < 2)
    {
        throw std::invalid_argument("calibrateFromViews: pattern must have at least 2x2 inner corners");
    }
    if (!(squareSizeM > 0.0))
    {
        throw std::invalid_argument("calibrateFromViews: square size must be positive");
    }
    const std::size_t expected = static_cast<std::size_t>(innerCorners.area());
    for (const auto & v : inputViews)
    {
        if (v.corners.size() != expected)
        {
            throw std::invalid_argument(std::format("calibrateFromViews: view has {} corners, expected {}", v.corners.size(), expected));
        }
    }
    if (static_cast<int>(inputViews.size()) < options.minViews)
    {
        throw std::runtime_error(std::format("only {} usable views, at least {} are required", inputViews.size(), options.minViews));
    }

    CalibrationResult result;
    const std::vector<cv::Point3f> objectPoints = boardObjectPoints(innerCorners, squareSizeM);
    std::vector<std::vector<cv::Point3f>> allObj(inputViews.size(), objectPoints);
    std::vector<std::vector<cv::Point2f>> allImg;
    allImg.reserve(inputViews.size());
    for (const auto & v : inputViews)
    {
        allImg.push_back(v.corners);
    }

    // Principal point: fixed at the image centre by default. Measured on the project data (views split by time and
    // by board tilt): the free principal point moved by up to 25 px between halves (outdoor) and the focal length
    // by up to 27% (indoor), while the fixed-centre model keeps outdoor fx within 0.8% at a 2% RMS cost.
    const int baseFlags = options.fixPrincipalPointAtCentre ? cv::CALIB_FIX_PRINCIPAL_POINT : 0;

    // Stage 1: fit the standard model, iteratively drop views with RMS > factor * median.
    std::vector<int> active(inputViews.size());
    for (std::size_t i = 0; i < active.size(); ++i)
    {
        active[i] = static_cast<int>(i);
    }
    auto gather = [&](const std::vector<int> & idx, std::vector<std::vector<cv::Point3f>> & o, std::vector<std::vector<cv::Point2f>> & im)
    {
        o.clear();
        im.clear();
        for (int i : idx)
        {
            o.push_back(allObj[static_cast<std::size_t>(i)]);
            im.push_back(allImg[static_cast<std::size_t>(i)]);
        }
    };
    Fit fit;
    for (int iteration = 0; iteration < 5; ++iteration)
    {
        std::vector<std::vector<cv::Point3f>> o;
        std::vector<std::vector<cv::Point2f>> im;
        gather(active, o, im);
        fit = fitIntrinsics(o, im, imageSize, baseFlags);
        requireFiniteOrThrow(fit.cvRms, "RMS");
        const Camera camera = cameraFromFit(fit.K, fit.dist, imageSize);
        std::vector<double> rms(active.size());
        for (std::size_t k = 0; k < active.size(); ++k)
        {
            const std::size_t i = static_cast<std::size_t>(active[k]);
            rms[k] = std::sqrt(viewSquaredError(camera, allObj[i], allImg[i], toVec3(fit.rvecs[k]), toVec3(fit.tvecs[k])) /
                               static_cast<double>(allImg[i].size()));
        }
        const double med = median(rms);
        const double threshold = options.outlierViewFactor * med;
        std::vector<int> keep;
        std::vector<int> drop;
        for (std::size_t k = 0; k < active.size(); ++k)
        {
            // Gross detection errors: one corner far from the model. The view-level RMS can be masked when
            // few views are available (the outlier pulls the fit towards itself), so check corners directly.
            double worstCorner = 0.0;
            {
                const std::size_t i = static_cast<std::size_t>(active[k]);
                cv::Mat R;
                cv::Rodrigues(toVec3(fit.rvecs[k]), R);
                Eigen::Matrix3d Re;
                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        Re(r, c) = R.at<double>(r, c);
                    }
                }
                const Eigen::Vector3d t(fit.tvecs[k].at<double>(0), fit.tvecs[k].at<double>(1), fit.tvecs[k].at<double>(2));
                for (std::size_t j = 0; j < allObj[i].size(); ++j)
                {
                    const Eigen::Vector3d P(allObj[i][j].x, allObj[i][j].y, allObj[i][j].z);
                    const Eigen::Vector2d e = camera.vectorToPixel(Re * P + t) - Eigen::Vector2d(allImg[i][j].x, allImg[i][j].y);
                    worstCorner = std::max(worstCorner, e.norm());
                }
            }
            if (rms[k] > threshold || worstCorner > options.grossCornerErrorPx)
            {
                drop.push_back(active[k]);
            }
            else
            {
                keep.push_back(active[k]);
            }
        }
        if (drop.empty())
        {
            break;
        }
        if (static_cast<int>(keep.size()) < options.minViews)
        {
            result.warnings.push_back(std::format("outlier rejection would leave {} views; keeping all {}", keep.size(), active.size()));
            break;
        }
        for (int i : drop)
        {
            result.rejectionLog.push_back(std::format("outlier view: frame {} (video {}) above view-RMS threshold {:.2f} px or corner gate {:.1f} px",
                                                      inputViews[static_cast<std::size_t>(i)].frameIndex,
                                                      inputViews[static_cast<std::size_t>(i)].videoIndex, threshold, options.grossCornerErrorPx));
        }
        active = keep;
    }

    // Stage 2: cross-validated distortion-model selection (standard 5 vs rational 8 coefficients).
    const ModelScore standardScore = crossValidate(allObj, allImg, active, imageSize, baseFlags, false, options.crossValidationFolds);
    const ModelScore rationalScore = crossValidate(allObj, allImg, active, imageSize, baseFlags | cv::CALIB_RATIONAL_MODEL, true, options.crossValidationFolds);
    result.modelScores = {standardScore, rationalScore};
    if (!standardScore.converged && !rationalScore.converged)
    {
        throw std::runtime_error("neither distortion model converged in cross-validation");
    }
    // Deterministic rule: the rational model is selected only when allowRationalModel is set; otherwise the
    // standard 5-coefficient model is used, even if the rational cross-validation is better (recorded as a warning).
    const bool rationalWouldWin = rationalScore.converged &&
        (!standardScore.converged || rationalScore.cvRmsPx < standardScore.cvRmsPx * (1.0 - options.modelSwitchMargin));
    const bool useRational = options.allowRationalModel && rationalWouldWin;
    if (rationalWouldWin && !options.allowRationalModel)
    {
        result.warnings.push_back("rational 8-coefficient model cross-validates better, but allowRationalModel is false; "
                                  "the standard 5-coefficient model is used");
    }
    result.selectedModel = modelName(useRational);

    // Stage 3: final fit on the inlier views with the selected model.
    std::vector<std::vector<cv::Point3f>> o;
    std::vector<std::vector<cv::Point2f>> im;
    gather(active, o, im);
    fit = fitIntrinsics(o, im, imageSize, baseFlags | (useRational ? cv::CALIB_RATIONAL_MODEL : 0));
    requireFiniteOrThrow(fit.cvRms, "RMS");
    const Camera camera = cameraFromFit(fit.K, fit.dist, imageSize);

    double sseAll = 0.0;
    std::size_t pointsAll = 0;
    for (std::size_t k = 0; k < active.size(); ++k)
    {
        const std::size_t i = static_cast<std::size_t>(active[k]);
        CalibrationView v = inputViews[i];
        v.rvec = toVec3(fit.rvecs[k]);
        v.tvec = toVec3(fit.tvecs[k]);
        const double sse = viewSquaredError(camera, allObj[i], allImg[i], v.rvec, v.tvec);
        v.rmsPx = std::sqrt(sse / static_cast<double>(v.corners.size()));
        sseAll += sse;
        pointsAll += v.corners.size();
        result.views.push_back(v);
    }
    result.numPoints = static_cast<int>(pointsAll);
    result.rmsPx = std::sqrt(sseAll / static_cast<double>(pointsAll));
    result.cvRmsPx = fit.cvRms;
    result.stdIntrinsics.clear();
    for (int i = 0; i < fit.stdIntrinsics.rows; ++i)
    {
        result.stdIntrinsics.push_back(fit.stdIntrinsics.at<double>(i));
    }

    // Build the camera with metadata.
    Camera::CalibrationMetadata meta;
    meta.rmsReprojectionErrorPx = result.rmsPx;
    meta.numViews = static_cast<int>(result.views.size());
    meta.numPoints = result.numPoints;
    meta.boardInnerCornersCols = innerCorners.width;
    meta.boardInnerCornersRows = innerCorners.height;
    meta.squareSizeM = squareSizeM;
    meta.distortionModel = result.selectedModel;
    result.camera = camera;
    result.camera.setMetadata(meta);

    // Sanity checks (reported, not fatal).
    const double cxRel = std::abs(camera.cx() - imageSize.width / 2.0) / imageSize.width;
    const double cyRel = std::abs(camera.cy() - imageSize.height / 2.0) / imageSize.height;
    const double aspect = std::abs(camera.fx() - camera.fy()) / camera.fx();
    auto check = [&](bool ok, const std::string & text)
    {
        result.checks.push_back(std::format("{} {}", ok ? "PASS" : "WARN", text));
        if (!ok)
        {
            result.warnings.push_back(text);
        }
    };
    check(result.rmsPx < 1.0, std::format("RMS reprojection error {:.3f} px < 1.0 px", result.rmsPx));
    check(std::abs(result.rmsPx - result.cvRmsPx) < 1e-3, std::format("in-sample RMS matches cv::calibrateCamera ({:.4f} vs {:.4f} px)", result.rmsPx, result.cvRmsPx));
    check(cxRel < 0.1 && cyRel < 0.1, std::format("principal point within 10% of image centre (dx={:.3f}, dy={:.3f} of size)", cxRel, cyRel));
    check(aspect < 0.02, std::format("square pixels: |fx-fy|/fx = {:.4f} < 0.02", aspect));
    double maxAbsK = 0.0;
    for (double k : camera.distortionCoefficients())
    {
        maxAbsK = std::max(maxAbsK, std::abs(k));
    }
    check(maxAbsK < 2.0, std::format("distortion coefficients moderate (max |k| = {:.4f})", maxAbsK));
    // Invertibility of the distortion map over the image (no fold-over inside the frame).
    double minDet = std::numeric_limits<double>::infinity();
    bool undefined = false;
    for (int gy = 0; gy <= 8; ++gy)
    {
        for (int gx = 0; gx <= 12; ++gx)
        {
            const Eigen::Vector2d p(gx * (imageSize.width - 1) / 12.0, gy * (imageSize.height - 1) / 8.0);
            const Eigen::Vector2d n = camera.pixelToNormalised(p);
            if (!n.allFinite())
            {
                undefined = true;
                continue;
            }
            minDet = std::min(minDet, camera.distortionJacobianDeterminant(n.x(), n.y()));
        }
    }
    check(!undefined && minDet > 0.0, std::format("distortion map invertible over the image (min det = {:.4f})", minDet));
    check(result.views.size() >= static_cast<std::size_t>(options.minViews), std::format("{} views >= {} required", result.views.size(), options.minViews));
    if (options.fixPrincipalPointAtCentre)
    {
        // Informational only: what the free principal point would have given on the same views.
        const Fit freeFit = fitIntrinsics(o, im, imageSize, useRational ? cv::CALIB_RATIONAL_MODEL : 0);
        result.checks.push_back(std::format("INFO free principal point (not used): cx={:.1f} cy={:.1f} fx={:.1f} fy={:.1f} RMS={:.4f} px",
                                            freeFit.K.at<double>(0, 2), freeFit.K.at<double>(1, 2), freeFit.K.at<double>(0, 0),
                                            freeFit.K.at<double>(1, 1), freeFit.cvRms));
    }
    return result;
}

CalibrationResult runCalibration(const std::filesystem::path & configPath, const CalibrationOptions & options)
{
    const auto start = std::chrono::steady_clock::now();
    const ChessboardConfig cfg = readChessboardConfig(configPath);
    const cv::Size inner(cfg.innerCornersPerRow, cfg.innerCornersPerColumn);
    if (options.verbose)
    {
        std::println("Chessboard: {}x{} inner corners, square {} m", inner.width, inner.height, cfg.squareSizeM);
        for (const auto & v : cfg.videos)
        {
            std::println("Calibration video: {}", v.string());
        }
    }
    if (options.frameStride < 1)
    {
        throw std::invalid_argument("frameStride must be >= 1");
    }

    // Stage A: decode, detect, collect every detection.
    std::vector<DetectionRecord> detections;
    cv::Size imageSize;
    int inspected = 0;
    int boards = 0;
    std::vector<std::string> rejections;
    std::vector<std::string> warnings;
    for (std::size_t vi = 0; vi < cfg.videos.size(); ++vi)
    {
        cv::VideoCapture cap = openVideo(cfg.videos[vi]);
        const cv::Size size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        if (vi == 0)
        {
            imageSize = size;
        }
        else if (size != imageSize)
        {
            throw std::runtime_error(std::format("{}: frame size {}x{} differs from {}x{}", cfg.videos[vi].string(), size.width, size.height, imageSize.width, imageSize.height));
        }
        if (!(fps > 0.0))
        {
            warnings.push_back(std::format("{}: frame rate unknown, timestamps set to 0", cfg.videos[vi].string()));
        }
        int frameIndex = 0;
        cv::Mat frame;
        while (true)
        {
            if (frameIndex % options.frameStride != 0)
            {
                if (!cap.grab())
                {
                    break;
                }
                ++frameIndex;
                continue;
            }
            if (!cap.read(frame) || frame.empty())
            {
                break;
            }
            ++inspected;
            cv::Mat gray;
            if (frame.channels() == 3)
            {
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            }
            else
            {
                gray = frame;
            }
            std::vector<cv::Point2f> corners;
            if (detectBoard(gray, inner, corners))
            {
                ++boards;
                DetectionRecord rec;
                rec.view.videoIndex = static_cast<int>(vi);
                rec.view.frameIndex = frameIndex;
                rec.view.timeSeconds = fps > 0.0 ? frameIndex / fps : 0.0;
                rec.view.corners = std::move(corners);
                rec.view.sharpness = boardSharpness(gray, rec.view.corners, inner);
                rec.view.boardAreaFraction = boardAreaFraction(rec.view.corners, imageSize);
                rec.view.gridCurvatureFraction = gridCurvatureFraction(rec.view.corners, inner);
                rec.cellSign = firstCellSign(gray, rec.view.corners, inner);
                detections.push_back(std::move(rec));
            }
            if (options.verbose && inspected % 100 == 0)
            {
                std::println("  inspected {} frames, {} boards (video {}, frame {})", inspected, boards, vi, frameIndex);
            }
            ++frameIndex;
        }
        if (options.verbose)
        {
            std::println("  {}: {} frames read, {} inspected in total", cfg.videos[vi].filename().string(), frameIndex, inspected);
        }
        cap.release();
    }
    if (detections.empty())
    {
        throw std::runtime_error(std::format("no chessboard detected in {} inspected frames", inspected));
    }

    // Stage B: canonical orientation. Reversing the corner order flips the first-cell brightness sign when
    // (cols + rows) is odd; the majority sign defines the canonical orientation.
    int flips = 0;
    if (((inner.width + inner.height) % 2) == 1)
    {
        int plus = 0;
        int minus = 0;
        for (const auto & d : detections)
        {
            plus += d.cellSign > 0 ? 1 : 0;
            minus += d.cellSign < 0 ? 1 : 0;
        }
        const int reference = plus >= minus ? 1 : -1;
        for (auto & d : detections)
        {
            if (d.cellSign != 0 && d.cellSign != reference)
            {
                std::reverse(d.view.corners.begin(), d.view.corners.end());
                ++flips;
            }
        }
        if (options.verbose)
        {
            std::println("orientation: {} views with first-cell sign +1, {} with -1; {} reversed to canonical", plus, minus, flips);
        }
    }
    else
    {
        warnings.push_back("(cols + rows) is even: colour-based orientation check unavailable; relying on detector order");
    }

    // Stage C: quality gates, then duplicate rejection (greedy, time order).
    std::vector<double> sharp;
    for (const auto & d : detections)
    {
        sharp.push_back(d.view.sharpness);
    }
    const double medianSharpness = median(sharp);
    const double separationPx = options.minSeparationFraction * imageSize.width;
    std::vector<CalibrationView> candidates;
    for (const auto & d : detections)
    {
        const CalibrationView & v = d.view;
        const std::string where = std::format("frame {} (video {})", v.frameIndex, v.videoIndex);
        if (v.boardAreaFraction < options.minBoardAreaFraction)
        {
            rejections.push_back(std::format("{}: board too small ({:.3f} of image area)", where, v.boardAreaFraction));
            continue;
        }
        if (!(v.gridCurvatureFraction <= options.maxGridCurvatureFraction))
        {
            rejections.push_back(std::format("{}: grid curvature {:.3f} above {:.3f} (corner outlier)", where, v.gridCurvatureFraction, options.maxGridCurvatureFraction));
            continue;
        }
        if (v.sharpness < options.minSharpnessFraction * medianSharpness)
        {
            rejections.push_back(std::format("{}: blurred (sharpness {:.1f} < {:.2f} x median {:.1f})", where, v.sharpness, options.minSharpnessFraction, medianSharpness));
            continue;
        }
        candidates.push_back(v);
    }
    std::vector<CalibrationView> accepted;
    for (const auto & c : candidates)
    {
        bool duplicate = false;
        for (const auto & a : accepted)
        {
            if (a.videoIndex == c.videoIndex && cornerRms(a.corners, c.corners) < separationPx)
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
        {
            rejections.push_back(std::format("frame {} (video {}): near-duplicate of an accepted view", c.frameIndex, c.videoIndex));
        }
        else
        {
            accepted.push_back(c);
        }
    }
    // Keep at most maxViews, evenly spread over the accepted sequence.
    std::vector<CalibrationView> selected;
    if (static_cast<int>(accepted.size()) > options.maxViews)
    {
        for (int k = 0; k < options.maxViews; ++k)
        {
            const std::size_t idx = static_cast<std::size_t>(std::llround(static_cast<double>(k) * (accepted.size() - 1) / (options.maxViews - 1)));
            selected.push_back(accepted[idx]);
        }
    }
    else
    {
        selected = accepted;
    }
    if (options.verbose)
    {
        std::println("views: {} detections, {} quality candidates, {} after duplicate rejection, {} selected",
                     detections.size(), candidates.size(), accepted.size(), selected.size());
    }
    if (static_cast<int>(selected.size()) < options.minViews)
    {
        throw std::runtime_error(std::format("only {} usable views after quality gates (need {}); see calibration log", selected.size(), options.minViews));
    }

    // Stage D: calibrate.
    CalibrationResult result = calibrateFromViews(imageSize, inner, cfg.squareSizeM, selected, options);
    result.numFramesInspected = inspected;
    result.numBoardsDetected = boards;
    result.numCandidateViews = static_cast<int>(candidates.size());
    result.numOrientationFlips = flips;
    result.rejectionLog.insert(result.rejectionLog.begin(), rejections.begin(), rejections.end());
    result.warnings.insert(result.warnings.begin(), warnings.begin(), warnings.end());
    std::string sources;
    for (const auto & v : cfg.videos)
    {
        sources += (sources.empty() ? "" : ", ") + v.filename().string();
    }
    Camera::CalibrationMetadata meta = result.camera.metadata();
    meta.sourceVideo = sources;
    result.camera.setMetadata(meta);
    result.elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (options.verbose)
    {
        printSummary(result);
    }
    if (!options.outputDirectory.empty())
    {
        writeVisualisation(result, cfg, options);
    }
    return result;
}

void calibrateCamera(const std::filesystem::path & configPath)
{
    CalibrationOptions options;
    // Visualisations go to <config dir>/../../out (the project's out/ folder for data/<scene>/config.xml).
    std::error_code ec;
    const std::filesystem::path outDir = std::filesystem::weakly_canonical(configPath.parent_path() / ".." / ".." / "out", ec);
    if (!ec)
    {
        options.outputDirectory = outDir;
    }

    const CalibrationResult result = runCalibration(configPath, options);

    const std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    // The .xml extension selects OpenCV XML (not YAML) output; the temporary name keeps it.
    const std::filesystem::path tmpPath = configPath.parent_path() / "camera.tmp.xml";
    result.camera.save(tmpPath);
    std::filesystem::rename(tmpPath, cameraPath);
    std::println("Wrote {}", cameraPath.string());
}
