#include "ArucoTags.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace tags
{

namespace
{

cv::aruco::DetectorParameters detectorParameters()
{
    cv::aruco::DetectorParameters p;
    p.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;   // sub-pixel corners: the measurement is the corner pixels
    return p;
}

} // namespace

Eigen::Matrix<double, 8, 1> TagDetection::cornerVector() const
{
    return cornersToVector(corners);
}

Eigen::Matrix<double, 4, 3> canonicalCorners(double side)
{
    if (!(side > 0.0) || !std::isfinite(side))
    {
        throw std::invalid_argument("tags::canonicalCorners: side length must be finite and positive");
    }
    const double h = 0.5 * side;
    Eigen::Matrix<double, 4, 3> c;
    c << -h, h, 0.0,
          h, h, 0.0,
          h, -h, 0.0,
         -h, -h, 0.0;
    return c;
}

Eigen::Matrix3d inPlaneRotation(int q)
{
    const int m = ((q % 4) + 4) % 4;
    const double a = 0.5 * std::numbers::pi_v<double> * static_cast<double>(m);
    const double c = std::cos(a);
    const double s = std::sin(a);
    Eigen::Matrix3d r;
    r << c, -s, 0.0,
         s, c, 0.0,
         0.0, 0.0, 1.0;
    return r;
}

Eigen::Matrix<double, 8, 1> cornersToVector(const Eigen::Matrix<double, 4, 2>& corners)
{
    Eigen::Matrix<double, 8, 1> v;
    for (int k = 0; k < 4; ++k)
    {
        v(2 * k) = corners(k, 0);
        v(2 * k + 1) = corners(k, 1);
    }
    return v;
}

Eigen::Matrix<double, 4, 2> vectorToCorners(const Eigen::Matrix<double, 8, 1>& v)
{
    Eigen::Matrix<double, 4, 2> c;
    for (int k = 0; k < 4; ++k)
    {
        c(k, 0) = v(2 * k);
        c(k, 1) = v(2 * k + 1);
    }
    return c;
}

ArucoTagDetector::ArucoTagDetector() : ArucoTagDetector(Options{}) {}

ArucoTagDetector::ArucoTagDetector(const Options& options)
    : options_(options),
      detector_(cv::aruco::getPredefinedDictionary(options.dictionary), detectorParameters())
{
    if (!(options_.minSidePx >= 0.0))
    {
        throw std::invalid_argument("ArucoTagDetector: minSidePx must be non-negative");
    }
}

std::vector<TagDetection> ArucoTagDetector::detect(const cv::Mat& image)
{
    if (image.empty())
    {
        throw std::invalid_argument("ArucoTagDetector::detect: empty image");
    }
    cv::Mat grey;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 1)
    {
        grey = image;
    }
    else
    {
        throw std::invalid_argument("ArucoTagDetector::detect: image must be grey or BGR");
    }

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;
    std::vector<int> ids;
    detector_.detectMarkers(grey, corners, ids, rejected);

    std::vector<TagDetection> out;
    out.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (ids[i] != options_.targetId || corners[i].size() != 4)
        {
            continue;
        }
        TagDetection d;
        d.id = ids[i];
        double side = 0.0;
        for (int k = 0; k < 4; ++k)
        {
            d.corners(k, 0) = static_cast<double>(corners[i][static_cast<std::size_t>(k)].x);
            d.corners(k, 1) = static_cast<double>(corners[i][static_cast<std::size_t>(k)].y);
        }
        for (int k = 0; k < 4; ++k)
        {
            side += (d.corners.row(k) - d.corners.row((k + 1) % 4)).norm();
        }
        d.sidePx = 0.25 * side;
        if (!d.corners.allFinite() || d.sidePx < options_.minSidePx)
        {
            continue;
        }
        out.push_back(d);
    }
    return out;
}

} // namespace tags
