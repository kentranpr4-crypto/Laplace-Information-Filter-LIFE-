// ArUco detection of identical-ID tags: canonical corner order, rotation, id filter and the in-plane frame rotations.
#include <cmath>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include "../../src/ArucoTags.h"
#include "testUtil.h"

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

namespace
{

/// Renders a 6X6_50 marker (id) with a white paper margin, warped so that its outer corners land on the given pixels.
/// The source corners are TL, TR, BR, BL of the printed pattern, which is the canonical tag order.
cv::Mat renderMarker(int id, const std::vector<cv::Point2f>& target, const cv::Size& size, int rotateQuarterTurns = 0)
{
    const int markerPx = 240;
    const int margin = 40;
    cv::Mat marker;
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50).generateImageMarker(id, markerPx, marker, 1);
    cv::Mat src(markerPx + 2 * margin, markerPx + 2 * margin, CV_8UC1, cv::Scalar(255));
    marker.copyTo(src(cv::Rect(margin, margin, markerPx, markerPx)));

    const std::vector<cv::Point2f> srcCorners = {
        cv::Point2f(static_cast<float>(margin), static_cast<float>(margin)),
        cv::Point2f(static_cast<float>(margin + markerPx), static_cast<float>(margin)),
        cv::Point2f(static_cast<float>(margin + markerPx), static_cast<float>(margin + markerPx)),
        cv::Point2f(static_cast<float>(margin), static_cast<float>(margin + markerPx))};
    const cv::Mat h = cv::getPerspectiveTransform(srcCorners, target);
    cv::Mat out;
    cv::warpPerspective(src, out, h, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(160));
    for (int q = 0; q < rotateQuarterTurns; ++q)
    {
        cv::Mat tmp;
        cv::rotate(out, tmp, cv::ROTATE_90_CLOCKWISE);
        out = tmp;
    }
    cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
    return out;
}

} // namespace

SCENARIO("canonical tag geometry and in-plane frame rotations")
{
    GIVEN("a tag of side 0.2 m")
    {
        const Matrix<double, 4, 3> c = tags::canonicalCorners(0.2);
        THEN("the corners are TL, TR, BR, BL in the tag frame (x right, y up, z towards the viewer)")
        {
            CHECK(c.row(0).isApprox(Eigen::RowVector3d(-0.1, 0.1, 0.0)));
            CHECK(c.row(1).isApprox(Eigen::RowVector3d(0.1, 0.1, 0.0)));
            CHECK(c.row(2).isApprox(Eigen::RowVector3d(0.1, -0.1, 0.0)));
            CHECK(c.row(3).isApprox(Eigen::RowVector3d(-0.1, -0.1, 0.0)));
        }
        THEN("the corner centroid is the tag origin and the corners form a square of side s")
        {
            CHECK(c.colwise().mean().norm() < 1e-14);
            CHECK(((c.row(0) - c.row(1)).norm() - 0.2) == doctest::Approx(0.0));
        }
    }
    GIVEN("the in-plane rotations Rz(q pi/2)")
    {
        for (int q = 0; q < 4; ++q)
        {
            const Matrix3d r = tags::inPlaneRotation(q);
            CHECK((r * r.transpose() - Matrix3d::Identity()).norm() < 1e-12);
            CHECK(r.determinant() == doctest::Approx(1.0));
            CHECK(r.col(2).isApprox(Vector3d::UnitZ()));   // the normal is kept
        }
        CHECK(tags::inPlaneRotation(4).isApprox(Matrix3d::Identity(), 1e-12));
        CHECK(tags::inPlaneRotation(-1).isApprox(tags::inPlaneRotation(3), 1e-12));
    }
}

SCENARIO("ArUco detection of the identical-ID tags with canonical corner order")
{
    GIVEN("a rendered id-5 marker of 6X6_50 (corners at a known quadrilateral)")
    {
        const std::vector<cv::Point2f> target = {cv::Point2f(300, 200), cv::Point2f(520, 215), cv::Point2f(505, 430),
                                                 cv::Point2f(290, 410)};
        const cv::Mat img = renderMarker(5, target, cv::Size(800, 600));
        tags::ArucoTagDetector detector;
        WHEN("the image is searched")
        {
            const std::vector<tags::TagDetection> dets = detector.detect(img);
            THEN("exactly one id-5 tag is found")
            {
                REQUIRE(dets.size() == 1);
                CHECK(dets[0].id == 5);
            }
            THEN("corner k is the k-th canonical corner of the printed pattern (TL, TR, BR, BL), within 1.5 px")
            {
                REQUIRE(dets.size() == 1);
                for (int k = 0; k < 4; ++k)
                {
                    const Vector2d expected(target[static_cast<std::size_t>(k)].x, target[static_cast<std::size_t>(k)].y);
                    CAPTURE_EIGEN(dets[0].corners.row(k));
                    CHECK((dets[0].corners.row(k).transpose() - expected).norm() < 1.5);
                }
            }
            THEN("the 8-vector convention matches the corner matrix")
            {
                REQUIRE(dets.size() == 1);
                CHECK(dets[0].cornerVector().isApprox(tags::cornersToVector(dets[0].corners)));
                CHECK(tags::vectorToCorners(dets[0].cornerVector()).isApprox(dets[0].corners));
            }
        }
    }
    GIVEN("the same marker rotated by 90 degrees in the image (the pattern is not rotation invariant)")
    {
        const std::vector<cv::Point2f> target = {cv::Point2f(300, 200), cv::Point2f(520, 215), cv::Point2f(505, 430),
                                                 cv::Point2f(290, 410)};
        const cv::Mat img = renderMarker(5, target, cv::Size(800, 600), 1);
        tags::ArucoTagDetector detector;
        WHEN("the image is searched")
        {
            const std::vector<tags::TagDetection> dets = detector.detect(img);
            THEN("the tag is still id 5, and corner 0 is still the physical TL corner (the rotation is resolved)")
            {
                REQUIRE(dets.size() == 1);
                CHECK(dets[0].id == 5);
                // A 90 deg clockwise rotation of an 800 x 600 image maps (x, y) -> (600 - 1 - y, x).
                const Vector2d tl(600.0 - 1.0 - target[0].y, target[0].x);
                CAPTURE_EIGEN(dets[0].corners.row(0));
                CHECK((dets[0].corners.row(0).transpose() - tl).norm() < 1.5);
            }
        }
    }
    GIVEN("a detector restricted to id 5 and a minimum size")
    {
        tags::ArucoTagDetector::Options opt;
        opt.minSidePx = 400.0;   // larger than the rendered tag (about 230 px)
        tags::ArucoTagDetector detector(opt);
        const std::vector<cv::Point2f> target = {cv::Point2f(300, 200), cv::Point2f(520, 215), cv::Point2f(505, 430),
                                                 cv::Point2f(290, 410)};
        const cv::Mat img = renderMarker(5, target, cv::Size(800, 600));
        THEN("the small tag is rejected by the size filter")
        {
            CHECK(detector.detect(img).empty());
        }
    }
    GIVEN("a rendered marker of another id")
    {
        const std::vector<cv::Point2f> target = {cv::Point2f(300, 200), cv::Point2f(520, 215), cv::Point2f(505, 430),
                                                 cv::Point2f(290, 410)};
        const cv::Mat img = renderMarker(7, target, cv::Size(800, 600));
        tags::ArucoTagDetector detector;
        THEN("the detector returns no tag (only the target id is kept)")
        {
            CHECK(detector.detect(img).empty());
        }
    }
    GIVEN("an image without markers")
    {
        tags::ArucoTagDetector detector;
        const cv::Mat blank(480, 640, CV_8UC1, cv::Scalar(128));
        THEN("no detections are returned, and an empty image is rejected")
        {
            CHECK(detector.detect(blank).empty());
            CHECK_THROWS_AS(detector.detect(cv::Mat()), std::invalid_argument);
        }
    }
}
