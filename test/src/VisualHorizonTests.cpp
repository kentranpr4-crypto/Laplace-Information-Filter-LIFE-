#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../../src/VisualHorizon.h"

namespace
{

/// Synthetic sky: bright, smooth, weakly noisy.
cv::Mat syntheticSky(int w, int h, unsigned seed)
{
    cv::RNG rng(seed);
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(225, 215, 205));
    cv::Mat noise(h, w, CV_8UC3);
    rng.fill(noise, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(3));
    cv::add(img, noise, img);
    return img;
}

/// Synthetic vegetation/ground: strongly textured, green-brown.
cv::Mat syntheticGround(int w, int h, unsigned seed)
{
    cv::RNG rng(seed);
    cv::Mat img(h, w, CV_8UC3);
    rng.fill(img, cv::RNG::UNIFORM, cv::Scalar(20, 60, 30), cv::Scalar(120, 170, 110));
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.7);
    return img;
}

/// Image with sky above row boundary and ground below, both textured as above.
cv::Mat skyOverGround(int w, int h, int boundary, unsigned seed)
{
    cv::Mat sky = syntheticSky(w, boundary, seed);
    cv::Mat ground = syntheticGround(w, h - boundary, seed + 1);
    cv::Mat out(h, w, CV_8UC3);
    sky.copyTo(out(cv::Rect(0, 0, w, boundary)));
    ground.copyTo(out(cv::Rect(0, boundary, w, h - boundary)));
    return out;
}

SkyGroundClassifier trainedClassifier()
{
    std::vector<cv::Mat> sky{syntheticSky(160, 90, 1u), syntheticSky(160, 90, 2u)};
    std::vector<cv::Mat> ground{syntheticGround(160, 90, 3u), syntheticGround(160, 90, 4u)};
    return SkyGroundClassifier::train(sky, ground);
}

} // namespace

SCENARIO("Sky/ground classifier separates synthetic sky from ground")
{
    GIVEN("a classifier trained on separate sky and ground patches")
    {
        const SkyGroundClassifier clf = trainedClassifier();
        REQUIRE(clf.trained());
        WHEN("it classifies an image whose horizon is at row 60 (new noise seeds)")
        {
            const cv::Mat img = skyOverGround(200, 120, 60, 77u);
            const cv::Mat lo = clf.logOdds(img);
            THEN("the log-odds have the expected sign away from the boundary band")
            {
                int skyOk = 0;
                int skyTotal = 0;
                int groundOk = 0;
                int groundTotal = 0;
                for (int y = 0; y < img.rows; ++y)
                {
                    for (int x = 0; x < img.cols; ++x)
                    {
                        const float v = lo.at<float>(y, x);
                        if (y < 52)
                        {
                            ++skyTotal;
                            skyOk += v > 0.0f ? 1 : 0;
                        }
                        else if (y > 68)
                        {
                            ++groundTotal;
                            groundOk += v < 0.0f ? 1 : 0;
                        }
                    }
                }
                CHECK(skyOk >= 0.98 * skyTotal);
                CHECK(groundOk >= 0.98 * groundTotal);
            }
        }
    }
}

SCENARIO("Horizon extraction recovers a smooth tilted boundary and rejects outliers")
{
    GIVEN("a smooth log-odds ramp across a boundary v = 40 + 0.05 (u - 100) at 1/4 analysis scale")
    {
        const int w = 200;
        const int h = 120;
        const double scale = 0.25;
        cv::Mat lo(h, w, CV_32FC1);
        auto boundaryRow = [](int u) { return 40.0 + 0.05 * (u - 100); };
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                lo.at<float>(y, x) = static_cast<float>(std::clamp(2.0 * (boundaryRow(x) - y), -5.0, 5.0));
            }
        }
        // Outliers: a cloud-like dark blob inside the sky and a bright patch inside the ground.
        cv::rectangle(lo, cv::Rect(20, 10, 10, 6), cv::Scalar(-4.0f), cv::FILLED);
        cv::rectangle(lo, cv::Rect(150, 95, 12, 8), cv::Scalar(4.0f), cv::FILLED);
        const int fullWidth = 800;
        WHEN("the horizon is extracted")
        {
            const HorizonEstimate est = extractHorizon(lo, scale, fullWidth);
            THEN("the estimate is valid with a wide span and many inliers")
            {
                CHECK(est.valid);
                CHECK(est.inliers.size() >= 40);
                CHECK(est.spanFraction > 0.8);
                CHECK(est.skyFraction > 0.3);
            }
            THEN("the fitted line lies within one full-resolution pixel of the true boundary")
            {
                for (int u = 10; u < w - 10; u += 10)
                {
                    // analysis (u, v) -> full-resolution pixel centres
                    const double uFull = (u + 0.5) / scale - 0.5;
                    const double vFull = (boundaryRow(u) + 0.5) / scale - 0.5;
                    const double dist = std::abs(est.line(0) * uFull + est.line(1) * vFull + est.line(2));
                    CAPTURE(u);
                    CHECK(dist < 1.0);
                }
                CHECK(std::abs(std::hypot(est.line(0), est.line(1)) - 1.0) < 1e-9);
            }
        }
    }
}

SCENARIO("Horizon extraction reports no horizon when the frame is all ground")
{
    GIVEN("a log-odds map that is entirely negative (camera looking down)")
    {
        cv::Mat lo(120, 200, CV_32FC1, cv::Scalar(-3.0f));
        WHEN("the horizon is extracted")
        {
            const HorizonEstimate est = extractHorizon(lo, 0.25, 800);
            THEN("the estimate is invalid and reports no sky")
            {
                CHECK_FALSE(est.valid);
                CHECK(est.skyFraction == doctest::Approx(0.0));
                CHECK(est.inliers.empty());
            }
        }
    }
}

SCENARIO("Horizon extraction rejects a horizon with too little span")
{
    GIVEN("a boundary that exists only in the left 10 percent of the image")
    {
        const int w = 200;
        const int h = 120;
        cv::Mat lo(h, w, CV_32FC1, cv::Scalar(-3.0f));
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < 20; ++x)
            {
                lo.at<float>(y, x) = static_cast<float>(std::clamp(2.0 * (40.0 - y), -5.0, 5.0));
            }
        }
        WHEN("the horizon is extracted")
        {
            const HorizonEstimate est = extractHorizon(lo, 0.25, 800);
            THEN("the span test marks it invalid")
            {
                CHECK_FALSE(est.valid);
            }
        }
    }
}

SCENARIO("Sky/ground classifier and extraction reject invalid use")
{
    THEN("an untrained classifier cannot classify and empty training sets are rejected")
    {
        SkyGroundClassifier untrained;
        CHECK_THROWS_AS(untrained.logOdds(syntheticSky(20, 20, 1u)), std::logic_error);
        CHECK_THROWS_AS(SkyGroundClassifier::train({}, {syntheticGround(20, 20, 1u)}), std::invalid_argument);
    }
    THEN("malformed maps and options are rejected")
    {
        CHECK_THROWS_AS(extractHorizon(cv::Mat(), 0.25, 800), std::invalid_argument);
        CHECK_THROWS_AS(extractHorizon(cv::Mat(10, 10, CV_8UC1), 0.25, 800), std::invalid_argument);
        CHECK_THROWS_AS(extractHorizon(cv::Mat(10, 10, CV_32FC1), 0.0, 800), std::invalid_argument);
    }
}

SCENARIO("Horizon boundary is not biased by the asymmetric log-odds magnitudes at a sky/ground edge (P2 regression)")
{
    GIVEN("a log-odds map with a sharp sky/ground edge between analysis rows 49 and 50 (full-resolution row 199.5)")
    {
        // Sky log-odds are small (+1e3..1e4) and ground log-odds are ten times larger in magnitude (-1e5), as measured on the
        // synthetic and real frames. A linear 3x3 mean of the raw log-odds moved the zero crossing ~1.2 rows into the sky.
        const int cols = 160;
        const int rows = 100;
        cv::Mat lo(rows, cols, CV_32FC1);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                lo.at<float>(r, c) = (r <= 49) ? static_cast<float>(8000.0 + 100.0 * (c % 7)) : -100000.0f;
            }
        }
        WHEN("the horizon is extracted at analysis scale 0.25 for a 640 px wide image")
        {
            const HorizonEstimate est = extractHorizon(lo, 0.25, 640);
            THEN("the horizon is valid and every boundary point lies within 0.5 px of the true edge")
            {
                REQUIRE(est.valid);
                for (const cv::Point2d& p : est.inliers)
                {
                    CHECK(std::abs(p.y - 199.5) < 0.5);
                }
            }
        }
    }
}

SCENARIO("Sub-pixel edge refinement moves boundary points onto the full-resolution sky/ground edge")
{
    GIVEN("a full-resolution gray image with sky (220) above row 200 and ground (110) below it, and a classifier map whose boundary lies 4 px high")
    {
        const int W = 640;
        const int H = 400;
        cv::Mat gray(H, W, CV_8UC1);
        gray.rowRange(0, 200).setTo(220);
        gray.rowRange(200, H).setTo(110);
        // Classifier boundary at analysis row 48.5 = full row 195.5 (edge at 199.5 is inside the +-6 px search window).
        cv::Mat lo(H / 4, W / 4, CV_32FC1);
        for (int r = 0; r < lo.rows; ++r)
        {
            for (int c = 0; c < lo.cols; ++c)
            {
                lo.at<float>(r, c) = (r <= 48) ? 5000.0f : -5000.0f;
            }
        }
        WHEN("the horizon is extracted with and without the gray image")
        {
            HorizonExtractionOptions noRefine;
            noRefine.refineToEdge = false;
            const HorizonEstimate coarse = extractHorizon(lo, 0.25, W, noRefine);
            const HorizonEstimate fine = extractHorizon(lo, 0.25, W, {}, gray);
            THEN("without refinement the boundary keeps the classifier position (about 195.5 px)")
            {
                REQUIRE(coarse.valid);
                CHECK(std::abs(coarse.inliers.front().y - 195.5) < 0.6);
            }
            THEN("with refinement every inlier lies on the true edge at 199.5 px (within 0.1 px)")
            {
                REQUIRE(fine.valid);
                for (const cv::Point2d& p : fine.inliers)
                {
                    CHECK(std::abs(p.y - 199.5) < 0.1);
                }
            }
        }
    }
}

