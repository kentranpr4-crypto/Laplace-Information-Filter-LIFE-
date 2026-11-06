#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../../src/ImageFlow.h"

namespace
{

/// Smooth random texture with fine detail (good corners for Shi-Tomasi and Lucas-Kanade).
cv::Mat makeTexture(int w, int h, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    cv::Mat noise(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            noise.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>(dist(rng));
        }
    }
    cv::Mat blurred;
    cv::GaussianBlur(noise, blurred, cv::Size(0, 0), 1.2);
    cv::Mat out;
    cv::normalize(blurred, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return out;
}

/// Translates an image by (dx, dy) with bilinear sub-pixel interpolation (content moves by +dx, +dy).
cv::Mat shifted(const cv::Mat& img, double dx, double dy)
{
    const cv::Matx23d m(1.0, 0.0, dx, 0.0, 1.0, dy);
    cv::Mat out;
    cv::warpAffine(img, out, m, img.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

} // namespace

SCENARIO("ImageFlow: tracks a known sub-pixel translation")
{
    GIVEN("a textured frame and the same frame translated by (3.4, -2.2) px")
    {
        const cv::Mat f0 = makeTexture(320, 240, 42u);
        const double dx = 3.4;
        const double dy = -2.2;
        const cv::Mat f1 = shifted(f0, dx, dy);
        ImageFlow flow;
        flow.update(f0);
        WHEN("the second frame is tracked")
        {
            const auto matches = flow.update(f1);
            THEN("many correspondences survive the forward-backward check")
            {
                CHECK(matches.size() > 50);
                CHECK(flow.stats().active >= static_cast<int>(matches.size()));
            }
            THEN("every surviving correspondence matches the true translation within a fraction of a pixel")
            {
                std::vector<double> err;
                for (const auto& m : matches)
                {
                    const cv::Point2d d = m.curr - m.prev;
                    err.push_back(std::hypot(d.x - dx, d.y - dy));
                }
                std::sort(err.begin(), err.end());
                REQUIRE(!err.empty());
                CHECK(err[err.size() / 2] < 0.1);
                CHECK(err.back() < 0.5);
            }
        }
    }
}

SCENARIO("ImageFlow: forward-backward check removes features behind an occluder")
{
    GIVEN("frame two with a black rectangle covering part of the scene")
    {
        const cv::Mat f0 = makeTexture(320, 240, 7u);
        cv::Mat f1 = shifted(f0, 2.0, 1.0);
        cv::rectangle(f1, cv::Rect(160, 60, 120, 120), cv::Scalar(0), cv::FILLED);
        ImageFlow flow;
        flow.update(f0);
        const auto matches = flow.update(f1);
        WHEN("the surviving matches are compared with the true translation (2, 1) px")
        {
            int gross = 0;
            for (const auto& m : matches)
            {
                const cv::Point2d d = m.curr - m.prev;
                if (std::hypot(d.x - 2.0, d.y - 1.0) > 1.0)
                {
                    ++gross;
                }
            }
            THEN("no surviving match carries a gross error (the forward-backward check rejects them)")
            {
                CHECK(gross == 0);
            }
            THEN("the forward-backward check actually removed some tracks")
            {
                CHECK(flow.stats().lostForwardBackward + flow.stats().lostLucasKanade > 0);
                CHECK(flow.stats().tracked > static_cast<int>(matches.size()) - 1);
            }
        }
    }
}

SCENARIO("ImageFlow: detection honours the mask and track identities persist")
{
    GIVEN("a textured frame and a detection mask that allows only the left half")
    {
        const cv::Mat f0 = makeTexture(320, 240, 11u);
        cv::Mat mask = cv::Mat::zeros(f0.size(), CV_8UC1);
        mask(cv::Rect(0, 0, 160, 240)).setTo(255);
        ImageFlow flow;
        flow.update(f0, mask);
        THEN("all newly detected features lie in the allowed region")
        {
            for (const auto& t : flow.activeTracks())
            {
                CHECK(t.curr.x < 161.0);
            }
            CHECK(flow.stats().detected > 0);
        }
        WHEN("the sequence advances by two small translations")
        {
            const cv::Mat f1 = shifted(f0, 1.0, 0.0);
            const cv::Mat f2 = shifted(f0, 2.0, 0.0);
            const auto m1 = flow.update(f1, mask);
            std::map<int, int> ageAfterFirst;
            for (const auto& m : m1)
            {
                ageAfterFirst[m.id] = m.age;
            }
            const auto m2 = flow.update(f2, mask);
            THEN("ids of surviving tracks are preserved and their age increases by one")
            {
                int common = 0;
                for (const auto& m : m2)
                {
                    const auto it = ageAfterFirst.find(m.id);
                    if (it != ageAfterFirst.end())
                    {
                        ++common;
                        CHECK(m.age == it->second + 1);
                    }
                }
                CHECK(common > 20);
            }
        }
    }
}

SCENARIO("ImageFlow: robust plane (homography) fit rejects outliers")
{
    GIVEN("correspondences from a planar motion with 30 percent random outliers")
    {
        std::mt19937 rng(99u);
        std::uniform_real_distribution<double> pos(0.0, 300.0);
        std::uniform_real_distribution<double> off(-80.0, 80.0);
        std::vector<ImageFlowMatch> matches;
        std::vector<bool> truthInlier;
        for (int i = 0; i < 200; ++i)
        {
            ImageFlowMatch m;
            m.id = i;
            m.age = 1;
            m.prev = cv::Point2d(pos(rng), pos(rng) * 0.7);
            const bool outlier = i % 10 < 3;
            // planar motion: mild perspective, small translation
            const double w = 1.0 + 1e-4 * m.prev.x;
            m.curr = cv::Point2d(m.prev.x * 1.01 / w + 3.0, m.prev.y * 0.995 / w - 1.5);
            if (outlier)
            {
                m.curr += cv::Point2d(off(rng), off(rng));
            }
            matches.push_back(m);
            truthInlier.push_back(!outlier);
        }
        WHEN("the RANSAC plane fit is computed with a 2 px threshold")
        {
            std::vector<std::uint8_t> mask;
            cv::Mat H;
            const int count = ImageFlow::robustHomographyInliers(matches, 2.0, mask, H);
            THEN("the inlier set agrees with the ground-truth inliers for almost every match")
            {
                CHECK(!H.empty());
                int agree = 0;
                for (std::size_t i = 0; i < matches.size(); ++i)
                {
                    agree += ((mask[i] != 0) == truthInlier[i]) ? 1 : 0;
                }
                CHECK(agree >= static_cast<int>(0.95 * static_cast<double>(matches.size())));
                CHECK(count >= 130);
            }
        }
    }
}

SCENARIO("ImageFlow: invalid inputs are rejected")
{
    ImageFlow flow;
    THEN("empty, colour or mismatched images throw")
    {
        CHECK_THROWS_AS(flow.update(cv::Mat()), std::invalid_argument);
        CHECK_THROWS_AS(flow.update(cv::Mat(10, 10, CV_8UC3)), std::invalid_argument);
        flow.update(cv::Mat::zeros(64, 64, CV_8UC1));
        CHECK_THROWS_AS(flow.update(cv::Mat::zeros(32, 32, CV_8UC1)), std::invalid_argument);
    }
    THEN("invalid options are rejected at construction")
    {
        ImageFlowOptions bad;
        bad.scale = 0.0;
        CHECK_THROWS_AS(ImageFlow{bad}, std::invalid_argument);
        ImageFlowOptions even;
        even.lkWindow = 20;
        CHECK_THROWS_AS(ImageFlow{even}, std::invalid_argument);
    }
    THEN("a homography fit with fewer than four matches reports failure")
    {
        std::vector<std::uint8_t> mask;
        cv::Mat H;
        CHECK(ImageFlow::robustHomographyInliers({}, 2.0, mask, H) == 0);
        CHECK(H.empty());
    }
}
