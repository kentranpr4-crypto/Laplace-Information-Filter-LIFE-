// ORB point features: Hamming distance, extraction contract, determinism and adverse inputs.
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "../../src/PointFeatures.h"

using slam::Descriptor;
using slam::PointFeature;
using slam::PointFeatureExtractor;

namespace
{

/// Deterministic textured grey image (smoothed random noise plus rectangles) of the given size.
cv::Mat texturedImage(int w, int h, unsigned seed)
{
    cv::RNG rng(seed);
    cv::Mat img(h, w, CV_8UC1);
    rng.fill(img, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(img, img, cv::Size(5, 5), 1.2);
    for (int k = 0; k < 40; ++k)
    {
        const int x = static_cast<int>(rng.uniform(0, w - 40));
        const int y = static_cast<int>(rng.uniform(0, h - 40));
        cv::rectangle(img, cv::Rect(x, y, 30, 25), cv::Scalar(static_cast<double>(rng.uniform(0, 255))), -1);
    }
    return img;
}

} // namespace

SCENARIO("hammingDistance counts differing bits")
{
    GIVEN("descriptors with known bit patterns")
    {
        Descriptor zero{};
        Descriptor ones{};
        ones.fill(0xFF);
        Descriptor one{};
        one[7] = 0x80;
        THEN("identical descriptors have distance 0, complementary ones 256, and a single bit 1")
        {
            CHECK(slam::hammingDistance(zero, zero) == 0);
            CHECK(slam::hammingDistance(zero, ones) == 256);
            CHECK(slam::hammingDistance(zero, one) == 1);
            CHECK(slam::hammingDistance(one, ones) == 255);
            CHECK(slam::hammingDistance(one, zero) == slam::hammingDistance(zero, one));
        }
    }
}

SCENARIO("PointFeatureExtractor: textured images give sorted features with 32-byte descriptors")
{
    GIVEN("a textured 640 x 480 grey image and an extractor with 300 features")
    {
        PointFeatureExtractor ex(300);
        const cv::Mat img = texturedImage(640, 480, 4242);
        WHEN("the features are extracted")
        {
            const std::vector<PointFeature> feats = ex.extract(img);
            THEN("there are features, at most the requested number, inside the image and sorted by response")
            {
                CHECK(feats.size() > 50);
                CHECK(feats.size() <= 300);
                for (std::size_t i = 0; i < feats.size(); ++i)
                {
                    CHECK(feats[i].pixel.x() >= 0.0);
                    CHECK(feats[i].pixel.x() < 640.0);
                    CHECK(feats[i].pixel.y() >= 0.0);
                    CHECK(feats[i].pixel.y() < 480.0);
                    if (i > 0)
                    {
                        CHECK(feats[i - 1].response >= feats[i].response);
                    }
                }
            }
            THEN("the descriptors are not degenerate (not all zero)")
            {
                int nonZero = 0;
                for (const PointFeature& f : feats)
                {
                    for (std::uint8_t b : f.descriptor)
                    {
                        if (b != 0)
                        {
                            ++nonZero;
                            break;
                        }
                    }
                }
                CHECK(nonZero > static_cast<int>(feats.size()) / 2);
            }
            THEN("a second extraction of the same image is identical (deterministic detector)")
            {
                const std::vector<PointFeature> again = ex.extract(img);
                REQUIRE(again.size() == feats.size());
                for (std::size_t i = 0; i < feats.size(); ++i)
                {
                    CHECK(again[i].pixel == feats[i].pixel);
                    CHECK(again[i].descriptor == feats[i].descriptor);
                }
            }
        }
    }
}

SCENARIO("PointFeatureExtractor: adverse inputs")
{
    GIVEN("an extractor")
    {
        PointFeatureExtractor ex(100);
        WHEN("a blank image is processed")
        {
            const cv::Mat blank = cv::Mat::zeros(240, 320, CV_8UC1);
            THEN("no features are returned and nothing fails")
            {
                CHECK(ex.extract(blank).empty());
            }
        }
        WHEN("an empty image is processed")
        {
            THEN("the call is rejected")
            {
                CHECK_THROWS_AS(ex.extract(cv::Mat()), std::invalid_argument);
            }
        }
        WHEN("a colour image is processed")
        {
            const cv::Mat colour(240, 320, CV_8UC3, cv::Scalar(10, 20, 30));
            THEN("the call is rejected (grey 8-bit input is required)")
            {
                CHECK_THROWS_AS(ex.extract(colour), std::invalid_argument);
            }
        }
    }
    GIVEN("a non-positive feature budget")
    {
        THEN("construction is rejected")
        {
            CHECK_THROWS_AS(PointFeatureExtractor(0), std::invalid_argument);
            CHECK_THROWS_AS(PointFeatureExtractor(-5), std::invalid_argument);
        }
    }
}
