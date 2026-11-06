#include "PointFeatures.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <opencv2/core.hpp>

namespace slam
{

int hammingDistance(const Descriptor& a, const Descriptor& b)
{
    int d = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        d += std::popcount(static_cast<unsigned int>(a[i] ^ b[i]));
    }
    return d;
}

PointFeatureExtractor::PointFeatureExtractor(int maxFeatures) : maxFeatures_(maxFeatures)
{
    if (maxFeatures_ <= 0)
    {
        throw std::invalid_argument("PointFeatureExtractor: maxFeatures must be positive");
    }
    // Defaults of cv::ORB with the Harris score (deterministic); patch and edge threshold 31 px.
    orb_ = cv::ORB::create(maxFeatures_, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 15);
}

std::vector<PointFeature> PointFeatureExtractor::extract(const cv::Mat& gray)
{
    if (gray.empty() || gray.type() != CV_8UC1)
    {
        throw std::invalid_argument("PointFeatureExtractor::extract: expected a non-empty 8-bit grey image");
    }
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    std::vector<PointFeature> out;
    out.reserve(keypoints.size());
    for (std::size_t i = 0; i < keypoints.size(); ++i)
    {
        if (descriptors.rows <= static_cast<int>(i) || descriptors.cols != 32)
        {
            break;
        }
        PointFeature f;
        f.pixel = Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y);
        f.response = keypoints[i].response;
        const std::uint8_t* row = descriptors.ptr<std::uint8_t>(static_cast<int>(i));
        std::copy_n(row, f.descriptor.size(), f.descriptor.begin());
        out.push_back(f);
    }
    // Sort by response (cv::ORB already returns retained keypoints sorted by score; keep it explicit).
    std::stable_sort(out.begin(), out.end(),
                     [](const PointFeature& a, const PointFeature& b) { return a.response > b.response; });
    return out;
}

void PointFeatureExtractor::describeAt(const cv::Mat& gray, const std::vector<Eigen::Vector2d>& pixels,
                                       std::vector<bool>& ok, std::vector<Descriptor>& descriptors)
{
    if (gray.empty() || gray.type() != CV_8UC1)
    {
        throw std::invalid_argument("PointFeatureExtractor::describeAt: expected a non-empty 8-bit grey image");
    }
    ok.assign(pixels.size(), false);
    descriptors.assign(pixels.size(), Descriptor{});
    if (pixels.empty())
    {
        return;
    }

    // Intensity-centroid orientation (radius 15 disc), as cv::ORB computes it for detected keypoints.
    constexpr int kRadius = 15;
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(pixels.size());
    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        const cv::Point c(cvRound(pixels[i].x()), cvRound(pixels[i].y()));
        double m10 = 0.0;
        double m01 = 0.0;
        if (c.x >= kRadius && c.y >= kRadius && c.x < gray.cols - kRadius && c.y < gray.rows - kRadius)
        {
            for (int dy = -kRadius; dy <= kRadius; ++dy)
            {
                const int half = static_cast<int>(std::sqrt(static_cast<double>(kRadius * kRadius - dy * dy)));
                const std::uint8_t* row = gray.ptr<std::uint8_t>(c.y + dy);
                for (int dx = -half; dx <= half; ++dx)
                {
                    const double v = static_cast<double>(row[c.x + dx]);
                    m10 += dx * v;
                    m01 += dy * v;
                }
            }
        }
        double angle = std::atan2(m01, m10) * 180.0 / std::numbers::pi;
        if (angle < 0.0)
        {
            angle += 360.0;
        }
        cv::KeyPoint kp(static_cast<float>(pixels[i].x()), static_cast<float>(pixels[i].y()), 31.0f,
                        static_cast<float>(angle), 0.0f, 0, static_cast<int>(i));
        keypoints.push_back(kp);
    }

    // compute() keeps the orientation given in each keypoint and drops points too close to the border; class_id
    // carries the index of the requested point through the filtering.
    cv::Mat desc;
    orb_->compute(gray, keypoints, desc);
    for (std::size_t j = 0; j < keypoints.size() && static_cast<int>(j) < desc.rows; ++j)
    {
        const std::size_t idx = static_cast<std::size_t>(keypoints[j].class_id);
        if (idx >= pixels.size() || desc.cols != 32)
        {
            continue;
        }
        ok[idx] = true;
        const std::uint8_t* row = desc.ptr<std::uint8_t>(static_cast<int>(j));
        std::copy_n(row, descriptors[idx].size(), descriptors[idx].begin());
    }
}

} // namespace slam
