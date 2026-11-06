#ifndef POINTFEATURES_H
#define POINTFEATURES_H

#include <array>
#include <cstdint>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/features.hpp>

namespace slam
{

/// 256-bit binary descriptor (ORB).
using Descriptor = std::array<std::uint8_t, 32>;

/**
 * @brief One detected image point with its binary descriptor.
 * pixel uses the OpenCV convention (centre of the top-left pixel at (0,0)).
 */
struct PointFeature
{
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    float response = 0.0f;   ///< detector score (Harris score of ORB); larger is stronger
    Descriptor descriptor{};
};

/** @brief Hamming distance (number of differing bits) between two 256-bit descriptors. */
int hammingDistance(const Descriptor& a, const Descriptor& b);

/**
 * @brief ORB detection and description with OpenCV (cv::ORB from <opencv2/features.hpp>).
 *
 * Output is sorted by decreasing response. The detector is deterministic for a given image.
 */
class PointFeatureExtractor
{
public:
    /**
     * @param maxFeatures  upper bound on the number of keypoints per frame
     * @throws std::invalid_argument if maxFeatures <= 0
     */
    explicit PointFeatureExtractor(int maxFeatures = 900);

    /**
     * @param gray  8-bit single-channel image
     * @return features sorted by decreasing response (possibly empty for a blank image)
     * @throws std::invalid_argument for an empty or non-8-bit image
     */
    std::vector<PointFeature> extract(const cv::Mat& gray);

    /** @brief Maximum number of features requested per frame. */
    int maxFeatures() const { return maxFeatures_; }

    /**
     * @brief ORB descriptors at given pixel positions (not necessarily detected keypoints).
     *
     * The orientation of each point is the intensity-centroid angle of its 31 x 31 patch, as in ORB detection, so the
     * descriptor is comparable with the descriptors of detected features at the same corner. Points closer than the
     * ORB border (31 px) to the image edge get ok = false and no descriptor.
     * @param gray       8-bit single-channel image
     * @param pixels     positions (OpenCV pixel convention)
     * @param ok         output, one flag per point
     * @param descriptors output, one descriptor per point (default-valued where ok is false)
     * @throws std::invalid_argument for an empty or non-8-bit image
     */
    void describeAt(const cv::Mat& gray, const std::vector<Eigen::Vector2d>& pixels, std::vector<bool>& ok,
                    std::vector<Descriptor>& descriptors);

private:
    int maxFeatures_;
    cv::Ptr<cv::ORB> orb_;
};

} // namespace slam

#endif
