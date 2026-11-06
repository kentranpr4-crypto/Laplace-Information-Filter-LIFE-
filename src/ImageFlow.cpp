#include "ImageFlow.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features.hpp>
#include <opencv2/video.hpp>
#include <opencv2/geometry.hpp>

namespace
{

/// Working-resolution pixel -> full-resolution pixel.
cv::Point2d toFull(const cv::Point2f& p, double scale)
{
    return cv::Point2d((p.x + 0.5) / scale - 0.5, (p.y + 0.5) / scale - 0.5);
}

} // namespace

ImageFlow::ImageFlow() : ImageFlow(ImageFlowOptions{}) {}

ImageFlow::ImageFlow(ImageFlowOptions options) : options_(options)
{
    if (!(options_.scale > 0.0 && options_.scale <= 1.0))
    {
        throw std::invalid_argument("ImageFlow: scale must be in (0, 1]");
    }
    if (options_.maxFeatures <= 0 || options_.lkWindow < 3 || options_.lkWindow % 2 == 0 || options_.lkLevels < 0 ||
        options_.lkMaxIterations <= 0 || !(options_.lkEpsilon > 0.0) || !(options_.forwardBackwardPx > 0.0) ||
        !(options_.minDistanceWorkPx > 0.0) || options_.blockSize < 3)
    {
        throw std::invalid_argument("ImageFlow: invalid options");
    }
}

void ImageFlow::reset()
{
    prevWork_.release();
    active_.clear();
    activeWorkPts_.clear();
    hasPrevious_ = false;
    stats_ = ImageFlowStats{};
}

std::vector<ImageFlowMatch> ImageFlow::update(const cv::Mat& gray, const cv::Mat& detectMask)
{
    if (gray.empty() || gray.type() != CV_8UC1)
    {
        throw std::invalid_argument("ImageFlow::update: gray must be a non-empty CV_8UC1 image");
    }
    if (!detectMask.empty() && (detectMask.size() != gray.size() || detectMask.type() != CV_8UC1))
    {
        throw std::invalid_argument("ImageFlow::update: detection mask must be CV_8UC1 with the image size");
    }
    if (hasPrevious_ && gray.size() != fullSize_)
    {
        throw std::invalid_argument("ImageFlow::update: image size changed between calls");
    }
    fullSize_ = gray.size();
    stats_ = ImageFlowStats{};

    cv::Mat work;
    if (options_.scale < 1.0)
    {
        cv::resize(gray, work, cv::Size(), options_.scale, options_.scale, cv::INTER_AREA);
    }
    else
    {
        work = gray.clone();
    }

    std::vector<ImageFlowMatch> matches;
    std::vector<ImageFlowMatch> nextActive;
    std::vector<cv::Point2f> nextWorkPts;

    const cv::Size winSize(options_.lkWindow, options_.lkWindow);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, options_.lkMaxIterations,
                                    options_.lkEpsilon);

    // 1. Track the active features from the previous frame: forward Lucas-Kanade, then backward check.
    if (hasPrevious_ && !activeWorkPts_.empty())
    {
        std::vector<cv::Point2f> currPts;
        std::vector<cv::Point2f> backPts;
        std::vector<unsigned char> statusF;
        std::vector<unsigned char> statusB;
        std::vector<float> errF;
        std::vector<float> errB;
        stats_.tracked = static_cast<int>(activeWorkPts_.size());
        cv::calcOpticalFlowPyrLK(prevWork_, work, activeWorkPts_, currPts, statusF, errF, winSize, options_.lkLevels,
                                 criteria);
        cv::calcOpticalFlowPyrLK(work, prevWork_, currPts, backPts, statusB, errB, winSize, options_.lkLevels,
                                 criteria);

        const double fbTol = options_.forwardBackwardPx;
        for (std::size_t i = 0; i < activeWorkPts_.size(); ++i)
        {
            if (statusF[i] == 0 || statusB[i] == 0 || !std::isfinite(currPts[i].x) || !std::isfinite(currPts[i].y))
            {
                ++stats_.lostLucasKanade;
                continue;
            }
            const cv::Point2d curr = toFull(currPts[i], options_.scale);
            if (curr.x < -0.5 || curr.y < -0.5 || curr.x > fullSize_.width - 0.5 || curr.y > fullSize_.height - 0.5)
            {
                ++stats_.lostLucasKanade;
                continue;
            }
            const cv::Point2d back = toFull(backPts[i], options_.scale);
            const cv::Point2d prevFull = toFull(activeWorkPts_[i], options_.scale);
            if (cv::norm(back - prevFull) > fbTol)
            {
                ++stats_.lostForwardBackward;
                continue;
            }
            ImageFlowMatch m;
            m.id = active_[i].id;
            m.age = active_[i].age + 1;
            m.prev = active_[i].curr;
            m.curr = curr;
            matches.push_back(m);
            nextActive.push_back(m);
            nextWorkPts.push_back(currPts[i]);
        }
    }

    // 2. Detect new features in the mask, away from the surviving tracks, until maxFeatures is reached.
    const int room = options_.maxFeatures - static_cast<int>(nextActive.size());
    if (room > 0)
    {
        cv::Mat detectWork;
        if (detectMask.empty())
        {
            detectWork = cv::Mat(work.size(), CV_8UC1, cv::Scalar(255));
        }
        else
        {
            cv::resize(detectMask, detectWork, work.size(), 0.0, 0.0, cv::INTER_NEAREST);
        }
        for (const cv::Point2f& p : nextWorkPts)
        {
            cv::circle(detectWork, cv::Point(cvRound(p.x), cvRound(p.y)), static_cast<int>(std::ceil(options_.minDistanceWorkPx)),
                       cv::Scalar(0), cv::FILLED);
        }
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(work, corners, room, options_.qualityLevel, options_.minDistanceWorkPx, detectWork,
                                options_.blockSize);
        for (const cv::Point2f& c : corners)
        {
            ImageFlowMatch m;
            m.id = nextId_++;
            m.age = 0;
            m.curr = toFull(c, options_.scale);
            nextActive.push_back(m);
            nextWorkPts.push_back(c);
        }
        stats_.detected = static_cast<int>(corners.size());
    }

    active_ = std::move(nextActive);
    activeWorkPts_ = std::move(nextWorkPts);
    prevWork_ = std::move(work);
    hasPrevious_ = true;
    stats_.active = static_cast<int>(active_.size());
    return matches;
}

int ImageFlow::robustHomographyInliers(const std::vector<ImageFlowMatch>& matches, double thresholdPx,
                                       std::vector<std::uint8_t>& inlierMask, cv::Mat& homography)
{
    inlierMask.assign(matches.size(), 0);
    homography = cv::Mat();
    if (matches.size() < 4)
    {
        return 0;
    }
    std::vector<cv::Point2d> src;
    std::vector<cv::Point2d> dst;
    src.reserve(matches.size());
    dst.reserve(matches.size());
    for (const ImageFlowMatch& m : matches)
    {
        src.push_back(m.prev);
        dst.push_back(m.curr);
    }
    cv::Mat mask;
    homography = cv::findHomography(src, dst, cv::RANSAC, thresholdPx, mask, 2000, 0.995);
    if (homography.empty() || mask.empty())
    {
        homography = cv::Mat();
        return 0;
    }
    int count = 0;
    for (std::size_t i = 0; i < matches.size(); ++i)
    {
        const bool in = mask.at<std::uint8_t>(static_cast<int>(i)) != 0;
        inlierMask[i] = in ? 1 : 0;
        count += in ? 1 : 0;
    }
    return count;
}
