#ifndef VISUALNAVIGATIONSUPPORT_H
#define VISUALNAVIGATIONSUPPORT_H

/**
 * @file visualNavigationSupport.h
 * @brief Test-only helpers shared by the BufferedVideo, Plot and visual-navigation tests.
 *
 * Everything that touches the file system writes into a fresh directory under
 * std::filesystem::temp_directory_path(), removed again by TempDir's destructor. No fixture under data/ is
 * modified. Videos are generated at test time with cv::VideoWriter.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "../../src/Camera.h"

namespace visTest
{

/// Fresh directory under the system temp directory, removed on destruction.
class TempDir
{
public:
    TempDir()
    {
        static std::atomic<unsigned> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
              / std::format("mcha4400_vis_{}_{}", static_cast<unsigned long long>(stamp), counter++);
        std::filesystem::create_directories(path_);
    }
    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir &) = delete;
    TempDir & operator=(const TempDir &) = delete;

    [[nodiscard]] const std::filesystem::path & path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

/// Pinhole camera without distortion, focal length 0.9 W, principal point at the image centre.
inline Camera makeCamera(int width, int height)
{
    Eigen::Matrix3d K;
    K << 0.9 * width, 0.0, width / 2.0,
         0.0, 0.9 * width, height / 2.0,
         0.0, 0.0, 1.0;
    return Camera(width, height, K, {});
}

/// Per-frame code colour of the synthetic videos (uniform frame, distinct for neighbouring indices).
inline cv::Vec3b frameCode(int index)
{
    return cv::Vec3b(static_cast<uchar>((index * 37) % 256), static_cast<uchar>((index * 91 + 40) % 256),
                     static_cast<uchar>((index * 17 + 120) % 256));
}

/**
 * @brief Writes a short synthetic video whose frame i is a uniform colour frameCode(i).
 * @param codec fourcc string; MJPG in .avi is used by default because it is decoded reliably by OpenCV here.
 */
inline void writeCodedVideo(const std::filesystem::path & path, cv::Size size, int nFrames, double fps,
                            const std::string & codec = "MJPG")
{
    cv::VideoWriter writer(path.string(), cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]), fps, size);
    if (!writer.isOpened())
        throw std::runtime_error(std::format("test: cannot open synthetic video writer for {}", path.string()));
    cv::Mat frame(size, CV_8UC3);
    for (int i = 0; i < nFrames; ++i)
    {
        frame.setTo(cv::Scalar(frameCode(i)[0], frameCode(i)[1], frameCode(i)[2]));
        // A small moving marker keeps the frames non-trivial for codecs and for the render pipeline.
        cv::rectangle(frame, cv::Rect(i % std::max(1, size.width - 8), 0, 8, std::min(8, size.height)),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        writer.write(frame);
    }
    writer.release();
}

/// Mean BGR colour of a frame.
inline cv::Scalar meanColour(const cv::Mat & frame)
{
    return cv::mean(frame);
}

/// Asymmetric BGR test pattern (no symmetry under flips): gradients in x (blue) and y (green), a red block in
/// the top-left corner and a white bar along the bottom-right.
inline cv::Mat makeAsymmetricOverlay(cv::Size size)
{
    cv::Mat img(size, CV_8UC3, cv::Scalar(30, 60, 90));
    for (int y = 0; y < size.height; ++y)
    {
        for (int x = 0; x < size.width; ++x)
        {
            cv::Vec3b & p = img.at<cv::Vec3b>(y, x);
            p[0] = static_cast<uchar>(255.0 * x / std::max(1, size.width - 1));
            p[1] = static_cast<uchar>(255.0 * y / std::max(1, size.height - 1));
        }
    }
    cv::rectangle(img, cv::Rect(0, 0, size.width / 4, size.height / 5), cv::Scalar(0, 0, 255), cv::FILLED);
    cv::rectangle(img, cv::Rect(size.width / 2, size.height - size.height / 10, size.width / 2, size.height / 10),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    return img;
}

/**
 * @brief Runs fn on a helper thread and reports whether it finished within the limit.
 * On timeout the helper thread is detached (it cannot be joined safely) and false is returned, so that the
 * test fails instead of hanging the whole test binary. Exceptions thrown by fn are rethrown.
 */
template<typename Fn>
bool finishesWithin(Fn && fn, std::chrono::milliseconds limit)
{
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    std::thread worker([fn = std::forward<Fn>(fn), promise]() mutable
    {
        try
        {
            fn();
            promise->set_value();
        }
        catch (...)
        {
            promise->set_exception(std::current_exception());
        }
    });
    if (future.wait_for(limit) == std::future_status::ready)
    {
        worker.join();
        future.get();   // rethrows an exception from fn
        return true;
    }
    worker.detach();
    return false;
}

} // namespace visTest

#endif
