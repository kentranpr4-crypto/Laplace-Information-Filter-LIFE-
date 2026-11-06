#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../../src/Camera.h"
#include "../../src/Plot.h"
#include "../../src/ScenarioRunner.h"
#include "visualNavigationSupport.h"

using visTest::makeAsymmetricOverlay;
using visTest::makeCamera;

namespace
{

/// Sets (or unsets, for a null value) an environment variable for the lifetime of the guard; restores it afterwards.
class ScopedEnvVar
{
public:
    ScopedEnvVar(const char * name, const char * value)
        : name_(name)
    {
        if (const char * old = std::getenv(name))
        {
            hadOld_ = true;
            old_ = old;
        }
        if (value)
            setenv(name_.c_str(), value, 1);
        else
            unsetenv(name_.c_str());
    }
    ~ScopedEnvVar()
    {
        if (hadOld_)
            setenv(name_.c_str(), old_.c_str(), 1);
        else
            unsetenv(name_.c_str());
    }
    ScopedEnvVar(const ScopedEnvVar &) = delete;
    ScopedEnvVar & operator=(const ScopedEnvVar &) = delete;

private:
    std::string name_;
    std::string old_;
    bool hadOld_ = false;
};

/// Scene with a trajectory, validation track, camera uncertainty, point and pose landmarks, both visible and
/// dimmed. Everything is inside a few metres so that the overview camera frames it.
PlotFrameData makeScene(const cv::Mat & overlay, bool withValidation = true)
{
    PlotFrameData d;
    d.overlay = overlay;
    d.rCNn = Eigen::Vector3d(0.6, 0.1, -0.4);
    d.Rnc = Eigen::Matrix3d::Identity();
    d.positionSqrtCov = Eigen::Matrix3d::Identity() * 0.05;
    for (int i = 0; i < 12; ++i)
        d.trajectory.emplace_back(0.05 * i, 0.02 * i * i / 3.0, -0.04 * i);
    d.trajectory.push_back(d.rCNn);
    if (withValidation)
    {
        for (int i = 0; i < 12; ++i)
            d.validationTrack.emplace_back(0.05 * i + 0.1, 0.02 * i, -0.04 * i - 0.05);
    }

    PlotFrameData::Landmark a;
    a.id = 1;
    a.rLNn = Eigen::Vector3d(1.5, 0.2, -0.6);
    a.positionSqrtCov = Eigen::Vector3d(0.10, 0.05, 0.02).asDiagonal();
    a.colourRGB = Eigen::Vector3d(1.0, 0.2, 0.2);
    a.visible = true;
    d.landmarks.push_back(a);

    PlotFrameData::Landmark b = a;
    b.id = 2;
    b.rLNn = Eigen::Vector3d(-0.8, 1.1, -0.2);
    b.colourRGB = Eigen::Vector3d(0.2, 1.0, 0.2);
    b.visible = false;   // dimmed
    d.landmarks.push_back(b);

    PlotFrameData::Landmark tag;
    tag.id = 10;
    tag.isPose = true;
    tag.rLNn = Eigen::Vector3d(0.3, -1.0, -0.5);
    const double th = 0.4;
    tag.Rnl << std::cos(th), -std::sin(th), 0.0,
               std::sin(th), std::cos(th), 0.0,
               0.0, 0.0, 1.0;
    tag.size = 0.3;
    tag.positionSqrtCov = Eigen::Vector3d(0.04, 0.03, 0.01).asDiagonal();
    tag.colourRGB = Eigen::Vector3d(0.3, 0.5, 1.0);
    tag.visible = true;
    d.landmarks.push_back(tag);

    d.statusLines = {"frame 12", "t = 0.40 s", "landmarks: 3"};
    return d;
}

/// Mean absolute difference over all channels of two equally sized BGR8 images.
double meanAbsDiff(const cv::Mat & a, const cv::Mat & b)
{
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::mean(diff)[0] / 3.0 + cv::mean(diff)[1] / 3.0 + cv::mean(diff)[2] / 3.0;
}

/// Fraction of pixels that differ from the scene background (BGR of RGB (0.10, 0.11, 0.14)).
double nonBackgroundFraction(const cv::Mat & bgr)
{
    const cv::Scalar background(36, 28, 26);   // 0.14, 0.11, 0.10 * 255 in BGR
    long count = 0;
    for (int y = 0; y < bgr.rows; ++y)
    {
        const cv::Vec3b * row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x)
        {
            const cv::Vec3b & p = row[x];
            if (std::abs(p[0] - background[0]) > 6 || std::abs(p[1] - background[1]) > 6 || std::abs(p[2] - background[2]) > 6)
                ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(bgr.rows * bgr.cols);
}

} // namespace

SCENARIO("Plot produces a 2W x H BGR frame whose left half is the scenario overlay")
{
    const int W = 320;
    const int H = 180;
    const cv::Size size(W, H);
    const cv::Mat overlay = makeAsymmetricOverlay(size);

    GIVEN("a camera calibrated for 320 x 180 and an asymmetric overlay")
    {
        const Camera camera = makeCamera(W, H);

        WHEN("the scene is updated, rendered offscreen and captured")
        {
            Plot plot(camera, size, Plot::WindowMode::Offscreen);
            plot.update(makeScene(overlay));
            plot.render();
            cv::Mat frame;
            plot.capture(frame);

            THEN("the frame is exactly 2W x H BGR8 and matches the reported frame size")
            {
                CHECK(frame.cols == 2 * W);
                CHECK(frame.rows == H);
                CHECK(frame.type() == CV_8UC3);
                CHECK(plot.frameSize() == cv::Size(2 * W, H));
                CHECK(plot.inputSize() == size);
                CHECK(frame.isContinuous());
            }

            THEN("the left half equals the overlay (no scaling, no vertical flip, correct channel order)")
            {
                const cv::Mat left = frame(cv::Rect(0, 0, W, H));
                const double diffNormal = meanAbsDiff(left, overlay);
                cv::Mat flipped;
                cv::flip(overlay, flipped, 0);
                const double diffFlipped = meanAbsDiff(left, flipped);
                INFO("mean abs diff to overlay = " << diffNormal << ", to vertically flipped overlay = " << diffFlipped);
                CHECK(diffNormal < 2.0);
                CHECK(diffFlipped > 20.0);
            }

            THEN("the red block of the overlay is top-left in the left pane")
            {
                const cv::Mat left = frame(cv::Rect(0, 0, W, H));
                // centroid of strongly red pixels (BGR: low B, low G, high R)
                double sx = 0.0;
                double sy = 0.0;
                double n = 0.0;
                for (int y = 0; y < H; ++y)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        const cv::Vec3b & p = left.at<cv::Vec3b>(y, x);
                        if (p[2] > 200 && p[1] < 60 && p[0] < 60)
                        {
                            sx += x;
                            sy += y;
                            n += 1.0;
                        }
                    }
                }
                REQUIRE(n > 100.0);
                CHECK(sx / n < W / 4.0);
                CHECK(sy / n < H / 5.0);
            }

            THEN("the white bar at the bottom-right of the overlay stays bottom-right in the left pane")
            {
                const cv::Mat left = frame(cv::Rect(0, 0, W, H));
                const cv::Vec3b p = left.at<cv::Vec3b>(H - 3, 3 * W / 4);
                CHECK(p[0] > 240);
                CHECK(p[1] > 240);
                CHECK(p[2] > 240);
            }

            THEN("the right half is not blank: the 3-D scene is drawn")
            {
                const cv::Mat right = frame(cv::Rect(W, 0, W, H));
                CHECK(nonBackgroundFraction(right) > 0.005);
            }

            THEN("the render timings are recorded and finite")
            {
                const Plot::Timings & t = plot.lastTimings();
                CHECK(std::isfinite(t.updateMs));
                CHECK(std::isfinite(t.renderMs));
                CHECK(std::isfinite(t.captureMs));
                CHECK(t.captureMs >= 0.0);
            }
        }

        WHEN("the scene is rendered twice with the trajectory growing in between")
        {
            Plot plot(camera, size, Plot::WindowMode::Offscreen);
            PlotFrameData d = makeScene(overlay, false);
            plot.update(d);
            plot.render();
            cv::Mat first;
            plot.capture(first);
            d.trajectory.emplace_back(1.0, 0.5, -0.3);
            d.trajectory.emplace_back(1.2, 0.6, -0.2);
            plot.update(d);
            cv::Mat second;
            plot.capture(second);

            THEN("both frames have the same size and the second one differs from the first in the right pane")
            {
                CHECK(second.size() == first.size());
                const cv::Mat r1 = first(cv::Rect(W, 0, W, H));
                const cv::Mat r2 = second(cv::Rect(W, 0, W, H));
                CHECK(meanAbsDiff(r1, r2) > 0.0);
            }
        }
    }
}

SCENARIO("Plot renders odd and non-16:9 frame sizes with the same layout")
{
    GIVEN("an odd-sized 121 x 75 input frame")
    {
        const cv::Size size(121, 75);
        const Camera camera = makeCamera(size.width, size.height);
        const cv::Mat overlay = makeAsymmetricOverlay(size);

        WHEN("the scene is rendered and captured")
        {
            Plot plot(camera, size, Plot::WindowMode::Offscreen);
            plot.update(makeScene(overlay));
            plot.render();
            cv::Mat frame;
            plot.capture(frame);

            THEN("the frame is 242 x 75 and its left half still matches the overlay")
            {
                CHECK(frame.cols == 242);
                CHECK(frame.rows == 75);
                const double d = meanAbsDiff(frame(cv::Rect(0, 0, size.width, size.height)), overlay);
                INFO("mean abs diff = " << d);
                CHECK(d < 2.0);
            }
        }
    }
}

SCENARIO("Plot accepts an empty overlay and grey overlays, and rejects wrong overlay sizes")
{
    const cv::Size size(160, 90);
    const Camera camera = makeCamera(size.width, size.height);

    GIVEN("a Plot for 160 x 90 input frames")
    {
        Plot plot(camera, size, Plot::WindowMode::Offscreen);

        WHEN("the overlay is empty")
        {
            PlotFrameData d = makeScene(cv::Mat());
            plot.update(d);
            plot.render();
            cv::Mat frame;
            plot.capture(frame);
            THEN("the left pane is black and the frame still has the full size")
            {
                CHECK(frame.size() == cv::Size(2 * size.width, size.height));
                CHECK(cv::countNonZero(frame(cv::Rect(0, 0, size.width, size.height)).reshape(1)) == 0);
            }
        }

        WHEN("the overlay is a single-channel image of the input size")
        {
            cv::Mat gray(size, CV_8UC1, cv::Scalar(128));
            CHECK_NOTHROW(plot.update(makeScene(gray)));
        }

        WHEN("the overlay has the wrong size")
        {
            cv::Mat wrong(size.height + 2, size.width, CV_8UC3, cv::Scalar(0, 0, 0));
            THEN("update() throws std::invalid_argument")
            {
                CHECK_THROWS_AS(plot.update(makeScene(wrong)), std::invalid_argument);
            }
        }

        WHEN("the overlay is not 8-bit BGR or grey")
        {
            cv::Mat floatImage(size, CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
            THEN("update() throws std::invalid_argument")
            {
                CHECK_THROWS_AS(plot.update(makeScene(floatImage)), std::invalid_argument);
            }
        }
    }
}

SCENARIO("Plot rejects inconsistent construction arguments")
{
    GIVEN("a camera calibrated for 160 x 90")
    {
        const Camera camera = makeCamera(160, 90);

        WHEN("the input size differs from the camera image size")
        {
            THEN("the constructor throws std::invalid_argument")
            {
                CHECK_THROWS_AS(Plot(camera, cv::Size(320, 180), Plot::WindowMode::Offscreen), std::invalid_argument);
            }
        }

        WHEN("the input size is degenerate")
        {
            THEN("the constructor throws std::invalid_argument")
            {
                CHECK_THROWS_AS(Plot(makeCamera(1, 1), cv::Size(1, 1), Plot::WindowMode::Offscreen), std::invalid_argument);
            }
        }
    }
}

SCENARIO("interactive = 0 (offscreen) uses the headless EGL backend even when DISPLAY is set")
{
    GIVEN("DISPLAY=:1 as on the reference machine, no VTK backend override, and a 160 x 90 camera")
    {
        ScopedEnvVar display("DISPLAY", ":1");
        ScopedEnvVar vtkOverride("VTK_DEFAULT_OPENGL_WINDOW", nullptr);
        const cv::Size size(160, 90);
        const Camera camera = makeCamera(size.width, size.height);
        const std::string required = Plot::requestedBackendClass(Plot::WindowMode::Offscreen);

        WHEN("an offscreen plot is constructed")
        {
            Plot plot(camera, size, Plot::WindowMode::Offscreen);
            INFO("backend in use: " << plot.backendDescription());
            THEN("the window class is the EGL offscreen class, never the X11 window")
            {
                CHECK(required == std::string("vtkEGLRenderWindow"));
                CHECK(plot.backendName() == required);
                CHECK(plot.backendName() != std::string("vtkXOpenGLRenderWindow"));
                CHECK(!plot.isOnscreen());
            }
            THEN("the backend description records the vendor and the OpenGL renderer string")
            {
                CHECK(plot.backendDescription().find("vendor '") != std::string::npos);
                CHECK(plot.backendDescription().find("renderer '") != std::string::npos);
            }
            THEN("the process environment is restored after construction")
            {
                CHECK(std::getenv("VTK_DEFAULT_OPENGL_WINDOW") == nullptr);
                const char * displayEnv = std::getenv("DISPLAY");
                CHECK(displayEnv != nullptr);
                CHECK(std::string(displayEnv ? displayEnv : "") == ":1");
            }
            THEN("frames can be rendered and captured through the offscreen backend")
            {
                plot.update(makeScene(makeAsymmetricOverlay(size)));
                plot.render();
                cv::Mat frame;
                plot.capture(frame);
                CHECK(frame.size() == cv::Size(2 * size.width, size.height));
                CHECK(nonBackgroundFraction(frame(cv::Rect(size.width, 0, size.width, size.height))) > 0.005);
            }
        }

        WHEN("a second offscreen plot is created after the first one")
        {
            Plot first(camera, size, Plot::WindowMode::Offscreen);
            Plot second(camera, size, Plot::WindowMode::Offscreen);
            THEN("both use the same backend and the environment override does not leak into later windows")
            {
                CHECK(second.backendName() == first.backendName());
                CHECK(second.backendName() == required);
            }
        }
    }
}

SCENARIO("offscreen plots do not depend on a display at all")
{
    GIVEN("no DISPLAY and no WAYLAND_DISPLAY")
    {
        ScopedEnvVar display("DISPLAY", nullptr);
        ScopedEnvVar wayland("WAYLAND_DISPLAY", nullptr);
        ScopedEnvVar vtkOverride("VTK_DEFAULT_OPENGL_WINDOW", nullptr);
        const cv::Size size(96, 54);
        const Camera camera = makeCamera(size.width, size.height);
        WHEN("an offscreen plot is constructed and captured")
        {
            Plot plot(camera, size, Plot::WindowMode::Offscreen);
            plot.update(makeScene(makeAsymmetricOverlay(size)));
            plot.render();
            cv::Mat frame;
            plot.capture(frame);
            THEN("the backend is EGL and the frame has the full 2W x H size")
            {
                CHECK(plot.backendName() == Plot::requestedBackendClass(Plot::WindowMode::Offscreen));
                CHECK(frame.size() == cv::Size(2 * size.width, size.height));
            }
        }
    }
}

SCENARIO("the backend selection logic and the GL report parser are deterministic")
{
    GIVEN("the two window modes")
    {
        THEN("offscreen requests the EGL class and onscreen leaves the VTK default (X11 on DISPLAY) in place")
        {
            CHECK(Plot::requestedBackendClass(Plot::WindowMode::Offscreen) == "vtkEGLRenderWindow");
            CHECK(Plot::requestedBackendClass(Plot::WindowMode::Onscreen).empty());
        }
    }
    GIVEN("a vtkRenderWindow::ReportCapabilities() style report")
    {
        const std::string report =
            "EGL vendor string:  NVIDIA\n"
            "OpenGL vendor string:  NVIDIA Corporation\n"
            "OpenGL renderer string:  NVIDIA GeForce RTX 3050 Laptop GPU/PCIe/SSE2  \r\n"
            "OpenGL version string:  4.6.0 NVIDIA 595.99.02\n"
            "OpenGL extensions:  \n";
        THEN("the renderer and vendor fields are extracted and trimmed")
        {
            CHECK(Plot::parseReportField(report, "OpenGL renderer string:") == "NVIDIA GeForce RTX 3050 Laptop GPU/PCIe/SSE2");
            CHECK(Plot::parseReportField(report, "OpenGL vendor string:") == "NVIDIA Corporation");
        }
        THEN("a missing field gives an empty string")
        {
            CHECK(Plot::parseReportField(report, "Nonexistent field:").empty());
            CHECK(Plot::parseReportField("", "OpenGL renderer string:").empty());
        }
    }
}

SCENARIO("a landmark that is no longer reported is removed from the batched geometry")
{
    GIVEN("a 160 x 90 plot and a scene with the base content plus a small landmark inside the content bounds")
    {
        const cv::Size size(160, 90);
        const Camera camera = makeCamera(size.width, size.height);
        const cv::Mat overlay = makeAsymmetricOverlay(size);
        PlotFrameData base = makeScene(overlay, false);
        // The extra landmark sits inside the bounding box of the base content, so the overview camera is unchanged.
        PlotFrameData::Landmark extra;
        extra.id = 77;
        extra.rLNn = Eigen::Vector3d(0.3, 0.05, -0.2);
        extra.positionSqrtCov = Eigen::Vector3d(0.03, 0.03, 0.03).asDiagonal();
        extra.colourRGB = Eigen::Vector3d(1.0, 0.0, 1.0);
        extra.visible = true;
        PlotFrameData withExtra = base;
        withExtra.landmarks.push_back(extra);

        WHEN("the landmark is drawn in frame 1, absent in frame 2 (and the reference plot never draws it)")
        {
            Plot plot(camera, size, Plot::WindowMode::Offscreen);
            Plot reference(camera, size, Plot::WindowMode::Offscreen);
            cv::Mat frame1;
            plot.update(withExtra);
            plot.render();
            plot.capture(frame1);
            cv::Mat frame2;
            plot.update(base);
            plot.render();
            plot.capture(frame2);
            cv::Mat refFrame;
            reference.update(base);
            reference.render();
            reference.capture(refFrame);

            // the extra landmark is pure magenta (1, 0, 1); the validation track is not used in this scene
            auto magentaPixels = [&](const cv::Mat & bgr)
            {
                const cv::Mat right = bgr(cv::Rect(size.width, 0, size.width, size.height));
                long count = 0;
                for (int y = 0; y < right.rows; ++y)
                {
                    const cv::Vec3b * row = right.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < right.cols; ++x)
                        count += (row[x][0] > 200 && row[x][2] > 200 && row[x][1] < 20) ? 1 : 0;
                }
                return count;
            };
            THEN("the landmark is drawn in frame 1 (the comparison is meaningful)")
            {
                INFO("magenta pixels in frame 1 = " << magentaPixels(frame1));
                CHECK(magentaPixels(frame1) > 3);
            }
            THEN("no pixel of the removed landmark is drawn in frame 2")
            {
                INFO("magenta pixels in frame 2 = " << magentaPixels(frame2));
                CHECK(magentaPixels(frame2) == 0);
            }
            THEN("frame 2 is identical to a fresh plot that never had the landmark (no stale cells are drawn)")
            {
                const double diff = meanAbsDiff(frame2(cv::Rect(size.width, 0, size.width, size.height)),
                                                refFrame(cv::Rect(size.width, 0, size.width, size.height)));
                INFO("right-pane mean abs diff frame 2 vs fresh reference = " << diff);
                CHECK(diff < 0.05);
            }
        }

        WHEN("the number of landmarks grows over several frames (batch slots are allocated on the fly)")
        {
            Plot plot(camera, size, Plot::WindowMode::Offscreen);
            PlotFrameData d = makeScene(overlay, false);
            for (int step = 0; step < 3; ++step)
            {
                for (int i = 0; i < 20; ++i)
                {
                    PlotFrameData::Landmark lm;
                    lm.id = 100 + step * 20 + i;
                    lm.isPose = (i % 5 == 0);
                    lm.rLNn = Eigen::Vector3d(-1.0 + 0.1 * i, 0.5 * step - 0.5, -0.2 - 0.02 * i);
                    lm.Rnl = Eigen::Matrix3d::Identity();
                    lm.size = 0.2;
                    lm.positionSqrtCov = Eigen::Vector3d(0.03, 0.02, 0.02).asDiagonal();
                    lm.colourRGB = Eigen::Vector3d(0.2 + 0.03 * i, 0.9, 0.4);
                    lm.visible = (i + step) % 2 == 0;
                    d.landmarks.push_back(lm);
                }
                plot.update(d);
                plot.render();
            }
            cv::Mat frame;
            plot.capture(frame);
            THEN("the 60 landmarks render into a non-empty, correctly sized frame")
            {
                CHECK(frame.size() == cv::Size(2 * size.width, size.height));
                CHECK(nonBackgroundFraction(frame(cv::Rect(size.width, 0, size.width, size.height))) > 0.01);
                CHECK(std::isfinite(plot.lastTimings().renderMs));
            }
        }
    }
}
