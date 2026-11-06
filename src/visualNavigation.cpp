#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <print>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include "BufferedVideo.h"
#include "Camera.h"
#include "FrameTimer.h"
#include "Plot.h"
#include "ScenarioRunner.h"
#include "visualNavigation.h"

namespace
{

constexpr std::size_t kReaderBufferFrames = 5;   ///< BufferedVideoReader queue depth (unchanged architecture)
constexpr std::size_t kWriterBufferFrames = 3;   ///< BufferedVideoWriter queue depth (unchanged architecture)
constexpr double kFallbackFps = 30.0;

using Clock = FrameTimer::Clock;

std::unique_ptr<ScenarioRunner> makeRunnerForScenario(int scenario)
{
    switch (scenario)
    {
    case 4:
        return makeScenario4Runner();
    case 5:
        return makeScenario5Runner();
    case 6:
        return makeScenario6Runner();
    default:
        return nullptr;
    }
}

void printCalibration(const Camera & camera, const std::filesystem::path & cameraPath)
{
    const Camera::CalibrationMetadata & md = camera.metadata();
    std::println("Camera calibration: {}", cameraPath.string());
    std::println("  image size      {} x {} px", camera.imageWidth(), camera.imageHeight());
    std::println("  fx, fy          {:.3f}, {:.3f} px", camera.fx(), camera.fy());
    std::println("  cx, cy          {:.3f}, {:.3f} px", camera.cx(), camera.cy());
    std::print("  distortion      [");
    for (std::size_t i = 0; i < camera.distortionCoefficients().size(); ++i)
        std::print("{}{:.6g}", i > 0 ? ", " : "", camera.distortionCoefficients()[i]);
    std::println("] ({})", md.distortionModel.empty() ? std::string("unspecified") : md.distortionModel);
    if (std::isfinite(md.rmsReprojectionErrorPx))
        std::println("  RMS reprojection error {:.3f} px over {} views / {} points", md.rmsReprojectionErrorPx,
                     md.numViews, md.numPoints);
    if (md.boardInnerCornersCols > 0)
        std::println("  calibration board {} x {} inner corners, square {:.4f} m", md.boardInnerCornersCols,
                     md.boardInnerCornersRows, md.squareSizeM);
}

/// Error path: shuts the buffered I/O threads down (flushing queued frames) and never throws, so that the original
/// exception that is unwinding the frame loop is the one the caller sees.
void stopPipeline(BufferedVideoWriter & writer, BufferedVideoReader & reader) noexcept
{
    try
    {
        writer.stop();
    }
    catch (...)
    {
    }
    try
    {
        reader.stop();
    }
    catch (...)
    {
    }
}

/// Normal exit: shuts both threads down and reports the first worker failure (a failed export encode must not be
/// silent). Both stops are always attempted.
void stopPipelineOrThrow(BufferedVideoWriter & writer, BufferedVideoReader & reader)
{
    std::exception_ptr first;
    try
    {
        writer.stop();
    }
    catch (...)
    {
        first = std::current_exception();
    }
    try
    {
        reader.stop();
    }
    catch (...)
    {
        if (!first)
            first = std::current_exception();
    }
    if (first)
        std::rethrow_exception(first);
}

} // namespace

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    if (scenario < 4 || scenario > 6)
        throw std::invalid_argument(std::format("unsupported scenario {}: visual navigation supports 4 (flight), "
                                                "5 (identical ArUco tags) and 6 (room point landmarks)", scenario));

    if (scenario == 4)
    {
        const std::filesystem::path subtitlePath = videoPath.parent_path() / (videoPath.stem().string() + ".SRT");
        std::println("Subtitle file: {}", subtitlePath.string());
    }

    std::unique_ptr<ScenarioRunner> runner = makeRunnerForScenario(scenario);
    if (!runner)
        throw std::invalid_argument(std::format("no runner available for scenario {}", scenario));

    runVisualNavigation(videoPath, cameraPath, std::move(runner), interactive, outputDirectory);
}

void runVisualNavigation(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, std::unique_ptr<ScenarioRunner> runner, int interactive, const std::filesystem::path & outputDirectory)
{
    if (!runner)
        throw std::invalid_argument("runVisualNavigation: no scenario runner given");
    if (interactive < 0 || interactive > 2)
        throw std::invalid_argument(std::format("interactive must be 0, 1 or 2 (got {})", interactive));
    if (videoPath.empty())
        throw std::invalid_argument("runVisualNavigation: no input video given");

    const bool interactiveWindow = interactive != 0;
    const bool doExport = !outputDirectory.empty();
    if (interactiveWindow && !Plot::displayAvailable())
        throw std::runtime_error(std::format("--interactive={} needs a display, but DISPLAY is not set. "
                                             "Re-run with --interactive=0 (no window) or from a desktop session.",
                                             interactive));

    // ----- 1. calibration -----
    if (!std::filesystem::exists(cameraPath))
        throw std::runtime_error(std::format("camera calibration not found: {}\n"
                                             "Run the calibration first:  ./a2 {} -c",
                                             cameraPath.string(),
                                             (cameraPath.parent_path() / "config.xml").string()));
    const Camera camera = Camera::load(cameraPath);
    printCalibration(camera, cameraPath);

    // ----- 2. input video and validation -----
    if (!std::filesystem::exists(videoPath))
        throw std::runtime_error(std::format("input video not found: {}", videoPath.string()));
    cv::VideoCapture cap(videoPath.string());
    if (!cap.isOpened())
        throw std::runtime_error(std::format("cannot open input video: {}", videoPath.string()));

    const cv::Size inputSize(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                             static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (inputSize != cv::Size(camera.imageWidth(), camera.imageHeight()))
        throw std::runtime_error(std::format("input frame size {}x{} does not match the calibrated camera {}x{} ({}); "
                                             "use the camera.xml generated for this video",
                                             inputSize.width, inputSize.height, camera.imageWidth(),
                                             camera.imageHeight(), cameraPath.string()));

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(std::isfinite(fps) && fps > 0.0))
    {
        std::println("warning: input video reports no frame rate; assuming {:.1f} fps", kFallbackFps);
        fps = kFallbackFps;
    }
    const int nFramesTotal = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));   // <= 0 if unknown
    std::println("Input video: {} ({}x{}, {:.3f} fps, {} frames)", videoPath.string(), inputSize.width,
                 inputSize.height, fps, nFramesTotal > 0 ? std::to_string(nFramesTotal) : std::string("unknown"));

    // ----- 3. export set-up -----
    const cv::Size outputSize(2 * inputSize.width, inputSize.height);
    std::filesystem::path outputPath;
    std::filesystem::path timingPath;
    if (doExport)
    {
        if (inputSize.height % 2 != 0)
            throw std::runtime_error(std::format("export needs an even frame height for mp4v (got {})", inputSize.height));
        std::filesystem::create_directories(outputDirectory);
        const std::string stem = videoPath.stem().string();
        outputPath = outputDirectory / (stem + "_out" + videoPath.extension().string());
        timingPath = outputDirectory / (stem + "_timing.csv");
    }

    // ----- 4. scenario initialisation -----
    const std::string scenarioName = runner->name();
    std::println("Scenario: {}", scenarioName);
    try
    {
        runner->initialise(camera, fps, inputSize, videoPath.string());
    }
    catch (const std::exception & e)
    {
        throw std::runtime_error(std::format("{}: initialisation failed: {}", scenarioName, e.what()));
    }

    // ----- 5. visualisation -----
    Plot plot(camera, inputSize, interactiveWindow ? Plot::WindowMode::Onscreen : Plot::WindowMode::Offscreen);
    std::println("Visualisation: {}x{} ({} window)", outputSize.width, outputSize.height,
                 interactiveWindow ? "on-screen" : "offscreen");
    std::println("Visualisation backend: {}", plot.backendDescription());
    if (!interactiveWindow && plot.backendName() != Plot::requestedBackendClass(Plot::WindowMode::Offscreen))
        std::println("warning: offscreen rendering requested {} but VTK created {}; readback will be slow",
                     Plot::requestedBackendClass(Plot::WindowMode::Offscreen), plot.backendName());

    // ----- 6. buffered I/O (declared in destruction-safe order: writer before the VideoWriter it uses) -----
    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(static_cast<int>(kWriterBufferFrames));
    BufferedVideoReader bufferedVideoReader(static_cast<int>(kReaderBufferFrames));

    if (doExport)
    {
        const int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoOut.open(outputPath.string(), codec, fps, outputSize);
        if (!videoOut.isOpened())
            throw std::runtime_error(std::format("cannot open output video for writing: {}", outputPath.string()));
        bufferedVideoWriter.start(videoOut);
        std::println("Export: {} ({}x{}, mp4v, {:.3f} fps)", outputPath.string(), outputSize.width,
                     outputSize.height, fps);
    }
    bufferedVideoReader.start(cap);

    // ----- 7. frame loop -----
    FrameTimer timer;
    cv::Mat imgout;
    int frameIndex = 0;
    bool stoppedByUser = false;
    const auto loopStart = Clock::now();
    const int progressStep = nFramesTotal > 0 ? std::max(1, nFramesTotal / 20) : 100;

    try
    {
        while (true)
        {
            // read (waits for the reader thread)
            const auto tRead0 = Clock::now();
            cv::Mat imgin = bufferedVideoReader.read();
            const auto tRead1 = Clock::now();
            if (imgin.empty())
                break;   // end of stream (or decode failure, reported below)
            if (imgin.size() != inputSize || imgin.type() != CV_8UC3)
                throw std::runtime_error(std::format("frame {}: unexpected size {}x{} / type {} (expected {}x{} BGR8)",
                                                     frameIndex, imgin.cols, imgin.rows, imgin.type(),
                                                     inputSize.width, inputSize.height));

            const double t = static_cast<double>(frameIndex) / fps;

            // process
            try
            {
                runner->processFrame(imgin, frameIndex, t);
            }
            catch (const std::exception & e)
            {
                throw std::runtime_error(std::format("{} failed at frame {} (t = {:.3f} s): {}", scenarioName,
                                                     frameIndex, t, e.what()));
            }
            const auto tProc1 = Clock::now();

            // visualise (+ capture when exporting)
            double plotDataMs = 0.0;
            try
            {
                const auto tData0 = Clock::now();
                const PlotFrameData data = runner->plotData();
                plotDataMs = FrameTimer::elapsedMs(tData0, Clock::now());
                plot.update(data);
                plot.render();
                if (doExport)
                    plot.capture(imgout);
            }
            catch (const std::exception & e)
            {
                throw std::runtime_error(std::format("visualisation failed at frame {} (t = {:.3f} s): {}",
                                                     frameIndex, t, e.what()));
            }
            const auto tPlot1 = Clock::now();

            // export enqueue (the encode runs on the writer thread)
            if (doExport)
                bufferedVideoWriter.write(imgout);
            const auto tWrite1 = Clock::now();

            FrameTimingSample sample;
            sample.frame = frameIndex;
            sample.t = t;
            sample.readMs = FrameTimer::elapsedMs(tRead0, tRead1);
            sample.processMs = FrameTimer::elapsedMs(tRead1, tProc1);
            sample.plotMs = FrameTimer::elapsedMs(tProc1, tPlot1);
            sample.writeMs = FrameTimer::elapsedMs(tPlot1, tWrite1);
            sample.totalMs = FrameTimer::elapsedMs(tRead0, tWrite1);
            sample.plotDataMs = plotDataMs;
            sample.plotUpdateMs = plot.lastTimings().updateMs;
            sample.plotRenderMs = plot.lastTimings().renderMs;
            sample.plotCaptureMs = doExport ? plot.lastTimings().captureMs : 0.0;
            timer.add(sample);

            // interactive mode 2: show this frame and wait (human time, not in the timings)
            if (interactive == 2 && !plot.showAndWait())
            {
                stoppedByUser = true;
            }

            ++frameIndex;
            if (frameIndex % progressStep == 0)
            {
                if (nFramesTotal > 0)
                    std::println("  frame {}/{} ({:.0f} %)", frameIndex, nFramesTotal,
                                 100.0 * frameIndex / nFramesTotal);
                else
                    std::println("  frame {}", frameIndex);
            }

            if (stoppedByUser)
            {
                std::println("Window closed / quit requested: stopping after frame {}", frameIndex - 1);
                break;
            }
        }
    }
    catch (...)
    {
        stopPipeline(bufferedVideoWriter, bufferedVideoReader);
        throw;
    }

    // ----- 8. shut down the I/O threads (flushes the exported video; a worker failure is reported, not dropped) -----
    stopPipelineOrThrow(bufferedVideoWriter, bufferedVideoReader);
    const double wallSeconds = FrameTimer::elapsedMs(loopStart, Clock::now()) / 1000.0;

    if (frameIndex == 0)
        throw std::runtime_error(std::format("no frames could be read from {}", videoPath.string()));
    if (!stoppedByUser && nFramesTotal > 0 && frameIndex < nFramesTotal)
        std::println("warning: input ended after {} of {} reported frames (decode stopped early)", frameIndex,
                     nFramesTotal);

    std::print("{}", timer.summary(wallSeconds));
    if (doExport)
    {
        timer.writeCsv(timingPath);
        std::println("Timing CSV: {}", timingPath.string());
        std::println("Exported {} frames to {}", frameIndex, outputPath.string());
    }

    // ----- 9. interactive mode 1: block on the last frame -----
    if (interactive == 1 && !stoppedByUser)
    {
        std::println("Showing the last frame. Close the window or press 'q' to exit.");
        plot.blockUntilClosed();
    }
}
