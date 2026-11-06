#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <format>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "../../src/Camera.h"
#include "../../src/FrameTimer.h"
#include "../../src/Plot.h"
#include "../../src/ScenarioRunner.h"
#include "../../src/visualNavigation.h"
#include "visualNavigationSupport.h"

using namespace std::chrono_literals;
using visTest::TempDir;
using visTest::finishesWithin;

namespace
{

/// Observations of a DummyScenarioRunner, shared with the test (the runner itself is owned by the loop).
struct DummyStats
{
    int initialiseCalls = 0;
    int processed = 0;
    std::vector<int> frameIndices;
    std::vector<double> frameTimes;
    double fps = 0.0;
    cv::Size frameSize;
    std::string videoPath;
};

/**
 * @brief Test double for the scenario estimators (Scenario 4/5/6 are not required here).
 *
 * The overlay is the input frame with a moving marker. The trajectory grows by one point per frame, five
 * landmarks alternate between visible and not visible, and an optional pose landmark is added.
 * failAtFrame makes processFrame() throw; failInitialise makes initialise() throw.
 */
class DummyScenarioRunner : public ScenarioRunner
{
public:
    struct Options
    {
        int failAtFrame = -1;
        bool failInitialise = false;
        bool poseLandmark = false;
    };

    DummyScenarioRunner(std::shared_ptr<DummyStats> stats, Options options)
        : stats_(std::move(stats))
        , options_(options)
    {}

    void initialise(const Camera & camera, double fps, cv::Size frameSize, const std::string & videoPath) override
    {
        stats_->initialiseCalls++;
        stats_->fps = fps;
        stats_->frameSize = frameSize;
        stats_->videoPath = videoPath;
        if (camera.imageWidth() != frameSize.width || camera.imageHeight() != frameSize.height)
            throw std::runtime_error("dummy: camera/frame size mismatch");
        camera_ = std::make_unique<Camera>(camera);
        if (options_.failInitialise)
            throw std::runtime_error("dummy init failure");
    }

    void processFrame(const cv::Mat & frame, int frameIndex, double t) override
    {
        if (frameIndex == options_.failAtFrame)
            throw std::runtime_error("dummy processing failure");
        overlay_ = frame.clone();
        cv::rectangle(overlay_, cv::Rect(frameIndex % std::max(1, frame.cols - 6), frame.rows / 2, 6, 6),
                      cv::Scalar(0, 0, 255), cv::FILLED);
        const double phase = 0.1 * frameIndex;
        trajectory_.emplace_back(0.3 * std::cos(phase), 0.05 * frameIndex, 0.3 * std::sin(phase));
        t_ = t;
        frameIndex_ = frameIndex;
        stats_->processed++;
        stats_->frameIndices.push_back(frameIndex);
        stats_->frameTimes.push_back(t);
    }

    PlotFrameData plotData() const override
    {
        PlotFrameData d;
        d.overlay = overlay_;
        d.rCNn = trajectory_.empty() ? Eigen::Vector3d::Zero() : trajectory_.back();
        d.Rnc = Eigen::Matrix3d::Identity();
        d.positionSqrtCov = Eigen::Matrix3d::Identity() * 0.02;
        d.trajectory = trajectory_;
        for (int i = 0; i < 5; ++i)
        {
            PlotFrameData::Landmark lm;
            lm.id = i;
            lm.rLNn = Eigen::Vector3d(1.0 * std::cos(0.9 * i), 0.5 * i - 1.0, 1.0 * std::sin(0.9 * i));
            lm.positionSqrtCov = Eigen::Vector3d(0.05, 0.03, 0.02).asDiagonal();
            lm.colourRGB = Eigen::Vector3d(0.2 + 0.15 * i, 1.0 - 0.15 * i, 0.5);
            lm.visible = (frameIndex_ + i) % 2 == 0;
            d.landmarks.push_back(lm);
        }
        if (options_.poseLandmark)
        {
            PlotFrameData::Landmark tag;
            tag.id = 100;
            tag.isPose = true;
            tag.rLNn = Eigen::Vector3d(-0.5, 0.0, 0.8);
            tag.size = 0.2;
            tag.positionSqrtCov = Eigen::Vector3d(0.02, 0.02, 0.01).asDiagonal();
            tag.colourRGB = Eigen::Vector3d(1.0, 0.8, 0.1);
            tag.visible = true;
            d.landmarks.push_back(tag);
        }
        d.validationTrack = trajectory_;
        for (auto & p : d.validationTrack)
            p += Eigen::Vector3d(0.05, 0.0, 0.05);
        d.statusLines = {std::format("frame {}", frameIndex_), std::format("t = {:.3f} s", t_)};
        return d;
    }

    std::string name() const override
    {
        return "Dummy scenario (test double)";
    }

private:
    std::shared_ptr<DummyStats> stats_;
    Options options_;
    std::unique_ptr<Camera> camera_;
    cv::Mat overlay_;
    std::vector<Eigen::Vector3d> trajectory_;
    double t_ = 0.0;
    int frameIndex_ = 0;
};

/// Writes a short coded input video and a matching camera.xml into dir. Returns the video path.
std::filesystem::path writeInputFixture(const std::filesystem::path & dir, cv::Size size, int nFrames, double fps,
                                        const std::string & stem = "synthetic", const std::string & ext = ".avi")
{
    const auto video = dir / (stem + ext);
    visTest::writeCodedVideo(video, size, nFrames, fps);
    visTest::makeCamera(size.width, size.height).save(dir / "camera.xml");
    return video;
}

/// Sets (or unsets) an environment variable for the lifetime of the guard and restores it afterwards.
class ScopedEnv
{
public:
    ScopedEnv(const char * name, const char * value)
        : name_(name)
    {
        const char * old = std::getenv(name);
        hadOld_ = old != nullptr;
        if (old)
            old_ = old;
        if (value)
            setenv(name_.c_str(), value, 1);
        else
            unsetenv(name_.c_str());
    }
    ~ScopedEnv()
    {
        if (hadOld_)
            setenv(name_.c_str(), old_.c_str(), 1);
        else
            unsetenv(name_.c_str());
    }
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv & operator=(const ScopedEnv &) = delete;

private:
    std::string name_;
    std::string old_;
    bool hadOld_ = false;
};

std::vector<std::string> readLines(const std::filesystem::path & path)
{
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line))
        lines.push_back(line);
    return lines;
}

} // namespace

SCENARIO("the frame loop runs end-to-end on a synthetic video and exports a 2W x H video and a timing CSV")
{
    GIVEN("a 30-frame 160 x 90 input video at 25 fps, a matching camera.xml, and an output directory")
    {
        auto dir = std::make_shared<TempDir>();
        const cv::Size size(160, 90);
        const auto video = writeInputFixture(dir->path(), size, 30, 25.0);
        const auto outDir = dir->path() / "out";
        auto stats = std::make_shared<DummyStats>();

        WHEN("runVisualNavigation runs the dummy scenario offscreen with export")
        {
            runVisualNavigation(video, dir->path() / "camera.xml",
                                std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{}), 0, outDir);

            THEN("the scenario is initialised once with the video metadata")
            {
                CHECK(stats->initialiseCalls == 1);
                CHECK(stats->frameSize == size);
                CHECK(stats->fps == doctest::Approx(25.0));
                CHECK(stats->videoPath == video.string());
            }

            THEN("every frame is processed exactly once, in order")
            {
                REQUIRE(stats->processed == 30);
                bool inOrder = true;
                for (int i = 0; i < 30; ++i)
                    inOrder = inOrder && stats->frameIndices[static_cast<std::size_t>(i)] == i
                                      && std::abs(stats->frameTimes[static_cast<std::size_t>(i)] - i / 25.0) < 1e-9;
                CHECK(inOrder);
            }

            THEN("the export is named out/<stem>_out<ext> and reopens as 2W x H with 30 frames at 25 fps")
            {
                const auto exported = outDir / "synthetic_out.avi";
                REQUIRE(std::filesystem::exists(exported));
                cv::VideoCapture cap(exported.string());
                REQUIRE(cap.isOpened());
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) == 2 * size.width);
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) == size.height);
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)) == 30);
                CHECK(cap.get(cv::CAP_PROP_FPS) == doctest::Approx(25.0).epsilon(0.02));

                cv::Mat frame;
                int decoded = 0;
                bool sizesOk = true;
                bool notBlank = true;
                while (cap.read(frame))
                {
                    sizesOk = sizesOk && frame.cols == 2 * size.width && frame.rows == size.height;
                    notBlank = notBlank && cv::countNonZero(frame.reshape(1)) > 0;
                    ++decoded;
                }
                CHECK(decoded == 30);
                CHECK(sizesOk);
                CHECK(notBlank);
            }

            THEN("the timing CSV has the exact header, one row per frame and consistent columns")
            {
                const auto csv = outDir / "synthetic_timing.csv";
                REQUIRE(std::filesystem::exists(csv));
                const std::vector<std::string> lines = readLines(csv);
                REQUIRE(lines.size() == 31u);
                CHECK(lines[0] == "frame,t,process_ms,plot_ms,write_ms,total_ms,read_ms,plot_data_ms,plot_update_ms,"
                                  "plot_render_ms,plot_capture_ms");
                for (std::size_t r = 1; r < lines.size(); ++r)
                {
                    std::vector<double> values;
                    std::string cell;
                    std::stringstream ss(lines[r]);
                    while (std::getline(ss, cell, ','))
                        values.push_back(std::stod(cell));
                    REQUIRE(values.size() == 11u);
                    CHECK(values[0] == doctest::Approx(static_cast<double>(r - 1)));
                    CHECK(values[1] == doctest::Approx(static_cast<double>(r - 1) / 25.0).epsilon(1e-5));
                    for (std::size_t c = 2; c < 11; ++c)
                        CHECK(values[c] >= 0.0);
                    // the stages are contiguous in the loop: total = read + process + plot + write (CSV rounding 1e-4)
                    const double read = values[6];
                    CHECK(std::abs(values[5] - (read + values[2] + values[3] + values[4])) < 1e-3);
                    // the plot stage contains data + update + render + capture
                    CHECK(values[3] + 1e-3 >= values[7] + values[8] + values[9] + values[10]);
                    // export is on, so the capture stage was measured
                    CHECK(values[10] > 0.0);
                }
            }
        }
    }
}

SCENARIO("without an output directory nothing is exported and no timing CSV is written")
{
    GIVEN("a 12-frame input and an empty output directory argument")
    {
        auto dir = std::make_shared<TempDir>();
        const auto video = writeInputFixture(dir->path(), cv::Size(96, 54), 12, 30.0);
        auto stats = std::make_shared<DummyStats>();

        WHEN("runVisualNavigation runs with no export")
        {
            runVisualNavigation(video, dir->path() / "camera.xml",
                                std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{}));
            THEN("all frames are still processed and the directory holds only the input and camera files")
            {
                CHECK(stats->processed == 12);
                std::size_t files = 0;
                for (const auto & entry : std::filesystem::directory_iterator(dir->path()))
                {
                    (void)entry;
                    ++files;
                }
                CHECK(files == 2u);   // synthetic.avi and camera.xml
            }
        }
    }
}

SCENARIO("a failing scenario stops the pipeline and reports the frame index")
{
    GIVEN("a 20-frame input whose runner throws at frame 7")
    {
        auto dir = std::make_shared<TempDir>();
        const auto video = writeInputFixture(dir->path(), cv::Size(96, 54), 20, 30.0);
        const auto outDir = dir->path() / "out";
        auto stats = std::make_shared<DummyStats>();

        WHEN("the export is running")
        {
            // Shared state: the worker may outlive this scope if the call hangs (then the test fails).
            auto outcome = std::make_shared<std::string>();
            auto threw = std::make_shared<bool>(false);
            const auto camPath = dir->path() / "camera.xml";
            const bool returned = finishesWithin([dir, video, outDir, stats, outcome, threw, camPath]
            {
                try
                {
                    runVisualNavigation(video, camPath,
                                        std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{.failAtFrame = 7}),
                                        0, outDir);
                }
                catch (const std::runtime_error & e)
                {
                    *threw = true;
                    *outcome = e.what();
                }
            }, 15000ms);

            THEN("the call returns (no thread left blocked) with a runtime_error naming frame 7")
            {
                CHECK(returned);
                CHECK(*threw);
                CHECK(outcome->find("frame 7") != std::string::npos);
                CHECK(outcome->find("dummy processing failure") != std::string::npos);
            }
            THEN("frames 0 to 6 were processed before the failure")
            {
                CHECK(stats->processed == 7);
            }
        }
    }
}

SCENARIO("a runner that fails in initialise() is reported with the scenario name")
{
    GIVEN("a valid input and a runner whose initialise() throws")
    {
        auto dir = std::make_shared<TempDir>();
        const auto video = writeInputFixture(dir->path(), cv::Size(96, 54), 5, 30.0);
        auto stats = std::make_shared<DummyStats>();

        WHEN("the visual navigation starts")
        {
            std::string message;
            try
            {
                runVisualNavigation(video, dir->path() / "camera.xml",
                                    std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{.failInitialise = true}));
            }
            catch (const std::runtime_error & e)
            {
                message = e.what();
            }
            THEN("the error names the scenario and the initialisation step, and no frame is processed")
            {
                CHECK(message.find("Dummy scenario") != std::string::npos);
                CHECK(message.find("initialisation failed") != std::string::npos);
                CHECK(stats->processed == 0);
            }
        }
    }
}

SCENARIO("missing or inconsistent calibration and video inputs are rejected with clear messages")
{
    GIVEN("a 160 x 90 input video in a directory without camera.xml")
    {
        auto dir = std::make_shared<TempDir>();
        const auto video = dir->path() / "nocal.avi";
        visTest::writeCodedVideo(video, cv::Size(160, 90), 3, 30.0);
        auto stats = std::make_shared<DummyStats>();

        WHEN("the camera file is missing")
        {
            std::string message;
            try
            {
                runVisualNavigation(video, dir->path() / "camera.xml",
                                    std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{}));
            }
            catch (const std::runtime_error & e)
            {
                message = e.what();
            }
            THEN("the error tells the user to run the calibration first, with the config.xml path")
            {
                CHECK(message.find("camera calibration not found") != std::string::npos);
                CHECK(message.find(" -c") != std::string::npos);
                CHECK(message.find("config.xml") != std::string::npos);
                CHECK(stats->initialiseCalls == 0);
            }
        }

        WHEN("camera.xml describes a 320 x 180 camera")
        {
            visTest::makeCamera(320, 180).save(dir->path() / "camera.xml");
            std::string message;
            try
            {
                runVisualNavigation(video, dir->path() / "camera.xml",
                                    std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{}));
            }
            catch (const std::runtime_error & e)
            {
                message = e.what();
            }
            THEN("the frame size mismatch is reported and nothing is initialised")
            {
                CHECK(message.find("does not match") != std::string::npos);
                CHECK(message.find("160x90") != std::string::npos);
                CHECK(stats->initialiseCalls == 0);
            }
        }

        WHEN("the input video does not exist")
        {
            visTest::makeCamera(160, 90).save(dir->path() / "camera.xml");
            bool threw = false;
            try
            {
                runVisualNavigation(dir->path() / "missing.avi", dir->path() / "camera.xml",
                                    std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{}));
            }
            catch (const std::runtime_error &)
            {
                threw = true;
            }
            THEN("a runtime_error is thrown")
            {
                CHECK(threw);
            }
        }
    }
}

SCENARIO("unsupported scenario numbers are rejected before any file is touched")
{
    GIVEN("a camera path and a video path that do not exist")
    {
        const std::filesystem::path none = "/nonexistent/path/that/does/not/exist/video.MOV";
        WHEN("the scenario is 0, 3 or 7")
        {
            THEN("runVisualNavigationFromVideo throws std::invalid_argument")
            {
                CHECK_THROWS_AS(runVisualNavigationFromVideo(none, none, 0), std::invalid_argument);
                CHECK_THROWS_AS(runVisualNavigationFromVideo(none, none, 3), std::invalid_argument);
                CHECK_THROWS_AS(runVisualNavigationFromVideo(none, none, 7), std::invalid_argument);
            }
        }
        WHEN("the interactive value is outside 0..2")
        {
            auto stats = std::make_shared<DummyStats>();
            THEN("runVisualNavigation throws std::invalid_argument")
            {
                CHECK_THROWS_AS(runVisualNavigation(none, none, std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{}), 3),
                                std::invalid_argument);
            }
        }
    }
}

SCENARIO("interactive modes 1 and 2 fail clearly when no display is available")
{
    GIVEN("an environment without DISPLAY or WAYLAND_DISPLAY and a valid input")
    {
        ScopedEnv display("DISPLAY", nullptr);
        ScopedEnv wayland("WAYLAND_DISPLAY", nullptr);
        auto dir = std::make_shared<TempDir>();
        const auto video = writeInputFixture(dir->path(), cv::Size(96, 54), 4, 30.0);
        auto stats = std::make_shared<DummyStats>();

        WHEN("visual navigation is started with --interactive=1 or 2")
        {
            THEN("it throws a runtime_error that says a display is needed, before any frame is processed")
            {
                for (int mode : {1, 2})
                {
                    std::string message;
                    try
                    {
                        runVisualNavigation(video, dir->path() / "camera.xml",
                                            std::make_unique<DummyScenarioRunner>(stats, DummyScenarioRunner::Options{}), mode);
                    }
                    catch (const std::runtime_error & e)
                    {
                        message = e.what();
                    }
                    INFO("interactive mode " << mode);
                    CHECK(message.find("display") != std::string::npos);
                }
                CHECK(stats->processed == 0);
                CHECK_FALSE(Plot::displayAvailable());
            }
        }
    }
}

SCENARIO("the timing helper computes percentiles and summary statistics")
{
    GIVEN("the samples 1, 2, 3, 4, 5")
    {
        const std::vector<double> v{5.0, 1.0, 4.0, 2.0, 3.0};
        THEN("the median is 3, the 95th percentile interpolates to 4.8, the mean is 3 and the max is 5")
        {
            CHECK(FrameTimer::percentile(v, 0.5) == doctest::Approx(3.0));
            CHECK(FrameTimer::percentile(v, 0.95) == doctest::Approx(4.8));
            CHECK(FrameTimer::percentile(v, 0.0) == doctest::Approx(1.0));
            CHECK(FrameTimer::percentile(v, 1.0) == doctest::Approx(5.0));
            const StageStatistics s = FrameTimer::statistics(v);
            CHECK(s.mean == doctest::Approx(3.0));
            CHECK(s.max == doctest::Approx(5.0));
            CHECK(s.p50 == doctest::Approx(3.0));
        }
        THEN("an empty set gives zeros")
        {
            CHECK(FrameTimer::percentile({}, 0.95) == 0.0);
            CHECK(FrameTimer::statistics({}).mean == 0.0);
        }
    }
}

SCENARIO("the timing CSV writer produces the documented header and one row per sample")
{
    GIVEN("two timing samples")
    {
        auto dir = std::make_shared<TempDir>();
        FrameTimer timer;
        FrameTimingSample a;
        a.frame = 0;
        a.t = 0.0;
        a.processMs = 1.5;
        a.plotMs = 2.25;
        a.writeMs = 0.5;
        a.totalMs = 4.5;
        FrameTimingSample b = a;
        b.frame = 1;
        b.t = 1.0 / 30.0;
        timer.add(a);
        timer.add(b);

        WHEN("the CSV is written")
        {
            const auto path = dir->path() / "timing.csv";
            timer.writeCsv(path);
            const auto lines = readLines(path);
            THEN("the header is the documented one (original six columns first, appended read and plot split) and rows follow in order")
            {
                REQUIRE(lines.size() == 3u);
                CHECK(lines[0] == "frame,t,process_ms,plot_ms,write_ms,total_ms,read_ms,plot_data_ms,plot_update_ms,"
                                  "plot_render_ms,plot_capture_ms");
                CHECK(lines[0] == FrameTimer::csvHeader());
                CHECK(lines[1].rfind("0,", 0) == 0);
                CHECK(lines[2].rfind("1,", 0) == 0);
                CHECK(lines[1].find("1.5000") != std::string::npos);
            }
        }
        WHEN("the CSV is written to a directory that does not exist")
        {
            THEN("writeCsv throws std::runtime_error")
            {
                CHECK_THROWS_AS(timer.writeCsv(dir->path() / "no_such_dir" / "t.csv"), std::runtime_error);
            }
        }
    }
}

SCENARIO("the timing summary lists read, process, plot (with its split), write and total")
{
    GIVEN("one timing sample with every stage filled in")
    {
        FrameTimer timer;
        FrameTimingSample a;
        a.readMs = 1.0;
        a.processMs = 2.0;
        a.plotMs = 3.0;
        a.plotDataMs = 0.1;
        a.plotUpdateMs = 1.2;
        a.plotRenderMs = 1.5;
        a.plotCaptureMs = 0.2;
        a.writeMs = 0.5;
        a.totalMs = 6.5;
        timer.add(a);
        WHEN("the summary is formatted")
        {
            const std::string text = timer.summary(1.0);
            THEN("every stage and the plot split appear on their own line")
            {
                for (const char * name : {"read", "process", "plot", "data", "update", "render", "capture", "write", "total"})
                    CHECK(text.find(name) != std::string::npos);
                CHECK(text.find("over 1 frames") != std::string::npos);
            }
        }
    }
}
