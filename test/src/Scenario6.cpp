// Scenario 6 adapter (ScenarioRunner contract): factory, initialisation checks, overlay and plot data.
#include <memory>
#include <stdexcept>
#include <doctest/doctest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "../../src/Camera.h"
#include "../../src/ScenarioRunner.h"

namespace
{

Camera scenario6Camera()
{
    return Camera(640, 480, (Eigen::Matrix3d() << 320.0, 0.0, 319.5, 0.0, 320.0, 239.5, 0.0, 0.0, 1.0).finished(),
                  {-0.02, 0.005, 0.0, 0.0, 0.0});
}

} // namespace

SCENARIO("Scenario 6 runner: factory, initialisation checks and the frame contract")
{
    GIVEN("the factory and a camera matching a 640 x 480 input")
    {
        std::unique_ptr<ScenarioRunner> runner = makeScenario6Runner();
        REQUIRE(runner != nullptr);
        THEN("the name identifies Scenario 6")
        {
            CHECK(runner->name().find("Scenario 6") != std::string::npos);
        }
        WHEN("a frame is processed before initialisation")
        {
            THEN("the runner refuses")
            {
                const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
                CHECK_THROWS_AS(runner->processFrame(frame, 0, 0.0), std::logic_error);
            }
        }
        WHEN("it is initialised with a mismatched frame size")
        {
            THEN("the initialisation is rejected")
            {
                CHECK_THROWS_AS(runner->initialise(scenario6Camera(), 30.0, cv::Size(320, 240), "x.MOV"), std::invalid_argument);
            }
        }
        WHEN("it is initialised correctly and given frames")
        {
            runner->initialise(scenario6Camera(), 30.0, cv::Size(640, 480), "x.MOV");
            cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
            cv::RNG rng(77);
            rng.fill(frame, cv::RNG::UNIFORM, 0, 255);
            cv::GaussianBlur(frame, frame, cv::Size(5, 5), 1.5);
            runner->processFrame(frame, 0, 0.0);
            runner->processFrame(frame, 1, 1.0 / 30.0);
            const PlotFrameData d = runner->plotData();
            THEN("the overlay is a BGR8 image of the input size and the status text is present")
            {
                CHECK(d.overlay.type() == CV_8UC3);
                CHECK(d.overlay.size() == frame.size());
                CHECK(!d.statusLines.empty());
            }
            THEN("the trajectory has one camera position per processed frame and the covariance is finite")
            {
                CHECK(d.trajectory.size() == 2);
                CHECK(d.rCNn.allFinite());
                CHECK(d.Rnc.allFinite());
                CHECK(d.positionSqrtCov.allFinite());
            }
            THEN("every landmark carries a finite position and an upper-triangular square-root covariance")
            {
                for (const PlotFrameData::Landmark& lm : d.landmarks)
                {
                    CHECK(lm.rLNn.allFinite());
                    CHECK(lm.positionSqrtCov.allFinite());
                    CHECK(lm.positionSqrtCov.triangularView<Eigen::StrictlyLower>().toDenseMatrix().isZero(0.0));
                }
            }
        }
        WHEN("a frame of the wrong type is processed")
        {
            runner->initialise(scenario6Camera(), 30.0, cv::Size(640, 480), "x.MOV");
            THEN("it is rejected")
            {
                const cv::Mat grey(480, 640, CV_8UC1, cv::Scalar(0));
                CHECK_THROWS_AS(runner->processFrame(grey, 0, 0.0), std::invalid_argument);
            }
        }
    }
}
