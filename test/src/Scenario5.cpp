// Scenario 5 runner contract: initialise / processFrame / plotData on a short synthetic sequence.
#include <memory>
#include <stdexcept>
#include <string>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "../../src/Camera.h"
#include "../../src/ScenarioRunner.h"
#include "testUtil.h"

namespace
{

Camera runnerCamera()
{
    return Camera(640, 480, (Eigen::Matrix3d() << 500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0).finished(), {});
}

} // namespace

std::unique_ptr<ScenarioRunner> makeScenario5Runner();

SCENARIO("Scenario 5 runner: the ScenarioRunner contract")
{
    GIVEN("the Scenario 5 runner initialised with a 640 x 480 camera at 29.97 fps")
    {
        const Camera cam = runnerCamera();
        std::unique_ptr<ScenarioRunner> runner = makeScenario5Runner();
        REQUIRE(runner != nullptr);
        runner->initialise(cam, 29.97, cv::Size(640, 480), "synthetic.MOV");
        THEN("its name identifies the scenario")
        {
            CHECK(runner->name().find("Scenario 5") != std::string::npos);
        }
        WHEN("a blank frame is processed")
        {
            const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(120, 120, 120));
            runner->processFrame(frame, 0, 0.0);
            const PlotFrameData d = runner->plotData();
            THEN("the overlay is a BGR8 image of the input size and there are no landmarks yet")
            {
                CHECK(d.overlay.size() == cv::Size(640, 480));
                CHECK(d.overlay.type() == CV_8UC3);
                CHECK(d.landmarks.empty());
                CHECK(d.trajectory.size() == 1);
            }
            THEN("the status lines and the position covariance are populated")
            {
                CHECK(!d.statusLines.empty());
                CHECK(d.positionSqrtCov.allFinite());
                CHECK(d.Rnc.allFinite());
            }
        }
        WHEN("a frame of the wrong size is processed")
        {
            const cv::Mat wrong(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
            THEN("the runner rejects it")
            {
                CHECK_THROWS_AS(runner->processFrame(wrong, 1, 0.03), std::invalid_argument);
            }
        }
    }
    GIVEN("a runner that has not been initialised")
    {
        std::unique_ptr<ScenarioRunner> runner = makeScenario5Runner();
        THEN("processing throws a logic error and plotData is empty")
        {
            const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
            CHECK_THROWS_AS(runner->processFrame(frame, 0, 0.0), std::logic_error);
            CHECK(runner->plotData().landmarks.empty());
        }
    }
    GIVEN("an invalid camera")
    {
        std::unique_ptr<ScenarioRunner> runner = makeScenario5Runner();
        THEN("initialisation is rejected")
        {
            CHECK_THROWS_AS(runner->initialise(Camera(), 30.0, cv::Size(640, 480), ""), std::invalid_argument);
        }
    }
}
