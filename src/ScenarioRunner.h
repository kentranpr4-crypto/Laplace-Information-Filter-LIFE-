#ifndef SCENARIORUNNER_H
#define SCENARIORUNNER_H

/**
 * @file ScenarioRunner.h
 * @brief Contract between the frame loop (visualNavigation.cpp), the scenario
 *        estimators (Scenario 4/5/6) and the VTK visualisation (Plot).
 *
 * Ownership: Lead. Scenario Peers implement ScenarioRunner; the visualisation
 * Peer consumes PlotFrameData. Changes to this header go through Lead.
 *
 * Conventions (see engineering-reports/architecture.md):
 *  - World frame n: z down (outdoor NED-like, indoor fixed at start).
 *  - Body b: x forward, y right, z down. Camera c: x right, y down, z optical axis.
 *  - All uncertainty is passed as an UPPER-triangular square-root covariance S
 *    with P = S^T S (obtained from GaussianInfo marginals; never inverted here).
 *  - Confidence regions are drawn with nSigma = 3.0 (GaussianBase conventions).
 */

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

class Camera;

/// Per-frame data handed from a scenario to the visualisation.
struct PlotFrameData
{
    /// Left pane: annotated image in BGR8 at the INPUT frame size (the
    /// scenario draws its overlays in the scenario-defined order and colours).
    cv::Mat overlay;

    /// Current camera pose in world: position rCNn and rotation Rnc.
    Eigen::Vector3d rCNn = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rnc = Eigen::Matrix3d::Identity();
    /// Upper-triangular sqrt covariance of the body/camera position (3x3).
    Eigen::Matrix3d positionSqrtCov = Eigen::Matrix3d::Zero();

    /// Estimated trajectory so far (camera positions in world), oldest first.
    std::vector<Eigen::Vector3d> trajectory;

    /// Optional validation-only reference track (e.g. GPS from flight.SRT
    /// converted to local metres). NEVER used by any estimator.
    std::vector<Eigen::Vector3d> validationTrack;

    struct Landmark
    {
        int id = -1;
        Eigen::Vector3d rLNn = Eigen::Vector3d::Zero();               ///< position mean
        Eigen::Matrix3d positionSqrtCov = Eigen::Matrix3d::Zero();    ///< upper-triangular, P = S^T S
        bool isPose = false;                                          ///< pose landmark (ArUco tag)
        Eigen::Matrix3d Rnl = Eigen::Matrix3d::Identity();            ///< orientation (pose landmarks only)
        double size = 0.0;                                            ///< tag side length [m] (pose landmarks only)
        Eigen::Vector3d colourRGB = Eigen::Vector3d(0.0, 1.0, 0.0);   ///< in [0,1]
        bool visible = false;                                         ///< observed in this frame
    };
    std::vector<Landmark> landmarks;

    /// Short text lines for an on-screen status overlay (frame, time, NIS, ...).
    std::vector<std::string> statusLines;
};

/// One scenario (4, 5 or 6). Called by the frame loop in this order:
///   initialise() once, then for every frame: processFrame(), then plotData().
class ScenarioRunner
{
public:
    virtual ~ScenarioRunner() = default;

    /// @param camera     loaded calibration of the camera that recorded the video
    /// @param fps        video frame rate
    /// @param frameSize  input frame size (must match camera image size; validated by caller)
    /// @param videoPath  input video path (Scenario 4 derives the .SRT from it)
    virtual void initialise(const Camera & camera, double fps, cv::Size frameSize,
                            const std::string & videoPath) = 0;

    /// Process one input frame (BGR8). t = frameIndex / fps (video time, seconds).
    virtual void processFrame(const cv::Mat & frame, int frameIndex, double t) = 0;

    /// Visualisation data for the most recent processed frame.
    virtual PlotFrameData plotData() const = 0;

    /// Human-readable name ("Scenario 4: flight", ...).
    virtual std::string name() const = 0;
};

/// Factory: implemented in each scenario's translation unit. Returns nullptr
/// for unsupported scenario numbers.
std::unique_ptr<ScenarioRunner> makeScenario4Runner();
std::unique_ptr<ScenarioRunner> makeScenario5Runner();
std::unique_ptr<ScenarioRunner> makeScenario6Runner();

#endif
