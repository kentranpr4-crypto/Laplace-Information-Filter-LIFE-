#ifndef PLOT_H
#define PLOT_H

/**
 * @file Plot.h
 * @brief Dual-pane VTK 9.5 visualisation of a visual-navigation run.
 *
 * One vtkRenderWindow of size 2W x H (W x H = input frame size, the export frame size is identical):
 *
 *   +----------------------+----------------------+
 *   | LEFT  (W x H)        | RIGHT (W x H)        |
 *   | scenario overlay     | 3-D scene            |
 *   | (image, no scaling)  | (see Plot.cpp docs)  |
 *   +----------------------+----------------------+
 *
 * Coordinate convention of the 3-D pane: the estimator world frame n is NED-like (z down). The scene
 * is displayed with the fixed proper rotation M (n -> VTK), M = [[1,0,0],[0,0,-1],[0,1,0]], so that
 * VTK +X = north (n_x), VTK +Y = up (-n_z), VTK +Z = east (n_y). The n-frame axes are drawn with the
 * same M so that their labels N / E / D are correct.
 *
 * Backends: Offscreen (interactive = 0) always uses headless EGL (vtkEGLRenderWindow) even when DISPLAY is set,
 * because the X11 path on this class of machine renders through a different GPU stack and reads back far
 * slower. Onscreen (interactive = 1, 2) uses the default VTK window (X11 on DISPLAY).
 *
 * All colours, render order and camera behaviour are documented in Plot.cpp and in
 * engineering-reports/architecture.md. They are ASSUMED (no visual specification was supplied).
 */

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include "ScenarioRunner.h"

class Camera;

class Plot
{
public:
    /// Offscreen: render to an off-screen buffer (EGL, no window, independent of DISPLAY).
    /// Onscreen: show an interactive window (requires a display).
    enum class WindowMode { Offscreen, Onscreen };

    /// Wall-clock cost of the last update / render / capture calls [ms].
    struct Timings
    {
        double updateMs = 0.0;
        double renderMs = 0.0;
        double captureMs = 0.0;
    };

    /// Confidence level of every drawn uncertainty region (AGENTS.md: nSigma = 3.0).
    static constexpr double kNSigma = 3.0;

    /**
     * @param camera     calibration of the recording camera (frustum drawing, pixel -> ray)
     * @param inputSize  input frame size W x H (must equal the camera image size)
     * @param mode       offscreen or onscreen window
     * @throws std::invalid_argument for invalid sizes, std::runtime_error if VTK cannot create the window
     *         or if Onscreen is requested without a display.
     */
    Plot(const Camera & camera, cv::Size inputSize, WindowMode mode);
    ~Plot();

    Plot(const Plot &) = delete;
    Plot & operator=(const Plot &) = delete;
    Plot(Plot &&) = delete;
    Plot & operator=(Plot &&) = delete;

    /// Input frame size W x H.
    [[nodiscard]] cv::Size inputSize() const noexcept;
    /// Output (window and export) size 2W x H.
    [[nodiscard]] cv::Size frameSize() const noexcept;
    [[nodiscard]] bool isOnscreen() const noexcept;

    /// VTK class name of the window that was actually created, e.g. "vtkEGLRenderWindow" or "vtkXOpenGLRenderWindow".
    [[nodiscard]] const std::string & backendName() const noexcept;
    /// Class name plus the OpenGL vendor and renderer strings of the live context (queried once, at construction).
    [[nodiscard]] const std::string & backendDescription() const noexcept;

    /// Updates all actors from the scenario data. Reuses actors and geometry buffers across calls (no per-frame
    /// actor creation). Uses data.overlay for the left pane.
    /// @throws std::invalid_argument if data.overlay is non-empty and is not BGR8 / GRAY8 at the input size.
    void update(const PlotFrameData & data);

    /// Renders the window (offscreen or onscreen) with the current actor state.
    void render();

    /// Renders (if needed) and reads back the window as BGR8 of exactly 2W x H (row 0 = top).
    /// The destination Mat is reused when it already has the right size and type.
    void capture(cv::Mat & bgr);

    [[nodiscard]] const Timings & lastTimings() const noexcept;

    /// Interactive mode 1: shows the window and blocks until it is closed (or 'q' / Escape is pressed).
    /// @throws std::logic_error if the plot was created offscreen.
    void blockUntilClosed();

    /// Interactive mode 2: shows the current frame and blocks until 'n', 'space' or 'c' continues, or the
    /// window is closed / 'q' / Escape is pressed.
    /// @return false if the user closed / quit (caller should stop the run), true to continue.
    /// @throws std::logic_error if the plot was created offscreen.
    bool showAndWait();

    /// True once the user has closed the window or pressed 'q' / Escape.
    [[nodiscard]] bool quitRequested() const noexcept;

    /// True if a graphical display is available for Onscreen mode (DISPLAY or WAYLAND_DISPLAY set on Linux).
    [[nodiscard]] static bool displayAvailable();

    /// VTK window class requested for a mode: Offscreen -> "vtkEGLRenderWindow" (headless, ignores DISPLAY);
    /// Onscreen -> "" (VTK default: X11 window on DISPLAY). This is the selection logic used by the constructor.
    [[nodiscard]] static std::string requestedBackendClass(WindowMode mode);

    /// Value of the "label: value" field of a vtkRenderWindow::ReportCapabilities() report (trimmed), or "" if absent.
    [[nodiscard]] static std::string parseReportField(const std::string & report, const std::string & label);

    // ----- Geometry helpers (pure functions, unit-tested) -----

    /**
     * @brief Linear map A such that the n-sigma confidence ellipsoid of N(mu, P) with P = S^T S, S upper triangular, is the
     *        image of the unit sphere: surface = mu + A u, |u| = 1, A = sqrt(c) S^T, c = GaussianBase::mahalanobisThresholdSq(nSigma, 3).
     */
    [[nodiscard]] static Eigen::Matrix3d ellipsoidLinearMap(const Eigen::Matrix3d & S, double nSigma = kNSigma);

    /// nLat x nLon surface points of the n-sigma ellipsoid (for tests / diagnostics).
    [[nodiscard]] static std::vector<Eigen::Vector3d> ellipsoidSurfacePoints(const Eigen::Vector3d & mu,
                                                                            const Eigen::Matrix3d & S,
                                                                            double nSigma, int nLat, int nLon);

    /// Fixed proper rotation from estimator frame n (z down) to VTK display coordinates (y up).
    [[nodiscard]] static Eigen::Matrix3d worldToDisplay();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
