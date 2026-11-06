#ifndef VISUALNAVIGATION_H
#define VISUALNAVIGATION_H

#include <filesystem>
#include <memory>

class ScenarioRunner;

/**
 * @brief Runs visual navigation on an input video with the scenario selected by number.
 *
 * @param videoPath        input video (Scenario 4 reads the .SRT next to it)
 * @param cameraPath       camera.xml written by --calibrate (must match the video frame size)
 * @param scenario         4 (flight), 5 (identical ArUco tags) or 6 (room point landmarks)
 * @param interactive      0: no window interaction (offscreen); 1: block on the last frame until the window is
 *                         closed; 2: block on every frame until 'n'/space continues or the window is closed.
 *                         1 and 2 need a display.
 * @param outputDirectory  if non-empty, export the 2W x H visualisation to <dir>/<stem>_out<ext> and write
 *                         <dir>/<stem>_timing.csv
 * @throws std::invalid_argument for an unsupported scenario or invalid arguments,
 *         std::runtime_error for missing/invalid files, size mismatches, missing display, or a failing scenario
 *         (the message names the frame index). Threads of the buffered reader/writer are stopped before the
 *         exception leaves this function.
 */
void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario = 6, int interactive = 0, const std::filesystem::path & outputDirectory = "");

/**
 * @brief Same as runVisualNavigationFromVideo for an already constructed scenario runner.
 *
 * Used by runVisualNavigationFromVideo and by the tests (DummyScenarioRunner). The runner's initialise() is
 * called once after the calibration and the input video have been validated.
 */
void runVisualNavigation(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, std::unique_ptr<ScenarioRunner> runner, int interactive = 0, const std::filesystem::path & outputDirectory = "");

#endif
