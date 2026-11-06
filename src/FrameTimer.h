#ifndef FRAMETIMER_H
#define FRAMETIMER_H

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

/**
 * @brief Per-frame stage timing for the visual-navigation frame loop.
 *
 * All durations are wall-clock milliseconds measured with std::chrono::steady_clock.
 * Stage definitions (see visualNavigation.cpp):
 *  - readMs:    time the frame loop waited for the BufferedVideoReader (includes decode time when the
 *               reader thread is behind).
 *  - processMs: ScenarioRunner::processFrame().
 *  - plotMs:    the whole visualisation stage = plotDataMs + plotUpdateMs + plotRenderMs + plotCaptureMs
 *               (up to the small loop overhead of the timer calls).
 *  - plotDataMs:    ScenarioRunner::plotData() (copies the scenario's visual data).
 *  - plotUpdateMs:  Plot::update() (actor geometry and overlay upload).
 *  - plotRenderMs:  Plot::render() (GPU draw on the window).
 *  - plotCaptureMs: Plot::capture() (readback, colour conversion, flip); 0 when not exporting.
 *  - writeMs:   enqueueing the captured frame into the BufferedVideoWriter (may block when the
 *               writer queue is full; the encode itself runs on the writer thread).
 *  - totalMs:   the whole loop iteration, i.e. readMs + processMs + plotMs + writeMs
 *               (excludes interactive blocking, which is a human wait, not compute).
 */
struct FrameTimingSample
{
    int frame = 0;          ///< 0-based input frame index
    double t = 0.0;         ///< video time [s] = frame / fps
    double readMs = 0.0;
    double processMs = 0.0;
    double plotMs = 0.0;
    double writeMs = 0.0;
    double totalMs = 0.0;
    double plotDataMs = 0.0;
    double plotUpdateMs = 0.0;
    double plotRenderMs = 0.0;
    double plotCaptureMs = 0.0;
};

/// Summary statistics of one stage [ms].
struct StageStatistics
{
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

/**
 * @brief Collects FrameTimingSample records, prints a summary and writes the timing CSV.
 *
 * CSV schema (header is exact, one row per processed frame). The first six columns keep their original positions;
 * the remaining columns were appended:
 * @code
 *   frame,t,process_ms,plot_ms,write_ms,total_ms,read_ms,plot_data_ms,plot_update_ms,plot_render_ms,plot_capture_ms
 * @endcode
 */
class FrameTimer
{
public:
    using Clock = std::chrono::steady_clock;

    /// Milliseconds between two steady_clock time points.
    [[nodiscard]] static double elapsedMs(Clock::time_point from, Clock::time_point to) noexcept;

    /// Linear-interpolated percentile (p in [0, 1]) of a sample set; 0 for an empty set.
    [[nodiscard]] static double percentile(std::vector<double> values, double p);

    /// Mean / median / 95th percentile / max of a sample set.
    [[nodiscard]] static StageStatistics statistics(std::vector<double> values);

    void add(const FrameTimingSample & sample);
    [[nodiscard]] const std::vector<FrameTimingSample> & samples() const noexcept { return samples_; }
    [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }

    /// Human-readable multi-line summary (per stage mean / p50 / p95 / max and throughput).
    [[nodiscard]] std::string summary(double wallSeconds) const;

    /// Writes the CSV described above. Overwrites the file. @throws std::runtime_error on I/O failure.
    void writeCsv(const std::filesystem::path & path) const;

    /// The exact CSV header line (no trailing newline).
    [[nodiscard]] static std::string csvHeader();

private:
    std::vector<FrameTimingSample> samples_;
};

#endif
