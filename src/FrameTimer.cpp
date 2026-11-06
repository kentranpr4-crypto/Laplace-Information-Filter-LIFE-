#include "FrameTimer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <print>
#include <stdexcept>
#include <format>

double FrameTimer::elapsedMs(Clock::time_point from, Clock::time_point to) noexcept
{
    return std::chrono::duration<double, std::milli>(to - from).count();
}

double FrameTimer::percentile(std::vector<double> values, double p)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double clamped = std::clamp(p, 0.0, 1.0);
    const double pos = clamped * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return values[lo] + frac * (values[hi] - values[lo]);
}

StageStatistics FrameTimer::statistics(std::vector<double> values)
{
    StageStatistics s;
    if (values.empty())
        return s;
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    s.max = *std::max_element(values.begin(), values.end());
    s.p50 = percentile(values, 0.50);
    s.p95 = percentile(std::move(values), 0.95);
    return s;
}

void FrameTimer::add(const FrameTimingSample & sample)
{
    samples_.push_back(sample);
}

std::string FrameTimer::csvHeader()
{
    return "frame,t,process_ms,plot_ms,write_ms,total_ms,read_ms,plot_data_ms,plot_update_ms,plot_render_ms,"
           "plot_capture_ms";
}

std::string FrameTimer::summary(double wallSeconds) const
{
    if (samples_.empty())
        return "Timing: no frames processed";

    auto collect = [this](auto member)
    {
        std::vector<double> v;
        v.reserve(samples_.size());
        for (const auto & s : samples_)
            v.push_back(s.*member);
        return v;
    };

    const StageStatistics read = statistics(collect(&FrameTimingSample::readMs));
    const StageStatistics proc = statistics(collect(&FrameTimingSample::processMs));
    const StageStatistics plot = statistics(collect(&FrameTimingSample::plotMs));
    const StageStatistics plotData = statistics(collect(&FrameTimingSample::plotDataMs));
    const StageStatistics plotUpdate = statistics(collect(&FrameTimingSample::plotUpdateMs));
    const StageStatistics plotRender = statistics(collect(&FrameTimingSample::plotRenderMs));
    const StageStatistics plotCapture = statistics(collect(&FrameTimingSample::plotCaptureMs));
    const StageStatistics write = statistics(collect(&FrameTimingSample::writeMs));
    const StageStatistics total = statistics(collect(&FrameTimingSample::totalMs));

    auto line = [](const char * name, const StageStatistics & s)
    {
        return std::format("  {:<12} mean {:9.3f}  p50 {:9.3f}  p95 {:9.3f}  max {:9.3f}  [ms/frame]\n",
                           name, s.mean, s.p50, s.p95, s.max);
    };

    std::string out = std::format("Timing summary over {} frames (wall {:.2f} s):\n", samples_.size(), wallSeconds);
    out += line("read", read);
    out += line("process", proc);
    out += line("plot", plot);
    out += line("  data", plotData);
    out += line("  update", plotUpdate);
    out += line("  render", plotRender);
    out += line("  capture", plotCapture);
    out += line("write", write);
    out += line("total", total);
    if (wallSeconds > 0.0)
        out += std::format("  throughput {:.2f} frames/s (loop, excluding interactive waits)\n",
                           static_cast<double>(samples_.size()) / wallSeconds);
    return out;
}

void FrameTimer::writeCsv(const std::filesystem::path & path) const
{
    std::ofstream os(path);
    if (!os)
        throw std::runtime_error(std::format("cannot open timing CSV for writing: {}", path.string()));

    os << csvHeader() << '\n';
    for (const auto & s : samples_)
    {
        os << std::format("{},{:.6f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n",
                          s.frame, s.t, s.processMs, s.plotMs, s.writeMs, s.totalMs, s.readMs, s.plotDataMs,
                          s.plotUpdateMs, s.plotRenderMs, s.plotCaptureMs);
    }
    os.flush();
    if (!os)
        throw std::runtime_error(std::format("failed while writing timing CSV: {}", path.string()));
}
