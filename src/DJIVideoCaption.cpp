#include "DJIVideoCaption.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace
{
constexpr std::size_t kMaxStoredWarnings = 200;
constexpr double kTimeToleranceSeconds = 1e-3;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::string_view trimView(std::string_view s) noexcept
{
    const auto isSpace = [](char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '\f' || ch == '\v'; };
    while (!s.empty() && isSpace(s.front())) s.remove_prefix(1);
    while (!s.empty() && isSpace(s.back())) s.remove_suffix(1);
    return s;
}

std::optional<long long> parseInteger(std::string_view s) noexcept
{
    s = trimView(s);
    if (s.empty())
    {
        return std::nullopt;
    }
    if (s.front() == '+')
    {
        s.remove_prefix(1);
    }
    long long v = 0;
    const char * first = s.data();
    const char * last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, v);
    if (ec != std::errc() || ptr != last)
    {
        return std::nullopt;
    }
    return v;
}

std::optional<double> parseReal(std::string_view s) noexcept
{
    s = trimView(s);
    if (s.empty())
    {
        return std::nullopt;
    }
    if (s.front() == '+')
    {
        s.remove_prefix(1);
    }
    double v = 0.0;
    const char * first = s.data();
    const char * last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, v);
    if (ec != std::errc() || ptr != last || !std::isfinite(v))
    {
        return std::nullopt;
    }
    return v;
}

/** @brief Parses "HH:MM:SS,mmm" (comma or dot separator) into seconds. */
std::optional<double> parseClock(std::string_view s) noexcept
{
    s = trimView(s);
    // Expect H..H:MM:SS,fff
    const std::size_t c1 = s.find(':');
    if (c1 == std::string_view::npos) return std::nullopt;
    const std::size_t c2 = s.find(':', c1 + 1);
    if (c2 == std::string_view::npos) return std::nullopt;
    const std::size_t sep = s.find_first_of(",.", c2 + 1);
    if (sep == std::string_view::npos) return std::nullopt;
    const auto h = parseInteger(s.substr(0, c1));
    const auto m = parseInteger(s.substr(c1 + 1, c2 - c1 - 1));
    const auto sec = parseInteger(s.substr(c2 + 1, sep - c2 - 1));
    const std::string_view frac = s.substr(sep + 1);
    if (!h || !m || !sec || frac.empty() || frac.size() > 9)
    {
        return std::nullopt;
    }
    const auto fracValue = parseInteger(frac);
    if (!fracValue || *fracValue < 0)
    {
        return std::nullopt;
    }
    if (*m < 0 || *m > 59 || *sec < 0 || *sec > 60 || *h < 0)
    {
        return std::nullopt;
    }
    // Exact integer arithmetic in nanoseconds (the fraction has at most 9 digits), and a single division at the end.
    // Summing h*3600 + m*60 + s + f*10^-k in floating point gives e.g. 0.70000000000000006661 for "00:00:00,700",
    // whereas the frame time 21/30 is 0.69999999999999995559 (C3 in the review).
    std::int64_t fractionNs = *fracValue;
    for (std::size_t digits = frac.size(); digits < 9; ++digits)
    {
        fractionNs *= 10;
    }
    const std::int64_t wholeSeconds = *h * 3600 + *m * 60 + *sec;
    const std::int64_t totalNs = wholeSeconds * 1000000000 + fractionNs;
    return static_cast<double>(totalNs) / 1.0e9;
}

/** @brief Parses "HH:MM:SS,mmm --> HH:MM:SS,mmm". */
bool parseTimingLine(std::string_view line, double & start, double & end) noexcept
{
    const std::size_t arrow = line.find("-->");
    if (arrow == std::string_view::npos)
    {
        return false;
    }
    const auto a = parseClock(line.substr(0, arrow));
    const auto b = parseClock(line.substr(arrow + 3));
    if (!a || !b || *b < *a)
    {
        return false;
    }
    start = *a;
    end = *b;
    return true;
}

/** @brief Parses the leading number after "key :" (e.g. "DiffTime : 16ms" -> 16). */
std::optional<double> numberAfterKey(std::string_view text, std::string_view key) noexcept
{
    const std::size_t pos = text.find(key);
    if (pos == std::string_view::npos)
    {
        return std::nullopt;
    }
    std::string_view rest = text.substr(pos + key.size());
    const std::size_t colon = rest.find(':');
    if (colon == std::string_view::npos)
    {
        return std::nullopt;
    }
    rest = trimView(rest.substr(colon + 1));
    std::size_t n = 0;
    while (n < rest.size() && (std::isdigit(static_cast<unsigned char>(rest[n])) || rest[n] == '-' || rest[n] == '+' || rest[n] == '.'))
    {
        ++n;
    }
    return parseReal(rest.substr(0, n));
}

/** @brief Key/value pairs of the bracketed telemetry, e.g. "[iso : 110]" -> ("iso", "110"). Keys are lower-cased. */
std::vector<std::pair<std::string, std::string>> bracketFields(std::string_view text)
{
    std::vector<std::pair<std::string, std::string>> fields;
    std::size_t pos = 0;
    while (true)
    {
        const std::size_t open = text.find('[', pos);
        if (open == std::string_view::npos)
        {
            break;
        }
        const std::size_t close = text.find(']', open + 1);
        if (close == std::string_view::npos)
        {
            break;
        }
        const std::string_view inner = text.substr(open + 1, close - open - 1);
        pos = close + 1;
        const std::size_t colon = inner.find(':');
        if (colon == std::string_view::npos)
        {
            continue;
        }
        std::string key(trimView(inner.substr(0, colon)));
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        fields.emplace_back(std::move(key), std::string(trimView(inner.substr(colon + 1))));
    }
    return fields;
}

/** @brief First payload line that looks like "YYYY-MM-DD HH:MM:SS,mmm,uuu". */
std::string_view wallClockLine(const std::vector<std::string_view> & lines, std::size_t begin, std::size_t end)
{
    for (std::size_t i = begin; i < end; ++i)
    {
        const std::string_view line = trimView(lines[i]);
        if (line.size() >= 19 && std::isdigit(static_cast<unsigned char>(line[0])) && line[4] == '-' && line[7] == '-')
        {
            return line;
        }
    }
    return {};
}

/** @brief Splits text into lines; accepts LF, CRLF and lone CR. Text must outlive the returned views. */
std::vector<std::string_view> splitLines(std::string_view text)
{
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    for (std::size_t i = 0; i < text.size(); ++i)
    {
        if (text[i] == '\n')
        {
            std::size_t end = i;
            if (end > start && text[end - 1] == '\r')
            {
                --end;
            }
            lines.push_back(text.substr(start, end - start));
            start = i + 1;
        }
        else if (text[i] == '\r' && (i + 1 >= text.size() || text[i + 1] != '\n'))
        {
            lines.push_back(text.substr(start, i - start));
            start = i + 1;
        }
    }
    if (start < text.size())
    {
        lines.push_back(text.substr(start));
    }
    return lines;
}
} // namespace

DJIVideoCaption DJIVideoCaption::load(const std::filesystem::path & srtPath, ParseMode mode, ParseReport * report)
{
    std::ifstream in(srtPath, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(std::format("cannot open subtitle file {}", srtPath.string()));
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(text, mode, report, srtPath.string());
}

DJIVideoCaption DJIVideoCaption::parse(std::string_view text, ParseMode mode, ParseReport * report, const std::string & sourceName)
{
    ParseReport localReport;
    ParseReport & rep = report ? *report : localReport;
    rep = ParseReport{};

    auto problem = [&](std::size_t lineNumber, const std::string & message, bool structural)
    {
        const std::string full = std::format("{}:{}: {}", sourceName, lineNumber, message);
        if (mode == ParseMode::Strict)
        {
            throw std::runtime_error(full);
        }
        if (rep.warnings.size() < kMaxStoredWarnings)
        {
            rep.warnings.push_back(full);
        }
        if (structural)
        {
            ++rep.blocksSkipped;
        }
        else
        {
            ++rep.fieldProblems;
        }
    };

    // Skip a UTF-8 byte-order mark.
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF && static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF)
    {
        text.remove_prefix(3);
    }
    const std::vector<std::string_view> lines = splitLines(text);

    // Lines containing "-->" delimit blocks, even if their times are malformed (so that a broken block cannot
    // leak its fields into the previous block). The block index is the line directly above such a line.
    std::vector<std::size_t> timingLines;
    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        if (lines[i].find("-->") != std::string_view::npos)
        {
            timingLines.push_back(i);
        }
    }
    DJIVideoCaption caption;
    if (timingLines.empty())
    {
        problem(1, "no SRT timing lines (HH:MM:SS,mmm --> HH:MM:SS,mmm) found", true);
        return caption;
    }

    for (std::size_t b = 0; b < timingLines.size(); ++b)
    {
        const std::size_t t = timingLines[b];
        // Index line directly above the timing line.
        std::optional<long long> index;
        std::size_t indexLine = t; // 0-based
        if (t >= 1)
        {
            index = parseInteger(lines[t - 1]);
            if (index)
            {
                indexLine = t - 1;
            }
        }
        // Payload runs until the next block's index line (or timing line).
        std::size_t payloadEnd = lines.size();
        if (b + 1 < timingLines.size())
        {
            const std::size_t nextT = timingLines[b + 1];
            payloadEnd = (nextT >= 1 && parseInteger(lines[nextT - 1])) ? nextT - 1 : nextT;
        }
        const std::size_t lineNo = indexLine + 1;

        if (!index)
        {
            problem(t + 1, "timing line without a numeric block index above it; block skipped", true);
            continue;
        }
        double startSeconds = 0.0;
        double endSeconds = 0.0;
        if (!parseTimingLine(lines[t], startSeconds, endSeconds))
        {
            problem(t + 1, "malformed timing line (expected HH:MM:SS,mmm --> HH:MM:SS,mmm); block skipped", true);
            continue;
        }

        // Gather the payload into one string for field scanning.
        std::string payload;
        for (std::size_t i = t + 1; i < payloadEnd; ++i)
        {
            payload += lines[i];
            payload += '\n';
        }

        DJIVideoCaptionRecord rec;
        rec.blockIndex = static_cast<int>(*index);
        rec.startSeconds = startSeconds;
        rec.endSeconds = endSeconds;

        // FrameCnt (required) and DiffTime (optional).
        const std::size_t fc = payload.find("FrameCnt");
        std::optional<long long> frameCnt;
        if (fc != std::string::npos)
        {
            std::string_view rest(payload);
            rest = rest.substr(fc + std::string_view("FrameCnt").size());
            const std::size_t colon = rest.find(':');
            if (colon != std::string_view::npos)
            {
                std::string_view digits = trimView(rest.substr(colon + 1));
                std::size_t n = 0;
                while (n < digits.size() && std::isdigit(static_cast<unsigned char>(digits[n])))
                {
                    ++n;
                }
                frameCnt = parseInteger(digits.substr(0, n));
            }
        }
        if (!frameCnt || *frameCnt < 0 || *frameCnt > std::numeric_limits<int>::max())
        {
            problem(lineNo, "missing or malformed FrameCnt; block skipped", true);
            continue;
        }
        rec.frameCount = static_cast<int>(*frameCnt);

        const std::size_t dt = payload.find("DiffTime");
        if (dt != std::string::npos)
        {
            const auto v = numberAfterKey(std::string_view(payload).substr(dt), "DiffTime");
            if (v)
            {
                rec.diffTimeMs = *v;
            }
            else
            {
                rec.diffTimeMs = kNaN;
                problem(lineNo, "malformed DiffTime (stored as NaN)", false);
            }
        }
        else
        {
            rec.diffTimeMs = kNaN;
        }

        rec.wallClock = std::string(wallClockLine(lines, t + 1, payloadEnd));

        // Bracketed telemetry fields.
        const auto fields = bracketFields(payload);
        std::optional<std::string> latText;
        std::optional<std::string> lonText;
        bool hasAltitude = false;
        for (const auto & [key, value] : fields)
        {
            if (key == "iso")
            {
                if (auto v = parseInteger(value))
                {
                    rec.iso = static_cast<int>(*v);
                }
                else
                {
                    problem(lineNo, std::format("malformed iso '{}'", value), false);
                }
            }
            else if (key == "shutter")
            {
                rec.shutterText = value;
                std::optional<double> seconds;
                const std::size_t slash = value.find('/');
                if (slash != std::string::npos)
                {
                    const auto num = parseReal(std::string_view(value).substr(0, slash));
                    const auto den = parseReal(std::string_view(value).substr(slash + 1));
                    if (num && den && *den > 0.0)
                    {
                        seconds = *num / *den;
                    }
                }
                else
                {
                    seconds = parseReal(value);
                }
                if (seconds)
                {
                    rec.shutterSeconds = seconds;
                }
                else
                {
                    problem(lineNo, std::format("malformed shutter '{}'", value), false);
                }
            }
            else if (key == "fnum")
            {
                if (auto v = parseReal(value))
                {
                    rec.fNumberRaw = v;
                }
                else
                {
                    problem(lineNo, std::format("malformed fnum '{}'", value), false);
                }
            }
            else if (key == "ev")
            {
                if (auto v = parseReal(value))
                {
                    rec.evCompensation = v;
                }
                else
                {
                    problem(lineNo, std::format("malformed ev '{}'", value), false);
                }
            }
            else if (key == "ct")
            {
                if (auto v = parseReal(value))
                {
                    rec.colourTemperatureK = static_cast<int>(std::lround(*v));
                }
                else
                {
                    problem(lineNo, std::format("malformed ct '{}'", value), false);
                }
            }
            else if (key == "color_md")
            {
                rec.colourMode = value;
            }
            else if (key == "focal_len")
            {
                if (auto v = parseReal(value))
                {
                    rec.focalLengthRaw = v;
                }
                else
                {
                    problem(lineNo, std::format("malformed focal_len '{}'", value), false);
                }
            }
            else if (key == "altitude")
            {
                hasAltitude = true;
                if (auto v = parseReal(value))
                {
                    rec.altitudeM = v;
                }
                else
                {
                    problem(lineNo, std::format("malformed altitude '{}'", value), false);
                }
            }
            else if (key == "latitude")
            {
                latText = value;
            }
            else if (key == "longtitude" || key == "longitude")
            {
                lonText = value;
            }
        }
        if (!hasAltitude)
        {
            ++rep.missingAltitude;
            if (mode == ParseMode::Strict)
            {
                problem(lineNo, "altitude field missing", false);
            }
        }

        // Validation-only geodetic values, kept apart from the estimator-facing record.
        DJIValidationOnlyGeodeticSample geo;
        geo.blockIndex = rec.blockIndex;
        if (latText)
        {
            const auto v = parseReal(*latText);
            if (v && std::abs(*v) <= 90.0)
            {
                geo.latitudeDeg = v;
            }
            else
            {
                problem(lineNo, std::format("malformed latitude '{}' (validation field ignored)", *latText), false);
            }
        }
        if (lonText)
        {
            const auto v = parseReal(*lonText);
            if (v && std::abs(*v) <= 180.0)
            {
                geo.longitudeDeg = v;
            }
            else
            {
                problem(lineNo, std::format("malformed longitude '{}' (validation field ignored)", *lonText), false);
            }
        }

        caption.records_.push_back(std::move(rec));
        caption.validationGeodetic_.push_back(geo);
        ++rep.blocksParsed;
    }

    // Enforce time order (stable), report if the file was not sorted.
    bool sorted = std::is_sorted(caption.records_.begin(), caption.records_.end(),
                                 [](const DJIVideoCaptionRecord & a, const DJIVideoCaptionRecord & b) { return a.startSeconds < b.startSeconds; });
    if (!sorted)
    {
        std::vector<std::size_t> order(caption.records_.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return caption.records_[a].startSeconds < caption.records_[b].startSeconds; });
        std::vector<DJIVideoCaptionRecord> r2;
        std::vector<DJIValidationOnlyGeodeticSample> g2;
        for (std::size_t i : order)
        {
            r2.push_back(caption.records_[i]);
            g2.push_back(caption.validationGeodetic_[i]);
        }
        caption.records_ = std::move(r2);
        caption.validationGeodetic_ = std::move(g2);
        problem(1, "block start times were not monotonic; records re-sorted by start time", false);
    }
    return caption;
}

std::size_t DJIVideoCaption::indexAtTime(double seconds) const noexcept
{
    constexpr std::size_t npos = static_cast<std::size_t>(-1);
    if (records_.empty() || !std::isfinite(seconds))
    {
        return npos;
    }
    const auto it = std::upper_bound(records_.begin(), records_.end(), seconds + kTimeToleranceSeconds,
                                     [](double value, const DJIVideoCaptionRecord & r) { return value < r.startSeconds; });
    if (it == records_.begin())
    {
        return npos;
    }
    const std::size_t idx = static_cast<std::size_t>(std::distance(records_.begin(), it)) - 1;
    // A block covers [start - tol, end - tol): the end boundary is shared with the next block's start.
    if (seconds + kTimeToleranceSeconds >= records_[idx].endSeconds)
    {
        return npos;
    }
    return idx;
}

const DJIVideoCaptionRecord * DJIVideoCaption::recordAtTime(double seconds) const noexcept
{
    const std::size_t idx = indexAtTime(seconds);
    return idx == static_cast<std::size_t>(-1) ? nullptr : &records_[idx];
}

const DJIVideoCaptionRecord * DJIVideoCaption::recordForFrame(std::size_t frameIndex, double fps) const noexcept
{
    if (!(fps > 0.0))
    {
        return nullptr;
    }
    return recordAtTime(static_cast<double>(frameIndex) / fps);
}

const DJIVideoCaptionRecord * DJIVideoCaption::recordForFrameCount(int frameCount) const noexcept
{
    for (const auto & r : records_)
    {
        if (r.frameCount == frameCount)
        {
            return &r;
        }
    }
    return nullptr;
}

std::optional<double> DJIVideoCaption::altitudeAtFrame(std::size_t frameIndex, double fps) const noexcept
{
    const DJIVideoCaptionRecord * r = recordForFrame(frameIndex, fps);
    return r ? r->altitudeM : std::nullopt;
}

std::optional<double> DJIVideoCaption::altitudeAtTime(double seconds) const noexcept
{
    const DJIVideoCaptionRecord * r = recordAtTime(seconds);
    return r ? r->altitudeM : std::nullopt;
}

std::vector<AltitudeSample> DJIVideoCaption::altitudeUpdates() const
{
    std::vector<AltitudeSample> out;
    std::optional<double> previous;
    for (const auto & r : records_)
    {
        if (!r.altitudeM)
        {
            continue;
        }
        if (!previous || *previous != *r.altitudeM)
        {
            out.push_back(AltitudeSample{r.blockIndex, r.startSeconds, *r.altitudeM});
        }
        previous = r.altitudeM;
    }
    return out;
}

std::optional<DJIValidationOnlyGeodeticSample> DJIVideoCaption::validationOnlyGeodeticForFrame(std::size_t frameIndex, double fps) const
{
    if (!(fps > 0.0))
    {
        return std::nullopt;
    }
    const std::size_t idx = indexAtTime(static_cast<double>(frameIndex) / fps);
    if (idx == static_cast<std::size_t>(-1))
    {
        return std::nullopt;
    }
    return validationGeodetic_[idx];
}
