#ifndef DJIVIDEOCAPTION_H
#define DJIVIDEOCAPTION_H

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief One block of a DJI subtitle (.SRT) file, as written next to a DJI video.
 *
 * Example block (flight.SRT, CRLF or LF line endings, optional <font> tags):
 * @code
 *   1
 *   00:00:00,000 --> 00:00:00,016
 *   <font size="36">FrameCnt : 1, DiffTime : 16ms
 *   2021-09-29 11:13:38,121,919
 *   [iso : 110] [shutter : 1/400.0] [fnum : 280] [ev : 0] [ct : 6000] [color_md : default] [focal_len : 280] [latitude : -32.868837] [longtitude : 151.684716] [altitude: 63.126999] </font>
 * @endcode
 *
 * This record is the ESTIMATOR-FACING view of a block. It deliberately contains no latitude or longitude:
 * geodetic values are only available through DJIVideoCaption::validationOnlyGeodeticForFrame().
 */
struct DJIVideoCaptionRecord
{
    int blockIndex = 0;                      ///< Number on the first line of the block (1-based)
    double startSeconds = 0.0;               ///< Block start on the video timeline [s]
    double endSeconds = 0.0;                 ///< Block end on the video timeline [s]
    int frameCount = 0;                      ///< "FrameCnt": 1-based frame counter written by DJI
    double diffTimeMs = 0.0;                 ///< "DiffTime": time since the previous frame [ms]
    std::string wallClock;                   ///< Wall-clock text as written by the camera (local time; not used for estimation)
    std::optional<int> iso;                  ///< "iso"
    std::string shutterText;                 ///< "shutter" as written, e.g. "1/400.0"
    std::optional<double> shutterSeconds;    ///< Exposure time [s] parsed from shutterText
    std::optional<double> fNumberRaw;        ///< "fnum" in raw DJI units (e.g. 280); units not documented
    std::optional<double> evCompensation;    ///< "ev"
    std::optional<int> colourTemperatureK;   ///< "ct" [K]
    std::string colourMode;                  ///< "color_md"
    std::optional<double> focalLengthRaw;    ///< "focal_len" in raw DJI units (e.g. 280); units not documented
    std::optional<double> altitudeM;         ///< "altitude" [m]; the ONLY telemetry value intended for estimation
};

/** @brief Altitude value change event (for altimeter measurement scheduling). */
struct AltitudeSample
{
    int blockIndex = 0;      ///< SRT block where the new value first appears
    double timeSeconds = 0.0; ///< Start time of that block [s]
    double altitudeM = 0.0;  ///< New altitude value [m]
};

/**
 * @brief Geodetic values from the SRT, VALIDATION AND VISUALISATION ONLY.
 *
 * USER_REQUIREMENT: latitude/longitude must never be used as filter measurements. This type is kept
 * separate from DJIVideoCaptionRecord so that estimator code cannot reach these values by accident;
 * the only accessor is DJIVideoCaption::validationOnlyGeodeticForFrame / validationOnlyGeodeticSamples.
 */
struct DJIValidationOnlyGeodeticSample
{
    int blockIndex = 0;                      ///< Block the values belong to
    std::optional<double> latitudeDeg;       ///< "latitude" [deg]; validation/visualisation only
    std::optional<double> longitudeDeg;      ///< "longtitude" (sic, as written by DJI) [deg]; validation/visualisation only
};

/**
 * @brief Parsed DJI SRT telemetry log with time/frame lookup.
 *
 * Parsing policy (documented behaviour):
 *  - A block starts at each line containing "-->" (its numeric index is the line directly above it); blank lines
 *    are not required between blocks. A "-->" line with malformed times skips its block (reported, never merged
 *    into the previous block). CRLF, CR and LF line endings and a UTF-8 BOM are accepted.
 *  - Lenient mode (default): a block whose index, timing or FrameCnt cannot be parsed is skipped and reported;
 *    an optional field that is present but malformed is stored as absent and reported. Each problem is
 *    reported as "<source>:<line>: message" in ParseReport::warnings (capped at 200 messages; the counters
 *    are always complete).
 *  - Strict mode: the first structural or malformed-field problem throws std::runtime_error with the line number.
 *  - Records are returned in time order. Non-monotonic input is sorted by start time and reported.
 *
 * Frame alignment (DESIGN_ASSUMPTION, verified on flight.SRT: block start times equal (FrameCnt - 1) / fps to within
 * 1 ms): video frame i (0-based, presentation time i / fps) corresponds to the SRT block whose FrameCnt is i + 1.
 * A block covers [start - 1 ms, end - 1 ms); the 1 ms tolerance absorbs the millisecond rounding of SRT timestamps.
 * The 9161st frame (index 9160) has no block: flight.SRT holds 9160 blocks.
 */
class DJIVideoCaption
{
public:
    /** @brief Parsing strictness. */
    enum class ParseMode
    {
        Lenient, ///< Skip/ignore problematic blocks or fields and report them
        Strict   ///< Throw on the first problem
    };

    /** @brief Counters and messages collected while parsing. */
    struct ParseReport
    {
        std::size_t blocksParsed = 0;       ///< Blocks turned into records
        std::size_t blocksSkipped = 0;      ///< Blocks dropped because a required element was malformed
        std::size_t fieldProblems = 0;      ///< Optional fields present but malformed (stored as absent)
        std::size_t missingAltitude = 0;    ///< Blocks without an altitude field
        std::vector<std::string> warnings;  ///< First 200 messages, "<source>:<line>: ..."
    };

    /** @brief Creates an empty log. */
    DJIVideoCaption() = default;

    /**
     * @brief Reads and parses an SRT file.
     * @throws std::runtime_error if the file cannot be read, or in Strict mode on any problem.
     */
    static DJIVideoCaption load(const std::filesystem::path & srtPath, ParseMode mode = ParseMode::Lenient, ParseReport * report = nullptr);

    /**
     * @brief Parses SRT text already in memory.
     * @param text SRT content.
     * @param sourceName Name used in diagnostics.
     * @throws std::runtime_error in Strict mode on any problem.
     */
    static DJIVideoCaption parse(std::string_view text, ParseMode mode = ParseMode::Lenient, ParseReport * report = nullptr,
                                 const std::string & sourceName = "<memory>");

    /** @brief All records in time order (estimator-facing; no geodetic values). */
    [[nodiscard]] std::span<const DJIVideoCaptionRecord> records() const noexcept { return records_; }
    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }

    /** @brief Record whose span contains the time t [s]: the last block with start <= t + 1 ms, provided t + 1 ms < end. nullptr if none. */
    [[nodiscard]] const DJIVideoCaptionRecord * recordAtTime(double seconds) const noexcept;

    /**
     * @brief Record for a 0-based video frame index, using time = frameIndex / fps.
     * @return nullptr if fps <= 0 or no record covers the frame (e.g. the video is longer than the SRT).
     */
    [[nodiscard]] const DJIVideoCaptionRecord * recordForFrame(std::size_t frameIndex, double fps) const noexcept;

    /** @brief Record with the given 1-based FrameCnt; nullptr if absent. */
    [[nodiscard]] const DJIVideoCaptionRecord * recordForFrameCount(int frameCount) const noexcept;

    /** @brief Altitude [m] at a video frame (nullopt if no record or altitude missing). */
    [[nodiscard]] std::optional<double> altitudeAtFrame(std::size_t frameIndex, double fps) const noexcept;

    /** @brief Altitude [m] at a time [s] (nullopt if no record or altitude missing). */
    [[nodiscard]] std::optional<double> altitudeAtTime(double seconds) const noexcept;

    /**
     * @brief Altitude value-change events: one entry per run of identical consecutive altitude values
     * (the first record with a given value). Used to schedule altimeter updates.
     */
    [[nodiscard]] std::vector<AltitudeSample> altitudeUpdates() const;

    /**
     * @brief VALIDATION ONLY. Latitude/longitude of the record covering a video frame.
     * Must not be used as a filter measurement. Returns nullopt if no record covers the frame.
     */
    [[nodiscard]] std::optional<DJIValidationOnlyGeodeticSample> validationOnlyGeodeticForFrame(std::size_t frameIndex, double fps) const;

    /**
     * @brief VALIDATION ONLY. Geodetic samples, parallel to records() (same index).
     * Must not be used as a filter measurement.
     */
    [[nodiscard]] std::span<const DJIValidationOnlyGeodeticSample> validationOnlyGeodeticSamples() const noexcept { return validationGeodetic_; }

private:
    std::vector<DJIVideoCaptionRecord> records_;
    std::vector<DJIValidationOnlyGeodeticSample> validationGeodetic_;

    /** @brief Index of the record covering time t, or npos. */
    [[nodiscard]] std::size_t indexAtTime(double seconds) const noexcept;
};

#endif
