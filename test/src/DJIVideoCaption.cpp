#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../src/DJIVideoCaption.h"

namespace
{
/** @brief NTSC 60 fps as used by the outdoor flight video (60000/1001 frames per second). */
constexpr double kFlightFps = 60000.0 / 1001.0;

std::string readFile(const std::filesystem::path & path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string replaceAll(std::string text, const std::string & from, const std::string & to)
{
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos)
    {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

/** @brief Builds one well-formed SRT block (LF line endings). */
std::string block(int index, int frameCount, const std::string & start, const std::string & end, double altitude)
{
    return std::to_string(index) + "\n" + start + " --> " + end + "\n<font size=\"36\">FrameCnt : " + std::to_string(frameCount) +
           ", DiffTime : 16ms\n2021-09-29 11:13:38,121,919\n[iso : 110] [shutter : 1/400.0] [fnum : 280] [ev : 0] [ct : 6000] "
           "[color_md : default] [focal_len : 280] [latitude : -32.868837] [longtitude : 151.684716] [altitude: " +
           std::to_string(altitude) + "] </font>\n\n";
}

const std::filesystem::path kExcerpt = std::filesystem::path(TEST_DATA_DIR) / "outdoor" / "flight_excerpt.SRT";
const std::filesystem::path kFullFlight = std::filesystem::path(TEST_DATA_DIR).parent_path().parent_path() / "data" / "outdoor" / "flight.SRT";

// Compile-time guarantees for the estimator-facing API: no latitude/longitude members anywhere on the
// record type, and the caption's public record accessor returns only that type.
template <typename T>
concept ExposesGeodetic = requires(const T & t) {
    t.latitudeDeg;
} || requires(const T & t) {
    t.longitudeDeg;
} || requires(const T & t) {
    t.latitude;
} || requires(const T & t) {
    t.longitude;
} || requires(const T & t) {
    t.longtitude;
};
static_assert(!ExposesGeodetic<DJIVideoCaptionRecord>, "estimator-facing record must not expose latitude/longitude");
static_assert(std::is_same_v<decltype(std::declval<const DJIVideoCaption &>().records()), std::span<const DJIVideoCaptionRecord>>,
              "records() must return the estimator-facing record type only");
static_assert(!std::is_same_v<DJIVideoCaptionRecord, DJIValidationOnlyGeodeticSample>, "geodetic samples are a separate type");
} // namespace

SCENARIO("the SRT excerpt parses with exact field values")
{
    REQUIRE(std::filesystem::exists(kExcerpt));
    GIVEN("the first 60 blocks of flight.SRT")
    {
        DJIVideoCaption::ParseReport report;
        const DJIVideoCaption log = DJIVideoCaption::load(kExcerpt, DJIVideoCaption::ParseMode::Strict, &report);
        THEN("all 60 blocks are parsed and nothing is skipped")
        {
            CHECK(log.size() == 60);
            CHECK(report.blocksParsed == 60);
            CHECK(report.blocksSkipped == 0);
            CHECK(report.fieldProblems == 0);
            CHECK(report.missingAltitude == 0);
        }
        THEN("the first block matches the file exactly")
        {
            REQUIRE(log.size() >= 1);
            const DJIVideoCaptionRecord & r = log.records()[0];
            CHECK(r.blockIndex == 1);
            CHECK(r.startSeconds == doctest::Approx(0.0));
            CHECK(r.endSeconds == doctest::Approx(0.016));
            CHECK(r.frameCount == 1);
            CHECK(r.diffTimeMs == doctest::Approx(16.0));
            CHECK(r.wallClock == "2021-09-29 11:13:38,121,919");
            REQUIRE(r.iso.has_value());
            CHECK(*r.iso == 110);
            CHECK(r.shutterText == "1/400.0");
            REQUIRE(r.shutterSeconds.has_value());
            CHECK(*r.shutterSeconds == doctest::Approx(1.0 / 400.0));
            REQUIRE(r.fNumberRaw.has_value());
            CHECK(*r.fNumberRaw == doctest::Approx(280.0));
            REQUIRE(r.evCompensation.has_value());
            CHECK(*r.evCompensation == doctest::Approx(0.0));
            REQUIRE(r.colourTemperatureK.has_value());
            CHECK(*r.colourTemperatureK == 6000);
            CHECK(r.colourMode == "default");
            REQUIRE(r.focalLengthRaw.has_value());
            CHECK(*r.focalLengthRaw == doctest::Approx(280.0));
            REQUIRE(r.altitudeM.has_value());
            CHECK(*r.altitudeM == doctest::Approx(63.126999));
        }
        THEN("the altitude values change at the reference FrameCnt values")
        {
            const auto updates = log.altitudeUpdates();
            REQUIRE(updates.size() == 6);
            CHECK(updates[0].blockIndex == 1);
            CHECK(updates[0].altitudeM == doctest::Approx(63.126999));
            CHECK(updates[1].blockIndex == 3);
            CHECK(updates[1].altitudeM == doctest::Approx(62.963001));
            CHECK(updates[2].blockIndex == 15);
            CHECK(updates[2].altitudeM == doctest::Approx(62.806999));
            CHECK(updates[3].blockIndex == 27);
            CHECK(updates[3].altitudeM == doctest::Approx(62.638));
            CHECK(updates[4].blockIndex == 39);
            CHECK(updates[4].altitudeM == doctest::Approx(62.522999));
            CHECK(updates[5].blockIndex == 51);
            CHECK(updates[5].altitudeM == doctest::Approx(62.423));
        }
    }
}

SCENARIO("SRT frame and time alignment follows FrameCnt = frame index + 1")
{
    REQUIRE(std::filesystem::exists(kExcerpt));
    const DJIVideoCaption log = DJIVideoCaption::load(kExcerpt);
    GIVEN("the 59.94 fps flight timeline")
    {
        WHEN("looking up video frame 0")
        {
            const auto * r = log.recordForFrame(0, kFlightFps);
            THEN("it maps to FrameCnt 1 starting at t = 0") {
                REQUIRE(r != nullptr);
                CHECK(r->frameCount == 1);
                CHECK(r->startSeconds == doctest::Approx(0.0));
            }
        }
        WHEN("looking up video frame 15, whose presentation time (250.25 ms) sits on the block boundary")
        {
            const auto * r = log.recordForFrame(15, kFlightFps);
            THEN("it maps to FrameCnt 16") {
                REQUIRE(r != nullptr);
                CHECK(r->frameCount == 16);
            }
        }
        WHEN("looking up the last frame of the excerpt")
        {
            const auto * r = log.recordForFrame(59, kFlightFps);
            THEN("it maps to FrameCnt 60") {
                REQUIRE(r != nullptr);
                CHECK(r->frameCount == 60);
            }
        }
        WHEN("looking up a frame after the excerpt ends")
        {
            THEN("no record is returned") {
                CHECK(log.recordForFrame(60, kFlightFps) == nullptr);
                CHECK_FALSE(log.altitudeAtFrame(200, kFlightFps).has_value());
            }
        }
        WHEN("the frame rate is invalid")
        {
            THEN("no record is returned") {
                CHECK(log.recordForFrame(3, 0.0) == nullptr);
                CHECK(log.recordForFrame(3, -30.0) == nullptr);
            }
        }
        WHEN("looking up by time at block boundaries")
        {
            THEN("the block containing the time is returned") {
                REQUIRE(log.recordAtTime(0.0) != nullptr);
                CHECK(log.recordAtTime(0.0)->frameCount == 1);
                REQUIRE(log.recordAtTime(0.0165) != nullptr);
                CHECK(log.recordAtTime(0.0165)->frameCount == 2);
                // Times up to 1 ms before a block start are attributed to that block (timestamp rounding tolerance).
                REQUIRE(log.recordAtTime(0.0145) != nullptr);
                CHECK(log.recordAtTime(0.0145)->frameCount == 1);
                REQUIRE(log.recordAtTime(0.0159) != nullptr);
                CHECK(log.recordAtTime(0.0159)->frameCount == 2);
            }
        }
        WHEN("looking up by 1-based FrameCnt")
        {
            THEN("the matching record is returned") {
                REQUIRE(log.recordForFrameCount(30) != nullptr);
                CHECK(log.recordForFrameCount(30)->blockIndex == 30);
                CHECK(log.recordForFrameCount(61) == nullptr);
            }
        }
        WHEN("the altitude is read at frame 0 and at time 0.5 s")
        {
            THEN("the dedicated altitude accessors return the record value") {
                REQUIRE(log.altitudeAtFrame(0, kFlightFps).has_value());
                CHECK(*log.altitudeAtFrame(0, kFlightFps) == doctest::Approx(63.126999));
                REQUIRE(log.altitudeAtTime(0.5).has_value());
                CHECK(*log.altitudeAtTime(0.5) == doctest::Approx(log.recordAtTime(0.5)->altitudeM.value()));
            }
        }
    }
}

SCENARIO("the full flight.SRT aligns with the 9161-frame video")
{
    // The full flight.SRT is a repository fixture: a missing file is a failure, never a silent pass.
    REQUIRE_MESSAGE(std::filesystem::exists(kFullFlight), "missing fixture data/outdoor/flight.SRT: ", kFullFlight.string());
    GIVEN("the full flight.SRT (9160 blocks) and flight.MOV (9161 frames at 60000/1001 fps)")
    {
        const auto start = std::chrono::steady_clock::now();
        DJIVideoCaption::ParseReport report;
        const DJIVideoCaption log = DJIVideoCaption::load(kFullFlight, DJIVideoCaption::ParseMode::Strict, &report);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        THEN("it parses strictly with 9160 records in time order and no altitude gaps")
        {
            CHECK(log.size() == 9160);
            CHECK(report.blocksSkipped == 0);
            CHECK(report.missingAltitude == 0);
            CHECK(seconds < 10.0);
            CHECK(log.records().front().frameCount == 1);
            CHECK(log.records().back().frameCount == 9160);
        }
        THEN("the last video frame (index 9160) has no SRT block, the first 9160 frames each have one")
        {
            REQUIRE(log.recordForFrame(0, kFlightFps) != nullptr);
            CHECK(log.recordForFrame(9159, kFlightFps)->frameCount == 9160);
            CHECK(log.recordForFrame(9160, kFlightFps) == nullptr);
        }
        THEN("the validation-only geodetic sample is paired with the same block")
        {
            const auto geo = log.validationOnlyGeodeticForFrame(0, kFlightFps);
            REQUIRE(geo.has_value());
            REQUIRE(geo->latitudeDeg.has_value());
            CHECK(*geo->latitudeDeg == doctest::Approx(-32.868837));
            CHECK(*geo->longitudeDeg == doctest::Approx(151.684716));
            CHECK(log.validationOnlyGeodeticSamples().size() == log.size());
        }
    }
}

SCENARIO("CRLF line endings, a BOM and missing <font> tags are handled")
{
    REQUIRE(std::filesystem::exists(kExcerpt));
    const std::string lf = readFile(kExcerpt);
    const DJIVideoCaption reference = DJIVideoCaption::parse(lf);
    GIVEN("the excerpt rewritten with CRLF line endings")
    {
        const std::string crlf = replaceAll(lf, "\n", "\r\n");
        const DJIVideoCaption parsed = DJIVideoCaption::parse(crlf, DJIVideoCaption::ParseMode::Strict);
        THEN("the records are identical to the LF parse")
        {
            REQUIRE(parsed.size() == reference.size());
            for (std::size_t i = 0; i < parsed.size(); ++i)
            {
                CHECK(parsed.records()[i].frameCount == reference.records()[i].frameCount);
                CHECK(parsed.records()[i].startSeconds == doctest::Approx(reference.records()[i].startSeconds));
                CHECK(parsed.records()[i].altitudeM == reference.records()[i].altitudeM);
            }
        }
    }
    GIVEN("the excerpt with a UTF-8 byte-order mark")
    {
        const DJIVideoCaption parsed = DJIVideoCaption::parse("\xEF\xBB\xBF" + lf, DJIVideoCaption::ParseMode::Strict);
        THEN("the BOM is ignored") { CHECK(parsed.size() == reference.size()); }
    }
    GIVEN("blocks without <font> tags and without blank lines between them")
    {
        std::string text = "1\n00:00:00,000 --> 00:00:00,016\nFrameCnt : 1, DiffTime : 16ms\n[altitude: 10.5] [latitude: -33.1]\n"
                           "2\n00:00:00,016 --> 00:00:00,033\nFrameCnt : 2, DiffTime : 17ms\n[altitude: 10.25]\n";
        const DJIVideoCaption parsed = DJIVideoCaption::parse(text, DJIVideoCaption::ParseMode::Strict);
        THEN("both blocks are parsed")
        {
            REQUIRE(parsed.size() == 2);
            CHECK(parsed.records()[1].altitudeM.value() == doctest::Approx(10.25));
            CHECK(parsed.records()[1].diffTimeMs == doctest::Approx(17.0));
        }
    }
}

SCENARIO("malformed blocks are skipped with line numbers in lenient mode and rejected in strict mode")
{
    GIVEN("an SRT with a missing FrameCnt, a malformed altitude and a broken timing line")
    {
        const std::string good1 = block(1, 1, "00:00:00,000", "00:00:00,016", 10.0);
        const std::string noFrameCount = "2\n00:00:00,016 --> 00:00:00,033\n[altitude: 10.1]\n\n";
        const std::string badAltitude = block(3, 3, "00:00:00,033", "00:00:00,050", 10.2);
        const std::string badAltitudeText = replaceAll(badAltitude, "altitude: 10.200000", "altitude: abc");
        const std::string badTiming = "4\n00:00:xx,050 --> 00:00:00,066\nFrameCnt : 4, DiffTime : 16ms\n[altitude: 10.3]\n\n";
        const std::string good5 = block(5, 5, "00:00:00,066", "00:00:00,083", 10.4);
        const std::string text = good1 + noFrameCount + badAltitudeText + badTiming + good5;

        WHEN("parsed leniently")
        {
            DJIVideoCaption::ParseReport report;
            const DJIVideoCaption parsed = DJIVideoCaption::parse(text, DJIVideoCaption::ParseMode::Lenient, &report, "bad.SRT");
            THEN("the good blocks survive and the bad block's altitude is absent")
            {
                REQUIRE(parsed.size() == 3);
                CHECK(parsed.records()[0].frameCount == 1);
                CHECK(parsed.records()[1].blockIndex == 3);
                CHECK_FALSE(parsed.records()[1].altitudeM.has_value());
                CHECK(parsed.records()[2].blockIndex == 5);
            }
            THEN("the report counts the skipped blocks and field problems")
            {
                CHECK(report.blocksParsed == 3);
                CHECK(report.blocksSkipped == 2);
                CHECK(report.fieldProblems == 1);
            }
            THEN("each problem is reported with its source name and line number")
            {
                REQUIRE(report.warnings.size() >= 3);
                bool sawFrameCount = false;
                bool sawAltitude = false;
                bool sawTiming = false;
                for (const auto & w : report.warnings)
                {
                    // Block 2 index is on line 7; block 4's broken timing line is line 18.
                    sawFrameCount = sawFrameCount || w.find("bad.SRT:7: missing or malformed FrameCnt") != std::string::npos;
                    sawAltitude = sawAltitude || (w.find("bad.SRT:") != std::string::npos && w.find("malformed altitude") != std::string::npos);
                    sawTiming = sawTiming || w.find("bad.SRT:18: malformed timing line") != std::string::npos;
                }
                CHECK(sawFrameCount);
                CHECK(sawAltitude);
                CHECK(sawTiming);
            }
        }
        WHEN("parsed strictly")
        {
            THEN("the first problem throws with the line number of the block")
            {
                try
                {
                    (void)DJIVideoCaption::parse(text, DJIVideoCaption::ParseMode::Strict, nullptr, "bad.SRT");
                    CHECK_MESSAGE(false, "strict parse should have thrown");
                }
                catch (const std::runtime_error & e)
                {
                    const std::string what = e.what();
                    CHECK(what.find("bad.SRT:") != std::string::npos);
                    CHECK(what.find("FrameCnt") != std::string::npos);
                }
            }
        }
    }

    GIVEN("text without any timing line")
    {
        WHEN("parsed leniently") {
            DJIVideoCaption::ParseReport report;
            const DJIVideoCaption parsed = DJIVideoCaption::parse("this is not an srt file\n", DJIVideoCaption::ParseMode::Lenient, &report);
            THEN("the result is empty and reported") {
                CHECK(parsed.empty());
                CHECK_FALSE(report.warnings.empty());
            }
        }
        WHEN("parsed strictly") {
            THEN("it throws") {
                CHECK_THROWS_AS(DJIVideoCaption::parse("not an srt", DJIVideoCaption::ParseMode::Strict), std::runtime_error);
            }
        }
    }

    GIVEN("a file that does not exist")
    {
        THEN("load throws") {
            CHECK_THROWS_AS(DJIVideoCaption::load(std::filesystem::path(TEST_DATA_DIR) / "outdoor" / "no_such.SRT"), std::runtime_error);
        }
    }
}

SCENARIO("out-of-order blocks are sorted by start time and reported")
{
    GIVEN("blocks 2 and 1 written in reverse order")
    {
        const std::string text = block(2, 2, "00:00:00,016", "00:00:00,033", 11.0) + block(1, 1, "00:00:00,000", "00:00:00,016", 10.0);
        DJIVideoCaption::ParseReport report;
        const DJIVideoCaption parsed = DJIVideoCaption::parse(text, DJIVideoCaption::ParseMode::Lenient, &report);
        THEN("records are in time order and a warning is recorded")
        {
            REQUIRE(parsed.size() == 2);
            CHECK(parsed.records()[0].frameCount == 1);
            CHECK(parsed.records()[1].frameCount == 2);
            CHECK(report.warnings.size() >= 1);
        }
    }
}

SCENARIO("the estimator-facing API does not expose latitude or longitude")
{
    REQUIRE(std::filesystem::exists(kExcerpt));
    GIVEN("a parsed SRT log")
    {
        const DJIVideoCaption log = DJIVideoCaption::load(kExcerpt);
        THEN("the record type carries altitude but no geodetic fields")
        {
            // Compile-time checks above guarantee this; the runtime check documents which field is used.
            const auto & r = log.records().front();
            CHECK(r.altitudeM.has_value());
            CHECK_FALSE(ExposesGeodetic<DJIVideoCaptionRecord>);
        }
        THEN("geodetic values are reachable only through the validation-only accessors")
        {
            const auto geo = log.validationOnlyGeodeticForFrame(0, kFlightFps);
            REQUIRE(geo.has_value());
            CHECK(geo->blockIndex == 1);
            CHECK(log.validationOnlyGeodeticSamples().size() == log.size());
        }
        THEN("the altitude accessor and the altitude update list do not touch geodetic data")
        {
            CHECK(log.altitudeUpdates().size() == 6);
            CHECK(log.altitudeAtFrame(0, kFlightFps).has_value());
        }
    }
}

SCENARIO("block start times are exact decimal times: frame and block times agree bit for bit (C3 regression)")
{
    GIVEN("an SRT with blocks starting at 00:00:00,700 (FrameCnt 22, frame index 21 at 30 fps) and at 00:01:02,345")
    {
        const std::string srt =
            "1\n00:00:00,000 --> 00:00:00,033\n<font size=\"36\">FrameCnt : 1, DiffTime : 33ms\n"
            "2021-09-29 11:13:38,000,000\n[altitude: 60.000000] </font>\n\n"
            "22\n00:00:00,700 --> 00:00:00,733\n<font size=\"36\">FrameCnt : 22, DiffTime : 33ms\n"
            "2021-09-29 11:13:38,700,000\n[altitude: 60.100000] </font>\n\n"
            "3\n00:01:02,345 --> 00:01:02,378\n<font size=\"36\">FrameCnt : 3, DiffTime : 33ms\n"
            "2021-09-29 11:14:40,345,000\n[altitude: 61.250000] </font>\n\n";
        const DJIVideoCaption caption = DJIVideoCaption::parse(srt, DJIVideoCaption::ParseMode::Strict);
        const auto& records = caption.records();
        REQUIRE(records.size() == 3);
        WHEN("the block start times are compared with the frame times they cover")
        {
            THEN("the block start equals the frame time exactly (no floating-point drift)")
            {
                // 0.70000000000000006661 (the former sum of 0 + 0 + 0 + 700 * 1e-3) differed from 21 / 30 = 0.69999999999999995559.
                CHECK(records[1].startSeconds == 21.0 / 30.0);
                CHECK(records[2].startSeconds == 62.345);
                CHECK(records[2].startSeconds == 62345.0 / 1000.0);
            }
            THEN("the frame lookup returns the block that starts at the frame time, and the altitude event uses the same time")
            {
                const DJIVideoCaptionRecord* at21 = caption.recordForFrame(21, 30.0);
                REQUIRE(at21 != nullptr);
                CHECK(at21->frameCount == 22);
                const auto updates = caption.altitudeUpdates();
                REQUIRE(updates.size() == 3);
                CHECK(updates[1].timeSeconds == records[1].startSeconds);
                CHECK(updates[1].timeSeconds == 21.0 / 30.0);
            }
        }
    }
}
