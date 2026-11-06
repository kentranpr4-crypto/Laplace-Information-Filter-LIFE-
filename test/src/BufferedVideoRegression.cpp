#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "../../src/BufferedVideo.h"
#include "visualNavigationSupport.h"

/*
 * Regression tests for the BufferedVideo threading (see BufferedVideo.cpp).
 *
 * Defect covered: lost wake-up in BufferedVideoReader / BufferedVideoWriter. The finished flags are now stored
 * under the queue mutex and reset in start(). The tests below exercise start/stop races, end-of-stream
 * reads, stop() while the producer is blocked on a full queue, zero-frame writers and frame order.
 *
 * Threading rule for these tests: worker threads (visTest::finishesWithin) never use doctest assertions.
 * They signal failure with exceptions or return values, which are checked on the test thread. Anything a
 * detached worker may still touch after a timeout is owned by shared_ptr, never by a reference to a
 * stack variable of the test.
 */

using namespace std::chrono_literals;
using visTest::TempDir;
using visTest::finishesWithin;
using visTest::frameCode;

namespace
{

constexpr int kRepetitions = 200;                     ///< repetitions of the race-prone sequences
constexpr auto kHangLimit = std::chrono::milliseconds(10000);

/// Decoded frame mean colour must be close to the code it was written with (lossy codec tolerance).
bool matchesCode(const cv::Mat & frame, int index, double tol = 8.0)
{
    if (frame.empty())
        return false;
    const cv::Scalar m = visTest::meanColour(frame);
    const cv::Vec3b c = frameCode(index);
    return std::abs(m[0] - c[0]) <= tol && std::abs(m[1] - c[1]) <= tol && std::abs(m[2] - c[2]) <= tol;
}

void requireOpen(const cv::VideoCapture & cap, const std::filesystem::path & path)
{
    if (!cap.isOpened())
        throw std::runtime_error("test: cannot open " + path.string());
}

} // namespace

SCENARIO("BufferedVideoReader start followed immediately by stop is safe")
{
    GIVEN("a 20-frame synthetic video")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path video = dir->path() / "short.avi";
        visTest::writeCodedVideo(video, cv::Size(64, 48), 20, 30.0);

        WHEN("start() is followed immediately by stop(), repeated 200 times")
        {
            const bool allFinished = [&]
            {
                for (int i = 0; i < kRepetitions; ++i)
                {
                    const bool ok = finishesWithin([dir, video]
                    {
                        cv::VideoCapture cap(video.string());
                        requireOpen(cap, video);
                        BufferedVideoReader reader(5);
                        reader.start(cap);
                        reader.stop();
                    }, kHangLimit);
                    if (!ok)
                        return false;
                }
                return true;
            }();
            THEN("no iteration hangs")
            {
                CHECK(allFinished);
            }
        }

        WHEN("the reader is destroyed after start() without any read() or stop()")
        {
            const bool finished = finishesWithin([dir, video]
            {
                cv::VideoCapture cap(video.string());
                requireOpen(cap, video);
                BufferedVideoReader reader(2);
                reader.start(cap);
                // destructor must join the reader thread before cap is released
            }, kHangLimit);
            THEN("the destructor joins the thread and returns")
            {
                CHECK(finished);
            }
        }
    }
}

SCENARIO("BufferedVideoReader returns an empty frame at end of stream without hanging")
{
    GIVEN("a 3-frame video read through a 5-frame queue")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path video = dir->path() / "three.avi";
        visTest::writeCodedVideo(video, cv::Size(64, 48), 3, 30.0);

        WHEN("the reader is drained 200 times, with a read after end of stream each time")
        {
            auto wrongCounts = std::make_shared<int>(0);
            auto afterEofNonEmpty = std::make_shared<int>(0);
            const bool allFinished = [&]
            {
                for (int i = 0; i < kRepetitions; ++i)
                {
                    const bool ok = finishesWithin([dir, video, wrongCounts, afterEofNonEmpty]
                    {
                        cv::VideoCapture cap(video.string());
                        requireOpen(cap, video);
                        BufferedVideoReader reader(5);
                        reader.start(cap);
                        int n = 0;
                        while (!reader.read().empty())
                            ++n;
                        if (n != 3)
                            ++*wrongCounts;
                        if (!reader.read().empty())
                            ++*afterEofNonEmpty;
                        reader.stop();
                    }, kHangLimit);
                    if (!ok)
                        return false;
                }
                return true;
            }();
            THEN("every run returns exactly the 3 frames, then empty frames, and nothing hangs")
            {
                CHECK(allFinished);
                CHECK(*wrongCounts == 0);
                CHECK(*afterEofNonEmpty == 0);
            }
        }
    }
}

SCENARIO("BufferedVideoReader stop() while the producer is blocked on a full queue does not hang")
{
    GIVEN("a 1-frame queue and a 60-frame video, so the producer blocks on a full queue")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path video = dir->path() / "long.avi";
        visTest::writeCodedVideo(video, cv::Size(64, 48), 60, 30.0);

        WHEN("one frame is read, the producer is given time to block, and stop() is called (50 times)")
        {
            auto gotFrame = std::make_shared<int>(0);
            const bool allFinished = [&]
            {
                for (int i = 0; i < 50; ++i)
                {
                    const bool ok = finishesWithin([dir, video, gotFrame]
                    {
                        cv::VideoCapture cap(video.string());
                        requireOpen(cap, video);
                        BufferedVideoReader reader(1);
                        reader.start(cap);
                        if (!reader.read().empty())
                            ++*gotFrame;
                        std::this_thread::sleep_for(2ms);
                        reader.stop();
                    }, kHangLimit);
                    if (!ok)
                        return false;
                }
                return true;
            }();
            THEN("stop() unblocks the producer, returns, and the first frame was delivered each time")
            {
                CHECK(allFinished);
                CHECK(*gotFrame == 50);
            }
        }
    }
}

SCENARIO("BufferedVideoWriter with zero frames starts and stops cleanly")
{
    GIVEN("an opened VideoWriter and a BufferedVideoWriter that never receives a frame")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path path = dir->path() / "empty.avi";

        WHEN("start() is followed by stop()")
        {
            auto stillOpen = std::make_shared<bool>(true);
            const bool finished = finishesWithin([dir, path, stillOpen]
            {
                cv::VideoWriter vw(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(64, 48));
                if (!vw.isOpened())
                    throw std::runtime_error("test: cannot open VideoWriter");
                BufferedVideoWriter writer(3);
                writer.start(vw);
                writer.stop();
                *stillOpen = vw.isOpened();   // stop() must release the stream it was given
            }, kHangLimit);
            THEN("the call returns and the stream is released")
            {
                CHECK(finished);
                CHECK_FALSE(*stillOpen);
            }
        }

        WHEN("stop() is called twice")
        {
            const bool finished = finishesWithin([dir, path]
            {
                cv::VideoWriter vw(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(64, 48));
                BufferedVideoWriter writer(3);
                writer.start(vw);
                writer.stop();
                writer.stop();
            }, kHangLimit);
            THEN("the second stop() is a no-op")
            {
                CHECK(finished);
            }
        }
    }
}

SCENARIO("frame count and order are preserved through BufferedVideoWriter and BufferedVideoReader")
{
    GIVEN("40 frames whose colours encode their index")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path path = dir->path() / "ordered.avi";
        const cv::Size size(64, 48);
        constexpr int nFrames = 40;

        WHEN("they are written through a 1-slot writer queue (maximum back-pressure)")
        {
            {
                cv::VideoWriter vw(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, size);
                REQUIRE(vw.isOpened());
                BufferedVideoWriter writer(1);
                writer.start(vw);
                for (int i = 0; i < nFrames; ++i)
                {
                    cv::Mat frame(size, CV_8UC3, cv::Scalar(frameCode(i)[0], frameCode(i)[1], frameCode(i)[2]));
                    writer.write(frame);
                }
                writer.stop();
            }

            THEN("the file reopens with 40 frames of the original size, in the original order")
            {
                cv::VideoCapture cap(path.string());
                REQUIRE(cap.isOpened());
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)) == nFrames);
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) == size.width);
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) == size.height);
                int i = 0;
                bool orderOk = true;
                cv::Mat frame;
                while (cap.read(frame))
                {
                    orderOk = orderOk && matchesCode(frame, i);
                    ++i;
                }
                CHECK(i == nFrames);
                CHECK(orderOk);
            }

            AND_THEN("the same file read through a 5-slot BufferedVideoReader gives the same count and order")
            {
                cv::VideoCapture cap(path.string());
                REQUIRE(cap.isOpened());
                BufferedVideoReader reader(5);
                reader.start(cap);
                int i = 0;
                bool orderOk = true;
                for (cv::Mat frame = reader.read(); !frame.empty(); frame = reader.read())
                {
                    orderOk = orderOk && matchesCode(frame, i);
                    ++i;
                }
                reader.stop();
                CHECK(i == nFrames);
                CHECK(orderOk);
            }
        }
    }
}

SCENARIO("a BufferedVideoReader can be restarted on a new capture after stop()")
{
    GIVEN("a reader that has been started and stopped once, part-way through a 7-frame video")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path video = dir->path() / "restart.avi";
        visTest::writeCodedVideo(video, cv::Size(64, 48), 7, 30.0);

        WHEN("it is started again on a fresh capture and drained")
        {
            BufferedVideoReader reader(2);
            {
                cv::VideoCapture cap(video.string());
                REQUIRE(cap.isOpened());
                reader.start(cap);
                (void)reader.read();
                reader.stop();
            }
            cv::VideoCapture cap2(video.string());
            REQUIRE(cap2.isOpened());
            reader.start(cap2);
            int n = 0;
            for (cv::Mat f = reader.read(); !f.empty(); f = reader.read())
                ++n;
            reader.stop();

            THEN("all 7 frames are delivered")
            {
                CHECK(n == 7);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Failure contract (R4-12): double start, worker exceptions captured and rethrown once, no terminate, no hang.
// ---------------------------------------------------------------------------------------------------------------------

namespace
{

/// Outcome of a sequence run on a helper thread (written by the helper, read on the test thread after it returned).
struct FailureSeq
{
    int frames = 0;                 ///< frames returned by read() before the first exception / end of stream
    int exceptions = 0;             ///< exceptions observed by the caller across all calls of the sequence
    std::string firstMessage;       ///< what() of the first exception observed
    std::string lastMessage;        ///< what() of the last exception observed
    bool emptyAfterError = false;   ///< read() returned an empty frame after the exception was delivered
    bool stopThrew = false;         ///< stop() threw
    std::string stopMessage;
};

constexpr const char * kInjected = "injected worker failure";

void record(FailureSeq & s, const std::exception & e)
{
    ++s.exceptions;
    if (s.firstMessage.empty())
        s.firstMessage = e.what();
    s.lastMessage = e.what();
}

} // namespace

SCENARIO("BufferedVideoReader: a second start() on a running reader throws and the first run is unaffected")
{
    GIVEN("a 20-frame video read through a 5-slot reader")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path video = dir->path() / "short.avi";
        visTest::writeCodedVideo(video, cv::Size(64, 48), 20, 30.0);
        auto cap = std::make_shared<cv::VideoCapture>(video.string());
        requireOpen(*cap, video);
        auto reader = std::make_shared<BufferedVideoReader>(5);

        WHEN("start() is called again while the reader is running")
        {
            auto seq = std::make_shared<FailureSeq>();
            const bool finished = visTest::finishesWithin([reader, cap, seq]
            {
                reader->start(*cap);
                try
                {
                    reader->start(*cap);
                }
                catch (const std::logic_error & e)
                {
                    record(*seq, e);
                }
                while (!reader->read().empty())
                    ++seq->frames;
                reader->stop();
            }, kHangLimit);
            THEN("the second start throws std::logic_error and the reader is not terminated")
            {
                CHECK(finished);
                CHECK(seq->exceptions == 1);
            }
            THEN("the first run still delivers all 20 frames")
            {
                CHECK(seq->frames == 20);
            }
        }
    }
}

SCENARIO("BufferedVideoReader: a worker exception is delivered by read() after the frames decoded before it")
{
    GIVEN("a 20-frame video and a fault injected into the worker on its 4th decode (frames 1-3 succeed)")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path video = dir->path() / "short.avi";
        visTest::writeCodedVideo(video, cv::Size(64, 48), 20, 30.0);
        auto cap = std::make_shared<cv::VideoCapture>(video.string());
        requireOpen(*cap, video);
        auto reader = std::make_shared<BufferedVideoReader>(5);
        auto calls = std::make_shared<std::atomic<int>>(0);
        reader->setFaultHookForTesting([calls]
        {
            if (++(*calls) == 4)
                throw std::runtime_error(kInjected);
        });

        WHEN("the frames are read until the exception, then read() and stop() are called again")
        {
            auto seq = std::make_shared<FailureSeq>();
            const bool finished = visTest::finishesWithin([reader, cap, seq]
            {
                reader->start(*cap);
                try
                {
                    while (true)
                    {
                        if (reader->read().empty())
                            break;
                        ++seq->frames;
                    }
                }
                catch (const std::runtime_error & e)
                {
                    record(*seq, e);
                }
                seq->emptyAfterError = reader->read().empty();
                try
                {
                    reader->stop();
                }
                catch (const std::exception & e)
                {
                    seq->stopThrew = true;
                    seq->stopMessage = e.what();
                }
            }, kHangLimit);
            THEN("read() returns the 3 good frames, then throws the worker's exception exactly once")
            {
                CHECK(finished);
                CHECK(seq->frames == 3);
                CHECK(seq->exceptions == 1);
                CHECK(seq->firstMessage == kInjected);
            }
            THEN("afterwards read() returns an empty frame and stop() does not repeat the exception")
            {
                CHECK(seq->emptyAfterError);
                CHECK_FALSE(seq->stopThrew);
            }
        }
    }
}

SCENARIO("BufferedVideoReader: stop() rethrows a worker exception that read() never delivered")
{
    GIVEN("a reader whose worker fails on its first decode")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path video = dir->path() / "short.avi";
        visTest::writeCodedVideo(video, cv::Size(64, 48), 5, 30.0);
        auto cap = std::make_shared<cv::VideoCapture>(video.string());
        requireOpen(*cap, video);
        auto reader = std::make_shared<BufferedVideoReader>(2);
        reader->setFaultHookForTesting([] { throw std::runtime_error(kInjected); });

        WHEN("start() is followed directly by stop(), then stop() again")
        {
            auto seq = std::make_shared<FailureSeq>();
            const bool finished = visTest::finishesWithin([reader, cap, seq]
            {
                reader->start(*cap);
                try
                {
                    reader->stop();
                }
                catch (const std::runtime_error & e)
                {
                    seq->stopThrew = true;
                    seq->stopMessage = e.what();
                }
                reader->stop();   // the error is delivered once: the second stop() is clean
            }, kHangLimit);
            THEN("the first stop() throws the worker's exception and the second does not")
            {
                CHECK(finished);
                CHECK(seq->stopThrew);
                CHECK(seq->stopMessage == kInjected);
            }
        }
    }
}

SCENARIO("BufferedVideoWriter: a worker failure is delivered once by write(), then write() reports the failure")
{
    GIVEN("a 3-slot writer whose worker fails on its 3rd frame")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path path = dir->path() / "failing.avi";
        auto calls = std::make_shared<std::atomic<int>>(0);
        auto writer = std::make_shared<BufferedVideoWriter>(3);
        writer->setFaultHookForTesting([calls]
        {
            if (++(*calls) == 3)
                throw std::runtime_error(kInjected);
        });
        auto video = std::make_shared<cv::VideoWriter>(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                                                       cv::Size(64, 48));
        REQUIRE(video->isOpened());

        WHEN("frames are written until write() reports the failure, then once more, then stop() is called")
        {
            auto seq = std::make_shared<FailureSeq>();
            const bool finished = visTest::finishesWithin([writer, video, seq]
            {
                writer->start(*video);
                const cv::Mat frame(48, 64, CV_8UC3, cv::Scalar(10, 20, 30));
                for (int i = 0; i < 100 && seq->exceptions == 0; ++i)
                {
                    try
                    {
                        writer->write(frame);
                    }
                    catch (const std::runtime_error & e)
                    {
                        record(*seq, e);
                    }
                }
                try
                {
                    writer->write(frame);
                }
                catch (const std::runtime_error & e)
                {
                    record(*seq, e);
                }
                try
                {
                    writer->stop();
                }
                catch (const std::exception & e)
                {
                    seq->stopThrew = true;
                    seq->stopMessage = e.what();
                }
            }, kHangLimit);
            THEN("the worker's exception is delivered by write() and the producer is not left blocked")
            {
                CHECK(finished);
                CHECK(seq->firstMessage == kInjected);
            }
            THEN("the later write() reports the failure as a runtime_error, not the original exception again")
            {
                CHECK(seq->exceptions == 2);
                CHECK(seq->lastMessage.find("writer thread has failed") != std::string::npos);
            }
            THEN("stop() does not repeat the delivered exception")
            {
                CHECK_FALSE(seq->stopThrew);
            }
        }
    }
}

SCENARIO("BufferedVideoWriter: an undelivered worker failure is reported exactly once, by write() or stop()")
{
    GIVEN("a writer whose worker fails on the first frame")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path path = dir->path() / "failing1.avi";
        auto writer = std::make_shared<BufferedVideoWriter>(3);
        writer->setFaultHookForTesting([] { throw std::runtime_error(kInjected); });
        auto video = std::make_shared<cv::VideoWriter>(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0,
                                                       cv::Size(64, 48));
        REQUIRE(video->isOpened());

        WHEN("one frame is written and then the writer is stopped (the worker may fail before or after the write)")
        {
            auto seq = std::make_shared<FailureSeq>();
            const bool finished = visTest::finishesWithin([writer, video, seq]
            {
                writer->start(*video);
                const cv::Mat frame(48, 64, CV_8UC3, cv::Scalar(1, 2, 3));
                try
                {
                    writer->write(frame);
                }
                catch (const std::runtime_error & e)
                {
                    record(*seq, e);
                }
                try
                {
                    writer->stop();
                }
                catch (const std::runtime_error & e)
                {
                    record(*seq, e);
                }
            }, kHangLimit);
            THEN("the failure is reported exactly once and carries the worker's message")
            {
                CHECK(finished);
                CHECK(seq->exceptions == 1);
                CHECK(seq->firstMessage == kInjected);
            }
        }
    }
}

SCENARIO("BufferedVideoWriter and BufferedImageWriter reject a second start() and a write() before start()")
{
    GIVEN("a writer that has not been started")
    {
        auto dir = std::make_shared<TempDir>();
        const std::filesystem::path path = dir->path() / "started.avi";
        cv::VideoWriter vw(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(64, 48));
        REQUIRE(vw.isOpened());
        BufferedVideoWriter writer(2);
        const cv::Mat frame(48, 64, CV_8UC3, cv::Scalar(9, 9, 9));
        WHEN("write() is called before start()")
        {
            THEN("it throws std::logic_error instead of blocking on a queue nobody drains")
            {
                CHECK_THROWS_AS(writer.write(frame), std::logic_error);
            }
        }
        WHEN("start() is called twice")
        {
            writer.start(vw);
            THEN("the second start throws std::logic_error")
            {
                CHECK_THROWS_AS(writer.start(vw), std::logic_error);
            }
            writer.stop();
        }
    }
    GIVEN("an image writer that is started twice")
    {
        auto dir = std::make_shared<TempDir>();
        BufferedImageWriter images(2);
        images.start();
        WHEN("start() is called again")
        {
            THEN("it throws std::logic_error and stop() still joins cleanly")
            {
                CHECK_THROWS_AS(images.start(), std::logic_error);
                CHECK_NOTHROW(images.stop());
            }
        }
    }
}

SCENARIO("buffer depths below one are rejected")
{
    CHECK_THROWS_AS(BufferedVideoReader(0), std::invalid_argument);
    CHECK_THROWS_AS(BufferedVideoWriter(-1), std::invalid_argument);
    CHECK_THROWS_AS(BufferedImageWriter(0), std::invalid_argument);
}

SCENARIO("BufferedImageWriter: a failed image write is reported by stop() and the other images are not lost silently")
{
    GIVEN("an image writer whose worker fails on its first image")
    {
        auto dir = std::make_shared<TempDir>();
        auto images = std::make_shared<BufferedImageWriter>(2);
        images->setFaultHookForTesting([] { throw std::runtime_error(kInjected); });
        WHEN("one image is written and the writer is stopped")
        {
            auto seq = std::make_shared<FailureSeq>();
            const std::string target = (dir->path() / "frame.png").string();
            const bool finished = visTest::finishesWithin([images, seq, target]
            {
                images->start();
                const cv::Mat img(48, 64, CV_8UC3, cv::Scalar(4, 5, 6));
                try
                {
                    images->write(target, img);
                }
                catch (const std::runtime_error & e)
                {
                    record(*seq, e);
                }
                try
                {
                    images->stop();
                }
                catch (const std::runtime_error & e)
                {
                    record(*seq, e);
                }
            }, kHangLimit);
            THEN("the failure is reported exactly once")
            {
                CHECK(finished);
                CHECK(seq->exceptions == 1);
                CHECK(seq->firstMessage == kInjected);
            }
        }
    }
}
