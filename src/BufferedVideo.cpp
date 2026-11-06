#include "BufferedVideo.h"

#include <stdexcept>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace
{

/// Validated queue depth. A depth below 1 would make the producer wait forever for a free slot.
std::size_t checkedCapacity(int bufferSize, const char * who)
{
    if (bufferSize < 1)
        throw std::invalid_argument(std::string(who) + ": buffer size must be at least 1");
    return static_cast<std::size_t>(bufferSize);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// BufferedVideoReader
// ---------------------------------------------------------------------------------------------------------------------

BufferedVideoReader::BufferedVideoReader(int bufferSize)
    : readBufferSize(checkedCapacity(bufferSize, "BufferedVideoReader"))
    , readFinished(true)   // nothing is running: read() returns the end-of-stream frame instead of waiting forever
    , pCap(nullptr)
{}

BufferedVideoReader::~BufferedVideoReader()
{
    // Joins the worker before the capture is released. Cannot throw: an undelivered worker error is dropped here.
    joinWorker();
}

void BufferedVideoReader::setFaultHookForTesting(std::function<void()> hook)
{
    if (videoReaderThread.joinable())
        throw std::logic_error("BufferedVideoReader::setFaultHookForTesting: call before start()");
    faultHook = std::move(hook);
}

void BufferedVideoReader::start(cv::VideoCapture & cap)
{
    if (videoReaderThread.joinable())
        throw std::logic_error("BufferedVideoReader::start: already started (call stop() first)");
    {   // reset the shared state before the thread exists, so a prompt stop() cannot be overwritten
        std::lock_guard<std::mutex> lock(readBufferMutex);
        readBuffer = std::queue<cv::Mat>();
        workerError = nullptr;
        readFinished = false;
    }
    pCap = &cap;
    videoReaderThread = std::thread(&BufferedVideoReader::videoReader, this, std::ref(cap));
}

void BufferedVideoReader::stop()
{
    joinWorker();
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(readBufferMutex);
        error = std::exchange(workerError, nullptr);
    }
    if (error)
        std::rethrow_exception(error);
}

void BufferedVideoReader::joinWorker() noexcept
{
    // Joining the thread and releasing pCap are independent concerns, each guarded separately.
    if (videoReaderThread.joinable())
    {   // Store the flag under the mutex so a waiter cannot miss the notification (lost wakeup)
        {
            std::lock_guard<std::mutex> lock(readBufferMutex);
            readFinished = true;
        }
        readBufferNotFull.notify_all();   // unblock producer
        readBufferNotEmpty.notify_all();  // unblock consumer
        videoReaderThread.join();
    }
    if (pCap)
    {
        pCap->release();
        pCap = nullptr;
    }
}

cv::Mat BufferedVideoReader::read()
{
    cv::Mat frame;
    std::exception_ptr error;
    bool popped = false;
    {   // begin scope for lock
        std::unique_lock<std::mutex> lock(readBufferMutex);

        // Wait until buffer not empty or video finished
        while (readBuffer.empty() && !readFinished)
            readBufferNotEmpty.wait(lock);

        if (readBuffer.empty())
        {
            // End of stream. A worker failure is delivered here, after every frame decoded before it.
            error = std::exchange(workerError, nullptr);
        }
        else
        {
            frame = readBuffer.front();
            readBuffer.pop();
            popped = true;
        }
    }   // end scope for lock
    if (popped)
        readBufferNotFull.notify_one();
    if (error)
        std::rethrow_exception(error);
    return frame;
}

void BufferedVideoReader::videoReader(cv::VideoCapture & cap)
{
    try
    {
        while (true)
        {
            if (faultHook)
                faultHook();

            cv::Mat frame;
            cap >> frame;

            if (frame.empty())
                break;

            {   // begin scope for lock
                std::unique_lock<std::mutex> lock(readBufferMutex);

                // Wait until buffer not full or video finished
                while (readBuffer.size() >= readBufferSize && !readFinished)
                    readBufferNotFull.wait(lock);

                if (readFinished)
                    break;

                readBuffer.emplace(std::move(frame));
            }   // end scope for lock
            readBufferNotEmpty.notify_one();
        }
    }
    catch (...)
    {   // Captured for the consumer; the thread itself must not leave this function by an exception.
        std::lock_guard<std::mutex> lock(readBufferMutex);
        workerError = std::current_exception();
    }
    {   // Store the flag under the mutex so a waiting read() cannot miss the notification (lost wakeup)
        std::lock_guard<std::mutex> lock(readBufferMutex);
        readFinished = true;
    }
    readBufferNotEmpty.notify_all(); // unblock waiting thread
}

// ---------------------------------------------------------------------------------------------------------------------
// BufferedVideoWriter
// ---------------------------------------------------------------------------------------------------------------------

BufferedVideoWriter::BufferedVideoWriter(int bufferSize)
    : writeBufferSize(checkedCapacity(bufferSize, "BufferedVideoWriter"))
    , writeFinished(false)
    , pVideo(nullptr)
{}

BufferedVideoWriter::~BufferedVideoWriter()
{
    // Joins the worker before the VideoWriter is released. Cannot throw: an undelivered worker error is dropped here.
    joinWorker();
}

void BufferedVideoWriter::setFaultHookForTesting(std::function<void()> hook)
{
    if (videoWriterThread.joinable())
        throw std::logic_error("BufferedVideoWriter::setFaultHookForTesting: call before start()");
    faultHook = std::move(hook);
}

void BufferedVideoWriter::start(cv::VideoWriter & video)
{
    if (videoWriterThread.joinable())
        throw std::logic_error("BufferedVideoWriter::start: already started (call stop() first)");
    {   // reset the shared state before the thread exists
        std::lock_guard<std::mutex> lock(writeBufferMutex);
        writeBuffer = std::queue<cv::Mat>();
        workerError = nullptr;
        writeFailed = false;
        writeFinished = false;
    }
    pVideo = &video;
    videoWriterThread = std::thread(&BufferedVideoWriter::videoWriter, this, std::ref(video));
}

void BufferedVideoWriter::stop()
{
    joinWorker();   // flushes the queue unless the worker failed
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(writeBufferMutex);
        error = std::exchange(workerError, nullptr);
    }
    if (error)
        std::rethrow_exception(error);
}

void BufferedVideoWriter::joinWorker() noexcept
{
    if (videoWriterThread.joinable())
    {   // Store the flag under the mutex so the writer thread cannot miss the notification (lost wakeup)
        {
            std::lock_guard<std::mutex> lock(writeBufferMutex);
            writeFinished = true;
        }
        writeBufferNotEmpty.notify_all(); // unblock waiting thread
        writeBufferNotFull.notify_all();  // unblock a producer that a failed worker left waiting
        videoWriterThread.join();
    }
    if (pVideo)
    {
        pVideo->release();
        pVideo = nullptr;
    }
}

void BufferedVideoWriter::write(const cv::Mat & frame)
{
    if (!videoWriterThread.joinable())
        throw std::logic_error("BufferedVideoWriter::write: writer not started");

    cv::Mat copy = frame.clone();   // the caller reuses its buffer
    std::exception_ptr error;
    bool failed = false;
    {   // begin scope for lock
        std::unique_lock<std::mutex> lock(writeBufferMutex);

        // Wait until buffer not full, or the writer has failed (then nothing will ever free a slot)
        while (writeBuffer.size() >= writeBufferSize && !writeFailed)
            writeBufferNotFull.wait(lock);

        failed = writeFailed;
        if (failed)
            error = std::exchange(workerError, nullptr);
        else
            writeBuffer.emplace(std::move(copy));
    }   // end scope for lock
    if (failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("BufferedVideoWriter::write: the writer thread has failed; the frame was not written");
    }
    writeBufferNotEmpty.notify_one();
}

void BufferedVideoWriter::videoWriter(cv::VideoWriter & video)
{
    try
    {
        while (true)
        {
            cv::Mat frame;
            {   // begin scope for lock
                std::unique_lock<std::mutex> lock(writeBufferMutex);

                // Wait until buffer not empty or video finished
                while (writeBuffer.empty() && !writeFinished)
                    writeBufferNotEmpty.wait(lock);

                if (writeBuffer.empty() && writeFinished)
                    break;

                frame = std::move(writeBuffer.front());
                writeBuffer.pop();
            }   // end scope for lock
            writeBufferNotFull.notify_one();

            if (faultHook)
                faultHook();
            video.write(frame);
        }
    }
    catch (...)
    {   // Record the failure, drop the queued frames (they cannot be written) and wake any blocked producer.
        {
            std::lock_guard<std::mutex> lock(writeBufferMutex);
            workerError = std::current_exception();
            writeFailed = true;
            writeBuffer = std::queue<cv::Mat>();
        }
        writeBufferNotFull.notify_all();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// BufferedImageWriter
// ---------------------------------------------------------------------------------------------------------------------

BufferedImageWriter::BufferedImageWriter(int bufferSize)
    : writeBufferSize(checkedCapacity(bufferSize, "BufferedImageWriter"))
    , writeFinished(false)
{}

BufferedImageWriter::~BufferedImageWriter()
{
    joinWorker();
}

void BufferedImageWriter::setFaultHookForTesting(std::function<void()> hook)
{
    if (writerThread.joinable())
        throw std::logic_error("BufferedImageWriter::setFaultHookForTesting: call before start()");
    faultHook = std::move(hook);
}

void BufferedImageWriter::start()
{
    if (writerThread.joinable())
        throw std::logic_error("BufferedImageWriter::start: already started (call stop() first)");
    {
        std::lock_guard<std::mutex> lock(writeBufferMutex);
        writeBuffer = std::queue<ImageWriteTask>();
        workerError = nullptr;
        writeFailed = false;
        writeFinished = false;
    }
    writerThread = std::thread(&BufferedImageWriter::imageWriter, this);
}

void BufferedImageWriter::stop()
{
    joinWorker();
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(writeBufferMutex);
        error = std::exchange(workerError, nullptr);
    }
    if (error)
        std::rethrow_exception(error);
}

void BufferedImageWriter::joinWorker() noexcept
{
    // Guard: stop() is safe to call when start() was never called (the thread is then not joinable).
    if (!writerThread.joinable())
        return;

    {   // Store the flag under the mutex so the writer thread cannot miss the notification (lost wakeup)
        std::lock_guard<std::mutex> lock(writeBufferMutex);
        writeFinished = true;
    }
    writeBufferNotEmpty.notify_all(); // unblock waiting thread
    writeBufferNotFull.notify_all();
    writerThread.join();
}

void BufferedImageWriter::write(const std::string & path, const cv::Mat & image)
{
    if (!writerThread.joinable())
        throw std::logic_error("BufferedImageWriter::write: writer not started");

    cv::Mat copy = image.clone();
    std::exception_ptr error;
    bool failed = false;
    {   // begin scope for lock
        std::unique_lock<std::mutex> lock(writeBufferMutex);

        // Wait until buffer not full, or the writer has failed
        while (writeBuffer.size() >= writeBufferSize && !writeFailed)
            writeBufferNotFull.wait(lock);

        failed = writeFailed;
        if (failed)
            error = std::exchange(workerError, nullptr);
        else
            writeBuffer.emplace(ImageWriteTask{path, std::move(copy)});
    }   // end scope for lock
    if (failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("BufferedImageWriter::write: the writer thread has failed; the image was not written");
    }
    writeBufferNotEmpty.notify_one();
}

void BufferedImageWriter::imageWriter()
{
    try
    {
        while (true)
        {
            ImageWriteTask task;
            {   // begin scope for lock
                std::unique_lock<std::mutex> lock(writeBufferMutex);

                // Wait until buffer not empty or finished
                while (writeBuffer.empty() && !writeFinished)
                    writeBufferNotEmpty.wait(lock);

                if (writeBuffer.empty() && writeFinished)
                    break;

                task = std::move(writeBuffer.front());
                writeBuffer.pop();
            }   // end scope for lock
            writeBufferNotFull.notify_one();

            if (faultHook)
                faultHook();
            cv::imwrite(task.path, task.image);
        }
    }
    catch (...)
    {
        {
            std::lock_guard<std::mutex> lock(writeBufferMutex);
            workerError = std::current_exception();
            writeFailed = true;
            writeBuffer = std::queue<ImageWriteTask>();
        }
        writeBufferNotFull.notify_all();
    }
}
