#ifndef BUFFEREDVIDEO_H
#define BUFFEREDVIDEO_H

/**
 * @file BufferedVideo.h
 * @brief Threaded, bounded-queue video reader/writer and image writer (architecture unchanged).
 *
 * Failure contract (R4-12):
 *  - start() on a running object throws std::logic_error (it never replaces a joinable thread).
 *  - An exception raised on the worker thread is captured, not allowed to terminate the process, and rethrown
 *    exactly once in the caller:
 *      BufferedVideoReader: by read(), after every frame decoded before the failure has been returned;
 *      BufferedVideoWriter: by the first write() that observes the failure, or by stop() if no write() did
 *                           (stop() flushes the queue, so a failed final encode is reported there);
 *      BufferedImageWriter: by write() or stop(), by the same rule.
 *    Later write() calls after a failure throw std::runtime_error; the producer is never left blocked.
 *  - The destructor joins the worker and never throws; an undelivered worker error is discarded there, so call
 *    stop() when the error matters.
 */

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

class BufferedVideoReader
{
public:
    /// @throws std::invalid_argument if bufferSize < 1
    explicit BufferedVideoReader(int bufferSize);
    ~BufferedVideoReader();
    /// Starts the reader thread on cap (which must outlive the reader or stop()).
    /// @throws std::logic_error if the reader is already running (call stop() first)
    void start(cv::VideoCapture & cap);
    /// Joins the reader thread and releases the capture.
    /// @throws the worker's exception if read() has not yet delivered it
    void stop();
    /// Next frame in order. Returns an empty frame at end of stream (or when not started).
    /// @throws the worker's exception once, after the frames decoded before the failure
    cv::Mat read();
    /// Test seam: called on the worker thread before each frame is decoded; an exception thrown here takes the
    /// worker-error path. Must be called before start().
    void setFaultHookForTesting(std::function<void()> hook);
private:
    std::size_t readBufferSize;
    std::queue<cv::Mat> readBuffer;
    std::mutex readBufferMutex;
    std::atomic_bool readFinished;
    std::exception_ptr workerError;                 ///< guarded by readBufferMutex
    std::condition_variable readBufferNotEmpty;
    std::condition_variable readBufferNotFull;
    std::thread videoReaderThread;
    cv::VideoCapture *pCap;
    std::function<void()> faultHook;
    void videoReader(cv::VideoCapture & cap);
    void joinWorker() noexcept;
};

class BufferedVideoWriter
{
public:
    /// @throws std::invalid_argument if bufferSize < 1
    explicit BufferedVideoWriter(int bufferSize);
    ~BufferedVideoWriter();
    /// Starts the writer thread on video (which must outlive the writer or stop()).
    /// @throws std::logic_error if the writer is already running (call stop() first)
    void start(cv::VideoWriter & video);
    /// Flushes the queue, joins the writer thread and releases the VideoWriter.
    /// @throws the worker's exception if write() has not yet delivered it
    void stop();
    /// Enqueues a deep copy of frame.
    /// @throws std::logic_error if not started; the worker's exception once; std::runtime_error after a failure
    void write(const cv::Mat & frame);
    /// Test seam: called on the worker thread before each frame is written. Must be called before start().
    void setFaultHookForTesting(std::function<void()> hook);
private:
    std::size_t writeBufferSize;
    std::queue<cv::Mat> writeBuffer;
    std::mutex writeBufferMutex;
    std::atomic_bool writeFinished;
    bool writeFailed = false;                       ///< guarded by writeBufferMutex
    std::exception_ptr workerError;                 ///< guarded by writeBufferMutex
    std::condition_variable writeBufferNotEmpty;
    std::condition_variable writeBufferNotFull;
    std::thread videoWriterThread;
    cv::VideoWriter *pVideo;
    std::function<void()> faultHook;
    void videoWriter(cv::VideoWriter & video);
    void joinWorker() noexcept;
};

class BufferedImageWriter
{
public:
    /// @throws std::invalid_argument if bufferSize < 1
    explicit BufferedImageWriter(int bufferSize);
    ~BufferedImageWriter();
    /// @throws std::logic_error if already running
    void start();
    /// Flushes the queue and joins the writer thread.
    /// @throws the worker's exception if write() has not yet delivered it
    void stop();
    /// @throws std::logic_error if not started; the worker's exception once; std::runtime_error after a failure
    void write(const std::string & path, const cv::Mat & image);
    /// Test seam: called on the worker thread before each image is written. Must be called before start().
    void setFaultHookForTesting(std::function<void()> hook);
private:
    struct ImageWriteTask
    {
        std::string path;
        cv::Mat image;
    };
    std::size_t writeBufferSize;
    std::queue<ImageWriteTask> writeBuffer;
    std::mutex writeBufferMutex;
    std::atomic_bool writeFinished;
    bool writeFailed = false;                       ///< guarded by writeBufferMutex
    std::exception_ptr workerError;                 ///< guarded by writeBufferMutex
    std::condition_variable writeBufferNotEmpty;
    std::condition_variable writeBufferNotFull;
    std::thread writerThread;
    std::function<void()> faultHook;
    void imageWriter();
    void joinWorker() noexcept;
};

#endif
