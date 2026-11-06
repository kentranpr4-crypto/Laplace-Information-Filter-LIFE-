// End-to-end synthetic check of SlamPointPipeline: a textured plane viewed by a camera translating sideways.
// Frames are exact plane homographies H = K (I - t n^T / d) K^{-1} of a texture canvas (no noise), so the feature
// tracks are known. Monocular scale is not observable: the check is on direction and a loose scale ratio.
//
// Run time: doctest re-executes the GIVEN/WHEN blocks for every THEN, so the 20-frame sequence is run once and its
// result is cached (see runSequence). The map capacity of the test configuration is 30 landmarks (the production
// default is 60): the update cost is cubic in the support 6 + 3N. Debug run time of this test: 54 s (sequence re-run
// per THEN, capacity 60) -> 12 s (sequence run once, capacity 40) -> 5 s (capacity 30), same assertions.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <doctest/doctest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include "../../src/Camera.h"
#include "../../src/SlamPointPipeline.h"

namespace
{

cv::Mat syntheticCanvas(int w, int h, unsigned seed)
{
    cv::RNG rng(seed);
    cv::Mat img(h, w, CV_8UC1);
    rng.fill(img, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    for (int k = 0; k < 900; ++k)
    {
        const int x = static_cast<int>(rng.uniform(0, w - 12));
        const int y = static_cast<int>(rng.uniform(0, h - 12));
        cv::rectangle(img, cv::Rect(x, y, 4 + k % 9, 3 + k % 7), cv::Scalar(static_cast<double>(rng.uniform(0, 255))), -1);
    }
    return img;
}

/// Everything the SCENARIO checks: the synthetic camera, the frames and the pipeline output after the whole sequence.
struct SequenceRun
{
    Camera camera;
    int nFrames = 0;
    double stepM = 0.0;
    std::vector<slam::SlamFrameStats> stats;
    std::vector<Eigen::Vector3d> trajectory;
    Eigen::MatrixXd sqrtInfo;
    Eigen::VectorXd mean;
    Eigen::Matrix3d positionSqrtCov;
    bool estimatorFinite = false;
};

/// Runs the 20-frame sideways-translation sequence once (process-wide cache; see the file comment).
const SequenceRun& runSequence()
{
    static const SequenceRun cached = []
    {
        // Small images keep the Debug unit-test run short; the release benchmark uses 1920 x 1080 video.
        const Eigen::Matrix3d kMat = (Eigen::Matrix3d() << 250.0, 0.0, 160.0, 0.0, 250.0, 120.0, 0.0, 0.0, 1.0).finished();
        const double depth = 2.0;
        const int nFrames = 20;
        const double stepM = 0.012;   // camera +x (= body +y, the right-hand direction) per frame
        const cv::Mat canvas = syntheticCanvas(960, 720, 9001);
        cv::Mat k(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                k.at<double>(r, c) = kMat(r, c);
            }
        }
        cv::Mat kInv = k.inv();
        cv::Mat canvasToFrame0 = cv::Mat(cv::Matx33d(1, 0, -320, 0, 1, -240, 0, 0, 1));

        std::vector<cv::Mat> frames;
        for (int i = 0; i < nFrames; ++i)
        {
            const double s = stepM * static_cast<double>(i);
            // Homography from frame-0 pixels to frame-i pixels for the plane z = depth in the start camera frame.
            cv::Mat tn = cv::Mat(cv::Vec3d(s, 0.0, 0.0));
            cv::Mat nT = cv::Mat(cv::Matx13d(0.0, 0.0, 1.0));
            cv::Mat inner = cv::Mat::eye(3, 3, CV_64F) - (tn * nT) / depth;
            cv::Mat hI = k * inner * kInv;
            cv::Mat m = hI * canvasToFrame0;
            cv::Mat frame;
            cv::warpPerspective(canvas, frame, m, cv::Size(320, 240), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            frames.push_back(frame);
        }

        SequenceRun out{Camera(320, 240, kMat, {}), nFrames, stepM, {}, {}, {}, {}, {}, false};
        slam::SlamConfig cfg{};
        cfg.maxLandmarks = 30;   // test capacity (see the file comment); the production default is 60
        slam::SlamPointPipeline pipe(out.camera, cfg);
        for (int i = 0; i < nFrames; ++i)
        {
            out.stats.push_back(pipe.step(frames[static_cast<std::size_t>(i)], i, i / 30.0));
        }
        out.trajectory = pipe.trajectory();
        out.sqrtInfo = pipe.estimator().state().sqrtInfo();
        out.mean = pipe.estimator().mean();
        out.positionSqrtCov = pipe.positionSqrtCov();
        out.estimatorFinite = pipe.estimator().mean().allFinite() && pipe.estimator().state().sqrtInfo().allFinite()
                              && out.positionSqrtCov.allFinite();
        for (const Eigen::Vector3d& p : out.trajectory)
        {
            out.estimatorFinite = out.estimatorFinite && p.allFinite();
        }
        return out;
    }();
    return cached;
}

} // namespace

SCENARIO("SlamPointPipeline: a sideways-translating camera over a textured plane is tracked and mapped")
{
    GIVEN("a 320 x 240 pinhole camera, a fronto-parallel plane at 2 m, and a 20-frame sideways motion of 1.2 cm per frame")
    {
        WHEN("every frame is processed by the pipeline (run once; the result is shared by the THEN blocks)")
        {
            const SequenceRun& run = runSequence();
            // Delayed (parallax-gated) initialisation: a landmark exists only after its triangulation, so the matches are
            // counted from the first birth onwards (before it, the frames cannot have matches by construction).
            int framesWithMatches = 0;
            int framesAfterFirstBirth = 0;
            int firstBirthFrame = -1;
            int maxLandmarks = 0;
            for (std::size_t i = 0; i < run.stats.size(); ++i)
            {
                const slam::SlamFrameStats& st = run.stats[i];
                if (firstBirthFrame < 0 && st.births > 0)
                {
                    firstBirthFrame = static_cast<int>(i);
                }
                if (firstBirthFrame >= 0 && static_cast<int>(i) >= firstBirthFrame)
                {
                    ++framesAfterFirstBirth;
                    if (st.associated > 0)
                    {
                        ++framesWithMatches;
                    }
                }
                maxLandmarks = std::max(maxLandmarks, st.landmarks);
                CHECK((!st.updated || std::isfinite(st.nis)));
            }
            THEN("landmarks are born and matched on most frames after the first birth")
            {
                CHECK(maxLandmarks >= 10);
                REQUIRE(firstBirthFrame >= 0);
                CAPTURE(firstBirthFrame);
                CHECK(framesWithMatches >= framesAfterFirstBirth - 3);
            }
            THEN("the estimated camera moves sideways (+y), dominating the other axes")
            {
                REQUIRE(run.trajectory.size() == static_cast<std::size_t>(run.nFrames));
                const Eigen::Vector3d d = run.trajectory.back() - run.trajectory.front();
                CAPTURE(d.transpose());
                CHECK(d.y() > 0.0);
                CHECK(std::abs(d.y()) > 2.0 * std::abs(d.x()));
                CHECK(std::abs(d.y()) > 2.0 * std::abs(d.z()));
            }
            THEN("the scale is within a factor of 3 of the truth (the depth prior sets the scale: ASSUMED, not observed)")
            {
                const double truth = run.stepM * static_cast<double>(run.nFrames - 1);
                const double est = run.trajectory.back().y() - run.trajectory.front().y();
                CAPTURE(est);
                CHECK(est / truth > 1.0 / 3.0);
                CHECK(est / truth < 3.0);
            }
            THEN("the state, the covariance factors and the trajectory are finite")
            {
                CHECK(run.mean.allFinite());
                CHECK(run.sqrtInfo.allFinite());
                CHECK(run.positionSqrtCov.allFinite());
                CHECK(run.estimatorFinite);
                for (const Eigen::Vector3d& p : run.trajectory)
                {
                    CHECK(p.allFinite());
                }
            }
        }
    }
}
