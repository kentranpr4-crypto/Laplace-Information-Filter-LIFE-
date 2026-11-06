// Synthetic walking-camera check of the Scenario 6 pipeline (keyframe flow + parallax-gated births).
//
// Scene: a textured box room, x in [0.5, 4.5], y in [-2, 2], z in [0, 2.5] m (world = start camera frame, origin
// at the start body position). Images are rendered by ray casting the six room faces with a speckle texture
// (undistorted pinhole, 320 x 240, f = 400 px). The camera walks 1.6 m sideways (+y) with a gentle yaw sweep of
// +/- 0.08 rad at a fixed height of 1.2 m. The ground truth (camera centres, room faces) is known exactly.
//
// Scale is NOT observable from monocular images (ASSUMED: the velocity random walk and the initial speed sigma set the
// metric scale of the estimate). The check is therefore scale-free on the shape and reports the recovered scale.
//
// Tolerances (fixed before the run, not loosened afterwards):
//   T1  recovered scale s (Umeyama similarity from the estimated to the true camera centres)      in [0.5, 2.0]
//       (the metric scale comes from the ASSUMED speed prior, equal to the true speed of this clock: 0.5 m/s)
//   T2  RMS of the scale-and-translation aligned trajectory error (rotation = identity, see the code)   <= 0.05 x 1.6 m = 0.08 m
//   T3  median distance of the aligned landmarks to the nearest room face                          <= 0.10 m
//   T4  fraction of Model R (pure-rotation) flow updates among frames with a keyframe age >= 3   <= 0.10
//   T5  delayed births >= 10 and landmarks alive at the end >= 10 (the map is not empty)
//   T6  all estimates finite, no MAP failures
// Run time: the sequence is run once (cached) because doctest re-runs the test case for every THEN block.
// Test capacity: 30 landmarks (production 60), see runWalk().
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "../../src/Camera.h"
#include "../../src/Pose.hpp"
#include "../../src/SlamPointPipeline.h"
#include "../../src/rotation.hpp"

namespace
{

constexpr double kRoomLo[3] = {0.5, -2.0, 0.0};
constexpr double kRoomHi[3] = {4.5, 2.0, 2.5};
constexpr double kTexPxPerM = 150.0;
constexpr int kFrames = 30;
constexpr double kWalkM = 1.6;
// Frame clock: the walk takes 3.2 s, i.e. the true speed is 0.5 m/s, equal to the ASSUMED speed prior of the
// configuration. The check therefore verifies that the prior anchors the scale (T1); it does not verify that 0.5 m/s is
// the speed of the real video. Per-frame displacement (5 cm) is the same as at 30 fps.
constexpr double kWalkDurationS = 3.2;
constexpr double kFrameDtS = kWalkDurationS / (kFrames - 1);

Camera walkingCamera()
{
    const Eigen::Matrix3d k = (Eigen::Matrix3d() << 400.0, 0.0, 159.5, 0.0, 400.0, 119.5, 0.0, 0.0, 1.0).finished();
    Camera cam(320, 240, k, {});
    const Eigen::Matrix3d rbc = (Eigen::Matrix3d() << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0).finished();
    cam.setBodyToCameraExtrinsics(rbc, Eigen::Vector3d::Zero());
    return cam;
}

cv::Mat speckleTexture()
{
    cv::RNG rng(4242);
    cv::Mat img(512, 512, CV_8UC1);
    rng.fill(img, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0.8);
    for (int k = 0; k < 1200; ++k)
    {
        const int x = static_cast<int>(rng.uniform(0, 500));
        const int y = static_cast<int>(rng.uniform(0, 500));
        cv::rectangle(img, cv::Rect(x, y, 4 + k % 9, 3 + k % 7), cv::Scalar(static_cast<double>(rng.uniform(0, 255))), -1);
    }
    return img;
}

/// Ray-cast rendering of the room faces (nearest positive hit, speckle texture on each face). The inner loop uses plain
/// doubles: Eigen objects are very slow in unoptimised (Debug) builds and this loop runs 77k times per frame.
cv::Mat renderRoom(const Camera& cam, const Eigen::Vector3d& centre, const Eigen::Matrix3d& rwc, const cv::Mat& tex)
{
    cv::Mat img(cam.imageHeight(), cam.imageWidth(), CV_8UC1, cv::Scalar(0));
    const double fx = cam.fx();
    const double fy = cam.fy();
    const double o[3] = {centre.x(), centre.y(), centre.z()};
    for (int v = 0; v < img.rows; ++v)
    {
        for (int u = 0; u < img.cols; ++u)
        {
            const double dc[3] = {(u - cam.cx()) / fx, (v - cam.cy()) / fy, 1.0};
            double d[3];
            for (int a = 0; a < 3; ++a)
            {
                d[a] = rwc(a, 0) * dc[0] + rwc(a, 1) * dc[1] + rwc(a, 2) * dc[2];
            }
            double bestT = std::numeric_limits<double>::infinity();
            int bestAxis = -1;
            double hit[3] = {0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a)
            {
                if (std::abs(d[a]) < 1e-12)
                {
                    continue;
                }
                for (int side = 0; side < 2; ++side)
                {
                    const double plane = side == 0 ? kRoomLo[a] : kRoomHi[a];
                    const double t = (plane - o[a]) / d[a];
                    if (!(t > 1e-9) || t >= bestT)
                    {
                        continue;
                    }
                    double p[3];
                    bool inside = true;
                    for (int b = 0; b < 3; ++b)
                    {
                        p[b] = o[b] + t * d[b];
                        if (b != a && (p[b] < kRoomLo[b] - 1e-9 || p[b] > kRoomHi[b] + 1e-9))
                        {
                            inside = false;
                        }
                    }
                    if (inside)
                    {
                        bestT = t;
                        bestAxis = a;
                        hit[0] = p[0];
                        hit[1] = p[1];
                        hit[2] = p[2];
                    }
                }
            }
            if (bestAxis < 0)
            {
                continue;
            }
            const int b = (bestAxis + 1) % 3;
            const int c = (bestAxis + 2) % 3;
            const int tu = static_cast<int>(std::floor(hit[b] * kTexPxPerM)) & 511;
            const int tv = static_cast<int>(std::floor(hit[c] * kTexPxPerM)) & 511;
            img.at<std::uint8_t>(v, u) = tex.at<std::uint8_t>(tv, tu);
        }
    }
    return img;
}

/// Camera centre and body Euler angles of frame i (ground truth).
void truthOf(int i, Eigen::Vector3d& centre, Eigen::Vector3d& theta)
{
    const double s = static_cast<double>(i) / static_cast<double>(kFrames - 1);
    centre = Eigen::Vector3d(1.2, -0.8 + kWalkM * s, 1.2);
    theta = Eigen::Vector3d(0.0, 0.0, 0.08 * std::sin(std::numbers::pi * s));
}

/// Scale and translation alignment dst ~ s src + t with the rotation fixed to the identity.
/// The estimator world is gauge-fixed to the start body frame (initialPoseSigma), whose orientation equals the true
/// start camera orientation; the walk is nearly collinear, so a free rotation (Umeyama) is not determined by the
/// trajectory and would be arbitrary. The scale is the least-squares scale of the centred trajectories.
struct Similarity
{
    double s = 1.0;
    Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

Similarity gaugeFixedAlignment(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst)
{
    const double n = static_cast<double>(src.size());
    Eigen::Vector3d ms = Eigen::Vector3d::Zero();
    Eigen::Vector3d md = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        ms += src[i];
        md += dst[i];
    }
    ms /= n;
    md /= n;
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        num += (dst[i] - md).dot(src[i] - ms);
        den += (src[i] - ms).squaredNorm();
    }
    Similarity out;
    out.s = den > 0.0 ? num / den : 0.0;
    out.t = md - out.s * ms;
    return out;
}

/// Distance of a point to the nearest of the six room faces (zero on the walls).
double distanceToRoomSurface(const Eigen::Vector3d& p)
{
    double d = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 3; ++a)
    {
        d = std::min(d, std::abs(p(a) - kRoomLo[a]));
        d = std::min(d, std::abs(p(a) - kRoomHi[a]));
    }
    return d;
}

struct WalkRun
{
    std::vector<slam::SlamFrameStats> stats;
    std::vector<Eigen::Vector3d> trajectory;
    std::vector<Eigen::Vector3d> truth;
    std::vector<Eigen::Vector3d> landmarks;
    int mapFailures = 0;
    bool finite = false;
};

const WalkRun& runWalk()
{
    static const WalkRun cached = []
    {
        WalkRun out;
        const Camera cam = walkingCamera();
        const cv::Mat tex = speckleTexture();
        // Test capacity 30 landmarks (production default 60): the MAP cost is cubic in the state, and this file runs in
        // unoptimised Debug builds within the 5 s budget of a unit test. All other settings are production defaults.
        slam::SlamConfig cfg{};
        cfg.maxLandmarks = 30;
        slam::SlamPointPipeline pipe(cam, cfg);
        for (int i = 0; i < kFrames; ++i)
        {
            Eigen::Vector3d centre;
            Eigen::Vector3d theta;
            truthOf(i, centre, theta);
            out.truth.push_back(centre);
            const Eigen::Matrix3d rwc = rigid::cameraRotationInWorld(theta, cam.Rbc());
            const cv::Mat frame = renderRoom(cam, centre, rwc, tex);
            out.stats.push_back(pipe.step(frame, i, kFrameDtS * i));
        }
        out.trajectory = pipe.trajectory();
        out.mapFailures = pipe.mapFailures();
        const Eigen::VectorXd mu = pipe.estimator().mean();
        for (const slam::SlamLandmarkRecord& rec : pipe.map().records())
        {
            out.landmarks.push_back(mu.segment<3>(pipe.model().landmarkOffset(rec.id)));
        }
        out.finite = mu.allFinite() && pipe.estimator().state().sqrtInfo().allFinite();
        for (const Eigen::Vector3d& p : out.trajectory)
        {
            out.finite = out.finite && p.allFinite();
        }
        return out;
    }();
    return cached;
}

} // namespace

SCENARIO("Scenario 6 pipeline recovers the walking-camera shape and a 3-D room map up to scale")
{
    GIVEN("a textured box room (4 x 4 x 2.5 m) and a camera walking 1.6 m sideways over 30 frames")
    {
        WHEN("the sequence is processed by SlamPointPipeline")
        {
            const WalkRun& run = runWalk();
            REQUIRE(run.trajectory.size() == static_cast<std::size_t>(kFrames));
            const Similarity sim = gaugeFixedAlignment(run.trajectory, run.truth);
            CAPTURE(sim.s);

            THEN("T6: the estimate is finite and no MAP step failed")
            {
                CHECK(run.finite);
                CHECK(run.mapFailures == 0);
            }
            THEN("T1: the recovered scale is within a factor of two of the ASSUMED metric scale")
            {
                CHECK(sim.s > 0.5);
                CHECK(sim.s < 2.0);
            }
            THEN("T2: the similarity-aligned trajectory follows the true walk")
            {
                double sq = 0.0;
                for (std::size_t i = 0; i < run.trajectory.size(); ++i)
                {
                    const Eigen::Vector3d e = sim.s * sim.r * run.trajectory[i] + sim.t - run.truth[i];
                    sq += e.squaredNorm();
                }
                const double rms = std::sqrt(sq / static_cast<double>(run.trajectory.size()));
                CAPTURE(rms);
                CHECK(rms <= 0.05 * kWalkM);
            }
            THEN("T3: the aligned landmarks lie on the room faces (median distance <= 0.10 m)")
            {
                std::vector<double> dist;
                for (const Eigen::Vector3d& p : run.landmarks)
                {
                    dist.push_back(distanceToRoomSurface(sim.s * sim.r * p + sim.t));
                }
                REQUIRE(!dist.empty());
                std::nth_element(dist.begin(), dist.begin() + static_cast<std::ptrdiff_t>(dist.size() / 2), dist.end());
                const double median = dist[dist.size() / 2];
                CAPTURE(median);
                CHECK(median <= 0.10);
            }
            THEN("T4: no pure-rotation absorption: Model R is rare once the keyframe has a baseline")
            {
                int used = 0;
                int rotation = 0;
                for (const slam::SlamFrameStats& st : run.stats)
                {
                    if (st.flowUpdated && st.keyframeAge >= 3 && !st.keyframeStarted)
                    {
                        ++used;
                        rotation += st.flowRotationModel ? 1 : 0;
                    }
                }
                CAPTURE(used);
                CAPTURE(rotation);
                REQUIRE(used > 0);
                CHECK(static_cast<double>(rotation) / static_cast<double>(used) <= 0.10);
            }
            THEN("T5: delayed births populate the map")
            {
                int births = 0;
                for (const slam::SlamFrameStats& st : run.stats)
                {
                    births += st.births;
                }
                CAPTURE(births);
                CHECK(births >= 10);
                CHECK(static_cast<int>(run.landmarks.size()) >= 10);
            }
        }
    }
}
