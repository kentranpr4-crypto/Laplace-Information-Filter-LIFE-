#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>   // test-only explicit-inverse reference (NIS check)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include "../../src/Camera.h"
#include "../../src/CoreLinalg.h"
#include "../../src/ImageFlow.h"
#include "../../src/MeasurementOutdoorFlowBundle.h"
#include "../../src/Pose.hpp"
#include "../../src/ScenarioRunner.h"
#include "../../src/Scenario4Runner.h"
#include "../../src/VisualHorizon.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;

namespace
{

// ----- Synthetic flight used by the Scenario 4 tests -----------------------------------------------------------
//
// Rendering model. The ground is the plane z = kGroundPlaneZ (world z down) with a smooth value-noise colour texture;
// the camera has a fixed attitude (pitch 0.05 rad) and moves along the world x axis. Every output pixel is the
// average of kSuper x kSuper sub-samples of the texture (area integration, as a real sensor does). Point sampling
// (one ray per pixel) was tried first and is NOT used: far-field texture is finer than a pixel footprint, so point
// samples alias, the aliasing changes with sub-pixel motion and Lucas-Kanade then under-measures the displacement
// (slope 0.29 at full resolution, measured; 0.93 with area integration). The sub-samples are evaluated on a world
// texture mosaic and resampled with warpPerspective (ground-plane homography), which is fast enough for Debug tests.

constexpr int kWidth = 480;   ///< f/W = 0.83 as the DJI camera (2236/2688): same field of view and pixel scale
constexpr int kHeight = 360;
constexpr double kFps = 30.0;
constexpr double kGroundPlaneZ = 60.0;     ///< world z (down) of the ground plane = initial AGL of 60 m
constexpr int kFrames = 60;                ///< 2 s of flight
constexpr int kSuper = 4;                  ///< sub-samples per pixel and axis (area integration)
constexpr double kTexel = 0.15;            ///< texture mosaic resolution [m]
constexpr double kMosaicX0 = -30.0;        ///< mosaic extent: forward [m]
constexpr double kMosaicX1 = 200.0;
constexpr double kMosaicY0 = -110.0;       ///< mosaic extent: lateral [m]
constexpr double kMosaicY1 = 110.0;
const cv::Scalar kTextureMeanBgr(90, 140, 105);   ///< colour of texture value h = 0.5 (outside the mosaic)

Camera syntheticCamera()
{
    Eigen::Matrix3d K;
    K << 400.0, 0.0, 240.0,
         0.0, 400.0, 180.0,
         0.0, 0.0, 1.0;
    return Camera(kWidth, kHeight, K, {});
}

/// Truth: constant forward speed 6 m/s (north), climb 0.2 m/s, level attitude with a 0.05 rad pitch.
Vector3d truthPosition(double t)
{
    return Vector3d(6.0 * t, 0.0, -0.2 * t);
}

Vector3d truthAngles()
{
    return Vector3d(0.0, 0.05, 0.0);
}

/// Deterministic hash of an integer lattice point to [0, 1).
double latticeValue(int ix, int iy)
{
    std::uint32_t h = static_cast<std::uint32_t>(ix) * 374761393u + static_cast<std::uint32_t>(iy) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return static_cast<double>(h) / 4294967296.0;
}

/// Smooth value noise on a lattice of spacing `cell` [m] (continuous, so that Lucas-Kanade has gradients everywhere).
double smoothNoise(double x, double y, double cell)
{
    const double gx = x / cell;
    const double gy = y / cell;
    const double fx0 = std::floor(gx);
    const double fy0 = std::floor(gy);
    const int ix = static_cast<int>(fx0);
    const int iy = static_cast<int>(fy0);
    const double tx = gx - fx0;
    const double ty = gy - fy0;
    const double sx = tx * tx * (3.0 - 2.0 * tx);
    const double sy = ty * ty * (3.0 - 2.0 * ty);
    const double v00 = latticeValue(ix, iy);
    const double v10 = latticeValue(ix + 1, iy);
    const double v01 = latticeValue(ix, iy + 1);
    const double v11 = latticeValue(ix + 1, iy + 1);
    return (1.0 - sy) * ((1.0 - sx) * v00 + sx * v10) + sy * ((1.0 - sx) * v01 + sx * v11);
}

/// Ground texture (BGR) at world ground coordinates (x north, y east): two octaves of smooth value noise.
cv::Vec3b groundColour(double x, double y)
{
    const double h = 0.6 * smoothNoise(x, y, 2.0) + 0.4 * smoothNoise(x, y, 0.7);
    return cv::Vec3b(static_cast<std::uint8_t>(60 + 60 * h), static_cast<std::uint8_t>(100 + 80 * h),
                     static_cast<std::uint8_t>(80 + 50 * h));
}

cv::Vec3b skyColour(int v)
{
    const double f = static_cast<double>(v) / kHeight;
    return cv::Vec3b(static_cast<std::uint8_t>(225 - 30 * f), static_cast<std::uint8_t>(215 - 30 * f),
                     static_cast<std::uint8_t>(205 - 20 * f));
}

/// Texture mosaic on the world grid (built once): texel (row i, col j) = groundColour(x0 + j d, y0 + i d).
const cv::Mat& groundMosaic()
{
    static const cv::Mat mosaic = []
    {
        const int cols = static_cast<int>(std::ceil((kMosaicX1 - kMosaicX0) / kTexel)) + 1;
        const int rows = static_cast<int>(std::ceil((kMosaicY1 - kMosaicY0) / kTexel)) + 1;
        cv::Mat m(rows, cols, CV_8UC3);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                m.at<cv::Vec3b>(i, j) = groundColour(kMosaicX0 + j * kTexel, kMosaicY0 + i * kTexel);
            }
        }
        return m;
    }();
    return mosaic;
}

/// Sky sub-sample mask and sky colours on the supersampled grid. The attitude is constant in this scene, so the
/// horizon is fixed and the layers are built once for the synthetic camera.
struct SkyLayers
{
    cv::Mat mask;     ///< CV_8UC1, 255 where the sub-sample ray is above the horizon
    cv::Mat colour;   ///< CV_8UC3, sky colour of the sub-sample row
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

SkyLayers buildSkyLayers(const Camera& camera)
{
    SkyLayers out;
    out.fx = camera.fx();
    out.fy = camera.fy();
    out.cx = camera.cx();
    out.cy = camera.cy();
    const Eigen::Matrix3d Rnc = rigid::cameraRotationInWorld(truthAngles(), camera.Rbc());
    const double a0 = Rnc(2, 0);
    const double a1 = Rnc(2, 1);
    const double a2 = Rnc(2, 2);
    const int W = kWidth * kSuper;
    const int H = kHeight * kSuper;
    out.mask = cv::Mat(H, W, CV_8UC1, cv::Scalar(0));
    out.colour = cv::Mat(H, W, CV_8UC3);
    const double s = 1.0 / kSuper;
    for (int V = 0; V < H; ++V)
    {
        // Full-resolution pixel row of this sub-sample (sub-sample centres, OpenCV pixel convention).
        const double fullV = (V + 0.5) * s - 0.5;
        const double ny = (fullV - camera.cy()) / camera.fy();
        const cv::Vec3b rowColour = skyColour(V / kSuper);
        for (int U = 0; U < W; ++U)
        {
            const double fullU = (U + 0.5) * s - 0.5;
            const double nx = (fullU - camera.cx()) / camera.fx();
            if (a0 * nx + a1 * ny + a2 <= 1e-9)
            {
                out.mask.at<std::uint8_t>(V, U) = 255;
            }
            out.colour.at<cv::Vec3b>(V, U) = rowColour;
        }
    }
    return out;
}

const SkyLayers& skyLayers(const Camera& camera)
{
    static const SkyLayers layers = buildSkyLayers(camera);
    if (layers.fx != camera.fx() || layers.fy != camera.fy() || layers.cx != camera.cx() || layers.cy != camera.cy())
    {
        throw std::logic_error("synthetic renderer: the sky layers were built for a different camera");
    }
    return layers;
}

/**
 * @brief Renders the truth view at time t (area-integrated, see the comment at the top of the file).
 *
 * Ground points are mapped by the plane homography G = S Hn Kinv Sup: Hn maps (n_x, n_y, 1) of the camera ray to
 * (X0, X1, X2) with P_x = X0 / X2 and P_y = X1 / X2 on z = kGroundPlaneZ; S maps world ground coordinates to mosaic
 * texel coordinates; Sup maps supersampled pixel indices to full-resolution pixels.
 */
cv::Mat renderFrame(const Camera& camera, double t)
{
    const SkyLayers& sky = skyLayers(camera);
    const Vector3d r = truthPosition(t);
    const Eigen::Matrix3d Rnc = rigid::cameraRotationInWorld(truthAngles(), camera.Rbc());
    const Eigen::Vector3d rowX = Rnc.row(0).transpose();
    const Eigen::Vector3d rowY = Rnc.row(1).transpose();
    const Eigen::Vector3d rowZ = Rnc.row(2).transpose();
    Eigen::Matrix3d Hn;
    Hn.row(0) = (r.x() * rowZ + (kGroundPlaneZ - r.z()) * rowX).transpose();
    Hn.row(1) = (r.y() * rowZ + (kGroundPlaneZ - r.z()) * rowY).transpose();
    Hn.row(2) = rowZ.transpose();
    Eigen::Matrix3d Kinv;
    Kinv << 1.0 / camera.fx(), 0.0, -camera.cx() / camera.fx(),
            0.0, 1.0 / camera.fy(), -camera.cy() / camera.fy(),
            0.0, 0.0, 1.0;
    const double s = 1.0 / kSuper;
    Eigen::Matrix3d sup;
    sup << s, 0.0, 0.5 * s - 0.5,
           0.0, s, 0.5 * s - 0.5,
           0.0, 0.0, 1.0;
    Eigen::Matrix3d S;
    S << 1.0 / kTexel, 0.0, -kMosaicX0 / kTexel,
         0.0, 1.0 / kTexel, -kMosaicY0 / kTexel,
         0.0, 0.0, 1.0;
    const Eigen::Matrix3d Hmap = S * Hn * Kinv * sup;   // supersampled pixel -> mosaic texel (inverse map)
    cv::Mat hm(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            hm.at<double>(i, j) = Hmap(i, j);
        }
    }
    cv::Mat big;
    cv::warpPerspective(groundMosaic(), big, hm, cv::Size(kWidth * kSuper, kHeight * kSuper),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT, kTextureMeanBgr);
    sky.colour.copyTo(big, sky.mask);
    cv::Mat out;
    cv::resize(big, out, cv::Size(kWidth, kHeight), 0.0, 0.0, cv::INTER_AREA);
    return out;
}

/// Ground-only mask (255 where the true view ray hits the ground), used as the tracker's detection mask.
cv::Mat groundMask(const Camera& camera)
{
    const Eigen::Matrix3d Rnc = rigid::cameraRotationInWorld(truthAngles(), camera.Rbc());
    cv::Mat m(kHeight, kWidth, CV_8UC1, cv::Scalar(0));
    for (int v = 0; v < kHeight; ++v)
    {
        for (int u = 0; u < kWidth; ++u)
        {
            const Eigen::Vector2d n = camera.pixelToNormalised(Eigen::Vector2d(u, v));
            const Vector3d dn = Rnc * Vector3d(n(0), n(1), 1.0);
            if (dn.z() > 1e-3)
            {
                m.at<std::uint8_t>(v, u) = 255;
            }
        }
    }
    return m;
}

/// Ground point seen at pixel p (full resolution) at time t, from the true pose.
Vector3d groundPointAt(const Camera& camera, double t, const cv::Point2d& p)
{
    const Vector3d r = truthPosition(t);
    const Eigen::Matrix3d Rnc = rigid::cameraRotationInWorld(truthAngles(), camera.Rbc());
    const Eigen::Vector2d n = camera.pixelToNormalised(Eigen::Vector2d(p.x, p.y));
    const Vector3d dn = Rnc * Vector3d(n(0), n(1), 1.0);
    const double lambda = (kGroundPlaneZ - r.z()) / dn.z();
    return r + lambda * dn;
}

/// Pixel of a world point in the camera at time t, from the true pose.
Eigen::Vector2d projectAt(const Camera& camera, double t, const Vector3d& p)
{
    const Vector3d pc = rigid::worldToCamera(p, truthPosition(t), truthAngles(), camera.Rbc(), Vector3d::Zero());
    return camera.vectorToPixel(pc);
}

/// Training patches at the full frame resolution (the runner rescales them to its analysis resolution).
std::vector<cv::Mat> skyPatches()
{
    return {cv::Mat(90, 160, CV_8UC3, cv::Scalar(225, 215, 205))};
}

std::vector<cv::Mat> groundPatches()
{
    cv::Mat a(90, 160, CV_8UC3);
    cv::Mat b(90, 160, CV_8UC3);
    for (int y = 0; y < a.rows; ++y)
    {
        for (int x = 0; x < a.cols; ++x)
        {
            a.at<cv::Vec3b>(y, x) = groundColour(x / 4.0, y / 4.0);
            b.at<cv::Vec3b>(y, x) = groundColour(x / 4.0 + 50.0, y / 4.0 + 30.0);
        }
    }
    return {a, b};
}

std::filesystem::path fixture(const std::string& name)
{
    return std::filesystem::path(TEST_DATA_DIR) / "outdoor" / name;
}

Scenario4Options syntheticOptions(const std::string& srt)
{
    Scenario4Options opt;
    opt.srtPath = fixture(srt);
    opt.trainingSky = skyPatches();
    opt.trainingGround = groundPatches();
    return opt;
}

/// The 60 synthetic frames (rendered once and shared by the tests).
const std::vector<cv::Mat>& syntheticFrames()
{
    static const std::vector<cv::Mat> frames = []
    {
        const Camera camera = syntheticCamera();
        std::vector<cv::Mat> out;
        out.reserve(kFrames);
        for (int k = 0; k < kFrames; ++k)
        {
            out.push_back(renderFrame(camera, k / kFps));
        }
        return out;
    }();
    return frames;
}

/// Everything the tests read from one processed synthetic flight.
struct SyntheticRun
{
    std::vector<Vector3d> trajectory;
    std::vector<Scenario4FrameDiagnostics> diagnostics;
    std::vector<Vector3d> validationTrack;
    PlotFrameData plot;
};

SyntheticRun runSynthetic(const std::string& srt)
{
    const Camera camera = syntheticCamera();
    Scenario4Runner runner(syntheticOptions(srt));
    runner.initialise(camera, kFps, cv::Size(kWidth, kHeight), "s4_synthetic.MOV");
    const auto& frames = syntheticFrames();
    for (int k = 0; k < kFrames; ++k)
    {
        runner.processFrame(frames[static_cast<std::size_t>(k)], k, k / kFps);
    }
    SyntheticRun out;
    out.trajectory = runner.trajectory();
    out.diagnostics = runner.diagnostics();
    out.validationTrack = runner.validationTrack();
    out.plot = runner.plotData();
    return out;
}

/// Run with the SRT whose latitude/longitude differ from file B (GPS must not influence the estimate).
const SyntheticRun& syntheticRunA()
{
    static const SyntheticRun run = runSynthetic("s4_synthetic_a.SRT");
    return run;
}

const SyntheticRun& syntheticRunB()
{
    static const SyntheticRun run = runSynthetic("s4_synthetic_b.SRT");
    return run;
}

std::string readFile(const std::filesystem::path& p)
{
    std::ifstream in(p, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("cannot read " + p.string());
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

SCENARIO("Scenario 4 factory returns a runner with the documented name")
{
    GIVEN("the factory used by the frame loop")
    {
        auto runner = makeScenario4Runner();
        THEN("a runner is returned and is named Scenario 4")
        {
            REQUIRE(runner != nullptr);
            CHECK(runner->name() == "Scenario 4: flight");
        }
    }
}

SCENARIO("Scenario 4 never uses flight.SRT latitude or longitude as an estimator input")
{
    GIVEN("two SRT files with identical altitudes and different latitude/longitude, processing the same frames")
    {
        const SyntheticRun& a = syntheticRunA();
        const SyntheticRun& b = syntheticRunB();
        WHEN("the two runs are compared")
        {
            THEN("the estimated trajectories are bitwise identical")
            {
                REQUIRE(a.trajectory.size() == b.trajectory.size());
                for (std::size_t k = 0; k < a.trajectory.size(); ++k)
                {
                    CHECK(a.trajectory[k] == b.trajectory[k]);
                }
            }
            THEN("the per-frame diagnostics (inliers, NIS, updates) are identical")
            {
                REQUIRE(a.diagnostics.size() == b.diagnostics.size());
                for (std::size_t k = 0; k < a.diagnostics.size(); ++k)
                {
                    const auto& da = a.diagnostics[k];
                    const auto& db = b.diagnostics[k];
                    CHECK(da.flowInliers == db.flowInliers);
                    CHECK(da.horizonInliers == db.horizonInliers);
                    CHECK(da.flowUpdated == db.flowUpdated);
                    CHECK(da.horizonUpdated == db.horizonUpdated);
                    CHECK(da.altitudeUpdated == db.altitudeUpdated);
                    CHECK(((std::isnan(da.altitudeNis) && std::isnan(db.altitudeNis)) || da.altitudeNis == db.altitudeNis));
                }
            }
            THEN("the validation-only track does differ, showing the GPS values are read only for validation")
            {
                REQUIRE(!a.validationTrack.empty());
                REQUIRE(a.validationTrack.size() == b.validationTrack.size());
                CHECK((a.validationTrack.back() - b.validationTrack.back()).norm() > 1.0);
            }
        }
    }
}

SCENARIO("Scenario 4 estimator sources contain no GPS access outside the validation block")
{
    GIVEN("the Scenario 4 implementation file")
    {
        const std::string src = readFile(std::filesystem::path(TEST_DATA_DIR) / ".." / ".." / "src" / "Scenario4.cpp");
        const std::size_t begin = src.find("// Validation-only local track");
        const std::size_t end = src.find("// Shared process model and prior");
        REQUIRE(begin != std::string::npos);
        REQUIRE(end != std::string::npos);
        WHEN("every occurrence of the geodetic accessors and fields is located")
        {
            THEN("they all lie inside the validation block")
            {
                for (const std::string token : {"validationOnlyGeodetic", "latitudeDeg", "longitudeDeg"})
                {
                    std::size_t pos = src.find(token);
                    while (pos != std::string::npos)
                    {
                        CAPTURE(token);
                        CHECK(pos >= begin);
                        CHECK(pos < end);
                        pos = src.find(token, pos + 1);
                    }
                }
            }
        }
    }
    GIVEN("the measurement and filter sources used by Scenario 4")
    {
        for (const char* file : {"MeasurementAltimeter.cpp", "MeasurementHorizon.cpp", "MeasurementOutdoorFlowBundle.cpp",
                                 "VisualHorizon.cpp", "ImageFlow.cpp"})
        {
            const std::string src = readFile(std::filesystem::path(TEST_DATA_DIR) / ".." / ".." / "src" / file);
            CAPTURE(file);
            THEN("they never mention latitude or longitude")
            {
                CHECK(src.find("latitude") == std::string::npos);
                CHECK(src.find("longitude") == std::string::npos);
                CHECK(src.find("validationOnly") == std::string::npos);
            }
        }
    }
}

SCENARIO("Scenario 4 synthetic flight: the filter recovers position and altitude without divergence")
{
    GIVEN("a rendered 2 s flight over a textured plane with a visible horizon, processed by the Scenario 4 runner")
    {
        const SyntheticRun& run = syntheticRunA();
        const Vector3d est = run.trajectory.back();
        const Vector3d truth = truthPosition((kFrames - 1) / kFps);
        THEN("the state is finite and the trajectory has one entry per frame")
        {
            CHECK(est.allFinite());
            CHECK(run.trajectory.size() == static_cast<std::size_t>(kFrames));
        }
        THEN("the horizon and flow are used on most frames")
        {
            int flow = 0;
            int hz = 0;
            for (const auto& d : run.diagnostics)
            {
                flow += d.flowUpdated ? 1 : 0;
                hz += d.horizonUpdated ? 1 : 0;
            }
            CHECK(flow >= 0.8 * (kFrames - 1));
            CHECK(hz >= 0.8 * kFrames);
        }
        THEN("the altitude matches the relative SRT altitude to within 0.5 m")
        {
            CAPTURE(-est.z());
            CHECK(std::abs(-est.z() - (-truth.z())) < 0.5);
        }
        THEN("the horizontal displacement is recovered to within 1.5 m")
        {
            CAPTURE(est.x());
            CAPTURE(est.y());
            CHECK(std::abs(est.x() - truth.x()) < 1.5);
            CHECK(std::abs(est.y() - truth.y()) < 1.5);
        }
        THEN("the overlay has the input frame size and the position uncertainty is finite")
        {
            CHECK(run.plot.overlay.size() == cv::Size(kWidth, kHeight));
            CHECK(run.plot.overlay.type() == CV_8UC3);
            CHECK(run.plot.positionSqrtCov.allFinite());
            CHECK(run.plot.trajectory.size() == static_cast<std::size_t>(kFrames));
            CHECK(!run.plot.statusLines.empty());
        }
    }
}

SCENARIO("Scenario 4 synthetic scene: the true-state flow model reproduces the rendered correspondences")
{
    GIVEN("ground pixels of frame 0 and their truth positions in frame 5 (5-frame keyframe baseline)")
    {
        const Camera camera = syntheticCamera();
        const double t0 = 0.0;
        const double t5 = 5.0 / kFps;
        Eigen::VectorXd x = Eigen::VectorXd::Zero(18);
        x.segment<3>(MeasurementOutdoorFlowBundle::kCurrentOffset) = truthPosition(t5);
        x.segment<3>(MeasurementOutdoorFlowBundle::kCurrentOffset + 3) = truthAngles();
        x.segment<3>(MeasurementOutdoorFlowBundle::kPreviousOffset) = truthPosition(t0);
        x.segment<3>(MeasurementOutdoorFlowBundle::kPreviousOffset + 3) = truthAngles();
        std::vector<OutdoorFlowPair> pairs;
        for (int v = 200; v < 350; v += 15)
        {
            for (int u = 20; u < 470; u += 30)
            {
                const Eigen::Vector2d tc = projectAt(camera, t5, groundPointAt(camera, t0, cv::Point2d(u, v)));
                if (!tc.allFinite())
                {
                    continue;
                }
                pairs.push_back(OutdoorFlowPair{camera.pixelToNormalised(Eigen::Vector2d(u, v)), camera.pixelToNormalised(tc)});
            }
        }
        REQUIRE(pairs.size() > 100);
        WHEN("the flow measurement is predicted at the true state")
        {
            MeasurementOutdoorFlowBundle meas(t5, pairs, 1.0 / 400.0, kGroundPlaneZ, camera.Rbc());
            const Eigen::VectorXd g = meas.predict(x, nullptr);
            THEN("the predicted pixels equal the ray-cast truth to numerical precision")
            {
                double maxPx = 0.0;
                for (std::size_t i = 0; i < pairs.size(); ++i)
                {
                    const Eigen::Vector2d gi = g.segment<2>(2 * static_cast<Eigen::Index>(i));
                    maxPx = std::max(maxPx, (gi - pairs[i].currNorm).norm() * camera.fx());
                }
                CHECK(maxPx < 1e-6);
            }
        }
    }
}

SCENARIO("Scenario 4 synthetic scene: Lucas-Kanade recovers the rendered displacement of ground tracks")
{
    GIVEN("six consecutive synthetic frames tracked with the ground-only detection mask")
    {
        const Camera camera = syntheticCamera();
        const auto& frames = syntheticFrames();
        const cv::Mat mask = groundMask(camera);
        ImageFlowOptions opt;
        opt.scale = 1.0;
        ImageFlow flow(opt);
        std::map<int, cv::Point2d> start;
        for (int k = 0; k <= 5; ++k)
        {
            cv::Mat gray;
            cv::cvtColor(frames[static_cast<std::size_t>(k)], gray, cv::COLOR_BGR2GRAY);
            flow.update(gray, mask);
            if (k == 0)
            {
                for (const ImageFlowMatch& tr : flow.activeTracks())
                {
                    start[tr.id] = tr.curr;
                }
            }
        }
        WHEN("each surviving track is compared with the true displacement of its ground point")
        {
            double sxy = 0.0;
            double sxx = 0.0;
            double sumAbsErr = 0.0;
            int n = 0;
            for (const ImageFlowMatch& tr : flow.activeTracks())
            {
                const auto it = start.find(tr.id);
                if (it == start.end())
                {
                    continue;
                }
                const Eigen::Vector2d tc = projectAt(camera, 5.0 / kFps, groundPointAt(camera, 0.0, it->second));
                const Eigen::Vector2d td = tc - Eigen::Vector2d(it->second.x, it->second.y);
                const Eigen::Vector2d ld(tr.curr.x - it->second.x, tr.curr.y - it->second.y);
                sxy += ld.dot(td);
                sxx += td.squaredNorm();
                sumAbsErr += (ld - td).norm();
                ++n;
            }
            THEN("there are enough tracks and the LK displacement is not attenuated (regression slope >= 0.85)")
            {
                REQUIRE(n >= 100);
                const double slope = sxy / sxx;
                CAPTURE(slope);
                CAPTURE(sumAbsErr / n);
                CHECK(slope >= 0.85);
                CHECK(slope <= 1.15);
            }
        }
    }
}

SCENARIO("Scenario 4 synthetic scene: the horizon boundary is unbiased on the rendered frame")
{
    GIVEN("the classifier trained as in the runner and the exact horizon line of the renderer")
    {
        const Camera camera = syntheticCamera();
        std::vector<cv::Mat> sky = skyPatches();
        std::vector<cv::Mat> ground = groundPatches();
        const double scale = 0.25;
        for (auto* set : {&sky, &ground})
        {
            for (cv::Mat& im : *set)
            {
                cv::resize(im, im, cv::Size(), scale, scale, cv::INTER_AREA);
            }
        }
        const SkyGroundClassifier classifier = SkyGroundClassifier::train(sky, ground);
        // Horizon of the renderer: sub-samples with Rnc(2,:) (n, 1) = 0.
        const Eigen::Matrix3d Rnc = rigid::cameraRotationInWorld(truthAngles(), camera.Rbc());
        const Eigen::Vector3d a = Rnc.row(2).transpose();
        const double aNorm = std::sqrt(a(0) * a(0) + a(1) * a(1));
        WHEN("the horizon is extracted from frame 0")
        {
            const cv::Mat& frame = syntheticFrames()[0];
            cv::Mat small;
            cv::resize(frame, small, cv::Size(), scale, scale, cv::INTER_AREA);
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            const HorizonEstimate est = extractHorizon(classifier.logOdds(small), scale, kWidth, {}, gray);
            THEN("the horizon is valid and the boundary points lie on the true line (mean |residual| < 0.25 px)")
            {
                REQUIRE(est.valid);
                double sum = 0.0;
                for (const cv::Point2d& p : est.inliers)
                {
                    const Eigen::Vector2d n = camera.pixelToNormalised(Eigen::Vector2d(p.x, p.y));
                    sum += (a(0) * n(0) + a(1) * n(1) + a(2)) / aNorm * camera.fy();
                }
                const double meanPx = sum / static_cast<double>(est.inliers.size());
                CAPTURE(meanPx);
                CHECK(std::abs(meanPx) < 0.25);
            }
        }
    }
}

SCENARIO("Scenario 4 rejects invalid initialisation and frame sequences")
{
    const Camera camera = syntheticCamera();
    GIVEN("a runner with synthetic options")
    {
        Scenario4Runner runner(syntheticOptions("s4_synthetic_a.SRT"));
        THEN("a frame size different from the camera is rejected")
        {
            CHECK_THROWS_AS(runner.initialise(camera, kFps, cv::Size(kWidth + 2, kHeight), "s4_synthetic.MOV"),
                            std::invalid_argument);
        }
        THEN("a non-positive frame rate is rejected")
        {
            CHECK_THROWS_AS(runner.initialise(camera, 0.0, cv::Size(kWidth, kHeight), "s4_synthetic.MOV"),
                            std::invalid_argument);
        }
        THEN("a processed frame before initialise is rejected")
        {
            Scenario4Runner fresh(syntheticOptions("s4_synthetic_a.SRT"));
            CHECK_THROWS_AS(fresh.processFrame(cv::Mat(kHeight, kWidth, CV_8UC3), 0, 0.0), std::logic_error);
        }
    }
    GIVEN("an initialised runner")
    {
        Scenario4Runner runner(syntheticOptions("s4_synthetic_a.SRT"));
        runner.initialise(camera, kFps, cv::Size(kWidth, kHeight), "s4_synthetic.MOV");
        const cv::Mat frame = syntheticFrames()[0];
        runner.processFrame(frame, 0, 0.0);
        THEN("frames out of order and empty or wrongly sized frames are rejected")
        {
            CHECK_THROWS_AS(runner.processFrame(frame, 0, 0.0), std::invalid_argument);
            CHECK_THROWS_AS(runner.processFrame(cv::Mat(), 1, 1.0 / kFps), std::invalid_argument);
            CHECK_THROWS_AS(runner.processFrame(cv::Mat(10, 10, CV_8UC3), 1, 1.0 / kFps), std::invalid_argument);
        }
    }
    GIVEN("a camera with a non-zero lever arm")
    {
        Camera lever = camera;
        lever.setBodyToCameraExtrinsics(lever.Rbc(), Vector3d(0.1, 0.0, 0.0));
        Scenario4Runner runner(syntheticOptions("s4_synthetic_a.SRT"));
        THEN("initialisation is rejected (the flow model assumes rCBb = 0)")
        {
            CHECK_THROWS_AS(runner.initialise(lever, kFps, cv::Size(kWidth, kHeight), "s4_synthetic.MOV"),
                            std::invalid_argument);
        }
    }
}

SCENARIO("Scenario 4 NIS (R4-11): the QR and triangular-solve evaluation equals the explicit covariance-form NIS")
{
    GIVEN("a random innovation nu (m = 12) with Jacobian B (k = 3) and the record the runner stores for it")
    {
        const MatrixXd b = testutil::randomMatrix(12, 3, 5101);
        const VectorXd nu = testutil::randomVector(12, 5102);
        Scenario4NisRecord rec;
        rec.dof = 12;
        rec.nu2 = nu.squaredNorm();
        rec.rB = core::upperFactor(b);
        rec.c = b.transpose() * nu;
        for (double sigma : {0.05, 0.5, 3.0})
        {
            WHEN("the NIS is evaluated at sigma = " + std::to_string(sigma))
            {
                // Reference (test only): S = sigma^2 I + B B^T and NIS = nu^T S^{-1} nu with an explicit inverse.
                const MatrixXd sCov = sigma * sigma * MatrixXd::Identity(12, 12) + b * b.transpose();
                const double ref = nu.dot(sCov.inverse() * nu);
                THEN("the factor evaluation matches the reference to 1e-9 relative")
                {
                    CAPTURE(sigma);
                    CHECK(scenario4NisAt(rec, sigma) == doctest::Approx(ref).epsilon(1e-9));
                }
            }
        }
        WHEN("sigma is not positive")
        {
            THEN("the evaluation is rejected")
            {
                CHECK_THROWS_AS(scenario4NisAt(rec, 0.0), std::invalid_argument);
            }
        }
    }
}

namespace
{

/// Simulated records whose innovations have covariance sigma^2 I + B B^T exactly (nu = sigma z1 + B z2).
std::vector<Scenario4NisRecord> simulatedRecords(Scenario4NisKind kind, int count, int firstFrame, int dof, int k,
                                                 double sigma, double bScale, unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Scenario4NisRecord> out;
    for (int i = 0; i < count; ++i)
    {
        MatrixXd b(dof, k);
        for (Eigen::Index r = 0; r < dof; ++r)
        {
            for (Eigen::Index c = 0; c < k; ++c)
            {
                b(r, c) = bScale * nd(gen);
            }
        }
        VectorXd nu(dof);
        for (Eigen::Index r = 0; r < dof; ++r)
        {
            nu(r) = sigma * nd(gen);
        }
        for (Eigen::Index c = 0; c < k; ++c)
        {
            const double z = nd(gen);
            nu += z * b.col(c);
        }
        Scenario4NisRecord rec;
        rec.kind = kind;
        rec.frameIndex = firstFrame + i;
        rec.dof = dof;
        rec.nu2 = nu.squaredNorm();
        rec.rB = core::upperFactor(b);
        rec.c = b.transpose() * nu;
        out.push_back(std::move(rec));
    }
    return out;
}

} // namespace

SCENARIO("Scenario 4 calibration (R4-06/R4-07): NIS matching recovers the noise of simulated innovations")
{
    GIVEN("400 flow-like records (dof 40, k 3) simulated with sigma = 0.2 in the first half of the flight")
    {
        std::vector<Scenario4NisRecord> records = simulatedRecords(Scenario4NisKind::Flow, 400, 0, 40, 3, 0.2, 0.3, 5201u);
        WHEN("the second half of the flight (a different sigma, 2.0) and altitude records are added")
        {
            const auto second = simulatedRecords(Scenario4NisKind::Flow, 400, 400, 40, 3, 2.0, 0.3, 5202u);
            const auto alt = simulatedRecords(Scenario4NisKind::Altitude, 400, 0, 1, 1, 0.05, 0.02, 5203u);
            records.insert(records.end(), second.begin(), second.end());
            records.insert(records.end(), alt.begin(), alt.end());
            const Scenario4Calibration cal = scenario4CalibrateSigma(records, Scenario4NisKind::Flow, 0, 400, 1e-3, 10.0);
            THEN("the first-half calibration recovers sigma = 0.2 within 5 %, and the second half is not used")
            {
                CAPTURE(cal.sigma);
                CHECK(cal.converged);
                CHECK(cal.records == 400);
                CHECK(cal.sigma == doctest::Approx(0.2).epsilon(0.05));
                CHECK(cal.medianPit == doctest::Approx(0.5).epsilon(1e-6));
            }
            THEN("the calibration is reproducible")
            {
                const Scenario4Calibration again = scenario4CalibrateSigma(records, Scenario4NisKind::Flow, 0, 400, 1e-3, 10.0);
                CHECK(again.sigma == cal.sigma);
            }
        }
    }
    GIVEN("a scalar altimeter-like family (dof 1, k 1): the calibrated sigma is the noise over the known prior spread")
    {
        const auto records = simulatedRecords(Scenario4NisKind::Altitude, 600, 0, 1, 1, 0.05, 0.03, 5301u);
        WHEN("it is calibrated on all frames")
        {
            const Scenario4Calibration cal = scenario4CalibrateSigma(records, Scenario4NisKind::Altitude, 0, 1000, 1e-3, 1.0);
            THEN("sigma is recovered to 8 % (one-dimensional NIS with 600 samples)")
            {
                CAPTURE(cal.sigma);
                CHECK(cal.converged);
                CHECK(cal.sigma == doctest::Approx(0.05).epsilon(0.08));
            }
        }
    }
    GIVEN("records whose NIS stays far above the chi-squared reference over the whole bracket")
    {
        const auto records = simulatedRecords(Scenario4NisKind::Flow, 50, 0, 40, 3, 1000.0, 0.3, 5401u);
        WHEN("it is calibrated with the bracket [1e-3, 10]")
        {
            const Scenario4Calibration cal = scenario4CalibrateSigma(records, Scenario4NisKind::Flow, 0, 50, 1e-3, 10.0);
            THEN("the result is reported as not converged (the nearest bound), not silently accepted")
            {
                CHECK(!cal.converged);
                CHECK(cal.sigma == doctest::Approx(10.0));
            }
        }
    }
}

SCENARIO("Scenario 4 MAP status (R4-01): counts and per-frame outcomes agree on the synthetic flight")
{
    GIVEN("the synthetic flight processed by the runner")
    {
        const SyntheticRun& run = syntheticRunA();
        WHEN("the per-frame outcomes are tallied")
        {
            int flowAttempts = 0;
            int flowApplied = 0;
            int horizonApplied = 0;
            int horizonFrames = 0;
            for (const auto& d : run.diagnostics)
            {
                if (d.flowOutcome)
                {
                    ++flowAttempts;
                    const bool applied = *d.flowOutcome == MapOutcome::Converged || *d.flowOutcome == MapOutcome::AcceptedMaxIterations;
                    CHECK(applied == d.flowUpdated);
                    flowApplied += applied ? 1 : 0;
                }
                if (d.horizonUpdated)
                {
                    ++horizonFrames;
                }
                if (d.horizonOutcome)
                {
                    const bool applied = *d.horizonOutcome == MapOutcome::Converged || *d.horizonOutcome == MapOutcome::AcceptedMaxIterations;
                    CHECK(applied == d.horizonUpdated);
                    horizonApplied += applied ? 1 : 0;
                }
            }
            THEN("every flow attempt has an outcome and the flow is applied on most frames")
            {
                CHECK(flowAttempts >= 0.8 * (kFrames - 1));
                CHECK(flowApplied >= 0.8 * flowAttempts);
                CHECK(horizonApplied >= 0.8 * kFrames);
                CHECK(horizonFrames == horizonApplied);
            }
        }
    }
    GIVEN("a short run of the synthetic flight with NIS records enabled")
    {
        Scenario4Options opt = syntheticOptions("s4_synthetic_a.SRT");
        opt.recordNisRecords = true;
        Scenario4Runner runner(opt);
        const Camera camera = syntheticCamera();
        runner.initialise(camera, kFps, cv::Size(kWidth, kHeight), "s4_synthetic.MOV");
        const auto& frames = syntheticFrames();
        const int nShort = 20;
        for (int k = 0; k < nShort; ++k)
        {
            runner.processFrame(frames[static_cast<std::size_t>(k)], k, k / kFps);
        }
        WHEN("the recorded statistics are re-evaluated at the sigma used in the run")
        {
            int matched = 0;
            double worst = 0.0;
            for (const auto& d : runner.diagnostics())
            {
                for (const auto& rec : runner.nisRecords())
                {
                    if (rec.frameIndex != d.frameIndex || rec.kind != Scenario4NisKind::Flow)
                    {
                        continue;
                    }
                    const double again = scenario4NisAt(rec, runner.options().flowSigmaPx / (0.5 * (camera.fx() + camera.fy())));
                    if (std::isfinite(d.flowNis))
                    {
                        worst = std::max(worst, std::abs(again - d.flowNis) / std::max(1.0, std::abs(d.flowNis)));
                        ++matched;
                    }
                }
            }
            THEN("each in-run flow NIS is reproduced from its record")
            {
                CAPTURE(worst);
                CHECK(matched >= 10);
                CHECK(worst < 1e-9);
            }
            THEN("the MAP counts equal the number of attempted flow updates")
            {
                std::size_t attempts = 0;
                for (const auto& d : runner.diagnostics())
                {
                    attempts += d.flowOutcome ? 1 : 0;
                }
                CHECK(runner.mapCounts().flow.updates == attempts);
                CHECK(runner.mapCounts().flow.skipped() + runner.mapCounts().flow.converged +
                          runner.mapCounts().flow.acceptedMaxIterations == attempts);
            }
        }
    }
}
