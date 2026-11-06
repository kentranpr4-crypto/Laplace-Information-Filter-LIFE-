// smoke.cpp - end-to-end smoke test of the native MCHA4400 A2 toolchain.
//
// Usage: a2_smoke <data_dir> <out_dir>
//   <data_dir>  the a2 data directory (contains indoor/ and outdoor/); opened read-only
//   <out_dir>   scratch directory for PNG/MP4 artefacts (never the data directory)
//
// Prints machine-readable lines "A2SMOKE <key> <value>" that verify-env.sh parses.
// Exit code is non-zero if any check fails.

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/version.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <vtkNew.h>
#include <vtkActor.h>
#include <vtkPNGWriter.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>
#include <vtkWindowToImageFilter.h>
#include <vtkVersion.h>

#include <core/session/onnxruntime_cxx_api.h>   // onnxruntime::onnxruntime exports include/onnxruntime

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void kv(const char *key, const std::string &value) {
    std::println("A2SMOKE {} {}", key, value);
}

void check(bool ok, const std::string &what) {
    std::println("A2SMOKE check {} {}", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

std::string fourcc_to_string(double value) {
    const int code = static_cast<int>(value);
    std::string s(4, ' ');
    for (int i = 0; i < 4; ++i) {
        const char c = static_cast<char>((code >> (8 * i)) & 0xFF);
        s[static_cast<std::size_t>(i)] = std::isprint(static_cast<unsigned char>(c)) ? c : '?';
    }
    return s;
}

// C++23 std::println + Eigen 5 + Boost.Math chi-squared quantile (used by GaussianBase::chi2inv)
void check_language_and_numerics() {
    std::println("A2SMOKE cxx_std __cplusplus={}", __cplusplus);
    check(__cplusplus >= 202302L, "C++23 language mode enabled (__cplusplus >= 202302L)");

    // Eigen >= 5 uses semantic versioning in EIGEN_{MAJOR,MINOR,PATCH}_VERSION (EIGEN_WORLD_VERSION stays 3).
    kv("Eigen_version", EIGEN_VERSION_STRING);
    kv("Eigen_semver", std::format("{}.{}.{}", EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION, EIGEN_PATCH_VERSION));
    check(EIGEN_MAJOR_VERSION == 5 && EIGEN_MINOR_VERSION == 0 && EIGEN_PATCH_VERSION == 1,
          "Eigen semver is 5.0.1");

    Eigen::Matrix3d A;
    A << 4, 1, 0,
         1, 3, 1,
         0, 1, 2;
    const Eigen::Vector3d b(1.0, 2.0, 3.0);
    const Eigen::Vector3d x = A.llt().solve(b);
    const double residual = (A * x - b).norm();
    kv("Eigen_llt_residual", std::format("{:.3e}", residual));
    check(residual < 1e-12, "Eigen 5 LLT solve residual < 1e-12");

    const double q = boost::math::quantile(boost::math::chi_squared(3), 0.997);
    kv("Boost_version", std::format("{}.{}.{}", BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000, BOOST_VERSION % 100));
    kv("Boost_chi2_quantile_3dof_0.997", std::format("{:.6f}", q));
    // Independent closed form for chi^2(3): CDF(x) = erf(sqrt(x/2)) - sqrt(2x/pi) exp(-x/2); root at 0.997
    // gives 13.931423 (bisection, checked outside this program).
    check(std::abs(q - 13.931423) < 1e-5, "Boost.Math chi-squared 3-dof 0.997 quantile == 13.931423");

    std::println("hello from std::println: C++23 ok, Eigen {}, OpenCV {}",
                 EIGEN_VERSION_STRING, cv::getVersionString());
}

void check_versions() {
    kv("OpenCV_version", cv::getVersionString());
    check(cv::getVersionMajor() == 5, "OpenCV major version is 5");
    kv("VTK_version", std::format("{}.{}.{}", VTK_MAJOR_VERSION, VTK_MINOR_VERSION, VTK_BUILD_VERSION));
    check(VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION == 5, "VTK version is 9.5");
    kv("ORT_version", Ort::GetVersionString());
}

// Reads frame 0 of each data video, checks codec / size / frame count, and writes a PNG preview.
void check_videos(const fs::path &data_dir, const fs::path &out_dir) {
    const std::vector<fs::path> videos = {
        data_dir / "indoor" / "room.MOV",
        data_dir / "indoor" / "calibration.MOV",
        data_dir / "indoor" / "tag_identical.MOV",
        data_dir / "outdoor" / "flight.MOV",
        data_dir / "outdoor" / "calibration.MOV",
    };
    for (const auto &path : videos) {
        const std::string label = path.parent_path().filename().string() + "/" + path.filename().string();
        cv::VideoCapture cap(path.string(), cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            check(false, "open " + label + " with CAP_FFMPEG");
            continue;
        }
        const std::string codec = fourcc_to_string(cap.get(cv::CAP_PROP_FOURCC));
        const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        const double nframes = cap.get(cv::CAP_PROP_FRAME_COUNT);
        kv("video", std::format("{} codec={} size={}x{} fps={:.3f} frames={:.0f} backend={}",
                                label, codec, w, h, fps, nframes, cap.getBackendName()));
        cv::Mat frame;
        const bool got = cap.read(frame);
        check(got && !frame.empty(), "decode first frame of " + label);
        if (got && !frame.empty()) {
            const double mean = cv::mean(frame)[0];
            check(mean > 1.0, std::format("first frame of {} is not blank (mean {:.2f})", label, mean));
            const fs::path png = out_dir / (path.parent_path().filename().string() + "_" +
                                            path.stem().string() + "_frame0.png");
            cv::Mat small;
            cv::resize(frame, small, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
            cv::imwrite(png.string(), small);
        }
    }
}

// Writes a short mp4v clip, then re-opens it and checks the frame count.
void check_mp4v_roundtrip(const fs::path &data_dir, const fs::path &out_dir) {
    cv::VideoCapture cap((data_dir / "indoor" / "room.MOV").string(), cv::CAP_FFMPEG);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        check(false, "read seed frame for mp4v round trip");
        return;
    }
    const fs::path out = out_dir / "mp4v_roundtrip.mp4";
    cv::VideoWriter writer(out.string(), cv::CAP_FFMPEG,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 29.97, frame.size());
    check(writer.isOpened(), "open VideoWriter mp4v (CAP_FFMPEG)");
    if (!writer.isOpened()) {
        return;
    }
    constexpr int kFrames = 10;
    for (int i = 0; i < kFrames; ++i) {
        writer.write(frame);
    }
    writer.release();

    cv::VideoCapture back(out.string(), cv::CAP_FFMPEG);
    const int got_frames = static_cast<int>(back.get(cv::CAP_PROP_FRAME_COUNT));
    const std::string codec = fourcc_to_string(back.get(cv::CAP_PROP_FOURCC));
    kv("mp4v_roundtrip", std::format("frames_reported={} codec={} size={}x{}", got_frames, codec,
                                     static_cast<int>(back.get(cv::CAP_PROP_FRAME_WIDTH)),
                                     static_cast<int>(back.get(cv::CAP_PROP_FRAME_HEIGHT))));
    int counted = 0;
    cv::Mat tmp;
    while (back.read(tmp)) {
        ++counted;
    }
    check(counted == kFrames, std::format("mp4v clip re-read frame count == {} (got {})", kFrames, counted));
}

// Offscreen VTK render of a sphere to PNG, then read back with OpenCV and count coloured pixels.
void check_vtk_offscreen(const fs::path &out_dir) {
    vtkNew<vtkSphereSource> sphere;
    sphere->SetRadius(0.5);
    sphere->SetThetaResolution(32);
    sphere->SetPhiResolution(32);
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(sphere->GetOutputPort());
    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(1.0, 0.4, 0.1);

    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(actor);
    renderer->SetBackground(0.0, 0.0, 0.0);

    vtkNew<vtkRenderWindow> window;
    window->SetOffScreenRendering(1);
    window->AddRenderer(renderer);
    window->SetSize(640, 480);
    window->Render();

    kv("VTK_render_window", window->GetClassName());
    vtkNew<vtkWindowToImageFilter> capture;
    capture->SetInput(window);
    capture->SetInputBufferTypeToRGB();
    capture->ReadFrontBufferOff();
    capture->Update();

    const fs::path png = out_dir / "vtk_offscreen_sphere.png";
    vtkNew<vtkPNGWriter> writer;
    writer->SetFileName(png.string().c_str());
    writer->SetInputConnection(capture->GetOutputPort());
    writer->Write();

    const cv::Mat img = cv::imread(png.string(), cv::IMREAD_COLOR);
    check(!img.empty() && img.cols == 640 && img.rows == 480, "VTK offscreen PNG readable at 640x480");
    if (!img.empty()) {
        int coloured = 0;
        for (int r = 0; r < img.rows; ++r) {
            const cv::Vec3b *row = img.ptr<cv::Vec3b>(r);
            for (int c = 0; c < img.cols; ++c) {
                if (row[c][0] > 8 || row[c][1] > 8 || row[c][2] > 8) {
                    ++coloured;
                }
            }
        }
        kv("VTK_coloured_pixels", std::to_string(coloured));
        check(coloured > 1000, "VTK offscreen render produced a non-blank sphere");
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::println(stderr, "usage: {} <data_dir> <out_dir>", argv[0]);
        return 2;
    }
    const fs::path data_dir = fs::path(argv[1]);
    const fs::path out_dir = fs::path(argv[2]);
    fs::create_directories(out_dir);

    check_language_and_numerics();
    check_versions();
    {
        std::string info = cv::getBuildInformation();
        const fs::path info_path = out_dir / "opencv_build_information.txt";
        if (FILE *f = std::fopen(info_path.c_str(), "w")) {
            std::fwrite(info.data(), 1, info.size(), f);
            std::fclose(f);
        }
    }
    check_videos(data_dir, out_dir);
    check_mp4v_roundtrip(data_dir, out_dir);
    check_vtk_offscreen(out_dir);

    std::println("A2SMOKE summary failures={}", failures);
    return failures == 0 ? 0 : 1;
}
