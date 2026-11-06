#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/calib.hpp>
#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "../../src/Camera.h"
#include "../../src/calibrate.h"

namespace
{
const cv::Size kInner(10, 7);
constexpr double kSquare = 0.022;

/** @brief Synthetic board views of a known camera with additive pixel noise. */
std::vector<CalibrationView> syntheticViews(const cv::Mat & K, const cv::Mat & dist, int count, double noisePx, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> angle(-0.5, 0.5);
    std::uniform_real_distribution<double> shift(-0.06, 0.06);
    std::uniform_real_distribution<double> depth(0.45, 0.9);
    std::normal_distribution<double> noise(0.0, noisePx);

    std::vector<cv::Point3d> board;
    for (int r = 0; r < kInner.height; ++r)
    {
        for (int c = 0; c < kInner.width; ++c)
        {
            board.emplace_back(c * kSquare, r * kSquare, 0.0);
        }
    }
    std::vector<CalibrationView> views;
    for (int v = 0; v < count; ++v)
    {
        const cv::Vec3d rvec(angle(gen), angle(gen), angle(gen) * 0.5);
        const cv::Vec3d tvec(shift(gen) - 0.1, shift(gen) - 0.05, depth(gen));
        std::vector<cv::Point2d> projected;
        cv::projectPoints(board, rvec, tvec, K, dist, projected);
        CalibrationView view;
        view.frameIndex = v * 6;
        view.timeSeconds = v * 0.2;
        for (const auto & p : projected)
        {
            view.corners.emplace_back(static_cast<float>(p.x + noise(gen)), static_cast<float>(p.y + noise(gen)));
        }
        view.sharpness = 1.0;
        view.boardAreaFraction = 0.1;
        views.push_back(view);
    }
    return views;
}

cv::Mat cameraMatrix(double fx, double fy, double cx, double cy)
{
    return cv::Mat(cv::Matx33d(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)).clone();
}

/** @brief Creates a unique scratch directory under the system temp dir; removed by the caller. */
std::filesystem::path makeScratchDir(const std::string & tag)
{
    const auto dir = std::filesystem::temp_directory_path() / ("a2_calib_" + tag + "_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    return dir;
}

void writeConfig(const std::filesystem::path & path, const std::string & body)
{
    std::ofstream out(path);
    out << "<?xml version=\"1.0\"?>\n<opencv_storage>\n<chessboard_data>\n" << body << "</chessboard_data>\n</opencv_storage>\n";
}
} // namespace

SCENARIO("calibrateFromViews recovers the intrinsics of a synthetic camera")
{
    GIVEN("a 1280x720 camera with standard distortion and 20 noisy board views (0.1 px noise)")
    {
        const cv::Mat K = cameraMatrix(900.0, 905.0, 639.5, 359.5);
        const cv::Mat dist = cv::Mat(cv::Matx<double, 1, 5>(-0.2, 0.05, 0.001, -0.002, 0.0)).clone();
        const auto views = syntheticViews(K, dist, 20, 0.1, 2024u);
        CalibrationOptions options;
        options.verbose = false;
        options.crossValidationFolds = 3; // keeps the Debug-build test fast; the default is 5
        WHEN("the views are calibrated")
        {
            const CalibrationResult result = calibrateFromViews(cv::Size(1280, 720), kInner, kSquare, views, options);
            THEN("focal lengths, principal point and distortion are recovered")
            {
                CHECK(result.camera.fx() == doctest::Approx(900.0).epsilon(0.002));
                CHECK(result.camera.fy() == doctest::Approx(905.0).epsilon(0.002));
                CHECK(result.camera.cx() == doctest::Approx(639.5).epsilon(0.002));
                CHECK(result.camera.cy() == doctest::Approx(359.5).epsilon(0.002));
                REQUIRE(result.camera.distortionCoefficients().size() == 5);
                CHECK(result.camera.distortionCoefficients()[0] == doctest::Approx(-0.2).epsilon(0.05));
            }
            THEN("the in-sample RMS is at the noise level and agrees with cv::calibrateCamera")
            {
                CHECK(result.rmsPx < 0.2);
                CHECK(result.rmsPx == doctest::Approx(result.cvRmsPx).epsilon(1e-6));
                CHECK(result.camera.metadata().numViews == 20);
                CHECK(result.camera.metadata().numPoints == 20 * 70);
            }
            THEN("the standard model is selected because the rational model does not beat it by the switch margin")
            {
                CHECK(result.selectedModel == "opencv-standard-5");
                REQUIRE(result.modelScores.size() == 2);
                CHECK(result.modelScores[0].converged);
                CHECK(result.modelScores[1].converged);
            }
            THEN("the sanity checks pass (INFO lines are informational)")
            {
                for (const auto & c : result.checks)
                {
                    CHECK((c.rfind("PASS", 0) == 0 || c.rfind("INFO", 0) == 0));
                }
            }
            THEN("the principal point is fixed at the image centre (default option) and the free value is reported as INFO")
            {
                CHECK(result.camera.cx() == doctest::Approx(639.5).epsilon(1e-9));
                CHECK(result.camera.cy() == doctest::Approx(359.5).epsilon(1e-9));
                bool info = false;
                for (const auto & c : result.checks)
                {
                    info = info || c.rfind("INFO free principal point", 0) == 0;
                }
                CHECK(info);
            }
        }
    }
}

SCENARIO("a grossly corrupted view is rejected as an outlier")
{
    GIVEN("20 clean synthetic views and one view where three corners are mis-detected by 25 px")
    {
        // A global shift of all corners would be absorbed by a board translation, so corrupt individual corners.
        const cv::Mat K = cameraMatrix(900.0, 900.0, 640.0, 360.0);
        const cv::Mat dist = cv::Mat(cv::Matx<double, 1, 5>(-0.1, 0.01, 0.0, 0.0, 0.0)).clone();
        auto views = syntheticViews(K, dist, 20, 0.1, 99u);
        for (std::size_t i : {5u, 33u, 61u})
        {
            views[7].corners[i].x += 25.0f;
        }
        CalibrationOptions options;
        options.verbose = false;
        options.crossValidationFolds = 3;
        WHEN("the views are calibrated")
        {
            const CalibrationResult result = calibrateFromViews(cv::Size(1280, 720), kInner, kSquare, views, options);
            THEN("the corrupted view is logged as an outlier and removed")
            {
                bool logged = false;
                for (const auto & line : result.rejectionLog)
                {
                    logged = logged || line.find("outlier") != std::string::npos;
                }
                CHECK(logged);
                CHECK(result.views.size() == 19);
            }
            THEN("the focal length is still recovered within 0.5 percent")
            {
                CHECK(result.camera.fx() == doctest::Approx(900.0).epsilon(0.005));
            }
        }
    }
}

SCENARIO("calibrateFromViews validates its inputs")
{
    GIVEN("too few views")
    {
        const cv::Mat K = cameraMatrix(900.0, 900.0, 640.0, 360.0);
        const cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
        const auto views = syntheticViews(K, dist, 5, 0.1, 5u);
        THEN("calibration throws runtime_error")
        {
            CHECK_THROWS_AS(calibrateFromViews(cv::Size(1280, 720), kInner, kSquare, views), std::runtime_error);
        }
    }
    GIVEN("a view with the wrong number of corners")
    {
        auto views = syntheticViews(cameraMatrix(900, 900, 640, 360), cv::Mat::zeros(1, 5, CV_64F), 12, 0.1, 6u);
        views[3].corners.pop_back();
        THEN("it throws invalid_argument")
        {
            CHECK_THROWS_AS(calibrateFromViews(cv::Size(1280, 720), kInner, kSquare, views), std::invalid_argument);
        }
    }
    GIVEN("a non-positive square size")
    {
        const auto views = syntheticViews(cameraMatrix(900, 900, 640, 360), cv::Mat::zeros(1, 5, CV_64F), 12, 0.1, 7u);
        THEN("it throws invalid_argument")
        {
            CHECK_THROWS_AS(calibrateFromViews(cv::Size(1280, 720), kInner, 0.0, views), std::invalid_argument);
        }
    }
}

SCENARIO("config.xml is validated and file_regex is resolved against the directory")
{
    GIVEN("a directory with calibration.MOV (empty placeholder) and a valid config.xml")
    {
        const auto dir = makeScratchDir("cfg");
        {
            std::ofstream(dir / "calibration.MOV").put('x');
        }
        const auto config = dir / "config.xml";
        writeConfig(config, "<file_regex>\"calibration.MOV\"</file_regex>\n<chessboard>\n<grid_width>10</grid_width>\n"
                            "<grid_height>7</grid_height>\n<square_size>0.022</square_size>\n</chessboard>\n");
        WHEN("it is read")
        {
            const ChessboardConfig cfg = readChessboardConfig(config);
            THEN("the quotes are stripped, the board is read and the video is resolved")
            {
                CHECK(cfg.fileRegex == "calibration.MOV");
                CHECK(cfg.innerCornersPerRow == 10);
                CHECK(cfg.innerCornersPerColumn == 7);
                CHECK(cfg.squareSizeM == doctest::Approx(0.022));
                REQUIRE(cfg.videos.size() == 1);
                CHECK(cfg.videos[0].filename() == "calibration.MOV");
            }
        }
        std::filesystem::remove_all(dir);
    }

    GIVEN("a config with several matching files")
    {
        const auto dir = makeScratchDir("multi");
        std::ofstream(dir / "calibration.MOV").put('x');
        std::ofstream(dir / "calibration2.MOV").put('x');
        std::ofstream(dir / "other.MOV").put('x');
        writeConfig(dir / "config.xml", "<file_regex>\"calibration.*\\.MOV\"</file_regex>\n<chessboard>\n<grid_width>10</grid_width>\n"
                                        "<grid_height>7</grid_height>\n<square_size>0.022</square_size>\n</chessboard>\n");
        WHEN("it is read")
        {
            const ChessboardConfig cfg = readChessboardConfig(dir / "config.xml");
            THEN("every matching file is used, sorted by name, and the non-matching file is ignored")
            {
                REQUIRE(cfg.videos.size() == 2);
                CHECK(cfg.videos[0].filename() == "calibration.MOV");
                CHECK(cfg.videos[1].filename() == "calibration2.MOV");
            }
        }
        std::filesystem::remove_all(dir);
    }

    GIVEN("invalid configurations")
    {
        const auto dir = makeScratchDir("bad");
        std::ofstream(dir / "calibration.MOV").put('x');
        const std::string chessboard = "<chessboard>\n<grid_width>10</grid_width>\n<grid_height>7</grid_height>\n<square_size>0.022</square_size>\n</chessboard>\n";
        THEN("a missing grid_width throws")
        {
            writeConfig(dir / "config.xml", "<file_regex>\"calibration.MOV\"</file_regex>\n<chessboard>\n<grid_height>7</grid_height>\n<square_size>0.022</square_size>\n</chessboard>\n");
            CHECK_THROWS_AS(readChessboardConfig(dir / "config.xml"), std::runtime_error);
        }
        THEN("a zero square size throws")
        {
            writeConfig(dir / "config.xml", "<file_regex>\"calibration.MOV\"</file_regex>\n<chessboard>\n<grid_width>10</grid_width>\n<grid_height>7</grid_height>\n<square_size>0</square_size>\n</chessboard>\n");
            CHECK_THROWS_AS(readChessboardConfig(dir / "config.xml"), std::runtime_error);
        }
        THEN("a grid of one row throws")
        {
            writeConfig(dir / "config.xml", "<file_regex>\"calibration.MOV\"</file_regex>\n<chessboard>\n<grid_width>10</grid_width>\n<grid_height>1</grid_height>\n<square_size>0.022</square_size>\n</chessboard>\n");
            CHECK_THROWS_AS(readChessboardConfig(dir / "config.xml"), std::runtime_error);
        }
        THEN("a regex that matches no file throws")
        {
            writeConfig(dir / "config.xml", "<file_regex>\"missing.MOV\"</file_regex>\n" + chessboard);
            CHECK_THROWS_AS(readChessboardConfig(dir / "config.xml"), std::runtime_error);
        }
        THEN("an invalid regular expression throws")
        {
            writeConfig(dir / "config.xml", "<file_regex>\"(calibration\"</file_regex>\n" + chessboard);
            CHECK_THROWS_AS(readChessboardConfig(dir / "config.xml"), std::runtime_error);
        }
        THEN("a missing file_regex throws")
        {
            writeConfig(dir / "config.xml", chessboard);
            CHECK_THROWS_AS(readChessboardConfig(dir / "config.xml"), std::runtime_error);
        }
        THEN("a missing configuration file throws")
        {
            CHECK_THROWS_AS(readChessboardConfig(dir / "absent.xml"), std::runtime_error);
        }
        std::filesystem::remove_all(dir);
    }
}

SCENARIO("the reprojection error recomputed from the saved camera and corner fixtures matches the stored RMS")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR);
    for (const char * scene : {"indoor", "outdoor"})
    {
        GIVEN("the camera.xml and calibration_views.xml fixtures")
        {
            INFO("scene: " << scene);
            const auto cameraPath = root / scene / "camera.xml";
            const auto viewsPath = root / scene / "calibration_views.xml";
            REQUIRE(std::filesystem::exists(cameraPath));
            REQUIRE(std::filesystem::exists(viewsPath));
            const Camera camera = Camera::load(cameraPath);
            cv::FileStorage fs(viewsPath.string(), cv::FileStorage::READ);
            REQUIRE(fs.isOpened());
            cv::FileNode views = fs["views"];
            REQUIRE(views.isSeq());

            std::vector<cv::Point3d> board;
            for (int r = 0; r < kInner.height; ++r)
            {
                for (int c = 0; c < kInner.width; ++c)
                {
                    board.emplace_back(c * kSquare, r * kSquare, 0.0);
                }
            }

            double sse = 0.0;
            std::size_t points = 0;
            int count = 0;
            double worstPerView = 0.0;
            for (auto it = views.begin(); it != views.end(); ++it, ++count)
            {
                cv::Mat rvec;
                cv::Mat tvec;
                cv::Mat corners;
                double storedRms = 0.0;
                (*it)["rvec"] >> rvec;
                (*it)["tvec"] >> tvec;
                (*it)["corners"] >> corners;
                (*it)["rms_px"] >> storedRms;
                REQUIRE(corners.rows == static_cast<int>(board.size()));

                cv::Mat R;
                cv::Rodrigues(rvec, R);
                Eigen::Matrix3d Re;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        Re(i, j) = R.at<double>(i, j);
                    }
                }
                const Eigen::Vector3d t(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                double viewSse = 0.0;
                for (std::size_t i = 0; i < board.size(); ++i)
                {
                    const Eigen::Vector3d P(board[i].x, board[i].y, board[i].z);
                    const Eigen::Vector2d pixel = camera.vectorToPixel(Re * P + t);
                    const Eigen::Vector2d measured(corners.at<double>(static_cast<int>(i), 0), corners.at<double>(static_cast<int>(i), 1));
                    viewSse += (pixel - measured).squaredNorm();
                }
                sse += viewSse;
                points += board.size();
                const double viewRms = std::sqrt(viewSse / static_cast<double>(board.size()));
                worstPerView = std::max(worstPerView, std::abs(viewRms - storedRms));
            }
            const double overall = std::sqrt(sse / static_cast<double>(points));
            WHEN("the per-view and overall RMS are recomputed with Camera::vectorToPixel")
            {
                THEN("every view agrees with its stored RMS to 1e-6 px")
                {
                    CHECK(count == camera.metadata().numViews);
                    CHECK(worstPerView < 1e-6);
                }
                THEN("the overall RMS agrees with camera.xml to 1e-6 px and is sub-pixel")
                {
                    CAPTURE(overall);
                    CHECK(overall == doctest::Approx(camera.metadata().rmsReprojectionErrorPx).epsilon(1e-6));
                    CHECK(overall < 1.5);
                }
            }
        }
    }
}

SCENARIO("the chessboard frame fixture is re-detected and its corners reproduce the stored calibration")
{
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR) / "indoor";
    const auto framePath = root / "chessboard_frame112.jpg";
    const auto viewsPath = root / "calibration_views.xml";
    GIVEN("a 1920x1080 JPEG extracted from calibration.MOV frame 112 (a stored view)")
    {
        REQUIRE(std::filesystem::exists(framePath));
        const cv::Mat image = cv::imread(framePath.string(), cv::IMREAD_COLOR);
        REQUIRE(!image.empty());
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::Mat found;
        const bool ok = cv::findChessboardCornersSB(gray, kInner, found, cv::CALIB_CB_EXHAUSTIVE);
        THEN("the 10x7 board is detected")
        {
            REQUIRE(ok);
            REQUIRE(found.total() == 70);
        }
        cv::FileStorage fs(viewsPath.string(), cv::FileStorage::READ);
        REQUIRE(fs.isOpened());
        cv::FileNode views = fs["views"];
        cv::Mat stored;
        for (auto it = views.begin(); it != views.end(); ++it)
        {
            int frame = -1;
            (*it)["frame_index"] >> frame;
            if (frame == 112)
            {
                (*it)["corners"] >> stored;
            }
        }
        REQUIRE(!stored.empty());
        WHEN("the detected corners are compared with the stored corners")
        {
            double worst = 0.0;
            for (int i = 0; i < 70; ++i)
            {
                const cv::Point2f d = found.at<cv::Point2f>(i) - cv::Point2f(static_cast<float>(stored.at<double>(i, 0)), static_cast<float>(stored.at<double>(i, 1)));
                worst = std::max(worst, static_cast<double>(std::sqrt(d.x * d.x + d.y * d.y)));
            }
            THEN("they agree to within 0.5 px (JPEG re-encoding and decoder differences)")
            {
                CAPTURE(worst);
                CHECK(worst < 0.5);
            }
        }
    }
}

SCENARIO("calibration model selection is deterministic: the rational model is used only when explicitly allowed (R4-15)")
{
    GIVEN("a synthetic camera with standard distortion and 20 noisy views, and a switch margin that makes the rational model win")
    {
        const cv::Mat K = cameraMatrix(900.0, 905.0, 639.5, 359.5);
        const cv::Mat dist = cv::Mat(cv::Matx<double, 1, 5>(-0.2, 0.05, 0.001, -0.002, 0.0)).clone();
        const auto views = syntheticViews(K, dist, 20, 0.1, 2024u);
        CalibrationOptions options;
        options.verbose = false;
        options.crossValidationFolds = 3;
        options.modelSwitchMargin = -10.0;   // the rational model is "better" by any margin when cross-validated
        WHEN("the default options are used (allowRationalModel = false)")
        {
            const CalibrationResult result = calibrateFromViews(cv::Size(1280, 720), kInner, kSquare, views, options);
            THEN("the standard model is used, the rational score is still reported, and a warning explains the decision")
            {
                CHECK(result.selectedModel == "opencv-standard-5");
                REQUIRE(result.modelScores.size() == 2);
                CHECK(result.modelScores[1].converged);
                bool warned = false;
                for (const std::string& w : result.warnings)
                {
                    warned = warned || (w.find("allowRationalModel") != std::string::npos);
                }
                CHECK(warned);
            }
        }
        WHEN("the rational model is explicitly allowed")
        {
            options.allowRationalModel = true;
            const CalibrationResult result = calibrateFromViews(cv::Size(1280, 720), kInner, kSquare, views, options);
            THEN("the rational model is selected")
            {
                CHECK(result.selectedModel == "opencv-rational-8");
            }
        }
    }
}
