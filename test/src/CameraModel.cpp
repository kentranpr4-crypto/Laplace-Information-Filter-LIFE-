#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "../../src/Camera.h"

namespace
{
/** @brief Fixed-seed scratch file in the system temporary directory; removed by the owning test. */
std::filesystem::path scratchFile(const std::string & name)
{
    return std::filesystem::temp_directory_path() / ("a2_camera_test_" + name);
}

Eigen::Matrix3d intrinsics(double fx, double fy, double cx, double cy, double skew = 0.0)
{
    Eigen::Matrix3d K;
    K << fx, skew, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;
    return K;
}

Camera makeCamera(const std::vector<double> & distortion = {-0.2, 0.05, 0.001, -0.002, 0.0})
{
    return Camera(1280, 720, intrinsics(900.0, 905.0, 640.0, 355.0), distortion);
}

void writeText(const std::filesystem::path & path, const std::string & text)
{
    std::ofstream out(path);
    out << text;
}
} // namespace

SCENARIO("camera construction validates its inputs")
{
    GIVEN("a valid intrinsic matrix and standard distortion")
    {
        const Eigen::Matrix3d K = intrinsics(900.0, 905.0, 640.0, 355.0);
        WHEN("the camera is constructed")
        {
            const Camera camera(1280, 720, K, {-0.1, 0.01, 0.0, 0.0, 0.0});
            THEN("it is valid and exposes the parameters")
            {
                CHECK(camera.isValid());
                CHECK(camera.imageWidth() == 1280);
                CHECK(camera.imageHeight() == 720);
                CHECK(camera.fx() == doctest::Approx(900.0));
                CHECK(camera.fy() == doctest::Approx(905.0));
                CHECK(camera.cx() == doctest::Approx(640.0));
                CHECK(camera.cy() == doctest::Approx(355.0));
                CHECK(camera.skew() == doctest::Approx(0.0));
                CHECK(camera.distortionCoefficients().size() == 5);
            }
        }
        WHEN("the matrix has a non-zero skew")
        {
            THEN("it is rejected, because the projection model (matching cv::projectPoints) has no skew term")
            {
                CHECK_THROWS_AS(Camera(1280, 720, intrinsics(900.0, 905.0, 640.0, 355.0, 0.5), {}), std::invalid_argument);
            }
        }
        WHEN("a default-constructed camera is inspected")
        {
            const Camera empty;
            THEN("it is reported invalid") { CHECK_FALSE(empty.isValid()); }
        }
    }

    GIVEN("invalid parameters")
    {
        const Eigen::Matrix3d good = intrinsics(900.0, 905.0, 640.0, 355.0);
        THEN("non-positive image size throws")
        {
            CHECK_THROWS_AS(Camera(0, 720, good, {}), std::invalid_argument);
            CHECK_THROWS_AS(Camera(1280, -1, good, {}), std::invalid_argument);
        }
        THEN("non-zero skew throws")
        {
            CHECK_THROWS_AS(Camera(1280, 720, intrinsics(900.0, 905.0, 640.0, 355.0, 1.0), {}), std::invalid_argument);
        }
        THEN("non-positive focal lengths throw")
        {
            CHECK_THROWS_AS(Camera(1280, 720, intrinsics(-900.0, 905.0, 640.0, 355.0), {}), std::invalid_argument);
            CHECK_THROWS_AS(Camera(1280, 720, intrinsics(900.0, 0.0, 640.0, 355.0), {}), std::invalid_argument);
        }
        THEN("non-finite intrinsics throw")
        {
            CHECK_THROWS_AS(Camera(1280, 720, intrinsics(std::numeric_limits<double>::quiet_NaN(), 905.0, 640.0, 355.0), {}), std::invalid_argument);
        }
        THEN("a matrix that is not of the form [fx s cx; 0 fy cy; 0 0 1] throws")
        {
            Eigen::Matrix3d bad = good;
            bad(1, 0) = 1.0;
            CHECK_THROWS_AS(Camera(1280, 720, bad, {}), std::invalid_argument);
        }
        THEN("unsupported distortion lengths throw")
        {
            CHECK_THROWS_AS(Camera(1280, 720, good, {0.1, 0.2, 0.3}), std::invalid_argument);
            CHECK_THROWS_AS(Camera(1280, 720, good, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}), std::invalid_argument);
            CHECK_THROWS_AS(Camera(1280, 720, good, {0.1, std::numeric_limits<double>::infinity(), 0.0, 0.0}), std::invalid_argument);
        }
        THEN("zero distortion coefficients (pinhole) are accepted")
        {
            const Camera pinhole(1280, 720, good, {});
            CHECK(pinhole.isValid());
            CHECK(pinhole.distortionCoefficients().empty());
        }
    }
}

SCENARIO("default body-to-camera extrinsics follow the documented axis convention")
{
    GIVEN("a camera constructed without explicit extrinsics")
    {
        const Camera camera = makeCamera();
        THEN("Rbc is a proper rotation")
        {
            CHECK(camera.Rbc().determinant() == doctest::Approx(1.0));
            CHECK((camera.Rbc().transpose() * camera.Rbc() - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() < 1e-12);
        }
        THEN("camera optical axis z maps to body forward x, camera x (right) to body y (right), camera y (down) to body z (down)")
        {
            const Eigen::Vector3d cx = camera.Rbc() * Eigen::Vector3d::UnitX();
            const Eigen::Vector3d cy = camera.Rbc() * Eigen::Vector3d::UnitY();
            const Eigen::Vector3d cz = camera.Rbc() * Eigen::Vector3d::UnitZ();
            CAPTURE(camera.Rbc());
            CHECK(cz.isApprox(Eigen::Vector3d::UnitX()));
            CHECK(cx.isApprox(Eigen::Vector3d::UnitY()));
            CHECK(cy.isApprox(Eigen::Vector3d::UnitZ()));
        }
        THEN("the camera origin coincides with the body origin")
        {
            CHECK(camera.rCBb().isApprox(Eigen::Vector3d::Zero()));
        }
    }

    GIVEN("explicit extrinsics")
    {
        Camera camera = makeCamera();
        const Eigen::AngleAxisd yaw(0.3, Eigen::Vector3d::UnitZ());
        const Eigen::Matrix3d R = camera.Rbc() * yaw.toRotationMatrix();
        const Eigen::Vector3d p(0.1, -0.02, 0.05);
        WHEN("they are set to a valid rotation")
        {
            camera.setBodyToCameraExtrinsics(R, p);
            THEN("they are returned unchanged")
            {
                CHECK(camera.Rbc().isApprox(R, 1e-14));
                CHECK(camera.rCBb().isApprox(p, 1e-14));
            }
        }
        WHEN("a reflection is supplied")
        {
            Eigen::Matrix3d reflection = Eigen::Matrix3d::Identity();
            reflection(0, 0) = -1.0;
            THEN("it is rejected") { CHECK_THROWS_AS(camera.setBodyToCameraExtrinsics(reflection, p), std::invalid_argument); }
        }
        WHEN("a non-finite translation is supplied")
        {
            THEN("it is rejected")
            {
                CHECK_THROWS_AS(camera.setBodyToCameraExtrinsics(R, Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0, 0)), std::invalid_argument);
            }
        }
    }
}

SCENARIO("camera.xml round trip preserves intrinsics, distortion, extrinsics and metadata")
{
    GIVEN("a rational-model camera with extrinsics and metadata")
    {
        Camera camera(1920, 1080, intrinsics(911.5657, 914.6668, 951.1927, 517.2400),
                      {-0.025072, -0.0066771, 0.000318028, -0.00211885, 0.00144189, 0.01, -0.002, 0.0003});
        const Eigen::AngleAxisd roll(0.11, Eigen::Vector3d::UnitX());
        camera.setBodyToCameraExtrinsics(makeCamera().Rbc() * roll.toRotationMatrix(), Eigen::Vector3d(0.12, -0.03, 0.07));
        Camera::CalibrationMetadata meta;
        meta.rmsReprojectionErrorPx = 0.6123456789;
        meta.numViews = 60;
        meta.numPoints = 4200;
        meta.boardInnerCornersCols = 10;
        meta.boardInnerCornersRows = 7;
        meta.squareSizeM = 0.022;
        meta.sourceVideo = "calibration.MOV";
        camera.setMetadata(meta);

        const auto path = scratchFile("roundtrip.xml");
        WHEN("it is saved and reloaded")
        {
            camera.save(path);
            const Camera loaded = Camera::load(path);
            std::filesystem::remove(path);
            THEN("intrinsics match exactly")
            {
                CHECK(loaded.imageWidth() == 1920);
                CHECK(loaded.imageHeight() == 1080);
                CHECK(loaded.cameraMatrix().isApprox(camera.cameraMatrix(), 1e-14));
                CHECK(loaded.skew() == doctest::Approx(0.0));
            }
            THEN("all eight distortion coefficients match")
            {
                REQUIRE(loaded.distortionCoefficients().size() == 8);
                for (std::size_t i = 0; i < 8; ++i)
                {
                    CHECK(loaded.distortionCoefficients()[i] == doctest::Approx(camera.distortionCoefficients()[i]).epsilon(1e-12));
                }
            }
            THEN("extrinsics match")
            {
                CHECK(loaded.Rbc().isApprox(camera.Rbc(), 1e-12));
                CHECK(loaded.rCBb().isApprox(camera.rCBb(), 1e-12));
            }
            THEN("calibration metadata matches")
            {
                CHECK(loaded.metadata().rmsReprojectionErrorPx == doctest::Approx(0.6123456789).epsilon(1e-12));
                CHECK(loaded.metadata().numViews == 60);
                CHECK(loaded.metadata().numPoints == 4200);
                CHECK(loaded.metadata().boardInnerCornersCols == 10);
                CHECK(loaded.metadata().boardInnerCornersRows == 7);
                CHECK(loaded.metadata().squareSizeM == doctest::Approx(0.022));
                CHECK(loaded.metadata().sourceVideo == "calibration.MOV");
                CHECK(loaded.metadata().distortionModel == "opencv-rational-8");
            }
        }
    }

    GIVEN("a camera with five coefficients and no extrinsics written explicitly")
    {
        const Camera camera = makeCamera();
        const auto path = scratchFile("standard.xml");
        WHEN("saved and reloaded")
        {
            camera.save(path);
            const Camera loaded = Camera::load(path);
            std::filesystem::remove(path);
            THEN("the default extrinsics are restored")
            {
                CHECK(loaded.Rbc().isApprox(camera.Rbc(), 1e-12));
                CHECK(loaded.rCBb().isApprox(Eigen::Vector3d::Zero()));
                CHECK(loaded.distortionCoefficients().size() == 5);
                CHECK(loaded.metadata().distortionModel == "opencv-standard-5");
            }
        }
    }
}

SCENARIO("loading rejects invalid camera files with a descriptive error")
{
    GIVEN("a path that does not exist")
    {
        THEN("loading throws runtime_error")
        {
            CHECK_THROWS_AS(Camera::load(scratchFile("does_not_exist.xml")), std::runtime_error);
        }
    }

    GIVEN("files with schema violations")
    {
        const std::string header = "<?xml version=\"1.0\"?>\n<opencv_storage>\n";
        const std::string footer = "</opencv_storage>\n";
        const std::string K = "<camera_matrix type_id=\"opencv-matrix\"><rows>3</rows><cols>3</cols><dt>d</dt>"
                              "<data>900 0 640 0 905 355 0 0 1</data></camera_matrix>\n";
        const std::string D5 = "<distortion_coefficients type_id=\"opencv-matrix\"><rows>1</rows><cols>5</cols><dt>d</dt>"
                               "<data>-0.1 0.01 0 0 0</data></distortion_coefficients>\n";
        const std::string size = "<image_width>1280</image_width>\n<image_height>720</image_height>\n";
        const std::string version = "<format_version>1</format_version>\n";

        struct Case
        {
            const char * name;
            std::string text;
        };
        const std::vector<Case> cases = {
            {"missing format_version", header + size + K + D5 + footer},
            {"unsupported format_version", header + "<format_version>2</format_version>\n" + size + K + D5 + footer},
            {"missing camera_matrix", header + version + size + D5 + footer},
            {"missing distortion_coefficients", header + version + size + K + footer},
            {"six distortion coefficients", header + version + size + K +
                "<distortion_coefficients type_id=\"opencv-matrix\"><rows>1</rows><cols>6</cols><dt>d</dt>"
                "<data>0.1 0.2 0 0 0.3 0.4</data></distortion_coefficients>\n" + footer},
            {"zero focal length", header + version + size +
                "<camera_matrix type_id=\"opencv-matrix\"><rows>3</rows><cols>3</cols><dt>d</dt>"
                "<data>0 0 640 0 905 355 0 0 1</data></camera_matrix>\n" + D5 + footer},
            {"non-positive image size", header + version +
                "<image_width>0</image_width>\n<image_height>720</image_height>\n" + K + D5 + footer},
            {"non-orthonormal body rotation", header + version + size + K + D5 +
                "<body_to_camera_rotation type_id=\"opencv-matrix\"><rows>3</rows><cols>3</cols><dt>d</dt>"
                "<data>1 0 0 0 1 0 0 0 2</data></body_to_camera_rotation>\n" + footer},
        };
        for (const auto & c : cases)
        {
            WHEN("the file has an invalid field")
            {
                INFO("case: " << c.name);
                const auto path = scratchFile("invalid.xml");
                writeText(path, c.text);
                THEN("loading throws runtime_error")
                {
                    CHECK_THROWS_AS(Camera::load(path), std::runtime_error);
                }
                std::filesystem::remove(path);
            }
        }
    }

    GIVEN("a valid file that also contains an unusual but legal camera")
    {
        const std::string text = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<format_version>1</format_version>\n"
                                 "<image_width>640</image_width>\n<image_height>480</image_height>\n"
                                 "<camera_matrix type_id=\"opencv-matrix\"><rows>3</rows><cols>3</cols><dt>d</dt>"
                                 "<data>500 0 320 0 500 240 0 0 1</data></camera_matrix>\n"
                                 "<distortion_coefficients type_id=\"opencv-matrix\"><rows>1</rows><cols>4</cols><dt>d</dt>"
                                 "<data>0 0 0 0</data></distortion_coefficients>\n</opencv_storage>\n";
        const auto path = scratchFile("minimal.xml");
        writeText(path, text);
        WHEN("it is loaded")
        {
            const Camera camera = Camera::load(path);
            std::filesystem::remove(path);
            THEN("optional fields take their documented defaults")
            {
                CHECK(camera.isValid());
                CHECK(camera.Rbc().isApprox(makeCamera().Rbc(), 1e-12));
                CHECK(camera.rCBb().isApprox(Eigen::Vector3d::Zero()));
                CHECK(std::isnan(camera.metadata().rmsReprojectionErrorPx));
                CHECK(camera.metadata().numViews == 0);
            }
        }
    }
}

SCENARIO("generated camera.xml fixtures reload and match the recorded calibration")
{
    const std::filesystem::path outdoor = std::filesystem::path(TEST_DATA_DIR) / "outdoor" / "camera.xml";
    const std::filesystem::path indoor = std::filesystem::path(TEST_DATA_DIR) / "indoor" / "camera.xml";
    GIVEN("the outdoor camera (Scenario 4 flight camera) fixture")
    {
        REQUIRE(std::filesystem::exists(outdoor));
        WHEN("it is loaded")
        {
            const Camera camera = Camera::load(outdoor);
            THEN("the image size is the flight.MOV resolution")
            {
                CHECK(camera.imageWidth() == 2688);
                CHECK(camera.imageHeight() == 1512);
            }
            THEN("the recorded RMS is sub-pixel and the view count matches the calibration run")
            {
                CHECK(camera.metadata().rmsReprojectionErrorPx < 1.0);
                CHECK(camera.metadata().numViews == 60);
            }
            THEN("intrinsics are physically plausible (positive, square-ish pixels, principal point near centre)")
            {
                CHECK(camera.fx() > 0.0);
                CHECK(std::abs(camera.fx() - camera.fy()) / camera.fx() < 0.02);
                CHECK(std::abs(camera.cx() - camera.imageWidth() / 2.0) < 0.1 * camera.imageWidth());
                CHECK(std::abs(camera.cy() - camera.imageHeight() / 2.0) < 0.1 * camera.imageHeight());
            }
        }
    }
    GIVEN("the indoor camera (Scenarios 5-6) fixture")
    {
        REQUIRE(std::filesystem::exists(indoor));
        WHEN("it is loaded")
        {
            const Camera camera = Camera::load(indoor);
            THEN("the image size is the tag_identical.MOV / room.MOV resolution")
            {
                CHECK(camera.imageWidth() == 1920);
                CHECK(camera.imageHeight() == 1080);
            }
            THEN("the view count matches the calibration run")
            {
                CHECK(camera.metadata().numViews == 60);
            }
        }
    }
}

SCENARIO("calibration image size matches the project videos (metadata only)")
{
    // Opens the container headers only (no decoding). A missing project video fails the test (R4-03).
    struct Expected
    {
        const char * path;
        int width;
        int height;
    };
    const std::filesystem::path root = std::filesystem::path(TEST_DATA_DIR).parent_path().parent_path() / "data";
    const std::vector<Expected> expected = {
        {"outdoor/flight.MOV", 2688, 1512},
        {"outdoor/calibration.MOV", 2688, 1512},
        {"indoor/tag_identical.MOV", 1920, 1080},
        {"indoor/room.MOV", 1920, 1080},
        {"indoor/calibration.MOV", 1920, 1080},
    };
    for (const auto & e : expected)
    {
        const std::filesystem::path video = root / e.path;
        // Project videos are repository fixtures: a missing file fails the test (it is never reported as passed).
        REQUIRE_MESSAGE(std::filesystem::exists(video), "missing fixture: ", video.string());
        GIVEN("a project video")
        {
            INFO("video: " << e.path);
            cv::VideoCapture cap(video.string());
            THEN("the video opens and has the expected frame size")
            {
                REQUIRE(cap.isOpened());
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) == e.width);
                CHECK(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) == e.height);
            }
        }
    }
}
