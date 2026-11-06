#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>

#include "../../src/Camera.h"

namespace
{
struct CameraConfig
{
    const char * name;
    double fx;
    double fy;
    double cx;
    double cy;
    double skew;
    std::vector<double> distortion;
};

/** @brief Distortion configurations covering every accepted coefficient count, plus unequal focal lengths. */
std::vector<CameraConfig> representativeConfigs()
{
    return {
        {"pinhole", 800.0, 790.0, 640.0, 360.0, 0.0, {}},
        {"radial-2", 800.0, 800.0, 640.0, 360.0, 0.0, {-0.3, 0.1, 0.0, 0.0}},
        {"tangential-4", 1000.0, 1010.0, 900.0, 500.0, 0.0, {0.02, -0.01, 0.003, -0.002}},
        {"standard-5", 911.57, 914.67, 951.19, 517.24, 0.0, {-0.025, -0.0067, 0.0003, -0.0021, 0.0014}},
        {"standard-5-wide", 620.0, 615.0, 320.0, 240.0, 0.0, {-0.2, 0.05, 0.001, -0.002, 0.01}},
        {"rational-8", 2208.5, 2220.3, 1288.0, 717.1, 0.0, {0.011, 0.0235, -0.00052, -0.0066, -0.131, 0.05, -0.01, 0.002}},
        {"aspect-5", 800.0, 805.0, 640.0, 360.0, 0.0, {-0.1, 0.02, 0.001, 0.001, 0.0}},
    };
}

Camera toCamera(const CameraConfig & c, int width = 1280, int height = 720)
{
    Eigen::Matrix3d K;
    K << c.fx, c.skew, c.cx,
         0.0, c.fy, c.cy,
         0.0, 0.0, 1.0;
    return Camera(width, height, K, c.distortion);
}

cv::Mat toCvK(const CameraConfig & c)
{
    return cv::Mat(cv::Matx33d(c.fx, c.skew, c.cx, 0.0, c.fy, c.cy, 0.0, 0.0, 1.0)).clone();
}

cv::Mat toCvDist(const std::vector<double> & d)
{
    cv::Mat m(1, static_cast<int>(d.size()), CV_64F);
    for (std::size_t i = 0; i < d.size(); ++i)
    {
        m.at<double>(0, static_cast<int>(i)) = d[i];
    }
    return m;
}

/** @brief Random camera-frame points in front of the camera with normalised coordinates within |x|,|y| <= 0.6. */
std::vector<Eigen::Vector3d> randomPoints(std::size_t n, unsigned seed, double maxNormalised = 0.6)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> depth(0.3, 8.0);
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const double Z = depth(gen);
        pts.emplace_back(unit(gen) * maxNormalised * Z, unit(gen) * maxNormalised * Z, Z);
    }
    return pts;
}
} // namespace

SCENARIO("vectorToPixel reproduces cv::projectPoints for every supported distortion model")
{
    for (const auto & config : representativeConfigs())
    {
        GIVEN("a camera configuration and 3000 random camera-frame points")
        {
            INFO("config: " << config.name);
            const Camera camera = toCamera(config);
            const cv::Mat K = toCvK(config);
            const cv::Mat dist = toCvDist(config.distortion);
            const auto points = randomPoints(3000, 1234u + static_cast<unsigned>(config.name[0]));

            std::vector<cv::Point3d> objectPoints;
            objectPoints.reserve(points.size());
            for (const auto & p : points)
            {
                objectPoints.emplace_back(p.x(), p.y(), p.z());
            }
            std::vector<cv::Point2d> cvPixels;
            cv::projectPoints(objectPoints, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, dist, cvPixels);

            double worst = 0.0;
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                const Eigen::Vector2d mine = camera.vectorToPixel(points[i]);
                worst = std::max(worst, std::hypot(mine.x() - cvPixels[i].x, mine.y() - cvPixels[i].y));
            }
            THEN("the largest pixel discrepancy is below 1e-8 px")
            {
                CAPTURE(worst);
                CHECK(worst < 1e-8);
            }
        }
    }
}

SCENARIO("pixelToVector inverts vectorToPixel")
{
    for (const auto & config : representativeConfigs())
    {
        GIVEN("a camera configuration")
        {
            INFO("config: " << config.name);
            const Camera camera = toCamera(config);
            const auto points = randomPoints(500, 42u, 0.5);
            WHEN("points are projected and back-projected")
            {
                double worstPixel = 0.0;
                double worstNormalised = 0.0;
                for (const auto & p : points)
                {
                    const Eigen::Vector2d px = camera.vectorToPixel(p);
                    const Eigen::Vector2d n = camera.pixelToNormalised(px);
                    worstNormalised = std::max(worstNormalised, (n - Eigen::Vector2d(p.x() / p.z(), p.y() / p.z())).norm());
                    const Eigen::Vector3d ray = camera.pixelToVector(px);
                    const Eigen::Vector2d back = camera.vectorToPixel(ray);
                    worstPixel = std::max(worstPixel, (back - px).norm());
                }
                THEN("normalised coordinates are recovered to 1e-9 and pixels to 1e-7 px")
                {
                    CAPTURE(worstNormalised);
                    CAPTURE(worstPixel);
                    CHECK(worstNormalised < 1e-9);
                    CHECK(worstPixel < 1e-7);
                }
            }
            WHEN("a grid of pixels is back-projected to unit rays")
            {
                bool allFinite = true;
                bool allUnit = true;
                for (int v = 0; v < camera.imageHeight(); v += 60)
                {
                    for (int u = 0; u < camera.imageWidth(); u += 80)
                    {
                        const Eigen::Vector3d ray = camera.pixelToVector(Eigen::Vector2d(u, v));
                        allFinite = allFinite && ray.allFinite();
                        allUnit = allUnit && std::abs(ray.norm() - 1.0) < 1e-12 && ray.z() > 0.0;
                    }
                }
                THEN("every ray is finite, unit length and in front of the camera")
                {
                    CHECK(allFinite);
                    CHECK(allUnit);
                }
            }
        }
    }
}

SCENARIO("points behind the camera and outside the image are not visible")
{
    GIVEN("a standard camera 1280x720 with a principal point at the image centre")
    {
        const Camera camera = toCamera(representativeConfigs()[3]);
        WHEN("the point lies behind the image plane or on it")
        {
            THEN("vectorToPixel returns NaN")
            {
                CHECK_FALSE(camera.vectorToPixel(Eigen::Vector3d(0.1, 0.1, -1.0)).allFinite());
                CHECK_FALSE(camera.vectorToPixel(Eigen::Vector3d(0.1, 0.1, 0.0)).allFinite());
            }
            THEN("it is not within the field of view")
            {
                CHECK_FALSE(camera.isVectorWithinFOV(Eigen::Vector3d(0.0, 0.0, -1.0)));
                CHECK_FALSE(camera.isVectorWithinFOV(Eigen::Vector3d(0.0, 0.0, 0.0)));
            }
            THEN("its Jacobian is NaN as well")
            {
                CHECK_FALSE(camera.vectorToPixelJacobian(Eigen::Vector3d(0.0, 0.0, -1.0)).allFinite());
            }
        }
        WHEN("the point projects inside the image")
        {
            const Eigen::Vector3d onAxis(0.0, 0.0, 2.0);
            THEN("it projects to the principal point and is visible")
            {
                CHECK(camera.vectorToPixel(onAxis).isApprox(Eigen::Vector2d(camera.cx(), camera.cy()), 1e-12));
                CHECK(camera.isVectorWithinFOV(onAxis));
            }
        }
        WHEN("the point projects beyond the right edge")
        {
            const Eigen::Vector3d outside(1.0, 0.0, 1.0);
            THEN("the pixel is outside the image and the point is not within the field of view")
            {
                CHECK_FALSE(camera.isWithinImage(camera.vectorToPixel(outside)));
                CHECK_FALSE(camera.isVectorWithinFOV(outside));
            }
        }
        WHEN("a margin is applied")
        {
            const Eigen::Vector3d nearEdge(0.0, 0.0, 1.0);
            THEN("a margin larger than the distance to the nearest edge rejects the point")
            {
                const Eigen::Vector2d pixel = camera.vectorToPixel(nearEdge);
                // The margin applies on all four sides: the nearest edge determines the threshold.
                const double distanceToEdge = std::min({pixel.x() + 0.5, camera.imageWidth() - 0.5 - pixel.x(),
                                                        pixel.y() + 0.5, camera.imageHeight() - 0.5 - pixel.y()});
                CHECK(camera.isVectorWithinFOV(nearEdge, 0.0));
                CHECK(camera.isVectorWithinFOV(nearEdge, distanceToEdge - 1.0));
                CHECK_FALSE(camera.isVectorWithinFOV(nearEdge, distanceToEdge + 1.0));
            }
        }
    }

    GIVEN("pixel-centre convention: a pixel is inside if -0.5 <= u < W - 0.5")
    {
        const Camera camera = toCamera(representativeConfigs()[0], 640, 480);
        THEN("u = -0.4 is inside and u = -0.6 is outside")
        {
            CHECK(camera.isWithinImage(Eigen::Vector2d(-0.4, 100.0)));
            CHECK_FALSE(camera.isWithinImage(Eigen::Vector2d(-0.6, 100.0)));
        }
        THEN("u = W - 0.5 is outside and u = W - 0.6 is inside")
        {
            CHECK_FALSE(camera.isWithinImage(Eigen::Vector2d(639.5, 100.0)));
            CHECK(camera.isWithinImage(Eigen::Vector2d(639.4, 100.0)));
        }
        THEN("non-finite pixels are never inside")
        {
            CHECK_FALSE(camera.isWithinImage(Eigen::Vector2d(std::nan(""), 10.0)));
        }
    }

    GIVEN("a strongly barrel-distorted camera whose distortion folds over at normalised radius 0.577")
    {
        const Camera camera = toCamera(CameraConfig{"fold", 500.0, 500.0, 320.0, 240.0, 0.0, {-1.0, 0.0, 0.0, 0.0}}, 640, 480);
        const Eigen::Vector3d folded(0.8, 0.0, 1.0);
        const Eigen::Vector2d pixel = camera.vectorToPixel(folded);
        THEN("the forward projection lands inside the image")
        {
            CHECK(camera.isWithinImage(pixel));
        }
        THEN("but the point is reported outside the field of view because the map is not invertible there")
        {
            CHECK(camera.distortionJacobianDeterminant(0.8, 0.0) < 0.0);
            CHECK_FALSE(camera.isVectorWithinFOV(folded));
        }
        THEN("a point before the fold is visible")
        {
            CHECK(camera.distortionJacobianDeterminant(0.3, 0.0) > 0.0);
            CHECK(camera.isVectorWithinFOV(Eigen::Vector3d(0.3, 0.0, 1.0)));
        }
    }
}

SCENARIO("the distortion map is locally invertible in the typical calibrated range")
{
    GIVEN("the standard and rational calibrated cameras")
    {
        for (const auto & config : {representativeConfigs()[3], representativeConfigs()[5]})
        {
            WHEN("the camera is sampled over its normalised field")
            {
                INFO("config: " << config.name);
                const Camera camera = toCamera(config);
                double minDet = 1e300;
                for (int i = -10; i <= 10; ++i)
                {
                    for (int j = -10; j <= 10; ++j)
                    {
                        minDet = std::min(minDet, camera.distortionJacobianDeterminant(0.05 * i, 0.05 * j));
                    }
                }
                THEN("the Jacobian determinant stays positive")
                {
                    CHECK(minDet > 0.0);
                }
            }
        }
    }
}
