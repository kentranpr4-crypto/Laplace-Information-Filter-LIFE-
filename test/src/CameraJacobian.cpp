#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "../../src/Camera.h"

namespace
{
struct JacobianConfig
{
    const char * name;
    double fx;
    double fy;
    double cx;
    double cy;
    double skew;
    std::vector<double> distortion;
};

Camera cameraFrom(const JacobianConfig & c)
{
    Eigen::Matrix3d K;
    K << c.fx, c.skew, c.cx,
         0.0, c.fy, c.cy,
         0.0, 0.0, 1.0;
    return Camera(1920, 1080, K, c.distortion);
}

/**
 * @brief Central finite-difference Jacobian of vectorToPixel with respect to the camera-frame vector.
 * Step is relative to the point magnitude so that the truncation and rounding errors balance.
 */
Eigen::Matrix<double, 2, 3> centralDifferenceJacobian(const Camera & camera, const Eigen::Vector3d & p)
{
    Eigen::Matrix<double, 2, 3> J;
    for (int i = 0; i < 3; ++i)
    {
        const double h = 1e-6 * std::max(1.0, std::abs(p(i)));
        Eigen::Vector3d plus = p;
        Eigen::Vector3d minus = p;
        plus(i) += h;
        minus(i) -= h;
        J.col(i) = (camera.vectorToPixel(plus) - camera.vectorToPixel(minus)) / (2.0 * h);
    }
    return J;
}

std::vector<Eigen::Vector3d> sampleStates(unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> depth(0.4, 12.0);
    std::vector<Eigen::Vector3d> pts;
    // Representative hand-picked states: on axis, near the image centre, near the edges, close and far.
    pts.emplace_back(0.0, 0.0, 1.0);
    pts.emplace_back(0.05, -0.03, 0.5);
    pts.emplace_back(0.6, 0.4, 2.0);
    pts.emplace_back(-0.7, 0.5, 0.8);
    pts.emplace_back(2.0, -1.5, 5.0);
    pts.emplace_back(0.1, 0.1, 10.0);
    for (int i = 0; i < 200; ++i)
    {
        const double Z = depth(gen);
        pts.emplace_back(unit(gen) * 0.7 * Z, unit(gen) * 0.6 * Z, Z);
    }
    return pts;
}
} // namespace

SCENARIO("analytic Jacobian of vectorToPixel matches central finite differences")
{
    const std::vector<JacobianConfig> configs = {
        {"pinhole", 800.0, 790.0, 640.0, 360.0, 0.0, {}},
        {"tangential-4", 1000.0, 1010.0, 900.0, 500.0, 0.0, {0.02, -0.01, 0.003, -0.002}},
        {"standard-5", 911.57, 914.67, 951.19, 517.24, 0.0, {-0.025, -0.0067, 0.0003, -0.0021, 0.0014}},
        {"standard-5-strong", 620.0, 615.0, 320.0, 240.0, 0.0, {-0.2, 0.05, 0.001, -0.002, 0.01}},
        {"rational-8", 2208.5, 2220.3, 1288.0, 717.1, 0.0, {0.011, 0.0235, -0.00052, -0.0066, -0.131, 0.05, -0.01, 0.002}},
        {"rational-8-denominator", 700.0, 700.0, 320.0, 240.0, 0.0, {0.1, -0.05, 0.001, 0.002, 0.01, 0.3, 0.1, -0.02}},
        {"aspect-5", 800.0, 805.0, 640.0, 360.0, 0.0, {-0.1, 0.02, 0.001, 0.001, 0.0}},
    };

    for (const auto & config : configs)
    {
        GIVEN("a camera configuration")
        {
            INFO("config: " << config.name);
            const Camera camera = cameraFrom(config);
            WHEN("the Jacobian is compared with central differences at 206 states")
            {
                double worstAbs = 0.0;
                double worstRel = 0.0;
                std::size_t checked = 0;
                for (const auto & p : sampleStates(7u))
                {
                    if (!camera.isVectorWithinFOV(p, -1e6)) // only states in front of the camera with a valid map
                    {
                        continue;
                    }
                    const auto [pixel, J] = camera.vectorToPixelWithJacobian(p);
                    const Eigen::Matrix<double, 2, 3> Jfd = centralDifferenceJacobian(camera, p);
                    const double err = (J - Jfd).cwiseAbs().maxCoeff();
                    const double scale = std::max(1.0, J.cwiseAbs().maxCoeff());
                    worstAbs = std::max(worstAbs, err);
                    worstRel = std::max(worstRel, err / scale);
                    ++checked;
                    CAPTURE(p.transpose());
                    CAPTURE(J);
                    CAPTURE(Jfd);
                    REQUIRE(pixel.allFinite());
                }
                THEN("every state agrees to a relative tolerance of 1e-6")
                {
                    CAPTURE(worstAbs);
                    CAPTURE(worstRel);
                    CHECK(checked > 150);
                    CHECK(worstRel < 1e-6);
                }
            }
            WHEN("the two Jacobian accessors are compared")
            {
                const Eigen::Vector3d p(0.3, -0.2, 2.5);
                THEN("vectorToPixelJacobian equals the Jacobian returned with the pixel")
                {
                    CHECK(camera.vectorToPixelJacobian(p).isApprox(camera.vectorToPixelWithJacobian(p).second, 1e-14));
                }
            }
        }
    }
}

SCENARIO("Jacobian of vectorToPixel is NaN behind the camera")
{
    GIVEN("a standard camera")
    {
        const Camera camera = cameraFrom(JacobianConfig{"standard", 900.0, 900.0, 640.0, 360.0, 0.0, {-0.1, 0.01, 0.0, 0.0, 0.0}});
        WHEN("the point is behind the camera")
        {
            const auto [pixel, J] = camera.vectorToPixelWithJacobian(Eigen::Vector3d(0.1, 0.2, -0.5));
            THEN("both pixel and Jacobian are NaN")
            {
                CHECK_FALSE(pixel.allFinite());
                CHECK_FALSE(J.allFinite());
            }
        }
    }
}

SCENARIO("analytic distortion Jacobian determinant matches finite differences of the projection")
{
    GIVEN("rational and standard cameras with zero skew")
    {
        const std::vector<JacobianConfig> configs = {
            {"standard-5", 911.57, 914.67, 951.19, 517.24, 0.0, {-0.025, -0.0067, 0.0003, -0.0021, 0.0014}},
            {"rational-8", 700.0, 700.0, 320.0, 240.0, 0.0, {0.1, -0.05, 0.001, 0.002, 0.01, 0.3, 0.1, -0.02}},
        };
        for (const auto & config : configs)
        {
            WHEN("the determinant of d(u,v)/d(x,y) is compared with the analytic value")
            {
                INFO("config: " << config.name);
                const Camera camera = cameraFrom(config);
                double worst = 0.0;
                for (double x = -0.6; x <= 0.6001; x += 0.15)
                {
                    for (double y = -0.5; y <= 0.5001; y += 0.125)
                    {
                        const double h = 1e-6;
                        // Finite-difference Jacobian of the normalised-to-pixel map (x, y, 1) -> (u, v).
                        const Eigen::Vector2d dx = (camera.vectorToPixel(Eigen::Vector3d(x + h, y, 1.0)) - camera.vectorToPixel(Eigen::Vector3d(x - h, y, 1.0))) / (2.0 * h);
                        const Eigen::Vector2d dy = (camera.vectorToPixel(Eigen::Vector3d(x, y + h, 1.0)) - camera.vectorToPixel(Eigen::Vector3d(x, y - h, 1.0))) / (2.0 * h);
                        const double detPixel = dx.x() * dy.y() - dy.x() * dx.y();
                        const double detNormalised = detPixel / (camera.fx() * camera.fy());
                        const double analytic = camera.distortionJacobianDeterminant(x, y);
                        worst = std::max(worst, std::abs(detNormalised - analytic) / std::max(1.0, std::abs(analytic)));
                    }
                }
                THEN("the analytic determinant agrees with the finite-difference determinant to 1e-6 relative")
                {
                    CAPTURE(worst);
                    CHECK(worst < 1e-6);
                }
            }
        }
    }
}
