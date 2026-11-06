// Regression tests for the Lead's R5 change: gates and classifiers evaluated in square-root form
// (USER rule: no covariance inversion anywhere). Explicit inverses appear ONLY here, as test references.

#include <cmath>
#include <random>
#include <vector>

#include <doctest/doctest.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "../../src/CoreLinalg.h"
#include "../../src/SlamAssociation.h"
#include "../../src/VisualHorizon.h"
#include "testUtil.h"

SCENARIO("slam::gateDistanceSquared equals the explicit Mahalanobis reference")
{
    std::mt19937 rng(4242);
    std::normal_distribution<double> n01(0.0, 1.0);
    GIVEN("random pixel innovations, state factors and measurement sigmas")
    {
        for (int trial = 0; trial < 200; ++trial)
        {
            Eigen::MatrixXd z(6, 2);   // whitened projection block (n x 2)
            for (int i = 0; i < z.size(); ++i)
            {
                z.data()[i] = 3.0 * n01(rng);
            }
            slam::LandmarkPrediction pred;
            pred.pixelCov = z.transpose() * z;
            pred.sigmaPx = 0.3 + std::abs(n01(rng));
            const Eigen::Vector2d e(5.0 * n01(rng), 5.0 * n01(rng));

            Eigen::Matrix2d s = pred.pixelCov;
            s += pred.sigmaPx * pred.sigmaPx * Eigen::Matrix2d::Identity();
            const double ref = e.dot(s.inverse() * e);   // test-only reference

            WHEN("only pixelCov is set (Cholesky path)")
            {
                const double d2 = slam::gateDistanceSquared(pred, e);
                THEN("the distance matches to 1e-10 relative")
                {
                    CHECK(std::abs(d2 - ref) <= 1e-10 * std::max(1.0, ref));
                }
            }
            WHEN("the square-root factor from the whitened projection is set")
            {
                Eigen::MatrixXd zz = z;
                pred.pixelSqrtCov = core::upperFactor<double>(zz);
                const double d2 = slam::gateDistanceSquared(pred, e);
                THEN("the distance matches to 1e-10 relative")
                {
                    CHECK(std::abs(d2 - ref) <= 1e-10 * std::max(1.0, ref));
                }
            }
        }
    }
    GIVEN("a prediction with zero state uncertainty")
    {
        slam::LandmarkPrediction pred;
        pred.sigmaPx = 2.0;
        const Eigen::Vector2d e(2.0, -4.0);
        THEN("the distance is ||e||^2 / sigma^2")
        {
            CHECK(slam::gateDistanceSquared(pred, e) == doctest::Approx(20.0 / 4.0).epsilon(1e-12));
        }
    }
}

SCENARIO("SkyGroundClassifier log-odds in square-root form equal the precision-matrix reference")
{
    GIVEN("textured synthetic sky and ground patches")
    {
        cv::RNG rng(7);
        cv::Mat sky(60, 80, CV_8UC3), ground(60, 80, CV_8UC3);
        rng.fill(sky, cv::RNG::NORMAL, cv::Scalar(200, 170, 150), cv::Scalar(12, 10, 8));
        rng.fill(ground, cv::RNG::NORMAL, cv::Scalar(60, 110, 90), cv::Scalar(25, 30, 20));
        const SkyGroundClassifier c = SkyGroundClassifier::train({sky}, {ground});

        WHEN("log-odds are evaluated on a mixed test image")
        {
            cv::Mat test(40, 40, CV_8UC3);
            rng.fill(test, cv::RNG::UNIFORM, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            const cv::Mat llr = c.logOdds(test);
            const cv::Mat feat = SkyGroundClassifier::features(test, 5);

            THEN("each pixel equals the explicit-inverse Gaussian log-likelihood ratio")
            {
                const auto& s = c.skyModel();
                const auto& g = c.groundModel();
                const Eigen::Matrix4d covS = s.sqrtCov.transpose() * s.sqrtCov;
                const Eigen::Matrix4d covG = g.sqrtCov.transpose() * g.sqrtCov;
                const Eigen::Matrix4d pS = covS.inverse();   // test-only reference
                const Eigen::Matrix4d pG = covG.inverse();
                double worst = 0.0;
                for (int y = 0; y < feat.rows; ++y)
                {
                    for (int x = 0; x < feat.cols; ++x)
                    {
                        const cv::Vec4f v = feat.at<cv::Vec4f>(y, x);
                        const Eigen::Vector4d f(v[0], v[1], v[2], v[3]);
                        const double ref = -0.5 * (f - s.mean).dot(pS * (f - s.mean))
                                         + 0.5 * (f - g.mean).dot(pG * (f - g.mean))
                                         + 0.5 * (std::log(covG.determinant()) - std::log(covS.determinant()));
                        const double got = llr.at<float>(y, x);
                        worst = std::max(worst, std::abs(got - ref) / std::max(1.0, std::abs(ref)));
                    }
                }
                CAPTURE(worst);
                CHECK(worst < 1e-5);   // float32 output image
            }
        }
    }
}
