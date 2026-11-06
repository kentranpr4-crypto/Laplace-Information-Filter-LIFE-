#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>

#include "../../src/ArucoTags.h"
#include "../../src/Camera.h"
#include "../../src/GaussianBase.h"
#include "../../src/GaussianInfo.h"
#include "../../src/SlamPoseAssociation.h"
#include "../../src/SlamPoseTracker.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "../../src/rotation.hpp"

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace
{

constexpr double kTagSide = 0.15;

Camera lifecycleCamera()
{
    return Camera(640, 480, (Matrix3d() << 500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0).finished(), {});
}

/// Estimator with the camera at the origin (eta = 0) and pose landmarks at the given [r; Theta] poses.
struct LandmarkScene
{
    std::shared_ptr<SystemVisualNav> model;
    std::unique_ptr<SystemEstimator> est;
    std::vector<Eigen::Index> ids;
};

LandmarkScene makeScene(const std::vector<Matrix<double, 6, 1>>& poses)
{
    LandmarkScene s;
    const Matrix<double, 6, 1> psd = Matrix<double, 6, 1>::Constant(0.1);
    s.model = std::make_shared<SystemVisualNav>(psd);
    s.est = std::make_unique<SystemEstimator>(
        s.model, GaussianInfo<double>::fromMoment(VectorXd::Zero(12), 1e-6 * MatrixXd::Identity(12, 12)), 0.0);
    for (const Matrix<double, 6, 1>& p : poses)
    {
        s.ids.push_back(s.model->addLandmark(*s.est, SystemVisualNav::LandmarkType::Pose,
                                             GaussianInfo<double>::fromMoment(p, 1e-6 * Matrix<double, 6, 6>::Identity())));
    }
    return s;
}

/// Tag pose [r; Theta] with its centre at r and its normal at the given tilt about the camera's vertical axis.
/// Theta = (0, tilt, yaw): R = Rz(yaw) Ry(tilt); the normal (landmark z axis) is (sin(tilt) cos(yaw), ., cos(tilt)).
Matrix<double, 6, 1> tagPose(const Vector3d& r, double tilt, double yaw)
{
    Matrix<double, 6, 1> p;
    p << r, 0.0, tilt, yaw;
    return p;
}

std::vector<slam::TagPredictionInput> inputsFor(const std::vector<Eigen::Index>& ids)
{
    std::vector<slam::TagPredictionInput> in;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        slam::TagPredictionInput t;
        t.record = i;
        t.id = ids[i];
        t.cornersLocal = tags::canonicalCorners(kTagSide);
        in.push_back(t);
    }
    return in;
}

const slam::TagCandidate& candidateOf(const slam::TagPredictionSet& set, Eigen::Index id)
{
    for (const slam::TagCandidate& c : set.candidates)
    {
        if (c.id == id)
        {
            return c;
        }
    }
    throw std::runtime_error("candidateOf: id not predicted");
}

} // namespace

SCENARIO("Tag lifecycle: a point is inside a tag quad only when it lies in the convex corner polygon")
{
    Matrix<double, 8, 1> quad;
    quad << 100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0;   // axis-aligned square, TL TR BR BL
    GIVEN("a square quad with corners (100,100)-(200,200)")
    {
        THEN("the centre is inside and points outside the square are not")
        {
            CHECK(slam::pointInTagQuad(Vector2d(150.0, 150.0), quad));
            CHECK_FALSE(slam::pointInTagQuad(Vector2d(90.0, 150.0), quad));
            CHECK_FALSE(slam::pointInTagQuad(Vector2d(150.0, 210.0), quad));
            CHECK_FALSE(slam::pointInTagQuad(Vector2d(210.0, 210.0), quad));
        }
        THEN("the orientation of the corner order does not matter")
        {
            Matrix<double, 8, 1> reversed;
            reversed << 100.0, 200.0, 100.0, 100.0, 200.0, 100.0, 200.0, 200.0;
            CHECK(slam::pointInTagQuad(Vector2d(150.0, 150.0), reversed));
            CHECK_FALSE(slam::pointInTagQuad(Vector2d(90.0, 150.0), reversed));
        }
    }
}

SCENARIO("Tag lifecycle: a tag behind a nearer tag on the same line of sight is occluded (not visible, not a miss)")
{
    const Camera cam = lifecycleCamera();
    GIVEN("a near tag at 2 m and a far tag at 3 m, both facing the camera, on the optical axis")
    {
        // Body = world at eta = 0; Rbc = defaultRbc so the optical axis is the body x axis (world +x forward).
        LandmarkScene s = makeScene({tagPose(Vector3d(2.0, 0.0, 0.0), std::numbers::pi / 4.0, std::numbers::pi),
                                     tagPose(Vector3d(3.0, 0.0, 0.0), std::numbers::pi / 4.0, std::numbers::pi)});
        WHEN("the predictions are computed")
        {
            const slam::TagPredictionSet set =
                slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), slam::TagPredictionOptions{});
            const slam::TagCandidate& nearC = candidateOf(set, s.ids[0]);
            const slam::TagCandidate& farC = candidateOf(set, s.ids[1]);
            THEN("the near tag is visible and the far tag is occluded by it")
            {
                CHECK(nearC.visible);
                CHECK_FALSE(nearC.occluded);
                CHECK(farC.occluded);
                CHECK_FALSE(farC.visible);
            }
        }
    }
    GIVEN("the same two tags placed side by side (no line-of-sight overlap)")
    {
        LandmarkScene s = makeScene({tagPose(Vector3d(2.0, -0.4, 0.0), std::numbers::pi / 4.0, std::numbers::pi),
                                     tagPose(Vector3d(3.0, 0.4, 0.0), std::numbers::pi / 4.0, std::numbers::pi)});
        WHEN("the predictions are computed")
        {
            const slam::TagPredictionSet set =
                slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), slam::TagPredictionOptions{});
            THEN("neither tag is occluded")
            {
                CHECK_FALSE(candidateOf(set, s.ids[0]).occluded);
                CHECK_FALSE(candidateOf(set, s.ids[1]).occluded);
                CHECK(candidateOf(set, s.ids[1]).visible);
            }
        }
    }
}

SCENARIO("Tag lifecycle: a tag facing away from the camera is not expected to be detected")
{
    const Camera cam = lifecycleCamera();
    GIVEN("a tag at 2 m whose normal points away from the camera, and one facing it")
    {
        // Normal = R e_z = (sin tilt cos yaw, sin tilt sin yaw, cos tilt): yaw = pi gives -x (towards the camera).
        LandmarkScene s = makeScene({tagPose(Vector3d(2.0, -0.5, 0.0), std::numbers::pi / 4.0, 0.0),
                                     tagPose(Vector3d(2.0, 0.5, 0.0), std::numbers::pi / 4.0, std::numbers::pi)});
        WHEN("the predictions are computed")
        {
            const slam::TagPredictionSet set =
                slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), slam::TagPredictionOptions{});
            THEN("the tag facing away is not visible, the facing tag is")
            {
                CHECK_FALSE(candidateOf(set, s.ids[0]).visible);
                CHECK(candidateOf(set, s.ids[1]).visible);
            }
        }
    }
}

SCENARIO("Re-identification: the archived-tag gate accepts a re-birth at the same place and rejects a different tag")
{
    const Matrix<double, 6, 1> mean = (Matrix<double, 6, 1>() << 1.0, 0.2, 2.0, 0.1, -0.2, 0.3).finished();
    const Matrix<double, 6, 6> sqrtCov = 0.02 * Matrix<double, 6, 6>::Identity();   // archived 2 cm / 0.02 rad std
    const double gate = GaussianBase::mahalanobisThresholdSq(3.0, 6.0);
    GIVEN("the archived mean and covariance, floors of 5 cm and 0.1 rad")
    {
        WHEN("the candidate is the same pose, or 3 cm and 0.03 rad away")
        {
            const double same = slam::tagArchiveDistanceSq(mean, sqrtCov, mean, 0.05, 0.1);
            Matrix<double, 6, 1> near = mean;
            near.head<3>() += Vector3d(0.03, -0.02, 0.01);
            near(5) += 0.03;
            const double d2near = slam::tagArchiveDistanceSq(mean, sqrtCov, near, 0.05, 0.1);
            THEN("the same pose has zero distance and the near pose passes the 3-sigma gate")
            {
                CHECK(same < 1e-12);
                CHECK(d2near < gate);
            }
        }
        WHEN("the candidate is 0.5 m away, or rotated by 1 rad about the normal")
        {
            Matrix<double, 6, 1> far = mean;
            far(0) += 0.5;
            Matrix<double, 6, 1> rotated = mean;
            rotated(5) += 1.0;
            const double d2far = slam::tagArchiveDistanceSq(mean, sqrtCov, far, 0.05, 0.1);
            const double d2rot = slam::tagArchiveDistanceSq(mean, sqrtCov, rotated, 0.05, 0.1);
            THEN("both are rejected by the gate")
            {
                CHECK(d2far > gate);
                CHECK(d2rot > gate);
            }
        }
        THEN("non-positive floors are rejected")
        {
            CHECK_THROWS_AS(slam::tagArchiveDistanceSq(mean, sqrtCov, mean, 0.0, 0.1), std::invalid_argument);
        }
    }
}
