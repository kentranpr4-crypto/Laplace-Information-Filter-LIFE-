// Geometric association of identical-ID tags: innovation covariances (triangular solves) against the reference covariance,
// gating, birth candidates, ambiguity rejection, conflicts and the joint NIS.
#include <cmath>
#include <memory>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/ArucoTags.h"
#include "../../src/Camera.h"
#include "../../src/GaussianBase.h"
#include "../../src/GaussianInfo.h"
#include "../../src/Pose.hpp"
#include "../../src/SlamPoseAssociation.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using slam::TagAssociationOptions;
using slam::TagPredictionInput;
using slam::TagPredictionOptions;
using slam::TagPredictionSet;

namespace
{

constexpr double kSide = 0.15;
constexpr double kSigmaPx = 2.0;

Camera assocCamera()
{
    Camera cam(1920, 1080, (Matrix3d() << 889.3, 0.0, 959.5, 0.0, 892.3, 539.5, 0.0, 0.0, 1.0).finished(),
               {-0.0246, -0.0061, 0.0005, 0.0001, 0.0013});
    cam.setBodyToCameraExtrinsics(rigid::defaultRbc(), Vector3d(0.05, -0.02, -0.10));
    return cam;
}

/// Tag landmark (6-vector) facing the camera at the given world centre: R_nL = [y, -z, -x] columns (normal towards -x).
Matrix<double, 6, 1> tagFacingCamera(const Vector3d& centre)
{
    Matrix3d r;
    r.col(0) = Vector3d(0.0, 1.0, 0.0);
    r.col(1) = Vector3d(0.0, 0.0, -1.0);
    r.col(2) = Vector3d(-1.0, 0.0, 0.0);
    Matrix<double, 6, 1> lm;
    lm.segment<3>(0) = centre;
    lm.segment<3>(3) = rot2rpy(r);
    return lm;
}

/// Estimator with the body at the origin (small covariance) and the given tag landmarks (Pose type).
struct Scene
{
    std::shared_ptr<SystemVisualNav> model;
    std::unique_ptr<SystemEstimator> est;
    std::vector<Index> ids;
};

Scene makeScene(const std::vector<Vector3d>& centres)
{
    Scene s;
    s.model = std::make_shared<SystemVisualNav>(Matrix<double, 6, 1>::Constant(0.02));
    VectorXd sig(12);
    // Pixel-level prior (about 2 px at 2.5 m): a 5 cm prior would project to ~18 px and gate everything.
    sig << 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002;
    s.est = std::make_unique<SystemEstimator>(
        s.model, GaussianInfo<double>::fromMoment(VectorXd::Zero(12), sig.cwiseProduct(sig).asDiagonal()), 0.0);
    for (const Vector3d& c : centres)
    {
        const Matrix<double, 6, 1> lm = tagFacingCamera(c);
        Matrix<double, 6, 1> sigL;
        sigL << 0.005, 0.005, 0.005, 0.002, 0.002, 0.002;
        s.ids.push_back(s.model->addLandmark(*s.est, SystemVisualNav::LandmarkType::Pose,
                                             GaussianInfo<double>::fromMoment(lm, sigL.cwiseProduct(sigL).asDiagonal())));
    }
    return s;
}

TagPredictionOptions predOpts()
{
    TagPredictionOptions o;
    o.sigmaPx = kSigmaPx;
    return o;
}

TagAssociationOptions assocOpts()
{
    TagAssociationOptions a;
    a.gateThresholdSq = GaussianBase::mahalanobisThresholdSq(3.0, 8.0);
    a.ambiguityMarginD2 = 4.0;
    return a;
}

/// Detection with the corners given as an 8-vector (canonical order).
tags::TagDetection detectionFrom(const Matrix<double, 8, 1>& uv)
{
    tags::TagDetection d;
    d.id = 5;
    d.corners = tags::vectorToCorners(uv);
    double side = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        side += (d.corners.row(k) - d.corners.row((k + 1) % 4)).norm();
    }
    d.sidePx = 0.25 * side;
    return d;
}

std::vector<TagPredictionInput> inputsFor(const std::vector<Index>& ids)
{
    std::vector<TagPredictionInput> in;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        TagPredictionInput p;
        p.record = i;
        p.id = ids[i];
        p.cornersLocal = tags::canonicalCorners(kSide);
        in.push_back(p);
    }
    return in;
}

} // namespace

SCENARIO("predicted innovation covariances from the union marginal match the reference covariance")
{
    const Camera cam = assocCamera();
    Scene s = makeScene({Vector3d(2.5, -0.15, 0.1), Vector3d(2.7, 0.2, -0.1)});
    GIVEN("two tags in front of the camera with a correlated state")
    {
        WHEN("the candidates are predicted from the triangular-solve path")
        {
            const TagPredictionSet set = slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), predOpts());
            REQUIRE(set.candidates.size() == 2);
            THEN("S_j equals H P H^T + sigma^2 I computed from the reference covariance (relative 1e-7)")
            {
                const MatrixXd p = s.est->covariance();
                for (const slam::TagCandidate& c : set.candidates)
                {
                    MatrixXd h = MatrixXd::Zero(8, s.est->dim());
                    h.block<8, 6>(0, 6) = c.pred.dEta;
                    h.block<8, 6>(0, c.offset) = c.pred.dLm;
                    const Matrix<double, 8, 8> sRef = h * p * h.transpose() + kSigmaPx * kSigmaPx * Matrix<double, 8, 8>::Identity();
                    CAPTURE_EIGEN(c.S);
                    CAPTURE_EIGEN(sRef);
                    CHECK(testutil::nearMatrix(c.S, sRef, 1e-7));
                }
            }
            THEN("both tags are visible (in front, inside the image, facing the camera)")
            {
                CHECK(set.candidates[0].visible);
                CHECK(set.candidates[1].visible);
                CHECK(set.candidates[0].depth > 2.0);
            }
        }
    }

    GIVEN("the joint NIS of the two tags")
    {
        const TagPredictionSet set = slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), predOpts());
        VectorXd innov(16);
        for (int k = 0; k < 16; ++k)
        {
            innov(k) = 1.5 * std::sin(0.9 * k + 0.2);
        }
        WHEN("the NIS is computed with cross-correlations")
        {
            const double nis = slam::jointNis(set, {0, 1}, innov, kSigmaPx);
            THEN("it equals e^T S^{-1} e for the reference joint innovation covariance (dof 16)")
            {
                const MatrixXd p = s.est->covariance();
                MatrixXd h = MatrixXd::Zero(16, s.est->dim());
                for (int j = 0; j < 2; ++j)
                {
                    h.block<8, 6>(8 * j, 6) = set.candidates[static_cast<std::size_t>(j)].pred.dEta;
                    h.block<8, 6>(8 * j, set.candidates[static_cast<std::size_t>(j)].offset) =
                        set.candidates[static_cast<std::size_t>(j)].pred.dLm;
                }
                const MatrixXd sJoint = h * p * h.transpose() + kSigmaPx * kSigmaPx * MatrixXd::Identity(16, 16);
                const double ref = innov.dot(sJoint.ldlt().solve(innov));
                CHECK(nis == doctest::Approx(ref).epsilon(1e-7));
            }
        }
    }
}

SCENARIO("geometric association: gating, birth candidates and the near-centre rule")
{
    const Camera cam = assocCamera();
    Scene s = makeScene({Vector3d(2.5, -0.5, 0.0), Vector3d(2.5, 0.5, 0.0)});
    const TagPredictionSet set = slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), predOpts());
    REQUIRE(set.candidates.size() == 2);

    GIVEN("a detection on each tag and one detection far from both")
    {
        std::vector<tags::TagDetection> dets;
        dets.push_back(detectionFrom(set.candidates[0].pred.uv));
        dets.push_back(detectionFrom(set.candidates[1].pred.uv));
        Matrix<double, 8, 1> far = set.candidates[0].pred.uv;
        for (int k = 0; k < 4; ++k)
        {
            far(2 * k) += 400.0;
            far(2 * k + 1) -= 250.0;
        }
        dets.push_back(detectionFrom(far));
        WHEN("the association is computed")
        {
            const slam::TagAssociationResult r = slam::associateTagDetections(set, dets, assocOpts());
            THEN("each tag detection is matched to its own tag, with no ambiguity or conflict")
            {
                CHECK(r.detectionCandidate[0] == 0);
                CHECK(r.detectionCandidate[1] == 1);
                CHECK(r.candidateDetection[0] == 0);
                CHECK(r.candidateDetection[1] == 1);
                CHECK_FALSE(r.ambiguous[0]);
                CHECK_FALSE(r.conflict[1]);
            }
            THEN("the far detection gates nothing and is a birth candidate")
            {
                CHECK(r.detectionCandidate[2] == -1);
                CHECK(r.birthCandidate[2]);
                CHECK(r.detectionD2[2] == std::numeric_limits<double>::infinity());
            }
        }
    }

    GIVEN("a detection shifted by 40 px from a tag (outside the gate, but within one tag size of its centre)")
    {
        Matrix<double, 8, 1> shifted = set.candidates[0].pred.uv;
        for (int k = 0; k < 4; ++k)
        {
            shifted(2 * k) += 40.0;
        }
        const std::vector<tags::TagDetection> dets = {detectionFrom(shifted)};
        WHEN("the association is computed")
        {
            const slam::TagAssociationResult r = slam::associateTagDetections(set, dets, assocOpts());
            THEN("it is not matched and it is not a birth (a missed association would duplicate the tag)")
            {
                CHECK(r.detectionCandidate[0] == -1);
                CHECK_FALSE(r.birthCandidate[0]);
            }
        }
    }

    GIVEN("two detections on the same tag")
    {
        const std::vector<tags::TagDetection> dets = {detectionFrom(set.candidates[0].pred.uv),
                                                      detectionFrom(set.candidates[0].pred.uv + Matrix<double, 8, 1>::Constant(0.5))};
        WHEN("the association is computed")
        {
            const slam::TagAssociationResult r = slam::associateTagDetections(set, dets, assocOpts());
            THEN("the tag keeps the better detection and the other one is a conflict, not a birth")
            {
                CHECK(r.candidateDetection[0] >= 0);
                CHECK(r.conflict[static_cast<std::size_t>(1 - r.candidateDetection[0])]);
                CHECK_FALSE(r.birthCandidate[1]);
            }
        }
    }
}

SCENARIO("ambiguity: a detection gated by two candidates with similar cost is rejected")
{
    const Camera cam = assocCamera();
    Scene s = makeScene({Vector3d(2.5, -0.5, 0.0)});
    TagPredictionSet set = slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), predOpts());
    REQUIRE(set.candidates.size() == 1);
    // Two candidates that predict the same corners (the identical-tag situation with an uncertain pose).
    set.candidates.push_back(set.candidates[0]);
    set.candidates[1].id = 999;
    set.candidates[1].record = 1;

    GIVEN("a detection exactly on the predicted corners of both candidates")
    {
        const std::vector<tags::TagDetection> dets = {detectionFrom(set.candidates[0].pred.uv)};
        WHEN("the association is computed")
        {
            const slam::TagAssociationResult r = slam::associateTagDetections(set, dets, assocOpts());
            THEN("the detection is flagged ambiguous, is not associated and is not a birth")
            {
                CHECK(r.ambiguous[0]);
                CHECK(r.detectionCandidate[0] == -1);
                CHECK(r.candidateDetection[0] == -1);
                CHECK(r.candidateDetection[1] == -1);
                CHECK_FALSE(r.birthCandidate[0]);
                CHECK(r.numGatedPairs == 2);
            }
        }
    }

    GIVEN("the same detection with the second candidate moved 30 px away")
    {
        set.candidates[1].pred.uv(0) += 30.0;
        set.candidates[1].pred.uv(2) += 30.0;
        set.candidates[1].pred.uv(4) += 30.0;
        set.candidates[1].pred.uv(6) += 30.0;
        // Recompute the innovation factor of the moved candidate (its covariance is unchanged).
        const std::vector<tags::TagDetection> dets = {detectionFrom(set.candidates[0].pred.uv)};
        WHEN("the association is computed")
        {
            const slam::TagAssociationResult r = slam::associateTagDetections(set, dets, assocOpts());
            THEN("the detection is associated to the candidate that predicts it (unambiguous)")
            {
                CHECK_FALSE(r.ambiguous[0]);
                CHECK(r.detectionCandidate[0] == 0);
            }
        }
    }
}

SCENARIO("prediction rejects invalid inputs")
{
    const Camera cam = assocCamera();
    Scene s = makeScene({Vector3d(2.5, 0.0, 0.0)});
    THEN("a non-positive sigma and an unset gate are rejected")
    {
        TagPredictionOptions bad = predOpts();
        bad.sigmaPx = 0.0;
        CHECK_THROWS_AS(slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), bad), std::invalid_argument);
        const TagPredictionSet set = slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), predOpts());
        CHECK_THROWS_AS(slam::associateTagDetections(set, {}, TagAssociationOptions{}), std::invalid_argument);
    }
    THEN("an empty input gives an empty set")
    {
        CHECK(slam::predictTagCandidates(*s.est, *s.model, cam, {}, predOpts()).candidates.empty());
    }
}

SCENARIO("innovation factors and the joint NIS equal the explicit-covariance references (square-root forms)")
{
    const Camera cam = assocCamera();
    Scene s = makeScene({Vector3d(2.5, -0.15, 0.1), Vector3d(2.7, 0.2, -0.1)});
    GIVEN("two predicted tag candidates with a correlated state")
    {
        const slam::TagPredictionSet set = slam::predictTagCandidates(*s.est, *s.model, cam, inputsFor(s.ids), predOpts());
        REQUIRE(set.candidates.size() == 2);
        WHEN("each stored factor is checked against S = X^T X + sigma^2 I and the gate distance is evaluated")
        {
            double maxFactorErr = 0.0;
            double maxGateErr = 0.0;
            for (const slam::TagCandidate& c : set.candidates)
            {
                // The gate's own factor R (no product or factorisation of S inside the gate) must satisfy R^T R = S.
                maxFactorErr = std::max(maxFactorErr, (c.sqrtS.transpose() * c.sqrtS - c.S).norm() / c.S.norm());
                Matrix<double, 8, 1> e;
                for (Index k = 0; k < 8; ++k)
                {
                    e(k) = 0.7 * static_cast<double>(k + 1) - 3.0;
                }
                const Matrix<double, 8, 1> q = c.sqrtS.transpose().triangularView<Eigen::Lower>().solve(e);
                const double gate = q.squaredNorm();
                const double ref = e.dot(c.S.ldlt().solve(e));   // test oracle: dense solve on the explicit S
                maxGateErr = std::max(maxGateErr, std::abs(gate - ref) / std::max(1.0, std::abs(ref)));
            }
            CAPTURE(maxFactorErr);
            CAPTURE(maxGateErr);
            THEN("R^T R equals S to 1e-10 (relative Frobenius)")
            {
                CHECK(maxFactorErr <= 1e-10);
            }
            THEN("the gate distance e^T S^{-1} e equals the dense reference to 1e-10 (relative)")
            {
                CHECK(maxGateErr <= 1e-10);
            }
        }
        WHEN("the joint NIS of both candidates is computed from the union factor")
        {
            const std::vector<std::size_t> matched = {0, 1};
            Eigen::VectorXd innov(16);
            for (Index k = 0; k < 16; ++k)
            {
                innov(k) = 0.4 * std::sin(0.9 * static_cast<double>(k)) + 0.05;
            }
            const double nis = slam::jointNis(set, matched, innov, kSigmaPx);
            // Reference: explicit joint S over the union marginal (test oracle), dense solve.
            const Index nu = static_cast<Index>(set.unionIndices.size());
            MatrixXd hStack = MatrixXd::Zero(16, nu);
            for (std::size_t k = 0; k < matched.size(); ++k)
            {
                const slam::TagCandidate& c = set.candidates[matched[k]];
                hStack.block<8, 6>(8 * static_cast<Index>(k), 0) = c.pred.dEta;
                hStack.block<8, 6>(8 * static_cast<Index>(k), c.unionOffset) = c.pred.dLm;
            }
            const MatrixXd xiInv = set.xiUnion.inverse();
            const MatrixXd pUnion = xiInv * xiInv.transpose();
            const MatrixXd sJoint = hStack * pUnion * hStack.transpose() + kSigmaPx * kSigmaPx * MatrixXd::Identity(16, 16);
            const double ref = innov.dot(sJoint.ldlt().solve(innov));
            CAPTURE(nis);
            CAPTURE(ref);
            THEN("the joint NIS equals the dense reference to 1e-10 (relative)")
            {
                CHECK(std::abs(nis - ref) / std::max(1.0, std::abs(ref)) <= 1e-10);
            }
        }
    }
}
