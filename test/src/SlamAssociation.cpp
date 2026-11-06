// Data association: chi2 gate, descriptor check, ambiguity and conflict rejection; predicted pixel covariance.
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include "../../src/Camera.h"
#include "../../src/GaussianInfo.h"
#include "../../src/MeasurementPointBundle.h"
#include "../../src/PointFeatures.h"
#include "../../src/SlamAssociation.h"
#include "../../src/SlamInitialisation.h"
#include "../../src/SlamLandmarkMap.h"
#include "../../src/SystemEstimator.h"
#include "../../src/SystemVisualNav.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using GaussianD = GaussianInfo<double>;
using slam::Descriptor;
using slam::LandmarkPrediction;
using slam::PointFeature;
using slam::PredictionSet;
using slam::SlamConfig;
using slam::SlamLandmarkMap;
using slam::SlamLandmarkRecord;
using testutil::nearMatrix;
using testutil::randomSpd;
using testutil::randomVector;

namespace
{

Camera assocCamera()
{
    return Camera(1920, 1080, (Matrix3d() << 889.3, 0.0, 959.5, 0.0, 892.3, 539.5, 0.0, 0.0, 1.0).finished(),
                  {-0.0246, -0.0061, 0.0005, 0.0001, 0.0013});
}

Descriptor descriptorWithBits(int ones)
{
    Descriptor d{};
    for (int i = 0; i < ones; ++i)
    {
        d[static_cast<std::size_t>(i / 8)] |= static_cast<std::uint8_t>(1u << (i % 8));
    }
    return d;
}

PointFeature feature(double u, double v, const Descriptor& d)
{
    PointFeature f;
    f.pixel = Vector2d(u, v);
    f.descriptor = d;
    f.response = 1.0f;
    return f;
}

LandmarkPrediction prediction(Index id, Index offset, double u, double v, double var, double sigma)
{
    LandmarkPrediction p;
    p.id = id;
    p.offset = offset;
    p.visible = true;
    p.depth = 3.0;
    p.pixel = Vector2d(u, v);
    p.pixelCov = var * Matrix2d::Identity();
    p.sigmaPx = sigma;
    return p;
}

PredictionSet makeSet(const std::vector<LandmarkPrediction>& preds)
{
    PredictionSet s;
    s.predictions = preds;
    // No state uncertainty in the synthetic set: the innovation covariance is R alone.
    s.whitenedProjection = MatrixXd::Zero(18, 2 * static_cast<Index>(preds.size()));
    return s;
}

SlamLandmarkMap mapWith(const std::vector<std::pair<Index, Descriptor>>& records, const SlamConfig& cfg)
{
    SlamLandmarkMap map(cfg);
    for (const auto& [id, d] : records)
    {
        SlamLandmarkRecord r;
        r.id = id;
        r.descriptor = d;
        map.insert(r);
    }
    return map;
}

} // namespace

SCENARIO("associateFeatures: a gated feature with a matching descriptor is accepted")
{
    GIVEN("one visible landmark predicted at (100, 100) with innovation variance 4 px^2 and sigma 2 px")
    {
        const SlamConfig cfg;
        const Camera cam = assocCamera();
        const Descriptor d = descriptorWithBits(0);
        SlamLandmarkMap map = mapWith({{0, d}}, cfg);
        const PredictionSet preds = makeSet({prediction(0, 12, 100.0, 100.0, 4.0, 2.0)});
        const std::vector<PointFeature> feats = {feature(101.0, 100.0, d), feature(300.0, 300.0, d)};
        WHEN("the association is computed")
        {
            const slam::AssociationResult res = slam::associateFeatures(preds, feats, map, cam, cfg);
            THEN("the nearby feature is matched and the far one is not used")
            {
                REQUIRE(res.observations.size() == 1);
                CHECK(res.matchedIds[0] == 0);
                CHECK(res.matchedFeature[0] == 0);
                CHECK(res.featureUsed[0]);
                CHECK(!res.featureUsed[1]);
                CHECK(res.observations[0].landmarkOffset == 12);
                CHECK(res.observations[0].pixel.isApprox(Vector2d(101.0, 100.0)));
            }
            THEN("the innovation is the feature minus the prediction and S = R (no state term in this set)")
            {
                CHECK(res.innovation.isApprox(Vector2d(1.0, 0.0)));
                const double sigmaSq = res.observations[0].sigmaPx * res.observations[0].sigmaPx;
                CHECK(nearMatrix(res.innovationCov, sigmaSq * MatrixXd::Identity(2, 2), 1e-12));
            }
        }
    }
}

SCENARIO("associateFeatures: a feature outside the 3-sigma chi2 gate is not a candidate")
{
    GIVEN("the same landmark and a feature 20 px away (Mahalanobis distance^2 = 50 > chi2(3, 2) = 11.8)")
    {
        const SlamConfig cfg;
        const Camera cam = assocCamera();
        const Descriptor d = descriptorWithBits(0);
        SlamLandmarkMap map = mapWith({{0, d}}, cfg);
        const PredictionSet preds = makeSet({prediction(0, 12, 100.0, 100.0, 4.0, 2.0)});
        const std::vector<PointFeature> feats = {feature(120.0, 100.0, d)};
        WHEN("the association is computed")
        {
            const slam::AssociationResult res = slam::associateFeatures(preds, feats, map, cam, cfg);
            THEN("no match is made and the landmark is counted as having no candidate")
            {
                CHECK(res.observations.empty());
                CHECK(res.numVisible == 1);
                CHECK(res.numNoCandidate == 1);
            }
        }
    }
}

SCENARIO("associateFeatures: a gated feature with a dissimilar descriptor is rejected")
{
    GIVEN("a feature inside the gate whose Hamming distance (100 bits) exceeds maxHamming (64)")
    {
        const SlamConfig cfg;
        const Camera cam = assocCamera();
        SlamLandmarkMap map = mapWith({{0, descriptorWithBits(0)}}, cfg);
        const PredictionSet preds = makeSet({prediction(0, 12, 100.0, 100.0, 4.0, 2.0)});
        const std::vector<PointFeature> feats = {feature(100.5, 100.0, descriptorWithBits(100))};
        WHEN("the association is computed")
        {
            const slam::AssociationResult res = slam::associateFeatures(preds, feats, map, cam, cfg);
            THEN("the descriptor check rejects it")
            {
                CHECK(res.observations.empty());
                CHECK(res.numNoCandidate == 1);
            }
        }
    }
}

SCENARIO("associateFeatures: two similar gated candidates are ambiguous and both are rejected")
{
    GIVEN("two features on either side of the prediction with identical descriptors")
    {
        const SlamConfig cfg;
        const Camera cam = assocCamera();
        const Descriptor d = descriptorWithBits(0);
        SlamLandmarkMap map = mapWith({{0, d}}, cfg);
        const PredictionSet preds = makeSet({prediction(0, 12, 100.0, 100.0, 4.0, 2.0)});
        const std::vector<PointFeature> feats = {feature(101.0, 100.0, d), feature(99.0, 100.0, d)};
        WHEN("the association is computed")
        {
            const slam::AssociationResult res = slam::associateFeatures(preds, feats, map, cam, cfg);
            THEN("the landmark is rejected as ambiguous (no arbitrary choice)")
            {
                CHECK(res.observations.empty());
                CHECK(res.numAmbiguous == 1);
                CHECK(!res.featureUsed[0]);
                CHECK(!res.featureUsed[1]);
            }
        }
    }
}

SCENARIO("associateFeatures: a clearly better second candidate resolves the ambiguity")
{
    GIVEN("a best candidate with 0 bits and a second gated candidate with 20 bits (margin 12)")
    {
        const SlamConfig cfg;
        const Camera cam = assocCamera();
        const Descriptor d = descriptorWithBits(0);
        SlamLandmarkMap map = mapWith({{0, d}}, cfg);
        const PredictionSet preds = makeSet({prediction(0, 12, 100.0, 100.0, 4.0, 2.0)});
        const std::vector<PointFeature> feats = {feature(101.0, 100.0, d), feature(99.0, 100.0, descriptorWithBits(20))};
        WHEN("the association is computed")
        {
            const slam::AssociationResult res = slam::associateFeatures(preds, feats, map, cam, cfg);
            THEN("the best candidate is accepted")
            {
                REQUIRE(res.observations.size() == 1);
                CHECK(res.matchedFeature[0] == 0);
                CHECK(res.numAmbiguous == 0);
            }
        }
    }
}

SCENARIO("associateFeatures: a feature claimed by two landmarks is rejected for both")
{
    GIVEN("two visible landmarks whose gates contain the same single feature")
    {
        const SlamConfig cfg;
        const Camera cam = assocCamera();
        const Descriptor d = descriptorWithBits(0);
        SlamLandmarkMap map = mapWith({{0, d}, {1, d}}, cfg);
        const PredictionSet preds = makeSet({prediction(0, 12, 100.0, 100.0, 16.0, 2.0), prediction(1, 15, 102.0, 100.0, 16.0, 2.0)});
        const std::vector<PointFeature> feats = {feature(101.0, 100.0, d)};
        WHEN("the association is computed")
        {
            const slam::AssociationResult res = slam::associateFeatures(preds, feats, map, cam, cfg);
            THEN("both are rejected as a conflict")
            {
                CHECK(res.observations.empty());
                CHECK(res.numConflict == 2);
                CHECK(!res.featureUsed[0]);
            }
        }
    }
}

SCENARIO("predictLandmarks: visibility and the pixel covariance J Sigma J^T agree with an explicit-inverse reference")
{
    GIVEN("a pose-and-points state with a full covariance (landmarks in front of the camera, one behind)")
    {
        const Camera cam = assocCamera();
        const SlamConfig cfg;
        auto model = std::make_shared<SystemVisualNav>(cfg.nuPsd);
        VectorXd mu = VectorXd::Zero(12);
        mu.segment<3>(6) = Vector3d(0.1, -0.05, 0.02);
        mu.segment<3>(9) = Vector3d(0.03, -0.04, 0.12);
        SystemEstimator est(model, GaussianD::fromMoment(mu, 0.02 * randomSpd(12, 7001)), 0.0);
        const std::vector<Vector3d> pts = {Vector3d(2.5, 0.4, 0.3), Vector3d(3.0, -0.6, -0.2), Vector3d(-2.0, 0.1, 0.0)};
        for (const Vector3d& p : pts)
        {
            model->addLandmark(est, SystemVisualNav::LandmarkType::Point,
                               GaussianD::fromMoment(p, 0.05 * Matrix3d::Identity()));
        }
        // Enrich the covariance with off-diagonal structure (pose-landmark correlations).
        VectorXd all = est.mean();
        est.setState(GaussianD::fromMoment(all, 0.03 * randomSpd(est.dim(), 7002)));
        WHEN("the landmarks are predicted")
        {
            const PredictionSet ps = slam::predictLandmarks(est, *model, cam, cfg);
            THEN("the first two landmarks are visible and the one behind the camera is not")
            {
                REQUIRE(ps.predictions.size() == 3);
                CHECK(ps.predictions[0].visible);
                CHECK(ps.predictions[1].visible);
                CHECK(!ps.predictions[2].visible);
                CHECK(ps.predictions[2].depth < 0.0);
            }
            THEN("each pixel covariance equals J Sigma J^T with Sigma = (Xi^T Xi)^{-1} (explicit inverse in the test only)")
            {
                const MatrixXd sigma = est.state().information().inverse();
                const VectorXd muNow = est.mean();
                for (std::size_t j = 0; j < pts.size(); ++j)
                {
                    const slam::PointPixelJacobian pj = slam::pointPixelJacobian(cam, muNow.segment<3>(6), muNow.segment<3>(9),
                                                                                  muNow.segment<3>(12 + 3 * static_cast<Index>(j)));
                    if (!pj.valid)
                    {
                        continue;
                    }
                    MatrixXd jac = MatrixXd::Zero(2, est.dim());
                    jac.block<2, 6>(0, 6) = pj.dUvdEta;
                    jac.block<2, 3>(0, 12 + 3 * static_cast<Index>(j)) = pj.dUvdLm;
                    const Matrix2d ref = jac * sigma * jac.transpose();
                    CAPTURE(j);
                    CHECK(nearMatrix(ps.predictions[j].pixelCov, ref, 1e-7));
                }
            }
        }
    }
}

SCENARIO("selectBirthFeatures: separation from predictions, spacing between births and the usage flags are respected")
{
    GIVEN("features in decreasing response order and one visible predicted landmark")
    {
        const SlamConfig cfg;
        PredictionSet ps = makeSet({prediction(0, 12, 495.0, 495.0, 1.0, 1.0)});
        const Descriptor d{};
        const std::vector<PointFeature> feats = {
            feature(100.0, 100.0, d),   // 0: birth
            feature(105.0, 100.0, d),   // 1: closer than the spacing to 0 -> skipped
            feature(300.0, 300.0, d),   // 2: birth
            feature(302.0, 300.0, d),   // 3: used by a landmark -> skipped
            feature(500.0, 500.0, d),   // 4: within birthSeparation of the prediction -> skipped
            feature(700.0, 300.0, d)};  // 5: birth
        std::vector<bool> used(feats.size(), false);
        used[3] = true;
        WHEN("births are selected with a generous limit")
        {
            const auto births = slam::selectBirthFeatures(feats, used, ps, cfg, 10);
            THEN("exactly features 0, 2 and 5 are selected, in response order")
            {
                REQUIRE(births.size() == 3);
                CHECK(births[0] == 0);
                CHECK(births[1] == 2);
                CHECK(births[2] == 5);
            }
        }
        WHEN("the limit is 1")
        {
            const auto births = slam::selectBirthFeatures(feats, used, ps, cfg, 1);
            THEN("only the strongest feature is taken")
            {
                REQUIRE(births.size() == 1);
                CHECK(births[0] == 0);
            }
        }
    }
}
