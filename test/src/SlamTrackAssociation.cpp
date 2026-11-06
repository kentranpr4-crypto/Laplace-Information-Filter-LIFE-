// Tests of SlamTrackAssociation (Scenario 6): identity association through the persistent LK track, descriptor
// re-acquisition of a free track, ambiguity between two equally good free tracks, and track ownership.
#include <cstdint>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Core>
#include "../../src/Camera.h"
#include "../../src/PointFeatures.h"
#include "../../src/SlamLandmarkMap.h"
#include "../../src/SlamTrackAssociation.h"

namespace
{

Camera associationCamera()
{
    return Camera(640, 480, (Eigen::Matrix3d() << 400.0, 0.0, 319.5, 0.0, 400.0, 239.5, 0.0, 0.0, 1.0).finished(), {});
}

slam::Descriptor descriptorWith(std::uint8_t value)
{
    slam::Descriptor d{};
    d.fill(value);
    return d;
}

/// One visible prediction at pixel px with a tight state covariance (the gate is then about 1 px + sigma).
slam::PredictionSet onePrediction(Eigen::Index id, const Eigen::Vector2d& px)
{
    slam::PredictionSet set;
    slam::LandmarkPrediction p;
    p.id = id;
    p.offset = 18;
    p.visible = true;
    p.depth = 3.0;
    p.pixel = px;
    p.pixelCov = Eigen::Matrix2d::Identity() * 0.25;
    p.sigmaPx = 1.0;
    set.predictions.push_back(p);
    set.whitenedProjection = Eigen::MatrixXd::Zero(18 + 3, 2);
    return set;
}

slam::SlamLandmarkRecord record(Eigen::Index id, int trackId, const slam::Descriptor& d)
{
    slam::SlamLandmarkRecord r;
    r.id = id;
    r.trackId = trackId;
    r.descriptor = d;
    return r;
}

slam::PointFeature track(const Eigen::Vector2d& px, const slam::Descriptor& d)
{
    slam::PointFeature f;
    f.pixel = px;
    f.descriptor = d;
    return f;
}

} // namespace

SCENARIO("track association: a landmark observes its own live track (identity) before any re-acquisition")
{
    GIVEN("one landmark observing track 7 at pixel (100, 100), and a second live track far away")
    {
        const Camera cam = associationCamera();
        slam::SlamConfig cfg{};
        slam::SlamLandmarkMap map(cfg);
        map.insert(record(0, 7, descriptorWith(0x0F)));
        const slam::PredictionSet preds = onePrediction(0, Eigen::Vector2d(100.0, 100.0));
        const std::vector<slam::PointFeature> tracks = {track(Eigen::Vector2d(101.0, 100.5), descriptorWith(0x0F)),
                                                        track(Eigen::Vector2d(300.0, 300.0), descriptorWith(0xF0))};
        const std::vector<int> ids = {7, 9};
        const std::vector<bool> described = {true, true};
        WHEN("the frame is associated")
        {
            const slam::AssociationResult a = slam::associateTracks(preds, tracks, ids, described, map, cam, cfg);
            THEN("the landmark is matched to its own track and the track pixel is the observation")
            {
                REQUIRE(a.observations.size() == 1);
                CHECK(a.matchedIds[0] == 0);
                CHECK(a.matchedFeature[0] == 0);
                CHECK(a.observations[0].pixel.isApprox(Eigen::Vector2d(101.0, 100.5)));
                CHECK(a.innovation.size() == 2);
                CHECK(a.innovation.isApprox(Eigen::Vector2d(1.0, 0.5)));
            }
        }
    }
}

SCENARIO("track association: a lost track is re-acquired by descriptor inside the gate; ownership and ambiguity are respected")
{
    GIVEN("a landmark whose track 7 is no longer alive, and a free track 9 inside its gate with the same descriptor")
    {
        const Camera cam = associationCamera();
        slam::SlamConfig cfg{};
        slam::SlamLandmarkMap map(cfg);
        map.insert(record(0, 7, descriptorWith(0x0F)));
        const slam::PredictionSet preds = onePrediction(0, Eigen::Vector2d(100.0, 100.0));
        WHEN("a free track with a close descriptor is inside the gate")
        {
            const std::vector<slam::PointFeature> tracks = {track(Eigen::Vector2d(102.0, 99.0), descriptorWith(0x0F))};
            const slam::AssociationResult a =
                slam::associateTracks(preds, tracks, {9}, {true}, map, cam, cfg);
            THEN("the free track is taken")
            {
                REQUIRE(a.observations.size() == 1);
                CHECK(a.matchedFeature[0] == 0);
                CHECK(a.observations[0].pixel.isApprox(Eigen::Vector2d(102.0, 99.0)));
            }
        }
        WHEN("the only candidate has a descriptor far away (Hamming distance above maxHamming)")
        {
            const std::vector<slam::PointFeature> tracks = {track(Eigen::Vector2d(102.0, 99.0), descriptorWith(0xF0))};
            const slam::AssociationResult a =
                slam::associateTracks(preds, tracks, {9}, {true}, map, cam, cfg);
            THEN("no observation is made and the gated prediction is counted as no-candidate")
            {
                CHECK(a.observations.empty());
                CHECK(a.numNoCandidate == 1);
            }
        }
        WHEN("two free tracks inside the gate have the same descriptor")
        {
            const std::vector<slam::PointFeature> tracks = {track(Eigen::Vector2d(101.0, 99.0), descriptorWith(0x0F)),
                                                            track(Eigen::Vector2d(99.0, 101.0), descriptorWith(0x0F))};
            const slam::AssociationResult a =
                slam::associateTracks(preds, tracks, {9, 10}, {true, true}, map, cam, cfg);
            THEN("the landmark is ambiguous and not associated")
            {
                CHECK(a.observations.empty());
                CHECK(a.numAmbiguous == 1);
            }
        }
        WHEN("the free-looking track is owned by another landmark")
        {
            slam::SlamLandmarkMap owned(cfg);
            owned.insert(record(0, 7, descriptorWith(0x0F)));
            owned.insert(record(1, 9, descriptorWith(0x0F)));
            const std::vector<slam::PointFeature> tracks = {track(Eigen::Vector2d(102.0, 99.0), descriptorWith(0x0F))};
            const slam::AssociationResult a =
                slam::associateTracks(preds, tracks, {9}, {true}, owned, cam, cfg);
            THEN("landmark 0 cannot take a track owned by landmark 1")
            {
                CHECK(a.observations.empty());
            }
        }
        WHEN("the candidate track has no valid descriptor (image border)")
        {
            const std::vector<slam::PointFeature> tracks = {track(Eigen::Vector2d(102.0, 99.0), descriptorWith(0x0F))};
            const slam::AssociationResult a =
                slam::associateTracks(preds, tracks, {9}, {false}, map, cam, cfg);
            THEN("it is not used for re-acquisition")
            {
                CHECK(a.observations.empty());
            }
        }
    }
}
