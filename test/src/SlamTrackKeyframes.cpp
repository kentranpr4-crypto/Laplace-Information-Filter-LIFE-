// Tests of SlamTrackKeyframes: keyframe anchoring of persistent LK tracks, birth bookkeeping, and the keyframe switching
// rule (boundaries and the ablation switch). Pure logic: no image or estimator is involved.
#include <cmath>
#include <limits>
#include <vector>
#include <doctest/doctest.h>
#include "../../src/ImageFlow.h"
#include "../../src/SlamLandmarkMap.h"
#include "../../src/SlamTrackKeyframes.h"

namespace
{

ImageFlowMatch track(int id, double x, double y, int age)
{
    ImageFlowMatch m;
    m.id = id;
    m.age = age;
    m.curr = cv::Point2d(x, y);
    return m;
}

} // namespace

SCENARIO("keyframe tracks are anchored at the keyframe and followed while they live")
{
    GIVEN("three live tracks at the keyframe frame 10")
    {
        slam::SlamTrackKeyframes kf;
        REQUIRE_FALSE(kf.hasKeyframe());
        const std::vector<ImageFlowMatch> atKeyframe = {track(1, 100.0, 50.0, 0), track(2, 200.0, 60.0, 3),
                                                        track(3, 300.0, 70.0, 7)};
        kf.start(10, atKeyframe);
        REQUIRE(kf.hasKeyframe());
        CHECK(kf.keyframeFrame() == 10);
        CHECK(kf.anchoredCount() == 3);

        WHEN("the next frame keeps tracks 2 and 3, drops track 1 and starts track 4")
        {
            const std::vector<ImageFlowMatch> next = {track(2, 205.0, 61.0, 4), track(3, 310.0, 69.0, 8),
                                                      track(4, 40.0, 40.0, 0)};
            const std::vector<slam::KeyframeTrack> alive = kf.alive(next);
            THEN("only the anchored tracks are alive, with their keyframe pixels and their current pixels")
            {
                REQUIRE(alive.size() == 2);
                CHECK(alive[0].id == 2);
                CHECK(alive[1].id == 3);
                CHECK(alive[0].kfPixel.isApprox(Eigen::Vector2d(200.0, 60.0)));
                CHECK(alive[0].pixel.isApprox(Eigen::Vector2d(205.0, 61.0)));
                CHECK(alive[1].kfPixel.isApprox(Eigen::Vector2d(300.0, 70.0)));
                CHECK(alive[1].age == 8);
            }
            THEN("a track that started after the keyframe is never a keyframe track")
            {
                for (const slam::KeyframeTrack& t : kf.alive(next))
                {
                    CHECK(t.id != 4);
                }
            }
            THEN("a track becomes unavailable for births once it has produced a landmark")
            {
                CHECK_FALSE(kf.isBorn(2));
                kf.markBorn(2);
                CHECK(kf.isBorn(2));
                CHECK_FALSE(kf.isBorn(3));
            }
        }
        WHEN("a new keyframe starts at frame 25")
        {
            kf.markBorn(1);
            const std::vector<ImageFlowMatch> later = {track(2, 210.0, 62.0, 9), track(4, 40.0, 40.0, 5)};
            kf.start(25, later);
            THEN("the keyframe moves, the births are forgotten and only the new tracks are anchored")
            {
                CHECK(kf.keyframeFrame() == 25);
                CHECK_FALSE(kf.isBorn(1));
                CHECK(kf.anchoredCount() == 2);
                const std::vector<slam::KeyframeTrack> alive = kf.alive(later);
                REQUIRE(alive.size() == 2);
                CHECK(alive[0].kfPixel.isApprox(Eigen::Vector2d(210.0, 62.0)));
            }
        }
    }
}

SCENARIO("the keyframe switching rule follows the translation parallax, the track count and the keyframe age")
{
    GIVEN("the default Scenario 6 configuration (keyframeParallaxPx 40, keyframeMinTracks 30, keyframeMaxAge 90)")
    {
        const slam::SlamConfig cfg{};
        const double nan = std::numeric_limits<double>::quiet_NaN();
        THEN("a small translation parallax keeps the keyframe")
        {
            CHECK_FALSE(slam::keyframeShouldStart(cfg, true, 10.0, 100, 3));
            CHECK_FALSE(slam::keyframeShouldStart(cfg, true, 40.0, 100, 3));   // boundary: strictly greater starts a keyframe
        }
        THEN("a parallax above the threshold starts a new keyframe")
        {
            CHECK(slam::keyframeShouldStart(cfg, true, 40.5, 100, 3));
        }
        THEN("too few live tracks, an invalid flow or an old keyframe start a new keyframe")
        {
            CHECK(slam::keyframeShouldStart(cfg, true, 10.0, 29, 3));
            CHECK(slam::keyframeShouldStart(cfg, false, nan, 100, 3));
            CHECK(slam::keyframeShouldStart(cfg, true, 10.0, 100, 90));
        }
        THEN("a keyframe started at this frame is never restarted in the same frame")
        {
            CHECK_FALSE(slam::keyframeShouldStart(cfg, false, nan, 0, 0));
            CHECK_FALSE(slam::keyframeShouldStart(cfg, true, 99.0, 100, 0));
        }
        THEN("an unknown parallax (NaN) with a valid flow does not start a keyframe by itself")
        {
            CHECK_FALSE(slam::keyframeShouldStart(cfg, true, nan, 100, 3));
        }
        THEN("the consecutive-frame ablation starts a keyframe at every frame")
        {
            slam::SlamConfig ablation = cfg;
            ablation.keyframeFlow = false;
            CHECK(slam::keyframeShouldStart(ablation, true, 0.0, 100, 1));
            CHECK_FALSE(slam::keyframeShouldStart(ablation, true, 0.0, 100, 0));
        }
    }
}
