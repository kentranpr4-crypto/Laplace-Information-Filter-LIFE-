#include "SlamTrackKeyframes.h"

#include <cmath>

namespace slam
{

void SlamTrackKeyframes::start(int frameIndex, const std::vector<ImageFlowMatch>& active)
{
    kfPixel_.clear();
    born_.clear();
    kfPixel_.reserve(active.size());
    for (const ImageFlowMatch& m : active)
    {
        kfPixel_[m.id] = Eigen::Vector2d(m.curr.x, m.curr.y);
    }
    keyframeFrame_ = frameIndex;
}

std::vector<KeyframeTrack> SlamTrackKeyframes::alive(const std::vector<ImageFlowMatch>& active) const
{
    std::vector<KeyframeTrack> out;
    out.reserve(active.size());
    for (const ImageFlowMatch& m : active)
    {
        const auto it = kfPixel_.find(m.id);
        if (it == kfPixel_.end())
        {
            continue;   // started after the keyframe
        }
        KeyframeTrack t;
        t.id = m.id;
        t.kfPixel = it->second;
        t.pixel = Eigen::Vector2d(m.curr.x, m.curr.y);
        t.age = m.age;
        out.push_back(t);
    }
    return out;
}

void SlamTrackKeyframes::markBorn(int trackId)
{
    born_.insert(trackId);
}

bool SlamTrackKeyframes::isBorn(int trackId) const
{
    return born_.count(trackId) > 0;
}

bool keyframeShouldStart(const SlamConfig& cfg, bool flowValid, double medianParallaxPx, int aliveTracks,
                         int framesSinceKeyframe)
{
    if (framesSinceKeyframe <= 0)
    {
        return false;   // the keyframe was started at this frame
    }
    if (!cfg.keyframeFlow)
    {
        return true;    // ablation: consecutive-frame flow (every frame is a keyframe)
    }
    if (framesSinceKeyframe >= cfg.keyframeMaxAge || aliveTracks < cfg.keyframeMinTracks || !flowValid)
    {
        return true;
    }
    return std::isfinite(medianParallaxPx) && medianParallaxPx > cfg.keyframeParallaxPx;
}

} // namespace slam
