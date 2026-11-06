#include "SlamLandmarkMap.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace slam
{

SlamLandmarkMap::SlamLandmarkMap(const SlamConfig& config) : config_(config)
{
    if (config_.maxLandmarks <= 0 || config_.confirmHits <= 0 || config_.tentativeMaxMisses <= 0
        || config_.confirmedMaxMisses <= 0 || config_.maxUnseenFrames <= 0 || config_.maxBehindFrames <= 0)
    {
        throw std::invalid_argument("SlamLandmarkMap: lifecycle limits must be positive");
    }
}

bool SlamLandmarkMap::contains(Eigen::Index id) const
{
    return std::any_of(records_.begin(), records_.end(), [id](const SlamLandmarkRecord& r) { return r.id == id; });
}

SlamLandmarkRecord& SlamLandmarkMap::at(Eigen::Index id)
{
    for (SlamLandmarkRecord& r : records_)
    {
        if (r.id == id)
        {
            return r;
        }
    }
    throw std::out_of_range("SlamLandmarkMap::at: unknown landmark id " + std::to_string(id));
}

const SlamLandmarkRecord& SlamLandmarkMap::at(Eigen::Index id) const
{
    for (const SlamLandmarkRecord& r : records_)
    {
        if (r.id == id)
        {
            return r;
        }
    }
    throw std::out_of_range("SlamLandmarkMap::at: unknown landmark id " + std::to_string(id));
}

void SlamLandmarkMap::insert(const SlamLandmarkRecord& record)
{
    if (contains(record.id))
    {
        throw std::invalid_argument("SlamLandmarkMap::insert: duplicate landmark id " + std::to_string(record.id));
    }
    records_.push_back(record);
}

void SlamLandmarkMap::erase(Eigen::Index id)
{
    const auto it = std::find_if(records_.begin(), records_.end(), [id](const SlamLandmarkRecord& r) { return r.id == id; });
    if (it == records_.end())
    {
        throw std::out_of_range("SlamLandmarkMap::erase: unknown landmark id " + std::to_string(id));
    }
    records_.erase(it);
}

std::vector<Eigen::Index> selectLandmarksToCull(const SlamLandmarkMap& map, int frame)
{
    const SlamConfig& cfg = map.config();
    std::vector<Eigen::Index> cull;
    std::vector<const SlamLandmarkRecord*> keep;
    for (const SlamLandmarkRecord& r : map.records())
    {
        const bool unconfirmedDead = !r.confirmed && r.misses >= cfg.tentativeMaxMisses;
        const bool confirmedDead = r.confirmed && r.misses >= cfg.confirmedMaxMisses;
        const bool behind = r.behindFrames >= cfg.maxBehindFrames;
        const bool unseen = (frame - r.lastSeenFrame) > cfg.maxUnseenFrames;
        if (unconfirmedDead || confirmedDead || behind || unseen)
        {
            cull.push_back(r.id);
        }
        else
        {
            keep.push_back(&r);
        }
    }

    // Capacity: remove the weakest survivors (most misses, then the oldest observation) until the map fits.
    const std::size_t remaining = keep.size();
    const std::size_t capacity = static_cast<std::size_t>(cfg.maxLandmarks);
    if (remaining > capacity)
    {
        std::sort(keep.begin(), keep.end(),
                  [](const SlamLandmarkRecord* a, const SlamLandmarkRecord* b)
                  {
                      if (a->misses != b->misses)
                      {
                          return a->misses > b->misses;
                      }
                      return a->lastSeenFrame < b->lastSeenFrame;
                  });
        for (std::size_t i = 0; i < remaining - capacity; ++i)
        {
            cull.push_back(keep[i]->id);
        }
    }
    return cull;
}

} // namespace slam
