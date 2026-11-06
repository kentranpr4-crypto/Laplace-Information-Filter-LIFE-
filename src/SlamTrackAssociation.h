#ifndef SLAMTRACKASSOCIATION_H
#define SLAMTRACKASSOCIATION_H

/**
 * @file SlamTrackAssociation.h
 * @brief Landmark-to-track association of Scenario 6: persistent LK track identity first, descriptor re-acquisition second.
 *
 * Each landmark records the ImageFlow track id it was born from (SlamLandmarkRecord::trackId). Per frame:
 *  1. Identity: if the landmark's track is still alive and its pixel passes the chi2 gate of the predicted pixel
 *     (S = pixelCov + sigma^2 I, nSigma = gateNSigma, dof 2), the track pixel is the observation.
 *  2. Re-acquisition: otherwise the landmark may take a free track (not owned by another landmark) inside the same gate
 *     whose descriptor is within maxHamming bits of the landmark descriptor. The descriptor of a track is computed at
 *     the track pixel by PointFeatureExtractor::describeAt, and the landmark descriptor was computed the same way at its
 *     birth, so the two are comparable (cv::ORB detections at coarser pyramid levels are not, see the file notes of
 *     SlamPointPipeline.h).
 * A landmark with two gated re-acquisition candidates within ambiguityMargin bits is ambiguous and not associated; a track
 * claimed by two landmarks is rejected for both (as in SlamAssociation).
 *
 * The result reuses AssociationResult, so the measurement update, the innovation covariance and the overlay are shared
 * with the ORB-based association.
 */

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "Camera.h"
#include "PointFeatures.h"
#include "SlamAssociation.h"
#include "SlamLandmarkMap.h"

namespace slam
{

/**
 * @brief Associates the predicted landmarks with the active LK tracks.
 * @param predictions   predicted pixels of all landmarks (SlamAssociation::predictLandmarks)
 * @param tracks        active tracks as features: pixel = current position, id = ImageFlow track id (in trackIds)
 * @param trackIds      ImageFlow track id of each entry of tracks
 * @param described     per track: true if its descriptor is valid (re-acquisition needs a descriptor)
 * @param map           landmark records (trackId, descriptor)
 * @param camera        camera (pixel noise)
 * @param cfg           configuration (gate, maxHamming, ambiguityMargin, pixel noise)
 * @return the accepted pairs in the AssociationResult layout (matchedFeature indexes tracks)
 */
AssociationResult associateTracks(const PredictionSet& predictions, const std::vector<PointFeature>& tracks,
                                  const std::vector<int>& trackIds, const std::vector<bool>& described,
                                  const SlamLandmarkMap& map, const Camera& camera, const SlamConfig& cfg);

} // namespace slam

#endif
