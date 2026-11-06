#ifndef S6KEYFRAMEFLOW_H
#define S6KEYFRAMEFLOW_H

/**
 * @file S6KeyframeFlow.h
 * @brief Keyframe flow front end of Scenario 6 (the keyframe-to-current epipolar measurement and its model choice).
 *
 * The measurement itself is MeasurementIndoorFlow (unchanged, shared with Scenario 5). This front end differs from
 * IndoorFlowFrontEnd (which Scenario 5 keeps) in two documented points:
 *  1. The epipolar consensus is the RANSAC inlier set of the five-point essential matrix (cv::findEssentialMat), which
 *     does not depend on the predicted state. IndoorFlowFrontEnd additionally gates the Sampson residual at the
 *     predicted state (1.5 px for the default sigma), which removed every correspondence in the first epipolar frames of
 *     the walking sequence: the prediction was not yet accurate, the keyframe restarted at every frame and the baseline
 *     never grew (observed, see test/src/S6WalkingCamera.cpp).
 *  2. The essential matrix also gives the relative pose (recoverPose, cheirality-resolved, unit translation). It seeds the
 *     velocity direction once (SlamPointPipeline::seedVelocityFromEssential); it is not a measurement.
 *
 * Model choice (unchanged in rule, see IndoorFlowFrontEnd.h): Model R when the median rotation-only misfit under the
 * predicted relative rotation is below rotationMisfitPx (no translation evidence); otherwise Model E. Because the
 * keyframe baseline grows over the keyframe's life, Model R is selected only while the keyframe has negligible parallax.
 */

#include <limits>
#include <vector>
#include <Eigen/Core>
#include "Camera.h"
#include "IndoorFlowFrontEnd.h"
#include "MeasurementBase.h"
#include "MeasurementIndoorFlow.h"
#include "SystemEstimator.h"

namespace slam
{

/** @brief Result of the keyframe flow front end for one frame. */
struct S6FlowFrame
{
    IndoorFlowFrame flow;                 ///< model, pairs (normalised), validity and counters (IndoorFlowFrontEnd layout)
    bool hasRelative = false;             ///< relative pose recovered from the essential matrix (Model E only)
    Eigen::Matrix3d rRel = Eigen::Matrix3d::Identity();   ///< p_B = rRel p_A + tRel (camera A = keyframe, B = current)
    Eigen::Vector3d tRel = Eigen::Vector3d::Zero();       ///< unit translation direction (scale not observable)
};

/**
 * @brief Builds the keyframe-to-current flow measurement inputs of one frame.
 * @param camera   calibrated camera
 * @param est      estimator after the time update (etaPrev = keyframe pose, eta = current pose)
 * @param matches  correspondences from the keyframe (prevPx) to the current frame (currPx)
 * @param opt      flow options (sigmaPx, ransacThresholdPx, rotationMisfitPx, minCorrespondences)
 * @throws std::invalid_argument if the camera is invalid or the state does not contain etaPrev
 */
S6FlowFrame buildKeyframeFlowFrame(const Camera& camera, const SystemEstimator& est,
                                   const std::vector<IndoorFlowPixelMatch>& matches, const IndoorFlowOptions& opt);

/**
 * @brief Cheirality census of correspondences under a relative pose p_B = R p_A + t (normalised rays x, y).
 *
 * For a correspondence the depths lambda (camera A) and mu (camera B) satisfy mu y - lambda R x = t. Taking cross
 * products with y and with R x gives the closed forms
 *   lambda = -((t x y) . (R x x y)) / |R x x y|^2,    mu = ((t x R x) . (y x R x)) / |y x R x|^2,
 * so the point is in front of both cameras iff lambda > 0 and mu > 0. Correspondences with (nearly) parallel rays have
 * no determined depth sign and are counted as undetermined. The epipolar residual cannot tell (R, t) from (R, -t): the
 * census is the cheirality test that does (see SlamPointPipeline::step).
 */
struct CheiralityCensus
{
    int total = 0;      ///< correspondences with a determined depth sign
    int positive = 0;   ///< of which positive depth in both cameras
    int undetermined = 0;   ///< correspondences with (nearly) parallel rays (not in total)

    /** @brief positive / total in [0, 1]; NaN if total is zero. */
    double fractionPositive() const;
};

/** @brief Cheirality census of the pairs under p_B = r p_A + t (t need not be unit). */
CheiralityCensus cheiralityCensus(const std::vector<IndoorFlowPair>& pairs, const Eigen::Matrix3d& r,
                                  const Eigen::Vector3d& t);

/** @brief Result of updateFlowWithCheirality. */
struct FlowUpdateResult
{
    MeasurementUpdateResult update;                 ///< MAP result of the kept update (the state holds its posterior)
    bool mirrored = false;                          ///< the translation branch was mirrored (t -> -t) and re-fitted
    double fractionBefore = std::numeric_limits<double>::quiet_NaN();  ///< positive-depth fraction of the first posterior
    double fractionAfter = std::numeric_limits<double>::quiet_NaN();   ///< positive-depth fraction of the kept posterior
};

/**
 * @brief Flow measurement update with a cheirality check of the translation branch (Scenario 6, see file documentation).
 *
 * The epipolar likelihood is invariant to t -> -t (and to the scale of t): the posterior translation can settle in the
 * mirrored branch, where most matched points lie behind the cameras. The update is run once. If the majority of the
 * correspondences is behind a camera at that posterior, the update is repeated from the PRIOR (before the first update,
 * so no flow information is counted twice) after mirrorKeyframeTranslation, and the branch with the larger positive-depth
 * fraction is kept. The census needs at least kMinCensusPairs determined correspondences.
 *
 * @param est     estimator, prior state at the measurement time (time update is done inside the measurement)
 * @param camera  calibrated camera (extrinsics for the relative pose)
 * @param fm      flow measurement (pairs and model)
 * @param opts    MAP options
 */
FlowUpdateResult updateFlowWithCheirality(SystemEstimator& est, const Camera& camera, const MeasurementIndoorFlow& fm,
                                          const FuncMinOptions& opts);

/**
 * @brief Information-form mirror of camera B about camera A: c_B -> 2 c_A - c_B and the body velocity v -> -v.
 *
 * Rotations are unchanged, so the relative pose becomes (R, -t): the other branch of the epipolar likelihood. The
 * velocity is mirrored with the position, so that the next time update moves B away from A on the same side. The shift is
 * a mean shift in information form (mean += Lambda delta = Xi^T (Xi delta)); no covariance is formed or inverted.
 * @throws std::invalid_argument if the state does not contain the velocity and the keyframe and current poses
 */
void mirrorKeyframeTranslation(SystemEstimator& est, const Camera& camera);

/// Minimum number of determined correspondences for a cheirality decision.
inline constexpr int kMinCensusPairs = 8;

} // namespace slam

#endif
