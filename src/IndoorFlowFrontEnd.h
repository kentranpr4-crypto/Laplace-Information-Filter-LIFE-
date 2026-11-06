#ifndef INDOORFLOWFRONTEND_H
#define INDOORFLOWFRONTEND_H

/**
 * @file IndoorFlowFrontEnd.h
 * @brief Per-frame preparation of the depth-free flow measurement (MeasurementIndoorFlow) for Scenarios 5 and 6.
 *
 * Steps (all evaluated once per frame, after the delayed time update, i.e. at the predicted state):
 *  1. Normalise the pixel correspondences with Camera::pixelToNormalised (undistorted rays; non-finite pairs dropped).
 *  2. Model selection from the prior relative rotation R_p (the rotation of the predicted states):
 *       misfit_i = f * || pi(R_p x_i) - y_i ||  [px],   median over the correspondences.
 *     If the median is below rotationMisfitPx the matches are explained by a pure rotation (small parallax), and
 *     Model R is used with the consensus set misfit_i <= ransacThresholdPx. Otherwise the parallax is significant and
 *     Model E (Sampson epipolar) is used, after RANSAC with cv::findEssentialMat (5-point, normalised units).
 *  3. Gate at the predicted state (outside the likelihood): Model E keeps |r_i^w| <= sqrt(chi2inv-based threshold,
 *     dof 1); the Model R consensus above is already a gate.
 *
 * A frame with fewer than minCorrespondences surviving pairs gets no flow update (valid = false).
 *
 * Why the model switch. The epipolar residual is scale-free in the translation and its Gauss-Newton curvature in the
 * translation direction is driven by noise when the parallax is below the pixel noise (see MeasurementIndoorFlow.h).
 * Selecting Model R there keeps the rotation information of the flow without that spurious translation curvature.
 * The switch is decided from the predicted state and the data, not from the posterior, so the likelihood of each frame is
 * a fixed model (a deterministic model choice, not a mixture).
 */

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "Camera.h"
#include "MeasurementIndoorFlow.h"
#include "SystemEstimator.h"

namespace slam
{

/** @brief One pixel correspondence between frame k-1 (prev) and frame k (curr). */
struct IndoorFlowPixelMatch
{
    Eigen::Vector2d prevPx = Eigen::Vector2d::Zero();   ///< pixel in frame k-1 (OpenCV convention)
    Eigen::Vector2d currPx = Eigen::Vector2d::Zero();   ///< pixel in frame k
};

/** @brief Options of the flow front end. Values marked ASSUMED are engineering choices (see the reports). */
struct IndoorFlowOptions
{
    double sigmaPx = 0.5;                  ///< ASSUMED (NIS-calibrated in-sample on tag_identical.MOV: median NIS/dof 0.43 at 0.5 px, 0.12 at 1 px): pixel noise of one correspondence coordinate [px]
    double ransacThresholdPx = 2.0;        ///< ASSUMED: RANSAC / consensus threshold [px]
    double rotationMisfitPx = 0.5;         ///< ASSUMED: median rotation-only misfit below this selects Model R [px] (near-zero parallax only)
    double gateNSigma = 3.0;               ///< chi2 gate on the Model E residuals (dof 1), GaussianBase convention
    int minCorrespondences = 12;           ///< ASSUMED: fewer surviving correspondences -> no flow update
};

/** @brief Result of the flow front end for one frame. */
struct IndoorFlowFrame
{
    bool valid = false;                    ///< enough correspondences for a flow update
    IndoorFlowModel model = IndoorFlowModel::Rotation;
    std::vector<IndoorFlowPair> pairs;     ///< normalised correspondences used by the measurement
    double sigmaNorm = 0.0;                ///< residual standard deviation in normalised units (sigmaPx / f)
    int input = 0;                         ///< pixel matches given
    int normalised = 0;                    ///< pairs with finite undistorted coordinates
    int consensus = 0;                     ///< after RANSAC (Model E) or the rotation consensus (Model R)
    int gated = 0;                         ///< pairs used (after the prior gate)
    double medianRotationMisfitPx = 0.0;   ///< median misfit under the prior rotation [px] (NaN if no pairs)
};

/**
 * @brief Builds the flow measurement inputs of one frame from pixel matches and the predicted estimator state.
 * @param camera   calibrated camera (undistortion, Rbc, rCBb)
 * @param est      estimator after the delayed time update (eta = eta_k, etaPrev = eta_{k-1})
 * @param matches  pixel correspondences between frame k-1 and frame k
 * @param opt      options
 * @throws std::invalid_argument if the camera is invalid or the state does not contain etaPrev
 */
IndoorFlowFrame buildIndoorFlowFrame(const Camera& camera, const SystemEstimator& est,
                                     const std::vector<IndoorFlowPixelMatch>& matches, const IndoorFlowOptions& opt);

} // namespace slam

#endif
