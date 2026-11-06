#ifndef SLAMASSOCIATION_H
#define SLAMASSOCIATION_H

#include <cstddef>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include "Camera.h"
#include "MeasurementPointBundle.h"
#include "PointFeatures.h"
#include "SlamLandmarkMap.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

namespace slam
{

/**
 * @brief Predicted pixel of one landmark in the current (predicted) state.
 *
 * pixelCov is the state-uncertainty part J Sigma J^T of the pixel prediction (2 x 2), with Sigma the marginal
 * covariance of (eta, landmark). It is obtained from triangular solves with the square-root information factor
 * (Sigma = Xi^{-1} Xi^{-T}); the information matrix is never inverted. The measurement noise is added on use.
 */
struct LandmarkPrediction
{
    Eigen::Index id = -1;                 ///< landmark id
    Eigen::Index offset = 0;              ///< landmark block offset in the state
    bool visible = false;                 ///< in front of the camera (depth in range) and inside the image with margin
    double depth = 0.0;                   ///< predicted camera-frame depth p_c.z [m]
    Eigen::Vector2d pixel = Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix2d pixelCov = Eigen::Matrix2d::Zero();   ///< J Sigma J^T (state part only; diagnostics/drawing)
    /// Upper-triangular square root R of pixelCov (R^T R = J Sigma J^T), from a QR of the whitened projection.
    /// All-zero means "not set" (e.g. hand-built test predictions); gateDistanceSquared() then factorises pixelCov.
    Eigen::Matrix2d pixelSqrtCov = Eigen::Matrix2d::Zero();
    double sigmaPx = 0.0;                 ///< measurement pixel sigma at the predicted pixel
};

/**
 * @brief Squared Mahalanobis distance e^T S^{-1} e of a pixel innovation, S = pixelCov + sigmaPx^2 I.
 *
 * Computed in square-root form without inverting S: R_S = qr([R_p; sigmaPx I]) (so R_S^T R_S = S) and
 * d^2 = ||R_S^{-T} e||^2 by one triangular solve. R_p is pred.pixelSqrtCov, or the Cholesky factor of pred.pixelCov
 * when the square root was not provided.
 */
double gateDistanceSquared(const LandmarkPrediction& pred, const Eigen::Vector2d& e);

/** @brief Predictions for all landmarks (model order) plus the whitened projection Xi^{-T} J^T (n x 2N). */
struct PredictionSet
{
    std::vector<LandmarkPrediction> predictions;
    Eigen::MatrixXd whitenedProjection;   ///< columns 2k, 2k+1 belong to prediction k
};

/**
 * @brief Predicts the pixel of every landmark registered in the model at the estimator's current state.
 * @throws std::invalid_argument if the estimator state does not match the model layout
 */
PredictionSet predictLandmarks(const SystemEstimator& est, const SystemVisualNav& model, const Camera& camera,
                               const SlamConfig& cfg);

/** @brief Result of the feature-to-landmark association of one frame. */
struct AssociationResult
{
    std::vector<MeasurementPointBundle::Observation> observations;  ///< accepted pairs (measurement order)
    std::vector<Eigen::Index> matchedIds;                          ///< landmark id per observation
    std::vector<std::size_t> matchedPrediction;                    ///< index into PredictionSet::predictions
    std::vector<int> matchedFeature;                               ///< index into the feature list
    std::vector<bool> featureUsed;                                 ///< feature matched to a landmark
    /// per prediction: at least one feature passed the gate and the descriptor test (ambiguous and conflicting
    /// candidates are gated; a visible prediction that is not gated is a miss, see SlamPointPipeline)
    std::vector<bool> predictionGated;

    int numVisible = 0;        ///< predicted visible landmarks
    int numNoCandidate = 0;    ///< visible landmarks with no gated, descriptor-compatible feature
    int numAmbiguous = 0;      ///< visible landmarks with two gated, similar candidates
    int numConflict = 0;       ///< landmarks whose chosen feature was claimed by another landmark too

    /// Innovation y - h(mu) of the matched set (2M) and its covariance S = Zm^T Zm + R (2M x 2M).
    Eigen::VectorXd innovation;
    Eigen::MatrixXd innovationCov;
};

/**
 * @brief Gated, descriptor-checked, ambiguity-rejecting association of features to predicted landmarks.
 *
 * A feature f is a candidate of visible landmark j if
 *   (1) the Mahalanobis distance of its pixel from the prediction, with S = pixelCov + sigma^2 I, is within the
 *       chi2 gate mahalanobisThresholdSq(gateNSigma, 2), and
 *   (2) the ORB Hamming distance to the landmark descriptor is at most maxHamming.
 * The landmark takes its best candidate (fewest bits) unless a second candidate is within ambiguityMargin bits
 * (then it is rejected as ambiguous). A feature claimed by two landmarks is rejected for both.
 */
AssociationResult associateFeatures(const PredictionSet& predictions, const std::vector<PointFeature>& features,
                                    const SlamLandmarkMap& map, const Camera& camera, const SlamConfig& cfg);

} // namespace slam

#endif
