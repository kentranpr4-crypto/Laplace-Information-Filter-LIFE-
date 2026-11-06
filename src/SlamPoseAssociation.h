#ifndef SLAMPOSEASSOCIATION_H
#define SLAMPOSEASSOCIATION_H

/**
 * @file SlamPoseAssociation.h
 * @brief Geometric data association for identical-ID tags (Scenario 5).
 *
 * The tags carry the same ArUco id, so appearance cannot distinguish them. Association uses only geometry:
 *  1. Every landmark that is predicted to be in front of the camera and near the image is a candidate. Its corner
 *     prediction z_j and its 8 x 8 innovation covariance S_j = H_j P H_j^T + sigma^2 I are computed from ONE marginal of
 *     the estimator over {eta} U {L_j} (GaussianInfo::marginal). With Xi the upper square-root information of the
 *     marginal (P = Xi^{-1} Xi^{-T}), X_j = Xi^{-T} H_j^T is obtained by a lower-triangular solve and
 *     S_j = X_j^T X_j + sigma^2 I. No covariance is inverted or factorised: the gate uses the upper factor
 *     R_j of S_j (Q-less QR of [X_j; sigma I]) and a triangular solve.
 *  2. Gate: d_ij^2 = (z_i - z_j)^T S_j^{-1} (z_i - z_j) <= chi2inv(erf(3/sqrt2), 8)  (GaussianBase::mahalanobisThresholdSq).
 *  3. Ambiguity: a detection with two or more gated candidates whose costs differ by less than ambiguityMarginD2 is
 *     rejected (not associated, not a birth). A candidate claimed by several detections keeps the best one; the others
 *     are conflicts (not births).
 *  4. Birth candidates: detections that no candidate gates and that are not close to any predicted tag centre.
 *
 * Corner correspondence is index-to-index: the detector returns the canonical corner order for every tag (see
 * ArucoTags.h), and the landmark frame keeps that order (Q c_k).
 */

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "ArucoTags.h"
#include "Camera.h"
#include "MeasurementTagBundle.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

namespace slam
{

/** @brief A tag landmark predicted for the current frame. */
struct TagCandidate
{
    std::size_t record = 0;                    ///< index of the tracker record
    Eigen::Index id = -1;                      ///< landmark id
    Eigen::Index offset = 0;                   ///< state offset of [r; Theta]
    Eigen::Index unionOffset = 0;              ///< column offset of this landmark in the union marginal
    Eigen::Matrix<double, 4, 3> cornersLocal = Eigen::Matrix<double, 4, 3>::Zero();
    TagCornerPrediction pred;                  ///< predicted corners and Jacobians (valid)
    Eigen::Matrix<double, 8, 8> S = Eigen::Matrix<double, 8, 8>::Zero();   ///< innovation covariance (with pixel noise); for display and tests only
    Eigen::Matrix<double, 8, 8> sqrtS = Eigen::Matrix<double, 8, 8>::Zero(); ///< upper R with R^T R = S, from a Q-less QR of [X; sigma I]; used by the gate
    bool visible = false;                      ///< strict visibility (counted as a miss when not detected): in the image,
                                               ///< large enough, facing the camera and not occluded
    bool occluded = false;                     ///< centre inside the predicted quad of a nearer candidate (see file doc)
    double depth = 0.0;                        ///< camera depth of the tag centre [m]
    double sidePx = 0.0;                       ///< predicted mean edge length [px]
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();  ///< predicted centre [px]
};

/** @brief Predicted candidates and the union marginal used to compute their innovation covariances. */
struct TagPredictionSet
{
    std::vector<TagCandidate> candidates;
    std::vector<Eigen::Index> unionIndices;    ///< state indices of the union marginal: [eta; L_1; ...; L_C]
    Eigen::MatrixXd xiUnion;                   ///< upper square-root information of the union marginal
};

/** @brief Input of the prediction: a record and its landmark. */
struct TagPredictionInput
{
    std::size_t record = 0;
    Eigen::Index id = -1;
    Eigen::Matrix<double, 4, 3> cornersLocal = Eigen::Matrix<double, 4, 3>::Zero();
};

/** @brief Options of the prediction and visibility rules. */
struct TagPredictionOptions
{
    double sigmaPx = 2.0;             ///< pixel std per corner component (R = sigma^2 I8)
    double minDepth = 0.2;            ///< [m] corners closer than this are not predicted
    double maxDepth = 30.0;           ///< [m] candidates farther than this are not predicted
    double candidateMarginPx = 100.0; ///< candidates whose corners lie within the image grown by this margin
    double visibleMarginPx = 5.0;     ///< strict visibility: corners inside the image shrunk by this margin
    double minVisibleSidePx = 20.0;   ///< strict visibility: predicted mean edge at least this long
    double minFacingCos = 0.3;        ///< strict visibility: cos of the angle between the tag normal and the view direction
    double occlusionDepthMarginM = 0.05;  ///< ASSUMED: a candidate occludes another one only if nearer by this margin [m]
};

/**
 * @brief True if the point lies inside the convex quadrilateral given by the 8-vector of corners (TL, TR, BR, BL order,
 * as produced by the canonical tag corners). Boundary points count as inside.
 */
bool pointInTagQuad(const Eigen::Vector2d& p, const Eigen::Matrix<double, 8, 1>& uv);

/**
 * @brief Predicts the candidates and their innovation covariances (one union marginal, triangular solves only).
 * @param est       estimator (state after the time update)
 * @param model     model with the landmark layout
 * @param camera    camera
 * @param inputs    records with landmark ids and landmark-frame corners
 * @param opts      options
 */
TagPredictionSet predictTagCandidates(const SystemEstimator& est, const SystemVisualNav& model, const Camera& camera,
                                      const std::vector<TagPredictionInput>& inputs, const TagPredictionOptions& opts);

/** @brief Result of the geometric association of one frame. */
struct TagAssociationResult
{
    std::vector<int> detectionCandidate;   ///< per detection: matched candidate index or -1
    std::vector<int> candidateDetection;   ///< per candidate: matched detection index or -1
    std::vector<bool> candidateGated;      ///< per candidate: at least one detection fell inside its gate
    std::vector<double> detectionD2;       ///< per detection: smallest gated d^2 (infinity if none)
    std::vector<bool> ambiguous;           ///< per detection: >= 2 gated candidates within the ambiguity margin
    std::vector<bool> conflict;            ///< per detection: lost a candidate to a better detection
    std::vector<bool> birthCandidate;      ///< per detection: not gated, not near a predicted tag
    int numGatedPairs = 0;                 ///< gated (detection, candidate) pairs
};

/** @brief Association options. */
struct TagAssociationOptions
{
    double gateThresholdSq = 0.0;          ///< chi2 threshold (dof 8, 3 sigma); must be set by the caller
    double ambiguityMarginD2 = 4.0;        ///< ASSUMED: second candidate within this d^2 of the best makes a detection ambiguous
    double nearCentreFactor = 1.0;         ///< detection closer than factor * max(side) to a predicted centre is not a birth
};

/** @brief Geometric association of the detections to the predicted candidates (see file documentation). */
TagAssociationResult associateTagDetections(const TagPredictionSet& predictions,
                                            const std::vector<tags::TagDetection>& detections,
                                            const TagAssociationOptions& opts);

/**
 * @brief Normalised innovation squared of a set of matched candidates, jointly (cross-correlations kept).
 * @param predictions  prediction set (union marginal)
 * @param matched      candidate indices (m)
 * @param innovation   z - z_hat stacked over matched candidates (8m)
 * @param sigmaPx      pixel std
 * @return e^T S^{-1} e with S = H P H^T + sigma^2 I over the stacked matched candidates (dof 8m)
 */
double jointNis(const TagPredictionSet& predictions, const std::vector<std::size_t>& matched,
                const Eigen::VectorXd& innovation, double sigmaPx);

} // namespace slam

#endif
