#ifndef VISUALHORIZON_H
#define VISUALHORIZON_H

#include <cstdint>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

/**
 * @brief Options of the two-class sky/ground pixel classifier.
 */
struct SkyGroundClassifierOptions
{
    int textureWindow = 5;       ///< side of the box window of the local texture (standard deviation of L) [px]
    int sampleStride = 3;        ///< training pixels are taken every sampleStride pixels in x and y
    double ridge = 1e-3;         ///< covariance ridge, relative to the mean variance of the class
};

/**
 * @brief Gaussian naive-free two-class classifier on per-pixel features f = (L, a, b, texture) in CIELAB.
 *
 * The class-conditional densities are full-covariance Gaussians fitted to labelled patches (sky, ground) with equal
 * priors. The log-odds log p(f|sky) - log p(f|ground) is a quadratic form in f,
 *   LLR(f) = f^T A f + b^T f + c,   A = -(1/2)(P_s - P_g),  b = P_s mu_s - P_g mu_g,
 * evaluated in square-root form: with Sigma = U^T U (upper Cholesky factor of the training covariance),
 *   LLR(f) = -1/2 ||U_s^{-T}(f - mu_s)||^2 + 1/2 ||U_g^{-T}(f - mu_g)||^2 - sum log diag U_s + sum log diag U_g,
 * using triangular solves only (no covariance is inverted anywhere; USER rule). Positive log-odds mean sky.
 */
class SkyGroundClassifier
{
public:
    /** @brief Creates an untrained classifier. */
    SkyGroundClassifier() = default;

    /**
     * @brief Fits the two Gaussian models from labelled BGR patches.
     * @param skyPatches    BGR images containing only sky
     * @param groundPatches BGR images containing only ground
     * @throws std::invalid_argument if a class has no usable pixels or the covariance is singular
     */
    static SkyGroundClassifier train(const std::vector<cv::Mat>& skyPatches, const std::vector<cv::Mat>& groundPatches,
                                     const SkyGroundClassifierOptions& options = {});

    /** @brief True after a successful train(). */
    bool trained() const { return trained_; }

    /**
     * @brief Per-pixel log-odds (CV_32FC1, same size as bgr; positive = sky).
     * @throws std::logic_error if untrained, std::invalid_argument for non-BGR8 input
     */
    cv::Mat logOdds(const cv::Mat& bgr) const;

    /** @brief Feature image (CV_32FC4: L, a, b, texture) of a BGR8 image. */
    static cv::Mat features(const cv::Mat& bgr, int textureWindow);

    /** @brief Class statistics: mean and upper-triangular square-root covariance (Sigma = U^T U). */
    struct ClassModel
    {
        Eigen::Vector4d mean = Eigen::Vector4d::Zero();
        Eigen::Matrix4d sqrtCov = Eigen::Matrix4d::Identity();   ///< U (upper triangular, positive diagonal)
    };
    const ClassModel& skyModel() const { return sky_; }
    const ClassModel& groundModel() const { return ground_; }

private:
    ClassModel sky_;
    ClassModel ground_;
    double constant_ = 0.0;   ///< sum log diag U_g - sum log diag U_s (equal class priors)
    int textureWindow_ = 5;
    bool trained_ = false;
};

/**
 * @brief Options of the horizon boundary extraction and robust line fit.
 */
struct HorizonExtractionOptions
{
    int columnStep = 4;               ///< analysis-resolution column spacing between candidate boundary points
    int minSkyRows = 3;               ///< analysis rows of sky required above the transition
    double ransacThresholdPx = 3.0;   ///< inlier distance to the fitted line [full-resolution px]
    int ransacIterations = 400;       ///< two-point hypotheses
    int minInliers = 15;              ///< minimum inlier count for a valid horizon
    double minSpanFraction = 0.3;     ///< minimum horizontal extent of the inliers (fraction of the image width)
    double minSkyFraction = 0.02;     ///< minimum fraction of the image classified as sky
    unsigned seed = 12345;            ///< deterministic RANSAC seed
    /// Refine every inlier to the sub-pixel sky->ground edge of the full-resolution gray image (needs grayFull).
    /// The classifier boundary is quantised to the analysis grid (4 full-resolution px) and sits ~0.5 px high on the
    /// synthetic horizon; the pitch estimate inherits that bias.
    bool refineToEdge = true;
    int refineRadiusPx = 6;           ///< search half-window around each boundary point [full-resolution px]
    double refineMinContrast = 5.0;   ///< minimum sky->ground drop over 2 px [gray levels] to accept an edge
};

/** @brief Horizon estimate of one frame (all coordinates in full-resolution pixels, OpenCV convention). */
struct HorizonEstimate
{
    bool valid = false;                      ///< true if enough consistent boundary points were found
    double skyFraction = 0.0;                ///< fraction of analysis pixels classified as sky
    std::vector<cv::Point2d> candidates;     ///< all sky->ground transitions found (before RANSAC)
    std::vector<cv::Point2d> inliers;        ///< RANSAC inliers used for the measurement
    Eigen::Vector3d line = Eigen::Vector3d::Zero();  ///< a u + b v + c = 0 with a^2 + b^2 = 1 (full-res px)
    double spanFraction = 0.0;               ///< horizontal extent of the inliers / image width
};

/**
 * @brief Extracts the sky/ground horizon from a per-pixel log-odds map.
 *
 * For every analysis column, the first top-down transition from positive (sky) to negative (ground) log-odds is
 * located with sub-pixel interpolation; RANSAC over two-point lines then selects the consistent boundary points, and
 * a total-least-squares refit gives the final line. The estimate is invalid when too few inliers, too little
 * horizontal span or too little sky are present (e.g. camera looking down, or a heavily occluded horizon).
 *
 * If grayFull is given (CV_8UC1, full resolution) and options.refineToEdge is set, every inlier is moved to the
 * sub-pixel position of the strongest sky->ground intensity drop in its column (within refineRadiusPx), and the line
 * is refitted on the refined points.
 *
 * @param logOddsMap   CV_32FC1 log-odds at analysis resolution (positive = sky); smoothed after clipping at +/-1
 * @param analysisScale analysis size / full size (same in x and y)
 * @param fullWidth    full image width [px]
 * @param options      extraction options
 * @param grayFull     optional full-resolution gray image for the sub-pixel refinement
 */
HorizonEstimate extractHorizon(const cv::Mat& logOddsMap, double analysisScale, int fullWidth,
                               const HorizonExtractionOptions& options = {}, const cv::Mat& grayFull = cv::Mat());

#endif
