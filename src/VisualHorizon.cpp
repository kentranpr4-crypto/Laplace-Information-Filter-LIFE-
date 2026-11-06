#include "VisualHorizon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>

namespace
{

/// Mean and square-root covariance of a pooled set of feature samples (rows of X), with a relative ridge on the covariance.
SkyGroundClassifier::ClassModel fitClass(const std::vector<Eigen::Vector4d>& samples, double ridge)
{
    if (samples.size() < 10)
    {
        throw std::invalid_argument("SkyGroundClassifier: too few training pixels for a class");
    }
    SkyGroundClassifier::ClassModel model;
    Eigen::Vector4d mean = Eigen::Vector4d::Zero();
    for (const Eigen::Vector4d& f : samples)
    {
        mean += f;
    }
    mean /= static_cast<double>(samples.size());
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
    for (const Eigen::Vector4d& f : samples)
    {
        const Eigen::Vector4d d = f - mean;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(samples.size() - 1);
    // Regularise: relative ridge plus an absolute floor, so that flat (constant-colour) training patches still give a
    // finite precision matrix. The floor is in feature units (L, a, b in CIELAB, texture in L units).
    const double meanVar = cov.trace() / 4.0;
    constexpr double kVarianceFloor = 1e-2;
    cov += (ridge * meanVar + kVarianceFloor) * Eigen::Matrix4d::Identity();
    const Eigen::LLT<Eigen::Matrix4d> llt(cov);   // factorisation only; the covariance is never inverted
    if (llt.info() != Eigen::Success)
    {
        throw std::invalid_argument("SkyGroundClassifier: class covariance Cholesky factorisation failed");
    }
    model.mean = mean;
    model.sqrtCov = llt.matrixU();
    return model;
}

/// log|Sigma| of a class model from its square-root factor: log|Sigma| = 2 sum log diag U.
double logDetCovariance(const SkyGroundClassifier::ClassModel& m)
{
    return 2.0 * m.sqrtCov.diagonal().array().log().sum();
}

/// Squared Mahalanobis distance ||U^{-T}(f - mu)||^2 by one triangular solve.
double mahalanobisSq(const SkyGroundClassifier::ClassModel& m, const Eigen::Vector4d& f)
{
    const Eigen::Vector4d w = m.sqrtCov.transpose().triangularView<Eigen::Lower>().solve(f - m.mean);
    return w.squaredNorm();
}

/// Total-least-squares line a u + b v + c = 0 (a^2 + b^2 = 1) through points (smallest-eigenvalue normal).
Eigen::Vector3d fitLineTls(const std::vector<cv::Point2d>& pts)
{
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (const cv::Point2d& p : pts)
    {
        centroid += Eigen::Vector2d(p.x, p.y);
    }
    centroid /= static_cast<double>(pts.size());
    Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
    for (const cv::Point2d& p : pts)
    {
        const Eigen::Vector2d d(p.x - centroid(0), p.y - centroid(1));
        scatter += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(scatter);
    const Eigen::Vector2d normal = es.eigenvectors().col(0);
    return Eigen::Vector3d(normal(0), normal(1), -normal.dot(centroid));
}

/**
 * @brief Sub-pixel sky->ground edge in one column of a full-resolution gray image.
 *
 * F(v) = S(v-1) - S(v+1), S = [1 2 1]/4-smoothed intensity: F is large and positive where intensity drops downwards
 * (sky above ground). The maximum of F within [vCentre - radius, vCentre + radius] is refined by a parabola through
 * three samples. Returns false if the drop is weaker than minContrast or the window leaves the image.
 */
bool refineEdgeInColumn(const cv::Mat& gray, int u, double vCentre, int radius, double minContrast, double& vOut)
{
    const int H = gray.rows;
    const int vLo = static_cast<int>(std::lround(vCentre)) - radius;
    const int vHi = static_cast<int>(std::lround(vCentre)) + radius;
    if (vLo < 3 || vHi > H - 4)
    {
        return false;
    }
    auto intensity = [&](int v) { return static_cast<double>(gray.at<std::uint8_t>(v, u)); };
    auto S = [&](int v) { return 0.25 * (intensity(v - 1) + 2.0 * intensity(v) + intensity(v + 1)); };
    auto F = [&](int v) { return S(v - 1) - S(v + 1); };
    int best = vLo;
    double bestF = F(vLo);
    for (int v = vLo + 1; v <= vHi; ++v)
    {
        if (F(v) > bestF)
        {
            bestF = F(v);
            best = v;
        }
    }
    if (bestF < minContrast)
    {
        return false;
    }
    const double fm = F(best - 1);
    const double f0 = F(best);
    const double fp = F(best + 1);
    const double denom = fm - 2.0 * f0 + fp;
    double offset = 0.0;
    if (denom < 0.0)   // a true maximum of the parabola
    {
        offset = std::clamp(0.5 * (fm - fp) / denom, -0.5, 0.5);
    }
    vOut = static_cast<double>(best) + offset;
    return true;
}

void collectSamples(const cv::Mat& bgr, int textureWindow, int stride, std::vector<Eigen::Vector4d>& out)
{
    const cv::Mat feat = SkyGroundClassifier::features(bgr, textureWindow);
    for (int y = 0; y < feat.rows; y += stride)
    {
        const cv::Vec4f* row = feat.ptr<cv::Vec4f>(y);
        for (int x = 0; x < feat.cols; x += stride)
        {
            const cv::Vec4f& f = row[x];
            if (std::isfinite(f[0]) && std::isfinite(f[1]) && std::isfinite(f[2]) && std::isfinite(f[3]))
            {
                out.emplace_back(f[0], f[1], f[2], f[3]);
            }
        }
    }
}

} // namespace

cv::Mat SkyGroundClassifier::features(const cv::Mat& bgr, int textureWindow)
{
    if (bgr.empty() || bgr.type() != CV_8UC3)
    {
        throw std::invalid_argument("SkyGroundClassifier::features: BGR8 image required");
    }
    cv::Mat f32;
    bgr.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    cv::Mat lab;
    cv::cvtColor(f32, lab, cv::COLOR_BGR2Lab);  // L in [0,100], a and b roughly in [-128,127]

    cv::Mat chans[3];
    cv::split(lab, chans);
    // Local texture: standard deviation of L over a box window (sky is smooth, vegetation/water is not).
    cv::Mat mean1;
    cv::Mat mean2;
    cv::Mat l2 = chans[0].mul(chans[0]);
    cv::boxFilter(chans[0], mean1, CV_32F, cv::Size(textureWindow, textureWindow), cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    cv::boxFilter(l2, mean2, CV_32F, cv::Size(textureWindow, textureWindow), cv::Point(-1, -1), true,
                  cv::BORDER_REFLECT);
    cv::Mat var = mean2 - mean1.mul(mean1);
    cv::max(var, 0.0, var);
    cv::Mat tex;
    cv::sqrt(var, tex);

    cv::Mat out;
    cv::Mat parts[4] = {chans[0], chans[1], chans[2], tex};
    cv::merge(parts, 4, out);
    return out;
}

SkyGroundClassifier SkyGroundClassifier::train(const std::vector<cv::Mat>& skyPatches,
                                               const std::vector<cv::Mat>& groundPatches,
                                               const SkyGroundClassifierOptions& options)
{
    if (skyPatches.empty() || groundPatches.empty())
    {
        throw std::invalid_argument("SkyGroundClassifier::train: both classes need at least one patch");
    }
    if (options.textureWindow < 1 || options.textureWindow % 2 == 0 || options.sampleStride < 1 || !(options.ridge >= 0.0))
    {
        throw std::invalid_argument("SkyGroundClassifier::train: invalid options");
    }
    std::vector<Eigen::Vector4d> skySamples;
    std::vector<Eigen::Vector4d> groundSamples;
    for (const cv::Mat& p : skyPatches)
    {
        collectSamples(p, options.textureWindow, options.sampleStride, skySamples);
    }
    for (const cv::Mat& p : groundPatches)
    {
        collectSamples(p, options.textureWindow, options.sampleStride, groundSamples);
    }

    SkyGroundClassifier c;
    c.sky_ = fitClass(skySamples, options.ridge);
    c.ground_ = fitClass(groundSamples, options.ridge);
    c.textureWindow_ = options.textureWindow;

    // LLR(f) = -1/2 d_s^2 + 1/2 d_g^2 + c, c = 1/2 (log|Sigma_g| - log|Sigma_s|), equal class priors.
    c.constant_ = 0.5 * (logDetCovariance(c.ground_) - logDetCovariance(c.sky_));
    c.trained_ = true;
    return c;
}

cv::Mat SkyGroundClassifier::logOdds(const cv::Mat& bgr) const
{
    if (!trained_)
    {
        throw std::logic_error("SkyGroundClassifier::logOdds: classifier is not trained");
    }
    const cv::Mat feat = features(bgr, textureWindow_);
    cv::Mat out(feat.size(), CV_32FC1);
    for (int y = 0; y < feat.rows; ++y)
    {
        const cv::Vec4f* in = feat.ptr<cv::Vec4f>(y);
        float* o = out.ptr<float>(y);
        for (int x = 0; x < feat.cols; ++x)
        {
            const Eigen::Vector4d f(in[x][0], in[x][1], in[x][2], in[x][3]);
            o[x] = static_cast<float>(-0.5 * mahalanobisSq(sky_, f) + 0.5 * mahalanobisSq(ground_, f) + constant_);
        }
    }
    return out;
}

HorizonEstimate extractHorizon(const cv::Mat& logOddsMap, double analysisScale, int fullWidth,
                               const HorizonExtractionOptions& options, const cv::Mat& grayFull)
{
    if (logOddsMap.empty() || logOddsMap.type() != CV_32FC1)
    {
        throw std::invalid_argument("extractHorizon: log-odds map must be a non-empty CV_32FC1 image");
    }
    if (!(analysisScale > 0.0) || fullWidth <= 0 || options.columnStep < 1 || options.minSkyRows < 1 ||
        options.minInliers < 2 || !(options.ransacThresholdPx > 0.0))
    {
        throw std::invalid_argument("extractHorizon: invalid options");
    }
    HorizonEstimate est;

    // Smooth the CLIPPED log-odds, not the raw ones. The raw log-odds are strongly asymmetric across a sky/ground edge
    // (ground-side magnitudes are ~10x the sky-side ones on the synthetic and real samples), so the linear 3x3 mean
    // moves the zero crossing into the sky by about one analysis row, i.e. the extracted horizon lay ~5 px above the
    // true one (measured in test/src/VisualHorizonTests.cpp). Clipping at +/-kLogOddsClip gives both sides equal weight.
    constexpr double kLogOddsClip = 1.0;
    cv::Mat clipped;
    cv::min(logOddsMap, kLogOddsClip, clipped);
    cv::max(clipped, -kLogOddsClip, clipped);
    cv::Mat sm;
    cv::boxFilter(clipped, sm, CV_32F, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    const int rows = sm.rows;
    const int cols = sm.cols;

    long skyCount = 0;
    for (int y = 0; y < rows; ++y)
    {
        const float* p = sm.ptr<float>(y);
        for (int x = 0; x < cols; ++x)
        {
            skyCount += (p[x] > 0.0f) ? 1 : 0;
        }
    }
    est.skyFraction = static_cast<double>(skyCount) / (static_cast<double>(rows) * static_cast<double>(cols));
    if (est.skyFraction < options.minSkyFraction)
    {
        return est;
    }

    // 1. Candidate boundary points: first top-down sky -> ground transition in each analysis column.
    for (int c = 0; c < cols; c += options.columnStep)
    {
        int transition = -1;
        for (int r = 1; r < rows; ++r)
        {
            if (sm.at<float>(r - 1, c) > 0.0f && sm.at<float>(r, c) <= 0.0f)
            {
                transition = r;
                break;
            }
        }
        if (transition < options.minSkyRows)
        {
            continue;
        }
        double skyAbove = 0.0;
        for (int r = transition - options.minSkyRows; r < transition; ++r)
        {
            skyAbove += sm.at<float>(r, c);
        }
        if (skyAbove <= 0.0)
        {
            continue;
        }
        const double a = sm.at<float>(transition - 1, c);
        const double b = sm.at<float>(transition, c);
        const double t = a / (a - b);   // zero crossing between rows transition-1 (a > 0) and transition (b <= 0)
        const double vAnalysis = (transition - 1) + t;
        const double u = (c + 0.5) / analysisScale - 0.5;
        const double v = (vAnalysis + 0.5) / analysisScale - 0.5;
        est.candidates.emplace_back(u, v);
    }
    if (static_cast<int>(est.candidates.size()) < options.minInliers)
    {
        return est;
    }

    // 2. RANSAC over two-point lines (deterministic seed).
    std::mt19937 rng(options.seed);
    std::uniform_int_distribution<std::size_t> pick(0, est.candidates.size() - 1);
    const double thr = options.ransacThresholdPx;
    int bestCount = -1;
    Eigen::Vector3d bestLine = Eigen::Vector3d::Zero();
    for (int it = 0; it < options.ransacIterations; ++it)
    {
        const std::size_t i = pick(rng);
        const std::size_t j = pick(rng);
        if (i == j)
        {
            continue;
        }
        const Eigen::Vector2d pi(est.candidates[i].x, est.candidates[i].y);
        const Eigen::Vector2d pj(est.candidates[j].x, est.candidates[j].y);
        Eigen::Vector2d d = pj - pi;
        const double len = d.norm();
        if (!(len > 1e-6))
        {
            continue;
        }
        const Eigen::Vector2d nrm(-d(1) / len, d(0) / len);
        int count = 0;
        for (const cv::Point2d& p : est.candidates)
        {
            const double dist = std::abs(nrm(0) * (p.x - pi(0)) + nrm(1) * (p.y - pi(1)));
            count += (dist <= thr) ? 1 : 0;
        }
        if (count > bestCount)
        {
            bestCount = count;
            bestLine << nrm(0), nrm(1), -(nrm(0) * pi(0) + nrm(1) * pi(1));
        }
    }
    if (bestCount < options.minInliers)
    {
        return est;
    }

    // 3. Total-least-squares refit on the inliers (twice: the inlier set is re-selected with the refit line).
    Eigen::Vector3d line = bestLine;
    std::vector<cv::Point2d> inl;
    for (int pass = 0; pass < 2; ++pass)
    {
        inl.clear();
        for (const cv::Point2d& p : est.candidates)
        {
            const double dist = std::abs(line(0) * p.x + line(1) * p.y + line(2));
            if (dist <= thr)
            {
                inl.push_back(p);
            }
        }
        if (static_cast<int>(inl.size()) < 2)
        {
            return est;
        }
        Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
        for (const cv::Point2d& p : inl)
        {
            centroid += Eigen::Vector2d(p.x, p.y);
        }
        centroid /= static_cast<double>(inl.size());
        Eigen::Matrix2d scatter = Eigen::Matrix2d::Zero();
        for (const cv::Point2d& p : inl)
        {
            const Eigen::Vector2d d(p.x - centroid(0), p.y - centroid(1));
            scatter += d * d.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(scatter);
        const Eigen::Vector2d normal = es.eigenvectors().col(0);   // smallest eigenvalue -> line normal
        line << normal(0), normal(1), -normal.dot(centroid);
    }

    // 4. Sub-pixel refinement of the inliers to the full-resolution sky->ground edge, then a TLS refit.
    if (options.refineToEdge && !grayFull.empty())
    {
        if (grayFull.type() != CV_8UC1 || grayFull.cols != fullWidth)
        {
            throw std::invalid_argument("extractHorizon: grayFull must be CV_8UC1 with the full image width");
        }
        for (cv::Point2d& p : inl)
        {
            const int u = std::clamp(static_cast<int>(std::lround(p.x)), 0, grayFull.cols - 1);
            double vEdge = 0.0;
            if (refineEdgeInColumn(grayFull, u, p.y, options.refineRadiusPx, options.refineMinContrast, vEdge))
            {
                p.y = vEdge;
            }
        }
        if (inl.size() >= 2)
        {
            line = fitLineTls(inl);
        }
    }
    est.line = line;
    est.inliers = inl;

    double uMin = inl.front().x;
    double uMax = inl.front().x;
    for (const cv::Point2d& p : inl)
    {
        uMin = std::min(uMin, p.x);
        uMax = std::max(uMax, p.x);
    }
    est.spanFraction = (uMax - uMin) / static_cast<double>(fullWidth);
    est.valid = static_cast<int>(inl.size()) >= options.minInliers && est.spanFraction >= options.minSpanFraction;
    return est;
}
