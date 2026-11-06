// Scenario 5 SLAM: lifecycle rules, and a synthetic multi-tag sequence (rendered identical-ID markers, known truth).
#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>
#include <doctest/doctest.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include "../../src/ArucoTags.h"
#include "../../src/Camera.h"
#include "../../src/Pose.hpp"
#include "../../src/SlamPoseTracker.h"
#include "../../src/rotation.hpp"
#include "testUtil.h"

using Eigen::Index;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using slam::SlamPoseConfig;
using slam::SlamPoseRecord;
using slam::SlamPoseTracker;

namespace
{

constexpr double kTagSide = 0.2;

/// Pinhole camera without distortion (exact synthetic rendering).
Camera syntheticCamera()
{
    return Camera(640, 480, (Matrix3d() << 500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0).finished(), {});
}

/// Renders the markers (id 5, world poses) seen from the body pose (rBNn, Theta_nb) into a grey image.
cv::Mat renderScene(const Camera& cam, const std::vector<Vector3d>& centres, const std::vector<Matrix3d>& tagRots,
                    const Vector3d& rBNn, const Vector3d& thetaNb)
{
    static const cv::Mat markerSource = [] {
        const int markerPx = 240;
        const int margin = 40;
        cv::Mat marker;
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50).generateImageMarker(5, markerPx, marker, 1);
        cv::Mat src(markerPx + 2 * margin, markerPx + 2 * margin, CV_8UC1, cv::Scalar(255));
        marker.copyTo(src(cv::Rect(margin, margin, markerPx, markerPx)));
        return src;
    }();
    const double margin = 40.0;
    const double markerPx = 240.0;
    const std::vector<cv::Point2f> srcCorners = {
        cv::Point2f(static_cast<float>(margin), static_cast<float>(margin)),
        cv::Point2f(static_cast<float>(margin + markerPx), static_cast<float>(margin)),
        cv::Point2f(static_cast<float>(margin + markerPx), static_cast<float>(margin + markerPx)),
        cv::Point2f(static_cast<float>(margin), static_cast<float>(margin + markerPx))};

    const Matrix3d Rnc = rpy2rot(thetaNb) * cam.Rbc();
    const Vector3d rCNn = rBNn + rpy2rot(thetaNb) * cam.rCBb();
    cv::Mat frame(cam.imageHeight(), cam.imageWidth(), CV_8UC1, cv::Scalar(90));
    const Matrix<double, 4, 3> c = tags::canonicalCorners(kTagSide);
    for (std::size_t j = 0; j < centres.size(); ++j)
    {
        std::vector<cv::Point2f> dst;
        bool ok = true;
        for (int k = 0; k < 4; ++k)
        {
            const Vector3d pn = centres[j] + tagRots[j] * c.row(k).transpose();
            const Vector3d pc = Rnc.transpose() * (pn - rCNn);
            if (!(pc.z() > 0.05))
            {
                ok = false;
                break;
            }
            dst.emplace_back(static_cast<float>(cam.fx() * pc.x() / pc.z() + cam.cx()),
                             static_cast<float>(cam.fy() * pc.y() / pc.z() + cam.cy()));
        }
        if (!ok)
        {
            continue;
        }
        const cv::Mat h = cv::getPerspectiveTransform(srcCorners, dst);
        cv::Mat warped;
        cv::Mat mask;
        cv::warpPerspective(markerSource, warped, h, frame.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::warpPerspective(cv::Mat(markerSource.size(), CV_8UC1, cv::Scalar(255)), mask, h, frame.size(), cv::INTER_NEAREST,
                            cv::BORDER_CONSTANT, cv::Scalar(0));
        warped.copyTo(frame, mask);
    }
    return frame;
}

/// Ground-truth scene and camera path for the synthetic sequence (world = start body frame).
struct SyntheticSequence
{
    std::vector<Vector3d> tagCentres;
    std::vector<Matrix3d> tagRots;
    std::vector<Vector3d> camPositions;
    std::vector<Vector3d> camAngles;
};

SyntheticSequence makeSequence(int frames)
{
    SyntheticSequence s;
    // Tags face the camera (normal towards -x, i.e. back at the start camera); pattern x-axis is world +y.
    Matrix3d facing;
    facing.col(0) = Vector3d(0.0, 1.0, 0.0);
    facing.col(1) = Vector3d(0.0, 0.0, -1.0);
    facing.col(2) = Vector3d(-1.0, 0.0, 0.0);
    const std::vector<Vector3d> centres = {Vector3d(2.6, -0.55, 0.25), Vector3d(2.8, 0.6, -0.2),
                                           Vector3d(3.2, -0.2, -0.3), Vector3d(3.4, 0.25, 0.35)};
    for (const Vector3d& c : centres)
    {
        s.tagCentres.push_back(c);
        s.tagRots.push_back(facing);
    }
    for (int k = 0; k < frames; ++k)
    {
        const double u = static_cast<double>(k) / static_cast<double>(frames - 1);
        s.camPositions.push_back(Vector3d(1.0 * u, 0.08 * std::sin(2.0 * std::numbers::pi_v<double> * u), 0.03 * u));
        s.camAngles.push_back(Vector3d(0.02 * std::sin(3.0 * u), 0.03 * u, 0.12 * u));   // starts at identity (the gauge)
    }
    return s;
}

SlamPoseConfig syntheticConfig()
{
    SlamPoseConfig cfg;
    cfg.tagSideM = kTagSide;
    // The renders have ~0.1 px corner noise, so the 2 px field default is replaced by 1 px for this check (NIS on the
    // synthetic data is far below its expectation with 2 px). The residual error budget is documented in the report:
    // the detector's corners are ~0.9 % inside the exact projection, a ~2-4 cm depth bias at 3 m.
    cfg.pixelSigmaPx = 1.0;
    return cfg;
}

SlamPoseRecord recordWith(Index id, int misses, int hits, bool confirmed, int lastSeen, int behind = 0)
{
    SlamPoseRecord r;
    r.id = id;
    r.misses = misses;
    r.hits = hits;
    r.confirmed = confirmed;
    r.lastSeenFrame = lastSeen;
    r.behindFrames = behind;
    return r;
}

} // namespace

SCENARIO("tag lifecycle: culling rules")
{
    // Explicit limits: the rules are checked against these values (the defaults are tuned for the real sequence).
    SlamPoseConfig cfg;
    cfg.confirmedMaxMisses = 10;
    cfg.maxUnseenFrames = 120;
    GIVEN("records with different counters at frame 100")
    {
        const std::vector<SlamPoseRecord> records = {
            recordWith(0, 3, 1, false, 99),            // unconfirmed, 3 misses -> culled
            recordWith(1, 2, 1, false, 99),            // unconfirmed, 2 misses -> kept
            recordWith(2, 10, 9, true, 99),            // confirmed, 10 misses -> culled
            recordWith(3, 9, 9, true, 99),             // confirmed, 9 misses -> kept
            recordWith(4, 0, 5, true, 99, 3),          // behind the camera for 3 frames -> culled
            recordWith(5, 0, 5, true, 10),             // not seen for 90 frames (<= 120) -> kept
            recordWith(6, 0, 5, true, -50),            // not seen for 150 frames -> culled
        };
        WHEN("the cull set is selected")
        {
            const std::vector<Index> ids = SlamPoseTracker::selectCulls(records, 100, cfg);
            THEN("exactly the records meeting one rule are removed")
            {
                CHECK(ids == std::vector<Index>{0, 2, 4, 6});
            }
        }
    }
    GIVEN("more records than the capacity")
    {
        SlamPoseConfig small = cfg;
        small.maxTags = 2;
        const std::vector<SlamPoseRecord> records = {
            recordWith(0, 1, 5, true, 99), recordWith(1, 2, 5, true, 99), recordWith(2, 0, 5, true, 99),
            recordWith(3, 0, 5, true, 98)};
        WHEN("the cull set is selected")
        {
            const std::vector<Index> ids = SlamPoseTracker::selectCulls(records, 100, small);
            THEN("the records with the most misses are removed until the capacity is met")
            {
                REQUIRE(ids.size() == 2);
                CHECK(ids[0] == 1);
                CHECK(ids[1] == 0);
            }
        }
    }
}

namespace
{

/// Result of the synthetic sequence (computed once: doctest re-runs GIVEN/WHEN for every THEN).
struct SequenceResult
{
    bool finite = false;
    int mapFailures = 0;
    int flowSkippedNoCurvature = 0;      ///< flow updates with no information (skipped, counted separately from failures)
    int totalMatched = 0;
    int totalBirths = 0;
    std::size_t records = 0;
    std::size_t trajectoryLength = 0;
    Vector3d camEstimate = Vector3d::Zero();
    Vector3d camTruth = Vector3d::Zero();
    double positionSqrtCovNorm = 0.0;
    std::vector<double> tagErrors;       ///< distance of each record to its nearest truth tag [m]
    std::vector<double> tiltCosines;     ///< |R(2,0)| of each record (gimbal-safety measure)
};

const SequenceResult& runSyntheticSequence()
{
    static const SequenceResult result = [] {
        const Camera cam = syntheticCamera();
        const int frames = 45;
        const SyntheticSequence truth = makeSequence(frames);
        SlamPoseTracker tracker(cam, syntheticConfig());
        SequenceResult r;
        for (int k = 0; k < frames; ++k)
        {
            const cv::Mat gray = renderScene(cam, truth.tagCentres, truth.tagRots, truth.camPositions[static_cast<std::size_t>(k)],
                                             truth.camAngles[static_cast<std::size_t>(k)]);
            const slam::SlamPoseFrameStats st = tracker.step(gray, k, static_cast<double>(k) / 29.97);
            r.totalMatched += st.matched;
            r.totalBirths += st.births;
        }
        r.mapFailures = tracker.mapFailures();
        r.flowSkippedNoCurvature = static_cast<int>(tracker.flowMapStatusCounts().skippedNoCurvature);
        r.records = tracker.records().size();
        r.trajectoryLength = tracker.trajectory().size();
        r.camEstimate = tracker.trajectory().back();
        r.camTruth = truth.camPositions.back();
        r.positionSqrtCovNorm = tracker.positionSqrtCov().norm();
        const Eigen::VectorXd mu = tracker.estimator().mean();
        r.finite = mu.allFinite();
        for (const SlamPoseRecord& rec : tracker.records())
        {
            const Vector3d rL = mu.segment<3>(tracker.model().landmarkOffset(rec.id));
            double best = 1e9;
            for (const Vector3d& c : truth.tagCentres)
            {
                best = std::min(best, (rL - c).norm());
            }
            r.tagErrors.push_back(best);
            const Matrix3d Rnl = rpy2rot(mu.segment<3>(tracker.model().landmarkOffset(rec.id) + 3));
            r.tiltCosines.push_back(std::abs(Rnl(2, 0)));
        }
        return r;
    }();
    return result;
}

} // namespace

SCENARIO("synthetic identical-tag sequence: the camera trajectory and tag poses are recovered")
{
    GIVEN("four identical-ID markers of side 0.2 m seen by a camera moving 1 m forward with sideways motion (45 frames)")
    {
        const SequenceResult& r = runSyntheticSequence();
        THEN("the MAP never fails, the state is finite and the tracking is continuous")
        {
            CHECK(r.mapFailures == 0);
            CHECK(r.finite);
            CHECK(r.totalMatched > 2 * 45 * 4 / 3);
            CHECK(r.trajectoryLength == 45);
        }
        THEN("the flow updates with no information (first frames, zero curvature) are skipped and counted apart from failures")
        {
            CHECK(r.flowSkippedNoCurvature >= 1);
            CHECK(r.mapFailures == 0);
        }
        THEN("exactly four tags are in the map and each physical marker was born once (no duplicates)")
        {
            CHECK(r.records == 4);
            CHECK(r.totalBirths == 4);
        }
        THEN("the final camera position is within 10 cm of the truth over a 1 m path")
        {
            // Budget: the first frames have no tag constraint (birth baseline), and the detector corners are ~0.9 % inside
            // the exact projection (a 2-4 cm depth bias at 3 m). Measured: about 7 cm.
            CAPTURE_EIGEN(r.camEstimate);
            CAPTURE_EIGEN(r.camTruth);
            CHECK((r.camEstimate - r.camTruth).norm() < 0.10);
        }
        THEN("each tag is recovered within 6 cm of the nearest truth tag")
        {
            REQUIRE(r.tagErrors.size() == 4);
            for (double e : r.tagErrors)
            {
                CHECK(e < 0.06);
            }
        }
        THEN("the tag orientations keep the Euler representation well conditioned and the covariance is finite")
        {
            for (double c : r.tiltCosines)
            {
                CHECK(c <= std::sqrt(0.5) + 0.2);
            }
            CHECK(std::isfinite(r.positionSqrtCovNorm));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// R4-09: the archive re-identification gate is evaluated with triangular solves on square-root factors. The reference
// below is computed in THIS test only (explicit P and a dense solve): the production code never forms P or factorises it.
// ---------------------------------------------------------------------------------------------------------------------

namespace
{

/// Pose difference of the gate: [r_c - r_m ; rotation vector of R_m^T R_c] (re-derived here, independent of the code).
Matrix<double, 6, 1> referenceDelta(const Matrix<double, 6, 1>& mean, const Matrix<double, 6, 1>& cand)
{
    Matrix<double, 6, 1> d;
    d.head<3>() = cand.head<3>() - mean.head<3>();
    const Matrix3d dR = rpy2rot(mean.tail<3>()).transpose() * rpy2rot(cand.tail<3>());
    const Eigen::AngleAxisd aa(dR);
    d.tail<3>() = aa.angle() * aa.axis();
    return d;
}

/// Reference d^2 = delta^T (P + F)^{-1} delta with P given explicitly (test oracle only).
double referenceD2(const Matrix<double, 6, 1>& mean, const Matrix<double, 6, 6>& p, const Matrix<double, 6, 1>& cand,
                   double fp, double fa)
{
    Matrix<double, 6, 6> pf = p;
    for (Index i = 0; i < 3; ++i)
    {
        pf(i, i) += fp * fp;
        pf(i + 3, i + 3) += fa * fa;
    }
    const Matrix<double, 6, 1> d = referenceDelta(mean, cand);
    return d.dot(pf.ldlt().solve(d));
}

} // namespace

SCENARIO("archive re-identification gate equals the explicit-covariance reference to 1e-10 (square-root forms)")
{
    GIVEN("300 random archived tag densities (cov and info factors) with random poses and floors")
    {
        std::mt19937 rng(20260506u);
        std::uniform_real_distribution<double> unit(-1.0, 1.0);
        std::uniform_real_distribution<double> logScale(0.0, 1.0);
        std::uniform_real_distribution<double> floorPos(0.01, 0.2);
        std::uniform_real_distribution<double> floorAng(0.02, 0.3);

        double maxErrCov = 0.0;     // worst relative error of tagArchiveDistanceSq (sqrtCov form)
        double maxErrInfo = 0.0;    // worst relative error of tagArchiveDistanceSqInfo (sqrtInfo form)
        double maxCondRatio = 0.0;  // largest diagonal ratio of the stored factors (to show the range is exercised)
        int finite = 0;

        for (int trial = 0; trial < 300; ++trial)
        {
            // Upper square-root information Xi with a log-uniform diagonal (sd 1 .. 1e3 per axis).
            Matrix<double, 6, 6> xi = Matrix<double, 6, 6>::Zero();
            for (Index i = 0; i < 6; ++i)
            {
                xi(i, i) = std::pow(10.0, 3.0 * logScale(rng));
                for (Index j = i + 1; j < 6; ++j)
                {
                    xi(i, j) = 0.3 * xi(i, i) * unit(rng);
                }
            }
            // Covariance factor S (upper) with the same structure at the reciprocal scale.
            Matrix<double, 6, 6> s = Matrix<double, 6, 6>::Zero();
            for (Index i = 0; i < 6; ++i)
            {
                s(i, i) = std::pow(10.0, -4.0 + 3.0 * logScale(rng));
                for (Index j = i + 1; j < 6; ++j)
                {
                    s(i, j) = 0.3 * s(i, i) * unit(rng);
                }
            }
            maxCondRatio = std::max(maxCondRatio, xi.diagonal().maxCoeff() / xi.diagonal().minCoeff());

            Matrix<double, 6, 1> mean;
            mean.head<3>() = Vector3d(unit(rng), unit(rng), unit(rng));
            mean.tail<3>() = Vector3d(0.5 * unit(rng), 0.5 * unit(rng), 0.5 * unit(rng));
            Matrix<double, 6, 1> cand = mean;
            cand.head<3>() += Vector3d(0.3 * unit(rng), 0.3 * unit(rng), 0.3 * unit(rng));
            cand.tail<3>() += Vector3d(0.4 * unit(rng), 0.4 * unit(rng), 0.4 * unit(rng));
            const double fp = floorPos(rng);
            const double fa = floorAng(rng);

            // P for the covariance form is S^T S; for the information form P = Xi^{-1} Xi^{-T} (test oracle).
            const Matrix<double, 6, 6> pCov = s.transpose() * s;
            const Matrix<double, 6, 6> xiInv = xi.inverse();
            const Matrix<double, 6, 6> pInfo = xiInv * xiInv.transpose();

            const double refCov = referenceD2(mean, pCov, cand, fp, fa);
            const double refInfo = referenceD2(mean, pInfo, cand, fp, fa);
            const double gotCov = slam::tagArchiveDistanceSq(mean, s, cand, fp, fa);
            const double gotInfo = slam::tagArchiveDistanceSqInfo(mean, xi, cand, fp, fa);
            if (std::isfinite(gotCov) && std::isfinite(gotInfo))
            {
                ++finite;
            }
            maxErrCov = std::max(maxErrCov, std::abs(gotCov - refCov) / std::max(1.0, std::abs(refCov)));
            maxErrInfo = std::max(maxErrInfo, std::abs(gotInfo - refInfo) / std::max(1.0, std::abs(refInfo)));
        }
        CAPTURE(maxErrCov);
        CAPTURE(maxErrInfo);
        CAPTURE(maxCondRatio);
        WHEN("the gate is evaluated by the triangular-solve implementation")
        {
            THEN("every evaluation is finite")
            {
                CHECK(finite == 300);
            }
            THEN("tagArchiveDistanceSq (covariance-factor input) matches the reference to 1e-10 (relative)")
            {
                CHECK(maxErrCov <= 1e-10);
            }
            THEN("tagArchiveDistanceSqInfo (information-factor input, the production path) matches the reference to 1e-10")
            {
                CHECK(maxErrInfo <= 1e-10);
            }
            THEN("the factor scales span several decades, so the comparison is not trivially well conditioned")
            {
                CHECK(maxCondRatio > 100.0);
            }
        }
    }
}

SCENARIO("archive re-identification gate: zero offset gives zero distance and a larger offset gives a larger distance")
{
    GIVEN("an archived tag with a unit information factor and the default floors")
    {
        const Matrix<double, 6, 6> xi = Matrix<double, 6, 6>::Identity() * 50.0;
        const Matrix<double, 6, 1> mean = (Matrix<double, 6, 1>() << 0.1, 0.2, 0.3, 0.01, 0.02, 0.03).finished();
        WHEN("the candidate equals the archived pose")
        {
            const double d2 = slam::tagArchiveDistanceSqInfo(mean, xi, mean, 0.05, 0.1);
            THEN("the distance is exactly zero")
            {
                CHECK(d2 == doctest::Approx(0.0).epsilon(1e-14));
            }
        }
        WHEN("the candidate is displaced by 1 cm and then by 10 cm")
        {
            Matrix<double, 6, 1> near = mean;
            near(0) += 0.01;
            Matrix<double, 6, 1> far = mean;
            far(0) += 0.10;
            const double dn = slam::tagArchiveDistanceSqInfo(mean, xi, near, 0.05, 0.1);
            const double df = slam::tagArchiveDistanceSqInfo(mean, xi, far, 0.05, 0.1);
            THEN("the distance grows monotonically with the offset")
            {
                CHECK(dn > 0.0);
                CHECK(df > dn);
            }
        }
    }
}
