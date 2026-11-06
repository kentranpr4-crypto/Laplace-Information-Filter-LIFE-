// Scenario 5: indoor identical-ID ArUco tags (optics: ArUco corners; SLAM: pose landmarks, geometric association).
// The estimator, association and lifecycle live in SlamPoseTracker; this file adapts them to the ScenarioRunner contract
// and draws the image overlay and the 3-D plot data.
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "Camera.h"
#include "GaussianBase.h"
#include "Pose.hpp"
#include "ScenarioRunner.h"
#include "SlamPoseTracker.h"
#include "SystemVisualNav.h"
#include "rotation.hpp"

namespace
{

// Overlay colours (BGR for OpenCV). ASSUMED: the assignment text does not fix Scenario 5 colours.
const cv::Scalar kColourDetected(0, 255, 255);      // yellow: every id-5 detection (corners and outline)
const cv::Scalar kColourMatched(0, 255, 0);         // green: predicted tag associated to a detection, with 3-sigma corner ellipses
const cv::Scalar kColourMissed(0, 0, 255);          // red: strictly visible predicted tag with no associated detection
const cv::Scalar kColourBirth(255, 0, 255);         // magenta: detection that starts a (pending) tag
const cv::Scalar kColourText(255, 255, 255);        // white: labels and status text, drawn last

// Tag colours in the 3-D view (RGB in [0,1]), cycled by landmark id. ASSUMED.
const std::array<Eigen::Vector3d, 8> kTagPalette = {
    Eigen::Vector3d(0.90, 0.30, 0.30), Eigen::Vector3d(0.30, 0.80, 0.30), Eigen::Vector3d(0.30, 0.50, 1.00),
    Eigen::Vector3d(0.95, 0.80, 0.20), Eigen::Vector3d(0.80, 0.40, 0.90), Eigen::Vector3d(0.20, 0.85, 0.85),
    Eigen::Vector3d(0.95, 0.55, 0.20), Eigen::Vector3d(0.70, 0.70, 0.70)};

/// Outline of one tag from its 8-vector of corners (closed polygon).
void drawOutline(cv::Mat& img, const Eigen::Matrix<double, 8, 1>& uv, const cv::Scalar& colour, int thickness)
{
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (int k = 0; k < 4; ++k)
    {
        poly.emplace_back(cvRound(uv(2 * k)), cvRound(uv(2 * k + 1)));
    }
    cv::polylines(img, poly, true, colour, thickness, cv::LINE_AA);
}

/// Draws the nSigma confidence ellipse of a 2-D covariance centred at c (OpenCV pixel coordinates).
void drawConfidenceEllipse(cv::Mat& img, const Eigen::Vector2d& c, const Eigen::Matrix2d& cov, double nSigma,
                           const cv::Scalar& colour)
{
    if (!cov.allFinite())
    {
        return;
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(0.5 * (cov + cov.transpose()));
    if (es.info() != Eigen::Success)
    {
        return;
    }
    const double r = std::sqrt(GaussianBase::mahalanobisThresholdSq(nSigma, 2.0));
    const Eigen::Vector2d axis = es.eigenvectors().col(1);     // major axis
    const double a = r * std::sqrt(std::max(0.0, es.eigenvalues()(1)));
    const double b = r * std::sqrt(std::max(0.0, es.eigenvalues()(0)));
    if (!(a < 1e3) || !(b < 1e3))
    {
        return;   // degenerate (very uncertain) ellipse: skipped to keep the overlay readable
    }
    const double angleDeg = std::atan2(axis.y(), axis.x()) * 180.0 / std::acos(-1.0);
    cv::ellipse(img, cv::RotatedRect(cv::Point2f(static_cast<float>(c.x()), static_cast<float>(c.y())),
                                     cv::Size2f(static_cast<float>(2.0 * a), static_cast<float>(2.0 * b)),
                                     static_cast<float>(angleDeg)),
                colour, 1, cv::LINE_AA);
}

class Scenario5Runner final : public ScenarioRunner
{
public:
    void initialise(const Camera& camera, double fps, cv::Size frameSize, const std::string& videoPath) override
    {
        (void)videoPath;
        if (!camera.isValid())
        {
            throw std::invalid_argument("Scenario 5: camera is not valid");
        }
        if (!(fps > 0.0) || frameSize.width <= 0 || frameSize.height <= 0)
        {
            throw std::invalid_argument("Scenario 5: fps and frame size must be positive");
        }
        if (frameSize.width != camera.imageWidth() || frameSize.height != camera.imageHeight())
        {
            throw std::invalid_argument("Scenario 5: frame size does not match the camera calibration");
        }
        fps_ = fps;
        frameSize_ = frameSize;
        tracker_ = std::make_unique<slam::SlamPoseTracker>(camera, slam::SlamPoseConfig{});
    }

    void processFrame(const cv::Mat& frame, int frameIndex, double t) override
    {
        if (!tracker_)
        {
            throw std::logic_error("Scenario 5: initialise() must be called before processFrame()");
        }
        if (frame.empty() || frame.type() != CV_8UC3 || frame.size() != frameSize_)
        {
            throw std::invalid_argument("Scenario 5: frame must be a BGR8 image of the initialised size");
        }
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        stats_ = tracker_->step(gray, frameIndex, t);

        // Plot data (computed once per frame; plotData() is const).
        const Eigen::VectorXd mu = tracker_->estimator().mean();
        const Camera& cam = tracker_->camera();
        const Eigen::Vector3d rBNn = mu.segment<3>(SystemVisualNav::kEtaOffset);
        const Eigen::Vector3d thetaNb = mu.segment<3>(SystemVisualNav::kEtaOffset + 3);
        Rnc_ = rigid::cameraRotationInWorld(thetaNb, cam.Rbc());
        rCNn_ = rigid::cameraOriginInWorld(thetaNb, rBNn, cam.rCBb());
        std::vector<Eigen::Index> offsets;
        offsets.push_back(SystemVisualNav::kEtaOffset);
        for (const slam::SlamPoseRecord& r : tracker_->records())
        {
            offsets.push_back(tracker_->model().landmarkOffset(r.id));
        }
        const std::vector<Eigen::Matrix3d> sq = tracker_->blockSqrtCovs(offsets);
        positionSqrtCov_ = sq.front();
        landmarkSqrtCov_.assign(sq.begin() + 1, sq.end());

        overlay_ = frame.clone();
        drawOverlay();
    }

    PlotFrameData plotData() const override
    {
        PlotFrameData d;
        if (!tracker_)
        {
            return d;
        }
        d.overlay = overlay_;
        d.Rnc = Rnc_;
        d.rCNn = rCNn_;
        d.positionSqrtCov = positionSqrtCov_;
        d.trajectory = tracker_->trajectory();

        const Eigen::VectorXd mu = tracker_->estimator().mean();
        const double side = tracker_->config().tagSideM;
        const std::vector<slam::SlamPoseRecord>& recs = tracker_->records();
        for (std::size_t k = 0; k < recs.size(); ++k)
        {
            const Eigen::Index off = tracker_->model().landmarkOffset(recs[k].id);
            PlotFrameData::Landmark lm;
            lm.id = static_cast<int>(recs[k].id);
            lm.isPose = true;
            lm.rLNn = mu.segment<3>(off);
            lm.Rnl = rpy2rot(mu.segment<3>(off + 3));
            lm.size = side;
            lm.positionSqrtCov = landmarkSqrtCov_[k];
            lm.colourRGB = kTagPalette[static_cast<std::size_t>(recs[k].id) % kTagPalette.size()];
            lm.visible = recs[k].associatedThisFrame;
            d.landmarks.push_back(lm);
        }

        const slam::SlamPoseFrameStats& s = stats_;
        d.statusLines = {
            "Scenario 5 (ASSUMED overlay colours)",
            "frame " + std::to_string(s.frame) + "  t=" + std::to_string(s.time).substr(0, 6) + " s",
            "detections " + std::to_string(s.detections) + "  matched " + std::to_string(s.matched)
                + "  ambiguous " + std::to_string(s.ambiguous) + "  conflicts " + std::to_string(s.conflicts),
            "tags " + std::to_string(s.tags) + "  births " + std::to_string(s.births) + "  pending "
                + std::to_string(s.pending) + "  culled " + std::to_string(s.culled),
            "NIS " + std::to_string(s.nis).substr(0, 6) + " (dof " + std::to_string(s.nisDof) + ")",
            std::string("flow ") + (s.flowUpdated ? (s.flowRotationModel ? "R (pure rotation)" : "E (epipolar)") : "none")
                + "  NIS/dof " + std::to_string(s.flowDof > 0 ? s.flowNis / s.flowDof : 0.0).substr(0, 5)
                + "  occluded " + std::to_string(s.occluded),
            "identities " + std::to_string(tracker_->identitiesCreated()) + "  re-identified " + std::to_string(s.reidentified),
        };
        return d;
    }

    std::string name() const override { return "Scenario 5: identical-ID ArUco tags"; }

private:
    /// Draw order (ASSUMED): predictions (red missed, green matched with 3-sigma ellipses), detections (yellow),
    /// birth candidates (magenta), labels, status text last.
    void drawOverlay()
    {
        const slam::TagPredictionSet& preds = tracker_->lastPredictions();
        const slam::TagAssociationResult& assoc = tracker_->lastAssociation();
        const std::vector<tags::TagDetection>& dets = tracker_->lastDetections();

        for (std::size_t j = 0; j < preds.candidates.size(); ++j)
        {
            const slam::TagCandidate& c = preds.candidates[j];
            if (assoc.candidateDetection[j] < 0 && c.visible)
            {
                drawOutline(overlay_, c.pred.uv, kColourMissed, 1);
            }
        }
        for (std::size_t j = 0; j < preds.candidates.size(); ++j)
        {
            const slam::TagCandidate& c = preds.candidates[j];
            if (assoc.candidateDetection[j] < 0)
            {
                continue;
            }
            drawOutline(overlay_, c.pred.uv, kColourMatched, 2);
            for (int k = 0; k < 4; ++k)
            {
                const Eigen::Vector2d corner = c.pred.uv.segment<2>(2 * k);
                drawConfidenceEllipse(overlay_, corner, c.S.block<2, 2>(2 * k, 2 * k), 3.0, kColourMatched);
            }
        }
        for (const tags::TagDetection& d : dets)
        {
            drawOutline(overlay_, d.cornerVector(), kColourDetected, 1);
            for (int k = 0; k < 4; ++k)
            {
                cv::circle(overlay_, cv::Point(cvRound(d.corners(k, 0)), cvRound(d.corners(k, 1))), 3, kColourDetected, -1,
                           cv::LINE_AA);
            }
        }
        for (std::size_t i = 0; i < dets.size() && i < assoc.birthCandidate.size(); ++i)
        {
            if (assoc.birthCandidate[i])
            {
                drawOutline(overlay_, dets[i].cornerVector(), kColourBirth, 2);
            }
        }
        // Labels: landmark ids (all tags share the same ArUco id 5, so the id here is only the birth index).
        for (const slam::TagCandidate& c : preds.candidates)
        {
            const cv::Point p(cvRound(c.centre.x()), cvRound(c.centre.y()));
            const std::string label = "#" + std::to_string(c.id);
            cv::putText(overlay_, label, p, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(overlay_, label, p, cv::FONT_HERSHEY_SIMPLEX, 0.6, kColourText, 1, cv::LINE_AA);
        }
        const std::string line = "S5 frame " + std::to_string(stats_.frame) + "  tags " + std::to_string(stats_.tags)
            + "  matched " + std::to_string(stats_.matched) + "  NIS " + std::to_string(stats_.nis).substr(0, 6);
        cv::putText(overlay_, line, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(overlay_, line, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, kColourText, 2, cv::LINE_AA);
    }

    std::unique_ptr<slam::SlamPoseTracker> tracker_;
    cv::Mat overlay_;
    slam::SlamPoseFrameStats stats_;
    Eigen::Matrix3d Rnc_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rCNn_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d positionSqrtCov_ = Eigen::Matrix3d::Zero();
    std::vector<Eigen::Matrix3d> landmarkSqrtCov_;
    double fps_ = 0.0;
    cv::Size frameSize_;
};

} // namespace

std::unique_ptr<ScenarioRunner> makeScenario5Runner()
{
    return std::make_unique<Scenario5Runner>();
}
