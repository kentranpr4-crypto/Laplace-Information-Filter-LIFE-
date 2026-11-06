// Scenario 6: indoor point-landmark visual SLAM (optical-flow-free ORB descriptors, information-form estimator).
// The estimator and the data association live in SlamPointPipeline; this file adapts them to the ScenarioRunner
// contract and draws the image overlay and the 3-D plot data.
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
#include "SlamPointPipeline.h"
#include "SystemVisualNav.h"
#include "rotation.hpp"

namespace
{

// Overlay colours (BGR for OpenCV). ASSUMED: the assignment text does not fix Scenario 6 colours.
const cv::Scalar kColourNewFeature(255, 0, 0);        // blue: features that started a new landmark this frame
const cv::Scalar kColourPredicted(0, 0, 255);         // red: visible predicted landmark, not associated
const cv::Scalar kColourAssociated(0, 255, 0);        // green: associated landmark, with its 3-sigma pixel ellipse
const cv::Scalar kColourText(255, 255, 255);          // white: status text, drawn last

// Landmark colours in the 3-D view (RGB in [0,1]), cycled by landmark id.
const std::array<Eigen::Vector3d, 8> kLandmarkPalette = {
    Eigen::Vector3d(0.90, 0.30, 0.30), Eigen::Vector3d(0.30, 0.80, 0.30), Eigen::Vector3d(0.30, 0.50, 1.00),
    Eigen::Vector3d(0.95, 0.80, 0.20), Eigen::Vector3d(0.80, 0.40, 0.90), Eigen::Vector3d(0.20, 0.85, 0.85),
    Eigen::Vector3d(0.95, 0.55, 0.20), Eigen::Vector3d(0.70, 0.70, 0.70)};

/// Draws the 3-sigma (nSigma) confidence ellipse of a 2-D covariance centred at c (OpenCV pixel coordinates).
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
    if (!(a < 1e4) || !(b < 1e4))
    {
        return;   // degenerate (very uncertain) ellipse: skipped to keep the overlay readable
    }
    const double angleDeg = std::atan2(axis.y(), axis.x()) * 180.0 / std::acos(-1.0);
    cv::ellipse(img, cv::RotatedRect(cv::Point2f(static_cast<float>(c.x()), static_cast<float>(c.y())),
                                     cv::Size2f(static_cast<float>(2.0 * a), static_cast<float>(2.0 * b)),
                                     static_cast<float>(angleDeg)),
                colour, 1, cv::LINE_AA);
}

void drawCross(cv::Mat& img, const Eigen::Vector2d& p, const cv::Scalar& colour)
{
    const cv::Point c(cvRound(p.x()), cvRound(p.y()));
    cv::drawMarker(img, c, colour, cv::MARKER_CROSS, 9, 1, cv::LINE_AA);
}

class Scenario6Runner final : public ScenarioRunner
{
public:
    void initialise(const Camera& camera, double fps, cv::Size frameSize, const std::string& videoPath) override
    {
        (void)videoPath;
        if (!camera.isValid())
        {
            throw std::invalid_argument("Scenario 6: camera is not valid");
        }
        if (!(fps > 0.0) || frameSize.width <= 0 || frameSize.height <= 0)
        {
            throw std::invalid_argument("Scenario 6: fps and frame size must be positive");
        }
        if (frameSize.width != camera.imageWidth() || frameSize.height != camera.imageHeight())
        {
            throw std::invalid_argument("Scenario 6: frame size does not match the camera calibration");
        }
        fps_ = fps;
        frameSize_ = frameSize;
        pipeline_ = std::make_unique<slam::SlamPointPipeline>(camera, slam::SlamConfig{});
    }

    void processFrame(const cv::Mat& frame, int frameIndex, double t) override
    {
        if (!pipeline_)
        {
            throw std::logic_error("Scenario 6: initialise() must be called before processFrame()");
        }
        if (frame.empty() || frame.type() != CV_8UC3 || frame.size() != frameSize_)
        {
            throw std::invalid_argument("Scenario 6: frame must be a BGR8 image of the initialised size");
        }
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        stats_ = pipeline_->step(gray, frameIndex, t);
        overlay_ = frame.clone();
        drawOverlay();
    }

    PlotFrameData plotData() const override
    {
        PlotFrameData d;
        if (!pipeline_)
        {
            return d;
        }
        d.overlay = overlay_;
        const Eigen::VectorXd mu = pipeline_->estimator().mean();
        const Eigen::Vector3d rBNn = mu.segment<3>(SystemVisualNav::kEtaOffset);
        const Eigen::Vector3d thetaNb = mu.segment<3>(SystemVisualNav::kEtaOffset + 3);
        const Camera& cam = pipeline_->camera();
        d.Rnc = rigid::cameraRotationInWorld(thetaNb, cam.Rbc());
        d.rCNn = rigid::cameraOriginInWorld(thetaNb, rBNn, cam.rCBb());
        d.positionSqrtCov = pipeline_->positionSqrtCov();
        d.trajectory = pipeline_->trajectory();

        std::vector<SystemVisualNav::Landmark> lms = pipeline_->model().landmarks();
        for (const slam::SlamLandmarkRecord& rec : pipeline_->map().records())
        {
            const auto it = std::find_if(lms.begin(), lms.end(),
                                         [&rec](const SystemVisualNav::Landmark& l) { return l.id == rec.id; });
            if (it == lms.end())
            {
                continue;
            }
            PlotFrameData::Landmark lm;
            lm.id = static_cast<int>(rec.id);
            lm.rLNn = mu.segment<3>(it->offset);
            lm.positionSqrtCov = pipeline_->landmarkSqrtCov(it->offset);
            lm.isPose = false;
            const Eigen::Vector3d rgb = kLandmarkPalette[static_cast<std::size_t>(rec.id) % kLandmarkPalette.size()];
            lm.colourRGB = rgb;
            lm.visible = rec.associatedThisFrame;
            d.landmarks.push_back(lm);
        }

        const double tNow = stats_.time;
        d.statusLines = {
            "Scenario 6 (ASSUMED overlay colours)",
            "frame " + std::to_string(stats_.frame) + "  t=" + std::to_string(tNow).substr(0, 6) + " s",
            "features " + std::to_string(stats_.features) + "  visible " + std::to_string(stats_.visible)
                + "  matched " + std::to_string(stats_.associated),
            "landmarks " + std::to_string(stats_.landmarks) + "  births " + std::to_string(stats_.births)
                + "  culled " + std::to_string(stats_.culled),
            "NIS " + std::to_string(stats_.nis).substr(0, 6) + " (dof " + std::to_string(stats_.nisDof) + ")",
            std::string("flow ") + (stats_.flowUpdated ? (stats_.flowRotationModel ? "R (pure rotation)" : "E (epipolar)") : "none")
                + "  NIS/dof " + std::to_string(stats_.flowDof > 0 ? stats_.flowNis / stats_.flowDof : 0.0).substr(0, 5),
        };
        return d;
    }

    std::string name() const override { return "Scenario 6: indoor point landmarks"; }

private:
    /// Draw order (ASSUMED): new-landmark features (blue), visible unassociated predictions (red),
    /// associated matches with 3-sigma pixel ellipses (green), status text last.
    void drawOverlay()
    {
        const slam::PredictionSet& preds = pipeline_->lastPredictions();
        const slam::AssociationResult& assoc = pipeline_->lastAssociation();
        const std::vector<slam::PointFeature>& feats = pipeline_->lastFeatures();

        for (const Eigen::Vector2d& p : pipeline_->lastBirthPixels())
        {
            cv::circle(overlay_, cv::Point(cvRound(p.x()), cvRound(p.y())), 4, kColourNewFeature, 1, cv::LINE_AA);
        }

        std::vector<bool> isMatched(preds.predictions.size(), false);
        for (std::size_t p : assoc.matchedPrediction)
        {
            isMatched[p] = true;
        }
        for (std::size_t p = 0; p < preds.predictions.size(); ++p)
        {
            const slam::LandmarkPrediction& pr = preds.predictions[p];
            if (pr.visible && !isMatched[p])
            {
                drawCross(overlay_, pr.pixel, kColourPredicted);
            }
        }

        for (std::size_t k = 0; k < assoc.matchedPrediction.size(); ++k)
        {
            const slam::LandmarkPrediction& pr = preds.predictions[assoc.matchedPrediction[k]];
            const double sig = assoc.observations[k].sigmaPx;
            Eigen::Matrix2d s = pr.pixelCov;
            s(0, 0) += sig * sig;
            s(1, 1) += sig * sig;
            drawConfidenceEllipse(overlay_, pr.pixel, s, 3.0, kColourAssociated);
            const Eigen::Vector2d& z = feats[static_cast<std::size_t>(assoc.matchedFeature[k])].pixel;
            cv::circle(overlay_, cv::Point(cvRound(z.x()), cvRound(z.y())), 3, kColourAssociated, -1, cv::LINE_AA);
        }

        const std::string line = "S6 frame " + std::to_string(stats_.frame) + "  landmarks " + std::to_string(stats_.landmarks)
            + "  matched " + std::to_string(stats_.associated) + "  NIS " + std::to_string(stats_.nis).substr(0, 6);
        cv::putText(overlay_, line, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(overlay_, line, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9, kColourText, 2, cv::LINE_AA);
    }

    std::unique_ptr<slam::SlamPointPipeline> pipeline_;
    cv::Mat overlay_;
    slam::SlamFrameStats stats_;
    double fps_ = 0.0;
    cv::Size frameSize_;
};

} // namespace

std::unique_ptr<ScenarioRunner> makeScenario6Runner()
{
    return std::make_unique<Scenario6Runner>();
}
