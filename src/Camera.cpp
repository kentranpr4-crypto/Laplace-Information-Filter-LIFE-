#include "Camera.h"

#include <cmath>
#include <format>
#include <stdexcept>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/** @brief Maximum Newton iterations for the undistortion inverse. */
constexpr int kMaxUndistortIterations = 50;

/** @brief Convergence tolerance on the Newton step in normalised coordinates. */
constexpr double kUndistortTolerance = 1e-13;

/** @brief Tolerance for rotation orthonormality checks. */
constexpr double kRotationTolerance = 1e-6;

bool isValidDistortionLength(std::size_t n) noexcept
{
    return n == 0 || n == 4 || n == 5 || n == 8;
}

/** @brief Copies an OpenCV 3x3 (or 3x1) CV_64F matrix into Eigen storage, checking the shape. */
Eigen::Matrix3d toEigenMatrix3x3(const cv::Mat & m)
{
    if (m.rows != 3 || m.cols != 3)
    {
        throw std::runtime_error(std::format("expected a 3x3 matrix, got {}x{}", m.rows, m.cols));
    }
    cv::Mat d;
    m.convertTo(d, CV_64F);
    Eigen::Matrix3d out;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            out(r, c) = d.at<double>(r, c);
        }
    }
    return out;
}

Eigen::Vector3d toEigenVector3(const cv::Mat & m)
{
    if (m.total() != 3)
    {
        throw std::runtime_error(std::format("expected 3 elements, got {}", m.total()));
    }
    cv::Mat d;
    m.convertTo(d, CV_64F);
    d = d.reshape(1, 3);
    return Eigen::Vector3d(d.at<double>(0, 0), d.at<double>(1, 0), d.at<double>(2, 0));
}

Eigen::Matrix3d defaultRbc()
{
    Eigen::Matrix3d R;
    R << 0.0, 0.0, 1.0,
         1.0, 0.0, 0.0,
         0.0, 1.0, 0.0;
    return R;
}

void checkRotation(const Eigen::Matrix3d & R)
{
    if (!R.allFinite())
    {
        throw std::invalid_argument("Rbc contains non-finite values");
    }
    if ((R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > kRotationTolerance)
    {
        throw std::invalid_argument("Rbc is not orthonormal");
    }
    if (R.determinant() < 0.0)
    {
        throw std::invalid_argument("Rbc is a reflection (det < 0)");
    }
}
} // namespace

Camera::Camera(int imageWidth, int imageHeight, const Eigen::Matrix3d & cameraMatrix, const std::vector<double> & distortionCoefficients)
    : imageWidth_(imageWidth)
    , imageHeight_(imageHeight)
    , K_(cameraMatrix)
    , Rbc_(defaultRbc())
    , rCBb_(Eigen::Vector3d::Zero())
{
    if (imageWidth_ <= 0 || imageHeight_ <= 0)
    {
        throw std::invalid_argument(std::format("image size must be positive, got {}x{}", imageWidth_, imageHeight_));
    }
    if (!K_.allFinite())
    {
        throw std::invalid_argument("camera matrix contains non-finite values");
    }
    if (K_(0, 0) <= 0.0 || K_(1, 1) <= 0.0)
    {
        throw std::invalid_argument(std::format("focal lengths must be positive, got fx={} fy={}", K_(0, 0), K_(1, 1)));
    }
    if (K_(1, 0) != 0.0 || K_(2, 0) != 0.0 || K_(2, 1) != 0.0 || K_(2, 2) != 1.0)
    {
        throw std::invalid_argument("camera matrix must have the form [fx 0 cx; 0 fy cy; 0 0 1]");
    }
    if (K_(0, 1) != 0.0)
    {
        throw std::invalid_argument(std::format("camera matrix skew must be zero (got {}): the projection model has no skew term", K_(0, 1)));
    }
    setDistortion(distortionCoefficients);
}

void Camera::setDistortion(const std::vector<double> & coefficients)
{
    if (!isValidDistortionLength(coefficients.size()))
    {
        throw std::invalid_argument(std::format("unsupported number of distortion coefficients: {} (accepted: 0, 4, 5, 8)", coefficients.size()));
    }
    for (double k : coefficients)
    {
        if (!std::isfinite(k))
        {
            throw std::invalid_argument("distortion coefficients contain non-finite values");
        }
    }
    distortion_ = coefficients;
    kd_.setZero();
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        kd_(static_cast<Eigen::Index>(i)) = coefficients[i];
    }
}

void Camera::setBodyToCameraExtrinsics(const Eigen::Matrix3d & Rbc, const Eigen::Vector3d & rCBb)
{
    checkRotation(Rbc);
    if (!rCBb.allFinite())
    {
        throw std::invalid_argument("rCBb contains non-finite values");
    }
    Rbc_ = Rbc;
    rCBb_ = rCBb;
}

Camera Camera::load(const std::filesystem::path & cameraXmlPath)
{
    const std::string where = cameraXmlPath.string();
    if (!std::filesystem::is_regular_file(cameraXmlPath))
    {
        throw std::runtime_error(std::format("{}: camera file not found", where));
    }

    cv::FileStorage fs(where, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw std::runtime_error(std::format("{}: cannot open camera file for reading", where));
    }

    auto fail = [&where](const std::string & message) -> std::runtime_error
    {
        return std::runtime_error(std::format("{}: {}", where, message));
    };

    try
    {
        if (fs["format_version"].isNone())
        {
            throw fail("missing format_version (expected 1)");
        }
        int version = 0;
        fs["format_version"] >> version;
        if (version != 1)
        {
            throw fail(std::format("unsupported format_version {} (expected 1)", version));
        }

        if (fs["image_width"].isNone() || fs["image_height"].isNone())
        {
            throw fail("missing image_width or image_height");
        }
        int width = 0;
        int height = 0;
        fs["image_width"] >> width;
        fs["image_height"] >> height;

        cv::Mat cameraMatrixNode;
        if (fs["camera_matrix"].isNone())
        {
            throw fail("missing camera_matrix");
        }
        fs["camera_matrix"] >> cameraMatrixNode;
        const Eigen::Matrix3d K = toEigenMatrix3x3(cameraMatrixNode);

        if (fs["distortion_coefficients"].isNone())
        {
            throw fail("missing distortion_coefficients");
        }
        cv::Mat distNode;
        fs["distortion_coefficients"] >> distNode;
        std::vector<double> distortion;
        if (!distNode.empty())
        {
            cv::Mat d;
            distNode.convertTo(d, CV_64F);
            distortion.assign(d.ptr<double>(0), d.ptr<double>(0) + d.total());
        }

        Camera camera(width, height, K, distortion);

        if (!fs["body_to_camera_rotation"].isNone())
        {
            cv::Mat rNode;
            fs["body_to_camera_rotation"] >> rNode;
            Eigen::Matrix3d Rbc = toEigenMatrix3x3(rNode);
            Eigen::Vector3d rCBb = Eigen::Vector3d::Zero();
            if (!fs["body_to_camera_position"].isNone())
            {
                cv::Mat pNode;
                fs["body_to_camera_position"] >> pNode;
                rCBb = toEigenVector3(pNode);
            }
            camera.setBodyToCameraExtrinsics(Rbc, rCBb);
        }

        CalibrationMetadata meta;
        if (!fs["rms_reprojection_error_px"].isNone())
        {
            fs["rms_reprojection_error_px"] >> meta.rmsReprojectionErrorPx;
        }
        if (!fs["num_views"].isNone())
        {
            fs["num_views"] >> meta.numViews;
        }
        if (!fs["num_points"].isNone())
        {
            fs["num_points"] >> meta.numPoints;
        }
        if (!fs["board_inner_corners_cols"].isNone())
        {
            fs["board_inner_corners_cols"] >> meta.boardInnerCornersCols;
        }
        if (!fs["board_inner_corners_rows"].isNone())
        {
            fs["board_inner_corners_rows"] >> meta.boardInnerCornersRows;
        }
        if (!fs["square_size_m"].isNone())
        {
            fs["square_size_m"] >> meta.squareSizeM;
        }
        if (!fs["source_video"].isNone())
        {
            fs["source_video"] >> meta.sourceVideo;
        }
        if (!fs["distortion_model"].isNone())
        {
            fs["distortion_model"] >> meta.distortionModel;
        }
        camera.setMetadata(meta);

        return camera;
    }
    catch (const std::invalid_argument & e)
    {
        throw fail(e.what());
    }
    catch (const cv::Exception & e)
    {
        throw fail(std::format("OpenCV error while reading: {}", e.what()));
    }
}

void Camera::save(const std::filesystem::path & cameraXmlPath) const
{
    if (!isValid())
    {
        throw std::runtime_error("cannot save an invalid (empty) camera");
    }
    const std::string where = cameraXmlPath.string();

    cv::Mat K(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            K.at<double>(r, c) = K_(r, c);
        }
    }
    cv::Mat R(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            R.at<double>(r, c) = Rbc_(r, c);
        }
    }
    cv::Mat p(3, 1, CV_64F);
    for (int r = 0; r < 3; ++r)
    {
        p.at<double>(r, 0) = rCBb_(r);
    }
    cv::Mat dist(1, static_cast<int>(distortion_.size()), CV_64F);
    for (std::size_t i = 0; i < distortion_.size(); ++i)
    {
        dist.at<double>(0, static_cast<int>(i)) = distortion_[i];
    }

    std::string modelLabel = metadata_.distortionModel;
    if (modelLabel.empty())
    {
        modelLabel = distortion_.size() == 8 ? "opencv-rational-8"
                   : distortion_.size() == 5 ? "opencv-standard-5"
                   : distortion_.size() == 4 ? "opencv-standard-4"
                   : "none";
    }

    cv::FileStorage fs(where, cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        throw std::runtime_error(std::format("{}: cannot open camera file for writing", where));
    }
    fs.writeComment("camera.xml schema version 1: see Camera.h for field documentation");
    fs << "format_version" << 1;
    fs << "image_width" << imageWidth_;
    fs << "image_height" << imageHeight_;
    fs << "camera_matrix" << K;
    fs << "distortion_coefficients" << dist;
    fs << "distortion_model" << modelLabel;
    fs << "body_to_camera_rotation" << R;
    fs << "body_to_camera_position" << p;
    if (std::isfinite(metadata_.rmsReprojectionErrorPx))
    {
        fs << "rms_reprojection_error_px" << metadata_.rmsReprojectionErrorPx;
    }
    if (metadata_.numViews > 0)
    {
        fs << "num_views" << metadata_.numViews;
    }
    if (metadata_.numPoints > 0)
    {
        fs << "num_points" << metadata_.numPoints;
    }
    if (metadata_.boardInnerCornersCols > 0)
    {
        fs << "board_inner_corners_cols" << metadata_.boardInnerCornersCols;
        fs << "board_inner_corners_rows" << metadata_.boardInnerCornersRows;
    }
    if (metadata_.squareSizeM > 0.0)
    {
        fs << "square_size_m" << metadata_.squareSizeM;
    }
    if (!metadata_.sourceVideo.empty())
    {
        fs << "source_video" << metadata_.sourceVideo;
    }
    fs.release();
}

Camera::Distortion Camera::evaluateDistortion(double x, double y) const noexcept
{
    const double k1 = kd_(0);
    const double k2 = kd_(1);
    const double p1 = kd_(2);
    const double p2 = kd_(3);
    const double k3 = kd_(4);
    const double k4 = kd_(5);
    const double k5 = kd_(6);
    const double k6 = kd_(7);

    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
    const double dnum = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;   // d num / d r2
    const double dden = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;   // d den / d r2

    const double radial = num / den;
    const double g = (dnum * den - num * dden) / (den * den); // d radial / d r2

    Distortion d;
    d.xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    d.yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    // Hand-derived: d radial/dx = 2 x g, d radial/dy = 2 y g, d r2/dx = 2x, d r2/dy = 2y.
    d.dxddx = radial + 2.0 * g * x * x + 2.0 * p1 * y + 6.0 * p2 * x;
    d.dxddy = 2.0 * g * x * y + 2.0 * p1 * x + 2.0 * p2 * y;
    d.dyddx = 2.0 * g * x * y + 2.0 * p1 * x + 2.0 * p2 * y;
    d.dyddy = radial + 2.0 * g * y * y + 6.0 * p1 * y + 2.0 * p2 * x;
    return d;
}

double Camera::distortionJacobianDeterminant(double x, double y) const noexcept
{
    const Distortion d = evaluateDistortion(x, y);
    return d.dxddx * d.dyddy - d.dxddy * d.dyddx;
}

Eigen::Vector2d Camera::vectorToPixel(const Eigen::Vector3d & rPCc) const noexcept
{
    if (!(rPCc.z() > 0.0))
    {
        return Eigen::Vector2d::Constant(kNaN);
    }
    const Distortion d = evaluateDistortion(rPCc.x() / rPCc.z(), rPCc.y() / rPCc.z());
    return Eigen::Vector2d(fx() * d.xd + cx(), fy() * d.yd + cy());
}

Eigen::Matrix<double, 2, 3> Camera::vectorToPixelJacobian(const Eigen::Vector3d & rPCc) const noexcept
{
    return vectorToPixelWithJacobian(rPCc).second;
}

std::pair<Eigen::Vector2d, Eigen::Matrix<double, 2, 3>> Camera::vectorToPixelWithJacobian(const Eigen::Vector3d & rPCc) const noexcept
{
    if (!(rPCc.z() > 0.0))
    {
        return { Eigen::Vector2d::Constant(kNaN), Eigen::Matrix<double, 2, 3>::Constant(kNaN) };
    }

    const double X = rPCc.x();
    const double Y = rPCc.y();
    const double Z = rPCc.z();
    const double x = X / Z;
    const double y = Y / Z;
    const Distortion d = evaluateDistortion(x, y);

    Eigen::Vector2d pixel(fx() * d.xd + cx(), fy() * d.yd + cy());

    // J = Jpix * Jdist(x,y) * d(x,y)/d(X,Y,Z), with
    //   Jpix = [fx s; 0 fy],   d(x,y)/d(X,Y,Z) = [1/Z 0 -X/Z^2; 0 1/Z -Y/Z^2].
    Eigen::Matrix2d Jdist;
    Jdist << d.dxddx, d.dxddy,
             d.dyddx, d.dyddy;
    Eigen::Matrix2d Jpix;
    Jpix << fx(), 0.0,
            0.0, fy();
    Eigen::Matrix<double, 2, 3> Jxyz;
    Jxyz << 1.0 / Z, 0.0, -X / (Z * Z),
            0.0, 1.0 / Z, -Y / (Z * Z);

    Eigen::Matrix<double, 2, 3> J = Jpix * Jdist * Jxyz;
    return { pixel, J };
}

Eigen::Vector2d Camera::pixelToNormalised(const Eigen::Vector2d & pixel) const noexcept
{
    // Target distorted normalised coordinates (xd, yd) from the pixel.
    const double ydTarget = (pixel.y() - cy()) / fy();
    const double xdTarget = (pixel.x() - cx()) / fx();

    // Newton iteration on F(x, y) = distort(x, y) - (xdTarget, ydTarget) = 0, starting at the target.
    double x = xdTarget;
    double y = ydTarget;
    for (int it = 0; it < kMaxUndistortIterations; ++it)
    {
        const Distortion d = evaluateDistortion(x, y);
        const double rx = d.xd - xdTarget;
        const double ry = d.yd - ydTarget;
        const double det = d.dxddx * d.dyddy - d.dxddy * d.dyddx;
        if (!std::isfinite(det) || std::abs(det) < 1e-14)
        {
            return Eigen::Vector2d::Constant(kNaN);
        }
        // Solve J [dx; dy] = [rx; ry] with the 2x2 inverse.
        const double dx = (d.dyddy * rx - d.dxddy * ry) / det;
        const double dy = (-d.dyddx * rx + d.dxddx * ry) / det;
        x -= dx;
        y -= dy;
        if (!std::isfinite(x) || !std::isfinite(y))
        {
            return Eigen::Vector2d::Constant(kNaN);
        }
        if (std::abs(dx) + std::abs(dy) < kUndistortTolerance)
        {
            return Eigen::Vector2d(x, y);
        }
    }
    return Eigen::Vector2d::Constant(kNaN);
}

Eigen::Vector3d Camera::pixelToVector(const Eigen::Vector2d & pixel) const noexcept
{
    const Eigen::Vector2d n = pixelToNormalised(pixel);
    if (!n.allFinite())
    {
        return Eigen::Vector3d::Constant(kNaN);
    }
    return Eigen::Vector3d(n.x(), n.y(), 1.0).normalized();
}

bool Camera::isWithinImage(const Eigen::Vector2d & pixel, double margin) const noexcept
{
    if (!pixel.allFinite())
    {
        return false;
    }
    const double lo = -0.5 + margin;
    const double uHi = static_cast<double>(imageWidth_) - 0.5 - margin;
    const double vHi = static_cast<double>(imageHeight_) - 0.5 - margin;
    return pixel.x() >= lo && pixel.x() < uHi && pixel.y() >= lo && pixel.y() < vHi;
}

bool Camera::isVectorWithinFOV(const Eigen::Vector3d & rPCc, double margin) const noexcept
{
    if (!(rPCc.z() > 0.0) || !rPCc.allFinite())
    {
        return false;
    }
    const double x = rPCc.x() / rPCc.z();
    const double y = rPCc.y() / rPCc.z();
    const Distortion d = evaluateDistortion(x, y);
    const double det = d.dxddx * d.dyddy - d.dxddy * d.dyddx;
    if (!(det > 0.0))
    {
        return false; // distortion folds over here: the point is not a valid image observation
    }
    const Eigen::Vector2d pixel(fx() * d.xd + cx(), fy() * d.yd + cy());
    return isWithinImage(pixel, margin);
}
