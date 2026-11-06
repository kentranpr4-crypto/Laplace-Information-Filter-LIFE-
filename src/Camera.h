#ifndef CAMERA_H
#define CAMERA_H

#include <cstddef>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

/**
 * @brief Pinhole camera with OpenCV-compatible lens distortion and fixed body-to-camera extrinsics.
 *
 * Coordinate conventions (DESIGN_ASSUMPTION, documented in architecture):
 *  - Camera frame c: x right, y down, z along the optical axis (OpenCV convention).
 *    rPCc is the position of a 3D point P relative to the camera origin C, expressed in frame c.
 *  - Body frame b: x forward, y right, z down. Rbc maps camera-frame vectors into body
 *    coordinates, i.e. rPBb = rCBb + Rbc * rPCc. Default Rbc = [[0,0,1],[1,0,0],[0,1,0]]
 *    (columns are the camera axes expressed in b: c_x = b_y, c_y = b_z, c_z = b_x), rCBb = 0.
 *  - Pixel frame: OpenCV convention. (u, v) = (column, row) with the centre of the top-left
 *    pixel at (0, 0). A pixel lies inside the image if -0.5 <= u < W - 0.5 and -0.5 <= v < H - 0.5.
 *
 * Projection model (identical to cv::projectPoints for the same coefficients):
 * @code
 *   x = X/Z,  y = Y/Z,  r2 = x^2 + y^2
 *   radial = (1 + k1 r2 + k2 r2^2 + k3 r2^3) / (1 + k4 r2 + k5 r2^2 + k6 r2^3)
 *   xd = x radial + 2 p1 x y + p2 (r2 + 2 x^2)
 *   yd = y radial + p1 (r2 + 2 y^2) + 2 p2 x y
 *   u  = fx xd + cx,          v = fy yd + cy
 * @endcode
 * The intrinsic matrix must have zero skew. cv::projectPoints (OpenCV 5) ignores the skew term, and
 * calibrateCamera never estimates it, so the model omits it to match the reference implementation exactly.
 * Distortion coefficients are stored in OpenCV order (k1, k2, p1, p2[, k3[, k4, k5, k6]]).
 * Accepted lengths: 0 (no distortion), 4, 5 (standard OpenCV model) and 8 (rational model).
 * Missing higher-order coefficients are zero, so the standard and rational models share one code path.
 *
 * Jacobians are derived by hand (no autodiff) and validated against central finite differences in
 * test/src/CameraJacobian.cpp.
 *
 * camera.xml schema (written/read with cv::FileStorage, version 1):
 * @code
 *   format_version              int, must be 1
 *   image_width, image_height   int, > 0
 *   camera_matrix               opencv-matrix 3x3 double  [fx 0 cx; 0 fy cy; 0 0 1] (zero skew required)
 *   distortion_coefficients     opencv-matrix 1xN double, N in {0,4,5,8}
 *   body_to_camera_rotation     optional opencv-matrix 3x3 Rbc (default: see above)
 *   body_to_camera_position     optional opencv-matrix 3x1 rCBb (default: zero)
 *   rms_reprojection_error_px   optional double (calibration metadata)
 *   num_views, num_points       optional int (calibration metadata)
 *   board_inner_corners_cols/rows, square_size_m, source_video, distortion_model   optional
 * @endcode
 */
class Camera
{
public:
    /** @brief Metadata recorded by the calibration procedure (informational, not used by the model). */
    struct CalibrationMetadata
    {
        double rmsReprojectionErrorPx = std::numeric_limits<double>::quiet_NaN(); ///< RMS reprojection error over all corners [px]
        int numViews = 0;                ///< Number of chessboard views used in the final fit
        int numPoints = 0;               ///< Number of 2D-3D correspondences used in the final fit
        int boardInnerCornersCols = 0;   ///< Chessboard inner corners per row (grid_width)
        int boardInnerCornersRows = 0;   ///< Chessboard inner corners per column (grid_height)
        double squareSizeM = 0.0;        ///< Chessboard square side length [m]
        std::string sourceVideo;         ///< File name of the calibration video
        std::string distortionModel;     ///< Informational label, e.g. "opencv-standard-5" or "opencv-rational-8"
    };

    /** @brief Number of distortion coefficients accepted by the model (padded internally to 8). */
    static constexpr int kMaxDistortionCoefficients = 8;

    /** @brief Constructs an empty (invalid) camera. Use load() or the parameterised constructor. */
    Camera() = default;

    /**
     * @brief Constructs a camera from intrinsics and distortion, using the default extrinsics.
     * @param imageWidth  Image width [px], > 0.
     * @param imageHeight Image height [px], > 0.
     * @param cameraMatrix 3x3 intrinsic matrix [fx s cx; 0 fy cy; 0 0 1] with fx, fy > 0.
     * @param distortionCoefficients OpenCV-ordered coefficients, size 0, 4, 5 or 8.
     * @throws std::invalid_argument if any parameter is invalid.
     */
    Camera(int imageWidth, int imageHeight, const Eigen::Matrix3d & cameraMatrix, const std::vector<double> & distortionCoefficients);

    /**
     * @brief Loads and validates camera.xml.
     * @throws std::runtime_error with a descriptive message if the file is missing, unreadable or invalid.
     */
    static Camera load(const std::filesystem::path & cameraXmlPath);

    /**
     * @brief Writes camera.xml (schema above). Overwrites the file.
     * @throws std::runtime_error if the camera is invalid or the file cannot be written.
     */
    void save(const std::filesystem::path & cameraXmlPath) const;

    /** @brief True if the camera has been constructed/loaded with valid parameters. */
    [[nodiscard]] bool isValid() const noexcept { return imageWidth_ > 0 && imageHeight_ > 0; }

    [[nodiscard]] int imageWidth() const noexcept { return imageWidth_; }
    [[nodiscard]] int imageHeight() const noexcept { return imageHeight_; }
    [[nodiscard]] const Eigen::Matrix3d & cameraMatrix() const noexcept { return K_; }
    [[nodiscard]] double fx() const noexcept { return K_(0, 0); }
    [[nodiscard]] double fy() const noexcept { return K_(1, 1); }
    [[nodiscard]] double cx() const noexcept { return K_(0, 2); }
    [[nodiscard]] double cy() const noexcept { return K_(1, 2); }
    /** @brief Skew K(0,1); always zero for a valid camera (see class documentation). */
    [[nodiscard]] double skew() const noexcept { return K_(0, 1); }

    /** @brief Distortion coefficients in OpenCV order, exactly as provided (size 0, 4, 5 or 8). */
    [[nodiscard]] const std::vector<double> & distortionCoefficients() const noexcept { return distortion_; }

    /** @brief Body-to-camera rotation Rbc (rPBb = rCBb + Rbc rPCc). */
    [[nodiscard]] const Eigen::Matrix3d & Rbc() const noexcept { return Rbc_; }
    /** @brief Position of the camera origin C relative to the body origin B, expressed in body coordinates. */
    [[nodiscard]] const Eigen::Vector3d & rCBb() const noexcept { return rCBb_; }
    /** @brief Sets the extrinsics. @throws std::invalid_argument if Rbc is not a proper rotation or rCBb is not finite. */
    void setBodyToCameraExtrinsics(const Eigen::Matrix3d & Rbc, const Eigen::Vector3d & rCBb);

    [[nodiscard]] const CalibrationMetadata & metadata() const noexcept { return metadata_; }
    void setMetadata(const CalibrationMetadata & metadata) { metadata_ = metadata; }

    /**
     * @brief Projects a camera-frame vector to pixel coordinates.
     * @param rPCc Point relative to the camera origin in camera coordinates.
     * @return Pixel (u, v). Returns NaN components if rPCc.z() <= 0 (behind the camera).
     */
    [[nodiscard]] Eigen::Vector2d vectorToPixel(const Eigen::Vector3d & rPCc) const noexcept;

    /**
     * @brief Projects and returns the analytic Jacobian d(u,v)/d(rPCc).
     * @return (pixel, J) with J a 2x3 matrix. Both contain NaN if rPCc.z() <= 0.
     */
    [[nodiscard]] std::pair<Eigen::Vector2d, Eigen::Matrix<double, 2, 3>> vectorToPixelWithJacobian(const Eigen::Vector3d & rPCc) const noexcept;

    /** @brief Jacobian d(u,v)/d(rPCc) only (2x3, NaN if rPCc.z() <= 0). */
    [[nodiscard]] Eigen::Matrix<double, 2, 3> vectorToPixelJacobian(const Eigen::Vector3d & rPCc) const noexcept;

    /**
     * @brief Back-projects a pixel to a unit direction vector in camera coordinates (undistorted ray).
     * @return Unit vector with z > 0, or NaN components if the undistortion did not converge
     *         (e.g. the pixel lies outside the region where the distortion map is invertible).
     */
    [[nodiscard]] Eigen::Vector3d pixelToVector(const Eigen::Vector2d & pixel) const noexcept;

    /** @brief Undistorted normalised image coordinates (x, y) = (X/Z, Y/Z) of a pixel, or NaN on failure. */
    [[nodiscard]] Eigen::Vector2d pixelToNormalised(const Eigen::Vector2d & pixel) const noexcept;

    /**
     * @brief True if the pixel lies inside the image, shrunk inwards by margin [px].
     * Pixel coordinates use the OpenCV convention (see class documentation).
     */
    [[nodiscard]] bool isWithinImage(const Eigen::Vector2d & pixel, double margin = 0.0) const noexcept;

    /**
     * @brief True if the camera-frame point is visible: in front of the camera (z > 0), inside the
     * image (shrunk by margin), and in the region where the distortion map is locally invertible
     * (det of the distortion Jacobian > 0, i.e. no fold-over).
     */
    [[nodiscard]] bool isVectorWithinFOV(const Eigen::Vector3d & rPCc, double margin = 0.0) const noexcept;

    /** @brief Determinant of d(xd,yd)/d(x,y) at normalised coordinates (x, y); > 0 means locally invertible. */
    [[nodiscard]] double distortionJacobianDeterminant(double x, double y) const noexcept;

private:
    /** @brief Distorted normalised coordinates and their Jacobian w.r.t. (x, y). */
    struct Distortion
    {
        double xd, yd;
        double dxddx, dxddy, dyddx, dyddy;
    };

    int imageWidth_ = 0;
    int imageHeight_ = 0;
    Eigen::Matrix3d K_ = Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
    std::vector<double> distortion_;
    Eigen::Matrix<double, kMaxDistortionCoefficients, 1> kd_ = Eigen::Matrix<double, kMaxDistortionCoefficients, 1>::Zero();
    Eigen::Matrix3d Rbc_ = Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector3d rCBb_ = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    CalibrationMetadata metadata_;

    [[nodiscard]] Distortion evaluateDistortion(double x, double y) const noexcept;
    void setDistortion(const std::vector<double> & coefficients);
};

#endif
