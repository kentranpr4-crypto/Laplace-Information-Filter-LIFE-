/**
 * @file Plot.cpp
 * @brief VTK 9.5 dual-pane visualisation (see Plot.h for the layout).
 *
 * ===================== Documentation of the visual design (ASSUMED) =====================
 * No visual specification was supplied with the assignment for the 3-D pane, so the choices below are
 * engineering assumptions. They are recorded here, in engineering-reports/architecture.md and printed
 * in the on-screen legend.
 *
 * Panes (window 2W x H, export frame identical):
 *   LEFT  viewport [0, 0.5] x [0, 1] : scenario overlay, one image pixel = one window pixel
 *                                      (nearest-neighbour sampling, no scaling, no vertical flip).
 *   RIGHT viewport [0.5, 1] x [0, 1] : 3-D scene (perspective camera, trackball in interactive modes).
 *
 * Right-pane coordinate system: estimator frame n (z down) is displayed through the fixed proper rotation
 *   M = [[1,0,0],[0,0,-1],[0,1,0]]  ->  VTK +X = north, VTK +Y = up, VTK +Z = east.
 * World axes are drawn with the same M: N (red), E (green), D (blue, pointing down); named in the legend.
 *
 * Render order (AddActor order; all 3-D geometry is opaque and depth tested, 2-D text is drawn last by VTK):
 *   1. world axes N/E/D (length 0.35 x scene scale)
 *   2. validation-only track (magenta, 2 px, validation only - never used by an estimator)
 *   3. estimated trajectory (yellow, 3 px)
 *   4. camera frustum (white, 2 px) and camera marker (white sphere, radius 0.012 x scale)
 *   5. camera 3-sigma position ellipsoid (cyan wireframe)
 *   6. landmarks, as three batches (one actor each, see "Landmark batching" below):
 *        6a. 3-sigma ellipsoid of every landmark (own colour, wireframe)
 *        6b. marker sphere of every point landmark (own colour, lit)
 *        6c. square of every pose landmark (own colour, white edges)
 *   7. text overlays: status lines (top-left) and the colour legend (bottom-left)
 *
 * Colours (RGB):
 *   background #1A1C23 (0.10, 0.11, 0.14); trajectory (1.00, 0.80, 0.10); validation (1.00, 0.25, 0.85);
 *   camera frustum (0.95, 0.95, 0.95); camera ellipsoid (0.20, 0.90, 1.00);
 *   landmarks: colourRGB from the scenario. Visible landmarks use full colour and a 1.5 x marker radius;
 *   landmarks that are not visible in the current frame are dimmed to 35 % brightness, with a 1.0 x marker.
 *   Pose-landmark square edges: white when visible, grey (0.5) when not visible.
 *
 * Uncertainty: all confidence regions use nSigma = 3.0 with c = chi2inv(P(3-sigma), 3). The ellipsoid is
 * drawn as the image of the unit sphere under A = sqrt(c) S^T (S upper triangular, P = S^T S), i.e.
 * surface = mu + A u, |u| = 1. Wireframe, not translucent: translucency would need depth peeling, which is
 * too slow for dozens of landmarks at 5376 x 1512.
 *
 * Camera behaviour (3-D pane, auto "overview" view):
 *   - Content = trajectory, validation track (if given), camera, landmark means +/- their 3-sigma radius.
 *   - Bounding box of the content (display coordinates) gives centre c and radius r (half diagonal).
 *   - Focal point and radius are smoothed with an exponential filter (alpha = 0.10 per frame); the radius
 *     grows immediately and shrinks slowly, so content is never cut off while the scene is growing.
 *   - View direction is fixed (oblique, from above), view angle 30 deg, distance r * 1.15 / sin(15 deg).
 *   - Frustum depth and marker sizes scale with the smoothed radius (floor 0.25 m) so they stay readable.
 *   - Interactive modes: the first trackball interaction turns the auto view off so the user keeps control;
 *     pressing 'r' turns it back on.
 *
 * Interaction (interactive modes only):
 *   trackball camera on the 3-D pane (the image pane is not interactive).
 *   n / space / c : continue to the next frame (mode 2) ; q / Escape : quit ; r : reset overview camera.
 *   Closing the window also quits.
 *
 * Backend (measured on the reference machine, DISPLAY=:1, RTX 3050 Laptop + Intel iGPU):
 *   Offscreen (interactive = 0): VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow is set only for the duration of
 *   the window construction, so the headless EGL window is used even when DISPLAY is set. Reasons: (a) the X11
 *   window of a 3840 x 1080 (or 5376 x 1512) frame does not fit the 1920 x 1080 X screen of the reference machine;
 *   (b) the X11 path is served by Mesa/Intel (integrated GPU) and its readback time varied between runs, from
 *   about 15 ms to about 0.9 s per 3840 x 1080 frame; (c) the EGL path renders on the NVIDIA GPU and its readback
 *   was about 4 to 8 ms in every measurement. The cause of the slow X11 episodes was not isolated.
 *   Onscreen (interactive = 1, 2): the default VTK window (X11 on DISPLAY).
 *
 * Landmark batching: landmarks are drawn from three persistent actors whose vtkPoints, colours and cell arrays
 * are updated in place. A landmark owns one slot in each batch while it is reported; slots of landmarks that
 * disappear are reused, so the batch size is the peak number of landmarks alive at once (not the number seen over
 * the run). Cells are rebuilt only when the set of drawn landmarks changes. Batching changes the draw order between landmarks from "per landmark" to
 * "all ellipsoids, then all markers, then all squares". All landmark primitives are opaque and depth tested, so
 * this order only matters for exactly coincident depths. Pixel comparison against the previous per-actor renderer
 * (same scene, same GPU, same tessellation) gives a mean absolute difference of about 0.001 grey levels over the
 * 3-D pane, i.e. unchanged apart from rasterisation noise.
 * Marker normals are the unit-sphere normals transformed by M (valid for the uniform scaling used for markers).
 * Vertex colours are 8-bit RGB (quantisation 1/255).
 * Slots that are not drawn in a frame are parked at the scene centre, so they never widen the polydata bounds
 * used by the overview camera.
 *
 * Capture: vtkRenderWindow::GetRGBACharPixelData on the full window (2W x H), RGBA -> BGR and vertical flip
 * (GL rows are bottom-up) in one pass. Offscreen: back buffer; onscreen: front buffer (after the buffer swap).
 * Multisampling is off (SetMultiSamples(0)) for speed; anti-aliasing is therefore not applied.
 * ========================================================================================
 */

#include "Plot.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkCommand.h>
#include <vtkFloatArray.h>
#include <vtkIdList.h>
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkImageProperty.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkUnsignedCharArray.h>

#include <opencv2/imgproc.hpp>

#include "Camera.h"
#include "GaussianBase.h"

namespace
{

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

/// Fixed proper rotation n -> display (see file documentation).
Mat3 displayRotation()
{
    Mat3 m;
    m << 1.0, 0.0, 0.0,
         0.0, 0.0, -1.0,
         0.0, 1.0, 0.0;
    return m;
}

constexpr std::array<double, 3> kBackgroundRGB{0.10, 0.11, 0.14};
constexpr std::array<double, 3> kTrajectoryRGB{1.00, 0.80, 0.10};
constexpr std::array<double, 3> kValidationRGB{1.00, 0.25, 0.85};
constexpr std::array<double, 3> kCameraRGB{0.95, 0.95, 0.95};
constexpr std::array<double, 3> kCameraEllipsoidRGB{0.20, 0.90, 1.00};
constexpr double kDimFactor = 0.35;
constexpr double kVisibleMarkerScale = 1.5;
constexpr double kHiddenMarkerScale = 1.0;
constexpr double kMarkerFraction = 0.012;          ///< marker radius / scene scale
constexpr double kFrustumDepthFraction = 0.10;     ///< frustum depth / scene scale
constexpr double kMinSceneScale = 0.25;            ///< [m] floor for scene scale
constexpr double kViewAngleDeg = 30.0;
constexpr double kViewSmoothing = 0.10;
constexpr double kViewMargin = 1.15;
constexpr int kSphereThetaResolution = 14;
constexpr int kSpherePhiResolution = 10;
constexpr double kEllipsoidLineWidth = 1.5;
constexpr std::array<double, 3> kViewDirection{0.55, 0.45, 0.70}; ///< from focus towards the camera (display coords)
constexpr const char * kOffscreenWindowClass = "vtkEGLRenderWindow";
constexpr const char * kWindowClassEnvironment = "VTK_DEFAULT_OPENGL_WINDOW";

Vec3 toVec3(const std::array<double, 3> & a)
{
    return Vec3(a[0], a[1], a[2]);
}

Vec3 clampColour(const Vec3 & c)
{
    return c.cwiseMax(0.0).cwiseMin(1.0);
}

bool isFinite(const Vec3 & v)
{
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

/// Colour in [0, 1] to 8-bit RGB (rounded).
std::array<unsigned char, 3> toBytes(const Vec3 & rgb)
{
    const Vec3 c = clampColour(rgb);
    std::array<unsigned char, 3> out{};
    for (int i = 0; i < 3; ++i)
        out[static_cast<std::size_t>(i)] = static_cast<unsigned char>(std::lround(255.0 * c(i)));
    return out;
}

/// Writes a 4x4 affine matrix [A t; 0 1] into a VTK matrix.
void setAffine(vtkMatrix4x4 * m, const Mat3 & A, const Vec3 & t)
{
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
            m->SetElement(r, c, A(r, c));
        m->SetElement(r, 3, t(r));
    }
    m->SetElement(3, 0, 0.0);
    m->SetElement(3, 1, 0.0);
    m->SetElement(3, 2, 0.0);
    m->SetElement(3, 3, 1.0);
}

void setColour(vtkActor * actor, const Vec3 & rgb)
{
    actor->GetProperty()->SetColor(rgb.x(), rgb.y(), rgb.z());
}

/// Sets an environment variable (or unsets it when value is null) and restores the previous state on destruction.
/// VTK reads VTK_DEFAULT_OPENGL_WINDOW when it creates a render window, so the override is scoped to that call.
class ScopedEnvironmentVariable
{
public:
    ScopedEnvironmentVariable(const char * name, const char * value)
        : name_(name)
    {
        if (const char * old = std::getenv(name))
        {
            hadOld_ = true;
            old_ = old;
        }
        set(value);
    }
    ~ScopedEnvironmentVariable()
    {
        set(hadOld_ ? old_.c_str() : nullptr);
    }
    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable & operator=(const ScopedEnvironmentVariable &) = delete;

private:
    void set(const char * value) const
    {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value)
            setenv(name_.c_str(), value, 1);
        else
            unsetenv(name_.c_str());
#endif
    }

    std::string name_;
    std::string old_;
    bool hadOld_ = false;
};

/// Creates the render window. Offscreen windows are requested from VTK's EGL class (headless, GPU, DISPLAY ignored);
/// onscreen windows use VTK's default class (X11 on DISPLAY) and leave the environment untouched.
vtkSmartPointer<vtkRenderWindow> createRenderWindow(Plot::WindowMode mode)
{
    const std::string requested = Plot::requestedBackendClass(mode);
    if (requested.empty())
        return vtkSmartPointer<vtkRenderWindow>::New();
    ScopedEnvironmentVariable scope(kWindowClassEnvironment, requested.c_str());
    return vtkSmartPointer<vtkRenderWindow>::New();
}

/// Unit-sphere triangulation (from vtkSphereSource, so that the tessellation is the one VTK draws everywhere else).
struct SphereTemplate
{
    std::vector<Vec3> points;                              ///< unit sphere, centred at the origin
    std::vector<std::array<vtkIdType, 3>> triangles;       ///< indices into points
};

SphereTemplate buildSphereTemplate(vtkPolyData * sphere)
{
    SphereTemplate tpl;
    const vtkIdType n = sphere->GetNumberOfPoints();
    tpl.points.resize(static_cast<std::size_t>(n));
    for (vtkIdType i = 0; i < n; ++i)
    {
        double q[3];
        sphere->GetPoint(i, q);
        tpl.points[static_cast<std::size_t>(i)] = Vec3(q[0], q[1], q[2]);
    }
    vtkCellArray * polys = sphere->GetPolys();
    if (polys == nullptr)
        throw std::logic_error("Plot: sphere template has no polygons");
    polys->InitTraversal();
    vtkIdType npts = 0;
    const vtkIdType * pts = nullptr;
    while (polys->GetNextCell(npts, pts))
    {
        // fan triangulation (vtkSphereSource emits triangles, the fan is a safe generalisation)
        for (vtkIdType k = 1; k + 1 < npts; ++k)
            tpl.triangles.push_back({pts[0], pts[k], pts[k + 1]});
    }
    if (tpl.triangles.empty())
        throw std::logic_error("Plot: sphere template has no triangles");
    return tpl;
}

/// Corners of the unit square used for pose landmarks (before scaling by the tag size and the rotation).
constexpr std::array<std::array<double, 3>, 4> kSquareCorners{{{-0.5, -0.5, 0.0}, {0.5, -0.5, 0.0},
                                                             {0.5, 0.5, 0.0}, {-0.5, 0.5, 0.0}}};

} // namespace

// ----------------------------------------------------------------------------------------------------
// Internal state
// ----------------------------------------------------------------------------------------------------

struct Plot::Impl
{
    /// Polyline (trajectory / validation track). Rebuilt only when its content changes.
    struct Polyline
    {
        vtkSmartPointer<vtkPoints> points;
        vtkSmartPointer<vtkCellArray> lines;
        vtkSmartPointer<vtkPolyData> poly;
        vtkSmartPointer<vtkPolyDataMapper> mapper;
        vtkSmartPointer<vtkActor> actor;
        std::vector<Vec3> finiteWorld;   ///< reused buffer (world coordinates, finite points only)
        std::vector<vtkIdType> ids;      ///< reused buffer
        Vec3 last = Vec3::Zero();
    };

    /// Kind of cells a batch draws for every active landmark slot.
    enum class BatchKind
    {
        Triangles,   ///< the sphere template (ellipsoids and markers)
        Squares      ///< two triangles (face) and one closed polyline (edges) per pose landmark
    };

    /**
     * @brief One actor drawing one small mesh per landmark slot.
     *
     * Every landmark owns slot k: points [k * pointsPerSlot, (k + 1) * pointsPerSlot). Points, colours and normals
     * are written in place every frame for the active slots. The cell arrays reference only the active slots and
     * are rebuilt only when the set of active slots changes.
     */
    struct Batch
    {
        BatchKind kind = BatchKind::Triangles;
        int pointsPerSlot = 0;
        vtkSmartPointer<vtkPoints> points;
        vtkSmartPointer<vtkUnsignedCharArray> colours;
        vtkSmartPointer<vtkFloatArray> normals;      ///< lit batches only
        vtkSmartPointer<vtkCellArray> polys;
        vtkSmartPointer<vtkCellArray> lines;
        vtkSmartPointer<vtkPolyData> poly;
        vtkSmartPointer<vtkPolyDataMapper> mapper;
        vtkSmartPointer<vtkActor> actor;
        std::vector<unsigned char> active;   ///< slot activity in this frame
        std::vector<unsigned char> built;    ///< slot activity the cell arrays were last built for
    };

    // ----- configuration -----
    Camera camera;
    cv::Size inputSize;
    bool onscreen = false;
    double sqrtChi2 = 1.0;                       ///< sqrt(c), c = mahalanobisThresholdSq(3, 3)
    std::array<Vec2, 4> cornersNormalised{};     ///< undistorted normalised image corners (z = 1)

    // ----- timings -----
    Plot::Timings timings;

    // ----- backend -----
    std::string backendClass;
    std::string backendDescription;

    // ----- VTK pipeline -----
    vtkSmartPointer<vtkRenderWindow> window;
    vtkSmartPointer<vtkRenderer> imageRenderer;
    vtkSmartPointer<vtkRenderer> sceneRenderer;
    vtkSmartPointer<vtkImageData> imageData;
    vtkSmartPointer<vtkImageActor> imageActor;
    vtkSmartPointer<vtkUnsignedCharArray> rgbaBuffer;

    // shared unit primitives
    vtkSmartPointer<vtkSphereSource> unitSphere;
    SphereTemplate sphere;
    vtkSmartPointer<vtkPolyData> unitQuad;

    // scene actors
    vtkSmartPointer<vtkAxesActor> axes;
    vtkSmartPointer<vtkMatrix4x4> axesM;
    Polyline trajectory;
    Polyline validation;
    vtkSmartPointer<vtkPoints> frustumPoints;
    vtkSmartPointer<vtkPolyData> frustumPoly;
    vtkSmartPointer<vtkActor> frustumActor;
    vtkSmartPointer<vtkActor> cameraMarker;
    vtkSmartPointer<vtkMatrix4x4> cameraMarkerM;
    vtkSmartPointer<vtkActor> cameraEllipsoid;
    vtkSmartPointer<vtkMatrix4x4> cameraEllipsoidM;

    // landmark batches (drawn in this order, after the camera actors)
    Batch ellipsoids;
    Batch markers;
    Batch squares;
    std::unordered_map<long long, std::size_t> slotIndex;   ///< key(id, isPose) -> slot, landmarks of this frame
    std::vector<std::size_t> freeSlots;                      ///< slots released by landmarks that disappeared
    std::size_t slotCount = 0;                               ///< slots allocated (peak number of landmarks alive)

    vtkSmartPointer<vtkTextActor> statusText;
    vtkSmartPointer<vtkTextActor> legendText;

    // ----- camera / view state -----
    bool autoView = true;
    bool viewInitialised = false;
    Vec3 focus = Vec3::Zero();
    double sceneRadius = kMinSceneScale;  ///< smoothed scene radius [display units]
    bool dirty = true;

    // ----- interaction -----
    vtkSmartPointer<vtkRenderWindowInteractor> interactor;
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style;
    vtkSmartPointer<vtkCallbackCommand> keyCommand;
    vtkSmartPointer<vtkCallbackCommand> exitCommand;
    vtkSmartPointer<vtkCallbackCommand> interactionCommand;
    bool interactorReady = false;
    bool continueRequested = false;
    bool quitRequested = false;

    // ----- scratch -----
    cv::Mat overlayBGR;    ///< grey overlays converted to BGR

    /// Creates the polyline actor with the given colour / width.
    void createPolyline(Polyline & pl, const std::array<double, 3> & rgb, double width)
    {
        pl.points = vtkSmartPointer<vtkPoints>::New();
        pl.lines = vtkSmartPointer<vtkCellArray>::New();
        pl.poly = vtkSmartPointer<vtkPolyData>::New();
        pl.poly->SetPoints(pl.points);
        pl.poly->SetLines(pl.lines);
        pl.mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        pl.mapper->SetInputData(pl.poly);
        pl.actor = vtkSmartPointer<vtkActor>::New();
        pl.actor->SetMapper(pl.mapper);
        pl.actor->GetProperty()->SetLighting(false);
        pl.actor->GetProperty()->SetLineWidth(static_cast<float>(width));   // VTK takes float; widths are a few px
        pl.actor->GetProperty()->SetColor(rgb[0], rgb[1], rgb[2]);
        sceneRenderer->AddActor(pl.actor);
    }

    /// Sets the polyline from world points (estimator frame). Skips non-finite points and unchanged content.
    void setPolyline(Polyline & pl, const std::vector<Vec3> & world, const Mat3 & M)
    {
        pl.finiteWorld.clear();
        for (const Vec3 & p : world)
        {
            if (isFinite(p))
                pl.finiteWorld.push_back(p);
        }
        const std::size_t n = pl.finiteWorld.size();
        if (static_cast<vtkIdType>(n) == pl.points->GetNumberOfPoints()
            && (n == 0 || pl.last == pl.finiteWorld.back()))
        {
            return;   // content unchanged (trajectories only grow, so this is the common path)
        }

        pl.points->SetNumberOfPoints(static_cast<vtkIdType>(n));
        for (std::size_t i = 0; i < n; ++i)
        {
            const Vec3 q = M * pl.finiteWorld[i];
            pl.points->SetPoint(static_cast<vtkIdType>(i), q.x(), q.y(), q.z());
        }
        pl.lines->Reset();
        if (n >= 2)
        {
            pl.ids.resize(n);
            std::iota(pl.ids.begin(), pl.ids.end(), vtkIdType{0});
            pl.lines->InsertNextCell(static_cast<vtkIdType>(n), pl.ids.data());
        }
        pl.last = n > 0 ? pl.finiteWorld.back() : Vec3::Zero();
        pl.points->Modified();
        pl.lines->Modified();
        pl.poly->Modified();
    }

    /// Creates an actor that shares the unit sphere output and owns a 4x4 user matrix (camera actors only).
    vtkSmartPointer<vtkActor> createSphereActor(vtkSmartPointer<vtkMatrix4x4> & matrix, bool wireframe)
    {
        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputConnection(unitSphere->GetOutputPort());
        auto actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        matrix = vtkSmartPointer<vtkMatrix4x4>::New();
        actor->SetUserMatrix(matrix);
        if (wireframe)
        {
            actor->GetProperty()->SetRepresentationToWireframe();
            actor->GetProperty()->SetLineWidth(kEllipsoidLineWidth);
            actor->GetProperty()->SetLighting(false);
        }
        sceneRenderer->AddActor(actor);
        return actor;
    }

    /// Creates the persistent actor of one landmark batch. All buffers are allocated here, once.
    void createBatch(Batch & b, BatchKind kind, int pointsPerSlot, bool lit, bool wireframe)
    {
        b.kind = kind;
        b.pointsPerSlot = pointsPerSlot;
        b.points = vtkSmartPointer<vtkPoints>::New();
        b.colours = vtkSmartPointer<vtkUnsignedCharArray>::New();
        b.colours->SetNumberOfComponents(3);
        b.polys = vtkSmartPointer<vtkCellArray>::New();
        b.lines = vtkSmartPointer<vtkCellArray>::New();
        b.poly = vtkSmartPointer<vtkPolyData>::New();
        b.poly->SetPoints(b.points);
        b.poly->SetPolys(b.polys);
        b.poly->SetLines(b.lines);
        b.poly->GetPointData()->SetScalars(b.colours);
        if (lit)
        {
            b.normals = vtkSmartPointer<vtkFloatArray>::New();
            b.normals->SetNumberOfComponents(3);
            b.poly->GetPointData()->SetNormals(b.normals);
        }
        b.mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        b.mapper->SetInputData(b.poly);
        b.mapper->SetColorModeToDirectScalars();
        b.mapper->ScalarVisibilityOn();
        if (kind == BatchKind::Squares)
        {
            // the face is pushed back so that the white edge lines drawn at the same depth stay visible
            b.mapper->SetResolveCoincidentTopologyToPolygonOffset();
            b.mapper->SetRelativeCoincidentTopologyPolygonOffsetParameters(1.0, 1.0);
        }
        b.actor = vtkSmartPointer<vtkActor>::New();
        b.actor->SetMapper(b.mapper);
        vtkProperty * property = b.actor->GetProperty();
        property->SetLighting(lit);
        property->SetBackfaceCulling(false);
        if (wireframe)
        {
            property->SetRepresentationToWireframe();
            property->SetLineWidth(kEllipsoidLineWidth);
        }
        sceneRenderer->AddActor(b.actor);
    }

    static long long landmarkKey(int id, bool isPose)
    {
        return 2LL * static_cast<long long>(id) + (isPose ? 1LL : 0LL);
    }

    /**
     * @brief Assigns the slots of this frame from the keys of the reported landmarks.
     *
     * A landmark keeps its slot while it is reported. Slots of landmarks that are no longer reported are released
     * and reused by new landmarks, so the batches are sized by the number of landmarks alive at the same time, not
     * by the number of landmarks ever seen. All growth happens here, before any point of this frame is written.
     */
    void assignSlots(const std::vector<long long> & keys)
    {
        const std::unordered_set<long long> present(keys.begin(), keys.end());
        for (auto it = slotIndex.begin(); it != slotIndex.end();)
        {
            if (present.count(it->first) == 0)
            {
                freeSlots.push_back(it->second);
                it = slotIndex.erase(it);
            }
            else
            {
                ++it;
            }
        }
        for (const long long key : keys)
        {
            if (slotIndex.count(key) != 0)
                continue;
            std::size_t slot = 0;
            if (!freeSlots.empty())
            {
                slot = freeSlots.back();
                freeSlots.pop_back();
            }
            else
            {
                slot = slotCount++;
                growBatches(slotCount);
            }
            slotIndex.emplace(key, slot);
        }
    }

    /// Resizes the buffers of all batches to `slots` landmark slots.
    void growBatches(std::size_t slots)
    {
        for (Batch * b : {&ellipsoids, &markers, &squares})
        {
            const auto n = static_cast<vtkIdType>(slots) * b->pointsPerSlot;
            b->points->SetNumberOfPoints(n);
            b->colours->SetNumberOfTuples(n);
            if (b->normals)
                b->normals->SetNumberOfTuples(n);
            b->active.resize(slots, 0);
        }
    }

    /// Writes the transformed unit sphere (surface = centre + A u) of one slot and marks the slot active.
    /// With withNormals the normals are the transformed unit normals (valid for uniform scaling, used for markers).
    void writeSphere(Batch & b, std::size_t slot, const Mat3 & A, const Vec3 & centre, const Vec3 & colour,
                     bool withNormals)
    {
        const vtkIdType base = static_cast<vtkIdType>(slot) * b.pointsPerSlot;
        const auto rgb = toBytes(colour);
        for (std::size_t i = 0; i < sphere.points.size(); ++i)
        {
            const Vec3 & u = sphere.points[i];
            const Vec3 q = A * u + centre;
            const auto id = base + static_cast<vtkIdType>(i);
            b.points->SetPoint(id, q.x(), q.y(), q.z());
            b.colours->SetTuple3(id, rgb[0], rgb[1], rgb[2]);
            if (withNormals)
            {
                const Vec3 n = A * u;
                const double len = n.norm();
                const Vec3 unit = len > 0.0 ? Vec3(n / len) : Vec3(0.0, 0.0, 1.0);
                b.normals->SetTuple3(id, unit.x(), unit.y(), unit.z());
            }
        }
        b.active[slot] = 1;
    }

    /// Writes one pose-landmark square of one slot: 4 face points (face colour) and 4 edge points (edge colour).
    /// The square is R * corner + centre with R = M Rnl size.
    void writeSquare(std::size_t slot, const Mat3 & R, const Vec3 & centre, const Vec3 & faceColour,
                     const Vec3 & edgeColour)
    {
        const vtkIdType base = static_cast<vtkIdType>(slot) * squares.pointsPerSlot;
        const auto face = toBytes(faceColour);
        const auto edge = toBytes(edgeColour);
        for (std::size_t k = 0; k < 4; ++k)
        {
            const Vec3 corner(kSquareCorners[k][0], kSquareCorners[k][1], kSquareCorners[k][2]);
            const Vec3 q = R * corner + centre;
            const auto f = base + static_cast<vtkIdType>(k);
            const auto e = base + static_cast<vtkIdType>(k + 4);
            squares.points->SetPoint(f, q.x(), q.y(), q.z());
            squares.colours->SetTuple3(f, face[0], face[1], face[2]);
            squares.points->SetPoint(e, q.x(), q.y(), q.z());
            squares.colours->SetTuple3(e, edge[0], edge[1], edge[2]);
        }
        squares.active[slot] = 1;
    }

    /// Moves the points of every slot that is not drawn in this frame to `point` (finite, inside the content).
    static void parkInactive(Batch & b, const Vec3 & point)
    {
        for (std::size_t slot = 0; slot < b.active.size(); ++slot)
        {
            if (b.active[slot])
                continue;
            const vtkIdType base = static_cast<vtkIdType>(slot) * b.pointsPerSlot;
            for (int i = 0; i < b.pointsPerSlot; ++i)
                b.points->SetPoint(base + i, point.x(), point.y(), point.z());
        }
    }

    /// Rebuilds the cell arrays of a batch when its activity pattern changed, then marks its data as modified.
    void commit(Batch & b)
    {
        if (b.active != b.built)
        {
            b.polys->Reset();
            b.lines->Reset();
            for (std::size_t slot = 0; slot < b.active.size(); ++slot)
            {
                if (!b.active[slot])
                    continue;
                const vtkIdType base = static_cast<vtkIdType>(slot) * b.pointsPerSlot;
                if (b.kind == BatchKind::Triangles)
                {
                    for (const auto & tri : sphere.triangles)
                    {
                        const std::array<vtkIdType, 3> ids{base + tri[0], base + tri[1], base + tri[2]};
                        b.polys->InsertNextCell(3, ids.data());
                    }
                }
                else
                {
                    const std::array<vtkIdType, 3> t0{base, base + 1, base + 2};
                    const std::array<vtkIdType, 3> t1{base, base + 2, base + 3};
                    b.polys->InsertNextCell(3, t0.data());
                    b.polys->InsertNextCell(3, t1.data());
                    const std::array<vtkIdType, 5> edge{base + 4, base + 5, base + 6, base + 7, base + 4};
                    b.lines->InsertNextCell(5, edge.data());
                }
            }
            b.built = b.active;
            b.polys->Modified();
            b.lines->Modified();
        }
        b.points->Modified();
        b.colours->Modified();
        if (b.normals)
            b.normals->Modified();
        b.poly->Modified();
    }
};

// ----------------------------------------------------------------------------------------------------
// Static helpers
// ----------------------------------------------------------------------------------------------------

Eigen::Matrix3d Plot::worldToDisplay()
{
    return displayRotation();
}

Eigen::Matrix3d Plot::ellipsoidLinearMap(const Eigen::Matrix3d & S, double nSigma)
{
    const double c = GaussianBase::mahalanobisThresholdSq(nSigma, 3.0);
    return std::sqrt(c) * S.transpose();
}

std::vector<Eigen::Vector3d> Plot::ellipsoidSurfacePoints(const Eigen::Vector3d & mu, const Eigen::Matrix3d & S,
                                                         double nSigma, int nLat, int nLon)
{
    if (nLat < 2 || nLon < 3)
        throw std::invalid_argument("ellipsoidSurfacePoints: grid too small");
    const Mat3 A = ellipsoidLinearMap(S, nSigma);
    std::vector<Vec3> pts;
    pts.reserve(static_cast<std::size_t>(nLat) * static_cast<std::size_t>(nLon));
    for (int i = 0; i < nLat; ++i)
    {
        const double th = std::numbers::pi * static_cast<double>(i) / static_cast<double>(nLat - 1);
        for (int j = 0; j < nLon; ++j)
        {
            const double ph = 2.0 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(nLon);
            const Vec3 u(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th));
            pts.push_back(mu + A * u);
        }
    }
    return pts;
}

bool Plot::displayAvailable()
{
#if defined(_WIN32) || defined(__APPLE__)
    return true;
#else
    const char * x11 = std::getenv("DISPLAY");
    const char * wayland = std::getenv("WAYLAND_DISPLAY");
    return (x11 != nullptr && *x11 != '\0') || (wayland != nullptr && *wayland != '\0');
#endif
}

std::string Plot::requestedBackendClass(WindowMode mode)
{
    return mode == WindowMode::Offscreen ? std::string(kOffscreenWindowClass) : std::string();
}

std::string Plot::parseReportField(const std::string & report, const std::string & label)
{
    std::size_t lineStart = 0;
    while (lineStart < report.size())
    {
        std::size_t lineEnd = report.find('\n', lineStart);
        if (lineEnd == std::string::npos)
            lineEnd = report.size();
        const std::string line = report.substr(lineStart, lineEnd - lineStart);
        const std::size_t at = line.find(label);
        if (at != std::string::npos)
        {
            std::string value = line.substr(at + label.size());
            const auto first = value.find_first_not_of(" \t\r");
            if (first == std::string::npos)
                return std::string();
            const auto last = value.find_last_not_of(" \t\r");
            return value.substr(first, last - first + 1);
        }
        lineStart = lineEnd + 1;
    }
    return std::string();
}

// ----------------------------------------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------------------------------------

Plot::Plot(const Camera & camera, cv::Size inputSize, WindowMode mode)
    : impl_(std::make_unique<Impl>())
{
    if (!camera.isValid())
        throw std::invalid_argument("Plot: camera is not valid");
    if (inputSize.width < 2 || inputSize.height < 2)
        throw std::invalid_argument(std::format("Plot: input size {}x{} too small", inputSize.width, inputSize.height));
    if (inputSize.width != camera.imageWidth() || inputSize.height != camera.imageHeight())
        throw std::invalid_argument(std::format("Plot: input size {}x{} does not match camera {}x{}",
                                                inputSize.width, inputSize.height,
                                                camera.imageWidth(), camera.imageHeight()));
    if (mode == WindowMode::Onscreen && !displayAvailable())
        throw std::runtime_error("interactive visualisation requires a graphical display (DISPLAY is not set); "
                                 "run with --interactive=0 or set DISPLAY");

    Impl & p = *impl_;
    p.camera = camera;
    p.inputSize = inputSize;
    p.onscreen = (mode == WindowMode::Onscreen);
    p.sqrtChi2 = std::sqrt(GaussianBase::mahalanobisThresholdSq(kNSigma, 3.0));

    // Undistorted normalised coordinates of the image corners (pixel edges, -0.5 .. W-0.5).
    const std::array<Vec2, 4> corners{Vec2(-0.5, -0.5), Vec2(inputSize.width - 0.5, -0.5),
                                      Vec2(inputSize.width - 0.5, inputSize.height - 0.5),
                                      Vec2(-0.5, inputSize.height - 0.5)};
    for (std::size_t i = 0; i < 4; ++i)
    {
        Vec2 n = camera.pixelToNormalised(corners[i]);
        if (!isFinite(Vec3(n.x(), n.y(), 1.0)))
            n = Vec2((corners[i].x() - camera.cx()) / camera.fx(), (corners[i].y() - camera.cy()) / camera.fy());
        p.cornersNormalised[i] = n;
    }

    // ----- window -----
    const int W2 = 2 * inputSize.width;
    const int H = inputSize.height;
    p.window = createRenderWindow(mode);
    p.window->SetOffScreenRendering(p.onscreen ? 0 : 1);
    p.window->SetMultiSamples(0);
    p.window->SetAlphaBitPlanes(0);
    p.window->SetSize(W2, H);
    p.window->SetWindowName("MCHA4400 visual navigation");

    // ----- left pane: overlay image -----
    p.imageRenderer = vtkSmartPointer<vtkRenderer>::New();
    p.imageRenderer->SetViewport(0.0, 0.0, 0.5, 1.0);
    p.imageRenderer->SetBackground(0.0, 0.0, 0.0);
    p.imageRenderer->SetInteractive(0);
    p.window->AddRenderer(p.imageRenderer);

    p.imageData = vtkSmartPointer<vtkImageData>::New();
    p.imageData->SetDimensions(inputSize.width, inputSize.height, 1);
    // Spacing W/(W-1) so the quad spans exactly [0, W] x [0, H] world units: one image pixel = one pane pixel.
    p.imageData->SetSpacing(static_cast<double>(inputSize.width) / (inputSize.width - 1),
                            static_cast<double>(inputSize.height) / (inputSize.height - 1), 1.0);
    p.imageData->SetOrigin(0.0, 0.0, 0.0);
    p.imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
    p.imageActor = vtkSmartPointer<vtkImageActor>::New();
    p.imageActor->SetInputData(p.imageData);
    p.imageActor->GetProperty()->SetInterpolationTypeToNearest();
    p.imageRenderer->AddViewProp(p.imageActor);
    {
        vtkCamera * cam = p.imageRenderer->GetActiveCamera();
        cam->ParallelProjectionOn();
        cam->SetFocalPoint(inputSize.width / 2.0, inputSize.height / 2.0, 0.0);
        cam->SetPosition(inputSize.width / 2.0, inputSize.height / 2.0, 1000.0);
        cam->SetViewUp(0.0, 1.0, 0.0);
        cam->SetParallelScale(inputSize.height / 2.0);
        p.imageRenderer->ResetCameraClippingRange();
    }

    // ----- right pane: 3-D scene -----
    p.sceneRenderer = vtkSmartPointer<vtkRenderer>::New();
    p.sceneRenderer->SetViewport(0.5, 0.0, 1.0, 1.0);
    p.sceneRenderer->SetBackground(kBackgroundRGB[0], kBackgroundRGB[1], kBackgroundRGB[2]);
    p.sceneRenderer->SetUseDepthPeeling(0);
    p.window->AddRenderer(p.sceneRenderer);

    p.unitSphere = vtkSmartPointer<vtkSphereSource>::New();
    p.unitSphere->SetRadius(1.0);
    p.unitSphere->SetThetaResolution(kSphereThetaResolution);
    p.unitSphere->SetPhiResolution(kSpherePhiResolution);
    p.unitSphere->Update();
    p.sphere = buildSphereTemplate(p.unitSphere->GetOutput());

    p.unitQuad = vtkSmartPointer<vtkPolyData>::New();
    {
        auto pts = vtkSmartPointer<vtkPoints>::New();
        pts->InsertNextPoint(-0.5, -0.5, 0.0);
        pts->InsertNextPoint(0.5, -0.5, 0.0);
        pts->InsertNextPoint(0.5, 0.5, 0.0);
        pts->InsertNextPoint(-0.5, 0.5, 0.0);
        auto polys = vtkSmartPointer<vtkCellArray>::New();
        const vtkIdType ids[4] = {0, 1, 2, 3};
        polys->InsertNextCell(4, ids);
        p.unitQuad->SetPoints(pts);
        p.unitQuad->SetPolys(polys);
    }

    // World axes N / E / D, drawn with the display rotation so that the labels are correct.
    p.axes = vtkSmartPointer<vtkAxesActor>::New();
    p.axes->SetAxisLabels(false);   // captions overlapped the scene; the legend names the axes instead
    p.axesM = vtkSmartPointer<vtkMatrix4x4>::New();
    setAffine(p.axesM, displayRotation(), Vec3::Zero());
    p.axes->SetUserMatrix(p.axesM);
    p.axes->SetTotalLength(1.0, 1.0, 1.0);
    p.sceneRenderer->AddActor(p.axes);

    // Validation track (magenta) first, then trajectory (yellow), so the estimate is drawn on top.
    p.createPolyline(p.validation, kValidationRGB, 2.0);
    p.createPolyline(p.trajectory, kTrajectoryRGB, 3.0);

    // Camera frustum: 5 points (apex + 4 corners), 8 edges.
    p.frustumPoints = vtkSmartPointer<vtkPoints>::New();
    for (int i = 0; i < 5; ++i)
        p.frustumPoints->InsertNextPoint(0.0, 0.0, 0.0);
    auto frustumLines = vtkSmartPointer<vtkCellArray>::New();
    const std::array<std::array<vtkIdType, 2>, 8> edges{{{0, 1}, {0, 2}, {0, 3}, {0, 4},
                                                         {1, 2}, {2, 3}, {3, 4}, {4, 1}}};
    for (const auto & e : edges)
        frustumLines->InsertNextCell(2, e.data());
    p.frustumPoly = vtkSmartPointer<vtkPolyData>::New();
    p.frustumPoly->SetPoints(p.frustumPoints);
    p.frustumPoly->SetLines(frustumLines);
    auto frustumMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    frustumMapper->SetInputData(p.frustumPoly);
    p.frustumActor = vtkSmartPointer<vtkActor>::New();
    p.frustumActor->SetMapper(frustumMapper);
    p.frustumActor->GetProperty()->SetLighting(false);
    p.frustumActor->GetProperty()->SetLineWidth(2.0);
    setColour(p.frustumActor, toVec3(kCameraRGB));
    p.sceneRenderer->AddActor(p.frustumActor);

    p.cameraMarker = p.createSphereActor(p.cameraMarkerM, false);
    setColour(p.cameraMarker, toVec3(kCameraRGB));
    p.cameraEllipsoid = p.createSphereActor(p.cameraEllipsoidM, true);
    setColour(p.cameraEllipsoid, toVec3(kCameraEllipsoidRGB));

    // Landmark batches: created once, geometry is updated in place (see the file documentation).
    const int sphereVerts = static_cast<int>(p.sphere.points.size());
    p.createBatch(p.ellipsoids, Impl::BatchKind::Triangles, sphereVerts, false, true);
    p.createBatch(p.markers, Impl::BatchKind::Triangles, sphereVerts, true, false);
    p.createBatch(p.squares, Impl::BatchKind::Squares, 8, false, false);

    // Text overlays (drawn last by VTK: vtkTextActor is a 2-D overlay prop).
    p.statusText = vtkSmartPointer<vtkTextActor>::New();
    p.statusText->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
    p.statusText->GetTextProperty()->SetJustificationToLeft();
    p.statusText->GetTextProperty()->SetVerticalJustificationToTop();
    p.statusText->SetPosition(12, H - 12);
    p.sceneRenderer->AddViewProp(p.statusText);

    p.legendText = vtkSmartPointer<vtkTextActor>::New();
    p.legendText->GetTextProperty()->SetColor(0.85, 0.85, 0.85);
    p.legendText->GetTextProperty()->SetJustificationToLeft();
    p.legendText->GetTextProperty()->SetVerticalJustificationToBottom();
    p.legendText->SetPosition(12, 12);
    p.legendText->SetInput("axes: red N (north), green E (east), blue D (down)\n"
                           "yellow: estimated trajectory   white: camera (3-sigma cyan)\n"
                           "magenta: validation GPS track (NOT used by estimator)\n"
                           "landmarks: own colour, 3-sigma wireframe; dim = not visible\n"
                           "drag: rotate   right-drag: zoom   'r': auto view   'q': quit");
    p.sceneRenderer->AddViewProp(p.legendText);

    // Overview camera defaults (updated on each update()).
    {
        vtkCamera * cam = p.sceneRenderer->GetActiveCamera();
        cam->SetViewAngle(kViewAngleDeg);
        cam->SetViewUp(0.0, 1.0, 0.0);
    }

    // Preallocate the RGBA read-back buffer once.
    p.rgbaBuffer = vtkSmartPointer<vtkUnsignedCharArray>::New();
    p.rgbaBuffer->SetNumberOfComponents(4);

    // ----- interaction (onscreen only) -----
    if (p.onscreen)
    {
        p.interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
        p.interactor->SetRenderWindow(p.window);
        p.style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
        p.interactor->SetInteractorStyle(p.style);

        Impl * implPtr = impl_.get();
        p.keyCommand = vtkSmartPointer<vtkCallbackCommand>::New();
        p.keyCommand->SetClientData(implPtr);
        p.keyCommand->SetCallback([](vtkObject * caller, unsigned long, void * clientData, void *)
        {
            auto * self = static_cast<Plot::Impl *>(clientData);
            auto * iren = vtkRenderWindowInteractor::SafeDownCast(caller);
            if (iren == nullptr || iren->GetKeySym() == nullptr)
                return;
            const std::string key = iren->GetKeySym();
            if (key == "n" || key == "space" || key == "c")
            {
                self->continueRequested = true;
                iren->TerminateApp();
            }
            else if (key == "q" || key == "Escape")
            {
                self->quitRequested = true;
                iren->TerminateApp();
            }
            else if (key == "r")
            {
                self->autoView = true;
                self->viewInitialised = false;
                self->dirty = true;
            }
        });
        p.interactor->AddObserver(vtkCommand::KeyPressEvent, p.keyCommand);

        p.exitCommand = vtkSmartPointer<vtkCallbackCommand>::New();
        p.exitCommand->SetClientData(implPtr);
        p.exitCommand->SetCallback([](vtkObject * caller, unsigned long, void * clientData, void *)
        {
            auto * self = static_cast<Plot::Impl *>(clientData);
            self->quitRequested = true;
            if (auto * iren = vtkRenderWindowInteractor::SafeDownCast(caller))
                iren->TerminateApp();
        });
        p.interactor->AddObserver(vtkCommand::ExitEvent, p.exitCommand);

        p.interactionCommand = vtkSmartPointer<vtkCallbackCommand>::New();
        p.interactionCommand->SetClientData(implPtr);
        p.interactionCommand->SetCallback([](vtkObject *, unsigned long, void * clientData, void *)
        {
            static_cast<Plot::Impl *>(clientData)->autoView = false;   // user takes control of the camera
        });
        p.style->AddObserver(vtkCommand::StartInteractionEvent, p.interactionCommand);
    }

    // ----- backend: initialise the context once and record the GL strings actually in use -----
    p.backendClass = p.window->GetClassName();
    p.window->Initialize();
    const char * report = p.window->ReportCapabilities();
    const std::string caps = report != nullptr ? report : "";
    const std::string vendor = parseReportField(caps, "OpenGL vendor string:");
    const std::string renderer = parseReportField(caps, "OpenGL renderer string:");
    p.backendDescription = std::format("{} (OpenGL vendor '{}', renderer '{}')", p.backendClass,
                                       vendor.empty() ? std::string("unknown") : vendor,
                                       renderer.empty() ? std::string("unknown") : renderer);
}

Plot::~Plot()
{
    if (!impl_)
        return;
    Impl & p = *impl_;
    if (p.interactor)
    {
        p.interactor->SetRenderWindow(nullptr);
    }
    if (p.window)
    {
        p.window->Finalize();
    }
}

// ----------------------------------------------------------------------------------------------------
// Accessors
// ----------------------------------------------------------------------------------------------------

cv::Size Plot::inputSize() const noexcept
{
    return impl_->inputSize;
}

cv::Size Plot::frameSize() const noexcept
{
    return cv::Size(2 * impl_->inputSize.width, impl_->inputSize.height);
}

bool Plot::isOnscreen() const noexcept
{
    return impl_->onscreen;
}

const std::string & Plot::backendName() const noexcept
{
    return impl_->backendClass;
}

const std::string & Plot::backendDescription() const noexcept
{
    return impl_->backendDescription;
}

const Plot::Timings & Plot::lastTimings() const noexcept
{
    return impl_->timings;
}

bool Plot::quitRequested() const noexcept
{
    return impl_->quitRequested;
}

// ----------------------------------------------------------------------------------------------------
// update(): all scene state from PlotFrameData
// ----------------------------------------------------------------------------------------------------

void Plot::update(const PlotFrameData & data)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    Impl & p = *impl_;
    const int W = p.inputSize.width;
    const int H = p.inputSize.height;

    // ----- left pane: BGR -> RGB, rows bottom-up, in one pass into the VTK buffer -----
    {
        auto * dst = static_cast<unsigned char *>(p.imageData->GetScalarPointer());
        const std::size_t rowBytes = static_cast<std::size_t>(W) * 3;
        if (!data.overlay.empty())
        {
            cv::Mat src = data.overlay;
            if (src.size() != p.inputSize)
                throw std::invalid_argument(std::format("Plot::update: overlay {}x{} does not match input {}x{}",
                                                        src.cols, src.rows, W, H));
            if (src.type() == CV_8UC1)
            {
                cv::cvtColor(src, p.overlayBGR, cv::COLOR_GRAY2BGR);
                src = p.overlayBGR;
            }
            else if (src.type() != CV_8UC3)
            {
                throw std::invalid_argument("Plot::update: overlay must be BGR8 or GRAY8");
            }
            // one SIMD colour conversion per row; the destination row index flips the image vertically
            for (int y = 0; y < H; ++y)
            {
                const cv::Mat srcRow(1, W, CV_8UC3, const_cast<unsigned char *>(src.ptr<unsigned char>(y)));
                cv::Mat dstRow(1, W, CV_8UC3, dst + static_cast<std::size_t>(H - 1 - y) * rowBytes);
                cv::cvtColor(srcRow, dstRow, cv::COLOR_BGR2RGB);
            }
        }
        else
        {
            std::fill_n(dst, static_cast<std::size_t>(W) * static_cast<std::size_t>(H) * 3, static_cast<unsigned char>(0));
        }
    }
    p.imageData->Modified();

    // ----- bounds of the content (display coordinates) and smoothed scene radius -----
    const Mat3 M = displayRotation();
    Vec3 lo = Vec3::Constant(std::numeric_limits<double>::infinity());
    Vec3 hi = -lo;
    // Axis-aligned box of the content in display coordinates; extent = bounding radius of a point's uncertainty.
    auto include = [&](const Vec3 & worldPoint, double extent)
    {
        if (!isFinite(worldPoint))
            return;
        const Vec3 q = M * worldPoint;
        lo = lo.cwiseMin(q - Vec3::Constant(extent));
        hi = hi.cwiseMax(q + Vec3::Constant(extent));
    };
    for (const Vec3 & t : data.trajectory)
        include(t, 0.0);
    for (const Vec3 & v : data.validationTrack)
        include(v, 0.0);
    include(data.rCNn, 0.0);
    for (const auto & lm : data.landmarks)
    {
        if (!isFinite(lm.rLNn) || !lm.positionSqrtCov.allFinite())
            continue;
        // ||S||_F >= ||S||_2, so sqrt(c) ||S||_F bounds the 3-sigma ellipsoid around its mean.
        include(lm.rLNn, p.sqrtChi2 * lm.positionSqrtCov.norm());
    }
    double radius = kMinSceneScale;
    if (std::isfinite(lo.x()) && std::isfinite(hi.x()))
        radius = std::max(kMinSceneScale, 0.5 * (hi - lo).norm());
    Vec3 centre = (std::isfinite(lo.x()) && std::isfinite(hi.x())) ? Vec3(0.5 * (lo + hi)) : M * data.rCNn;
    if (!isFinite(centre))
        centre = Vec3::Zero();

    // Smooth the scene radius (grows at once, shrinks slowly) so markers and frustum do not pulse.
    if (!p.viewInitialised)
        p.sceneRadius = radius;
    else
        p.sceneRadius = std::max(radius, (1.0 - kViewSmoothing) * p.sceneRadius + kViewSmoothing * radius);
    const double scale = std::max(p.sceneRadius, kMinSceneScale);

    // ----- trajectory and validation (display coordinates via M inside setPolyline) -----
    p.setPolyline(p.trajectory, data.trajectory, M);
    p.setPolyline(p.validation, data.validationTrack, M);
    p.validation.actor->SetVisibility(!data.validationTrack.empty());

    // ----- camera frustum, marker and position ellipsoid -----
    const double depth = kFrustumDepthFraction * scale;
    const Vec3 apexWorld = data.rCNn;
    if (isFinite(apexWorld) && data.Rnc.allFinite())
    {
        const Vec3 apex = M * apexWorld;
        p.frustumPoints->SetPoint(0, apex.x(), apex.y(), apex.z());
        for (std::size_t i = 0; i < 4; ++i)
        {
            const Vec3 cam(p.cornersNormalised[i].x() * depth, p.cornersNormalised[i].y() * depth, depth);
            const Vec3 q = M * (apexWorld + data.Rnc * cam);
            p.frustumPoints->SetPoint(static_cast<vtkIdType>(i + 1), q.x(), q.y(), q.z());
        }
        p.frustumPoints->Modified();
        p.frustumPoly->Modified();
        p.frustumActor->SetVisibility(true);

        setAffine(p.cameraMarkerM, Mat3::Identity() * (kMarkerFraction * scale), apex);
        p.cameraMarker->SetVisibility(true);

        if (data.positionSqrtCov.allFinite())
        {
            setAffine(p.cameraEllipsoidM, M * Plot::ellipsoidLinearMap(data.positionSqrtCov, kNSigma),
                      apex);
            p.cameraEllipsoid->SetVisibility(true);
        }
        else
        {
            p.cameraEllipsoid->SetVisibility(false);
        }
    }
    else
    {
        p.frustumActor->SetVisibility(false);
        p.cameraMarker->SetVisibility(false);
        p.cameraEllipsoid->SetVisibility(false);
    }

    // ----- landmarks: batched geometry, updated in place -----
    // Pass 1: assign the slots of this frame. Growing the arrays may leave their storage unset, so no point is
    // written before every slot exists (writing before growing left uninitialised points in the bounds).
    std::vector<long long> keys;
    keys.reserve(data.landmarks.size());
    for (const auto & lm : data.landmarks)
    {
        if (isFinite(lm.rLNn) && lm.positionSqrtCov.allFinite())
            keys.push_back(Impl::landmarkKey(lm.id, lm.isPose));
    }
    p.assignSlots(keys);
    std::fill(p.ellipsoids.active.begin(), p.ellipsoids.active.end(), static_cast<unsigned char>(0));
    std::fill(p.markers.active.begin(), p.markers.active.end(), static_cast<unsigned char>(0));
    std::fill(p.squares.active.begin(), p.squares.active.end(), static_cast<unsigned char>(0));
    // Pass 2: write the drawn landmarks.
    for (const auto & lm : data.landmarks)
    {
        if (!isFinite(lm.rLNn) || !lm.positionSqrtCov.allFinite())
            continue;
        const std::size_t slot = p.slotIndex.find(Impl::landmarkKey(lm.id, lm.isPose))->second;
        const Vec3 colour = clampColour(lm.colourRGB) * (lm.visible ? 1.0 : kDimFactor);
        const Vec3 mean = M * lm.rLNn;

        p.writeSphere(p.ellipsoids, slot, M * Plot::ellipsoidLinearMap(lm.positionSqrtCov, kNSigma), mean, colour,
                      false);
        if (lm.isPose)
        {
            const double side = std::max(0.0, lm.size);
            if (side > 0.0)
            {
                const Vec3 edge = lm.visible ? Vec3::Ones() : Vec3::Constant(0.5);
                p.writeSquare(slot, M * lm.Rnl * side, mean, colour, edge);
            }
        }
        else
        {
            const double r = kMarkerFraction * scale * (lm.visible ? kVisibleMarkerScale : kHiddenMarkerScale);
            p.writeSphere(p.markers, slot, M * (Mat3::Identity() * r), mean, colour, true);
        }
    }
    // Pass 3: slots that are not drawn this frame are parked at the scene centre. Their points still belong to the
    // polydata bounds (VTK computes them from all points), so they must stay finite and inside the content.
    p.parkInactive(p.ellipsoids, centre);
    p.parkInactive(p.markers, centre);
    p.parkInactive(p.squares, centre);
    p.commit(p.ellipsoids);
    p.commit(p.markers);
    p.commit(p.squares);

    // ----- world axes (length tracks the scene) -----
    const double axisLength = 0.35 * scale;
    p.axes->SetTotalLength(axisLength, axisLength, axisLength);

    // ----- status text -----
    std::string status;
    for (std::size_t i = 0; i < data.statusLines.size(); ++i)
    {
        if (i > 0)
            status += '\n';
        status += data.statusLines[i];
    }
    p.statusText->SetInput(status.c_str());
    p.statusText->GetTextProperty()->SetFontSize(std::max(10, H / 50));
    p.legendText->GetTextProperty()->SetFontSize(std::max(9, H / 60));

    // ----- overview camera -----
    if (p.autoView)
    {
        if (!p.viewInitialised)
        {
            p.focus = centre;
            p.viewInitialised = true;
        }
        else
        {
            p.focus = (1.0 - kViewSmoothing) * p.focus + kViewSmoothing * centre;
        }
        const double halfAngle = 0.5 * kViewAngleDeg * std::numbers::pi / 180.0;
        const double distance = kViewMargin * scale / std::sin(halfAngle);
        const Vec3 dir = Vec3(kViewDirection[0], kViewDirection[1], kViewDirection[2]).normalized();
        vtkCamera * cam = p.sceneRenderer->GetActiveCamera();
        const Vec3 pos = p.focus + distance * dir;
        cam->SetFocalPoint(p.focus.x(), p.focus.y(), p.focus.z());
        cam->SetPosition(pos.x(), pos.y(), pos.z());
        cam->SetViewUp(0.0, 1.0, 0.0);
        cam->SetViewAngle(kViewAngleDeg);
        p.sceneRenderer->ResetCameraClippingRange();
    }

    p.dirty = true;
    p.timings.updateMs = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------------------------------
// render() / capture()
// ----------------------------------------------------------------------------------------------------

void Plot::render()
{
    using clock = std::chrono::steady_clock;
    Impl & p = *impl_;
    const auto t0 = clock::now();
    p.window->Render();
    p.dirty = false;
    p.timings.renderMs = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

void Plot::capture(cv::Mat & bgr)
{
    using clock = std::chrono::steady_clock;
    Impl & p = *impl_;
    if (p.dirty)
        render();
    const auto t0 = clock::now();

    const int W2 = 2 * p.inputSize.width;
    const int H = p.inputSize.height;
    // front buffer (1) for onscreen windows (after the swap), back buffer (0) for offscreen.
    const int front = p.onscreen ? 1 : 0;
    const int ok = p.window->GetRGBACharPixelData(0, 0, W2 - 1, H - 1, front, p.rgbaBuffer);
    if (ok != 1 || p.rgbaBuffer->GetNumberOfTuples() != static_cast<vtkIdType>(W2) * H)
        throw std::runtime_error("Plot::capture: VTK failed to read back the render window");

    if (bgr.size() != cv::Size(W2, H) || bgr.type() != CV_8UC3 || !bgr.isContinuous())
        bgr.create(H, W2, CV_8UC3);

    // RGBA (bottom-up, GL) -> BGR (top-down) in a single pass.
    // one SIMD colour conversion per row; the source row index flips the image vertically
    unsigned char * src = p.rgbaBuffer->GetPointer(0);
    const std::size_t srcRowBytes = static_cast<std::size_t>(W2) * 4;
    for (int y = 0; y < H; ++y)
    {
        const cv::Mat srcRow(1, W2, CV_8UC4, src + static_cast<std::size_t>(H - 1 - y) * srcRowBytes);
        cv::Mat dstRow(1, W2, CV_8UC3, bgr.ptr<unsigned char>(y));
        cv::cvtColor(srcRow, dstRow, cv::COLOR_RGBA2BGR);
    }

    p.timings.captureMs = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------------------------------
// Interactive modes
// ----------------------------------------------------------------------------------------------------

void Plot::blockUntilClosed()
{
    Impl & p = *impl_;
    if (!p.onscreen)
        throw std::logic_error("Plot::blockUntilClosed requires an onscreen plot");

    render();
    if (!p.interactorReady)
    {
        p.interactor->Initialize();
        p.interactorReady = true;
    }
    while (!p.quitRequested)
    {
        p.interactor->SetDone(false);
        p.interactor->Start();   // returns on TerminateApp (key or window close)
        if (!p.quitRequested)
        {
            // 'n' / 'space' / 'c' in mode 1 has no meaning: keep the window open until closed.
            p.continueRequested = false;
            render();
        }
    }
}

bool Plot::showAndWait()
{
    Impl & p = *impl_;
    if (!p.onscreen)
        throw std::logic_error("Plot::showAndWait requires an onscreen plot");

    render();
    if (!p.interactorReady)
    {
        p.interactor->Initialize();
        p.interactorReady = true;
    }
    p.continueRequested = false;
    p.interactor->SetDone(false);
    p.interactor->Start();   // returns on continue / quit / close
    return !p.quitRequested;
}
