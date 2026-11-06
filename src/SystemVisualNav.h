#ifndef SYSTEMVISUALNAV_H
#define SYSTEMVISUALNAV_H

#include <vector>
#include <Eigen/Core>
#include "GaussianInfo.h"
#include "SystemBase.h"
#include "SystemEstimator.h"

/**
 * @brief Shared rigid-body model for all visual-navigation scenarios.
 *
 * State layout (all blocks are contiguous, in this order):
 *   - nu      [0, 6)   : vBNb (3, body velocity), omegaBNb (3, body angular velocity)       random walk
 *   - eta     [6, 12)  : rBNn (3, position of body in world), Thetanb (3, ZYX Euler angles) kinematics
 *   - etaPrev [12, 18) : delayed previous pose eta_{k-1} (present only when hasDelayedPose())
 *   - landmarks          : N blocks, 3 (point: r_jn^n) or 6 (pose: [r_jn^n; Theta_nj]); static
 *
 * Dynamics (continuous time):
 *   nudot = 0 (random walk driven by dW with PSD diag(nuPsd)),
 *   etadot = [Rnb(Theta) vBNb ; TK(Theta) omegaBNb],
 *   etaPrevdot = 0, landmarks static.
 *
 * Discrete process noise over one step dt (linearised about the state at the start of the step):
 *   w_nu  = W(dt),  I = int_0^dt W(s) ds,  Delta eta = T I,  with T = blockdiag(Rnb, TK).
 *   Per axis (variance q = nuPsd):   Cov[W, I] = [[q dt, q dt^2/2],[q dt^2/2, q dt^3/3]],
 *   factor Lz = [[D1, D2],[0, D3]] with D1 = sqrt(q dt), D2 = dt^2/2 sqrt(q/dt), D3 = sqrt(q dt^3/12),
 *   Qd = G Cz G^T, G = blockdiag(I, T), so Qd^{-1} = M^T M with M = Lz^{-T} G^{-1}.
 *   Qd is full rank over the 12 dynamic states iff all nuPsd > 0 and T is invertible. The body pitch is limited to
 *   |theta| <= 85 deg (requireKinematicPitchSupported, rotation.hpp): dynamics() and processNoiseSqrtInfo() throw
 *   std::domain_error beyond it.
 *
 * The Jacobians of the flow are hand-written (rotation.hpp) and validated against central differences.
 */
class SystemVisualNav : public SystemBase
{
public:
    /** @brief Landmark parameterisation. */
    enum class LandmarkType
    {
        Point,  ///< 3 states: r_jn^n
        Pose    ///< 6 states: [r_jn^n ; Theta_nj] (ZYX Euler angles of the landmark frame)
    };

    /** @brief Description of one landmark block in the current state layout. */
    struct Landmark
    {
        Eigen::Index id = -1;        ///< stable identifier returned by addLandmark
        LandmarkType type = LandmarkType::Point;
        Eigen::Index offset = 0;     ///< first index of the block in the state
        Eigen::Index dim = 0;        ///< 3 or 6
    };

    static constexpr Eigen::Index kNuOffset = 0;        ///< [vBNb; omegaBNb]
    static constexpr Eigen::Index kEtaOffset = 6;       ///< [rBNn; Thetanb]
    static constexpr Eigen::Index kEtaPrevOffset = 12;  ///< delayed pose (when present)
    static constexpr Eigen::Index kDynamicDim = 12;     ///< nu + eta
    static constexpr Eigen::Index kPoseDim = 6;         ///< dimension of nu / eta / etaPrev blocks

    /**
     * @brief Constructs the model.
     * @param nuPsd  continuous-time PSD diagonal of the random walk on [vBNb; omegaBNb] (units (m/s)^2/s, (rad/s)^2/s);
     *               every entry must be strictly positive so that Qd is full rank.
     * @throws std::invalid_argument for non-positive or non-finite entries
     */
    explicit SystemVisualNav(const Eigen::Matrix<double, 6, 1>& nuPsd);

    /** @brief Flow xdot = f(t, x) and its Jacobian (see class documentation). */
    Eigen::VectorXd dynamics(double t, const Eigen::VectorXd& x, Eigen::MatrixXd* A) const override;

    /** @brief Stochastic block: indices 0..11 (nu and eta). */
    std::vector<Eigen::Index> stochasticIndices() const override;

    /** @brief The flow depends on nu and eta only (12 states); etaPrev and landmarks are static. */
    Eigen::Index dynamicBlockDim() const override { return kDynamicDim; }

    /**
     * @brief Process-noise square-root information (12 x 12, upper triangular, L^T L = Qd^{-1}).
     * @throws std::invalid_argument unless dt > kZeroTimeStep (Qd^{-1} grows like 1/dt^3; zero steps are no-ops).
     */
    Eigen::MatrixXd processNoiseSqrtInfo(double dt, const Eigen::VectorXd& x) const override;

    /** @brief Expected state dimension for the current layout (dynamic + delayed pose + landmarks). */
    Eigen::Index expectedDim() const;

    /** @brief True when the delayed pose block etaPrev is part of the state. */
    bool hasDelayedPose() const { return hasDelayedPose_; }

    /** @brief Landmark blocks in state order, with their current offsets. */
    std::vector<Landmark> landmarks() const;

    /** @brief Index of the first element of the landmark with the given id. @throws std::out_of_range if unknown. */
    Eigen::Index landmarkOffset(Eigen::Index id) const;

    /** @brief Dimension (3 or 6) of a landmark type. */
    static Eigen::Index landmarkDim(LandmarkType type);

    /**
     * @brief Adds a landmark block with the given initial density (augmentation with an independent block).
     * @param est      estimator whose state is augmented
     * @param type     landmark type; the density dimension must equal landmarkDim(type)
     * @return stable landmark id
     */
    Eigen::Index addLandmark(SystemEstimator& est, LandmarkType type, const GaussianInfo<double>& initial);

    /**
     * @brief Adds a landmark block that is an affine function of the current state plus independent noise:
     *        x_L = G x + c + w,  w ~ N(0, Qw),  Qw^{-1} = lS^T lS.
     *
     * The joint density of (x, x_L) keeps the cross-covariance Cov(x, x_L) = G P exactly (to the linearisation
     * of G), unlike addLandmark which appends an independent block. Implemented with GaussianInfo::linearisedTransition.
     * @param est   estimator whose state is augmented (its dimension must equal expectedDim())
     * @param type  landmark type
     * @param g     landmarkDim(type) x est.dim() matrix
     * @param c     landmarkDim(type) constant term
     * @param lS    landmarkDim(type) square invertible sqrt-information of w
     * @return stable landmark id
     */
    Eigen::Index addLandmarkLinear(SystemEstimator& est, LandmarkType type, const Eigen::MatrixXd& g,
                                   const Eigen::VectorXd& c, const Eigen::MatrixXd& lS);

    /** @brief Removes a landmark block by marginalisation. @throws std::out_of_range if the id is unknown. */
    void removeLandmark(SystemEstimator& est, Eigen::Index id);

    /**
     * @brief Time update with the delayed-state shift: etaPrev(new) = eta(old).
     *
     * The previous etaPrev is marginalised out. Afterwards hasDelayedPose() is true.
     *
     * Zero-length steps (|tNew - time| <= kZeroTimeStep) use the exact zero-step semantics instead of the noise
     * model: nu and eta are carried over unchanged (no flow, no process noise) and the new etaPrev is a copy of
     * eta_k, realised as eta_k + w with w ~ N(0, 1e-8 I_6) (independent jitter, see kDelayedCopyStd in the source).
     * An exact copy is a singular joint density and cannot be stored in information form; the jitter keeps the mean
     * exact and changes the covariance of etaPrev by 1e-8 I_6 only.
     */
    void predictDelayed(SystemEstimator& est, double tNew);

    /** @brief Time update keeping etaPrev (if present) unchanged. */
    void predictPlain(SystemEstimator& est, double tNew);

    /** @brief Removes the delayed pose block by marginalisation (no-op error if absent). */
    void discardDelayedPose(SystemEstimator& est);

private:
    Eigen::Matrix<double, 6, 1> nuPsd_;
    bool hasDelayedPose_ = false;
    std::vector<Eigen::Index> landmarkIds_;
    std::vector<LandmarkType> landmarkTypes_;
    Eigen::Index nextId_ = 0;

    Eigen::Index landmarkBase() const { return kEtaPrevOffset + (hasDelayedPose_ ? kPoseDim : 0); }
};

#endif
