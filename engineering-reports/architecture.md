# Architecture and mathematical design — MCHA4400 Assignment 2 (T-A2-001)

Labels: **VERIFIED** (checked by test or measurement), **ASSUMED** (design choice without a written spec; `project.txt`
contains only the project name), **USER** (user decision), **DECISION** (engineering decision with rationale).

## 1. Overview

```
main.cpp (CLI) ──> calibrateCamera()                              (-c)
              └──> runVisualNavigationFromVideo()                 (--scenario=4|5|6)
                     BufferedVideoReader ─> ScenarioRunner::processFrame ─> plotData ─> Plot (VTK, EGL) ─> BufferedVideoWriter
                                               │
             Scenario4 (flight)  Scenario5 (identical tags)  Scenario6 (room points)
                     └──────────── shared estimation core ────────────┘
   GaussianInfo (sqrt information form) · SystemVisualNav (dynamics) · SystemEstimator (predict)
   MeasurementBase (Laplace MAP update) · funcmin (trust-region Newton) · Camera · Pose/rotation
```

- `ScenarioRunner.h` (Lead-owned contract): `initialise(camera, fps, frameSize, videoPath)`, `processFrame(frame, k, t)`,
  `plotData() -> PlotFrameData` (overlay image, camera pose, 3x3 position sqrt-covariance, trajectory, landmarks with
  sqrt-covariances, validation-only track, status lines).
- Frame loop: `visualNavigation.cpp`; per-stage timing via `FrameTimer` → `out/<stem>_timing.csv`.
- Export: `out/<stem>_out<ext>`, mp4v, size 2W x H, input fps (VERIFIED by ffprobe for all three videos).

## 2. Frames and conventions

- World `n`: z down. Outdoor: origin at the initial body position, x = initial forward direction (NED-like, heading
  ASSUMED 0 because it is unobservable). Indoor: origin and axes at the initial pose.
- Body `b`: x forward, y right, z down. Camera `c`: x right, y down, z optical axis.
  `Rbc = [[0,0,1],[1,0,0],[0,1,0]]` (columns = camera axes in b), `rCBb = 0` (ASSUMED; no extrinsic spec).
- Attitude: `Theta = [phi, theta, psi]`, `R_nb = Rz(psi) Ry(theta) Rx(phi)`; `Thetadot = TK(Theta) omega_b`.
  Singular at |pitch| = 90 deg (documented; not reached in the data).

## 3. Square-root information form (GaussianInfo)

- Density stored as information vector `nu` and UPPER-triangular `Xi` with `Lambda = Xi^T Xi`, `nu = Lambda mu`.
- Operations by Householder Q-less QR (`core::triangularise`) and triangular solves only: marginal (permute eliminated
  variables first, QR, keep trailing block), conditional, join, information update (stack rows, QR), affine/nonlinear
  transform, log-density with gradient and Hessian, confidence ellipse/ellipsoid (nSigma = 3).
- USER rule: **no covariance inversion anywhere** (filter, gates, diagnostics, classifiers). Remaining `inverse()` calls
  are of Jacobian/kinematic maps (`TK`, birth Jacobians), not covariances. Covariances are formed only in tests (as
  references) and for drawing/diagnostics from `S^T S`.
- Gate convention (DECISION): `nSigma -> p = erf(nSigma/sqrt2)`, threshold `chi2inv(p, dim)` via
  `GaussianBase::mahalanobisThresholdSq`. nSigma=3: 9.0 (1-D), 11.83 (2-D), 14.16 (3-D).

## 4. Process model (SystemVisualNav)

- State `x = [nu(6): vBNb, omegaBNb; eta(6): rBNn, Thetanb; etaPrev(6); landmarks]`, landmarks Point(3) or
  Pose(6) = `[r_jn^n; Theta_nj]`.
- `nudot = w` (random walk, diagonal PSD per scenario); `etadot = [R_nb v; TK omega]`; etaPrev and landmarks static.
- Discretisation: RK4 on the 12-state dynamic block with its state-transition matrix (block-wise; exact because the
  full STM is blockdiag(Phi12, I)); process noise from the integrated random walk, `L^T L = Qd^{-1}` (full rank for
  positive PSD and |pitch| < 90 deg).
- Information-form time update: joint over `[x_k; x_{k+1,S}]` with rows `[-L Phi_S | L]`, then elimination of the old
  variables by QR. Delayed variant shifts `eta_k` into `etaPrev`. `|dt| <= 1e-9 s` is a no-op (fix C1); a zero-step
  delayed shift adds 1e-4 jitter so the joint stays representable in information form (DECISION, documented).
- VERIFIED: STM vs FD, prediction vs an independent EKF reference including cross-covariances, block vs full STM to 1e-10.

## 5. Measurement update (MeasurementBase)

- `process(system)`: predict to `t`, then Laplace update: minimise `V(x) = -log p(y|x) - log p(x)` with a trust-region
  Newton method (`funcmin`, exact eigen subproblem incl. hard case; Cholesky fast path), then the posterior factor from
  the Hessian at the MAP.
- Square-root update (fix C2): Cholesky path if `K` is PD; else eigen factor `F = diag(sqrt(lambda+)) V^T`,
  `informationUpdate(F, w)`; truly indefinite `K` → flagged fallback (`precisionIsExact=false`, not reached in production).
- MAP outcome policy (R4-01): Converged / AcceptedMaxIterations (Newton norm <= 0.05) / Skipped{MaxIterations,
  NumericalFailure, NoCurvature}; skipped updates keep the prediction and are counted.
- Support restriction: MAP over the observed sub-state only (exact by the Gaussian conditional identity).
- Gauss-Newton Hessian by default for residual models; the omitted second-order term is zero at zero residual
  (VERIFIED by FD at zero residual for each measurement).

## 6. Camera and calibration

- OpenCV 5-coefficient model; projection identical to `cv::projectPoints` (VERIFIED to 1e-12); hand-written
  `d(pixel)/d(rPCc)` (FD-verified); `pixelToVector` via undistortion.
- Calibration (DECISION): chessboard 10x7, 22 mm; `findChessboardCornersSB` (EXHAUSTIVE) on strided frames, quality and
  duplicate-pose rejection, 60 views, principal point fixed at the image centre (free principal point was unstable across
  view subsets), standard 5 coefficients (5-fold CV did not favour the rational model).
  Outdoor RMS 0.667 px (fx 2236), indoor RMS 1.118 px (fx 889, weakly determined: ±5 % across subsets).

## 7. Scenario 4 — flight (flow + horizon + altimeter)

- State `[nu; eta; etaPrev]` (18). Priors: v 5 m/s, omega 0.5 rad/s, xy 1 m, z 0.1 m, roll/pitch 0.1 rad, yaw 0.5 rad.
  nuPsd = [1,1,1, 0.05,0.05,0.05].
- **Optical-flow plane bundle** (support eta, etaPrev): LK tracks (ImageFlow, half resolution, forward–backward check)
  from a keyframe every 5 frames; homography `H = R_BA + t_BA n_A^T / d_A` for the ground plane `z = Hg`
  (**Hg = first SRT altitude, ASSUMED**: ground under the start point at the altimeter reference); plane RANSAC; sigma
  1 px; 12 hand-written `dH/dtheta`.
- **Visual horizon** (support Theta): CIELAB + texture Gaussian sky/ground classifier trained on patches cropped from
  `flight.MOV` (`data/outdoor/sky_*.png`, `ground_*.png`; no samples were supplied), evaluated in square-root form;
  per-column boundary with clipped smoothing (bias fix P2) and sub-pixel gradient refinement; 2-point RANSAC; residual
  `a·p/||a_xy||`, `a = Rbc^T R_nb^T e3`; sigma 5 px (PROVISIONAL).
- **Altimeter** (support z): event-driven on each SRT altitude change (~5 Hz, 760 events), `y = h - h0 = -(z - z0) + v`,
  sigma 0.02 m (ASSUMED; the NIS-calibrated 0.066 m was rejected out-of-sample; see test-report).
- Observability: flow gives relative rotation and `t/d`; the altimeter fixes scale; horizon fixes roll/pitch.
  **Yaw and horizontal position are unobservable** (no compass, GPS excluded) and are dead-reckoned (USER-approved
  reporting). SRT latitude/longitude are converted to a local track for the 3-D view ONLY (VERIFIED by a test that
  two SRTs differing only in lat/lon give bitwise-identical estimates).
- Overlay (ASSUMED colours, BGR, draw order): frame → sky tint (200,120,40) α0.3 → horizon candidates orange → horizon
  inliers red → predicted horizon yellow → flow inliers green / rejected magenta → status text.

## 8. Scenario 5 — identical-ID ArUco tags

- DICT_6X6_50, every tag ID 5 (VERIFIED on the video). Tag side **0.15 m ASSUMED** (metric scale ∝ side).
- Pose landmarks `[r; Theta]`; measurement = 8 corner pixels, corners at ±s/2 in the tag frame, hand-written
  Jacobians w.r.t. eta and the landmark pose (FD-verified), GN Hessian; pixel sigma 0.629 px (CALIBRATED on the first
  half of the video, validated on the second half).
- Geometric association (IDs identical): predicted corners + innovation factor, gate `mahalanobisThresholdSq(3, 8)`,
  conflicts/ambiguity rejection; birth by IPPE (`SOLVEPNP_IPPE_SQUARE`) with ambiguity resolution and correlated
  augmentation `addLandmarkLinear`; occlusion-aware visibility; archive re-identification gate in square-root form
  (R4-09); culling after 60 misses.
- etaPrev used by the depth-free flow measurement (below) between consecutive frames.
- Overlay (ASSUMED): detections yellow, matched green with 3-sigma ellipses, missed red, birth magenta, labels last.

## 9. Scenario 6 — room point SLAM

- Point landmarks; keyframe pose in the etaPrev slot; persistent LK tracks with descriptor re-acquisition; parallax-
  gated two-view triangulation births (angle >= 2 deg, depth 0.3–15 m), correlated augmentation; association by track
  identity + gate (square-root form); speed prior |v| ~ N(0.5, 0.25²) m/s (**ASSUMED, fixes monocular scale**);
  cheirality repair (R5): mirrored translation branch detected by positive-depth counts and re-seeded from recoverPose.
- **Quality: INADEQUATE** (see final-report): sparse map (mean 8.4 landmarks), short landmark lifetimes.

## 10. Depth-free flow measurement (MeasurementIndoorFlow, S5/S6)

- Relative pose `R = R_cB^T R_cA`, `t = R_cB^T (c_A - c_B)`; Sampson epipolar residual on normalised rays
  (`E = [t]x R`) or rotation-only model when the parallax is negligible; hand-written Jacobians w.r.t. eta and etaPrev
  (FD-verified); RANSAC front end; translation scale unobservable (degeneracy documented).

## 11. Visualisation (Plot)

- One VTK window 2W x H: left = overlay image (1:1), right = 3-D scene (axes N/E/D red/green/blue, trajectory yellow,
  validation track magenta, camera frustum white, camera 3-sigma ellipsoid cyan, landmarks in own colour with 3-sigma
  wireframes, pose landmarks as squares, text last). ASSUMED colours (no spec).
- `--interactive=0`: EGL offscreen (NVIDIA) regardless of DISPLAY (DECISION: 5–10x faster than X/Mesa); modes 1/2: X window.
- Batched landmark geometry, in-place point updates, SIMD readback.

## 12. Build-system decisions

- `autodiff 1.1.2` (USER B1), Boost 1.88.0 (USER B2), hand-written derivatives (USER B3).
- HDF5 imported targets' `NDEBUG;_FORTIFY_SOURCE=2` stripped so Debug asserts are active (VERIFIED in build.ninja).
- `EIGEN_INITIALIZE_MATRICES_BY_NAN` in Debug (kept).
