# Laplace Information Filter for Visual Navigation and SLAM

A C++23 implementation of a **square-root information-form filter with Laplace (MAP) updates**, applied to
camera-based state estimation on real video. The same estimator drives three independent pipelines:

| Scenario | Task | Sensors and measurements |
|---|---|---|
| **4. Flight** | Drone trajectory from visual odometry | Lucas-Kanade optical flow, visual horizon, altimeter |
| **5. Identical tags** | SLAM with ArUco markers that all share one ID | Tag corners, optical flow, geometric data association |
| **6. Room** | Monocular point-landmark SLAM in a room | ORB/LK tracks, keyframe epipolar flow, point landmarks |

> **Status: two of three scenarios meet their goals; Scenario 6 does not.**
> All three run end to end on the real videos and export annotated video, but the Scenario 6 room map is
> reported as inadequate (see [Results](#results) and [Known limitations](#known-limitations)).

---

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Testing and verification](#testing-and-verification)
- [Known limitations](#known-limitations)
- [Assumptions](#assumptions)
- [Documentation](#documentation)

---

## How it works

### The estimator

The state is a Gaussian stored in **square-root information form**: an information vector and an upper-triangular
factor `R` with `Λ = RᵀR`. Every operation (marginalisation, conditioning, prediction, measurement updates) is done
with Q-less QR and triangular solves. **A covariance matrix is never formed or inverted anywhere in the filter
path**, which is checked by a source audit and by regression tests that compare against explicit-inverse references
that exist only inside the tests.

Measurement updates use a **Laplace approximation**: find the MAP point of the posterior with a trust-region
Newton method, then take the Gaussian whose information matrix is the Hessian of the negative log posterior at that
point. For a linear-Gaussian measurement this reproduces the Kalman update exactly (tested).

All Jacobians and Hessians are hand-derived and validated against central finite differences in the test suite.

### Shared model

One rigid-body model serves all scenarios. The state holds body velocities, the current pose, a **delayed previous
pose** (used by the optical-flow likelihoods), and landmarks. Velocities follow a random walk, propagated with RK4
and its transition matrix in information form.

### Data flow per scenario

```
Scenario 4   frame ──► Lucas-Kanade flow ──────► ground-plane flow likelihood ─┐
             frame ──► sky/ground classifier ──► horizon likelihood            ├─► predict ─► Laplace update ─► pose, 3σ
             .SRT  ──► altitude only ──────────► altimeter likelihood          ┘

Scenario 5   frame ──► ArUco detection ─► geometric association ─► tag corner likelihood / new tag birth ─┐
             frame ──► flow tracking ───► depth-free flow odometry likelihood ────────────────────────────┴─► update ─► pose + tag map

Scenario 6   frame ──► LK tracks + ORB ─► keyframes ─► epipolar flow + speed prior ──────┐
                                       ├─► parallax-gated triangulation (landmark birth) ├─► update ─► pose + point map
                                       └─► track association (id, then descriptor) ──────┘
```

Each run renders a two-pane VTK view (image on the left, 3D scene with 3σ ellipsoids on the right) and exports it
as video through a threaded buffered reader and writer.

---

## Results

All numbers are from one reference machine (Ubuntu 22.04.5, x86_64, 16 cores, NVIDIA GPU), Release build, run through
the real command-line interface. Run-to-run variance is about ±40% (laptop power governor).

| Scenario | Frames | Export | Mean ms/frame | Wall time | Verdict |
|---|---|---|---|---|---|
| 4. Flight | 9161 | 5376×1512 | 168 | about 1542 s | Acceptable within observability limits |
| 5. Identical tags | 1074 | 3840×1080 | 108 | about 117 s | Usable |
| 6. Room | 1470 | 3840×1080 | 107 | about 158 s | **Inadequate** |

None runs in real time.

**Camera calibration** (10×7 chessboard, principal point fixed at image centre, 5 distortion coefficients):
outdoor RMS 0.667 px, indoor RMS 1.118 px.

### Scenario 4 detail
- Net displacement is within +4% of the GPS track. GPS is used **for validation only and never enters the filter**
  (a test shows two subtitle files differing only in latitude/longitude give bitwise-identical estimates).
- Position error against GPS is 34.8 m after a full similarity fit and 56.8 m after yaw-only alignment.
- Heading is not observable with these sensors, so it drifts (dead reckoning). This dominates the horizontal error.
- Altitude agrees with the subtitle altimeter to 0.023 m RMSE, but that is the same sensor, so it is a consistency
  check and not independent evidence.

### Scenario 5 detail
- Tag outlines match the markers in the exported frames. The map holds 6 tags at the end of the video (39 identities
  were created along the way, so some identity churn remains). There is no ground-truth tag count.
- On held-out frames, 82.9% fall inside the 95% pixel-noise band.

### Scenario 6 detail
The room map misses the targets that were fixed before the final run: mean map size 8.4 landmarks (target 30 or more),
median landmark lifetime 7 frames (target 30 or more), and no recognisable room structure. A cheirality fix cut
points triangulated behind the camera from 54% to 6.7%, but not enough. These targets were chosen by the project and are
not from an external specification.

---

## Quick start

Full instructions, including Ubuntu 24.04/26.04 and WSL2 notes, are in
[HUONG-DAN-CHAY.md](HUONG-DAN-CHAY.md) (Vietnamese) and [ENVIRONMENT.md](ENVIRONMENT.md) (English, technical).

### Requirements

- x86_64 Linux (Ubuntu 22.04 or newer), normal user account, no root needed for the toolchain
- At least 12 GiB free disk for the install, plus space for data and builds
- The videos are **not** in this repository (see [Data](#data))

### Install, build, run

```bash
scripts/setup-env.sh                     # installs GCC 14, OpenCV 5, Eigen 5, VTK 9.5, ... into ~/opt/mcha4400-a2
scripts/verify-env.sh --project          # checks every dependency
scripts/build.sh Release -- --target a2  # builds build-release/a2

cd build-release
./a2 ../data/outdoor/config.xml -c                        # calibrate outdoor camera
./a2 ../data/indoor/config.xml  -c                        # calibrate indoor camera
./a2 ../data/outdoor/flight.MOV       --scenario=4 -e     # flight
./a2 ../data/indoor/tag_identical.MOV --scenario=5 -e     # identical tags
./a2 ../data/indoor/room.MOV          --scenario=6 -e     # room
```

Flags: `-c/--calibrate`, `-s=N/--scenario=N` (4, 5 or 6), `-i=N/--interactive=N` (0, 1 or 2), `-e/--export`.
Write `--scenario=5` or `-s=5`; a space-separated `-s 5` is rejected by the argument parser.
Exports go to `out/<video>_out.MOV` with per-frame timings in `out/<video>_timing.csv`.

### Data

Place the input videos under `data/outdoor/` (`calibration.MOV`, `flight.MOV`, `flight.SRT`, `config.xml`) and
`data/indoor/` (`calibration.MOV`, `tag_identical.MOV`, `room.MOV`, `config.xml`). For Scenario 4 also add your own
`sky_*.png` and `ground_*.png` sample crops in `data/outdoor/` for the visual-horizon classifier. `data/` is git-ignored.

### Toolchain

| Component | Version |
|---|---|
| GCC | 14.4.0 (C++23) |
| CMake / Ninja | 3.31.8 / 1.13.2 |
| OpenCV | 5.0.0 (headless build) |
| Eigen | 5.0.1 |
| VTK | 9.5.2 (offscreen EGL rendering) |
| Boost | 1.88.0 (header-only Boost.Math used) |
| doctest | 2.5.3 |

Installed into one isolated prefix from a pinned lock file. autodiff 1.1.2 is declared in CMake but the project
does not use it: its Eigen integration header does not compile with Eigen 5.0.1, so derivatives are hand-written.
Boost 1.88.0 replaces 1.74 because VTK 9.5.2 requires exactly that version.

---

## Repository layout

```
src/        estimator, measurement models, front ends, SLAM layers, visualisation, CLI
test/       doctest suite and small fixtures
scripts/    setup-env, env, verify-env, build, test, and a CMake probe project
engineering-reports/   plan, architecture, test report, code review, performance, final report
```

Source modules by layer:

| Layer | Files |
|---|---|
| Math | `rotation.hpp`, `Pose.hpp`, `CoreLinalg.h`, `funcmin.*` |
| Filter core | `GaussianInfo.h`, `SystemVisualNav.*`, `SystemEstimator.*`, `MeasurementBase.*` |
| Camera and input | `Camera.*`, `calibrate.*`, `DJIVideoCaption.*`, `BufferedVideo.*` |
| Measurement models | `MeasurementAltimeter`, `MeasurementHorizon`, `MeasurementOutdoorFlowBundle`, `MeasurementIndoorFlow`, `MeasurementPointBundle`, `MeasurementTagBundle` |
| Image front ends | `ImageFlow.*`, `VisualHorizon.*`, `PointFeatures.*`, `ArucoTags.*`, `IndoorFlowFrontEnd.*`, `S6KeyframeFlow.*` |
| SLAM | `Slam*.{h,cpp}` (landmark map, association, initialisation, triangulation, tag tracker, point pipeline) |
| Drivers and output | `Scenario4/5/6`, `Plot.*`, `FrameTimer.*`, `visualNavigation.*`, `main.cpp` |

---

## Testing and verification

```bash
scripts/build.sh Debug -- --target a2 --target tests   # Debug build with assertions enabled
scripts/test.sh                                        # runs ctest
```

- **244 test cases, 243 pass.** The one failing case is the Scenario 6 map-quality test
  (`test/src/S6WalkingCamera.cpp`, 0.134 m against a 0.10 m limit). It is failing on purpose: the tolerance was not
  loosened to hide the quality gap.
- Tests cover Jacobians and Hessians against finite differences, QR and information-form consistency, prediction
  against an independent EKF reference, chi-squared gating, calibration reload, SRT parsing and the no-GPS-leakage rule.
- An independent review found 0 critical and 0 high issues outside Scenario 6, plus 10 medium and 12 low findings.
  All medium findings were addressed or documented. See [engineering-reports/code-review.md](engineering-reports/code-review.md).
- An AddressSanitizer and UBSan run of the suite was clean, but it predates the final fixes and was not repeated.

---

## Known limitations

- **Scenario 6 is inadequate** (see [Results](#results)). Likely next steps: longer-lived tracks with descriptor
  re-identification across keyframes, a better landmark birth policy, rejection of unstable features on windows and
  curtains, and windowed bundle adjustment.
- **Scale is not observable** for the monocular room scenario. It rests on an assumed speed prior (0.5 ± 0.25 m/s)
  that the data do not support (measured median about 0.08 m/s).
- **Scenario 4 heading drifts**, and the noise statistics are not consistent (flow is too conservative, the altimeter
  is too tight). Calibrated noise values were tried and rejected because they failed on held-out data.
- Not real time on the reference machine.
- **Untested:** installation on a clean machine, Ubuntu 24.04 and 26.04, WSL2, machines without an NVIDIA GPU,
  Doxygen generation, and interactive window modes beyond a smoke test. Portability is not claimed.
- No ground truth exists for any trajectory or map.

---

## Assumptions

The original task specification was not available, so these choices were made and are documented in
[engineering-reports/architecture.md](engineering-reports/architecture.md):

- overlay colours and draw order for the visualisation
- altimeter reference (first subtitle sample), update rate (about 5 Hz) and noise (0.02 m)
- Scenario 4 ground-plane height and initial heading (unobservable, set to zero)
- Scenario 5 tag side length of 0.15 m, which sets the metric scale of the tag map
- Scenario 6 speed prior
- body-to-camera extrinsics (axis permutation, zero lever arm)
- visual-horizon training patches cropped from the flight video

---

## Documentation

| Document | Content |
|---|---|
| [ENVIRONMENT.md](ENVIRONMENT.md) | Toolchain install, verification, pinned versions, known blockers |
| [HUONG-DAN-CHAY.md](HUONG-DAN-CHAY.md) | Running the project on another machine: Ubuntu and WSL2 (Vietnamese) |
| [engineering-reports/final-report.md](engineering-reports/final-report.md) | Outcome, evidence, limitations |
| [engineering-reports/architecture.md](engineering-reports/architecture.md) | State layout, models, equations, design decisions |
| [engineering-reports/test-report.md](engineering-reports/test-report.md) | Test commands and results |
| [engineering-reports/performance.md](engineering-reports/performance.md) | Release benchmarks |
| [engineering-reports/code-review.md](engineering-reports/code-review.md) | Review findings and resolutions |

## License

No license has been specified for this repository. All rights reserved by the author until one is added.
