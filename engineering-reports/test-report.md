# Test report (T-A2-001)

All numbers below were observed on the integrated tree on the reference machine (i7-12650H, 16 threads, 15 GB RAM,
RTX 3050 Laptop, Ubuntu 22.04.5). "Peer driver" means a scratch driver program linked against the project library,
used for per-measurement diagnostics. The CLI runs are the real `a2` binary.

## 1. Environment verification

| Command | Result |
|---|---|
| `scripts/verify-env.sh --project` | **78 PASS, 0 FAIL, 33 INFO**, exit 0 |

The INFO lines include the known autodiff-Eigen 5 header limitation. The project does not use that header (user decision B3).

## 2. Unit and integration tests (doctest, `unit_tests` via CTest)

| Build | Command | Test cases | Assertions | Wall |
|---|---|---|---|---|
| Debug (asserts live, NaN init) | `scripts/build.sh Debug` | 244: **243 pass, 1 fail** | 8515: 8514 pass | 82.8 s |
| Release (-O3, NDEBUG) | `scripts/build.sh Release` | 244: **243 pass, 1 fail** | 8515: 8514 pass | 24.3 s |
| ASan + UBSan Debug (review Peer, pre-S6-rework snapshot) | scratch build | 201/201 | 7616/7616 | 153 s, 0 sanitizer reports |

**The single failure, kept on purpose.** `test/src/S6WalkingCamera.cpp:296` checks a synthetic walking-camera room:
the median landmark-to-wall distance must be <= 0.10 m. It measures 0.134 m. The tolerance was fixed before the run and
was not loosened; the user decided "no tuning to look good". This failure is the Scenario 6 quality gap (section 5).

**Important build fix.** Before R3, Debug builds silently compiled with `-DNDEBUG`, because conda's HDF5 target leaked
it through VTK. Counts from before that fix were obtained with assertions off. All counts above have assertions active.

### Coverage by area (representative, all passing)

- GaussianInfo: moment/info/sqrt round trips; Q-less QR (tall, wide, rank-deficient); marginal and conditional
  against moment-form references; join and product; information update; affine and nonlinear transforms;
  log-density gradient and Hessian vs finite differences (FD); chi2 quantiles; ellipse and ellipsoid.
- Rotation and pose: orthonormality, rpy round trip, every derivative vs FD; pitch guard (R4-17).
- funcmin: Rosenbrock; hard case; subproblem exactness; NaN rejection; objective exceptions contained (R4-02);
  Cholesky fast path.
- SystemVisualNav: dynamics and STM vs FD; process-noise factor; information-form predict vs an EKF reference
  (incl. cross-covariances); block-wise STM equal to full STM to 1e-10; delayed shift; tiny-dt no-op (C1).
- MeasurementBase: linear-Gaussian Laplace update equals the Kalman update; square-root eigen update vs a
  moment-form reference (C2); MAP outcome policy (R4-01/16).
- Camera: projection vs `cv::projectPoints` and Jacobian vs FD (many random points); camera.xml reload;
  missing fixture fails (R4-03).
- Calibration: synthetic recovery; outlier rejection; RMS recomputed from camera.xml (to 1e-6 px);
  deterministic model selection (R4-15).
- SRT: field values; CRLF/BOM/malformed input; integer-ms timing (C3, frame 21); frame alignment.
  **GPS leakage test:** two SRTs differing only in lat/lon give bitwise-identical estimates.
- S4:
  - Jacobians/Hessians vs FD: flow homography, horizon, altimeter.
  - LK slope 0.94 on an area-integrated synthetic scene (P1 regression).
  - Horizon bias -0.02 px (P2 regression).
  - Synthetic flight: x error 1.24 m vs a 1.5 m tolerance.
- S5:
  - Corner Jacobians vs FD.
  - IPPE ambiguity.
  - Gating, ambiguity, conflict.
  - Occlusion lifecycle and re-identification.
  - Re-identification gate in square-root form vs reference (R4-09).
  - Synthetic 4-tag scene.
- S6:
  - Triangulation and birth Jacobians vs FD.
  - Speed prior.
  - Track association.
  - Cheirality repair regression.
  - Walking camera: T1 scale 1.029 PASS, T2 RMS 0.029 m PASS, **T3 0.134 m FAIL**, T4 no rotation absorption PASS.
- Visualisation and I/O:
  - Plot left pane orientation.
  - Ellipsoid geometry.
  - EGL backend selection.
  - Export round trip (2W x H, frame count, fps).
  - Frame loop with a dummy runner.
  - BufferedVideo race and failure-contract regressions (R4-12).
- Square-root form (Lead, R5): S6 gate distance and sky/ground classifier log-odds equal the explicit-inverse
  references, which are computed only inside the tests.

## 3. Calibration (Release, CLI `./a2 ../data/<dir>/config.xml -c`)

| Camera | Image | Views | RMS | fx, fy | Matches video |
|---|---|---|---|---|---|
| Outdoor | 2688x1512 | 60 (4200 corners) | 0.667 px | 2236.4, 2237.9 | flight.MOV ✓ |
| Indoor | 1920x1080 | 60 (4200 corners) | 1.118 px | 889.3, 892.3 | tag_identical.MOV, room.MOV ✓ |

Indoor fx varies by about ±5 % across view subsets, so the indoor intrinsics are weakly determined.

## 4. Scenario acceptance runs (Release CLI from `build-release/` with `../data` paths, after code freeze)

See `performance.md` for timings. Every export was reopened with ffprobe:

| Scenario | Command | Exit | Export reopened |
|---|---|---|---|
| 4 | `./a2 ../data/outdoor/flight.MOV --scenario=4 -e` | 0 | mpeg4 5376x1512, 59.94 fps, 9161/9161 frames |
| 5 | `./a2 ../data/indoor/tag_identical.MOV --scenario=5 -e` | 0 | mpeg4 3840x1080, 29.97 fps, 1074/1074 frames |
| 6 | `./a2 ../data/indoor/room.MOV --scenario=6 -e` | 0 | mpeg4 3840x1080, 29.97 fps, 1470/1470 frames |

All three runs printed `Visualisation backend: vtkEGLRenderWindow (... NVIDIA GeForce RTX 3050 ...)`. The Lead inspected
exported frames visually:
- S5: tag outlines sit on the markers.
- S6: a cluster of landmarks on the window curtain with large ellipsoids, consistent with the INADEQUATE verdict.
- S4: horizon and flow overlays present.

## 5. Estimate quality (Peer drivers on the full videos, same code paths and parameters as the CLI)

### Scenario 4: flight (`out/scenario4_r5_refA_flight_run_summary.txt`, final parameters)

**MAP outcomes.** Flow 9160/9160 updates converged, horizon 9159/9159, altimeter 760/760. No skips and no MaxIterations.

**Altitude.** At the SRT change instants, post-update RMSE is 0.023 m (max 0.099 m). The altimeter IS this SRT field,
so this is a consistency check, not an independent accuracy result.

**Validation only (GPS never enters the estimator).**

| Metric | Estimate | GPS |
|---|---|---|
| End-to-end displacement | 828 m | 795 m (+4 %) |
| Horizontal path, raw | 2415 m (inflated by frame jitter) | 1832 m |
| Horizontal path, ±0.5 s smoothed | 2109 m | 1832 m |

Fits of the estimated track to GPS:

| Fit | RMSE | Max error | Parameters |
|---|---|---|---|
| Similarity | 34.8 m | 74 m | scale 1.108 |
| Rigid, best yaw | 56.8 m | 93.9 m | — |

Heading and horizontal position cannot be observed with these sensors; heading is dead-reckoned.

**NIS consistency is NOT achieved** (honest; recorded as review findings R4-06/07):

| Measurement | NIS/dof median | Inside 95 % band |
|---|---|---|
| Flow | 0.008 | 0 % |
| Horizon | 0.57–0.82 | 24–58 % |
| Altitude | 1.75–1.88 | 73–77 % |

Calibrated sigmas were tried and rejected out-of-sample (code-review.md §7).

**Verdict: acceptable within the stated observability limits.** Altitude and scale are consistent; heading drifts; NIS
is not consistent.

### Scenario 5: identical tags (`out/scenario5_r5_summary.txt`, sigma 0.629 px)

- Pixel sigma calibrated on the first half of the video. On the out-of-sample second half: NIS/dof mean 1.06,
  median 0.98, with 82.9 % of frames in the 95 % band.
- MAP outcomes: tag updates 959/959 converged; flow updates 1068 converged, 2 skipped for no curvature; 0 failures.
- Map: 43 births, 37 culls, 4 re-identified, 39 identities, 6 tags at the end, no duplicate pairs.
- There is no ground-truth tag count. The tag side of 0.15 m is ASSUMED, so metric scale scales with it.
- **Verdict: usable.** Identity churn remains.

### Scenario 6: room points (`out/scenario6_r5_summary.txt`, after the user-authorised cheirality round)

Criteria fixed before the run, and the results:

| Criterion | Target | Result | Status |
|---|---|---|---|
| Behind-camera triangulations | < 10 % | 6.7 % (was 54 %) | met |
| Mean map size | >= 30 | 8.4 | **not met** |
| Landmark median lifetime | >= 30 frames | 7 | **not met** |
| Recognisable structure in top view | yes | no (`out/scenario6_r5_map_top.png`) | **not met** |
| Walking-camera test T3 | <= 0.10 m | 0.134 m | **not met** |

Other numbers:

- Flow NIS/dof median 1.36; point NIS/dof median 0.047.
- Speed median 0.079 m/s against the ASSUMED 0.5 m/s prior, so scale is not trustworthy.
- Trajectory extent about 1.2 x 0.8 m, path 12.1 units.
- Map failures: 0.

**Verdict: INADEQUATE.** Reported as final per the user rule; no further tuning.

## 6. NOT RUN / UNTESTED

- Clean-machine install (another Ubuntu, a container, no GPU): **UNTESTED**.
- Doxygen (`-DBUILD_DOCUMENTATION=ON`): **NOT RUN**.
- Interactive modes 1 and 2: smoke-tested only (window opens and runs until a timeout); the mode-2 key advance was
  not exercised.
- No ground truth exists for any trajectory or map. Accuracy statements rest only on GPS (S4, validation only),
  synthetic tests, and internal consistency.
- ASan/UBSan was run before the S6 rework and R5 fixes; it was not re-run on the final tree.
