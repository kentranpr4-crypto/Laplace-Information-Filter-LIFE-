# Plan — audit, requirement matrix, executed plan (T-A2-001)

## 1. Audit at start (Phase 0)

- Not a git repository. `project.txt` = "a2" only, so no assignment text or visual spec was available.
  No Lab 4–10 or Assignment 1 code was present.
- Source at start: `main.cpp` (CLI), `BufferedVideo.*` (threaded I/O), stub `calibrate.cpp`, stub `visualNavigation.cpp`
  (frame loop without processing). `test/src/main.cpp` held only the doctest entry point, and there were 0 tests.
- Data: `data/outdoor/{calibration.MOV, flight.MOV (HEVC 2688x1512, 59.94 fps, 9161 frames), flight.SRT, config.xml}`,
  `data/indoor/{calibration.MOV, tag_identical.MOV (1074 frames), room.MOV (1470 frames), config.xml}`.
  No `sky*/ground*` images were present.
- Environment: Ubuntu 22.04, GCC 11/12, OpenCV 4.5, Eigen 3.4, VTK 9.1, no root. The required versions had to be
  installed in user space.

## 2. Requirement matrix (final status)

| Requirement (AGENTS.md / user) | Status | Evidence |
|---|---|---|
| Native C++23 toolchain at required versions, no Docker/root | verified (2 user-approved deviations) | verify-env 78 PASS / 0 FAIL |
| Reproducible setup/verify/build/test scripts + README | complete; clean-machine **UNTESTED** | scripts/, README.md |
| Camera calibration (both cameras), camera.xml | verified | RMS 0.667 / 1.118 px; reload tests |
| SRT parsing, altitude only, no GPS leakage | verified | DJIVideoCaption tests; leakage test |
| GaussianInfo sqrt-information filter, QR, triangular solves | verified | GaussianInfo/Core tests; review probes |
| Dynamics, STM, process noise, info-form predict | verified | SystemVisualNav tests vs EKF reference |
| Laplace measurement update | verified | linear = Kalman test; regression tests |
| No covariance inversion anywhere (user rule) | complete | grep audit + regression tests (R4-09, Lead R5) |
| Hand-written Jacobians/Hessians, FD-validated (user B3) | verified | FD tests for every measurement |
| S4 flow + horizon + altimeter → trajectory | complete; heading unobservable | CLI run + test-report |
| S5 identical-tag SLAM with delayed pose | complete; scale ASSUMED | CLI run + test-report |
| S6 point SLAM with flow | complete but **INADEQUATE quality** | test-report, final-report |
| VTK dual-pane visualisation, export, interactive modes | complete (modes 1/2 smoke-tested only) | Plot tests; exports reopened |
| Debug tests + Release benchmarks | verified | test-report, performance |
| Independent five-pass review, fixes | complete | code-review.md |
| Doxygen (BUILD_DOCUMENTATION=ON) | **NOT RUN** | — |

## 3. Executed plan (milestones)

1. **M1 Environment**: micromamba prefix from a conda-forge lock file, plus scripts and README.
   Blockers were escalated to the user: B1 (autodiff 1.1.3 does not exist), B2 (Boost 1.74 is incompatible with
   VTK 9.5.2), B3 (autodiff's Eigen 5 integration is broken).
2. **M2 Core + camera**: GaussianInfo, rotation/Pose, funcmin, SystemVisualNav, MeasurementBase, Camera, calibration,
   SRT parsing. Also a BufferedVideo lost-wakeup race fix and the Lead-owned `ScenarioRunner.h` contract.
3. **M3 Scenarios + visualisation** (rounds 1–5, parallel Peers with disjoint file ownership):
   - S4: flow, horizon, altimeter. The P1 root cause was synthetic-renderer aliasing, not the tracker; P2 horizon bias fixed.
   - S5: ArUco SLAM with re-identification.
   - S6: point SLAM, rework, cheirality round.
   - VIS: Plot with EGL offscreen rendering.
   - Core fixes: C1 tiny-dt, C2 square-root update, HDF5 NDEBUG leak.
4. **M4 Review/fix/benchmark/report**: independent review (0 CRIT / 0 HIGH / 10 MED / 12 LOW), fixes R4-01..R4-10,
   README refresh, code freeze, idle-machine Release CLI benchmarks, these reports.

## 4. Plan changes (recorded)

- Peers resumed via SendMessage re-sent old reports instead of working. From then on, fresh foreground Peers were
  spawned instead.
- Scenario 6 was reworked twice (round 4, then the user-authorised cheirality round 5). Per the user rule it is
  reported INADEQUATE rather than tuned further.
- Disk ran low (2.4 GB free) from Peer build trees. Regenerable scratch trees were deleted, which restored 18 GB.
