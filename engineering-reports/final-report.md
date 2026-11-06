# Final report — MCHA4400 Assignment 2 (T-A2-001)

User goal: "Hoàn thành project này hết sức có thể với mục tiêu hoàn thành các track đã nói nếu viết bằng C++ càng tốt nhé",
with a native install of the exact environment (no Docker), reproducible scripts, Debug tests, Release benchmarks,
and an independent review.

## 1. Outcome at a glance

| Track | Status | Short verdict |
|---|---|---|
| Environment (GCC 14, Ninja, OpenCV 5.0, Eigen 5.0, VTK 9.5, autodiff, Boost, doctest, onnxruntime) | DONE | verify-env 78 PASS / 0 FAIL. Two user-approved deviations. |
| Calibration (outdoor + indoor) | DONE | RMS 0.667 px / 1.118 px |
| Scenario 4: flight (LK flow, horizon, altimeter, LIF) | DONE | Acceptable within observability limits (heading unobservable). NIS not consistent. |
| Scenario 5: identical-ID ArUco SLAM | DONE | Usable. Tag size ASSUMED. Some identity churn. |
| Scenario 6: room point SLAM | DONE (runs, exports) | **INADEQUATE quality** (final, per user rule) |
| VTK dual-pane visualisation and export | DONE | EGL offscreen; all exports reopen correctly |
| Tests | DONE | 244 cases: 243 pass, 1 honest failure (S6 quality) |
| Independent review + fixes | DONE | 0 CRIT, 0 HIGH (non-S6), 10 MED, 12 LOW. All MED addressed. |
| Release benchmarks | DONE | 5.95–9.34 frames/s; not real time |

## 2. VERIFIED (observed by the Lead on the integrated, frozen tree)

- `scripts/verify-env.sh --project`: 78 PASS, 0 FAIL, 33 INFO. Installed versions:
  - GCC/g++ 14.4.0, CMake 3.31.8, Ninja 1.13.2
  - OpenCV 5.0.0 (headless, FFmpeg 8.1.2), Eigen 5.0.1, VTK 9.5.2
  - autodiff 1.1.2, Boost 1.88.0, doctest 2.5.3, onnxruntime 1.28.0, Doxygen 1.17.0
  - All in `~/opt/mcha4400-a2` (3.8 GB), from the lock file `scripts/conda-lock-linux-64.txt`.
- Tests, Debug with assertions live: 244 cases, 243 pass, 8514/8515 assertions, 82.8 s. Release: same counts, 24.3 s.
  The one failure is `S6WalkingCamera.cpp:296` (0.134 m vs 0.10 m). The tolerance was kept unchanged on purpose.
- All three scenarios ran through the real CLI from `build-release/` with `../data` paths, exit 0. Exports reopened:

  | Scenario | Export | Frames | Mean ms/frame | Wall |
  |---|---|---|---|---|
  | S4 | 5376x1512 | 9161/9161 | 168.0 | 1542 s |
  | S5 | 3840x1080 | 1074/1074 | 108.1 | 117 s |
  | S6 | 3840x1080 | 1470/1470 | 106.9 | 158 s |

- Exported frames inspected by eye:
  - S4: horizon on the true horizon; flow tracks; trajectory shaped like the validation GPS track.
  - S5: tag outlines on the markers.
  - S6: sparse landmark cluster with large ellipsoids.
- No covariance inversion anywhere in project source, which the user required. Checked with a grep audit, and
  regression tests compare against explicit-inverse references that exist only inside the tests.
- No GPS in the estimator. A test shows that two SRT files differing only in lat/lon give bitwise-identical estimates.
- The review Peer ran an ASan + UBSan Debug suite with 0 reports. That run predates the S6 rework (see §5).
- Fixed defects that had evidence:
  - BufferedVideo lost-wakeup races.
  - Tiny-dt information blow-up (C1).
  - Non-square-root Laplace update (C2).
  - HDF5 leaking `NDEBUG` into Debug builds.
  - S4 horizon bias (P2).
  - The S4 synthetic-renderer aliasing (P1) was a test bug; the tracker was unbiased.
  - S6 translation-sign flips (behind-camera triangulations 54 % → 6.7 %).
  - Plot slot leak; X/Mesa render slowness.
  - Two covariance inversions in S6 gates and one in the horizon classifier.

## 3. ASSUMED (no specification available; `project.txt` is a one-line stub)

- Overlay colours and draw order for all scenarios, and the 3-D view colours (`architecture.md` §7–11).
- Altimeter: relative to the first SRT sample, event-driven at about 5 Hz, sigma 0.02 m.
  Its NIS-calibrated value was rejected out-of-sample.
- S4 ground plane Hg = first SRT altitude; initial heading 0 (unobservable).
- S5 tag side 0.15 m; metric scale is proportional to it.
- S6 speed prior 0.5 ± 0.25 m/s; monocular scale is fixed only by this prior.
- Body-to-camera extrinsic: axis permutation, zero lever arm.
- Calibration model: principal point fixed at the image centre, 5 distortion coefficients (evidence-based decision).
- Visual-horizon training patches cropped from `flight.MOV`; no sample images were supplied.

## 4. USER decisions applied

- B1: autodiff 1.1.2 in `CMakeLists.txt`; 1.1.3 does not exist.
- B2: Boost 1.88.0; VTK 9.5.2 requires exactly that version.
- B3: hand-written, FD-validated Jacobians and Hessians; autodiff's Eigen-5 header is broken and unused.
- S6: one cheirality root-cause round, then report INADEQUATE if still inadequate. Done: INADEQUATE.
- No covariance inversion anywhere, including the S5 re-identification gate. Done.

## 5. Unresolved issues and limitations

1. **S6 room map is INADEQUATE.**
   - Mean map size is 8.4 landmarks, median landmark lifetime 7 frames, and there is no recognisable room structure.
   - The walking-camera map test fails.
   - Scale is set only by an assumed speed prior, which the data do not support (median 0.08 m/s).
   - Likely next steps (not done, per user rule): longer-lived tracks with descriptor re-identification across
     keyframes, a better birth policy, and window/curtain texture rejection.
2. **S4 NIS inconsistency.** Flow is far too conservative (NIS/dof about 0.008) and the altimeter too tight
   (about 1.8). Calibrated sigmas failed out-of-sample checks, because the measurement errors are correlated.
   Heading and horizontal position are dead-reckoned. Validation-only GPS: similarity RMSE 34.8 m, rigid best-yaw
   RMSE 56.8 m, end-to-end displacement +4 %.
3. **S5 identity churn.** 39 identities, 6 tags at the end, with no ground-truth tag count. NIS is 82.9 % in band
   out-of-sample, against an expected 95 %.
4. Not real time. On this laptop, Release runs vary by about ±40 % between runs (powersave governor, thermals).
   The cost of the square-root classifier change was not isolated.
5. Review LOW items left open: R4-18 (onnxruntime linked but unused, kept per spec), part of R4-19
   (`-Wextra`/`-Wconversion` sites), R4-20 (S4 synthetic tolerances kept as written).
6. CLI: `-s 5` with a space is rejected by `cv::CommandLineParser`. Use `--scenario=5` or `-s=5` (documented).
7. The ASan/UBSan run predates the S6 rework, the R5 fixes and the Lead's square-root changes. It was not repeated.

## 6. UNTESTED / NOT RUN

- **Clean-machine install: UNTESTED.** Only the reference machine was used; no container or second machine.
  Portability is not claimed.
- **Doxygen (`-DBUILD_DOCUMENTATION=ON`): NOT RUN.**
- Interactive modes 1 and 2: smoke-tested only; the mode-2 key advance was not exercised.
- Machines without an EGL-capable NVIDIA GPU: offscreen performance not measured.
- No ground truth for any trajectory or map.

## 7. Where things are

- Reports: `engineering-reports/{plan,architecture,test-report,code-review,performance,final-report}.md`, `status.json`.
- Scripts: `scripts/{setup-env,env,verify-env,build,test}.sh`; README.md describes installation and running.
- Exports and timings: `out/{flight,tag_identical,room}_out.MOV`, `out/*_timing.csv`.
- Diagnostics: `out/scenario{4,5,6}_r5_*`.
