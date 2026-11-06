# Code review: Phase 4 five-pass independent review (T-A2-001 / M4-REVIEW)

- Reviewer: independent Peer (read-only on source, tests, CMake, scripts and data; scratch builds only).
- Date: 2026-11-06.
- Source state: all non-S6 files were byte-identical to the live tree at snapshot time (about 17:09) and re-checked at about 17:45. S6 files changed during the review (Peer rework) and are reviewed only at interface/safety level (section 4).
- Scratch trees (deleted at the end): an ASan+UBSan Debug build of the source snapshot and a copy of the flight clip. No file under `src/`, `test/`, `CMakeLists.txt`, `scripts/`, `data/` or `out/` was written.

## 1. Summary

Verdict: **no CRITICAL and no HIGH finding is supported by evidence.** The core estimator mathematics (hand-written Jacobians, process noise, time update, Laplace update, information-form primitives, camera model) passes independent checks at 1e-10 or better. The defects found are robustness, statistical-consistency, validation-methodology and specification-traceability issues, rated MEDIUM at most. One known S6 defect (reported by the Lead, S6 under rework) is carried but was not re-verified (section 4).

Counts by severity and specialist (22 findings):

| Specialist    | CRITICAL | HIGH | MEDIUM | LOW | Total |
|---------------|---------:|-----:|-------:|----:|------:|
| Mathematics   | 0 | 0 | 2 | 0 | 2 |
| Numerical     | 0 | 0 | 3 | 3 | 6 |
| C++23         | 0 | 0 | 1 | 3 | 4 |
| Performance   | 0 | 0 | 0 | 1 | 1 |
| Specification | 0 | 0 | 4 | 5 | 9 |
| **Total**     | **0** | **0** | **10** | **12** | **22** |

Primary owners: core 4 (R4-02, R4-15, R4-16, R4-17); S4 4 (R4-06, R4-07, R4-11, R4-20); S5 2 (R4-09, R4-10); S6 2 (R4-21, R4-22); VIS 1 (R4-12); Lead 7 (R4-03, R4-04, R4-05, R4-08, R4-13, R4-14, R4-18); multi-owner 2 (R4-01: core with S4/S5/S6 call sites; R4-19: VIS/core/S5/Lead per file).

Scope note: the requested deep review of S6 was not performed because S6 is under rework. The S6 items are listed only at interface/safety level and are marked "S6 under rework - re-review needed".

## 2. Evidence base (what was actually run)

- ASan+UBSan Debug build of the snapshot (`-fsanitize=address,undefined -fno-sanitize-recover=undefined`, `EIGEN_INITIALIZE_MATRICES_BY_NAN` from CMake). The `run_tests` target ran during the build: `[doctest] test cases: 201 | 201 passed | 0 failed`, `assertions: 7616 | 7616 passed`, `Test #1: unit_tests Passed 153.36 sec`. Log grep for `runtime error`, `AddressSanitizer`, `LeakSanitizer`: 0 matches.
- Extra-warnings syntax-only pass (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wno-unused-parameter`) over 74 project translation units: 0 errors, 14 project warnings (list in R4-19). The total is a lower bound: some `src/` and `test/src/` files share basenames and the per-file logs overwrote each other.
- Mutation check: sign of the flow-bundle derivative term `d(1/d_A)/d r_pz` flipped in a scratch copy and the tests rebuilt. Filtered run (`*Outdoor*,*Horizon*,*Scenario 4*`): 24 cases, 2 failed, 9 failed assertions of 662. The suite detects this real derivative error. (The scratch copy was restored and the library rebuilt; the mutant is not in the source tree.)
- Independent probe (scratch, linked against the project static library with the project's own flags). All reference quantities re-derived explicitly, not by calling the project's tests:
  - A: dynamics Jacobian vs central FD, 50 random states, |theta| to 1.2 rad: max rel err 1.9e-10.
  - B: process-noise factor: `L^T L Qd - I` with `Qd = G Cz G^T` built explicitly: 3.4e-14.
  - C1: flow prediction Jacobian (80x12) vs central FD: 8.9e-11. C2: flow log-likelihood gradient vs FD: 3.8e-11.
  - C3 (realistic residuals): GN Hessian vs FD of gradient, relative gap 7.7e-6 (0.25 px), 8.7e-5 (2 px), 2.2e-4 (10 px).
  - D1: horizon residual Jacobian vs FD: 4.1e-11.
  - E1-E4: GaussianInfo marginal and conditional mean/covariance vs explicit Schur algebra: <= 1e-15 (absolute).
  - F1-F2: linearised transition (time update) vs `Phi P Phi^T + Qd` and `Phi mu + c`: <= 1.2e-15 (absolute).
  - G: trust-region with an objective that throws at a trial point: the exception escapes (R4-02).
- Camera (`data/outdoor/camera.xml`, 2688x1512, 5 coefficients): pixel -> normalised -> pixel round trip max 4.7e-13 px; `vectorToPixel` identical to `cv::projectPoints` (same model); `pixelToNormalised` matches `cv::undistortPoints` to 3.6e-12.
- S4 short clip (150 frames, first 150 SRT blocks, ASan+UBSan Debug build, `--scenario=4 --interactive=0 --export`): exit 0, 0 sanitizer lines. Export reopened with ffprobe: mpeg4, 5376x1512 (2W x H), 150 frames, 2997/50 fps. These timings are Debug+ASan and are not performance results.
- Read-only analysis of the Lead's existing full-flight S4 CSV and summary (`out/scenario4_round2_flight_run.csv`, `out/scenario4_round2_flight_run_summary.txt`) and S5 summary (`out/scenario5_final_run_summary.txt`). No full-length runs were performed by the reviewer.
- Static audits: covariance/information inversion grep, lat/lon grep, assert grep, numeric-parse grep, try/catch structure, BufferedVideo locking and copy semantics, CLI parsing (`-s 5`, `-s=5`, `-s5`, `--scenario=5`, `-i 2`, `-s 7`, `--scenario=7`).

## 3. Findings

Each entry: ID, severity, specialist, location, description, reproduction/evidence, impact, proposed fix, validation, owner.

### R4-01 MEDIUM - Numerical (Mathematics secondary) - core, with S4/S5/S6 call sites
- Location: `src/funcmin.cpp:248` (returns `MaxIterations`); `src/MeasurementBase.cpp:152` and `:190` (only `NumericalFailure` handled); `src/OutdoorLaplaceUpdate.cpp:102-103` (status ignored; only non-finite throws at line 115); `src/SlamPoseTracker.cpp:301, 402`; `src/SlamPointPipeline.cpp:497, 525, 576`.
- Description: The MAP status is consumed only as `== NumericalFailure`. A `MaxIterations` (non-stationary) result is applied to the posterior as if it were the MAP. The tests assert only `!= NumericalFailure` (for example `test/src/Measurement.cpp:125`, `test/src/CoreRegressionTests.cpp:291`), so non-convergence is never tested. The S4 pipeline does not log the MAP status per frame, so the rate of non-convergence in real runs is unknown.
- Reproduction: code inspection; no test asserts convergence status. Counts not measured (NOT_RUN).
- Impact: a non-converged MAP yields a posterior that is not the Laplace posterior at the mode, silently. The consistency of the Laplace step (posterior mean equals MAP) holds only at stationarity.
- Proposed fix: (1) treat `MaxIterations` as a diagnosed outcome (count and log per frame in S4/S5/S6 diagnostics); (2) add a test asserting `GradientConverged` or `StepConverged` on the representative measurements, and a test with `maxIterations = 1` asserting `MaxIterations` is reported; (3) decide, and document, whether a non-converged update is applied or reverted.
- Validation: unit tests above; on a 150-frame S4 clip and the S5 video, record `map.status` counts (NOT_RUN by reviewer).
- Owner: core (MeasurementBase, OutdoorLaplaceUpdate, funcmin); S4, S5, S6 to record statuses.

### R4-02 MEDIUM - C++23 (Numerical) - core
- Location: `src/funcmin.cpp:185` (`const double fTrial = f(xTrial, nullptr, nullptr);` not guarded); thrown from `src/MeasurementOutdoorFlowBundle.cpp` (`predict` and `logLikelihood` throw `std::domain_error` on `q(2) <= 0`; `homography` throws when `d_A <= 0`); caught only at `src/Scenario4.cpp:452-455` and `:493-496`.
- Description: an exception thrown by the objective at a trial point escapes `trustRegionMinimise` and aborts the whole MAP, instead of being treated as a rejected step (as non-finite values are, `funcmin.cpp` lines 188-194). The S5 tag likelihood instead returns NaN for points behind the camera (`Camera::vectorToPixel`, `Camera.cpp:404`), which the trust region handles. The error contract is therefore inconsistent between scenarios.
- Reproduction: probe G (scratch): an objective `(x-3)^2` that throws `std::domain_error` for `x > 0.5`, started at `x0 = 0.4` with radius 10. Output: `G. trust region with domain violation at a trial step: EXCEPTION ESCAPES (behind camera ...)`.
- Impact: one trial step outside the model domain discards the whole frame's flow update (S4 "update skipped" note). Not observed on the real flight (flow update fraction 0.9999 in the Lead's summary), so no wrong result is evidenced.
- Proposed fix: in `trustRegionMinimise`, catch `std::exception` (or the domain exceptions) around trial evaluations and treat it as a non-finite trial (radius shrink). Alternatively return NaN from the flow model, as the camera does. Add the probe G scenario as a regression test.
- Validation: probe G must return a finite result with the trial rejected; the existing funcmin and outdoor-flow tests must still pass.
- Owner: core (funcmin); S4 to confirm the update is no longer skipped.

### R4-03 MEDIUM - Specification (test honesty) - Lead
- Location: `test/src/CameraModel.cpp:412-416` (`MESSAGE("skipped, missing: ...")` then `continue`); `test/src/DJIVideoCaption.cpp:213-216` (`WARN_MESSAGE(false, ...)` then `return`).
- Description: when a fixture is absent the test body does not fail; doctest counts it as passed. A missing data file therefore reports PASS rather than NOT_RUN, which conflicts with the "honest NOT_RUN labelling" requirement.
- Reproduction: code inspection (both paths exit without a failed CHECK). In this environment the fixtures are present, so the suite does not hit them.
- Impact: a fresh clone without `data/*.MOV` or `data/outdoor/flight.SRT` would show a green suite with unexecuted checks.
- Proposed fix: use `doctest::skip` (or an explicit `FAIL` for required fixtures), or a documented NOT_RUN counter printed at the end of the run.
- Validation: move `data/outdoor/flight.SRT` aside in a scratch copy and confirm the run reports skipped or failed, not passed.
- Owner: Lead (test infrastructure).

### R4-04 MEDIUM - Specification (documentation) - Lead
- Location: `README.md:10-11` ("does not configure yet"), `:53-61` ("both Debug and Release fail at configure time"), `:65-67` ("suite contains 0 test cases"), `:90-91` ("Not runnable yet").
- Description: the README is an environment-setup document that is now stale: CMakeLists uses autodiff 1.1.2 (recorded decision), the project configures and builds, and the suite has 201 test cases. It documents no project usage, scenario behaviour, CLI forms or ASSUMED values.
- Reproduction: `CMakeLists.txt:61` finds autodiff 1.1.2; ctest reports 201 cases (see section 2).
- Impact: misleading for any reviewer or grader; the deliverable has no usage documentation.
- Proposed fix: rewrite the README status to the current state, add the Scenario 4/5/6 run commands, the accepted CLI forms (R4-14), the ASSUMED-values table and the NOT_RUN list.
- Validation: each command in the README is executed once in a scratch copy.
- Owner: Lead.

### R4-05 MEDIUM - Specification (traceability) - Lead
- Location: `project.txt` (one line: `a2`); colours, overlay order and altitude update frequency marked ASSUMED in `src/Scenario4.cpp:579, 586, 595, 627`, `src/Scenario5.cpp` (`drawOverlay`, status line "ASSUMED overlay colours"), `src/Scenario4Runner.h:60`.
- Description: the repository does not contain the authoritative scenario specification (AGENTS.md refers to `project.txt`). Overlay order, colours, the altitude reference and "update frequency specified in project.txt" therefore cannot be verified against a written source; they are recorded as ASSUMED.
- Reproduction: `cat project.txt` shows only the project name.
- Impact: the Specification pass can check only internal consistency, not compliance with the assignment wording.
- Proposed fix: the Lead should place the authoritative requirement text (or a link/extract) in the repo, and map each ASSUMED visual or timing item to it, or record it as unverified in the final report.
- Validation: each ASSUMED item has a requirement reference or is listed as UNKNOWN in `final-report.md`.
- Owner: Lead.

### R4-06 MEDIUM - Mathematics (statistical consistency) - S4
- Location: `src/Scenario4Runner.h:60` (`altitudeSigmaM = 0.02 ... ASSUMED, tuned by NIS`); evidence `out/scenario4_round2_flight_run_summary.txt` and `out/scenario4_round2_flight_run.csv`.
- Description: the summary reports altitude NIS mean 4.14, median 1.82, and only 75.3% of frames inside the central 95% chi-squared band (dof 1, so about 95% expected). My analysis of the CSV: median altimeter innovation |y - h| = 0.050 m, p95 0.141 m, max 0.408 m (sigma 0.020 m). The "tuned by NIS" label is therefore not supported by the Lead's own NIS statistics: sigma 0.02 m is inconsistent with the innovations.
- Impact: the altimeter is over-weighted relative to the prior, and altitude feeds the AGL that sets the flow scale (`Scenario4.cpp:425`). Post-update altitude RMS against the SRT is 0.023 m over the 760 update instants, so the error is not large at the update instants, but the uncertainty is under-stated.
- Proposed fix: either re-tune sigma so that NIS is about 1 (about 0.05 m from these innovations) and record the value and evidence, or relabel as ASSUMED-inconsistent and report the consequence. Add a synthetic NIS-consistency test for the altimeter.
- Validation: NIS mean and in-band fraction on the full flight after the change (NOT_RUN; Lead to run).
- Owner: S4 (decision: Lead).

### R4-07 MEDIUM - Numerical (statistical consistency) - S4
- Location: `src/Scenario4Runner.h:55` (`flowSigmaPx = 1.0 ... tuned`); `src/Scenario4.cpp:438`; evidence `out/scenario4_round2_flight_run_summary.txt`.
- Description: flow NIS/dof median 0.0084 (mean 0.0101) with 0% of frames inside the 95% band. Post-update flow residual RMS is 0.095 px against sigma 1.0 px, so the flow noise model is about 10x conservative relative to the realised residuals.
- Impact: flow information is under-used (about 100x less precision than the residual scatter implies), so flow contributes little; the chosen value is a conservative choice rather than a consistent one. Correlated Lucas-Kanade errors across about 400 features argue for some inflation, but the factor should be documented.
- Proposed fix: decide and document the flow noise model (for example per-feature sigma with an inflation factor for correlation), and report the NIS target alongside the tuned value.
- Validation: NIS/dof on the full flight and on the synthetic test (NOT_RUN for the full flight).
- Owner: S4 (decision: Lead).

### R4-08 MEDIUM - Specification (validation methodology) - Lead
- Location: validation metrics in `out/scenario4_round2_flight_run_summary.txt` (altitude RMSE 0.1547 m, max 0.9376 m; path length 2415 m vs 1832 m, ratio 1.318; yaw-aligned RMSE 56.78 m); the CSV columns `srt_h_rel_m` and `gps_*`.
- Description, altitude: `srt_h_rel_m` is a sample-and-hold reference (updated only at value changes). The estimate is continuous. During the climb at about 3.7 m/s (122.5-127.5 s; SRT rises 18.5 m in 5 s) the estimate runs ahead of the staircase between SRT changes; the maximum 0.94 m error (frame 7506, t = 125.23 s) is this artefact. At the 760 update instants the post-update RMS is 0.023 m; at t = 124.22 s the estimate is 18.161 m against the SRT value 18.158 m (3 mm). The reported RMSE and max therefore overstate the altitude error.
- Description, path length: the path-length ratio 1.318 compares sums of frame-to-frame steps, which include high-frequency estimate jitter. End-to-end displacement agrees: estimate 828.5 m vs GPS 795.4 m (4%). A similarity fit (validation only) gives scale 0.888 with RMSE 31.1 m (scale fixed at 1: 43.9 m with the same rotation). The ratio is not evidence of a scale bias on its own; the candidate cause (ASSUMED ground depth Hg, `Scenario4.cpp:162`) remains untested.
- Impact: misleading acceptance metrics; risk of a wrong conclusion about scale or altitude accuracy.
- Proposed fix: report altitude error at update instants (or against an interpolated reference), report RMSE after similarity alignment (scale and rotation estimated, validation only), and drop the path-length ratio as evidence. Run a sensitivity check on Hg (for example +/-10%) to test the ASSUMED ground depth.
- Validation: the recomputed metrics from the existing CSV (reproducible with the reviewer's script, not stored in the repo).
- Owner: Lead (validation); S4 (Hg sensitivity).

### R4-09 MEDIUM - Numerical (specification: no covariance inversion) - S5
- Location: `src/SlamPoseTracker.cpp:79-101` (`tagArchiveDistanceSq`: `P = SqrtCov^T SqrtCov` at line 91, `LLT(P)` at 96, `llt.solve` at 101); called at line 673 for archive re-identification.
- Description: the re-identification gate forms the covariance explicitly and inverts it by Cholesky. This is outside the estimator update, but the user requirement reads "no explicit covariance inversion in the filter path"; the gate uses the archived filter covariance.
- Impact: none observed on the S5 video (mapFailures 0; re-identified births 3). It is a requirement-compliance and numerical-conditioning issue (forming `P` squares the condition number).
- Proposed fix: compute the squared Mahalanobis distance by a triangular solve with the stored square-root factor: `d^2 = || S^{-T} delta ||^2`, with the floors added as extra rows of a QR factor, so no `P` is formed.
- Validation: unit test that the new function matches the current one to 1e-10 on random archives; the S5 short-clip run should give identical births and culls (NOT_RUN).
- Owner: S5.

### R4-10 MEDIUM - Mathematics (consistency and evidence) - S5
- Location: `out/scenario5_final_run_summary.txt` (flow ON section: `NIS/dof ... inside 95% chi2 band 556 / 996 (55.8 %)`; `total births 38  culls 33  final 5  identities 35`).
- Description: the tag NIS is inconsistent with the 95% band (55.8% in band, mean 0.667), in the same direction as the S4 flow (over-conservative). The identity churn (38 births, 33 culls, 35 identities) is large for a scene with a handful of tags, but no ground-truth tag count is recorded, so churn cannot be judged as error or as correct behaviour. The summary timing is labelled "NOT a clean benchmark" (machine load), which is honest.
- Impact: the S5 identity and uncertainty results are not validated against truth; the NIS level is not consistent.
- Proposed fix: record ground-truth (or manually labelled) tag identities for a short segment and compute identity-switch and track-continuity rates; check the tag sigma against the NIS target.
- Validation: labelled-segment metrics (NOT_RUN).
- Owner: S5.

### R4-11 LOW - Performance - S4
- Location: `src/Scenario4.cpp:445, 485, 526` (`supportInformation(...)`, which builds the marginal and its information matrix); `src/Scenario4.cpp:41-52` (`nisFromInnovation`, LDLT per update).
- Description: NIS diagnostics run unconditionally in the production frame loop. They are "diagnostics only" but form an information matrix and an LDLT per update.
- Impact: small at n = 18 (the Lead's Release timings show per-frame cost dominated by flow and horizon), but unnecessary work in production and an explicit matrix factorisation of the type the requirement discourages.
- Proposed fix: gate the diagnostics behind an option (default off for timing runs) or compute them from the already-available QR factors.
- Validation: Release timing before and after (NOT_RUN by reviewer).
- Owner: S4.

### R4-12 LOW - C++23 (thread robustness) - VIS
- Location: `src/BufferedVideo.cpp:33` and `:128-132` (`start()` assigns a new `std::thread` to a possibly joinable member, which calls `std::terminate` if called twice); `src/BufferedVideo.cpp:192` (`video.write(frame)` on the worker thread, not guarded; an exception from OpenCV would terminate).
- Description: the threading architecture is otherwise sound: lost-wakeup-safe notifications, `write()` clones the frame (`:168`), `stop()` joins and releases. These two paths are not reachable in the current call sites (`visualNavigation.cpp` starts each buffer once).
- Impact: defence-in-depth only.
- Proposed fix: guard `start()` against double start (throw `std::logic_error`), and catch and record exceptions in the worker thread, rethrowing on the next `write()` or `stop()`.
- Validation: a unit test that double-start throws; a writer whose `write()` throws is reported, not terminated.
- Owner: VIS.

### R4-13 LOW - C++23 (error handling) - Lead
- Location: `src/main.cpp:82` (`catch (const std::exception & e)` only).
- Description: a non-standard exception escapes `main` and terminates the process.
- Impact: low (all project code throws std exceptions).
- Proposed fix: add `catch (...)` that prints a generic error and returns `EXIT_FAILURE`.
- Validation: unit or CLI test is optional.
- Owner: Lead.

### R4-14 LOW - Specification (CLI) - Lead
- Location: `src/main.cpp:18-26` (keys), `:40` (parser check).
- Description: OpenCV's parser accepts `-s=5`, `-s5` and `--scenario=5` (and `-i=2`, `--interactive=2`). It rejects `-s 5` and `-i 2` (space-separated) with a usage dump and exit 1 and no specific message: observed `./a2 -s 5 data/indoor/room.MOV` prints `Usage: a2 [params] input` and returns 1. AGENTS.md documents `-s|--scenario=4|5|6` and `--scenario=4` as the examples, so the `=` forms are the documented ones.
- Impact: usability; the documented examples work. Not a spec violation on the evidence.
- Proposed fix: document the accepted forms in README (R4-04) and print an explicit error for a missing value.
- Validation: `./a2 -s 5 ...` prints a clear message (optional).
- Owner: Lead.

### R4-15 LOW - Specification (decision drift) - core (calibration)
- Location: `src/calibrate.cpp:1074-1078` (cross-validated switch to the 8-coefficient rational model when it beats the standard 5-coefficient model by `modelSwitchMargin`).
- Description: the recorded decision is "5-coefficient distortion". The code can select 8 coefficients automatically. The shipped `data/outdoor/camera.xml` and `data/indoor/camera.xml` both use `opencv-standard-5` (verified), so the switch is not triggered on the shipped data.
- Impact: the decision can be silently overridden by a different calibration video.
- Proposed fix: gate the rational model behind an explicit option (default off), or record the decision as superseded in `decisions.md`.
- Validation: calibration run on the shipped video reproduces the standard-5 output (NOT_RUN by reviewer).
- Owner: core (calibrate).

### R4-16 LOW - Numerical - core
- Location: `src/MeasurementBase.cpp:197-212` (absolute floor `scale = max(1, max|lambda|)`; `if (r == 0)` at 209 applies `setState(prior)`); `precisionIsExact` remains at its default `true` on that branch (`:209-212`).
- Description: when the Laplace curvature has no eigenvalue above `1e-10 * max(1, ...)`, the measurement is silently discarded and the result reports `precisionIsExact = true`. A measurement with information below about 1e-10 is dropped without a flag.
- Impact: not reached with the current sigma values (altimeter 0.02 m, flow 1 px). Could hide a mis-scaled measurement.
- Proposed fix: add an `updateSkipped` flag to `MeasurementUpdateResult` and set `precisionIsExact = false` (or the new flag) on this branch.
- Validation: unit test with sigma 1e5 asserting the flag.
- Owner: core.

### R4-17 LOW - Numerical - core
- Location: `src/rotation.hpp:137` (`kinematicMatrixTK`, `1/cos(theta)` term); `src/SystemVisualNav.cpp:102` (`kinematicMatrixTK(theta).inverse()`); `src/SystemVisualNav.h` documents the restriction |theta| < 90 deg.
- Description: the singularity at pitch = +/-90 deg is documented but not checked at runtime. A pitch near 90 deg would produce large or non-finite `TK` and `T^{-1}` and thus a non-finite process noise factor (the estimator then throws only through `allFinite` checks at frame level).
- Impact: not reached in the flight or indoor data (the Lead's summary shows finite states throughout). Robustness only.
- Proposed fix: validate `|theta_pitch| < 85 deg` in `processNoiseSqrtInfo` and throw a documented error, or clamp.
- Validation: unit test with pitch 89.9 deg asserting the documented error.
- Owner: core.

### R4-18 LOW - Specification (build hygiene) - Lead
- Location: `CMakeLists.txt:20` (`add_compile_options(-Wall) # -Wextra -Wpedantic`); `CMakeLists.txt:77` and `:127` (`find_package(onnxruntime REQUIRED)`, linked; `grep -l onnx src/* test/src/*` returns nothing).
- Description: the warning level is the minimum (`-Wall` only). The onnxruntime dependency is required and linked but not used by any source file.
- Impact: unused dependency increases configure fragility (the README shows onnxruntime was a previously reported environment issue); low warning level hides the R4-19 items.
- Proposed fix: enable `-Wextra -Wpedantic` (fix or justify the 14 items) and remove the onnxruntime requirement unless a planned use exists (AGENTS.md: avoid unnecessary dependencies).
- Validation: configure and build with the new flags; `grep` for onnx.
- Owner: Lead.

### R4-19 LOW - C++23 (warnings) - VIS / core / S5 / Lead (per file)
- Locations and warnings (extra flags):
  - `src/BufferedVideo.cpp:14, 113, 197` (`-Wsign-conversion`, int -> size_t).
  - `src/calibrate.cpp:531, 535, 536, 699, 1370` (`-Wconversion`, size_t -> double).
  - `src/Plot.cpp:448` (`-Wfloat-conversion`, double -> float).
  - `src/Plot.cpp:1132` (`-Wsign-conversion`).
  - `test/src/ImageFlowTests.cpp:202` (`-Wconversion`).
  - `test/src/SlamPipeline.cpp:72, 79, 80` (`-Wdeprecated-declarations`, OpenCV comma initialiser).
- Impact: none of these changes behaviour for the sizes used; they are the warnings the requirement would expect to be clean.
- Proposed fix: explicit `static_cast`s and `cv::Matx` or `at` initialisation instead of the deprecated comma initialiser.
- Validation: rebuild with the flags; test suite unchanged.
- Owner: VIS (BufferedVideo, Plot); core (calibrate); Lead (ImageFlowTests); S5 (SlamPipeline, test).

### R4-20 LOW - Specification (test strength) - S4 / Lead
- Location: `test/src/Scenario4Tests.cpp:490-530` (synthetic flight: altitude tolerance 0.5 m, horizontal 1.5 m); `test/src/Scenario4Tests.cpp:446-488` (source-text search for "latitude"/"longitude"/"validationOnly" in `src/*.cpp`).
- Description: the synthetic tolerances are loose relative to the altimeter sigma (0.02 m); a tighter bound would catch more regressions. The source-text tests are weak as proof of "no GPS access", but they are complemented by the behavioural test at `Scenario4Tests.cpp:405-443` (bitwise-identical trajectories for two SRT files with different lat/lon), which is strong and passes.
- Impact: regressions in accuracy under about 0.5 m would not be detected by the test.
- Proposed fix: tighten the synthetic tolerances to values derived from the rendered truth and the altimeter sigma; keep the behavioural lat/lon test as the primary guard.
- Validation: the tightened tests pass on the current code and fail on a seeded 0.5 m altitude bias.
- Owner: S4 (tests), Lead (review).

### R4-21 LOW - Specification (artefact hygiene) - S6 / Lead
- Location: `test/src/S6ZZDebugTemp.cpp` (new in the S6 rework; a temporary debug file in the test suite); `out/scenario6_*` (pre-rework S6 outputs).
- Description: a debug file in the test suite will be compiled and run with the production tests. The `out/scenario6_*` files pre-date the S6 rework and must not be cited as evidence for the reworked S6.
- Proposed fix: remove `S6ZZDebugTemp.cpp` before the final review; regenerate or label the S6 outputs.
- Validation: `find test/src -name '*ZZ*'` is empty; the S6 summary cites the regenerated outputs only.
- Owner: S6 (Lead to check).

### R4-22 LOW - Numerical (S6 under rework - re-review needed) - S6
- Location: `src/SlamAssociation.cpp:99` (`const Eigen::Matrix2d sInv = s.inverse();`, gate only); `src/SlamInitialisation.cpp:113` (`lS = dInv * jn.inverse()`, explicit 3x3 inverse of a Jacobian to form the noise factor).
- Description: the first is an explicit inverse of a 2x2 innovation covariance for gating (same class of issue as R4-09). The second inverts a 3x3 transform (not a covariance) and the commented derivation (noise factor, not filter information) is correct in form.
- Proposed fix: replace the gate with a triangular solve; the noise-factor inverse is acceptable and may remain.
- Validation: as R4-09.
- Owner: S6 (re-review required after the rework).

## 4. S6 under rework - re-review needed

Interface and safety level only (S6 files are being rewritten):
- Carried from the Lead's report, not re-verified by me: translation absorbed as rotation; depth never resolved. If still present at the end of the rework this is HIGH (wrong estimates) and should be re-reviewed in full.
- MAP status handling (R4-01): `src/SlamPointPipeline.cpp:497, 525, 576` check only `NumericalFailure`.
- Explicit inverses (R4-22): `src/SlamAssociation.cpp:99`, `src/SlamInitialisation.cpp:113`.
- Temporary debug file `test/src/S6ZZDebugTemp.cpp` (R4-21).
- New files from the rework (not reviewed): `src/SlamTrackKeyframes.{h,cpp}`, `src/SlamTrackTriangulation.{h,cpp}`, `test/src/SlamTrackKeyframes.cpp`, `test/src/SlamTrackTriangulation.cpp`, `test/src/S6WalkingCamera.cpp`.

## 5. Verified OK (evidence-backed absence of findings)

- Covariance and information handling in the estimator path: no `.inverse()`, `LLT`, `LDLT` or `solve` on a covariance or information matrix in `GaussianInfo`, `SystemEstimator`, `SystemVisualNav`, `MeasurementBase` (square-root paths) or `MeasurementOutdoorFlowBundle`/`MeasurementHorizon`/`MeasurementAltimeter`. Updates use Q-less Householder QR (`core::triangularise`) and triangular solves. The explicit paths are: `GaussianInfo::covariance`/`sqrtCov` (reference only; `affineTransform`/`nonlinearTransform` are called only from tests), the documented indefinite fallback (`MeasurementBase.cpp:234`, flagged `precisionIsExact = false`, not reached in production), and the diagnostics (R4-11).
- Math identities (probe A-F above): dynamics Jacobian including the Euler-kinematics derivative; process-noise `L^T L = Qd^{-1}` with `Qd = G Cz G^T`; time update against explicit linear-Gaussian algebra; marginal and conditional against Schur complements; flow and horizon Jacobians; GN Hessian gap at operating residuals.
- Homography and plane geometry: `H = R_BA + t n_A^T/d_A`, `n_A` and `d_A` derivations and the altimeter convention (`h = -z`, `z(t0) = 0`, `AGL = Hg - z = h0 + h_rel`) are consistent (`MeasurementOutdoorFlowBundle.cpp`, `Scenario4.cpp:425`).
- Horizon residual `r = a.p / ||(a1,a2)||`, its derivative, and the overlay line equation are consistent.
- FD-based and mutation-based test strength: the outdoor-flow FD tests catch a real sign error in a derivative (section 2).
- Lat/lon never reach the estimator: `Scenario4.cpp:150-186` uses lat/lon only to build `validationLocal` (validation track); `MeasurementAltimeter.cpp`, `MeasurementHorizon.cpp`, `MeasurementOutdoorFlowBundle.cpp`, `VisualHorizon.cpp`, `ImageFlow.cpp` contain no latitude or longitude; the behavioural test (`Scenario4Tests.cpp:405-443`) passes.
- SRT altitude timing: 9160 blocks, one block per frame; altitude-change events (760) land within 0.97 ms of the first frame at or after the event (computed from `data/outdoor/flight.SRT`). The event-driven altimeter therefore has no material latency.
- Delayed-state and keyframe bookkeeping (S4): `Scenario4.cpp:377-388` (shift at keyframe+1) and `:538-547` (keyframe refresh) are consistent with the flow baseline; the S5 sequence (`SlamPoseTracker.cpp:264-290`) is consistent.
- Exception and thread handling in the frame loop: `visualNavigation.cpp` (try/catch around the loop stops both buffers and rethrows; runner and visualisation errors are re-thrown with frame index); `BufferedVideo.cpp` lost-wakeup-safe; `write()` clones the frame (no race on the reused `imgout`).
- Assertions: no `assert(` in `src/` (grep), so Debug and Release run the same validation (runtime checks throughout).
- Numeric parsing: `DJIVideoCaption.cpp:45, 67` uses `std::from_chars` with error checks.
- CLI: documented flags, `--help`, runtime validation of scenario and interactive values (`main.cpp:69-77`); `-s=5`, `-s5`, `--scenario=5`, `--scenario=7` (error message) behave as documented (R4-14).
- Export: `<stem>_out<ext>` naming and 2W x H size (clip: 5376x1512, 150 frames).
- Calibration: shipped `camera.xml` files load and use the standard 5-coefficient model; the camera model matches OpenCV to machine precision.
- Scripts: `scripts/build.sh`, `test.sh`, `env.sh` use `set -euo pipefail` and no destructive commands; `setup-env.sh` has the documented `--force` guard for an existing prefix.
- CMake: `NDEBUG`/`EIGEN_NO_DEBUG` in Release, `EIGEN_INITIALIZE_MATRICES_BY_NAN` in Debug, HDF5 flags stripped (consistent with the recorded decisions).

## 6. NOT_RUN and not verified

- Release-build test suite: not re-run by the reviewer (Lead reports 201/201).
- Full-length S4, S5 and S6 runs: not run by the reviewer. S4 and S5 analysis is read-only on the Lead's existing outputs. S6 not run (under rework).
- MAP status counts in real runs (R4-01): not measured.
- Flow and altimeter NIS after any re-tuning (R4-06, R4-07): not measured.
- Hg (ground depth) sensitivity (R4-08): not run.
- S5 identity ground truth (R4-10): not available.
- Performance (Release): not re-measured.
- Interactive display modes (`--interactive=1|2`) and VTK visual correctness of overlays and colours: not exercised; colours are ASSUMED (R4-05).
- Static analysis beyond the compiler warnings (clang-tidy, cppcheck, Doxygen): not run.
- The S6 known defect (translation absorbed as rotation, depth unresolved): not reproduced by the reviewer (S6 under rework).
- Reviewer scratch artefacts were deleted at the end (build trees, probe binaries, clip). The probe sources and logs were kept in the scratch directory only.

## 7. Resolutions (round R5, after this review; verified on the integrated tree by the Lead)

Integrated-tree evidence after all fixes:
- Debug (asserts live): 244 test cases, 243 pass, 8514/8515 assertions.
- Release: identical counts.
- The single failure is `S6WalkingCamera.cpp:296`, the honest Scenario 6 map-quality test (tolerance unchanged).
- `verify-env.sh --project`: 78 PASS, 0 FAIL.

| ID | Sev | Resolution | Validation |
|---|---|---|---|
| R4-01 | MED | `MapOutcome` policy in `MeasurementBase` (Converged / AcceptedMaxIterations if Newton norm <= 0.05 / Skipped*). Outcomes counted per measurement type in S4 and S5. S6 inherits the skip behaviour through `process()`. | Measurement/Core regression tests; S4 run summary: 0 skipped, 0 MaxIterations over 9160/9159/760 updates; S5: 0 MAP failures |
| R4-02 | MED | `funcmin` catches exceptions thrown by the objective at a trial point and treats them as a rejected step; `MeasurementBase` maps a failure to `SkippedNumericalFailure`. | funcmin regression test |
| R4-03 | MED | Missing fixtures now fail through `REQUIRE`/`REQUIRE_MESSAGE` (`CameraModel.cpp`, `DJIVideoCaption.cpp`). | grep audit; tests pass with fixtures present |
| R4-04 | MED | README rewritten: status, build, run (correct `--scenario=` syntax), user decisions B1–B3, verify result. | README.md |
| R4-05 | MED | Not fixable (no spec exists). Every visual and sensor choice is labelled ASSUMED in `architecture.md`. | architecture.md |
| R4-06 | MED | Altimeter sigma calibrated by NIS-median matching on the first half (0.0656 m). **Rejected** because the out-of-sample NIS got worse (median 2.44 vs 1.75). Value kept at 0.02 m and marked NOT NIS-consistent. | `out/scenario4_r5_altcal_flight_run_summary.txt`, `Scenario4Runner.h` comment |
| R4-07 | MED | Flow sigma calibrated to 0.083 px. **Rejected** because the synthetic-flight regression then misses its unchanged 1.5 m tolerance (LK errors are correlated). Kept 1 px; documented as NOT NIS-consistent. | `Scenario4Tests` |
| R4-08 | MED | Validation methodology corrected in `test-report.md`: net displacement and similarity-aligned error are reported instead of jitter-inflated path length. Altitude is reported as a consistency check, not an independent accuracy result. | test-report.md |
| R4-09 | MED | S5 re-identification gate rewritten as QR of stacked square-root factors plus a triangular solve. | gate check max rel. error 4.5e-16 (300 cases); SlamPoseTracker test |
| R4-10 | MED | S5 pixel sigma calibrated on the first half (0.629 px) and validated on the second half: NIS/dof mean 1.06, 82.9 % of frames in the 95 % band. Shortfall in the tails documented, not tuned away. | `out/scenario5_r5_summary.txt` |
| R4-11 | LOW | NIS evaluation is optional (`computeNis`) and uses the QR + triangular-solve form. | Scenario4Tests R4-11 |
| R4-12 | LOW | BufferedVideo: double `start()` throws; worker exceptions are captured and rethrown once. | BufferedVideoRegression R4-12 |
| R4-13 | LOW | `main.cpp` adds `catch (...)`. | build + CLI error-path tests |
| R4-14 | LOW | CLI syntax documented: `--scenario=N` / `-s=N`; `cv::CommandLineParser` rejects `-s 5`. Parser kept, to preserve the `--help` contract. | README §7 |
| R4-15 | LOW | Calibration uses the rational model only when explicitly allowed. | calibrateSynthetic R4-15 |
| R4-16 | LOW | A measurement with no curvature is skipped and flagged (`SkippedNoCurvature`), not silently applied. | Measurement R4-16 |
| R4-17 | LOW | Pitch guard at 85 deg in the dynamics and noise factor. | rotation/SystemVisualNav R4-17 |
| R4-18 | LOW | Open (accepted): CMake keeps `-Wall` and links onnxruntime, as the original `CMakeLists.txt` and spec require; onnxruntime is unused by project code. | — |
| R4-19 | LOW | Partially fixed: deprecated `cv::Mat_ <<` initialisers in `SlamPipeline.cpp` replaced. Remaining `-Wextra/-Wconversion` sites open (not enabled in the project flags). | Debug build: no project warnings with `-Wall` |
| R4-20 | LOW | Open: S4 synthetic tolerances are kept as written. They were not tightened after seeing results. | — |
| R4-21 | LOW | `S6ZZDebugTemp.cpp` removed. Pre-rework `out/scenario6_*` artefacts are kept as history; the current S6 evidence is `out/scenario6_r5_*`. | ls test/src |
| R4-22 | LOW→**HIGH (S6)** | S6 re-review after rework and the cheirality round: estimates are still inadequate (sparse map, short lifetimes). **Open, reported INADEQUATE** per user rule. | final-report.md |

Additional Lead fixes found during R5 integration (no-covariance-inversion rule "everywhere"):

- **S6 association gates** (`SlamAssociation.cpp`, `SlamTrackAssociation.cpp`) inverted a 2x2 innovation covariance.
  Replaced by `slam::gateDistanceSquared` (QR of `[R_p; sigma I]` plus a triangular solve).
- **Sky/ground classifier** (`VisualHorizon.cpp`) inverted its class covariance (`precision = LDLT.solve(I)`).
  Replaced by Cholesky square-root factors with triangular solves.
- Both are covered by `test/src/LeadSqrtFormRegression.cpp`; explicit inverses appear only there, as references.
- Release link failure: `Scenario4Tests.cpp` lacked `<Eigen/LU>` for its test-only inverse reference. Fixed.
