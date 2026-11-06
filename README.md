# MCHA4400 Assignment 2 — visual navigation and SLAM (C++23)

This directory contains the C++23 project (`CMakeLists.txt`, `src/`, `test/`) and the scripts that
install and verify its native, user-space toolchain **without Docker, without root, and without touching
system libraries or `~/.bashrc`**. The engineering reports are in `engineering-reports/`
(plan, architecture, test-report, code-review, performance, final-report, plus `status.json`).

Status summary:

- Toolchain: GCC 14.4.0, CMake 3.31.8, Ninja, OpenCV 5.0.0, Eigen 5.0.1, VTK 9.5.2, Boost 1.88.0, autodiff 1.1.2,
  doctest 2.5.3, onnxruntime 1.28.0, installed into one prefix from a pinned lock file.
  Two USER-APPROVED deviations: **autodiff 1.1.2** (1.1.3 does not exist upstream) and **Boost 1.88.0** (VTK 9.5.2
  requires exactly 1.88.0). See section 9.
- The project configures and builds in Debug and Release. The doctest suite runs with assertions enabled in Debug.
- All three scenarios run end to end through the CLI on the real videos and export `out/<stem>_out.MOV`.
  Scenario 6 (room point SLAM) is reported **INADEQUATE** in quality; see `engineering-reports/final-report.md`.
- Portability: **UNTESTED on a clean machine.** Installed and verified on one machine only
  (Ubuntu 22.04.5, x86_64, NVIDIA GPU, `DISPLAY=:1`).

---

## 1. Prerequisites

| Item | Requirement | Notes |
|---|---|---|
| OS / arch | Ubuntu 22.04 (glibc >= 2.34), x86_64 | Only `linux-64` is pinned. |
| User | a normal user account | `setup-env.sh` refuses to run as root and never calls `sudo`. |
| Network | HTTPS to `github.com` (micromamba release, nanobench), `conda.anaconda.org` (conda-forge packages) | `build.sh` also fetches nanobench from GitHub at configure time (FetchContent). |
| Disk | ~3.8 GB for the installed prefix; package cache up to ~6.4 GB during install | `setup-env.sh` refuses to start with less than 12 GiB free (override: `MCHA4400_A2_MIN_FREE_GB`), so >= 5 GiB stays free. The package cache is removed afterwards. |
| Tools already on the system | `bash`, `curl`, `sha256sum`, `df`, `awk`, `sed`, `python3` (system Python is only used to parse JSON in `verify-env.sh` via the prefix's Python) | No compiler or CMake from the system is used. |
| Display / GPU | `--interactive=0` renders offscreen via EGL; modes 1/2 need `DISPLAY` | Verified with an NVIDIA RTX 3050 (EGL) and `DISPLAY=:1` (X11/Mesa for interactive windows). EGL also worked with `DISPLAY` unset. |

Nothing else is needed: no `apt`, no `pip`, no conda on your PATH, no Docker.

## 2. Setup (one time)

```bash
cd <path-to>/a2
scripts/setup-env.sh          # installs from scripts/conda-lock-linux-64.txt (exact URLs + md5)
```

What it does:

1. Downloads the pinned **micromamba 2.10.0-0** static binary from
   `github.com/mamba-org/micromamba-releases` and verifies its SHA-256 against the GitHub release digest
   (`701f39c0...8bd`). A mismatch aborts.
2. Creates the prefix **`$HOME/opt/mcha4400-a2`** from the explicit lock file (284 pinned conda-forge packages).
   Override with `MCHA4400_A2_PREFIX=/path`. Re-running is a no-op if the prefix was created by this script
   and matches the lock; a prefix it did not create is never modified.
3. Writes a marker `.mcha4400-a2-install.json` into the prefix (lock hash, micromamba version, date).
4. Removes the micromamba package cache (`$HOME/.cache/mcha4400-a2/mamba`); pass `--keep-cache` to keep it.
   The micromamba binary itself stays in `$HOME/.cache/mcha4400-a2/bin/`.

Nothing is written to `~/.bashrc`, `/usr`, `/etc`, `~/.condarc` or `~/.mamba`.

To re-solve the environment from `environment.yml` (only when deliberately updating pins):
`scripts/setup-env.sh --from-spec` (regenerates `scripts/conda-lock-linux-64.txt`).

## 3. Activate (every shell)

```bash
source scripts/env.sh
```

`env.sh` is for the current shell only. It puts the prefix first on `PATH`, sets `CC`/`CXX` to conda-forge
GCC 14.4.0, sets `CMAKE_PREFIX_PATH` to the prefix, sets `PKG_CONFIG_LIBDIR` to the prefix, and sets
`CFLAGS`/`CXXFLAGS` to `-isystem <prefix>/include` (the conda-forge convention). `LDFLAGS` gets
`-Wl,-rpath,<prefix>/lib`, so built binaries load the prefix's `libstdc++`, OpenCV, VTK, etc. at run time.
`LD_LIBRARY_PATH` is **not** set. Re-sourcing is safe.

Compilers run directly from the shell do not read `CXXFLAGS`; pass `$CXXFLAGS` yourself for ad-hoc compiles
(CMake builds already get it).

## 4. Verify

```bash
scripts/verify-env.sh            # 8 sections, a few minutes; exit 0 only if every check passes
scripts/verify-env.sh --project  # also configures the real a2 project (scratch dir) and shows CMake's error
```

`verify-env.sh` prints the exact version and location of every dependency, taken from the **CMake config
files** that `find_package()` actually reads, and from the conda package records. It checks that:

- gcc/g++ 14.4.0, CMake 3.31.8, Ninja 1.13.2, Doxygen 1.17.0 come from the prefix;
- `<print>` compiles with `-std=c++23` against the prefix's libstdc++;
- `scripts/cmake-probe` (same `find_package` calls as `CMakeLists.txt`) configures, builds with GCC 14 + Ninja,
  and every package resolves **inside the prefix**;
- OpenCV 5.0.0 has the required headers and module libraries (`core`, `imgproc`, `imgcodecs`, `videoio`,
  `highgui`, `calib`, `stereo`, `geometry`, `features`, `video`, `objdetect`) and its FFmpeg backend is enabled;
- the smoke program (`scripts/cmake-probe/src/smoke.cpp`) runs: C++23 `std::println`, an Eigen 5 solve,
  a Boost.Math chi-squared quantile, OpenCV decode of the first frame of **all five data videos**
  (`indoor/room.MOV`, `indoor/calibration.MOV`, `indoor/tag_identical.MOV` are H.264; `outdoor/flight.MOV`,
  `outdoor/calibration.MOV` are HEVC), an `mp4v` write + re-read, an offscreen VTK render to PNG, and an
  onnxruntime link. Its artefacts are in a scratch directory, never in `data/`;
- every shared library of the smoke program resolves inside the prefix (no system `libstdc++`);
- all VTK module libraries referenced by VTK's own CMake target files exist;
- autodiff checks (see "Known blockers").

Latest result on the reference machine (`verify-env.sh --project`, after user decisions B1–B3): **78 PASS, 0 FAIL,
33 INFO**, exit 0. The INFO lines include the two known limitations: the original autodiff 1.1.3 spec does not exist,
and autodiff's Eigen-integration header does not compile with Eigen 5.0.1. The project does not use that header.

Scratch output (probe build tree, smoke PNG/MP4, logs) goes to `$HOME/.cache/mcha4400-a2/scratch`
(override with `MCHA4400_A2_SCRATCH`). Use `--keep-scratch` to keep the smoke outputs.

## 5. Build

```bash
scripts/build.sh                 # Debug   -> build/           (GCC 14, Ninja; CMakeLists.txt adds -O0 -g and EIGEN_INITIALIZE_MATRICES_BY_NAN)
scripts/build.sh Release         # Release -> build-release/   (CMakeLists.txt adds -O3, NDEBUG, EIGEN_NO_DEBUG)
scripts/build.sh Debug -DBUILD_DOCUMENTATION=ON -- -j8 --target a2
```

Arguments after the build type are passed to `cmake` (configure); arguments after `--` go to `cmake --build`.
`build.sh` sources `scripts/env.sh` itself and always uses `-G Ninja`, the prefix compilers, and
`-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF`. It does not modify `CMakeLists.txt`.

`CMakeLists.txt` strips `NDEBUG` and `_FORTIFY_SOURCE=2` that conda-forge's HDF5 imported targets (pulled in by VTK)
would otherwise inject into every compile, so `assert()` is active in Debug builds. Release keeps `-O3 -DNDEBUG
-DEIGEN_NO_DEBUG`.

The `run_tests` target runs the unit tests after every build; `build.sh` therefore fails if a test fails.
Expect a `-Wcpp _FORTIFY_SOURCE requires compiling with optimization` note in `-O0` builds (conda GCC specs; harmless).

## 6. Test

```bash
scripts/test.sh                  # ctest in build/   (after a successful build)
scripts/test.sh Release          # ctest in build-release/
scripts/test.sh probe            # ctest of the toolchain probe (doctest self-test); needs verify-env.sh first
```

The project registers `unit_tests` with CTest; the `run_tests` target also runs it after a build. Current counts and
the one known failing case (Scenario 6 map-quality test, kept failing on purpose) are in
`engineering-reports/test-report.md`. Run a subset directly, e.g. `build/tests -tc="*GaussianInfo*"`.

## 7. Run

Run from the build directory, because scenario paths are relative (see `AGENTS.md`):

```bash
cd build-release                                        # or build/ for Debug
./a2 ../data/outdoor/config.xml -c                      # calibrate the outdoor camera -> data/outdoor/camera.xml
./a2 ../data/indoor/config.xml -c                       # calibrate the indoor camera  -> data/indoor/camera.xml
./a2 ../data/outdoor/flight.MOV --scenario=4 -e         # flight: flow + horizon + altimeter
./a2 ../data/indoor/tag_identical.MOV --scenario=5 -e   # identical-ID ArUco SLAM
./a2 ../data/indoor/room.MOV --scenario=6 -e            # point-landmark SLAM
./a2 ../data/indoor/room.MOV --scenario=6 --interactive=1   # VTK window, blocks after the last frame
```

- Use `--scenario=N` / `-s=N` / `--interactive=N` / `-i=N`. `cv::CommandLineParser` reads a space-separated
  `-s 5` as a boolean and rejects it.
- `-e` writes `out/<stem>_out.MOV` (mp4v, 2W x H, input fps) and always `out/<stem>_timing.csv` (per-frame
  read/process/plot/write ms).
- `--interactive=0` renders offscreen through EGL (GPU) even when `DISPLAY` is set; modes 1 and 2 open an X window.
- `camera.xml` must exist beside the video (run `-c` first); a missing file or a frame-size mismatch is a clean error.
- `flight.SRT` (beside `flight.MOV`) is read for the altimeter only; latitude/longitude are validation-only (magenta
  track in the 3-D view) and never enter the estimator.

---

## 8. Pinned versions (installed, verified)

| Component | Required by project | Installed | Source (conda-forge) | Verified by |
|---|---|---|---|---|
| GCC / g++ | GCC 14 (`std::print`) | **14.4.0** | `gxx_linux-64 14.4.0` | `verify-env` §1 |
| libstdc++ / libgcc (runtime) | matches compiler | **14.4.0** | `libstdcxx`, `libgcc` | `verify-env` §2 |
| CMake | `cmake_minimum_required(3.31)` | **3.31.8** | `cmake 3.31.8` | `verify-env` §1 |
| Ninja | `-G Ninja` | **1.13.2** | `ninja 1.13.2` | `verify-env` §1 |
| OpenCV | 5.0 | **5.0.0** (headless, FFmpeg 8.1.2) | `libopencv 5.0.0 headless_h8197073_0` | CMake config + smoke |
| Eigen | 5.0 (`NO_MODULE`) | **5.0.1** (see note) | `eigen 5.0.1` | CMake config + smoke |
| VTK | 9.5 | **9.5.2** | `vtk-base 9.5.2` + `vtk-io-ffmpeg 9.5.2` | CMake config + smoke |
| Boost | 1.74 (`find_package(Boost 1.74)`) | **1.88.0** (deviation) | `libboost-devel 1.88.0` | CMake config |
| autodiff | 1.1.2 (`CMakeLists.txt`; spec said 1.1.3, which does not exist) | **1.1.2** | `autodiff 1.1.2` | CMake config + probe |
| doctest | (find_package) | **2.5.3** | `doctest 2.5.3` | CMake config + probe test |
| onnxruntime | (find_package) | **1.28.0** (C++ SDK) | `onnxruntime-cpp 1.28.0` | CMake config + link |
| Doxygen | optional | **1.17.0** | `doxygen 1.17.0` | `verify-env` §1 |
| nanobench | FetchContent v4.3.11 | v4.3.11 from GitHub at configure time | `martinus/nanobench` | project configure |
| Python (prefix; VTK/onnx/doxygen deps only) | — | 3.12.14 | `python 3.12.14` | `verify-env` §1 |

Full list: `scripts/conda-lock-linux-64.txt` (exact URLs and md5). Human-edited pins with rationale:
`environment.yml`.

Notes on the table:

- **Eigen 5.0.1 (conda build):** `Eigen/Version` in the conda package reports `EIGEN_VERSION_STRING "5.0.1-dev"`.
  The upstream `5.0.1` tarball from `gitlab.com/libeigen/eigen` reports `5.0.1`. I diffed every header in
  `Eigen/`: that one file is the only difference. Semantic-version macros are `5.0.1` in both. Eigen 5 keeps
  `EIGEN_WORLD_VERSION` at 3 by design.
- **OpenCV is headless:** `GUI: NONE` in `cv::getBuildInformation()`. All required modules are present, and
  VideoCapture/VideoWriter (FFmpeg) work. `cv::imshow`/`namedWindow` are **not** available. The current
  project source does not call them. If interactive `imshow` is needed, a GUI build is required; the only
  GUI-enabled conda-forge build I found requires Qt 6.11, while VTK 9.5.2 (also needed) pins Qt 6.10, so both
  cannot be installed together today. Visualization in the project should go through VTK.
- **OpenCV build choice:** the newest 5.0.0 builds need libstdc++ 15, and newer FFmpeg 9 or hdf5 2.x. The chosen
  build is the newest one that satisfies libstdc++ 14, FFmpeg 8.1.2, hdf5 1.14 (VTK compatible) and OpenVINO
  2026.2 (with TBB 2022.3).

## 9. Known blockers and limitations

### 9.1 autodiff 1.1.2 instead of 1.1.3 (USER DECISION B1)

- The original spec asked for `find_package(autodiff 1.1.3 REQUIRED)`. No autodiff 1.1.3 exists: the GitHub tags end at
  `v1.1.2`, `main` declares `project(autodiff VERSION 1.1.2)`, conda-forge's newest is 1.1.2.
- By user decision, `CMakeLists.txt` requires **1.1.2** (commented in the file). No shim or fake config was created.

### 9.2 autodiff's Eigen integration does not compile with Eigen 5.0.1 (USER DECISION B3: not used)

- `#include <autodiff/forward/dual/eigen.hpp>` fails (`autodiff/common/eigen.hpp:105`, `Eigen::IndexedView` template
  changed in Eigen 5); upstream `main` has the same code. `verify-env.sh` reports it as INFO.
- By user decision all Jacobians/Hessians are hand-written and validated against central finite differences in the
  doctest suite. autodiff is linked (as the original CMakeLists does) but not used by project code.

### 9.3 Boost 1.88.0 is installed, not Boost 1.74 (USER DECISION B2)

- The project asks for `find_package(Boost 1.74 REQUIRED)`. Boost 1.74 cannot be used with this environment:
  1. conda-forge `boost-cpp 1.74.0` pins `icu 70` and `zstd 1.4`; it does not co-install with CMake 3.31,
     OpenCV 5 and VTK 9.5 (solver conflict, verified);
  2. the conda-forge **VTK 9.5.2 CMake config calls `find_package(Boost 1.88.0 EXACT)`** from its XDMF3 module when
     `find_package(VTK REQUIRED)` is used, so any configure with Boost 1.74 fails (verified).
- Therefore `libboost-devel 1.88.0` is installed (headers + `BoostConfig.cmake`, which CMake 3.31 requires because
  `cmake_minimum_required(3.31)` removes the old FindBoost module). `find_package(Boost 1.74)` accepts 1.88.
- The project only needs header-only Boost Math (`chi2inv`). The Boost libraries are installed as a dependency but
  are not linked.
- Note: `CMakeLists.txt` sets `${Boost_INCLUDE_DIR}` for `include_directories`. With BoostConfig that variable is
  empty; the Boost headers are reachable because the prefix include directory is added by `-isystem` (via CFLAGS/
  CXXFLAGS and other imported targets). Verified: `boost/math/distributions/chi_squared.hpp` compiles with GCC 14.

### 9.4 Other notes

- **Qt / GUI:** VTK's conda build needs Qt 6.10 (`qt6-main`, `GUISupportQt`) present in the prefix. This is not used
  by the project.
- **VTK's CUDA-free VTKm:** `viskores` is pinned to a CPU build. The CUDA build makes VTK's config call
  `enable_language(CUDA)`, which fails without `nvcc`.
- **Extra development packages:** VTK's conda config requires several `-devel` packages that the `vtk` metapackage
  normally brings in (`libgl-devel`, `libopengl-devel`, `libxml2-devel`, `zlib` (headers), `expat` 2.8.1,
  `liblzma-devel`, `tbb-devel` 2022.3). Without them CMake falls back to the **system** copies in `/usr`
  (zlib 1.2.11, expat 2.4.7, liblzma 5.2.5, libxml2 2.9.13) or fails. They are all pinned in `environment.yml`.
- **expat / libgcc:** the newest expat (2.9.x) requires libgcc 15; the pinned 2.8.1 build keeps the runtime at libgcc 14.
- **TBB:** the libgcc-14 `tbb-devel` builds are 2022.3 (2023.x `tbb-devel` builds need libgcc 15).
- **Debug warning:** `-Wcpp: _FORTIFY_SOURCE requires compiling with optimization` appears in `-O0` builds from
  conda-forge's default GCC specs. It is harmless.
- **nanobench** is the only dependency fetched at configure time (GitHub, `GIT_TAG v4.3.11`). Configure needs network access
  to GitHub. It is not pinned by hash in the repository (as in the original `CMakeLists.txt`).
- **Lint:** `shellcheck` is not installed on the reference machine, so the scripts were checked with `bash -n` only.

## 10. Portability (UNTESTED)

What was tested on the reference machine only (Ubuntu 22.04.5, 16 cores, 15 GB RAM, `DISPLAY=:1`):

- `setup-env.sh --from-spec` (solve + write lock), then `setup-env.sh` again from the lock file alone on a
  removed prefix, then an idempotent re-run (no-op + cache cleaned);
- `verify-env.sh --project` (result in `engineering-reports/test-report.md`);
- `build.sh Debug` and `build.sh Release` with the full test suite, and the CLI runs of section 7;
- `test.sh probe`: passes;
- the safety gates: low-disk refusal (exit 2) and refusal to touch a prefix it did not create (exit 1).

Not tested: a fresh machine or container (no clean-room run), a different glibc/Ubuntu release, a network-restricted
machine, a machine with no X display, machines without an NVIDIA/EGL-capable GPU for the offscreen path (VTK falls back to another
window class; not measured), GPU/CUDA builds (the CUDA build of `viskores` is avoided on purpose). The environment is pinned, so the same lock file should
reproduce on a similar machine, but that has not been demonstrated.

## 11. Files

| File | Purpose |
|---|---|
| `environment.yml` | Human-edited exact pins with rationale (source of truth for re-solving) |
| `scripts/conda-lock-linux-64.txt` | Explicit lock file: exact conda-forge URLs and md5 for all 284 packages (install path) |
| `scripts/setup-env.sh` | Bootstrap micromamba (checksum-verified), create the prefix, clean caches; disk and root guards |
| `scripts/env.sh` | `source` it: activates the prefix for the current shell |
| `scripts/verify-env.sh` | Checks and prints versions and discovery; exit 1 on any FAIL |
| `scripts/build.sh` | Debug -> `build/`, Release -> `build-release/`, Ninja, passes extra args |
| `scripts/test.sh` | CTest in a chosen build directory (`Debug`, `Release`, `probe`) |
| `scripts/cmake-probe/` | Probe project: the same `find_package` calls as `CMakeLists.txt`, plus a C++23/OpenCV/Eigen/Boost/VTK/ONNX/video smoke program and a doctest self-test |
| `.gitignore` | Ignores `build/` and `build-release/` (and `build-probe/`) |

Data: `data/` is read-only input, except the generated `data/{outdoor,indoor}/camera.xml` and the visual-horizon
training patches `data/outdoor/sky_*.png`, `ground_*.png` (cropped from `flight.MOV`; no sample images were supplied).
`project.txt` contains only the project name, so the visual spec (colours, draw order) and the altimeter reference are
ASSUMED and documented in `engineering-reports/architecture.md`.
