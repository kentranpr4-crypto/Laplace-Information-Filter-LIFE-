#!/usr/bin/env bash
# verify-env.sh - print and CHECK the exact versions and discoverability of every dependency of
# MCHA4400 Assignment 2 in the native environment. Exit status 0 only if every check passes.
#
# Usage:
#   scripts/verify-env.sh [--project] [--keep-scratch]
#     --project       also configure the REAL a2 project (scratch build dir, no compile) and print
#                     CMake's verbatim result (expected to PASS since user decision B1: autodiff 1.1.2)
#     --keep-scratch  keep scratch build trees and smoke outputs under $MCHA4400_A2_SCRATCH
#                     (default $HOME/.cache/mcha4400-a2/scratch)
#
# Checks (each line prints PASS / FAIL / INFO):
#   1. build tools from the prefix: gcc/g++ 14.4.0, cmake 3.31.8, ninja 1.13.2, doxygen 1.17.0
#   2. conda package records in the prefix against the pins in environment.yml
#   3. CMake config discovery (scripts/cmake-probe, same find_package calls as CMakeLists.txt):
#      doctest, Boost, Eigen3, OpenCV, VTK, onnxruntime, Doxygen - version AND location inside prefix
#   4. OpenCV 5.0.0 contents: required headers, module libraries, build info (FFMPEG, GUI)
#   5. probe build with GCC 14 + Ninja and smoke run: C++23 std::println, Eigen 5 solve, Boost.Math,
#      OpenCV decode of one frame of each data video (H.264 / HEVC), mp4v write + re-read,
#      VTK offscreen render to PNG, onnxruntime link, RUNPATH / no system libstdc++ at run time
#   6. VTK module libraries that VTK's config references exist in the prefix
#   7. autodiff: the version required by CMakeLists.txt (1.1.2 since user decision B1; the original
#      spec asked for 1.1.3, which does not exist upstream) must be found. The autodiff/Eigen 5
#      integration header is checked and reported as INFO: it does not compile with Eigen 5.0.1, and the
#      project deliberately does not use it (user decision B3: hand-written Jacobians).
#   8. free disk space on the prefix filesystem (FAIL below 5 GiB)
#
# Nothing here modifies the repository. Scratch output goes to $MCHA4400_A2_SCRATCH.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROBE_SRC="${SCRIPT_DIR}/cmake-probe"
DATA_DIR="${A2_DIR}/data"

RUN_PROJECT=0
KEEP_SCRATCH=0
for arg in "$@"; do
    case "$arg" in
        --project) RUN_PROJECT=1 ;;
        --keep-scratch) KEEP_SCRATCH=1 ;;
        -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg (see --help)" >&2; exit 1 ;;
    esac
done

# ---- expected versions -------------------------------------------------------------------------
EXPECT_GCC="14.4.0"
EXPECT_CMAKE="3.31.8"
EXPECT_NINJA="1.13.2"
EXPECT_DOXYGEN="1.17.0"
EXPECT_OPENCV="5.0.0"
EXPECT_EIGEN="5.0.1"              # EIGEN_VERSION_STRING of the conda build is "5.0.1-dev" (see ENVIRONMENT.md)
EXPECT_VTK="9.5.2"
EXPECT_BOOST_MIN="1.74"           # what CMakeLists.txt requires (find_package(Boost 1.74))
EXPECT_BOOST_INSTALLED="1.88.0"   # what is installed (VTK 9.5.2 requires exactly 1.88.0)
EXPECT_DOCTEST="2.5.3"
EXPECT_ONNXRUNTIME="1.28.0"
EXPECT_AUTODIFF_REQUIRED="$(sed -n 's/^find_package(autodiff \([0-9.]*\) REQUIRED).*/\1/p' "$(dirname "${BASH_SOURCE[0]}")/../CMakeLists.txt" | head -n 1)"  # what CMakeLists.txt requires (1.1.2, user decision B1)
EXPECT_AUTODIFF_REQUIRED="${EXPECT_AUTODIFF_REQUIRED:-1.1.2}"
EXPECT_AUTODIFF_ORIGINAL_SPEC="1.1.3"  # original assignment spec; does not exist upstream
EXPECT_AUTODIFF_INSTALLED="1.1.2" # newest upstream release
MIN_FREE_GB=5

# ---- reporting ----------------------------------------------------------------------------------
N_PASS=0; N_FAIL=0; N_INFO=0
FAILS=()
pass()    { N_PASS=$((N_PASS+1)); printf '  PASS  %s\n' "$*"; }
fail()    { N_FAIL=$((N_FAIL+1)); FAILS+=("$*"); printf '  FAIL  %s\n' "$*"; }
info()    { N_INFO=$((N_INFO+1)); printf '  INFO  %s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }
indent()  { sed 's/^/        | /'; }

check_eq() {  # check_eq <label> <actual> <expected>
    if [[ "$2" == "$3" ]]; then pass "$1: $2"; else fail "$1: got '$2', expected '$3'"; fi
}
version_ge() {  # version_ge <a> <b> : true if a >= b (dotted numeric)
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

# ---- activate -----------------------------------------------------------------------------------
# shellcheck source=scripts/env.sh
if ! source "${SCRIPT_DIR}/env.sh"; then
    echo "verify-env: could not activate the environment (run scripts/setup-env.sh)" >&2
    exit 1
fi
P="${MCHA4400_A2_PREFIX}"
SCRATCH="${MCHA4400_A2_SCRATCH:-$HOME/.cache/mcha4400-a2/scratch}"
mkdir -p "$SCRATCH"

echo "verify-env: prefix=${P}"
echo "verify-env: scratch=${SCRATCH}"
echo "verify-env: host=$(. /etc/os-release && echo "${PRETTY_NAME}") kernel=$(uname -r) cpus=$(nproc) DISPLAY=${DISPLAY:-unset}"
echo "verify-env: tooling -> $(command -v cmake) | $(command -v ninja) | $CXX"

in_prefix() { case "$1" in "$P"/*) return 0 ;; *) return 1 ;; esac; }

# ---- 1. tools -----------------------------------------------------------------------------------
section "1. build tools (must come from the prefix, never /usr)"
GCC_VER="$("$CC" -dumpfullversion 2>/dev/null)"
GXX_VER="$("$CXX" -dumpfullversion 2>/dev/null)"
check_eq "gcc" "$GCC_VER" "$EXPECT_GCC"
check_eq "g++" "$GXX_VER" "$EXPECT_GCC"
in_prefix "$(command -v "$CXX")" && pass "CXX resolves inside prefix" || fail "CXX resolves outside prefix: $(command -v "$CXX")"
PRINT_CHECK="${SCRATCH}/std_print_check.cpp"
printf '#include <print>\nint main() { std::println("std::println ok"); return 0; }\n' > "$PRINT_CHECK"
if "$CXX" -std=c++23 -fsyntax-only "$PRINT_CHECK" > "${SCRATCH}/std_print_check.log" 2>&1; then
    pass "g++ -std=c++23 compiles <print> (std::println) with the prefix libstdc++"
else
    fail "g++ -std=c++23 cannot compile <print> (see ${SCRATCH}/std_print_check.log)"
fi
CMAKE_VER="$(cmake --version 2>/dev/null | head -n 1 | awk '{print $3}')"
check_eq "cmake" "$CMAKE_VER" "$EXPECT_CMAKE"
in_prefix "$(command -v cmake)" && pass "cmake resolves inside prefix" || fail "cmake resolves outside prefix: $(command -v cmake)"
version_ge "$CMAKE_VER" "3.31" && pass "cmake >= 3.31 (cmake_minimum_required 3.31)" || fail "cmake $CMAKE_VER < 3.31"
NINJA_VER="$(ninja --version 2>/dev/null)"
check_eq "ninja" "$NINJA_VER" "$EXPECT_NINJA"
DOX_VER="$(doxygen --version 2>/dev/null || true)"
check_eq "doxygen (optional; BUILD_DOCUMENTATION)" "$DOX_VER" "$EXPECT_DOXYGEN"
info "pkg-config $(pkg-config --version 2>/dev/null || echo missing); python $("$P/bin/python" --version 2>&1 | awk '{print $2}') (prefix)"

# ---- 2. conda package records -------------------------------------------------------------------
section "2. installed conda packages (conda-meta records in the prefix)"
pkg_records() {
    "$P/bin/python" -I - "$P" <<'PY'
import glob, json, os, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "conda-meta", "*.json"))):
    with open(path) as f:
        rec = json.load(f)
    print(rec.get("name"), rec.get("version"), rec.get("build"))
PY
}
REC="$(pkg_records)"
rec_field() { awk -v n="$1" '$1==n {print $2; exit}' <<< "$REC"; }
rec_build() { awk -v n="$1" '$1==n {print $3; exit}' <<< "$REC"; }
for pkg in gxx_impl_linux-64 gcc_impl_linux-64 libstdcxx libgcc cmake ninja libopencv eigen vtk-base vtk-io-ffmpeg \
           autodiff doctest onnxruntime-cpp doxygen libboost-devel zlib expat liblzma-devel tbb-devel libxml2-devel \
           libopenvino ffmpeg python; do
    if [[ -n "$(rec_field "$pkg")" ]]; then
        info "$(printf '%-18s %-14s %s' "$pkg" "$(rec_field "$pkg")" "$(rec_build "$pkg")")"
    else
        fail "conda record missing: $pkg"
    fi
done
check_eq "record libopencv" "$(rec_field libopencv)" "$EXPECT_OPENCV"
check_eq "record eigen" "$(rec_field eigen)" "$EXPECT_EIGEN"
check_eq "record vtk-base" "$(rec_field vtk-base)" "$EXPECT_VTK"
check_eq "record onnxruntime-cpp" "$(rec_field onnxruntime-cpp)" "$EXPECT_ONNXRUNTIME"
check_eq "record doctest" "$(rec_field doctest)" "$EXPECT_DOCTEST"
check_eq "record cmake" "$(rec_field cmake)" "$EXPECT_CMAKE"
check_eq "record libstdcxx (runtime == compiler)" "$(rec_field libstdcxx)" "$EXPECT_GCC"
check_eq "record libboost-devel (installed Boost)" "$(rec_field libboost-devel)" "$EXPECT_BOOST_INSTALLED"
check_eq "record autodiff (installed)" "$(rec_field autodiff)" "$EXPECT_AUTODIFF_INSTALLED"

# ---- 3. CMake discovery (probe) -----------------------------------------------------------------
section "3. CMake find_package() discovery (scripts/cmake-probe, same calls as CMakeLists.txt)"
PROBE_BUILD="${SCRATCH}/probe-build"
PROBE_LOG="${SCRATCH}/probe-configure.log"
rm -rf "$PROBE_BUILD"
if cmake -S "$PROBE_SRC" -B "$PROBE_BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
        -DA2_DATA_DIR="$DATA_DIR" > "$PROBE_LOG" 2>&1; then
    pass "probe configure: find_package doctest, Boost 1.74, Eigen3 5.0, OpenCV 5.0, VTK 9.5, onnxruntime, Doxygen"
else
    fail "probe configure failed (see $PROBE_LOG)"; tail -n 25 "$PROBE_LOG" | indent
fi

probe_line() { sed -n "s/^-- A2PROBE $1 //p" "$PROBE_LOG" | head -n 1; }
probe_check() {  # probe_check <name> <expected_version_or_>=> ; checks version (exact) and prefix location
    local name="$1" expect="$2"
    local line; line="$(probe_line "$name")"
    if [[ -z "$line" ]]; then fail "$name: not reported by probe"; return; fi
    local ver="${line%% *}"; local loc="${line#* }"
    if [[ "$ver" == "$expect" ]]; then pass "$name version $ver at ${loc}"; else fail "$name version '$ver', expected '$expect' (at ${loc})"; fi
    in_prefix "$loc" && pass "$name located inside prefix" || fail "$name located OUTSIDE prefix: ${loc}"
}
probe_check doctest "$EXPECT_DOCTEST"
probe_check Eigen3 "$EXPECT_EIGEN"
probe_check OpenCV "$EXPECT_OPENCV"
probe_check VTK "$EXPECT_VTK"
probe_check onnxruntime "$EXPECT_ONNXRUNTIME"
# Boost: project minimum 1.74; installed 1.88.0 (deviation, see ENVIRONMENT.md). Must be inside the prefix.
BOOST_LINE="$(probe_line Boost)"
BOOST_VER="${BOOST_LINE%% *}"; BOOST_LOC="${BOOST_LINE#* }"
if [[ -n "$BOOST_LINE" ]] && version_ge "$BOOST_VER" "$EXPECT_BOOST_MIN"; then
    pass "Boost $BOOST_VER satisfies find_package(Boost ${EXPECT_BOOST_MIN}) at ${BOOST_LOC}"
else
    fail "Boost '${BOOST_VER}' does not satisfy >= ${EXPECT_BOOST_MIN}"
fi
check_eq "Boost (installed, VTK requires exactly this)" "$BOOST_VER" "$EXPECT_BOOST_INSTALLED"
in_prefix "$BOOST_LOC" && pass "Boost located inside prefix" || fail "Boost located OUTSIDE prefix: ${BOOST_LOC}"
info "DEVIATION: project asks Boost 1.74; installed Boost is ${BOOST_VER} (see ENVIRONMENT.md 'Known blockers')"
DOX_LINE="$(probe_line Doxygen)"; info "Doxygen (find_package QUIET): ${DOX_LINE:-not found}"

# ---- 4. OpenCV contents -------------------------------------------------------------------------
section "4. OpenCV 5.0 contents (headers, module libraries, build info)"
OCV_INC="$P/include/opencv5"
for h in opencv2/core.hpp opencv2/imgproc.hpp opencv2/imgcodecs.hpp opencv2/videoio.hpp opencv2/highgui.hpp \
         opencv2/calib.hpp opencv2/stereo.hpp opencv2/geometry.hpp opencv2/features.hpp opencv2/video.hpp \
         opencv2/objdetect/aruco_detector.hpp; do
    [[ -f "$OCV_INC/$h" ]] && pass "header $h" || fail "header $h missing under $OCV_INC"
done
for m in core imgproc imgcodecs videoio highgui calib stereo geometry features video objdetect; do
    if compgen -G "$P/lib/libopencv_${m}.so.5.0*" > /dev/null || compgen -G "$P/lib/libopencv_${m}.so" > /dev/null; then
        pass "module library libopencv_${m}"
    else
        fail "module library libopencv_${m} missing"
    fi
done
# build info is written by the smoke program (below); checked after step 5
OCV_INFO="${SCRATCH}/smoke-out/opencv_build_information.txt"

# ---- 5. probe build + smoke run -----------------------------------------------------------------
section "5. probe build (GCC 14 + Ninja) and smoke program"
if cmake --build "$PROBE_BUILD" > "${SCRATCH}/probe-build.log" 2>&1; then
    pass "probe build (a2_smoke, a2_doctest_probe)"
else
    fail "probe build failed (see ${SCRATCH}/probe-build.log)"; tail -n 25 "${SCRATCH}/probe-build.log" | indent
fi
SMOKE_OUT="${SCRATCH}/smoke-out"
rm -rf "$SMOKE_OUT"; mkdir -p "$SMOKE_OUT"
SMOKE_LOG="${SCRATCH}/smoke.log"
if [[ -x "$PROBE_BUILD/a2_smoke" ]]; then
    timeout 900 "$PROBE_BUILD/a2_smoke" "$DATA_DIR" "$SMOKE_OUT" > "$SMOKE_LOG" 2>&1
    SMOKE_RC=$?
    grep -E '^A2SMOKE' "$SMOKE_LOG" | sed 's/^A2SMOKE /        /'
    [[ $SMOKE_RC -eq 0 ]] && pass "a2_smoke exit 0 (all checks)" || fail "a2_smoke exit $SMOKE_RC (see $SMOKE_LOG)"
    grep -q '^A2SMOKE check FAIL' "$SMOKE_LOG" && fail "a2_smoke reported FAIL checks (see $SMOKE_LOG)"
    grep -q 'check PASS decode first frame of outdoor/flight.MOV' "$SMOKE_LOG" && pass "HEVC decode (outdoor/flight.MOV, 2688x1512) on a real frame"
    grep -q 'check PASS decode first frame of indoor/room.MOV' "$SMOKE_LOG" && pass "H.264 decode (indoor/room.MOV, 1920x1080) on a real frame"
    if [[ -f "$OCV_INFO" ]]; then
        if grep -a -q 'FFMPEG:[[:space:]]*YES' "$OCV_INFO"; then pass "OpenCV build: FFMPEG backend YES"; else fail "OpenCV build: FFMPEG backend not enabled"; fi
        GUI_LINE="$(grep -a -m1 'GUI:' "$OCV_INFO" | sed 's/^ *//')"
        if grep -a -q 'GUI:[[:space:]]*NONE' "$OCV_INFO"; then
            info "OpenCV build: ${GUI_LINE} -> cv::imshow/namedWindow are NOT available (headless build; see ENVIRONMENT.md)"
        else
            pass "OpenCV build: ${GUI_LINE}"
        fi
    else
        fail "OpenCV build information not written by smoke program"
    fi
    # Run-time linkage: every prefix library must resolve inside the prefix; no system libstdc++.
    LDD_OUT="$(ldd "$PROBE_BUILD/a2_smoke" 2>/dev/null)"
    if echo "$LDD_OUT" | grep -q "not found"; then
        fail "a2_smoke has unresolved shared libraries:"; echo "$LDD_OUT" | grep "not found" | indent
    else
        pass "a2_smoke: all shared libraries resolve (ldd, no 'not found')"
    fi
    if echo "$LDD_OUT" | grep -E '=> /' | grep -v -E "=> ${P}/|linux-vdso|ld-linux" | grep -q .; then
        echo "$LDD_OUT" | grep -E '=> /' | grep -v -E "=> ${P}/" | indent
        info "a2_smoke: libraries NOT from the prefix (expected only glibc/system core libs such as libc/libm/libpthread)"
    fi
    LIBSTDCXX_PATH="$(echo "$LDD_OUT" | awk '/libstdc\+\+/ {print $3; exit}')"
    if [[ -n "$LIBSTDCXX_PATH" ]] && in_prefix "$LIBSTDCXX_PATH"; then
        pass "libstdc++ at run time: $LIBSTDCXX_PATH (prefix)"
    else
        fail "libstdc++ at run time is not the prefix copy: ${LIBSTDCXX_PATH:-none}"
    fi
    for lib in libopencv_core libvtkRenderingOpenGL2 libonnxruntime; do
        if echo "$LDD_OUT" | grep -q "${lib}.*=> ${P}/"; then pass "runtime link: ${lib} from prefix"; else fail "runtime link: ${lib} not from prefix"; fi
    done
else
    fail "a2_smoke not built"
fi
if [[ -x "$PROBE_BUILD/a2_doctest_probe" ]]; then
    if "$PROBE_BUILD/a2_doctest_probe" --no-intro > "${SCRATCH}/doctest_probe.log" 2>&1; then
        pass "doctest probe runs ($(grep -o 'doctest [0-9.]*' "${SCRATCH}/doctest_probe.log" | head -n 1 || echo ok))"
    else
        fail "doctest probe failed (see ${SCRATCH}/doctest_probe.log)"
    fi
fi
if [[ -n "${DISPLAY:-}" ]]; then
    info "DISPLAY=${DISPLAY}: VTK offscreen render used $(grep -a -m1 'VTK_render_window' "$SMOKE_LOG" | awk '{print $3}')"
else
    info "DISPLAY unset: VTK offscreen render requires an X/GL context (not verified in this session)"
fi

# ---- 6. VTK module libraries referenced by VTK's config -----------------------------------------
section "6. VTK 9.5.2 module libraries referenced by its CMake config"
VTK_TARGET_FILES=("$P"/lib/cmake/vtk-9.5/VTK-targets*.cmake)
if [[ -e "${VTK_TARGET_FILES[0]}" ]]; then
    missing=0; total=0
    while read -r so; do
        total=$((total+1))
        [[ -e "$P/lib/$so" ]] || { missing=$((missing+1)); echo "        | missing lib/$so"; }
    done < <(grep -h -o 'libvtk[A-Za-z0-9_]*-9\.5\.so\.9\.5' "${VTK_TARGET_FILES[@]}" | sort -u)
    if [[ $total -gt 0 && $missing -eq 0 ]]; then
        pass "all ${total} VTK module libraries referenced by VTK's target files exist"
    else
        fail "${missing} of ${total} VTK module libraries missing (or none referenced)"
    fi
    for m in RenderingOpenGL2 RenderingContextOpenGL2 InteractionStyle IOImage RenderingAnnotation RenderingFreeType IOFFMPEG; do
        [[ -e "$P/lib/libvtk${m}-9.5.so.9.5" ]] && pass "VTK module ${m}" || fail "VTK module ${m} missing"
    done
else
    fail "VTK-targets.cmake not found"
fi

# ---- 7. autodiff --------------------------------------------------------------------------------
section "7. autodiff (CMakeLists.txt requires ${EXPECT_AUTODIFF_REQUIRED}; installed ${EXPECT_AUTODIFF_INSTALLED})"
info "original assignment spec asked for autodiff ${EXPECT_AUTODIFF_ORIGINAL_SPEC}, which does not exist upstream (latest tag v1.1.2); CMakeLists.txt uses 1.1.2 by user decision B1"
AD_REQ_BUILD="${SCRATCH}/probe-autodiff-${EXPECT_AUTODIFF_REQUIRED}"
AD_REQ_LOG="${SCRATCH}/probe-autodiff-${EXPECT_AUTODIFF_REQUIRED}.log"
rm -rf "$AD_REQ_BUILD"
if cmake -S "$PROBE_SRC" -B "$AD_REQ_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
        -DA2_PROBE_AUTODIFF_VERSION="${EXPECT_AUTODIFF_REQUIRED}" \
        -DA2_DATA_DIR="$DATA_DIR" > "$AD_REQ_LOG" 2>&1; then
    pass "find_package(autodiff ${EXPECT_AUTODIFF_REQUIRED}) succeeded"
else
    fail "find_package(autodiff ${EXPECT_AUTODIFF_REQUIRED} REQUIRED) FAILS (CMakeLists.txt requirement)."
    echo "        CMake said:"
    sed -n '/Could not find a configuration file for package "autodiff"/,/^$/p' "$AD_REQ_LOG" | head -n 12 | indent
fi
AD_INST_BUILD="${SCRATCH}/probe-autodiff-${EXPECT_AUTODIFF_INSTALLED}"
AD_INST_LOG="${SCRATCH}/probe-autodiff-${EXPECT_AUTODIFF_INSTALLED}.log"
rm -rf "$AD_INST_BUILD"
if cmake -S "$PROBE_SRC" -B "$AD_INST_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
        -DA2_PROBE_AUTODIFF_VERSION="${EXPECT_AUTODIFF_INSTALLED}" \
        -DA2_DATA_DIR="$DATA_DIR" > "$AD_INST_LOG" 2>&1 \
   && cmake --build "$AD_INST_BUILD" --target a2_autodiff_smoke > "${SCRATCH}/probe-autodiff-build.log" 2>&1 \
   && "$AD_INST_BUILD/a2_autodiff_smoke" > "${SCRATCH}/autodiff_smoke.log" 2>&1; then
    AD_VER="$(sed -n 's/^-- A2PROBE autodiff //p' "$AD_INST_LOG" | awk '{print $1}')"
    check_eq "installed autodiff version" "$AD_VER" "$EXPECT_AUTODIFF_INSTALLED"
    grep -q 'check PASS' "${SCRATCH}/autodiff_smoke.log" && pass "autodiff ${EXPECT_AUTODIFF_INSTALLED} forward-mode derivatives match closed form"
else
    fail "autodiff ${EXPECT_AUTODIFF_INSTALLED} probe build/run failed (see ${AD_INST_LOG})"
fi
info "Eigen note: conda-forge eigen 5.0.1 Eigen/Version reports EIGEN_VERSION_STRING \"5.0.1-dev\"; upstream tag 5.0.1 reports \"5.0.1\". All other headers are byte-identical to the upstream tarball (diff checked)."
EIG_CHECK="${SCRATCH}/autodiff-eigen-check.cpp"
printf '#include <autodiff/forward/dual/eigen.hpp>\nint main() { return 0; }\n' > "$EIG_CHECK"
if "$CXX" -std=c++23 -fsyntax-only -isystem "$P/include" -isystem "$P/include/eigen3" "$EIG_CHECK" > "${SCRATCH}/autodiff-eigen-check.log" 2>&1; then
    pass "autodiff ${EXPECT_AUTODIFF_INSTALLED} Eigen integration header (autodiff/forward/dual/eigen.hpp) compiles with Eigen ${EXPECT_EIGEN}"
else
    first_err="$(grep -m1 'error' "${SCRATCH}/autodiff-eigen-check.log" | sed "s|$P/||" | cut -c1-200)"
    info "KNOWN LIMITATION (not used by the project, user decision B3): autodiff ${EXPECT_AUTODIFF_INSTALLED} Eigen integration header (autodiff/forward/dual/eigen.hpp) does NOT compile with Eigen ${EXPECT_EIGEN}: ${first_err}. Only the scalar dual API is usable; the project uses hand-written Jacobians."
fi

# ---- 8. optional: real project configure --------------------------------------------------------
if [[ "$RUN_PROJECT" -eq 1 ]]; then
    section "7b. real a2 project configure (--project; scratch build dir, no compile)"
    PROJ_BUILD="${SCRATCH}/a2-configure"
    PROJ_LOG="${SCRATCH}/a2-configure.log"
    rm -rf "$PROJ_BUILD"
    if cmake -S "$A2_DIR" -B "$PROJ_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
            -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF > "$PROJ_LOG" 2>&1; then
        pass "a2 project configures (Debug)"
    else
        fail "a2 project configure FAILED. CMake output:"
        sed -n '/CMake Error/,$p' "$PROJ_LOG" | head -n 30 | indent
    fi
fi

# ---- 9. disk ------------------------------------------------------------------------------------
section "8. disk"
FREE_GB="$(df -BG --output=avail "$P" | tail -n 1 | tr -dc '0-9')"
info "prefix size: $(du -sh "$P" 2>/dev/null | cut -f1)"
if (( FREE_GB < MIN_FREE_GB )); then fail "free space ${FREE_GB} GiB < ${MIN_FREE_GB} GiB"; else pass "free space ${FREE_GB} GiB (>= ${MIN_FREE_GB} GiB)"; fi

# ---- summary ------------------------------------------------------------------------------------
section "summary"
printf '  %d PASS, %d FAIL, %d INFO\n' "$N_PASS" "$N_FAIL" "$N_INFO"
if [[ $N_FAIL -gt 0 ]]; then
    echo "  failing checks:"
    for f in "${FAILS[@]}"; do echo "    - $f" | cut -c1-260; done
fi
if [[ "$KEEP_SCRATCH" -eq 0 ]]; then
    rm -rf "$SMOKE_OUT"
fi
[[ $N_FAIL -eq 0 ]]
