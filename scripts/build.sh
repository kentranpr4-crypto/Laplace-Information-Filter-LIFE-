#!/usr/bin/env bash
# build.sh - configure and build the a2 project with the native toolchain (Ninja, GCC 14.4).
#
# Usage:
#   scripts/build.sh [Debug|Release] [cmake-configure-args ...] [-- cmake-build-args ...]
#
#   Debug   (default) -> build/           (matches AGENTS.md: cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug)
#   Release           -> build-release/   (-O3, NDEBUG, EIGEN_NO_DEBUG; use for performance numbers only)
#
# Examples:
#   scripts/build.sh                                  # Debug build
#   scripts/build.sh Release -DBUILD_DOCUMENTATION=OFF
#   scripts/build.sh Debug -DBUILD_DOCUMENTATION=ON -- -j8 --target a2
#
# Notes:
#   * Sources the environment (scripts/env.sh) itself; nothing else needs to be set up.
#   * Refuses to run if setup-env.sh has not been run. Does not use sudo.
#   * The project's CMakeLists.txt is NOT modified by this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _a in "$@"; do
    case "$_a" in
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done
A2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="Debug"
if [[ $# -gt 0 && "$1" != -* ]]; then
    case "$1" in
        Debug|Release) BUILD_TYPE="$1" ;;
        *) echo "build.sh: build type must be Debug or Release (got '$1')" >&2; exit 1 ;;
    esac
    shift
fi

CONFIGURE_ARGS=()
BUILD_ARGS=()
seen_sep=0
for arg in "$@"; do
    if [[ "$arg" == "--" && $seen_sep -eq 0 ]]; then
        seen_sep=1
        continue
    fi
    if [[ $seen_sep -eq 0 ]]; then
        CONFIGURE_ARGS+=("$arg")
    else
        BUILD_ARGS+=("$arg")
    fi
done

if [[ "$BUILD_TYPE" == "Release" ]]; then
    BUILD_DIR="${A2_DIR}/build-release"
else
    BUILD_DIR="${A2_DIR}/build"
fi

# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

echo "[build.sh] type=${BUILD_TYPE} build_dir=${BUILD_DIR}"
echo "[build.sh] CXX=${CXX}"
"${CXX}" --version | head -n 1
cmake --version | head -n 1
ninja --version | sed 's/^/ninja /'

cmake -S "${A2_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF \
    "${CONFIGURE_ARGS[@]}"

cmake --build "${BUILD_DIR}" "${BUILD_ARGS[@]}"
echo "[build.sh] OK: ${BUILD_DIR}"
