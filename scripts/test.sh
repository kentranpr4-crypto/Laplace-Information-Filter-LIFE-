#!/usr/bin/env bash
# test.sh - run the test suite of a configured build directory via CTest (the project registers
# the doctest executable `tests` as CTest test `unit_tests`).
#
# Usage:
#   scripts/test.sh [Debug|Release|probe] [ctest-args ...]
#
#   Debug   (default) -> build/            (run scripts/build.sh first)
#   Release           -> build-release/
#   probe             -> scratch build of scripts/cmake-probe (run scripts/verify-env.sh first)
#
# Examples:
#   scripts/test.sh                       # all unit tests in build/
#   scripts/test.sh Debug -R unit_tests   # select by name
#   scripts/test.sh probe                 # doctest + smoke probe of the toolchain
#
# The project's own `run_tests` target also runs ctest after a build; this script is for
# re-running tests without rebuilding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for _a in "$@"; do
    case "$_a" in
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    esac
done
A2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="Debug"
if [[ $# -gt 0 && "$1" != -* ]]; then
    TARGET="$1"
    shift
fi

case "$TARGET" in
    Debug)   BUILD_DIR="${A2_DIR}/build" ;;
    Release) BUILD_DIR="${A2_DIR}/build-release" ;;
    probe)   BUILD_DIR="${MCHA4400_A2_SCRATCH:-$HOME/.cache/mcha4400-a2/scratch}/probe-build" ;;
    *) echo "test.sh: expected Debug, Release or probe (got '$TARGET')" >&2; exit 1 ;;
esac

# shellcheck source=scripts/env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ ! -f "${BUILD_DIR}/CTestTestfile.cmake" ]]; then
    echo "test.sh: ${BUILD_DIR} is not configured. Run scripts/build.sh ${TARGET} (or verify-env.sh for probe) first." >&2
    exit 1
fi

ctest --test-dir "${BUILD_DIR}" --output-on-failure "$@"
