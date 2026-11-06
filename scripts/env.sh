# env.sh - activate the MCHA4400 Assignment 2 native toolchain for the CURRENT shell only.
#
#   source scripts/env.sh          (must be sourced, not executed)
#
# Does not modify ~/.bashrc, does not use sudo, and does not set LD_LIBRARY_PATH: the built binaries
# get RUNPATH to the prefix (LDFLAGS below), so the system libstdc++ / OpenCV 4.5 / VTK 9.1 / Eigen 3.4
# under /usr are not used at run time. Re-sourcing is safe (idempotent).
#
# Effects:
#   PATH                 <prefix>/bin first (GCC 14.4, CMake 3.31.8, Ninja 1.13.2, doxygen, ctest)
#   CC / CXX             conda-forge GCC 14.4.0 (x86_64-conda-linux-gnu-gcc / -g++), not /usr/bin/g++
#   CFLAGS / CXXFLAGS    -isystem <prefix>/include  (as conda-forge's compiler activation does)
#   LDFLAGS              -Wl,-rpath,<prefix>/lib -L<prefix>/lib
#   CMAKE_PREFIX_PATH    <prefix>  (find_package searches this first; system copies are a fallback
#                                   only, and verify-env.sh fails if any package resolves outside it)
#   PKG_CONFIG_LIBDIR    <prefix>/lib/pkgconfig:<prefix>/share/pkgconfig (PKG_CONFIG_PATH unset)
#   MCHA4400_A2_PREFIX, MCHA4400_A2_ENV_ACTIVE
#
# Override the prefix with MCHA4400_A2_PREFIX (must match the one used by setup-env.sh).

if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
    echo "error: scripts/env.sh must be sourced:  source scripts/env.sh" >&2
    exit 1
fi

export MCHA4400_A2_PREFIX="${MCHA4400_A2_PREFIX:-$HOME/opt/mcha4400-a2}"

if [[ ! -x "${MCHA4400_A2_PREFIX}/bin/cmake" ]]; then
    echo "error: toolchain prefix not found at ${MCHA4400_A2_PREFIX} (run scripts/setup-env.sh)" >&2
    return 1
fi

_a2_p="${MCHA4400_A2_PREFIX}"
_a2_triple="x86_64-conda-linux-gnu"

# Remove any earlier activation of this prefix from PATH / flags (idempotent re-sourcing).
if [[ -n "${_a2_active_prefix:-}" ]]; then
    PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -x -F "${_a2_active_prefix}/bin" | paste -sd: -)"
    LDFLAGS="$(printf '%s' "${LDFLAGS:-}" | sed -e "s| *-Wl,-rpath,${_a2_active_prefix}/lib||" -e "s| *-L${_a2_active_prefix}/lib||")"
    CFLAGS="$(printf '%s' "${CFLAGS:-}" | sed -e "s| *-isystem ${_a2_active_prefix}/include||")"
    CXXFLAGS="$(printf '%s' "${CXXFLAGS:-}" | sed -e "s| *-isystem ${_a2_active_prefix}/include||")"
fi
export _a2_active_prefix="${_a2_p}"

export PATH="${_a2_p}/bin:${PATH}"
export CC="${_a2_p}/bin/${_a2_triple}-gcc"
export CXX="${_a2_p}/bin/${_a2_triple}-g++"
# Same convention as conda-forge's compiler activation: prefix headers as -isystem (not -I), so that
# ad-hoc compiles see e.g. <boost/...>, <Eigen/...>, <opencv2/...> from the prefix, never /usr/include.
export CFLAGS="-isystem ${_a2_p}/include${CFLAGS:+ ${CFLAGS}}"
export CXXFLAGS="-isystem ${_a2_p}/include${CXXFLAGS:+ ${CXXFLAGS}}"
export LDFLAGS="-Wl,-rpath,${_a2_p}/lib -L${_a2_p}/lib${LDFLAGS:+ ${LDFLAGS}}"
export CMAKE_PREFIX_PATH="${_a2_p}"
export PKG_CONFIG_LIBDIR="${_a2_p}/lib/pkgconfig:${_a2_p}/share/pkgconfig"
unset PKG_CONFIG_PATH
export MCHA4400_A2_ENV_ACTIVE=1

unset _a2_p _a2_triple
