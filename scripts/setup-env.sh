#!/usr/bin/env bash
# setup-env.sh - native, user-space toolchain + dependency install for MCHA4400 Assignment 2.
#
# No Docker, no root, no sudo, no edits to ~/.bashrc or system directories. Everything goes into one
# isolated prefix outside the repository plus a micromamba cache directory:
#   $MCHA4400_A2_PREFIX  (default $HOME/opt/mcha4400-a2)
#       GCC 14.4.0 (+ libstdc++ 14.4), CMake 3.31.8, Ninja 1.13.2, OpenCV 5.0.0 (headless),
#       Eigen 5.0.1, VTK 9.5.2, autodiff 1.1.2, doctest 2.5.3, onnxruntime 1.28.0 (C++),
#       libboost-headers 1.88.0, Doxygen 1.17.0.   Exact pins: environment.yml / conda-lock-linux-64.txt
#   $MCHA4400_A2_CACHE   (default $HOME/.cache/mcha4400-a2)
#       micromamba binary (pinned, sha256-checked) and the package cache (removed after install
#       unless --keep-cache)
#
# Idempotent: if the prefix was created by this script from the same lock file (marker file written
# at the end of a successful install), it is left alone. Use --force to rebuild such a prefix.
# A prefix that exists but was NOT created by this script is never modified.
#
# Usage: scripts/setup-env.sh [--from-spec] [--keep-cache] [--force] [-h|--help]
#   --from-spec   solve environment.yml instead of installing the exact lock file (refreshes pins)
#   --keep-cache  keep the micromamba package cache (~6 GB) after install
#   --force       remove and recreate a prefix that this script created
#
# Environment overrides:
#   MCHA4400_A2_PREFIX    default $HOME/opt/mcha4400-a2
#   MCHA4400_A2_CACHE     default $HOME/.cache/mcha4400-a2
#   MCHA4400_A2_MIN_FREE_GB  refuse to start if free space on the cache filesystem is below this
#                         (default 12 GiB: the package cache reached ~6.4 GiB during the reference install,
#                         so 12 GiB keeps >= 5 GiB free at the peak)
#
# Exit codes: 0 ok, 1 usage/error, 2 refused (root, platform, disk), 3 checksum mismatch.

set -euo pipefail

# ---- pinned bootstrap --------------------------------------------------------------------------
# micromamba 2.10.0-0 static binary from github.com/mamba-org/micromamba-releases (official release).
# sha256 = GitHub release asset digest for micromamba-linux-64 at tag 2.10.0-0.
MICROMAMBA_TAG="2.10.0-0"
MICROMAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_TAG}/micromamba-linux-64"
MICROMAMBA_SHA256="701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd"
MARKER_NAME=".mcha4400-a2-install.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PREFIX="${MCHA4400_A2_PREFIX:-$HOME/opt/mcha4400-a2}"
CACHE="${MCHA4400_A2_CACHE:-$HOME/.cache/mcha4400-a2}"
MIN_FREE_GB="${MCHA4400_A2_MIN_FREE_GB:-12}"

MAMBA_ROOT="${CACHE}/mamba"
MICROMAMBA="${CACHE}/bin/micromamba"
LOCK_FILE="${SCRIPT_DIR}/conda-lock-linux-64.txt"
SPEC_FILE="${A2_DIR}/environment.yml"

FROM_SPEC=0
KEEP_CACHE=0
FORCE=0

log() { printf '[setup-env] %s\n' "$*"; }
die() { local code="$1"; shift; printf '[setup-env] ERROR: %s\n' "$*" >&2; exit "$code"; }

for arg in "$@"; do
    case "$arg" in
        --from-spec)  FROM_SPEC=1 ;;
        --keep-cache) KEEP_CACHE=1 ;;
        --force)      FORCE=1 ;;
        -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die 1 "unknown argument: $arg (see --help)" ;;
    esac
done

# ---- safety checks -----------------------------------------------------------------------------
if [[ "$(id -u)" -eq 0 ]]; then
    die 2 "refusing to run as root. This setup is user-space only; run it as your normal user."
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    die 2 "the pinned environment is linux-64 only (found $(uname -s)/$(uname -m))."
fi
for tool in curl sha256sum df awk; do
    command -v "$tool" >/dev/null 2>&1 || die 1 "required tool '$tool' not found on PATH."
done

free_gb() {
    # free GiB (integer) on the filesystem that holds path $1 (walks up to an existing directory)
    local target="$1"
    while [[ ! -d "$target" ]]; do target="$(dirname "$target")"; done
    df -BG --output=avail "$target" | tail -n 1 | tr -dc '0-9'
}

mkdir -p "$(dirname "$PREFIX")" "$CACHE"
FREE="$(free_gb "$CACHE")"
log "free space on $(df -P "$CACHE" | tail -n 1 | awk '{print $1}'): ${FREE} GiB (minimum required: ${MIN_FREE_GB} GiB)"
if (( FREE < MIN_FREE_GB )); then
    die 2 "only ${FREE} GiB free, below MCHA4400_A2_MIN_FREE_GB=${MIN_FREE_GB}. Free disk space first (keep >= 5 GiB free)."
fi

# Keep every micromamba artefact under $CACHE: do not read or write ~/.mamba, ~/.condarc, ~/.mambarc.
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
export CONDA_PKGS_DIRS="${MAMBA_ROOT}/pkgs"
export MAMBA_NO_BANNER=1

# ---- bootstrap micromamba ----------------------------------------------------------------------
mkdir -p "$(dirname "$MICROMAMBA")"
if [[ ! -x "$MICROMAMBA" ]] || ! echo "${MICROMAMBA_SHA256}  ${MICROMAMBA}" | sha256sum -c --status 2>/dev/null; then
    log "downloading micromamba ${MICROMAMBA_TAG} from ${MICROMAMBA_URL}"
    tmp="${MICROMAMBA}.part"
    curl -fsSL --retry 3 --retry-delay 5 -o "$tmp" "$MICROMAMBA_URL"
    if ! echo "${MICROMAMBA_SHA256}  ${tmp}" | sha256sum -c --status; then
        rm -f "$tmp"
        die 3 "micromamba checksum mismatch (expected ${MICROMAMBA_SHA256}). Refusing to use it."
    fi
    chmod +x "$tmp"
    mv "$tmp" "$MICROMAMBA"
fi
log "micromamba: $("$MICROMAMBA" --version)"

# ---- create the prefix -------------------------------------------------------------------------
SOURCE="$LOCK_FILE"
if [[ "$FROM_SPEC" -eq 1 ]]; then
    SOURCE="$SPEC_FILE"
    FORCE=1   # a spec solve always re-creates a prefix this script made (it refreshes the lock)
fi
[[ -f "$SOURCE" ]] || die 1 "missing ${SOURCE}"
SRC_SHA="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
MARKER="${PREFIX}/${MARKER_NAME}"

if [[ -e "$PREFIX" ]]; then
    if [[ -f "$MARKER" ]] && grep -q "\"source_sha256\": \"${SRC_SHA}\"" "$MARKER" && [[ "$FORCE" -eq 0 ]]; then
        log "prefix already installed from $(basename "$SOURCE") (${PREFIX}); nothing to do. Use --force to rebuild."
        SKIP_CREATE=1
    elif [[ -f "$MARKER" ]]; then
        log "removing previous prefix created by this script: ${PREFIX}"
        rm -rf -- "$PREFIX"
    else
        die 1 "${PREFIX} exists but was not created by this script (no ${MARKER_NAME}). Refusing to modify it. Choose another location with MCHA4400_A2_PREFIX."
    fi
fi

if [[ "${SKIP_CREATE:-0}" -eq 0 ]]; then
    log "installing into ${PREFIX} from $(basename "$SOURCE") (sha256 ${SRC_SHA:0:12}...)"
    if [[ "$FROM_SPEC" -eq 1 ]]; then
        "$MICROMAMBA" create -y --no-rc -r "$MAMBA_ROOT" -p "$PREFIX" -f "$SPEC_FILE"
        # Refresh the pinned explicit lock file from the freshly solved prefix (exact URLs + md5).
        "$MICROMAMBA" env export -p "$PREFIX" --explicit --md5 --no-rc -r "$MAMBA_ROOT" > "${LOCK_FILE}.tmp"
        mv "${LOCK_FILE}.tmp" "$LOCK_FILE"
        log "wrote refreshed lock file: ${LOCK_FILE} ($(grep -c '^https://' "$LOCK_FILE") packages)"
        SRC_SHA="$(sha256sum "$LOCK_FILE" | cut -d' ' -f1)"
        SOURCE="$LOCK_FILE"
    else
        "$MICROMAMBA" create -y --no-rc -r "$MAMBA_ROOT" -p "$PREFIX" -f "$LOCK_FILE"
    fi
    cat > "$MARKER" <<EOF
{
  "source": "$(basename "$SOURCE")",
  "source_sha256": "${SRC_SHA}",
  "micromamba": "${MICROMAMBA_TAG}",
  "installed_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    log "install complete."
fi

# ---- clean caches -----------------------------------------------------------------------------
if [[ "$KEEP_CACHE" -eq 0 ]]; then
    log "cleaning micromamba package cache under ${MAMBA_ROOT}"
    # micromamba 2.x takes the root from MAMBA_ROOT_PREFIX (exported above); it has no -r for 'clean'.
    # Only unused cache entries / tarballs are removed; files linked into the prefix are hard links and stay.
    "$MICROMAMBA" clean -y --all >/dev/null 2>&1 || log "warning: cache clean reported an error (non-fatal)"
fi

log "free space now: $(free_gb "$CACHE") GiB; prefix size: $(du -sh "$PREFIX" 2>/dev/null | cut -f1)"
log "next: source scripts/env.sh && scripts/verify-env.sh"
