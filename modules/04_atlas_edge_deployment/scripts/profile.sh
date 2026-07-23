#!/usr/bin/env bash
# Run the Atlas CLI with CANN profiling enabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"

source "$SCRIPT_DIR/env_setup.sh"

PROFILE_DIR="${PROFILE_DIR:-$REPO_ROOT/atlas-results/profile-$(date +%Y%m%d-%H%M%S)}"
BINARY="${CAR_ASR_BINARY:-$REPO_ROOT/build/atlas/car-asr-cli}"
mkdir -p "$PROFILE_DIR"

if [[ ! -x "$BINARY" ]]; then
    echo "car-asr-cli not found or not executable: $BINARY" >&2
    exit 2
fi

export ACL_PROFILING=ON
export PROFILING_DIR="$PROFILE_DIR"
export PROFILING_OPTIONS="${PROFILING_OPTIONS:-task_trace:on,op_trace:on}"

echo "Profiling output: $PROFILE_DIR"
"$BINARY" "$@"

export ACL_PROFILING=OFF
echo "Analyze with: msprof --input=$PROFILE_DIR"
