#!/usr/bin/env bash
# Build and run a repeatable smoke test on an Atlas self-hosted runner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to an absolute .om file}"
TOKENS_PATH="${TOKENS_PATH:?Set TOKENS_PATH to an absolute tokens.txt file}"
WAV_PATH="${WAV_PATH:?Set WAV_PATH to an absolute mono PCM16 WAV file}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/atlas}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/atlas-results}"
DEVICE_ID="${DEVICE_ID:-0}"
EXPECTED_TEXT="${EXPECTED_TEXT:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RESULT_ROOT/$RUN_ID"

for path in "$MODEL_PATH" "$TOKENS_PATH" "$WAV_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "Required artifact not found: $path" >&2
        exit 2
    fi
done
mkdir -p "$RUN_DIR"

source "$SCRIPT_DIR/env_setup.sh"

cmake -S "$MODULE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-2}"
ctest --test-dir "$BUILD_DIR" --output-on-failure \
    2>&1 | tee "$RUN_DIR/host-tests.log"

set +e
"$BUILD_DIR/acl-hello" 2>&1 | tee "$RUN_DIR/acl-hello.log"
ACL_STATUS=${PIPESTATUS[0]}
"$BUILD_DIR/car-asr-cli" \
    --model "$MODEL_PATH" \
    --tokens "$TOKENS_PATH" \
    --wav "$WAV_PATH" \
    --device "$DEVICE_ID" \
    2>&1 | tee "$RUN_DIR/asr-smoke.log"
ASR_STATUS=${PIPESTATUS[0]}
set -e

REPORT_ARGS=(
    --run-id "$RUN_ID"
    --model "$MODEL_PATH"
    --tokens "$TOKENS_PATH"
    --wav "$WAV_PATH"
    --acl-status "$ACL_STATUS"
    --asr-status "$ASR_STATUS"
    --log "$RUN_DIR/asr-smoke.log"
    --output "$RUN_DIR/report.json"
)
if [[ -n "$EXPECTED_TEXT" ]]; then
    REPORT_ARGS+=(--expected-text "$EXPECTED_TEXT")
fi
python3 "$SCRIPT_DIR/write_board_report.py" \
    "${REPORT_ARGS[@]}"

echo "Atlas result bundle: $RUN_DIR"
if [[ "$ACL_STATUS" -ne 0 || "$ASR_STATUS" -ne 0 ]]; then
    exit 1
fi
