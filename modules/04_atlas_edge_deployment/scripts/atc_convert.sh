#!/usr/bin/env bash
# Convert a fixed-shape Paraformer ONNX model into an Atlas OM artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 <model.onnx> [fp16|origin|quantized]"
    echo ""
    echo "Environment overrides:"
    echo "  SOC_VERSION=Ascend310B1"
    echo "  INPUT_SHAPE=speech:1,300,560"
    echo "  OUTPUT_DIR=/path/to/output"
    echo "  DRY_RUN=1"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

ONNX_PATH="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
MODE="${2:-fp16}"
SOC_VERSION="${SOC_VERSION:-Ascend310B1}"
INPUT_SHAPE="${INPUT_SHAPE:-speech:1,300,560}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$ONNX_PATH")}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$ONNX_PATH" ]]; then
    echo "ONNX model not found: $ONNX_PATH" >&2
    exit 2
fi
case "$MODE" in
    fp16|origin|quantized) ;;
    *)
        echo "Unsupported mode: $MODE" >&2
        usage
        exit 2
        ;;
esac

source "$SCRIPT_DIR/env_setup.sh"

MODEL_STEM="$(basename "$ONNX_PATH" .onnx)"
OUTPUT_PREFIX="$OUTPUT_DIR/${MODEL_STEM}_${MODE}"
LOG_PATH="${OUTPUT_PREFIX}.atc.log"
mkdir -p "$OUTPUT_DIR"

ATC_ARGS=(
    "--model=$ONNX_PATH"
    "--framework=5"
    "--soc_version=$SOC_VERSION"
    "--input_shape=$INPUT_SHAPE"
    "--input_format=ND"
    "--output=$OUTPUT_PREFIX"
    "--log=info"
)

case "$MODE" in
    fp16)
        ATC_ARGS+=("--precision_mode=allow_fp32_to_fp16")
        ;;
    origin)
        ATC_ARGS+=("--precision_mode=must_keep_origin_dtype")
        ;;
    quantized)
        # AMCT-quantized deployable models must preserve their quantization
        # graph. Do not add high-precision flags that disable quantization.
        ;;
esac

echo "ATC conversion"
echo "  input:       $ONNX_PATH"
echo "  mode:        $MODE"
echo "  SoC:         $SOC_VERSION"
echo "  input shape: $INPUT_SHAPE"
echo "  output:      ${OUTPUT_PREFIX}.om"

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'atc'
    printf ' %q' "${ATC_ARGS[@]}"
    printf '\n'
    exit 0
fi

if ! command -v atc >/dev/null 2>&1; then
    echo "atc not found; source a valid CANN toolkit environment" >&2
    exit 127
fi

atc "${ATC_ARGS[@]}" 2>&1 | tee "$LOG_PATH"

OM_PATH="${OUTPUT_PREFIX}.om"
if [[ ! -s "$OM_PATH" ]]; then
    echo "ATC completed without a non-empty OM artifact: $OM_PATH" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/write_model_manifest.py" \
    --onnx "$ONNX_PATH" \
    --om "$OM_PATH" \
    --mode "$MODE" \
    --soc-version "$SOC_VERSION" \
    --input-shape "$INPUT_SHAPE" \
    --atc-log "$LOG_PATH" \
    --output "${OUTPUT_PREFIX}.manifest.json"

echo "OM ready: $OM_PATH"
