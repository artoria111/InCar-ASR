#!/usr/bin/env bash
# ============================================================
# ONNX → OM 模型转换脚本
# ============================================================
# Usage:
#   ./scripts/atc_convert.sh <onnx_path> [input_name] [input_frames] [soc_version]
# ============================================================
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/env_setup.sh"

ONNX_PATH="${1:?Usage: $0 <onnx_path> [input_name] [input_frames] [soc_version]}"
INPUT_NAME="${2:-speech}"
INPUT_FRAMES="${3:-498}"
SOC_VERSION="${4:-Ascend310B1}"
OUTPUT_DIR="$(cd "$(dirname "$ONNX_PATH")" && pwd)"
MODEL_NAME="$(basename "$ONNX_PATH" .onnx)"

echo "============================================"
echo "  ATC Model Conversion"
echo "  Input:  $ONNX_PATH"
echo "  Input:  ${INPUT_NAME}:1,${INPUT_FRAMES},80"
echo "  SoC:    $SOC_VERSION"
echo "  Output: $OUTPUT_DIR/${MODEL_NAME}_fp16.om"
echo "============================================"

# ATC common args
ATC_ARGS=(
    --framework=5                    # 5=ONNX
    --soc_version="$SOC_VERSION"
    --input_shape="${INPUT_NAME}:1,${INPUT_FRAMES},80"
    --input_format=ND
    --output="$OUTPUT_DIR/${MODEL_NAME}_fp16"
    --log=error
    --enable_small_channel=1
)

ATC_ARGS+=(--precision_mode=allow_fp32_to_fp16)

# Optional: insert AIPP config for hardware preprocessing
# AIPP_CFG="$OUTPUT_DIR/aipp_paraformer.cfg"
# if [ -f "$AIPP_CFG" ]; then
#     ATC_ARGS+=(--insert_op_conf=$AIPP_CFG)
# fi

echo "Running: atc --model=$ONNX_PATH ${ATC_ARGS[@]}"

atc --model="$ONNX_PATH" "${ATC_ARGS[@]}"

echo ""
echo "Done. Output: $OUTPUT_DIR/${MODEL_NAME}_fp16.om"
ls -lh "$OUTPUT_DIR/${MODEL_NAME}_fp16.om"
