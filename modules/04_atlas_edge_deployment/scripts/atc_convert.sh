#!/bin/bash
# ============================================================
# ONNX → OM 模型转换脚本
# ============================================================
# Usage:
#   ./scripts/atc_convert.sh <onnx_path> [fp16|int8]
# ============================================================
set -e

source $(dirname $0)/env_setup.sh

ONNX_PATH="${1:?Usage: $0 <onnx_path> [fp16|int8]}"
MODE="${2:-fp16}"
OUTPUT_DIR="$(dirname $ONNX_PATH)"
MODEL_NAME="$(basename $ONNX_PATH .onnx)"

echo "============================================"
echo "  ATC Model Conversion"
echo "  Input:  $ONNX_PATH"
echo "  Mode:   $MODE"
echo "  Output: $OUTPUT_DIR/${MODEL_NAME}_${MODE}.om"
echo "============================================"

# ATC common args
ATC_ARGS=(
    --framework=5                    # 5=ONNX
    --soc_version=Ascend310B1        # Atlas 200I DK A2
    --input_shape="speech:1,-1,80"   # dynamic time axis
    --input_format=ND
    --output="$OUTPUT_DIR/${MODEL_NAME}_${MODE}"
    --log=error
    --enable_small_channel=1
)

# Mode-specific args
if [ "$MODE" == "int8" ]; then
    ATC_ARGS+=(--precision_mode=force_fp16)  # 若ONNX已量化为INT8则用int8
else
    ATC_ARGS+=(--precision_mode=allow_fp32_to_fp16)
fi

# Optional: insert AIPP config for hardware preprocessing
# AIPP_CFG="$OUTPUT_DIR/aipp_paraformer.cfg"
# if [ -f "$AIPP_CFG" ]; then
#     ATC_ARGS+=(--insert_op_conf=$AIPP_CFG)
# fi

echo "Running: atc --model=$ONNX_PATH ${ATC_ARGS[@]}"

atc --model="$ONNX_PATH" "${ATC_ARGS[@]}"

echo ""
echo "Done. Output: $OUTPUT_DIR/${MODEL_NAME}_${MODE}.om"
ls -lh "$OUTPUT_DIR/${MODEL_NAME}_${MODE}.om"
