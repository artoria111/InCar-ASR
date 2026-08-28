#!/usr/bin/env bash
# ============================================================
# Ascend Profiling 性能分析脚本
# ============================================================
# Usage: ./scripts/profile.sh
# ============================================================
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/env_setup.sh"

PROFILE_DIR="${PROFILE_DIR:-${module_root}/profile_output}"
mkdir -p "$PROFILE_DIR"

echo "Starting profiling..."
echo "Output dir: $PROFILE_DIR"

# 设置Profiling环境变量
export ACL_PROFILING=ON
export PROFILING_DIR=$PROFILE_DIR
export PROFILING_OPTIONS="task_trace:on,op_trace:on"

# 运行推理程序
"${module_root}/build/car-asr-cli" "$@"

# 关闭Profiling
export ACL_PROFILING=OFF

echo ""
echo "Profiling data saved to: $PROFILE_DIR"
echo "Analyze with: msprof --input=$PROFILE_DIR"
