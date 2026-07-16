#!/bin/bash
# ============================================================
# 环境变量配置脚本 — 在编译/运行前 source 此脚本
# ============================================================
# Usage: source scripts/env_setup.sh
# ============================================================

export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LITE_HOME=/home/mindspore-lite-2.2.10-linux-aarch64

# AscendCL
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$ASCEND_HOME/acllib/lib64:$LD_LIBRARY_PATH
export PATH=$ASCEND_HOME/atc/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$PATH
export PYTHONPATH=$ASCEND_HOME/pyACL/python/site-packages/acl:$PYTHONPATH

# MindSpore Lite
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH

echo "[env_setup] ASCEND_HOME=$ASCEND_HOME"
echo "[env_setup] LITE_HOME=$LITE_HOME"
echo "[env_setup] ATC=$(which atc 2>/dev/null || echo 'not found')"
echo "[env_setup] Environment ready."
