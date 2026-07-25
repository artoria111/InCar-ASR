# Atlas 端侧 CTC 推理模块

目标设备为 Atlas 200I DK A2 / Ascend 310B 系列。当前模块实现：

- AscendCL 初始化、OM 加载、H2D/执行/D2H 和安全释放。
- 16 kHz PCM16 mono WAV 严格解析。
- 80 维 power FBank 和 radix-2 FFT。
- 自适应能量 VAD。
- CTC greedy 与 prefix beam search。
- 单条 CLI 及末行 JSON 输出。
- 不依赖 CANN 的便携核心测试。

## 模型要求

当前 C++ 运行时要求单输入、单输出、float32 I/O：

```text
input : [1, T, 80] float32 FBank
output: [1, U, V] float32 CTC logits
```

完整 Paraformer、多输入模型、560 维 LFR 输入和非 float32 I/O 不能直接运行。正式模型必须先与 `configs/model_contract_atlas_ctc.example.json` 对齐。

## 编译

```bash
source scripts/env_setup.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

## 运行

```bash
./build/acl-hello --device 0

./build/car-asr-cli \
  --model /path/to/model.om \
  --tokens /path/to/tokens.txt \
  --wav /path/to/16k_mono_pcm16.wav \
  --device 0 \
  --vad-mode 2 \
  --beam-size 1 \
  --json
```

设备组完整步骤、批测和回传要求见仓库根目录的 `docs/设备组运行与交接手册.md`。

## 无设备核心测试

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -Iinclude \
  tests/test_core.cpp \
  src/audio_preprocess.cpp \
  src/vad_detector.cpp \
  src/ctc_decoder.cpp \
  src/utils.cpp \
  -o /tmp/incar-asr-core-tests

/tmp/incar-asr-core-tests
```

这项测试不验证 ACL、OM 数值一致性、真机延迟或功耗。
