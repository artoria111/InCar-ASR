# Car-ASR Engine — 车载端侧语音识别推理引擎

基于华为 Atlas 200I DK A2（昇腾310B）的车载环境离线语音识别推理引擎。

## 环境要求

| 组件 | 版本 | 路径 |
|------|------|------|
| CANN | 7.0.RC1 | `/usr/local/Ascend/ascend-toolkit/latest` |
| MindSpore Lite | 2.2.10 | `/home/mindspore-lite-2.2.10-linux-aarch64` |
| CMake | ≥ 3.14 | system |
| g++ | ≥ 11 | system |
| NPU | Ascend 310B4 | Atlas 200I DK A2 |

## 目录结构

```
car-asr-engine/
├── CMakeLists.txt              # CMake构建
├── cmake/                      # CMake模块
├── include/                    # 头文件
│   ├── asr_engine.h            # 引擎统一接口
│   ├── ascend_inference.h      # AscendCL推理封装
│   ├── audio_preprocess.h      # FBank特征提取
│   ├── vad_detector.h          # 语音活动检测
│   ├── ctc_decoder.h           # CTC解码器
│   └── common.h                # 公共定义
├── src/                        # 实现代码
│   ├── main.cpp                # CLI入口
│   ├── asr_engine.cpp          # 引擎实现
│   ├── ascend_inference.cpp    # ACL推理实现
│   ├── audio_preprocess.cpp    # FBank实现
│   ├── vad_detector.cpp        # VAD实现
│   ├── ctc_decoder.cpp         # CTC解码实现
│   └── utils.cpp               # 工具函数
├── test/                       # 测试
├── model/                      # 模型文件目录
│   └── (放置 .om 离线模型)
├── scripts/
│   ├── env_setup.sh            # 环境变量配置
│   ├── atc_convert.sh          # ONNX→OM转换
│   └── profile.sh              # 性能分析
└── python_bindings/            # Python封装（Week 4）
```

## 快速开始

### 1. 环境配置

```bash
source scripts/env_setup.sh
```

### 2. 编译

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. 运行

```bash
# 识别WAV文件
./build/car-asr-cli \
    --model model/paraformer_small_fp16.om \
    --wav test/test_audio.wav \
    --tokens model/tokens.txt
```

## ONNX→OM模型转换

```bash
# 等成员C提供ONNX模型后：
./scripts/atc_convert.sh model/paraformer_small.onnx fp16
```

## 性能指标

| 指标 | 目标 | 当前 |
|------|:----:|:----:|
| RTF | < 0.1 | TBD |
| 端到端延迟 | < 500ms | TBD |
| 模型体积 (INT8) | < 50MB | TBD |
| 推理功耗 | < 8W | ~6.6W |

## 相关文档

- 项目书：`项目书-车载环境端侧语音识别系统.md`
- 任务计划：`任务.md`
