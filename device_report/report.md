# Atlas 200I DK A2 设备组测试报告

## 设备信息

| 项目 | 详情 |
|------|------|
| 平台 | Linux aarch64, Ubuntu 22.04 |
| NPU | Ascend 310B4, 健康, 6.6W, 42°C |
| CANN | 7.0.RC1 |
| CMake | 3.22.1 |
| 编译器 | g++ 11.3.0 |
| Python | 3.10.12 |

## NPU 链路验证

```
ALL CHECKS PASSED
Ascend NPU Link:      VERIFIED
Device Memory R/W:    VERIFIED (H→D→H)
CANN Runtime:         OPERATIONAL
```

详见 `acl_hello_output.log`

## 模型信息

| 文件 | SHA256 | 大小 |
|------|------|------|
| model.int8.onnx | 3ef6c193... | 78MB |
| sherpa_tokens.txt | 4b2d964e... | 74KB |

## Golden Sample 测试

| # | 参考文本 | 识别结果 | 延迟 | RTF |
|:--:|------|------|:--:|:--:|
| 1 | 温度调高 | 温度调高 | 391ms | 0.096 |
| 2 | 打开空调 | 打开空调 | 321ms | 0.107 |

详见 `golden_sample_infer.json`

## C++ 编译

- CMake 配置 + 编译日志：`cmake_config.log`, `cmake_build.log`
- 产物：libcarasr.so, car-asr-cli, acl-hello

## ATC 转换状态

**未完成** — 完整 Paraformer 模型编译需要 >3.4GB 内存，板端无法完成。
小模型 (MatMul 1.3MB) 的 ATC→OM 转换已验证通过。
较大内存交叉编译环境可解决问题。

## 限制说明

- C++ 引擎当前走 80 维 CTC 路径；实际部署使用 560 维 Paraformer + ONNX Runtime
- OM 模型缺失 → Python ONNX vs Atlas OM 对比无法完成
- 详见 `docs/项目改动说明.md` §5 和 §6
