# InCar-ASR

面向车载离线语音识别的可复现实验、指令理解与 Atlas 端侧部署工具链。

本仓库不提交模型、音频、数据集和 profiling 结果等大文件。代码组可在普通电脑上完成模型接口验证、批量评测、噪声混合、指令纠错、意图/槽位计算和流式 VAD；设备组使用同一份 manifest 在 Atlas 上运行，并回传可追溯结果。

## 当前能力

- 严格的模型 I/O 契约：输入名、形状、dtype、前端、解码器和 token 文件统一配置。
- NumPy 参考前端：PCM WAV、重采样、80 维 FBank、LFR、CMVN。
- 三种后端：确定性 mock、ONNX Runtime、Atlas CLI 子进程。
- 真实批量评测：动态按 SNR 混噪，计算 CER、指令准确率、意图准确率、槽位 F1、延迟、RTF 和分组结果。
- 283 条车载指令目录，支持常见同音错误纠正及温度、风量、联系人、目的地等槽位。
- 流式自适应能量 VAD：预卷、起止判决、静音挂起、最短/最长语音限制。
- Atlas C++：FBank、VAD、CTC greedy/prefix beam、ACL 推理、JSON CLI、WAV 严格校验。
- Python 与便携 C++ 单元测试，以及 GitHub Actions CI。

## 代码组快速运行

要求 Python 3.9+。以下命令不会下载模型或音频：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

incar-asr validate-contract \
  --contract configs/model_contract.example.json

incar-asr build-catalog \
  --output artifacts/command_catalog.json

incar-asr validate-manifest \
  --manifest examples/manifest.example.jsonl \
  --allow-missing-audio
```

有最终 ONNX 和 token 文件后，先复制并修改契约，再运行：

```bash
cp configs/model_contract.example.json configs/model_contract.json
incar-asr validate-contract \
  --contract configs/model_contract.json \
  --require-artifacts

python -m pip install -e '.[onnx]'
incar-asr infer \
  --backend onnx \
  --contract configs/model_contract.json \
  --audio /path/to/16k_mono.wav
```

批量评测：

```bash
incar-asr evaluate \
  --backend onnx \
  --contract configs/model_contract.json \
  --manifest /path/to/test_manifest.jsonl \
  --output reports/onnx
```

输出包含 `results.jsonl`、`summary.json` 和 `report.md`，全部来自当次推理，不使用随机或手填实验结果。

## Atlas 设备组快速运行

完整步骤见 [设备组运行与交接手册](docs/设备组运行与交接手册.md)。最短路径如下：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

source modules/04_atlas_edge_deployment/scripts/env_setup.sh
cmake -S modules/04_atlas_edge_deployment \
  -B modules/04_atlas_edge_deployment/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build modules/04_atlas_edge_deployment/build -j"$(nproc)"

modules/04_atlas_edge_deployment/build/acl-hello --device 0

./scripts/run_device_benchmark.sh \
  /path/to/test_manifest.jsonl \
  /path/to/model.om \
  /path/to/tokens.txt \
  device_report
```

设备组需要回传 `device_info.json`、`results.jsonl`、`summary.json` 和 `report.md`。

## 重要的模型边界

仓库提供两个示例契约：

- `configs/model_contract.example.json`：560 维 LFR 的 Paraformer/ONNX 参考路径。
- `configs/model_contract_atlas_ctc.example.json`：80 维 FBank、单输入单输出、float32 I/O 的 Atlas C++ CTC 路径。

两者不是可以随意互换的同一个模型接口。当前 Atlas C++ 运行时只接受单输入、单输出、float32 I/O 的 CTC logits 模型；如果最终模型是完整 Paraformer、多输入模型或 560 维 LFR 模型，必须先统一导出接口或扩展 C++ 解码路径，不能只改文件名。

## 文档

- [项目改动说明](docs/项目改动说明.md)
- [现有功能手册](docs/现有功能手册.md)
- [设备组运行与交接手册](docs/设备组运行与交接手册.md)
- [模型接口契约](docs/模型接口契约.md)

## 当前验证范围

普通电脑上已验证 Python 单元测试和不依赖 CANN 的 C++ 核心测试。由于本机没有 Atlas、CANN 和最终模型，ACL 链路、OM 模型数值一致性、真机延迟、功耗和准确率必须由设备组按手册实测，仓库不会把目标值写成已完成结果。
