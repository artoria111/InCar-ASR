# InCar-ASR

面向车载端侧部署的离线语音识别原型，目标平台为华为 Atlas 200I DK A2。

项目当前包含数据构建、Paraformer 训练、模型压缩、Atlas 部署和系统测试五个模块。可在普通电脑上运行的 CPU 基线已经使用官方 Sherpa-ONNX Paraformer-small INT8 模型统一；Atlas OM 推理仍需在配置好 CANN 的真实开发板上验证。

## 可复现的 CPU Demo

环境要求：

- macOS、Linux 或 Windows WSL
- Python 3.10 或更高版本
- 首次运行需要下载约 74MB 的模型文件

执行一条命令：

```bash
make demo
```

首次执行会：

1. 创建隔离环境 `.venv-demo`；
2. 安装固定版本的 `sherpa-onnx`；
3. 从官方发布页下载 Paraformer-small INT8；
4. 校验模型压缩包 SHA-256；
5. 识别模型自带的中文测试 WAV；
6. 输出识别文本、推理延迟和 RTF。

识别自己的音频：

```bash
./scripts/run_demo.sh /path/to/mono-pcm16.wav
```

机器可读输出：

```bash
.venv-demo/bin/python scripts/transcribe.py \
  /path/to/mono-pcm16.wav \
  --json
```

输入必须是单声道、16-bit PCM WAV；采样率可以不是 16kHz，Sherpa-ONNX 会在内部重采样。

## 实时麦克风与仪表盘

仪表盘不依赖第三方 Web 框架：

```bash
python3 apps/dashboard/server.py
```

另一个终端安装可选麦克风依赖并启动监听：

```bash
.venv-demo/bin/pip install -r requirements-microphone.txt
.venv-demo/bin/python scripts/microphone_demo.py \
  --dashboard http://127.0.0.1:8765
```

打开 `http://127.0.0.1:8765` 查看识别文本、延迟和 RTF。该 Demo 使用实时
采音与静音断句，句尾调用离线 Paraformer；不是逐字流式模型。没有麦克风时可用
`--simulate /path/to/test.wav` 验证完整上报链路。

## 测试

运行新增的可复现基线测试：

```bash
make test
```

运行数据构建模块测试：

```bash
python3 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r modules/01_dataset_building/requirements.txt
python -m pytest modules/01_dataset_building/tests -q
```

## 项目模块

| 模块 | 已有内容 | 当前状态 |
| --- | --- | --- |
| `01_dataset_building` | 录音、预处理、切分、校验、噪声增强 | 基本可用 |
| `02_asr_model_training` | Paraformer 微调与 CER 报告脚本 | 缺少可分发的训练音频和特征缓存 |
| `03_model_compression_distillation` | 蒸馏损失与压缩报告 | 实验框架，INT8/OM 尚未完成 |
| `04_atlas_edge_deployment` | AscendCL C++ 引擎、ATC 和 Profiling 脚本 | 需要真实开发板复验 |
| `05_system_integration_testing` | 文件/批量推理和 1200 条测试计划 | 测试计划尚未全部执行 |

详细完成度和已知缺口见 [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)。
数据、缓存、训练、CER 和 ONNX 对齐命令见
[`docs/TRAINING_AND_EVALUATION.md`](docs/TRAINING_AND_EVALUATION.md)。

## 协作方式

- 普通电脑负责数据、训练、ONNX、测试和代码审查。
- 唯一一块 Atlas 开发板作为共享的远程测试节点。
- 日常 Pull Request 运行普通 CI。
- Atlas 测试通过手动触发的 self-hosted runner 排队运行。
- 仓库不提交大型模型、原始音频、缓存特征或 OM 文件。

共享开发板的配置和安全要求见 [`docs/REMOTE_ATLAS.md`](docs/REMOTE_ATLAS.md)。
发布验收规则见 [`docs/RELEASE.md`](docs/RELEASE.md)。

## 重要说明

- `modules/02_asr_model_training/outputs/cer_report.md` 中的历史指标是已有实验记录，不代表当前仓库可以从零复现所有数据。
- `modules/05_system_integration_testing/outputs/test_plan.jsonl` 是测试计划，不是 1200 条已执行结果。
- 板端延迟、功耗和 OM 精度必须由 Atlas runner 或开发板持有人重新测试，不能由 CPU Demo 推断。

## License

[MIT](LICENSE)
