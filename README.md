# InCar-ASR

Lightweight automatic speech recognition for in-car edge deployment.

面向车载端侧部署的轻量级自动语音识别系统。

The project uses FunASR/Paraformer-small as its baseline and targets offline inference on Atlas 200I DK A2. The repository is divided into five independent modules that match the team responsibilities in the project report.

本项目以 FunASR/Paraformer-small 为基线模型，目标是在 Atlas 200I DK A2 上完成离线推理。仓库按照项目报告中的团队分工划分为五个相互独立的模块。

## Project modules / 项目模块

| No. | Module | Responsibility |
| --- | --- | --- |
| 01 | [`dataset_building`](modules/01_dataset_building/) | Audio collection, preprocessing, vehicle-noise organization, and dataset validation.<br>音频采集、预处理、车载噪声整理及数据集校验。 |
| 02 | [`asr_model_training`](modules/02_asr_model_training/) | Reproduce and fine-tune Paraformer with FunASR.<br>基于 FunASR 复现并微调 Paraformer 模型。 |
| 03 | [`model_compression_distillation`](modules/03_model_compression_distillation/) | Knowledge distillation, quantization, and compression evaluation.<br>知识蒸馏、模型量化及压缩效果评估。 |
| 04 | [`atlas_edge_deployment`](modules/04_atlas_edge_deployment/) | Convert and deploy the offline model on Atlas 200I DK A2.<br>在 Atlas 200I DK A2 上转换并部署离线模型。 |
| 05 | [`system_integration_testing`](modules/05_system_integration_testing/) | Integrate the ASR pipeline, develop the demo, and test in-car scenarios.<br>集成 ASR 推理管线、开发演示界面并完成车载场景测试。 |

The dataset-building module is currently implemented. The other four directories intentionally contain only `.gitkeep` placeholders so that each teammate can develop their module without mixing responsibilities.

目前已经实现数据集构建模块。其余四个目录暂时只包含用于 Git 跟踪空目录的 `.gitkeep` 占位文件，方便各成员独立开发，避免不同分工的代码混放。

## Getting started / 开始使用

See the bilingual [dataset-building guide](modules/01_dataset_building/README.md) for installation, audio collection, preprocessing, validation, and testing commands.

安装、音频采集、预处理、数据校验及测试命令，请查看中英双语的[数据集构建说明](modules/01_dataset_building/README.md)。

## Collaboration / 协作方式

Develop each responsibility on a separate feature branch and open a pull request into `main` after review. Do not commit raw audio or generated datasets to normal Git history.

每项分工应在独立功能分支上开发，审查完成后通过 Pull Request 合并到 `main`。请勿将原始音频或生成的数据集提交到普通 Git 历史中。
