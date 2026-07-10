# Dataset building / 数据集构建

This module provides audio collection, preprocessing, metadata management, and dataset validation for the InCar-ASR vehicle-noise dataset.

本模块为 InCar-ASR 车载噪声数据集提供音频采集、预处理、元数据管理和数据集校验功能。

The default configuration standardizes audio to 16 kHz mono PCM WAV, preserves natural noise levels, creates deterministic train/validation/test splits, and checks format errors, clipping, weak signals, duplicate audio, metadata gaps, and split leakage.

默认配置会将音频统一为 16 kHz 单声道 PCM WAV，保留自然噪声音量，确定性地划分训练集、验证集和测试集，并检查格式错误、削波、低电平、重复音频、元数据缺失以及数据集泄漏。

The configuration contains 12 common in-car noise labels. The project target is at least 10 categories for later clean-speech mixing at controlled SNRs such as 5–15 dB.

配置中包含 12 类常见车载噪声。项目目标是收集不少于 10 类噪声，供后续按照 5–15 dB 等指定信噪比与干净语音混合。

## Module layout / 模块结构

```text
01_dataset_building/
├── configs/audio.yaml
├── data/
│   ├── README.md
│   └── metadata_template.csv
├── docs/data_collection.md
├── scripts/audio/
│   ├── collect_audio.py
│   ├── common.py
│   ├── preprocess_audio.py
│   └── validate_dataset.py
├── tests/test_audio_pipeline.py
└── requirements.txt
```

## Setup / 环境安装

Python 3.9 or later is recommended. Run these commands from the repository root.

推荐使用 Python 3.9 或更高版本。请在仓库根目录运行以下命令。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r modules/01_dataset_building/requirements.txt
```

If `sounddevice` cannot find an input device, install PortAudio for the operating system and retry. List available devices with:

如果 `sounddevice` 无法找到输入设备，请安装当前操作系统对应的 PortAudio 后重试。可以使用以下命令查看可用设备：

```bash
python modules/01_dataset_building/scripts/audio/collect_audio.py --list-devices
```

## Record a labelled sample / 采集带标签的音频

The collection script saves a PCM WAV file and appends one metadata row automatically.

采集脚本会保存 PCM WAV 文件，并自动追加一行元数据。

```bash
python modules/01_dataset_building/scripts/audio/collect_audio.py \
  --category engine_idle \
  --duration 30 \
  --vehicle-state idling \
  --window-state closed \
  --microphone-position center_console
```

Raw WAV files are stored below `modules/01_dataset_building/data/raw/<category>/`, and metadata is stored in `modules/01_dataset_building/data/metadata.csv`. Neither path is tracked by normal Git.

原始 WAV 文件保存在 `modules/01_dataset_building/data/raw/<category>/` 下，元数据保存在 `modules/01_dataset_building/data/metadata.csv` 中。这两个路径都不会被普通 Git 跟踪。

Read the [collection protocol](docs/data_collection.md) before recording in a vehicle.

在车内采集前，请先阅读[采集规范](docs/data_collection.md)。

## Import existing data / 导入已有数据

Place licensed WAV or FLAC files under category folders such as `data/raw/wind_noise/`. Copy `data/metadata_template.csv` to `data/metadata.csv`, then record the source URL, license, category, and privacy status for every recording.

将获得合法授权的 WAV 或 FLAC 文件放入 `data/raw/wind_noise/` 等类别目录。复制 `data/metadata_template.csv` 为 `data/metadata.csv`，并为每段录音填写来源网址、许可证、类别和隐私状态。

## Preprocess / 预处理

```bash
python modules/01_dataset_building/scripts/audio/preprocess_audio.py
```

The script generates the following structure while leaving every raw recording unchanged:

脚本会生成以下目录结构，同时保持所有原始录音不变：

```text
data/processed/
├── train/<category>/*.wav
├── validation/<category>/*.wav
├── test/<category>/*.wav
└── metadata.csv
```

RMS normalization, silence trimming, and high-pass filtering are disabled by default because they may remove useful noise characteristics. Enable them in `configs/audio.yaml` only for a documented experiment.

RMS 归一化、静音裁剪和高通滤波默认关闭，因为这些处理可能移除有价值的噪声特征。只有在明确记录的实验中，才应在 `configs/audio.yaml` 中启用。

## Validate / 校验数据集

```bash
python modules/01_dataset_building/scripts/audio/validate_dataset.py \
  --report reports/audio_validation.json
```

The validator returns a non-zero status for format errors or split leakage. Add `--fail-on-warning` when warnings should also fail an automated check.

发现格式错误或数据集划分泄漏时，校验脚本会返回非零状态码。如果希望警告也导致自动检查失败，请添加 `--fail-on-warning`。

## Tests / 测试

```bash
python -m pytest modules/01_dataset_building/tests -q
```

The tests use synthetic audio and a simulated microphone; they do not activate the real microphone.

测试使用合成音频和模拟麦克风，不会启用真实麦克风。

## Data policy / 数据管理原则

Do not commit raw or processed audio to standard Git history. Keep immutable raw data in the storage location agreed by the group and document access plus checksums in `data/README.md`. Do not publish identifiable cabin conversations or precise personal/location information.

请勿将原始音频或处理后的音频提交到普通 Git 历史中。不可变的原始数据应存放在小组约定的存储位置，并在 `data/README.md` 中记录访问方式和校验值。不得公开可识别身份的车内对话或精确的个人及位置信息。
