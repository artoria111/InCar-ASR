# Training and evaluation workflow

This workflow keeps audio outside Git and stores relative paths in manifests.
Commands below run from the repository root.

## 1. Prepare data

Create a UTF-8 transcript file with one item per line:

```text
command_0001 打开空调
command_0002 导航到公司
```

WAV basenames must match the first column.

```bash
python modules/02_asr_model_training/scripts/01_prepare_manifest.py \
  --audio-root /data/incar-asr/train \
  --transcripts /data/incar-asr/train.txt \
  --domain car_command \
  --output work/manifests/train.jsonl
```

Set `INCAR_ASR_DATA_ROOT=/data/incar-asr/train` or pass `--data-root` to
downstream commands. Do not rewrite manifests with machine-specific absolute
paths.

The existing AISHELL manifests resolve against the AISHELL dataset root (the
directory containing `wav/`). The small `train*.jsonl`/`val.jsonl` command
manifests resolve against the private directory containing `cmd_*.wav`.

## 2. Download the shared baseline and build caches

```bash
make download-model
python modules/02_asr_model_training/scripts/02_build_feature_cache.py \
  --manifest work/manifests/train.jsonl \
  --data-root /data/incar-asr/train \
  --tokens /path/to/the-training-model/tokens.json \
  --output work/cache/train.npz
```

The cache stores padded float32 features, feature lengths, token IDs, target
lengths, and sample keys. It is generated data and must not be committed.

## 3. Fine-tune

```bash
python modules/02_asr_model_training/scripts/02_finetune.py \
  --train_cache work/cache/train.npz \
  --val_cache work/cache/val.npz \
  --tokens models/sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt
```

Training still requires a compatible FunASR Paraformer checkpoint. Record the
base-model revision, data-manifest checksum, random seed, command, and produced
checkpoint checksum with every experiment.

## 4. CER and bad cases

```bash
python modules/02_asr_model_training/scripts/03_evaluate_baseline.py \
  --manifest work/manifests/test.jsonl \
  --data-root /data/incar-asr/test \
  --output reports/baseline-evaluation.json
```

The JSON contains every prediction and error; the sibling Markdown file
contains corpus CER, latency percentiles, RTF, and the worst cases. A failed
audio item is counted separately and never silently converted into an empty
hypothesis.

## 5. ONNX export and parity

Optional teacher-to-student distillation:

```bash
python modules/03_model_compression_distillation/scripts/01_knowledge_distillation.py \
  --teacher /path/to/teacher \
  --student /path/to/student \
  --train-cache work/cache/train.npz \
  --output-dir work/distillation
```

If teacher and student vocabularies differ, hidden-state and supervised CTC
losses remain active while logit KL is explicitly disabled and recorded in the
report. Training completion is not an accuracy pass; run CER afterward.

```bash
python modules/03_model_compression_distillation/scripts/02_export_onnx.py \
  --model /path/to/funasr-model \
  --checkpoint /path/to/best_model.pt \
  --output-dir work/exports/paraformer

python modules/03_model_compression_distillation/scripts/03_compare_onnx.py \
  --reference /path/to/reference.onnx \
  --candidate work/exports/paraformer/model.onnx \
  --cache work/cache/val.npz \
  --output reports/onnx-parity.json
```

Do not call a quantized/exported model aligned unless the parity command passes
the configured cosine and maximum-error thresholds and text/CER evaluation is
also run on the same manifest.
