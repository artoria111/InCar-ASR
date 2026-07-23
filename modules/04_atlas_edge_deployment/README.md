# Atlas edge deployment

AscendCL C++ inference prototype for Atlas 200I DK A2. Hardware-independent
frontend, VAD, and decoder tests also run on ordinary Linux/macOS hosts.

## Host tests

```bash
cmake -S . -B build/host \
  -DCAR_ASR_BUILD_ENGINE=OFF \
  -DCAR_ASR_BUILD_HOST_TESTS=ON
cmake --build build/host
ctest --test-dir build/host --output-on-failure
```

## ONNX to OM

The converter requires CANN `atc` and a fixed-shape Paraformer ONNX model.

```bash
INPUT_SHAPE='speech:1,300,560' \
SOC_VERSION=Ascend310B1 \
./scripts/atc_convert.sh /path/to/model.onnx fp16
```

Use `DRY_RUN=1` to inspect the command without CANN. A successful conversion
creates the OM, ATC log, and a checksum manifest. Conversion alone does not
prove the model runs correctly; board verification is recorded separately.

Before conversion, verify the same full Paraformer graph on CPU:

```bash
python scripts/asr_infer.py \
  --model /path/to/full-paraformer.onnx \
  --tokens /path/to/tokens.json \
  --wav /path/to/smoke.wav
```

## Board build and smoke test

```bash
export MODEL_PATH=/opt/incar-asr/models/paraformer.om
export TOKENS_PATH=/opt/incar-asr/models/tokens.txt
export WAV_PATH=/opt/incar-asr/samples/smoke.wav
./scripts/atlas_smoke.sh
```

Results are written to `atlas-results/<run-id>/` and include raw logs, file
checksums, NPU information, exit codes, and parsed performance fields.
`.github/workflows/atlas-smoke.yml` runs exactly this command on the shared
self-hosted runner and serializes tasks with GitHub Actions concurrency.

## Current model contract

- mono PCM16 WAV at 16 kHz;
- 80-bin log-Mel frontend;
- LFR stack/stride 7/6, producing 560 values per model frame;
- fixed ONNX/OM input shape chosen at conversion time;
- token lines may be either `token` or `token id`;
- Paraformer non-autoregressive token decoding is the default; CTC repeat
  collapsing is opt-in.

The checked-in C++ code and ATC script have host-side tests. OM execution,
ONNX/OM text parity, latency, memory, temperature, and power remain unverified
until a report produced by the physical Atlas runner is attached.
