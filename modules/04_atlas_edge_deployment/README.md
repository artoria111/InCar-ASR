# Atlas Edge Deployment — Huawei Atlas 200I DK A2

This module contains everything needed to take a trained ASR model and run it on the Huawei Atlas 200I DK A2 board (Ascend 310B NPU, 24 TOPS INT8, 8 W, 3.4 GB RAM).

The code is split into two clearly-labeled directories so it is obvious what runs where:

```
modules/04_atlas_edge_deployment/
├── README.md             ← this file
├── local/                ← Source code that is developed on the PC, NOT deployed
├── board/                ← Source code that runs on the Atlas 200I DK A2 board
└── test/                 ← Local tests for the native C++ engine (no Atlas required)
```

## `local/` — code developed on the PC

Files here are **never** copied to the board. They exist for development, benchmarking, and the project's evaluation framework.

| File / Dir | Purpose |
|---|---|
| `src/` | C++ source for the native Atlas ASR engine (`ascend_inference.cpp`, `audio_preprocess.cpp`, `vad_detector.cpp`, `ctc_decoder.cpp`, `acl_hello.cpp`, `asr_engine.cpp`, `main.cpp`, `utils.cpp`). |
| `include/` | C++ headers (`asr_engine.h`, `acl_hello.h`, `vad_detector.h`, etc.). |
| `tests/` | xUnit-style `test_core.cpp` — runs without CANN, useful for CI. |
| `CMakeLists.txt` | CMake build for the native engine (`car-asr-cli` executable, `libcarasr.so`). |
| `atc_convert.sh` | ONNX → OM offline conversion. |
| `env_setup.sh` | Sets `ASCEND_HOME` and other env vars for CANN. |
| `profile.sh` | `profile.sh --model ... --wav ...` for board-side NPU profiling. |
| `cockpit_server.py` | Older prototype server, superseded by `board/cockpit_v2.py`. Kept for historical comparison. |
| `streaming_demo.py` | Older streaming demo, superseded by `incar-asr stream` in `src/incar_asr/streaming.py`. |

## `board/` — code that actually runs on the Atlas 200I DK A2

Files here mirror what is checked out at `/root/work/car-asr-engine/` on the board itself. They are intentionally **not** in the Python package's import path — they live outside the src tree because they are deployable artifacts.

| File | Purpose |
|---|---|
| `cockpit_v2.py` | Flask HTTP service. Loads sherpa-onnx Paraformer-zh-small INT8 (79 MB), runs FBank/LFR/CMVN front-end, decodes tokens, applies fuzzy ASR correction (~50-entry `_FUZZY_MAP`), parses commands, returns JSON. End-to-end ~370 ms on CPU. |
| `cockpit_index.html` | Browser UI — Chinese-localised fork of Liquid4All audio-car-cockpit (HVAC, windows, music, navigation, doors, vehicle controls). |
| `cockpit_asr_bridge.js` | WebRTC push-to-talk bridge — captures WebM blob, POSTs to `/api/recognize`, dispatches returned `actions` to `cockpitController`. |
| `README.md` | Detailed deploy instructions, endpoints, design notes for `cockpit_v2.py`. |

> Files in this directory are **also deployed** at `/root/work/car-asr-engine/scripts/` and `/root/work/car-asr-engine/static/` on the board.

## `test/` — local C++ engine tests

Standalone tests that don't require CANN or an Atlas board. Used in CI to catch regressions on the C++ engine before deploying.

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

## Reproducing the on-board deploy

```bash
# On the PC
cmake -S modules/04_atlas_edge_deployment/local \
      -B modules/04_atlas_edge_deployment/local/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build modules/04_atlas_edge_deployment/local/build -j

# Sync the board-side artefacts
scp modules/04_atlas_edge_deployment/local/build/car-asr-cli root@192.168.X.X:/root/work/car-asr-engine/scripts/
scp modules/04_atlas_edge_deployment/board/cockpit_v2.py root@192.168.X.X:/root/work/car-asr-engine/scripts/
scp modules/04_atlas_edge_deployment/board/cockpit_index.html root@192.168.X.X:/root/work/car-asr-engine/static/
scp modules/04_atlas_edge_deployment/board/cockpit_asr_bridge.js root@192.168.X.X:/root/work/car-asr-engine/static/

# On the board
setsid /usr/bin/python3 -u /root/work/car-asr-engine/scripts/cockpit_v2.py &>/tmp/cockpit.log &
```

## Why is the layout like this?

The C++ NPU engine (`local/`) is cross-compiled on the PC and copied to the board as a binary. The Python cockpit (`board/`) is pushed as plain source because it's small and tied to a specific model + corpus. Keeping them in separate directories makes it obvious at a glance which side a file belongs to.