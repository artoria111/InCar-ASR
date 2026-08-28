# Cockpit V2 — Atlas 200I DK A2 车载语音控制系统

This directory contains the on-device ASR + command-understanding stack deployed on the Huawei Atlas 200I DK A2 board. It powers the demo "cockpit" UI shown in the project presentation.

## Files

| File | Purpose |
|---|---|
| `cockpit_v2.py` | Flask HTTP service. Loads the sherpa-onnx Paraformer-zh-small INT8 model, runs FBank/LFR/CMVN front-end, decodes tokens, applies fuzzy ASR correction, parses commands, returns JSON. |
| `cockpit_index.html` | Browser UI (Liquid4All audio-car-cockpit fork). Chinese-localised car dashboard with HVAC, windows, music, navigation, door/vehicle controls. |
| `cockpit_asr_bridge.js` | WebRTC push-to-talk → /api/recognize → applies returned actions to cockpitController (UI state machine). |

## Model artifacts (board-side, not in repo)

| File | Source |
|---|---|
| `/root/work/car-asr-engine/model/model.int8.onnx` | sherpa-onnx Paraformer-zh-small 2024-03-09 INT8 |
| `/root/work/car-asr-engine/model/sherpa_tokens.txt` | 8359 token vocabulary |
| `/root/work/car-asr-engine/model/sherpa_cmvn.npz` | Kaldi-format CMVN stats |
| `/root/work/car-asr-engine/static/style.css` / `script.js` | Liquid4All framework files |

## Run on Atlas 200I DK A2

```bash
# Start the cockpit service
setsid /usr/bin/python3 -u /root/work/car-asr-engine/scripts/cockpit_v2.py \
    &>/tmp/cockpit.log &

# Expose via ngrok (optional, for remote demo)
setsid ngrok http 5000 --log=stdout &>/tmp/ngrok.log &
curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

# Health check
curl -s http://localhost:5000/api/health
# {"status": "ok"}

# Single recognition test
curl -s -X POST -F 'file=@test.wav' http://localhost:5000/api/recognize
```

## Key design choices

- **Voice activity detection** is not on the server side; the browser captures the entire push-to-talk utterance, then POSTs the WebM blob once released.
- **ONNX Runtime CPU inference** is used instead of NPU (CANN ATC conversion OOM on this board's 3.4 GB RAM). 335 ms per 2 s utterance. The native NPU C++ pipeline (see `src/ascend_inference.cpp`) achieves 12 ms on the same model when ATC succeeds.
- **Two-layer fuzzy correction** (`fuzzy_correct`) rewrites the ASR text before parsing: a ~50-entry `_FUZZY_MAP` covers the highest-frequency homophone errors observed on the board (空调 ↔ 空条, 导航 ↔ 到行, etc.).
- **Chinese numerals** are handled by `_cn2int` so 二十六度 / 26度 both resolve to temperature 26.
- **Direction disambiguation** is rule-based: `热`/`冷`/`打开`/`关闭` keywords steer ambiguous directions.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | `index.html` |
| GET | `/<file>` | | static asset |
| GET | `/api/health` | | `{"status":"ok"}` |
| POST | `/api/recognize` | `multipart/form-data` with `file=<webm/wav>` | `{"text":..., "delay_ms":..., "rtf":..., "duration_s":..., "command": {"actions": [...]}}` |

## Browser push-to-talk

`cockpit_asr_bridge.js` is a self-contained IIFE attached on `DOMContentLoaded`. It binds mousedown/touchstart to start recording and mouseup/touchend to stop+POST. The response is rendered into `#audio-status-text` and each `action` is dispatched via the action-type switch.