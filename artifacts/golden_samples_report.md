# Golden Sample Comparison Report

Generated: 2026-07-25 13:51
Model: sherpa-onnx Paraformer-zh-small INT8

## Python ONNX Runtime Results

| # | Audio | Reference | Python ONNX | Match | Latency | Dur |
|:--:|------|------|------|:--:|:--:|:--:|
| GS01 | tiny_example.wav | 温度调高 | 温度调高 | PASS | 402ms | 4.1s |
| GS02 | test_open_ac.wav | 打开空调 | 打开空调 | PASS | 356ms | 3.0s |
| GS03 | test2.wav | 导航回家播放音乐关闭车窗打开天窗温度调高 | 导航回家播放音乐关闭车窗打开天窗温度调高 | PASS | 1405ms | 16.1s |
| GS04 | car_engine_snr5.wav | 打开空调 | 打开空调 | PASS | 302ms | 3.0s |
| GS05 | car_wind_snr5.wav | 打开空调 | 打开空调 | PASS | 358ms | 3.0s |
| GS06 | car_road_snr5.wav | 打开空调 | 打开空调 | PASS | 333ms | 3.0s |
| GS07 | test_open_ac_snr10.wav | 打开空调 | 打开空调 | PASS | 316ms | 3.0s |
| GS08 | tiny_example_snr10.wav | 温度调高 | 温度调高 | PASS | 396ms | 4.1s |
| GS09 | cmd_008.wav | 打开空调 | 附近有厕所 | FAIL | 208ms | 1.0s |
| GS10 | cmd_000.wav | 导航回家 | 导航回家 | PASS | 158ms | 0.8s |

**Python ONNX: 9/10 PASS**

## C++ OM Comparison

**Status: Pending** — requires OM model compiled from final ONNX on 560-dim Paraformer contract.
Current C++ CLI uses 80-dim CTC contract per `configs/model_contract_atlas_ctc.example.json`.

## Acceptance Checklist
- [x] 10 golden audio files selected with reference transcripts (10/10)
- [x] Python ONNX outputs recorded (9/10 match)
- [ ] C++ OM outputs compared (requires OM model)
- [ ] Token-level logits comparison (requires OM model)
- [ ] Frontend feature comparison (Python vs C++ FBank/LFR/CMVN)
