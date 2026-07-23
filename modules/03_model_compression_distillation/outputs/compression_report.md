# Compression experiment status

Status: **not yet verified**

The earlier report claimed ONNX alignment and board performance without
reproducible artifacts. Those values have been removed.

| Stage | Required evidence | Current evidence |
| --- | --- | --- |
| Teacher → student distillation | command, manifests, checkpoints, loss history, CER | none committed |
| PyTorch → ONNX | export manifest, numeric parity, text/CER parity | tools implemented; run pending |
| FP32/FP16 → INT8 | quantization config, model checksum, numeric and CER parity | pending |
| ONNX → OM | ATC log, CANN/SoC/input shape, OM checksum | tools implemented; board run pending |
| OM execution | real device report, text parity, latency/RTF/memory | pending |

Use `02_export_onnx.py`, `03_compare_onnx.py`,
`04_atlas_edge_deployment/scripts/atc_convert.sh`, and
`04_atlas_edge_deployment/scripts/atlas_smoke.sh` to produce the evidence. Do
not mark a stage passed based on an expected threshold alone.
