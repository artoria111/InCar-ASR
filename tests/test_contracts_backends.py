import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from incar_asr.backends import SubprocessBackend
from incar_asr.contracts import ModelContract
from incar_asr.device_info import collect_device_info


class ContractAndBackendTest(unittest.TestCase):
    def test_contract_validates_frontend_dimension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root / "contract.json"
            contract = {
                "version": 1,
                "model_name": "test",
                "backend": "onnxruntime",
                "model_path": "model.onnx",
                "tokens_path": "tokens.txt",
                "sample_rate": 16000,
                "input": {
                    "features_name": "speech",
                    "lengths_name": None,
                    "shape": [1, -1, 560],
                    "dtype": "float32",
                },
                "output": {
                    "logits_name": "logits",
                    "lengths_name": None,
                    "dtype": "float32",
                },
                "decoder": {
                    "type": "paraformer",
                    "blank_id": 0,
                    "special_token_ids": [0],
                },
                "frontend": {
                    "sample_rate": 16000,
                    "mel_bins": 80,
                    "lfr_m": 7,
                    "lfr_n": 6,
                },
            }
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            loaded = ModelContract.load(contract_path)
            self.assertEqual(loaded.frontend.output_dim, 560)
            contract["input"]["shape"][-1] = 80
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                ModelContract.load(contract_path)

    def test_subprocess_backend_reads_final_json_line(self):
        backend = SubprocessBackend(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys;"
                    "print('device log');"
                    "print(json.dumps({'text':'打开空调','inference_ms':3.5}))"
                ),
                "{wav}",
            ]
        )
        result = backend.recognize(np.ones(800, dtype=np.float32) * 0.01, 16000)
        self.assertEqual(result.text, "打开空调")
        self.assertEqual(result.inference_ms, 3.5)

    def test_device_info_hashes_only_requested_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "token.txt"
            artifact.write_text("token", encoding="utf-8")
            output = root / "device.json"
            payload = collect_device_info(output, [artifact])
            self.assertTrue(output.is_file())
            self.assertEqual(payload["artifacts"][0]["bytes"], 5)
            self.assertEqual(len(payload["artifacts"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
