import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "02_asr_model_training"
    / "scripts"
    / "02_build_feature_cache.py"
)
SPEC = importlib.util.spec_from_file_location("build_feature_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class FeatureCacheTest(unittest.TestCase):
    def test_builds_portable_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_path = root / "sample.wav"
            time = np.arange(8000, dtype=np.float32) / 16000
            samples = (3000 * np.sin(2 * np.pi * 440 * time)).astype("<i2")
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(samples.tobytes())
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {"key": "sample", "source": "sample.wav", "target": "打开"}
                )
                + "\n",
                encoding="utf-8",
            )
            tokens = root / "tokens.txt"
            tokens.write_text(
                "<blank> 0\n<unk> 1\n打 2\n开 3\n", encoding="utf-8"
            )
            cache_path = root / "cache.npz"

            summary = MODULE.build_cache(
                manifest, root, tokens, cache_path
            )

            cache = np.load(cache_path, allow_pickle=False)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["feature_width"], 560)
            self.assertEqual(cache["token_ids"][0, :2].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
