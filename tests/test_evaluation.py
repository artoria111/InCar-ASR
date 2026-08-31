import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from incar_asr.audio import write_wav
from incar_asr.backends import MockBackend
from incar_asr.evaluation import load_manifest, run_evaluation


class EvaluationTest(unittest.TestCase):
    def test_real_report_bundle_from_dynamic_mix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            time = np.arange(16000, dtype=np.float32) / 16000
            clean = 0.12 * np.sin(2 * np.pi * 330 * time)
            noise = np.random.default_rng(2).normal(0, 0.08, 20000).astype(np.float32)
            write_wav(root / "clean.wav", clean)
            write_wav(root / "noise.wav", noise)
            manifest = root / "manifest.jsonl"
            row = {
                "sample_id": "case-001",
                "clean_audio": "clean.wav",
                "noise_audio": "noise.wav",
                "noise_type": "engine",
                "snr_db": 5,
                "transcript": "打开空调",
                "intent": "climate.open",
                "slots": {},
                "speaker_id": "speaker-01",
                "split": "test",
            }
            manifest.write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            output = root / "report"
            summary = run_evaluation(
                manifest,
                MockBackend(hypotheses={"case-001": "打开空条"}),
                output,
            )
            self.assertEqual(summary["sample_count"], 1)
            self.assertGreater(summary["raw"]["corpus_cer"], 0)
            self.assertEqual(summary["command_aware"]["corpus_cer"], 0)
            for name in ("results.jsonl", "summary.json", "report.md"):
                self.assertTrue((output / name).is_file())
            result = json.loads((output / "results.jsonl").read_text().splitlines()[0])
            self.assertAlmostEqual(result["achieved_snr_db"], 5.0, places=3)

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            rows = [
                {
                    "sample_id": "a",
                    "clean_audio": "a.wav",
                    "transcript": "a",
                    "speaker_id": "same",
                    "split": "train",
                },
                {
                    "sample_id": "b",
                    "clean_audio": "b.wav",
                    "transcript": "b",
                    "speaker_id": "same",
                    "split": "test",
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "speaker leakage"):
                load_manifest(manifest, require_audio=False)


if __name__ == "__main__":
    unittest.main()
