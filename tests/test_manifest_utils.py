from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.manifest_utils import (
    character_error_rate,
    edit_counts,
    resolve_audio_path,
)


class ManifestUtilsTests(unittest.TestCase):
    def test_character_error_rate(self):
        self.assertEqual(edit_counts("打开 空调", "打开空调"), (0, 4, 4))
        self.assertAlmostEqual(character_error_rate("打开空调", "打开车窗"), 0.5)
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "误识"), 1.0)

    def test_resolves_relative_audio_against_data_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "speaker" / "sample.wav"
            audio.parent.mkdir()
            audio.touch()
            manifest = root / "manifests" / "test.jsonl"
            manifest.parent.mkdir()

            resolved = resolve_audio_path(
                "speaker/sample.wav",
                manifest,
                data_root=root,
            )
            self.assertEqual(resolved, audio.resolve())

    def test_environment_root_preserves_nested_relative_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "wav" / "train" / "sample.wav"
            audio.parent.mkdir(parents=True)
            audio.touch()
            previous = os.environ.get("INCAR_ASR_DATA_ROOT")
            os.environ["INCAR_ASR_DATA_ROOT"] = str(root)
            try:
                resolved = resolve_audio_path(
                    "wav/train/sample.wav",
                    root / "manifest.jsonl",
                )
            finally:
                if previous is None:
                    os.environ.pop("INCAR_ASR_DATA_ROOT", None)
                else:
                    os.environ["INCAR_ASR_DATA_ROOT"] = previous
            self.assertEqual(resolved, audio.resolve())


if __name__ == "__main__":
    unittest.main()
