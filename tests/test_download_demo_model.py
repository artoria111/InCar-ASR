from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.download_demo_model import (
    MODEL_NAME,
    extract_archive,
    model_is_complete,
    sha256_file,
)


class DemoModelDownloadTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "payload"
            path.write_bytes(b"InCar-ASR")
            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(b"InCar-ASR").hexdigest(),
            )

    def test_model_completeness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self.assertFalse(model_is_complete(model_dir))
            (model_dir / "test_wavs").mkdir()
            (model_dir / "model.int8.onnx").touch()
            (model_dir / "tokens.txt").touch()
            (model_dir / "test_wavs" / "0.wav").touch()
            self.assertTrue(model_is_complete(model_dir))

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "unsafe.tar.bz2"
            with tarfile.open(archive_path, "w:bz2") as archive:
                info = tarfile.TarInfo("../outside")
                payload = b"unsafe"
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

            with self.assertRaises(ValueError):
                extract_archive(archive_path, Path(temp_dir) / "models")

    def test_extracts_expected_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "model.tar.bz2"
            with tarfile.open(archive_path, "w:bz2") as archive:
                for relative_path in (
                    "model.int8.onnx",
                    "tokens.txt",
                    "test_wavs/0.wav",
                ):
                    data = b"test"
                    info = tarfile.TarInfo(f"{MODEL_NAME}/{relative_path}")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))

            model_dir = extract_archive(archive_path, Path(temp_dir) / "models")
            self.assertTrue(model_is_complete(model_dir))


if __name__ == "__main__":
    unittest.main()
