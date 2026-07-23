from __future__ import annotations

import struct
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.transcribe import read_pcm16_wave


class WaveReaderTests(unittest.TestCase):
    def test_reads_pcm16_mono(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(struct.pack("<hhh", 0, 16384, -16384))

            samples, sample_rate = read_pcm16_wave(path)
            self.assertEqual(sample_rate, 8000)
            self.assertEqual(samples.shape, (3,))
            self.assertAlmostEqual(float(samples[1]), 0.5)
            self.assertAlmostEqual(float(samples[2]), -0.5)

    def test_rejects_stereo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stereo.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(struct.pack("<hhhh", 0, 0, 1, 1))

            with self.assertRaisesRegex(ValueError, "must be mono"):
                read_pcm16_wave(path)


if __name__ == "__main__":
    unittest.main()
