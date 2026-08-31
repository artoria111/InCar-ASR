import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from incar_asr.audio import (
    extract_features,
    mix_at_snr,
    read_wav,
    resample_audio,
    write_wav,
)
from incar_asr.contracts import FrontendConfig


class AudioTest(unittest.TestCase):
    def test_wav_round_trip_and_resample(self):
        samples = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audio.wav"
            write_wav(path, samples, sample_rate=8000)
            loaded = read_wav(path, target_rate=16000)
        self.assertEqual(loaded.sample_rate, 16000)
        self.assertEqual(loaded.samples.shape, (1600,))
        self.assertTrue(np.all(np.isfinite(loaded.samples)))

    def test_feature_dimensions_follow_contract(self):
        time = np.arange(16000, dtype=np.float32) / 16000
        samples = 0.2 * np.sin(2 * math.pi * 440 * time)
        plain = extract_features(samples, FrontendConfig())
        stacked = extract_features(samples, FrontendConfig(lfr_m=7, lfr_n=6))
        self.assertEqual(plain.features.shape[1], 80)
        self.assertEqual(stacked.features.shape[1], 560)
        self.assertEqual(stacked.lfr_frames, math.ceil(stacked.fbank_frames / 6))

    def test_mix_reaches_requested_snr(self):
        rng = np.random.default_rng(3)
        clean = 0.1 * np.sin(
            2 * math.pi * 220 * np.arange(16000, dtype=np.float32) / 16000
        )
        noise = rng.normal(0, 0.1, 24000).astype(np.float32)
        mixed = mix_at_snr(clean, noise, snr_db=5.0, seed=8)
        self.assertAlmostEqual(mixed.achieved_snr_db, 5.0, places=4)
        self.assertLessEqual(float(np.max(np.abs(mixed.samples))), 0.991)

    def test_resample_rejects_invalid_rate(self):
        with self.assertRaises(ValueError):
            resample_audio(np.ones(10, dtype=np.float32), 0, 16000)


if __name__ == "__main__":
    unittest.main()
