from __future__ import annotations

import unittest

import numpy as np

from scripts.frontend import extract_lfr_features, resample_linear


class FrontendTests(unittest.TestCase):
    def test_resample_length(self):
        samples = np.zeros(8000, dtype=np.float32)
        result = resample_linear(samples, 8000, 16000)
        self.assertEqual(result.shape, (16000,))

    def test_feature_shape_and_finiteness(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        samples = 0.1 * np.sin(2 * np.pi * 440 * time)
        features = extract_lfr_features(samples, sample_rate)
        self.assertGreater(features.shape[0], 0)
        self.assertEqual(features.shape[1], 560)
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(features.mean()), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
