import unittest

import numpy as np

from incar_asr.streaming import EnergyVAD, VADConfig, simulate_stream


class StreamingTest(unittest.TestCase):
    def test_two_utterances_are_segmented(self):
        rate = 16000
        silence = np.zeros(int(rate * 0.4), dtype=np.float32)
        tone = 0.1 * np.sin(
            2 * np.pi * 440 * np.arange(int(rate * 0.7), dtype=np.float32) / rate
        )
        audio = np.concatenate((silence, tone, silence, tone, silence))
        config = VADConfig(
            sample_rate=rate,
            start_frames=2,
            end_frames=6,
            pre_roll_ms=80,
            min_utterance_ms=200,
        )
        segments = simulate_stream(audio, EnergyVAD(config), chunk_ms=40)
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(segment.samples.size > 0 for segment in segments))
        self.assertLess(segments[0].end_sample, segments[1].start_sample)


if __name__ == "__main__":
    unittest.main()
