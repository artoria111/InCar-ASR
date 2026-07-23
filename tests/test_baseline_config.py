from __future__ import annotations

import unittest

from scripts.baseline_config import load_baseline_config


class BaselineConfigTests(unittest.TestCase):
    def test_required_runtime_contract(self):
        config = load_baseline_config()
        self.assertEqual(config["model"]["family"], "paraformer")
        self.assertEqual(config["model"]["precision"], "int8")
        self.assertEqual(config["frontend"]["sample_rate"], 16000)
        self.assertEqual(config["frontend"]["feature_dim"], 80)
        self.assertEqual(config["decoder"]["method"], "greedy_search")
        self.assertEqual(len(config["model"]["archive_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
