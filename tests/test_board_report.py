import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "04_atlas_edge_deployment"
    / "scripts"
    / "write_board_report.py"
)
SPEC = importlib.util.spec_from_file_location("write_board_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class BoardReportTest(unittest.TestCase):
    def test_extract_metrics(self):
        metrics = MODULE.extract_metrics(
            """
=== Recognition Result ===
打开空调
==========================
  Total latency:   42.5 ms
  RTF:             0.021
"""
        )
        self.assertEqual(metrics["text"], "打开空调")
        self.assertEqual(metrics["latency_ms"], 42.5)
        self.assertEqual(metrics["rtf"], 0.021)
        self.assertIsNone(metrics["peak_memory_mb"])


if __name__ == "__main__":
    unittest.main()
