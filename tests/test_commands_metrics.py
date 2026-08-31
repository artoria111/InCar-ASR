import unittest

from incar_asr.commands import CommandMatcher, build_default_catalog
from incar_asr.metrics import edit_counts, summarize_results


class CommandAndMetricsTest(unittest.TestCase):
    def test_catalog_and_correction(self):
        catalog = build_default_catalog()
        self.assertGreaterEqual(len(catalog), 200)
        match = CommandMatcher(catalog).match("打开空条")
        self.assertFalse(match.rejected)
        self.assertEqual(match.corrected_text, "打开空调")
        self.assertEqual(match.intent, "climate.open")

    def test_dynamic_temperature_slot(self):
        match = CommandMatcher().match("把温度调到二十六度")
        self.assertEqual(match.intent, "climate.set_temperature")
        self.assertEqual(match.slots, {"temperature": 26})

    def test_empty_command_is_rejected(self):
        self.assertTrue(CommandMatcher().match("  ，。").rejected)

    def test_edit_counts_and_summary(self):
        counts = edit_counts("打开空调", "打开空条")
        self.assertEqual(counts.substitutions, 1)
        rows = [
            {
                "reference": "打开空调",
                "hypothesis": "打开空条",
                "corrected_text": "打开空调",
                "expected_intent": "climate.open",
                "predicted_intent": "climate.open",
                "expected_slots": {},
                "predicted_slots": {},
                "command_rejected": False,
                "noise_type": "engine",
                "requested_snr_db": 5,
                "total_ms": 12.0,
                "rtf": 0.02,
                "error": None,
            }
        ]
        summary = summarize_results(rows)
        self.assertGreater(summary["raw"]["corpus_cer"], 0)
        self.assertEqual(summary["command_aware"]["corpus_cer"], 0)
        self.assertEqual(summary["intent_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
