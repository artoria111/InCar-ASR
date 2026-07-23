import unittest

from apps.dashboard.server import HISTORY, add_result


class DashboardTest(unittest.TestCase):
    def setUp(self):
        HISTORY.clear()

    def test_add_result_validates_and_normalizes(self):
        result = add_result(
            {
                "text": "打开空调",
                "duration_seconds": 1.5,
                "elapsed_seconds": 0.1,
                "rtf": 0.0667,
                "source": "test",
            }
        )
        self.assertEqual(result["text"], "打开空调")
        self.assertEqual(result["source"], "test")
        self.assertIsInstance(result["id"], int)

    def test_add_result_rejects_missing_metrics(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            add_result({"text": "缺少性能数据"})


if __name__ == "__main__":
    unittest.main()
