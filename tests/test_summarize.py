import json
import tempfile
import unittest
from pathlib import Path

from overcooked_benchmark.summarize import flatten_run, summarize_files, summarize_rows


class SummarizeTest(unittest.TestCase):
    def test_flatten_run_extracts_nested_metrics(self):
        row = flatten_run(
            {
                "pair": "llm-llm",
                "model": "gpt-test",
                "task_id": "task",
                "score": 20,
                "success": True,
                "metrics": {
                    "tes": 0.8,
                    "ticks": 42,
                    "invalid_action_count": 1,
                    "progress_completeness": {"pc": 0.7},
                },
            }
        )
        self.assertEqual(row["pair"], "llm-llm")
        self.assertEqual(row["model"], "gpt-test")
        self.assertEqual(row["pc"], 0.7)
        self.assertEqual(row["invalid"], 1)

    def test_summarize_rows_groups_runs(self):
        summaries = summarize_rows(
            [
                {"pair": "llm-llm", "model": "gpt-test", "task_id": "task", "success": True, "score": 20, "tes": 1, "pc": 1, "ticks": 10, "invalid": 0},
                {"pair": "llm-llm", "model": "gpt-test", "task_id": "task", "success": False, "score": 0, "tes": 0.5, "pc": 0.25, "ticks": 20, "invalid": 2},
            ]
        )
        self.assertEqual(summaries[0]["success"], "1/2")
        self.assertEqual(summaries[0]["avg_score"], 10)
        self.assertEqual(summaries[0]["avg_invalid"], 1)

    def test_summarize_files_prints_table(self):
        payload = {
            "pair": "llm-llm",
            "model": "gpt-test",
            "results": [
                {
                    "pair": "llm-llm",
                    "model": "gpt-test",
                    "task_id": "task",
                    "score": 20,
                    "success": True,
                    "metrics": {
                        "tes": 0.8,
                        "ticks": 42,
                        "invalid_action_count": 0,
                        "progress_completeness": {"pc": 0.7},
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps(payload))
            table = summarize_files([path])
        self.assertIn("llm-llm", table)
        self.assertIn("gpt-test", table)


if __name__ == "__main__":
    unittest.main()
