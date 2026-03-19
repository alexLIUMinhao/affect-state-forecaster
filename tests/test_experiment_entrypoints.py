from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.utils.sentiment import weak_label_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


ratio_sweep = load_module("scripts/run_ratio_sweep.py", "run_ratio_sweep")
cross_event = load_module("scripts/run_cross_event_suite.py", "run_cross_event_suite")
labeler_robustness = load_module("scripts/run_labeler_robustness_suite.py", "run_labeler_robustness_suite")


class ExperimentEntrypointTests(unittest.TestCase):
    def test_conservative_labeler_is_supported(self) -> None:
        self.assertEqual(weak_label_text("good", labeler="lexicon_conservative"), "neutral")
        self.assertEqual(weak_label_text("good and safe", labeler="lexicon_conservative"), "positive")

    def test_ratio_sweep_uses_requested_ratio_files(self) -> None:
        args = type("Args", (), {"data_dir": Path("data/processed"), "device": "cpu", "epochs": 1, "batch_size": 2, "runs_root": Path("runs"), "experiments_root": Path("experiments"), "tag_prefix": "ratio_sweep", "models": ["text_baseline"]})()
        command = ratio_sweep.build_command(args, "30")
        command_text = " ".join(command)
        self.assertIn("pheme_forecast_ratio_30_train.jsonl", command_text)
        self.assertIn("pheme_forecast_ratio_30_test.jsonl", command_text)

    def test_cross_event_split_holds_out_single_event(self) -> None:
        records = [
            {"thread_id": "1", "event_name": "a", "split": "train"},
            {"thread_id": "2", "event_name": "a", "split": "train"},
            {"thread_id": "3", "event_name": "b", "split": "train"},
            {"thread_id": "4", "event_name": "c", "split": "train"},
        ]
        train_records, val_records, test_records = cross_event.build_cross_event_splits(records, "a")
        self.assertTrue(all(item["event_name"] == "a" for item in test_records))
        self.assertTrue(all(item["split"] == "test" for item in test_records))
        self.assertTrue(all(item["event_name"] != "a" for item in train_records + val_records))
        self.assertEqual({item["split"] for item in train_records}, {"train"})
        self.assertEqual({item["split"] for item in val_records}, {"val"})

    def test_labeler_robustness_uses_benchmark_ratio_suffix(self) -> None:
        self.assertEqual(labeler_robustness.ratio_suffix("30"), "03")
        self.assertEqual(labeler_robustness.ratio_suffix("50"), "05")
        self.assertEqual(labeler_robustness.ratio_suffix("70"), "07")


if __name__ == "__main__":
    unittest.main()
