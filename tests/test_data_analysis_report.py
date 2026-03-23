from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

matplotlib_stub = ModuleType("matplotlib")
matplotlib_stub.use = lambda *_args, **_kwargs: None
pyplot_stub = ModuleType("matplotlib.pyplot")
matplotlib_stub.pyplot = pyplot_stub
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts/generate_data_analysis_report.py"
SPEC = importlib.util.spec_from_file_location("generate_data_analysis_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class DataAnalysisReportTests(unittest.TestCase):
    def test_resolve_ratio_paths_accepts_03_and_30_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processed = Path(tmpdir)
            for name in (
                "pheme_forecast_ratio_30.jsonl",
                "pheme_forecast_ratio_03_train.jsonl",
                "pheme_forecast_ratio_03_val.jsonl",
                "pheme_forecast_ratio_03_test.jsonl",
            ):
                (processed / name).write_text("{}\n", encoding="utf-8")
            resolved = MODULE.resolve_ratio_paths(processed, "30")
            self.assertTrue(resolved["all"].name.endswith("ratio_30.jsonl"))
            self.assertTrue(resolved["train"].name.endswith("ratio_03_train.jsonl"))

    def test_model_bias_frame_contains_all_main_table_models(self) -> None:
        frame = MODULE.model_bias_frame()
        self.assertEqual(
            set(frame["model_name"]),
            {
                "text_baseline",
                "temporal_baseline",
                "structure_baseline",
                "affect_state_forecaster",
                "patchtst_baseline",
                "timesnet_baseline",
                "thread_transformer_baseline",
            },
        )

    def test_build_sample_frame_extracts_expected_columns(self) -> None:
        sample = {
            "thread_id": "root",
            "event_name": "charliehebdo",
            "split": "train",
            "observation_ratio": 0.5,
            "source_text": "source post",
            "conversation_tree": {"root": None, "r1": "root", "r2": "r1"},
            "observed_replies": [
                {"id": "r1", "parent_id": "root", "text": "bad warning", "created_at": "2020-01-01T00:00:00+00:00", "sentiment_label": "negative"},
                {"id": "r2", "parent_id": "r1", "text": "good support", "created_at": "2020-01-01T00:05:00+00:00", "sentiment_label": "positive"},
            ],
            "forecast_replies": [
                {"id": "r3", "parent_id": "root", "text": "neutral", "created_at": "2020-01-01T00:10:00+00:00", "sentiment_label": "neutral"},
            ],
            "observed_neg_ratio": 0.5,
            "observed_neu_ratio": 0.0,
            "observed_pos_ratio": 0.5,
            "observed_majority_sentiment": "negative",
            "future_neg_ratio": 0.0,
            "future_neu_ratio": 1.0,
            "future_pos_ratio": 0.0,
            "future_majority_sentiment": "neutral",
        }
        frame = MODULE.build_sample_frame([sample])
        self.assertEqual(frame.loc[0, "observed_reply_count"], 2)
        self.assertEqual(frame.loc[0, "forecast_reply_count"], 1)
        self.assertIn("avg_depth", frame.columns)
        self.assertIn("future_neg_bucket", frame.columns)


if __name__ == "__main__":
    unittest.main()
