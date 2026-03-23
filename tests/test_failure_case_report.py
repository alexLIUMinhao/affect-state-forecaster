from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

matplotlib_stub = ModuleType("matplotlib")
matplotlib_stub.use = lambda *_args, **_kwargs: None
pyplot_stub = ModuleType("matplotlib.pyplot")
for name in ("subplots", "close"):
    setattr(pyplot_stub, name, lambda *_args, **_kwargs: None)
matplotlib_stub.pyplot = pyplot_stub
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts/generate_failure_case_report.py"
SPEC = importlib.util.spec_from_file_location("generate_failure_case_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FailureCaseReportTests(unittest.TestCase):
    def test_bucket_helpers_follow_fixed_rules(self) -> None:
        self.assertEqual(MODULE.target_bucket(0.0), "zero")
        self.assertEqual(MODULE.target_bucket(0.05), "low")
        self.assertEqual(MODULE.target_bucket(0.2), "mid")
        self.assertEqual(MODULE.target_bucket(0.35), "high")
        self.assertEqual(MODULE.length_bucket(3), "short")
        self.assertEqual(MODULE.length_bucket(7), "medium")
        self.assertEqual(MODULE.length_bucket(9), "long")
        self.assertEqual(MODULE.flip_bucket(0.0, 0.2), "flip_up")
        self.assertEqual(MODULE.flip_bucket(0.2, 0.0), "flip_down")
        self.assertEqual(MODULE.flip_bucket(0.0, 0.0), "stable_zero")
        self.assertEqual(MODULE.flip_bucket(0.1, 0.2), "stable_nonzero")

    def test_align_predictions_adds_abs_errors(self) -> None:
        gold = {
            "t1": {
                "thread_id": "t1",
                "event_name": "sydneysiege",
                "split": "test",
                "observation_ratio": 0.5,
                "source_text": "source",
                "observed_replies": [],
                "forecast_replies": [],
                "observed_neg_ratio": 0.0,
                "future_neg_ratio": 0.3,
                "observed_reply_count": 0,
                "forecast_reply_count": 0,
                "target_bucket": "mid",
                "length_bucket": "short",
                "flip_bucket": "flip_up",
            }
        }
        prediction_maps = {
            "affect_state_forecaster": {"t1": {"predicted_future_neg_ratio": 0.1}},
            "patchtst_baseline": {"t1": {"predicted_future_neg_ratio": 0.28}},
            "thread_transformer_baseline": {"t1": {"predicted_future_neg_ratio": 0.22}},
        }
        rows = MODULE.align_predictions(gold, prediction_maps)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["affect_state_forecaster_abs_error"], 0.2)
        self.assertAlmostEqual(rows[0]["patchtst_baseline_abs_error"], 0.02)
        self.assertAlmostEqual(rows[0]["thread_transformer_baseline_abs_error"], 0.08)

    def test_case_bucket_selection_and_note_generation(self) -> None:
        rows = []
        for index in range(12):
            rows.append(
                {
                    "thread_id": f"t{index}",
                    "event_name": "sydneysiege",
                    "source_text": "source",
                    "observed_replies": [{"text": "reply"}],
                    "future_neg_ratio": 0.4,
                    "observed_neg_ratio": 0.0,
                    "observed_reply_count": 2,
                    "forecast_reply_count": 2,
                    "target_bucket": "high",
                    "length_bucket": "short",
                    "flip_bucket": "flip_up",
                    "affect_state_forecaster_pred": 0.05,
                    "patchtst_baseline_pred": 0.36,
                    "thread_transformer_baseline_pred": 0.18,
                    "affect_state_forecaster_abs_error": 0.35,
                    "patchtst_baseline_abs_error": 0.04,
                    "thread_transformer_baseline_abs_error": 0.22,
                    "mean_abs_error": 0.203,
                }
            )
        buckets = MODULE.select_case_buckets(rows)
        self.assertEqual(len(buckets["shared_failures"]), 10)
        self.assertEqual(len(buckets["flip_up_cases"]), 10)
        note = MODULE.derive_case_note(rows[0])
        self.assertEqual(note, "均值回归失败")

    def test_render_html_mentions_models_and_case_buckets(self) -> None:
        output_html = PROJECT_ROOT / "experiments/html/diagnostics/demo_failure_case.html"
        html_text = MODULE.render_html(
            output_html=output_html,
            report_run_id="demo_failure_case",
            overall_rows=[
                {"model": "affect_state_forecaster", "num_threads": 92, "mae": 0.1, "rmse": 0.2, "pearson": 0.0, "spearman": 0.1},
                {"model": "patchtst_baseline", "num_threads": 92, "mae": 0.11, "rmse": 0.21, "pearson": 0.1, "spearman": 0.2},
                {"model": "thread_transformer_baseline", "num_threads": 92, "mae": 0.12, "rmse": 0.22, "pearson": 0.2, "spearman": 0.3},
            ],
            slice_rows=[
                {"slice_type": "target_bucket", "slice_value": "high", "model": "affect_state_forecaster", "num_threads": 4, "mae": 0.2, "rmse": 0.3, "pearson": 0.1, "spearman": 0.2},
            ],
            case_rows=[
                {
                    "bucket_name": "shared_failures",
                    "thread_id": "t1",
                    "event_name": "sydneysiege",
                    "observed_neg_ratio": 0.0,
                    "future_neg_ratio": 0.4,
                    "observed_reply_count": 2,
                    "forecast_reply_count": 3,
                    "affect_state_forecaster_pred": 0.1,
                    "patchtst_baseline_pred": 0.2,
                    "thread_transformer_baseline_pred": 0.15,
                    "affect_state_forecaster_abs_error": 0.3,
                    "patchtst_baseline_abs_error": 0.2,
                    "thread_transformer_baseline_abs_error": 0.25,
                    "case_note": "均值回归失败",
                    "source_excerpt": "source text",
                    "reply_excerpt_1": "reply one",
                    "reply_excerpt_2": "",
                    "reply_excerpt_3": "",
                }
            ],
            figures={
                "slice_heatmap": PROJECT_ROOT / "experiments/figures/failure_case/demo_slice_heatmap.png",
                "gold_vs_prediction": PROJECT_ROOT / "experiments/figures/failure_case/demo_gold_vs_prediction.png",
                "absolute_error_by_bucket": PROJECT_ROOT / "experiments/figures/failure_case/demo_absolute_error_by_bucket.png",
                "pairwise_disagreement": PROJECT_ROOT / "experiments/figures/failure_case/demo_pairwise_disagreement.png",
            },
        )
        self.assertIn("Affect-State Forecaster", html_text)
        self.assertIn("PatchTST", html_text)
        self.assertIn("shared_failures", html_text)
        self.assertIn("thread_transformer_unique_wins", html_text)
        self.assertIn("没有符合当前阈值的样本", html_text)
        self.assertIn("../../figures/failure_case/demo_slice_heatmap.png", html_text)


if __name__ == "__main__":
    unittest.main()
