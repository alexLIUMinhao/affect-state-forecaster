from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.datasets.pheme_forecast_dataset import PHEMEForecastDataset, collate_forecast_batch
from src.train import build_model, model_forward
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
capacity_matched = load_module("scripts/run_capacity_matched_suite.py", "run_capacity_matched_suite")
fusion_diagnostic = load_module("scripts/run_fusion_diagnostic_suite.py", "run_fusion_diagnostic_suite")
source_gate_validation = load_module("scripts/run_source_gate_validation.py", "run_source_gate_validation")
topconf_suite = load_module("scripts/run_topconf_baseline_suite.py", "run_topconf_baseline_suite")


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

    def test_capacity_matched_search_targets_structure_baseline_window(self) -> None:
        args = type(
            "Args",
            (),
            {
                "target_model": "structure_baseline",
                "target_param_tolerance": 0.05,
                "search_hidden_dim": [64, 72, 80, 88, 96, 104, 112, 120, 128],
                "search_affect_state_dim": [8, 16, 24, 32, 40],
            },
        )()
        matched = capacity_matched.choose_matched_asf_config(args)
        target = matched["target_param_count"]
        gap_ratio = abs(matched["param_count"] - target) / target
        self.assertLessEqual(gap_ratio, 0.05)
        self.assertLess(matched["hidden_dim"], 128)

    def test_fusion_diagnostic_includes_gate_variants(self) -> None:
        variants = fusion_diagnostic.variant_matrix()
        variant_names = {name for _model, name, _overrides in variants}
        self.assertIn("asf_scalar_gate", variant_names)
        self.assertIn("asf_vector_gate", variant_names)
        self.assertIn("asf_softmax_router", variant_names)
        self.assertIn("structure_baseline", {model for model, _name, _overrides in variants})

    def test_source_gate_validation_aggregates_by_capacity_and_variant(self) -> None:
        rows = [
            {
                "seed": 13,
                "capacity_group": "matched",
                "model_name": "affect_state_forecaster",
                "fusion_variant": "source_gate_only",
                "param_count": 2634355.0,
                "mae": 0.1240,
                "rmse": 0.1750,
                "pearson": 0.01,
                "spearman": -0.02,
                "gate_source_mean": 0.48,
                "gate_temporal_mean": 1.0,
                "gate_structure_mean": 1.0,
            },
            {
                "seed": 42,
                "capacity_group": "matched",
                "model_name": "affect_state_forecaster",
                "fusion_variant": "source_gate_only",
                "param_count": 2634355.0,
                "mae": 0.1260,
                "rmse": 0.1740,
                "pearson": 0.00,
                "spearman": -0.05,
                "gate_source_mean": 0.49,
                "gate_temporal_mean": 1.0,
                "gate_structure_mean": 1.0,
            },
        ]
        aggregates = source_gate_validation.aggregate_rows(rows)
        self.assertEqual(len(aggregates), 1)
        aggregate = aggregates[0]
        self.assertEqual(aggregate["capacity_group"], "matched")
        self.assertEqual(aggregate["fusion_variant"], "source_gate_only")
        self.assertEqual(aggregate["seeds"], "13,42")
        self.assertAlmostEqual(aggregate["mae"], 0.1250)
        self.assertGreaterEqual(aggregate["mae_std"], 0.0)

    def test_topconf_suite_builds_expected_models(self) -> None:
        args = type(
            "Args",
            (),
            {
                "train_path": Path("data/processed/pheme_forecast_ratio_05_train.jsonl"),
                "val_path": Path("data/processed/pheme_forecast_ratio_05_val.jsonl"),
                "test_path": Path("data/processed/pheme_forecast_ratio_05_test.jsonl"),
                "models": ["patchtst_baseline", "timesnet_baseline", "thread_transformer_baseline"],
                "device": "cpu",
                "epochs": 1,
                "batch_size": 2,
                "runs_root": Path("runs"),
                "experiments_root": Path("experiments"),
                "tag_prefix": "topconf_baselines",
                "special_settings": "baseline_family=topconf",
            },
        )()
        command = topconf_suite.build_command(args)
        command_text = " ".join(command)
        self.assertIn("patchtst_baseline", command_text)
        self.assertIn("timesnet_baseline", command_text)
        self.assertIn("thread_transformer_baseline", command_text)

    def test_collate_generates_topconf_features(self) -> None:
        sample = self._write_minimal_dataset()
        dataset = PHEMEForecastDataset(sample)
        batch = collate_forecast_batch([dataset[0]])
        self.assertEqual(tuple(batch["binned_time_series"].shape), (1, 8, 8))
        self.assertEqual(tuple(batch["binned_time_series_mask"].shape), (1, 8))
        self.assertEqual(len(batch["reply_depths"][0]), 2)
        self.assertEqual(len(batch["reply_parent_positions"][0]), 2)
        self.assertEqual(len(batch["reply_time_deltas"][0]), 2)

    def test_new_models_support_minimal_forward(self) -> None:
        sample = self._write_minimal_dataset()
        dataset = PHEMEForecastDataset(sample)
        batch = collate_forecast_batch([dataset[0]])
        for model_name in ("patchtst_baseline", "timesnet_baseline", "thread_transformer_baseline"):
            model = build_model(
                model_name,
                hidden_dim=32,
                vocab_size=1000,
                dropout=0.1,
                affect_state_dim=16,
                num_bins=8,
                max_replies=16,
                time_series_dim=8,
                patch_len=2,
                stride=1,
                n_heads=4,
                n_layers=2,
            )
            output = model_forward(model, model_name, batch)
            self.assertIn("predicted_future_neg_ratio", output)
            self.assertIn("predicted_future_majority_logits", output)
            self.assertEqual(output["predicted_future_neg_ratio"].shape[0], 1)

    def _write_minimal_dataset(self) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "sample.jsonl"
        record = {
            "thread_id": "root",
            "event_name": "event_a",
            "split": "train",
            "observation_ratio": 0.5,
            "source_text": "source post",
            "conversation_tree": {"r1": "root", "r2": "r1"},
            "observed_replies": [
                {
                    "id": "r1",
                    "parent_id": "root",
                    "text": "bad warning",
                    "created_at": "2020-01-01T00:00:00+00:00",
                    "sentiment_label": "negative",
                },
                {
                    "id": "r2",
                    "parent_id": "r1",
                    "text": "good support",
                    "created_at": "2020-01-01T00:05:00+00:00",
                    "sentiment_label": "positive",
                },
            ],
            "forecast_replies": [],
            "observed_neg_ratio": 0.5,
            "observed_neu_ratio": 0.0,
            "observed_pos_ratio": 0.5,
            "observed_majority_sentiment": "negative",
            "future_neg_ratio": 0.5,
            "future_neu_ratio": 0.25,
            "future_pos_ratio": 0.25,
            "future_majority_sentiment": "negative",
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
