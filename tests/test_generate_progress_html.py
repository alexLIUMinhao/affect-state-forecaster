from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "skills/research-progress-html/scripts/generate_progress_html.py"
SPEC = importlib.util.spec_from_file_location("generate_progress_html", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class GenerateProgressHtmlTests(unittest.TestCase):
    def test_intro_mentions_affect_state_when_ours_supported(self) -> None:
        context = {
            "dataset": "PHEME",
            "task": "以 future_neg_ratio 为主任务，预测未来窗口中的负面情绪比例",
            "method": "Affect-State Forecaster",
        }
        evidence = {
            "best_model": "affect_state_forecaster",
            "best_mae": 0.0917,
            "mae_gain_vs_text": 0.1821,
            "ours_vs_structure": 0.0269,
            "ours_vs_temporal": 0.0888,
            "evidence_status": "ours_supported",
            "anomalies": [],
        }

        paragraphs = MODULE.build_introduction(context, evidence)

        self.assertIn("affect-state 建模获得了初步支持", "".join(paragraphs))
        self.assertIn("相对文本基线的 MAE 改进", paragraphs[1])

    def test_intro_is_conservative_when_structure_baseline_leads(self) -> None:
        context = {
            "dataset": "PHEME",
            "task": "以 future_neg_ratio 为主任务，预测未来窗口中的负面情绪比例",
            "method": "Affect-State Forecaster",
        }
        evidence = {
            "best_model": "structure_baseline",
            "best_mae": 0.1186,
            "mae_gain_vs_text": 0.1552,
            "ours_vs_structure": -0.0270,
            "ours_vs_temporal": 0.0810,
            "evidence_status": "diagnostic_needed",
            "anomalies": ["结构基线领先，说明结构信息有效，但 affect-state 叙事尚未闭合。"],
        }

        paragraphs = MODULE.build_introduction(context, evidence)
        joined = "".join(paragraphs)

        self.assertIn("任务本身的可做性", joined)
        self.assertIn("仍需进一步诊断", joined)
        self.assertNotIn("初步支持", joined)

    def test_metric_reading_handles_split_winners(self) -> None:
        evidence = {
            "best_model": "structure_baseline",
            "best_mae_model": "structure_baseline",
            "best_mae": 0.1170,
            "best_rmse_model": "affect_state_forecaster",
            "best_rmse": 0.1775,
            "best_corr_model": "affect_state_forecaster",
            "mae_gain_vs_text": 0.0608,
            "ours_vs_structure": -0.0125,
            "ours_vs_temporal": 0.0395,
            "evidence_status": "diagnostic_needed",
            "anomalies": [],
        }
        results = {
            "models": {
                "structure_baseline": {"metrics": {"pearson": -0.2484, "spearman": 0.0661}},
                "affect_state_forecaster": {"metrics": {"pearson": 0.0126, "spearman": 0.2499}},
            }
        }
        reading = MODULE.build_metric_reading(evidence, results)
        self.assertIn("MAE 更低", reading)
        self.assertIn("RMSE 更低", reading)
        self.assertIn("相关性偏弱", reading)

    def test_generate_html_contains_table_and_plan(self) -> None:
        args = argparse.Namespace(
            idea_path=PROJECT_ROOT / "idea.md",
            experiments_root=PROJECT_ROOT / "experiments",
            output_html=PROJECT_ROOT / "experiments/html/test_paper_progress.html",
            mode="auto",
            result_source="structured",
        )

        html_text, _context, _evidence = MODULE.generate_html(args)

        self.assertIn("下一步实验计划", html_text)
        self.assertIn("Compared Baselines", html_text)
        self.assertIn("affect_state_forecaster", html_text)
        self.assertIn("当前模型架构与 I/O", html_text)
        self.assertIn("模型架构", html_text)
        self.assertIn("输入", html_text)
        self.assertIn("输出", html_text)
        self.assertIn("MAE", html_text)
        self.assertIn("RMSE", html_text)
        self.assertIn("Pearson", html_text)
        self.assertIn("Spearman", html_text)
        self.assertIn("Metric Reading", html_text)

    def test_reason_diagnosis_marks_fusion_as_strong_when_gate_beats_full(self) -> None:
        results = {
            "models": {
                "text_baseline": {"metrics": {"mae": 0.18, "rmse": 0.23, "pearson": 0.02, "spearman": 0.03}},
                "structure_baseline": {"metrics": {"mae": 0.12, "rmse": 0.18, "pearson": 0.01, "spearman": 0.04}},
                "temporal_baseline": {"metrics": {"mae": 0.17, "rmse": 0.22, "pearson": 0.00, "spearman": 0.01}},
                "affect_state_forecaster": {"metrics": {"mae": 0.13, "rmse": 0.17, "pearson": 0.05, "spearman": 0.20}},
            },
            "fusion_diagnostic": {
                "variants": [
                    {"model_name": "affect_state_forecaster", "fusion_variant": "full", "metrics": {"mae": 0.13, "rmse": 0.17, "pearson": 0.05, "spearman": 0.20}},
                    {"model_name": "affect_state_forecaster", "fusion_variant": "scalar_gate", "metrics": {"mae": 0.11, "rmse": 0.169, "pearson": 0.06, "spearman": 0.24}},
                ]
            },
        }
        evidence = MODULE.compute_evidence(results)
        diagnosis = MODULE.compute_reason_diagnosis(results, evidence)
        fusion_item = next(item for item in diagnosis if item["title"] == "融合筛选问题")
        self.assertEqual(fusion_item["status"], "当前证据最强")
        self.assertIn("scalar_gate", fusion_item["reason"])


if __name__ == "__main__":
    unittest.main()
