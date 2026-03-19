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


if __name__ == "__main__":
    unittest.main()
