from __future__ import annotations

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

from src.experiment_reporting import (
    build_index_html,
    classify_html_category,
    html_output_path,
)


class HtmlOrganizationTests(unittest.TestCase):
    def test_classify_html_category_maps_expected_run_types(self) -> None:
        self.assertEqual(classify_html_category("20260323_091059_main_plus_topconf"), "main")
        self.assertEqual(classify_html_category("20260323_data_analysis_report"), "data_analysis")
        self.assertEqual(classify_html_category("20260323_091019_topconf_baselines"), "topconf")
        self.assertEqual(classify_html_category("20260323_091235_ratio_sweep_ratio_30"), "ratio_sweep")
        self.assertEqual(classify_html_category("20260323_091509_cross_event_charliehebdo"), "cross_event")
        self.assertEqual(classify_html_category("20260323_092441_seed_sweep_seed_13"), "seed_sweep")
        self.assertEqual(classify_html_category("20260319_134508_smoke_ratio05"), "smoke")
        self.assertEqual(classify_html_category("20260319_211003_capacity_matched_main_default"), "diagnostics")
        self.assertEqual(classify_html_category("20260323_091059_main_plus_topconf_failure_case"), "diagnostics")
        self.assertEqual(classify_html_category("20260323_092808_paper_progress"), "progress")
        self.assertEqual(classify_html_category("import_20260323_091059_main_plus_topconf"), "imports")

    def test_html_output_path_uses_category_subdirectory(self) -> None:
        html_root = Path("experiments/html")
        path = html_output_path(html_root, "20260323_091509_cross_event_charliehebdo")
        self.assertEqual(path, html_root / "cross_event" / "20260323_091509_cross_event_charliehebdo.html")

    def test_root_index_contains_category_descriptions_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            html_root = Path(tmpdir) / "experiments" / "html"
            (html_root / "progress").mkdir(parents=True)
            (html_root / "data_analysis").mkdir(parents=True)
            (html_root / "progress" / "20260323_092808_paper_progress.html").write_text("<html></html>", encoding="utf-8")
            (html_root / "data_analysis" / "index.html").write_text("<html></html>", encoding="utf-8")
            (html_root / "data_analysis" / "feature_spaces.html").write_text("<html></html>", encoding="utf-8")
            rows = [
                {
                    "run_id": "20260323_091059_main_plus_topconf",
                    "created_at": "2026-03-23T09:10:59",
                    "experiment_goal": "main comparison",
                    "best_model": "affect_state_forecaster",
                    "best_mae": "0.1052",
                    "hypothesis_summary": "main summary",
                    "decision_action": "continue",
                    "html_path": str(html_root / "main" / "20260323_091059_main_plus_topconf.html"),
                }
            ]
            page = build_index_html(rows, html_root)
            self.assertIn("主实验", page)
            self.assertIn("数据分析", page)
            self.assertIn("论文进展", page)
            self.assertIn("main/index.html", page)
            self.assertIn("data_analysis/index.html", page)
            self.assertIn("progress/index.html", page)


if __name__ == "__main__":
    unittest.main()
