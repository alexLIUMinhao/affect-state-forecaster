from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_reporting import build_manifest, ensure_experiment_paths, load_hypothesis_config, persist_manifest_outputs, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import existing runs/ results into the experiment records system.")
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--config_path", type=Path, default=Path("configs/research_hypotheses.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_experiment_paths(args.experiments_root)
    config = load_hypothesis_config(args.config_path)

    run_roots = []
    if args.run_name:
        run_roots = [args.runs_root / args.run_name]
    else:
        run_roots = sorted(path for path in args.runs_root.iterdir() if path.is_dir())

    for run_root in run_roots:
        run_id = args.run_id if args.run_id else f"import_{slugify(run_root.name)}"
        plan = {
            "run_id": run_id,
            "experiment_type": "历史结果导入",
            "target_hypotheses": ["H1", "H2", "H3", "H4"],
            "question": f"回填历史实验 `{run_root.name}`，确认其对当前研究主线的支持程度。",
            "success_criteria": "能够生成完整报告，并对 idea.md 给出可读判断。",
            "failure_criteria": "若历史结果缺失严重，则仅标记证据不足，不给过度结论。",
            "idea_sections": ["三、核心研究问题", "七、实验设计"],
            "dataset": str(run_root),
            "ratio_labels": [],
            "models": [],
            "epochs": "unknown",
            "batch_size": "unknown",
            "device": "unknown",
            "special_settings": "来自既有 runs 目录的历史导入",
        }
        manifest = build_manifest(run_id=run_id, run_root=run_root, config=config, experiment_plan=plan)
        persist_manifest_outputs(paths, manifest)
        print(f"synced_run={run_root}")
        print(f"saved_manifest={manifest['manifest_path']}")
        print(f"saved_report={manifest['report_path']}")


if __name__ == "__main__":
    main()
