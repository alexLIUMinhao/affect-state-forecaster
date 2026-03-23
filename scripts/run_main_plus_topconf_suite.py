from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_MODELS = ("text_baseline", "temporal_baseline", "structure_baseline", "affect_state_forecaster")
TOPCONF_MODELS = ("patchtst_baseline", "timesnet_baseline", "thread_transformer_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main baselines plus top-conference-inspired baselines in one tracked suite.")
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_test.jsonl"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="main_plus_topconf")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/run_experiment_suite.py",
        "--train_path",
        str(args.train_path),
        "--val_path",
        str(args.val_path),
        "--test_path",
        str(args.test_path),
        "--models",
        *(list(MAIN_MODELS) + list(TOPCONF_MODELS)),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--runs_root",
        str(args.runs_root),
        "--experiments_root",
        str(args.experiments_root),
        "--tag",
        args.tag_prefix,
        "--experiment_type",
        "主实验+顶会Baseline",
        "--question",
        "统一训练设置下，现有方法与顶会可适配baseline的相对排序如何。",
        "--success_criteria",
        "输出可直接进入论文主表的统一对比结果，并识别最强可复现baseline。",
        "--failure_criteria",
        "若新增baseline训练不稳定或整体无增益，则回退到主实验配置。",
        "--special_settings",
        "baseline_family=main_plus_topconf",
    ]


def main() -> None:
    args = parse_args()
    result = subprocess.run(build_command(args), cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
