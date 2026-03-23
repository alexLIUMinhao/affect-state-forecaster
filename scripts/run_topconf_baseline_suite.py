from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPCONF_MODELS = ("patchtst_baseline", "timesnet_baseline", "thread_transformer_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-conference-inspired baselines on the forecasting benchmark.")
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_test.jsonl"))
    parser.add_argument("--models", nargs="+", default=list(TOPCONF_MODELS))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="topconf_baselines")
    parser.add_argument("--special_settings", type=str, default="baseline_family=topconf")
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
        *args.models,
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
        "顶会Baseline对比",
        "--question",
        "顶会可适配baseline在PHEME forecasting设定下是否优于现有自研基础线。",
        "--success_criteria",
        "至少一个顶会baseline在主要回归指标上进入前二，并可稳定复现。",
        "--failure_criteria",
        "若新增baseline全部落后于现有强基线，则停止继续扩展。",
        "--special_settings",
        args.special_settings,
    ]


def main() -> None:
    args = parse_args()
    result = subprocess.run(build_command(args), cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
