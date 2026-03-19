from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("text_baseline", "temporal_baseline", "structure_baseline", "affect_state_forecaster")
RATIO_CHOICES = ("30", "50", "70")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main four-model comparison across multiple observation ratios.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--ratios", nargs="+", default=list(RATIO_CHOICES), choices=list(RATIO_CHOICES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="ratio_sweep")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def dataset_triplet(data_dir: Path, ratio: str) -> tuple[Path, Path, Path]:
    train_path = data_dir / f"pheme_forecast_ratio_{ratio}_train.jsonl"
    val_path = data_dir / f"pheme_forecast_ratio_{ratio}_val.jsonl"
    test_path = data_dir / f"pheme_forecast_ratio_{ratio}_test.jsonl"
    return train_path, val_path, test_path


def build_command(args: argparse.Namespace, ratio: str) -> list[str]:
    train_path, val_path, test_path = dataset_triplet(args.data_dir, ratio)
    return [
        sys.executable,
        "scripts/run_experiment_suite.py",
        "--train_path",
        str(train_path),
        "--val_path",
        str(val_path),
        "--test_path",
        str(test_path),
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
        f"{args.tag_prefix}_ratio_{ratio}",
        "--question",
        f"验证四个模型在 observation ratio {ratio} 下对 future_neg_ratio 的预测表现。",
        "--success_criteria",
        "affect_state_forecaster 在主要回归指标上稳定优于文本基线，并最好保持最优。",
        "--failure_criteria",
        "如果 affect_state_forecaster 不能稳定领先，则转入诊断与消融实验。",
        "--special_settings",
        f"ratio={ratio}",
        "--models",
        *args.models,
    ]


def main() -> None:
    args = parse_args()
    for ratio in args.ratios:
        command = build_command(args, ratio)
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0 and not args.continue_on_error:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
