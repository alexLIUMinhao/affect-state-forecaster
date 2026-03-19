from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("text_baseline", "temporal_baseline", "structure_baseline", "affect_state_forecaster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat the same experiment setting across multiple random seeds.")
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_50_train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_50_val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_50_test.jsonl"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 77])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="seed_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        command = [
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
            "--seed",
            str(seed),
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
            f"{args.tag_prefix}_seed_{seed}",
            "--question",
            f"验证固定数据设置下 seed={seed} 时主实验结果是否稳定。",
            "--success_criteria",
            "affect_state_forecaster 的领先关系在不同随机种子下保持一致。",
            "--failure_criteria",
            "若不同 seed 下排序波动明显，则先做稳健性分析。",
            "--special_settings",
            f"seed={seed}",
        ]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
