from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first-wave ablations for Affect-State Forecaster.")
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_50_train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_50_val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_50_test.jsonl"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs_root", type=Path, default=Path("runs/ablations"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def variant_matrix() -> list[tuple[str, list[str]]]:
    return [
        ("asf_full", []),
        ("asf_no_affect_state", ["--disable_affect_state"]),
        ("asf_no_structure", ["--disable_structure"]),
        ("asf_no_temporal", ["--disable_temporal"]),
        ("asf_source_only", ["--input_view", "source_only"]),
        ("asf_replies_only", ["--input_view", "replies_only"]),
    ]


def train_eval(args: argparse.Namespace, variant_name: str, extra_train_args: list[str]) -> None:
    artifact_dir = args.runs_root / variant_name / "artifacts"
    eval_dir = args.runs_root / variant_name / "eval"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "src/train.py",
        "--train_path",
        str(args.train_path),
        "--val_path",
        str(args.val_path),
        "--model",
        "affect_state_forecaster",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(artifact_dir),
        *extra_train_args,
    ]
    subprocess.run(train_cmd, cwd=PROJECT_ROOT, check=True)

    prefix = f"affect_state_forecaster_{args.train_path.stem}"
    eval_cmd = [
        sys.executable,
        "src/evaluate.py",
        "--data_path",
        str(args.test_path),
        "--model",
        "affect_state_forecaster",
        "--model_path",
        str(artifact_dir / f"{prefix}.pt"),
        "--config_path",
        str(artifact_dir / f"{prefix}.json"),
        "--device",
        args.device,
        "--output_dir",
        str(eval_dir),
    ]
    subprocess.run(eval_cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for variant_name, extra_train_args in variant_matrix():
        train_eval(args, variant_name, extra_train_args)


if __name__ == "__main__":
    main()
