from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.sentiment import WEAK_LABELERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark and model comparisons across weak sentiment labelers.")
    parser.add_argument("--threads_path", type=Path, default=Path("data/processed/pheme_threads.jsonl"))
    parser.add_argument("--output_root", type=Path, default=Path("data/processed/robustness"))
    parser.add_argument("--ratio", type=str, default="50", choices=["30", "50", "70"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def ratio_float(ratio: str) -> str:
    return {"30": "0.3", "50": "0.5", "70": "0.7"}[ratio]


def ratio_suffix(ratio: str) -> str:
    return {"30": "03", "50": "05", "70": "07"}[ratio]


def main() -> None:
    args = parse_args()
    suffix = ratio_suffix(args.ratio)
    for labeler in WEAK_LABELERS:
        label_output = args.output_root / f"pheme_threads_{labeler}.jsonl"
        benchmark_dir = args.output_root / labeler
        subprocess.run(
            [
                sys.executable,
                "scripts/label_pheme_sentiment.py",
                "--input_path",
                str(args.threads_path),
                "--output_path",
                str(label_output),
                "--labeler",
                labeler,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/build_forecasting_benchmark.py",
                "--input_path",
                str(label_output),
                "--output_dir",
                str(benchmark_dir),
                "--output_prefix",
                "pheme_forecast",
                "--observation_ratios",
                ratio_float(args.ratio),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/run_experiment_suite.py",
                "--train_path",
                str(benchmark_dir / f"pheme_forecast_ratio_{suffix}_train.jsonl"),
                "--val_path",
                str(benchmark_dir / f"pheme_forecast_ratio_{suffix}_val.jsonl"),
                "--test_path",
                str(benchmark_dir / f"pheme_forecast_ratio_{suffix}_test.jsonl"),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--tag",
                f"labeler_{labeler}_ratio_{suffix}",
                "--question",
                f"比较 weak labeler={labeler} 时主实验结果是否稳定。",
                "--success_criteria",
                "主结论对不同弱标注策略保持一致。",
                "--failure_criteria",
                "若主结论随 labeler 明显波动，则需要先修正弱标注策略。",
                "--special_settings",
                f"labeler={labeler}",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
