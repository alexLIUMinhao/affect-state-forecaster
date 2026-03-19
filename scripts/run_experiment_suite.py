from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_reporting import (
    build_manifest,
    default_run_id,
    ensure_experiment_paths,
    load_hypothesis_config,
    persist_manifest_outputs,
    run_command_with_logging,
)


DEFAULT_MODELS = ("text_baseline", "temporal_baseline", "structure_baseline", "affect_state_forecaster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a server-side experiment suite with logging and post-run analysis.")
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--val_path", type=Path, default=Path(""))
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--tag", type=str, default="experiment")
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--config_path", type=Path, default=Path("configs/research_hypotheses.json"))
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or default_run_id(args.tag)
    run_root = args.runs_root / run_id
    paths = ensure_experiment_paths(args.experiments_root)
    config = load_hypothesis_config(args.config_path)
    log_path = paths.logs / f"{run_id}.log"

    for model in args.models:
        artifact_dir = run_root / model / "artifacts"
        eval_dir = run_root / model / "eval"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "src/train.py",
            "--train_path",
            str(args.train_path),
            "--model",
            model,
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--output_dir",
            str(artifact_dir),
        ]
        if str(args.val_path):
            train_cmd.extend(["--val_path", str(args.val_path)])
        train_code = run_command_with_logging(train_cmd, log_path, PROJECT_ROOT)
        if train_code != 0 and not args.continue_on_error:
            raise SystemExit(train_code)
        if train_code != 0:
            continue

        prefix = f"{model}_{args.train_path.stem}"
        eval_cmd = [
            sys.executable,
            "src/evaluate.py",
            "--data_path",
            str(args.test_path),
            "--model",
            model,
            "--model_path",
            str(artifact_dir / f"{prefix}.pt"),
            "--config_path",
            str(artifact_dir / f"{prefix}.json"),
            "--device",
            args.device,
            "--output_dir",
            str(eval_dir),
        ]
        eval_code = run_command_with_logging(eval_cmd, log_path, PROJECT_ROOT)
        if eval_code != 0 and not args.continue_on_error:
            raise SystemExit(eval_code)

    manifest = build_manifest(run_id=run_id, run_root=run_root, config=config, log_path=log_path)
    persist_manifest_outputs(paths, manifest)
    print(f"completed_run={run_id}")
    print(f"saved_manifest={manifest['manifest_path']}")
    print(f"saved_report={manifest['report_path']}")


if __name__ == "__main__":
    main()
