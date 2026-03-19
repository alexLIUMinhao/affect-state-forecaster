from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fusion diagnostic experiments for Affect-State Forecaster.")
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_test.jsonl"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="fusion_diagnostic")
    return parser.parse_args()


def slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in normalized.split("_") if part)


def default_run_id(tag: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{slugify(tag)}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def variant_matrix() -> list[tuple[str, str, dict[str, str]]]:
    return [
        ("structure_baseline", "baseline", {}),
        ("affect_state_forecaster", "asf_full", {"fusion_variant": "full"}),
        ("affect_state_forecaster", "asf_scalar_gate", {"fusion_variant": "scalar_gate"}),
        ("affect_state_forecaster", "asf_vector_gate", {"fusion_variant": "vector_gate"}),
        ("affect_state_forecaster", "asf_softmax_router", {"fusion_variant": "softmax_router"}),
        ("affect_state_forecaster", "asf_structure_gate_only", {"fusion_variant": "structure_gate_only"}),
        ("affect_state_forecaster", "asf_source_gate_only", {"fusion_variant": "source_gate_only"}),
        ("affect_state_forecaster", "asf_reply_gate_only", {"fusion_variant": "reply_gate_only"}),
    ]


def run_variant(
    args: argparse.Namespace,
    run_root: Path,
    model_name: str,
    variant_name: str,
    overrides: dict[str, str],
) -> None:
    artifact_dir = run_root / variant_name / "artifacts"
    eval_dir = run_root / variant_name / "eval"
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
        model_name,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--capacity_group",
        "default",
        "--output_dir",
        str(artifact_dir),
    ]
    for key, value in overrides.items():
        train_cmd.extend([f"--{key}", value])
    subprocess.run(train_cmd, cwd=PROJECT_ROOT, check=True)

    prefix = f"{model_name}_{args.train_path.stem}"
    eval_cmd = [
        sys.executable,
        "src/evaluate.py",
        "--data_path",
        str(args.test_path),
        "--model",
        model_name,
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


def summary_row(run_root: Path, variant_name: str, model_name: str, fusion_variant: str) -> dict[str, object]:
    artifact_dir = run_root / variant_name / "artifacts"
    eval_dir = run_root / variant_name / "eval"
    config_rows = sorted(path for path in artifact_dir.glob("*.json") if not path.name.endswith("_summary.json"))
    config = {}
    if config_rows:
        import json

        config = json.loads(config_rows[0].read_text(encoding="utf-8"))
    eval_rows = read_csv_rows(eval_dir / "results_summary.csv")
    overall = next((row for row in eval_rows if row.get("group_name") == "overall"), {})
    return {
        "model_name": model_name,
        "fusion_variant": fusion_variant,
        "param_count": config.get("param_count", ""),
        "mae": overall.get("mae", ""),
        "rmse": overall.get("rmse", ""),
        "pearson": overall.get("pearson", ""),
        "spearman": overall.get("spearman", ""),
        "gate_source_mean": overall.get("gate_source_mean", ""),
        "gate_temporal_mean": overall.get("gate_temporal_mean", ""),
        "gate_structure_mean": overall.get("gate_structure_mean", ""),
    }


def write_summary(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "fusion_variant",
        "param_count",
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "gate_source_mean",
        "gate_temporal_mean",
        "gate_structure_mean",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_id = default_run_id(args.tag_prefix)
    run_root = args.runs_root / run_id
    rows: list[dict[str, object]] = []

    for model_name, variant_name, overrides in variant_matrix():
        run_variant(args, run_root, model_name, variant_name, overrides)
        rows.append(summary_row(run_root, variant_name, model_name, overrides.get("fusion_variant", variant_name)))

    summary_path = args.experiments_root / "records" / f"{run_id}_fusion_diagnostic_summary.csv"
    write_summary(summary_path, rows)
    print(f"run_id={run_id}")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
