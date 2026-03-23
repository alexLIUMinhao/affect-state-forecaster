from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_model, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run capacity-control experiments for the best fusion gate variants.")
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_test.jsonl"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="fusion_capacity_control")
    parser.add_argument("--target_param_tolerance", type=float, default=0.05)
    parser.add_argument("--search_hidden_dim", nargs="+", type=int, default=[64, 72, 80, 88, 96, 104, 112, 120, 128])
    parser.add_argument("--search_affect_state_dim", nargs="+", type=int, default=[8, 16, 24, 32, 40])
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


def model_param_count(fusion_variant: str = "full", hidden_dim: int = 128, affect_state_dim: int = 32) -> int:
    model = build_model(
        "affect_state_forecaster",
        hidden_dim=hidden_dim,
        vocab_size=20000,
        dropout=0.1,
        affect_state_dim=affect_state_dim,
        fusion_variant=fusion_variant,
    )
    return count_trainable_parameters(model)


def structure_param_count() -> int:
    model = build_model(
        "structure_baseline",
        hidden_dim=128,
        vocab_size=20000,
        dropout=0.1,
        affect_state_dim=32,
    )
    return count_trainable_parameters(model)


def choose_matched_config(args: argparse.Namespace, fusion_variant: str) -> dict[str, int]:
    target = structure_param_count()
    best: tuple[int, dict[str, int]] | None = None
    best_within: tuple[int, dict[str, int]] | None = None
    for hidden_dim in args.search_hidden_dim:
        for affect_state_dim in args.search_affect_state_dim:
            param_count = model_param_count(fusion_variant=fusion_variant, hidden_dim=hidden_dim, affect_state_dim=affect_state_dim)
            gap = abs(param_count - target)
            payload = {
                "hidden_dim": hidden_dim,
                "affect_state_dim": affect_state_dim,
                "param_count": param_count,
                "target_param_count": target,
            }
            if best is None or gap < best[0]:
                best = (gap, payload)
            if gap / target <= args.target_param_tolerance:
                if best_within is None or gap < best_within[0]:
                    best_within = (gap, payload)
    return (best_within or best)[1]


def variant_specs() -> list[tuple[str, str]]:
    return [
        ("asf_full", "full"),
        ("asf_vector_gate", "vector_gate"),
        ("asf_source_gate_only", "source_gate_only"),
    ]


def run_variant(
    args: argparse.Namespace,
    run_root: Path,
    variant_name: str,
    fusion_variant: str,
    hidden_dim: int,
    affect_state_dim: int,
    capacity_group: str,
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
        "affect_state_forecaster",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--hidden_dim",
        str(hidden_dim),
        "--affect_state_dim",
        str(affect_state_dim),
        "--fusion_variant",
        fusion_variant,
        "--capacity_group",
        capacity_group,
        "--output_dir",
        str(artifact_dir),
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


def run_structure_baseline(args: argparse.Namespace, run_root: Path, capacity_group: str) -> None:
    artifact_dir = run_root / "structure_baseline" / "artifacts"
    eval_dir = run_root / "structure_baseline" / "eval"
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
        "structure_baseline",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--capacity_group",
        capacity_group,
        "--output_dir",
        str(artifact_dir),
    ]
    subprocess.run(train_cmd, cwd=PROJECT_ROOT, check=True)
    prefix = f"structure_baseline_{args.train_path.stem}"
    eval_cmd = [
        sys.executable,
        "src/evaluate.py",
        "--data_path",
        str(args.test_path),
        "--model",
        "structure_baseline",
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


def collect_summary(run_root: Path, capacity_group: str, output_path: Path) -> None:
    fieldnames = [
        "capacity_group",
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
    rows: list[dict[str, object]] = []
    for variant_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        artifact_dir = variant_dir / "artifacts"
        eval_dir = variant_dir / "eval"
        config_path = next(path for path in artifact_dir.glob("*.json") if not path.name.endswith("_summary.json"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        summary_rows = read_csv_rows(eval_dir / "results_summary.csv")
        overall = next(row for row in summary_rows if row.get("group_name") == "overall")
        rows.append(
            {
                "capacity_group": capacity_group,
                "model_name": config.get("model", ""),
                "fusion_variant": config.get("fusion_variant", variant_dir.name),
                "param_count": config.get("param_count", ""),
                "mae": overall.get("mae", ""),
                "rmse": overall.get("rmse", ""),
                "pearson": overall.get("pearson", ""),
                "spearman": overall.get("spearman", ""),
                "gate_source_mean": overall.get("gate_source_mean", ""),
                "gate_temporal_mean": overall.get("gate_temporal_mean", ""),
                "gate_structure_mean": overall.get("gate_structure_mean", ""),
            }
        )
    write_mode = "a" if output_path.exists() else "w"
    with output_path.open(write_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_id = default_run_id(args.tag_prefix)
    summary_path = args.experiments_root / "records" / f"{base_id}_fusion_capacity_summary.csv"

    for capacity_group in ("default", "matched"):
        run_root = args.runs_root / f"{base_id}_{capacity_group}"
        run_structure_baseline(args, run_root, capacity_group)
        for variant_name, fusion_variant in variant_specs():
            hidden_dim = 128
            affect_state_dim = 32
            if capacity_group == "matched":
                matched = choose_matched_config(args, fusion_variant)
                hidden_dim = matched["hidden_dim"]
                affect_state_dim = matched["affect_state_dim"]
            run_variant(args, run_root, variant_name, fusion_variant, hidden_dim, affect_state_dim, capacity_group)
        collect_summary(run_root, capacity_group, summary_path)

    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
