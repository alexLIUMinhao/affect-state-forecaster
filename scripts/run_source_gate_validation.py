from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_fusion_capacity_control import choose_matched_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate source_gate_only stability across seeds under default and matched capacity.")
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_05_test.jsonl"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 77])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="source_gate_validation")
    parser.add_argument("--target_param_tolerance", type=float, default=0.05)
    parser.add_argument("--search_hidden_dim", nargs="+", type=int, default=[64, 72, 80, 88, 96, 104, 112, 120, 128])
    parser.add_argument("--search_affect_state_dim", nargs="+", type=int, default=[8, 16, 24, 32, 40])
    return parser.parse_args()


def slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in normalized.split("_") if part)


def default_run_id(tag: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(tag)}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_model(
    args: argparse.Namespace,
    run_root: Path,
    model_name: str,
    variant_name: str,
    capacity_group: str,
    seed: int,
    overrides: dict[str, str | int],
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
        str(seed),
        "--capacity_group",
        capacity_group,
        "--output_dir",
        str(artifact_dir),
    ]
    for key, value in overrides.items():
        train_cmd.extend([f"--{key}", str(value)])
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


def collect_row(run_root: Path, variant_name: str, capacity_group: str, seed: int) -> dict[str, object]:
    artifact_dir = run_root / variant_name / "artifacts"
    eval_dir = run_root / variant_name / "eval"
    config_path = next(path for path in artifact_dir.glob("*.json") if not path.name.endswith("_summary.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_rows = read_csv_rows(eval_dir / "results_summary.csv")
    overall = next(row for row in summary_rows if row.get("group_name") == "overall")
    return {
        "seed": seed,
        "capacity_group": capacity_group,
        "model_name": config.get("model", ""),
        "fusion_variant": config.get("fusion_variant", "baseline"),
        "param_count": float(config.get("param_count", 0) or 0),
        "mae": float(overall.get("mae", 0) or 0),
        "rmse": float(overall.get("rmse", 0) or 0),
        "pearson": float(overall.get("pearson", 0) or 0),
        "spearman": float(overall.get("spearman", 0) or 0),
        "gate_source_mean": float(overall.get("gate_source_mean", 0) or 0),
        "gate_temporal_mean": float(overall.get("gate_temporal_mean", 0) or 0),
        "gate_structure_mean": float(overall.get("gate_structure_mean", 0) or 0),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["capacity_group"]), str(row["model_name"]), str(row["fusion_variant"]))
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    metric_keys = [
        "param_count",
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "gate_source_mean",
        "gate_temporal_mean",
        "gate_structure_mean",
    ]
    for (capacity_group, model_name, fusion_variant), items in sorted(grouped.items()):
        aggregate = {
            "capacity_group": capacity_group,
            "model_name": model_name,
            "fusion_variant": fusion_variant,
            "seeds": ",".join(str(int(item["seed"])) for item in items),
        }
        for key in metric_keys:
            values = [float(item[key]) for item in items]
            aggregate[key] = mean(values)
            aggregate[f"{key}_std"] = stdev(values)
        aggregates.append(aggregate)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_id = default_run_id(args.tag_prefix)
    raw_rows: list[dict[str, object]] = []
    full_matched = choose_matched_config(args, "full")
    source_matched = choose_matched_config(args, "source_gate_only")

    for capacity_group in ("default", "matched"):
        for seed in args.seeds:
            seed_root = args.runs_root / f"{run_id}_{capacity_group}_seed_{seed}"
            run_model(args, seed_root, "structure_baseline", "structure_baseline", capacity_group, seed, {})

            full_overrides: dict[str, str | int] = {"fusion_variant": "full"}
            source_overrides: dict[str, str | int] = {"fusion_variant": "source_gate_only"}
            if capacity_group == "matched":
                full_overrides.update(
                    {"hidden_dim": full_matched["hidden_dim"], "affect_state_dim": full_matched["affect_state_dim"]}
                )
                source_overrides.update(
                    {"hidden_dim": source_matched["hidden_dim"], "affect_state_dim": source_matched["affect_state_dim"]}
                )

            run_model(args, seed_root, "affect_state_forecaster", "asf_full", capacity_group, seed, full_overrides)
            run_model(
                args,
                seed_root,
                "affect_state_forecaster",
                "asf_source_gate_only",
                capacity_group,
                seed,
                source_overrides,
            )

            raw_rows.append(collect_row(seed_root, "structure_baseline", capacity_group, seed))
            raw_rows.append(collect_row(seed_root, "asf_full", capacity_group, seed))
            raw_rows.append(collect_row(seed_root, "asf_source_gate_only", capacity_group, seed))

    raw_path = args.experiments_root / "records" / f"{run_id}_source_gate_validation_raw.csv"
    aggregate_path = args.experiments_root / "records" / f"{run_id}_source_gate_validation_summary.csv"
    raw_fields = [
        "seed",
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
    aggregate_fields = [
        "capacity_group",
        "model_name",
        "fusion_variant",
        "seeds",
        "param_count",
        "param_count_std",
        "mae",
        "mae_std",
        "rmse",
        "rmse_std",
        "pearson",
        "pearson_std",
        "spearman",
        "spearman_std",
        "gate_source_mean",
        "gate_source_mean_std",
        "gate_temporal_mean",
        "gate_temporal_mean_std",
        "gate_structure_mean",
        "gate_structure_mean_std",
    ]
    write_csv(raw_path, raw_rows, raw_fields)
    write_csv(aggregate_path, aggregate_rows(raw_rows), aggregate_fields)
    print(f"run_id={run_id}")
    print(f"raw_csv={raw_path}")
    print(f"summary_csv={aggregate_path}")


if __name__ == "__main__":
    main()
