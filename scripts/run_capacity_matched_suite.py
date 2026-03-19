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


DEFAULT_MODELS = ("text_baseline", "temporal_baseline", "structure_baseline", "affect_state_forecaster")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run default and capacity-matched main experiments.")
    parser.add_argument("--train_path", type=Path, required=True)
    parser.add_argument("--val_path", type=Path, default=Path(""))
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--target_model", type=str, default="structure_baseline", choices=list(DEFAULT_MODELS))
    parser.add_argument("--target_param_tolerance", type=float, default=0.05)
    parser.add_argument("--search_hidden_dim", nargs="+", type=int, default=[64, 72, 80, 88, 96, 104, 112, 120, 128])
    parser.add_argument("--search_affect_state_dim", nargs="+", type=int, default=[8, 16, 24, 32, 40])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="capacity_matched_main")
    return parser.parse_args()


def slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in normalized.split("_") if part)


def default_run_id(tag: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{slugify(tag)}"


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def model_param_count(model_name: str, hidden_dim: int = 128, affect_state_dim: int = 32) -> int:
    model = build_model(
        model_name,
        hidden_dim=hidden_dim,
        vocab_size=20000,
        dropout=0.1,
        affect_state_dim=affect_state_dim,
    )
    return count_trainable_parameters(model)


def choose_matched_asf_config(args: argparse.Namespace) -> dict[str, int]:
    target_param_count = model_param_count(args.target_model)
    best_within: tuple[int, dict[str, int]] | None = None
    best_overall: tuple[int, dict[str, int]] | None = None

    for hidden_dim in args.search_hidden_dim:
        for affect_state_dim in args.search_affect_state_dim:
            param_count = model_param_count("affect_state_forecaster", hidden_dim=hidden_dim, affect_state_dim=affect_state_dim)
            gap = abs(param_count - target_param_count)
            config = {
                "hidden_dim": hidden_dim,
                "affect_state_dim": affect_state_dim,
                "param_count": param_count,
                "target_param_count": target_param_count,
            }
            if best_overall is None or gap < best_overall[0]:
                best_overall = (gap, config)
            if gap / target_param_count <= args.target_param_tolerance:
                if best_within is None or gap < best_within[0]:
                    best_within = (gap, config)

    selected = best_within[1] if best_within else best_overall[1]
    if selected is None:
        raise ValueError("Unable to find a capacity-matched ASF configuration.")
    return selected


def write_model_configs(path: Path, matched_config: dict[str, int]) -> None:
    payload = {
        "affect_state_forecaster": {
            "hidden_dim": matched_config["hidden_dim"],
            "affect_state_dim": matched_config["affect_state_dim"],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_suite(
    args: argparse.Namespace,
    run_id: str,
    model_configs_path: Path | None,
    special_settings: str,
    capacity_group: str,
) -> None:
    command = [
        sys.executable,
        "scripts/run_experiment_suite.py",
        "--train_path",
        str(args.train_path),
        "--test_path",
        str(args.test_path),
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
        "--run_id",
        run_id,
        "--runs_root",
        str(args.runs_root),
        "--experiments_root",
        str(args.experiments_root),
        "--tag",
        args.tag_prefix,
        "--experiment_type",
        "主实验",
        "--question",
        "在固定任务、数据划分和训练设置下，比较默认配置与参数量对齐配置是否改变 affect-state 方法的相对优势。",
        "--success_criteria",
        "参数量对齐后能够判断 affect-state 优势是否仍然成立，并输出可解释的对比表。",
        "--failure_criteria",
        "若参数量对齐后结论仍混乱，则先停止扩模，转入容量与校准诊断。",
        "--special_settings",
        special_settings,
    ]
    if str(args.val_path):
        command.extend(["--val_path", str(args.val_path)])
    if model_configs_path is not None:
        command.extend(["--model_configs_path", str(model_configs_path)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def load_manifest(path: Path) -> dict[str, object]:
    return read_json(path)


def param_count_from_run(run_root: Path, model_name: str) -> int | None:
    artifact_dir = run_root / model_name / "artifacts"
    config_paths = sorted(artifact_dir.glob("*.json"))
    for path in config_paths:
        if path.name.endswith("_summary.json"):
            continue
        payload = read_json(path)
        param_count = payload.get("param_count")
        if isinstance(param_count, int):
            return param_count
    return None


def write_summary_csv(
    output_path: Path,
    default_manifest: dict[str, object],
    matched_manifest: dict[str, object],
    default_run_root: Path,
    matched_run_root: Path,
) -> None:
    fieldnames = [
        "run_id",
        "capacity_group",
        "model_name",
        "param_count",
        "mae",
        "rmse",
        "pearson",
        "spearman",
    ]
    rows: list[dict[str, object]] = []
    for capacity_group, manifest, run_root in (
        ("default", default_manifest, default_run_root),
        ("matched", matched_manifest, matched_run_root),
    ):
        for model_entry in manifest.get("models", []):
            metrics = model_entry.get("overall_metrics", {})
            rows.append(
                {
                    "run_id": manifest.get("run_id", ""),
                    "capacity_group": capacity_group,
                    "model_name": model_entry.get("model", ""),
                    "param_count": param_count_from_run(run_root, model_entry.get("model", "")) or "",
                    "mae": metrics.get("mae", ""),
                    "rmse": metrics.get("rmse", ""),
                    "pearson": metrics.get("pearson", ""),
                    "spearman": metrics.get("spearman", ""),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_id = default_run_id(args.tag_prefix)
    default_run_id_value = f"{base_id}_default"
    matched_run_id_value = f"{base_id}_matched"
    matched_config = choose_matched_asf_config(args)
    model_configs_path = args.experiments_root / "manifests" / f"{base_id}_matched_model_configs.json"
    write_model_configs(model_configs_path, matched_config)

    run_suite(
        args,
        default_run_id_value,
        None,
        f"capacity_group=default; target_model={args.target_model}; target_param_tolerance={args.target_param_tolerance}",
        "default",
    )
    run_suite(
        args,
        matched_run_id_value,
        model_configs_path,
        (
            f"capacity_group=matched; target_model={args.target_model}; "
            f"target_param_tolerance={args.target_param_tolerance}; "
            f"matched_asf_hidden_dim={matched_config['hidden_dim']}; "
            f"matched_asf_affect_state_dim={matched_config['affect_state_dim']}; "
            f"matched_asf_param_count={matched_config['param_count']}; "
            f"target_param_count={matched_config['target_param_count']}"
        ),
        "matched",
    )

    manifest_dir = args.experiments_root / "manifests"
    default_manifest = load_manifest(manifest_dir / f"{default_run_id_value}.json")
    matched_manifest = load_manifest(manifest_dir / f"{matched_run_id_value}.json")
    summary_csv = args.experiments_root / "records" / f"{base_id}_capacity_summary.csv"
    write_summary_csv(
        summary_csv,
        default_manifest,
        matched_manifest,
        args.runs_root / default_run_id_value,
        args.runs_root / matched_run_id_value,
    )

    print(f"default_run_id={default_run_id_value}")
    print(f"matched_run_id={matched_run_id_value}")
    print(f"matched_asf_hidden_dim={matched_config['hidden_dim']}")
    print(f"matched_asf_affect_state_dim={matched_config['affect_state_dim']}")
    print(f"matched_asf_param_count={matched_config['param_count']}")
    print(f"target_param_count={matched_config['target_param_count']}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
