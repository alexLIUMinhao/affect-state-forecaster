from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_reporting import (
    build_manifest,
    default_run_id,
    ensure_experiment_paths,
    finalize_log,
    initialize_log,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--tag", type=str, default="experiment")
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--config_path", type=Path, default=Path("configs/research_hypotheses.json"))
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--experiment_type", type=str, default="主实验")
    parser.add_argument("--target_hypotheses", nargs="+", default=["H1", "H2", "H3"])
    parser.add_argument("--question", type=str, default="验证当前设置下哪些模型最能预测 future_neg_ratio。")
    parser.add_argument("--success_criteria", type=str, default="至少一个模型优于文本基线，并减少当前研究不确定性。")
    parser.add_argument("--failure_criteria", type=str, default="若核心假设被削弱或结果无法解释，则转入诊断实验。")
    parser.add_argument("--idea_sections", nargs="+", default=["三、核心研究问题", "六、方法方案", "七、实验设计"])
    parser.add_argument("--special_settings", type=str, default="")
    parser.add_argument("--model_configs_path", type=Path, default=Path(""))
    parser.add_argument("--capacity_group", type=str, default="default")
    return parser.parse_args()


OVERRIDABLE_MODEL_ARGS = (
    "hidden_dim",
    "vocab_size",
    "dropout",
    "affect_state_dim",
    "num_bins",
    "max_replies",
    "time_series_dim",
    "patch_len",
    "stride",
    "n_heads",
    "n_layers",
    "classification_loss_weight",
    "affect_state_weight",
    "input_view",
    "disable_temporal",
    "disable_structure",
    "disable_affect_state",
    "fusion_variant",
    "capacity_group",
)


def load_model_configs(path: Path) -> dict[str, dict[str, object]]:
    if not str(path) or str(path) == "." or not path.exists() or path.is_dir():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def apply_model_overrides(command: list[str], overrides: dict[str, object]) -> list[str]:
    merged = list(command)
    for key in OVERRIDABLE_MODEL_ARGS:
        if key not in overrides:
            continue
        value = overrides[key]
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                merged.append(flag)
            continue
        merged.extend([flag, str(value)])
    return merged


def main() -> None:
    args = parse_args()
    run_id = args.run_id or default_run_id(args.tag)
    run_root = args.runs_root / run_id
    paths = ensure_experiment_paths(args.experiments_root)
    config = load_hypothesis_config(args.config_path)
    model_configs = load_model_configs(args.model_configs_path)
    log_path = paths.logs / f"{run_id}.log"
    ratio_label = args.train_path.stem.replace("pheme_forecast_ratio_", "ratio_")
    plan = {
        "run_id": run_id,
        "experiment_type": args.experiment_type,
        "target_hypotheses": args.target_hypotheses,
        "question": args.question,
        "success_criteria": args.success_criteria,
        "failure_criteria": args.failure_criteria,
        "idea_sections": args.idea_sections,
        "dataset": str(args.train_path),
        "ratio_labels": [ratio_label],
        "models": list(args.models),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "special_settings": args.special_settings or "无",
        "capacity_group": args.capacity_group,
        "model_configs_path": str(args.model_configs_path) if str(args.model_configs_path) else "",
        "model_configs": model_configs,
    }
    initialize_log(log_path, plan)

    for model in args.models:
        artifact_dir = run_root / model / "artifacts"
        eval_dir = run_root / model / "eval"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"==> train model={model}\n")

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
            "--seed",
            str(args.seed),
            "--capacity_group",
            args.capacity_group,
            "--output_dir",
            str(artifact_dir),
        ]
        train_cmd = apply_model_overrides(train_cmd, model_configs.get(model, {}))
        if str(args.val_path):
            train_cmd.extend(["--val_path", str(args.val_path)])
        train_code = run_command_with_logging(train_cmd, log_path, PROJECT_ROOT)
        if train_code != 0 and not args.continue_on_error:
            raise SystemExit(train_code)
        if train_code != 0:
            continue
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"==> evaluate model={model}\n")

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
        eval_cmd = apply_model_overrides(eval_cmd, model_configs.get(model, {}))
        eval_code = run_command_with_logging(eval_cmd, log_path, PROJECT_ROOT)
        if eval_code != 0 and not args.continue_on_error:
            raise SystemExit(eval_code)

    preliminary_manifest = build_manifest(run_id=run_id, run_root=run_root, config=config, log_path=log_path, experiment_plan=plan)
    finalize_log(log_path, preliminary_manifest["anomalies"])
    manifest = build_manifest(run_id=run_id, run_root=run_root, config=config, log_path=log_path, experiment_plan=plan)
    persist_manifest_outputs(paths, manifest)
    print(f"completed_run={run_id}")
    print(f"saved_manifest={manifest['manifest_path']}")
    print(f"saved_report={manifest['report_path']}")


if __name__ == "__main__":
    main()
