from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("text_baseline", "temporal_baseline", "structure_baseline", "affect_state_forecaster")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_cross_event_splits(records: list[dict[str, Any]], heldout_event: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_candidates = [record for record in records if str(record["event_name"]) != heldout_event]
    test_records = [record for record in records if str(record["event_name"]) == heldout_event]
    remaining_events = sorted({str(record["event_name"]) for record in train_candidates})
    if not remaining_events or not test_records:
        raise ValueError(f"Unable to build split for held-out event {heldout_event}")
    val_event = remaining_events[-1]
    train_records = []
    val_records = []
    for record in train_candidates:
        event_name = str(record["event_name"])
        updated = dict(record)
        if event_name == val_event:
            updated["split"] = "val"
            val_records.append(updated)
        else:
            updated["split"] = "train"
            train_records.append(updated)
    test_records = [{**record, "split": "test"} for record in test_records]
    return train_records, val_records, test_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-event validation by holding out each event in turn.")
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/pheme_forecast_ratio_50.jsonl"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    parser.add_argument("--tag_prefix", type=str, default="cross_event")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.data_path)
    event_names = sorted({str(record["event_name"]) for record in records})
    with tempfile.TemporaryDirectory(prefix="pheme_cross_event_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for heldout_event in event_names:
            train_records, val_records, test_records = build_cross_event_splits(records, heldout_event)
            train_path = temp_dir / f"{heldout_event}_train.jsonl"
            val_path = temp_dir / f"{heldout_event}_val.jsonl"
            test_path = temp_dir / f"{heldout_event}_test.jsonl"
            write_jsonl(train_path, train_records)
            write_jsonl(val_path, val_records)
            write_jsonl(test_path, test_records)
            command = [
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
                f"{args.tag_prefix}_{heldout_event}",
                "--question",
                f"验证 held-out event={heldout_event} 时各模型的 cross-event 泛化能力。",
                "--success_criteria",
                "affect_state_forecaster 在 held-out event 上保持与主实验一致的领先趋势。",
                "--failure_criteria",
                "若 held-out event 掉点显著，则后续优先做泛化与正则化。",
                "--special_settings",
                f"heldout_event={heldout_event}",
                "--models",
                *args.models,
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
