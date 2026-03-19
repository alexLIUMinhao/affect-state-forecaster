from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.pheme_forecast_dataset import PHEMEForecastDataset, collate_forecast_batch
from src.eval.metrics import compute_metrics
from src.train import MODEL_NAMES, build_model, model_forward
from src.utils.sentiment import id_to_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forecasting models and export summaries.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="text_baseline", choices=list(MODEL_NAMES))
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--affect_state_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--input_view", type=str, default="full", choices=["full", "source_only", "replies_only"])
    parser.add_argument("--disable_temporal", action="store_true")
    parser.add_argument("--disable_structure", action="store_true")
    parser.add_argument("--disable_affect_state", action="store_true")
    return parser.parse_args()


def maybe_load_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config_path:
        return args

    with open(args.config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    args.model = config.get("model", args.model)
    args.hidden_dim = int(config.get("hidden_dim", args.hidden_dim))
    args.vocab_size = int(config.get("vocab_size", args.vocab_size))
    args.dropout = float(config.get("dropout", args.dropout))
    args.affect_state_dim = int(config.get("affect_state_dim", args.affect_state_dim))
    args.input_view = str(config.get("input_view", args.input_view))
    args.disable_temporal = bool(config.get("disable_temporal", args.disable_temporal))
    args.disable_structure = bool(config.get("disable_structure", args.disable_structure))
    args.disable_affect_state = bool(config.get("disable_affect_state", args.disable_affect_state))
    return args


def normalize_model_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return (
        output["predicted_future_neg_ratio"],
        output.get("predicted_future_majority_logits"),
        output.get("predicted_current_affect_state"),
    )


def default_majority_prediction(pred_value: float) -> str:
    if pred_value >= 0.5:
        return "negative"
    if pred_value <= 0.15:
        return "positive"
    return "neutral"


def logits_to_labels(logits: torch.Tensor | None, preds: np.ndarray) -> list[str]:
    if logits is None:
        return [default_majority_prediction(float(pred)) for pred in preds]
    label_ids = logits.argmax(dim=1).cpu().tolist()
    return [id_to_label(label_id) for label_id in label_ids]


def build_prediction_rows(
    model_name: str,
    batch: dict[str, Any],
    preds: np.ndarray,
    pred_labels: list[str],
    affect_states: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, pred in enumerate(preds):
        row = {
            "model": model_name,
            "thread_id": batch["thread_ids"][index],
            "event_name": batch["event_names"][index],
            "split": batch["splits"][index],
            "observation_ratio": batch["observation_ratios"][index],
            "target_future_neg_ratio": float(batch["targets"][index].item()),
            "predicted_future_neg_ratio": float(pred),
            "target_future_majority_sentiment": batch["future_majority_sentiments"][index],
            "predicted_future_majority_sentiment": pred_labels[index],
        }
        if affect_states is not None:
            row["predicted_current_affect_state"] = affect_states[index].tolist()
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], group_name: str, group_value: str) -> dict[str, Any]:
    preds = np.array([row["predicted_future_neg_ratio"] for row in rows], dtype=np.float32)
    targets = np.array([row["target_future_neg_ratio"] for row in rows], dtype=np.float32)
    pred_labels = [str(row["predicted_future_majority_sentiment"]) for row in rows]
    true_labels = [row["target_future_majority_sentiment"] for row in rows]
    summary: dict[str, Any] = {
        "group_name": group_name,
        "group_value": group_value,
        "num_threads": len(rows),
    }
    summary.update(compute_metrics(preds, targets, pred_labels=pred_labels, true_labels=true_labels))
    return summary


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_name",
        "group_value",
        "num_threads",
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "majority_sentiment_accuracy",
        "majority_sentiment_macro_f1",
        "majority_sentiment_weighted_f1",
        "loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def write_event_error_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["event_name"]), []).append(row)

    event_rows = []
    for event_name, event_samples in sorted(grouped.items()):
        preds = np.array([row["predicted_future_neg_ratio"] for row in event_samples], dtype=np.float32)
        targets = np.array([row["target_future_neg_ratio"] for row in event_samples], dtype=np.float32)
        metrics = compute_metrics(preds, targets)
        event_rows.append(
            {
                "event_name": event_name,
                "num_threads": len(event_samples),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "pearson": metrics["pearson"],
                "spearman": metrics["spearman"],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_name", "num_threads", "mae", "rmse", "pearson", "spearman"])
        writer.writeheader()
        for row in event_rows:
            writer.writerow(row)


def main() -> None:
    args = maybe_load_config(parse_args())
    dataset = PHEMEForecastDataset(args.data_path)
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_forecast_batch)
    model = build_model(
        args.model,
        args.hidden_dim,
        args.vocab_size,
        args.dropout,
        args.affect_state_dim,
        disable_temporal=args.disable_temporal,
        disable_structure=args.disable_structure,
        disable_affect_state=args.disable_affect_state,
    ).to(args.device)
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    criterion = nn.MSELoss()

    all_rows: list[dict[str, Any]] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(args.device)
            raw_output = model_forward(model, args.model, batch, input_view=args.input_view)
            preds_tensor, cls_logits, affect_state = normalize_model_output(raw_output)
            total_loss += float(criterion(preds_tensor, targets).item())
            preds = preds_tensor.cpu().numpy()
            pred_labels = logits_to_labels(cls_logits, preds)
            affect_state_np = affect_state.cpu().numpy() if affect_state is not None else None
            all_rows.extend(build_prediction_rows(args.model, batch, preds, pred_labels, affect_state_np))

    summaries = [summarize_rows(all_rows, "overall", "all")]
    split_values = sorted({row["split"] for row in all_rows})
    ratio_values = sorted({row["observation_ratio"] for row in all_rows})
    event_values = sorted({row["event_name"] for row in all_rows})

    for split_value in split_values:
        rows = [row for row in all_rows if row["split"] == split_value]
        summaries.append(summarize_rows(rows, "split", split_value))
    for ratio_value in ratio_values:
        rows = [row for row in all_rows if row["observation_ratio"] == ratio_value]
        summaries.append(summarize_rows(rows, "observation_ratio", str(ratio_value)))
    for event_value in event_values:
        rows = [row for row in all_rows if row["event_name"] == event_value]
        summaries.append(summarize_rows(rows, "event", event_value))

    summaries[0]["loss"] = total_loss / max(len(loader), 1)

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "results_summary.csv"
    event_error_path = output_dir / "error_analysis_by_event.csv"
    write_predictions(predictions_path, all_rows)
    write_summary_csv(summary_path, summaries)
    write_event_error_csv(event_error_path, all_rows)

    for summary in summaries[: min(len(summaries), 12)]:
        parts = [
            f"group={summary['group_name']}",
            f"value={summary['group_value']}",
            f"num_threads={summary['num_threads']}",
            f"mae={summary['mae']:.4f}",
            f"rmse={summary['rmse']:.4f}",
        ]
        print(" ".join(parts))

    print(f"saved_predictions={predictions_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_event_errors={event_error_path}")


if __name__ == "__main__":
    main()
