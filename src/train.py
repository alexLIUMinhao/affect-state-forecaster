from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.pheme_forecast_dataset import PHEMEForecastDataset, collate_forecast_batch
from src.eval.metrics import compute_metrics
from src.models.affect_state_forecaster import AffectStateForecaster
from src.models.structure_baseline import StructureBaseline
from src.models.temporal_baseline import TemporalBaseline
from src.models.text_baseline import TextBaseline
from src.utils.sentiment import id_to_label


MODEL_NAMES = ("text_baseline", "temporal_baseline", "structure_baseline", "affect_state_forecaster")


def build_model(name: str, hidden_dim: int, vocab_size: int, dropout: float, affect_state_dim: int) -> nn.Module:
    if name == "text_baseline":
        return TextBaseline(hidden_dim=hidden_dim, vocab_size=vocab_size, dropout=dropout)
    if name == "temporal_baseline":
        return TemporalBaseline(hidden_dim=hidden_dim, vocab_size=vocab_size, dropout=dropout)
    if name == "structure_baseline":
        return StructureBaseline(hidden_dim=hidden_dim, vocab_size=vocab_size, dropout=dropout)
    if name == "affect_state_forecaster":
        return AffectStateForecaster(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            dropout=dropout,
            affect_state_dim=affect_state_dim,
        )
    raise ValueError(f"Unknown model: {name}")


def model_forward(model: nn.Module, model_name: str, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    if model_name == "text_baseline":
        return model(batch["concat_texts"])
    if model_name == "temporal_baseline":
        return model(batch["observed_reply_texts"])
    if model_name == "structure_baseline":
        return model(
            batch["thread_ids"],
            batch["source_texts"],
            batch["observed_replies"],
            batch["conversation_trees"],
        )
    if model_name == "affect_state_forecaster":
        return model(
            batch["thread_ids"],
            batch["source_texts"],
            batch["observed_reply_texts"],
            batch["observed_replies"],
            batch["conversation_trees"],
        )
    raise ValueError(f"Unsupported model: {model_name}")


def unpack_predictions(output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    reg_preds = output["predicted_future_neg_ratio"]
    cls_logits = output.get("predicted_future_majority_logits")
    return reg_preds, cls_logits


def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    mse_criterion: nn.Module,
    cls_criterion: nn.Module,
    device: str,
    classification_loss_weight: float,
    affect_state_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = batch["targets"].to(device)
    reg_preds, cls_logits = unpack_predictions(output)
    reg_loss = mse_criterion(reg_preds, targets)
    loss = reg_loss
    loss_parts = {"regression_loss": float(reg_loss.item())}

    if cls_logits is not None and classification_loss_weight > 0.0:
        cls_loss = cls_criterion(cls_logits, batch["majority_targets"].to(device))
        loss = loss + classification_loss_weight * cls_loss
        loss_parts["classification_loss"] = float(cls_loss.item())

    if affect_state_weight > 0.0 and "predicted_current_affect_state" in output:
        affect_state = output["predicted_current_affect_state"]
        affect_proxy = torch.sigmoid(affect_state.mean(dim=1))
        affect_target = batch["observed_neg_ratios"].to(device)
        affect_loss = mse_criterion(affect_proxy, affect_target)
        loss = loss + affect_state_weight * affect_loss
        loss_parts["affect_state_loss"] = float(affect_loss.item())

    loss_parts["total_loss"] = float(loss.item())
    return loss, loss_parts


def summarize_predictions(
    reg_preds: list[torch.Tensor],
    reg_targets: list[torch.Tensor],
    cls_logits: list[torch.Tensor],
    cls_targets: list[torch.Tensor],
) -> dict[str, float]:
    preds_np = torch.cat(reg_preds).numpy() if reg_preds else torch.empty(0).numpy()
    targets_np = torch.cat(reg_targets).numpy() if reg_targets else torch.empty(0).numpy()
    pred_labels = None
    true_labels = None
    if cls_logits and cls_targets:
        pred_ids = torch.cat(cls_logits).argmax(dim=1).tolist()
        true_ids = torch.cat(cls_targets).tolist()
        pred_labels = [id_to_label(label_id) for label_id in pred_ids]
        true_labels = [id_to_label(label_id) for label_id in true_ids]
    if len(preds_np) == 0:
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}
    return compute_metrics(preds_np, targets_np, pred_labels=pred_labels, true_labels=true_labels)


def evaluate_model(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: str,
    classification_loss_weight: float,
    affect_state_weight: float,
) -> tuple[float, dict[str, float]]:
    model.eval()
    mse_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    reg_preds: list[torch.Tensor] = []
    reg_targets: list[torch.Tensor] = []
    cls_logits: list[torch.Tensor] = []
    cls_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            output = model_forward(model, model_name, batch)
            loss, _ = compute_loss(
                output,
                batch,
                mse_criterion,
                cls_criterion,
                device,
                classification_loss_weight,
                affect_state_weight,
            )
            total_loss += float(loss.item())
            reg_pred, logits = unpack_predictions(output)
            reg_preds.append(reg_pred.cpu())
            reg_targets.append(batch["targets"].cpu())
            if logits is not None:
                cls_logits.append(logits.cpu())
                cls_targets.append(batch["majority_targets"].cpu())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, summarize_predictions(reg_preds, reg_targets, cls_logits, cls_targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future affect forecasting models.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--model", type=str, default="text_baseline", choices=list(MODEL_NAMES))
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--affect_state_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--classification_loss_weight", type=float, default=0.2)
    parser.add_argument("--affect_state_weight", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def artifact_prefix(model_name: str, dataset_path: str) -> str:
    return f"{model_name}_{Path(dataset_path).stem}"


def main() -> None:
    args = parse_args()
    train_dataset = PHEMEForecastDataset(args.train_path)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_forecast_batch,
    )
    val_loader = None
    if args.val_path:
        val_dataset = PHEMEForecastDataset(args.val_path)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_forecast_batch,
            )

    model = build_model(
        args.model,
        args.hidden_dim,
        args.vocab_size,
        args.dropout,
        args.affect_state_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = artifact_prefix(args.model, args.train_path)
    model_path = output_dir / f"{prefix}.pt"
    config_path = output_dir / f"{prefix}.json"
    summary_path = output_dir / f"{prefix}_summary.json"
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            output = model_forward(model, args.model, batch)
            loss, _ = compute_loss(
                output,
                batch,
                mse_criterion,
                cls_criterion,
                args.device,
                args.classification_loss_weight,
                args.affect_state_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        epoch_summary: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
        }

        if val_loader is not None:
            val_loss, val_metrics = evaluate_model(
                model,
                args.model,
                val_loader,
                args.device,
                args.classification_loss_weight,
                args.affect_state_weight,
            )
            epoch_summary["val_loss"] = val_loss
            epoch_summary.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)

        history.append(epoch_summary)
        print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in epoch_summary.items()))

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "hidden_dim": args.hidden_dim,
                "vocab_size": args.vocab_size,
                "dropout": args.dropout,
                "affect_state_dim": args.affect_state_dim,
                "classification_loss_weight": args.classification_loss_weight,
                "affect_state_weight": args.affect_state_weight,
            },
            handle,
            indent=2,
        )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"history": history}, handle, indent=2)

    print(f"saved_model={model_path}")
    print(f"saved_config={config_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
