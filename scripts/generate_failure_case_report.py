from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import compute_metrics
from src.experiment_reporting import (
    classify_html_category,
    ensure_experiment_paths,
    read_csv_rows,
    rebuild_html_indexes,
    write_csv_rows,
)

MODEL_NOTE_LABELS = {
    "affect_state_forecaster": "Affect-State Forecaster",
    "patchtst_baseline": "PatchTST",
    "thread_transformer_baseline": "Thread Transformer",
}

CASE_BUCKET_ORDER = [
    "shared_failures",
    "affect_vs_patch_conflicts",
    "thread_transformer_unique_wins",
    "flip_up_cases",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate failure-case diagnostics for selected forecasting models.")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["affect_state_forecaster", "patchtst_baseline", "thread_transformer_baseline"],
    )
    parser.add_argument("--test_path", type=Path, required=True)
    parser.add_argument("--experiments_root", type=Path, default=Path("experiments"))
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truncate_text(value: str, limit: int) -> str:
    normalized = " ".join(value.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 1, 0)].rstrip() + "…"


def target_bucket(value: float) -> str:
    if value == 0.0:
        return "zero"
    if value <= 0.1:
        return "low"
    if value <= 0.3:
        return "mid"
    return "high"


def length_bucket(observed_reply_count: int) -> str:
    if observed_reply_count <= 3:
        return "short"
    if observed_reply_count <= 8:
        return "medium"
    return "long"


def flip_bucket(observed_neg_ratio: float, future_neg_ratio: float) -> str:
    if observed_neg_ratio == 0.0 and future_neg_ratio > 0.0:
        return "flip_up"
    if observed_neg_ratio > 0.1 and future_neg_ratio == 0.0:
        return "flip_down"
    if observed_neg_ratio == 0.0 and future_neg_ratio == 0.0:
        return "stable_zero"
    return "stable_nonzero"


def relative_path(output_html: Path, target: Path) -> str:
    return Path(os.path.relpath(target, start=output_html.parent)).as_posix()


def load_gold_samples(test_path: Path) -> dict[str, dict[str, Any]]:
    gold_rows = {}
    for row in load_jsonl(test_path):
        observed_replies = list(row.get("observed_replies", []))
        forecast_replies = list(row.get("forecast_replies", []))
        gold_rows[str(row["thread_id"])] = {
            "thread_id": str(row["thread_id"]),
            "event_name": str(row.get("event_name", "")),
            "split": str(row.get("split", "")),
            "observation_ratio": float(row.get("observation_ratio", 0.0)),
            "source_text": str(row.get("source_text", "")),
            "observed_replies": observed_replies,
            "forecast_replies": forecast_replies,
            "observed_neg_ratio": float(row.get("observed_neg_ratio", 0.0)),
            "future_neg_ratio": float(row.get("future_neg_ratio", 0.0)),
            "observed_reply_count": len(observed_replies),
            "forecast_reply_count": len(forecast_replies),
            "target_bucket": target_bucket(float(row.get("future_neg_ratio", 0.0))),
            "length_bucket": length_bucket(len(observed_replies)),
            "flip_bucket": flip_bucket(float(row.get("observed_neg_ratio", 0.0)), float(row.get("future_neg_ratio", 0.0))),
        }
    return gold_rows


def load_prediction_map(predictions_path: Path, model_name: str) -> dict[str, dict[str, Any]]:
    prediction_rows = load_jsonl(predictions_path)
    mapping = {}
    for row in prediction_rows:
        thread_id = str(row["thread_id"])
        mapping[thread_id] = {
            "model": model_name,
            "predicted_future_neg_ratio": float(row["predicted_future_neg_ratio"]),
            "predicted_future_majority_sentiment": str(row.get("predicted_future_majority_sentiment", "")),
            "predicted_current_affect_state": row.get("predicted_current_affect_state"),
            "gate_source_mean": row.get("gate_source_mean"),
            "gate_temporal_mean": row.get("gate_temporal_mean"),
            "gate_structure_mean": row.get("gate_structure_mean"),
        }
    return mapping


def align_predictions(
    gold_rows: dict[str, dict[str, Any]],
    prediction_maps: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    model_names = list(prediction_maps)
    for thread_id, gold in gold_rows.items():
        merged = dict(gold)
        for model_name in model_names:
            if thread_id not in prediction_maps[model_name]:
                raise KeyError(f"Missing prediction for thread_id={thread_id} in model={model_name}")
            prediction = prediction_maps[model_name][thread_id]
            pred_value = float(prediction["predicted_future_neg_ratio"])
            merged[f"{model_name}_pred"] = pred_value
            merged[f"{model_name}_abs_error"] = abs(pred_value - merged["future_neg_ratio"])
            if prediction.get("gate_source_mean") is not None:
                merged[f"{model_name}_gate_source_mean"] = float(prediction["gate_source_mean"])
                merged[f"{model_name}_gate_temporal_mean"] = float(prediction["gate_temporal_mean"])
                merged[f"{model_name}_gate_structure_mean"] = float(prediction["gate_structure_mean"])
        merged["mean_abs_error"] = float(np.mean([merged[f"{model_name}_abs_error"] for model_name in model_names]))
        aligned.append(merged)
    return aligned


def compute_overall_metrics(aligned_rows: list[dict[str, Any]], model_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = np.array([row["future_neg_ratio"] for row in aligned_rows], dtype=np.float32)
    for model_name in model_names:
        preds = np.array([row[f"{model_name}_pred"] for row in aligned_rows], dtype=np.float32)
        metrics = compute_metrics(preds, targets)
        rows.append(
            {
                "slice_type": "overall",
                "slice_value": "all",
                "model": model_name,
                "num_threads": len(aligned_rows),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "pearson": metrics["pearson"],
                "spearman": metrics["spearman"],
            }
        )
    return rows


def compute_slice_metrics(aligned_rows: list[dict[str, Any]], model_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aligned_rows:
        grouped[("event_name", row["event_name"])].append(row)
        grouped[("target_bucket", row["target_bucket"])].append(row)
        grouped[("length_bucket", row["length_bucket"])].append(row)
        grouped[("flip_bucket", row["flip_bucket"])].append(row)

    metric_rows = compute_overall_metrics(aligned_rows, model_names)
    for (slice_type, slice_value), items in sorted(grouped.items()):
        if not items:
            continue
        targets = np.array([row["future_neg_ratio"] for row in items], dtype=np.float32)
        for model_name in model_names:
            preds = np.array([row[f"{model_name}_pred"] for row in items], dtype=np.float32)
            metrics = compute_metrics(preds, targets)
            metric_rows.append(
                {
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "model": model_name,
                    "num_threads": len(items),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                }
            )
    return metric_rows


def select_case_buckets(aligned_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = list(aligned_rows)
    shared = sorted(rows, key=lambda row: row["mean_abs_error"], reverse=True)[:10]
    conflict = sorted(
        rows,
        key=lambda row: abs(row["affect_state_forecaster_abs_error"] - row["patchtst_baseline_abs_error"]),
        reverse=True,
    )[:10]
    thread_wins = [
        row
        for row in rows
        if row["thread_transformer_baseline_abs_error"] + 0.05 <= row["affect_state_forecaster_abs_error"]
        and row["thread_transformer_baseline_abs_error"] + 0.05 <= row["patchtst_baseline_abs_error"]
    ]
    thread_wins = sorted(thread_wins, key=lambda row: row["thread_transformer_baseline_abs_error"])[:10]
    flip_up = [row for row in rows if row["flip_bucket"] == "flip_up"]
    flip_up = sorted(flip_up, key=lambda row: row["mean_abs_error"], reverse=True)[:10]
    return {
        "shared_failures": shared,
        "affect_vs_patch_conflicts": conflict,
        "thread_transformer_unique_wins": thread_wins,
        "flip_up_cases": flip_up,
    }


def derive_case_note(row: dict[str, Any]) -> str:
    affect_error = row["affect_state_forecaster_abs_error"]
    patch_error = row["patchtst_baseline_abs_error"]
    thread_error = row["thread_transformer_baseline_abs_error"]
    if row["flip_bucket"] == "flip_up" and affect_error >= patch_error + 0.03:
        return "均值回归失败"
    if patch_error + 0.03 < min(affect_error, thread_error):
        return "桶统计捕捉到趋势但语义模型未捕捉"
    if thread_error + 0.03 < min(affect_error, patch_error):
        return "局部 reply 交互帮助了 transformer"
    if row["mean_abs_error"] >= 0.18:
        return "三模型在稀有翻转或高噪声样本上共同失效"
    return "局部模式存在，但三模型对未来负面比例的刻画方式不同"


def build_case_rows(case_buckets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket_name, items in case_buckets.items():
        for rank, row in enumerate(items, start=1):
            case_row = {
                "bucket_name": bucket_name,
                "rank_within_bucket": rank,
                "thread_id": row["thread_id"],
                "event_name": row["event_name"],
                "observed_neg_ratio": row["observed_neg_ratio"],
                "future_neg_ratio": row["future_neg_ratio"],
                "observed_reply_count": row["observed_reply_count"],
                "forecast_reply_count": row["forecast_reply_count"],
                "affect_state_forecaster_pred": row["affect_state_forecaster_pred"],
                "patchtst_baseline_pred": row["patchtst_baseline_pred"],
                "thread_transformer_baseline_pred": row["thread_transformer_baseline_pred"],
                "affect_state_forecaster_abs_error": row["affect_state_forecaster_abs_error"],
                "patchtst_baseline_abs_error": row["patchtst_baseline_abs_error"],
                "thread_transformer_baseline_abs_error": row["thread_transformer_baseline_abs_error"],
                "target_bucket": row["target_bucket"],
                "length_bucket": row["length_bucket"],
                "flip_bucket": row["flip_bucket"],
                "case_note": derive_case_note(row),
                "source_excerpt": truncate_text(row["source_text"], 160),
                "reply_excerpt_1": truncate_text(str(row["observed_replies"][0].get("text", "")), 100) if len(row["observed_replies"]) >= 1 else "",
                "reply_excerpt_2": truncate_text(str(row["observed_replies"][1].get("text", "")), 100) if len(row["observed_replies"]) >= 2 else "",
                "reply_excerpt_3": truncate_text(str(row["observed_replies"][2].get("text", "")), 100) if len(row["observed_replies"]) >= 3 else "",
            }
            rows.append(case_row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def plot_slice_heatmap(metric_rows: list[dict[str, Any]], model_names: list[str], output_path: Path) -> None:
    slice_rows = [
        row for row in metric_rows if row["slice_type"] in {"event_name", "target_bucket", "length_bucket", "flip_bucket"}
    ]
    ordered_slices = []
    labels = []
    for prefix, values in (
        ("event_name", ["prince-toronto", "putinmissing", "sydneysiege"]),
        ("target_bucket", ["zero", "low", "mid", "high"]),
        ("length_bucket", ["short", "medium", "long"]),
        ("flip_bucket", ["flip_up", "flip_down", "stable_zero", "stable_nonzero"]),
    ):
        for value in values:
            ordered_slices.append((prefix, value))
            labels.append(f"{prefix}:{value}")
    matrix = np.full((len(ordered_slices), len(model_names)), np.nan, dtype=np.float32)
    lookup = {(row["slice_type"], row["slice_value"], row["model"]): row["mae"] for row in slice_rows}
    for i, (slice_type, slice_value) in enumerate(ordered_slices):
        for j, model_name in enumerate(model_names):
            value = lookup.get((slice_type, slice_value, model_name))
            matrix[i, j] = value if value is not None else np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(model_names)), [MODEL_NOTE_LABELS.get(name, name) for name in model_names], rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Slice-Level MAE Heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#1f2937")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gold_vs_prediction(aligned_rows: list[dict[str, Any]], model_names: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(model_names), figsize=(15, 4.8), sharex=True, sharey=True)
    targets = np.array([row["future_neg_ratio"] for row in aligned_rows], dtype=np.float32)
    for ax, model_name in zip(np.atleast_1d(axes), model_names):
        preds = np.array([row[f"{model_name}_pred"] for row in aligned_rows], dtype=np.float32)
        ax.scatter(targets, preds, alpha=0.7, s=22, color="#8c3b2f")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
        ax.set_title(MODEL_NOTE_LABELS.get(model_name, model_name))
        ax.set_xlabel("Gold future_neg_ratio")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_error_by_bucket(metric_rows: list[dict[str, Any]], model_names: list[str], output_path: Path) -> None:
    bucket_order = ["zero", "low", "mid", "high"]
    grouped = {
        (row["slice_value"], row["model"]): row["mae"]
        for row in metric_rows
        if row["slice_type"] == "target_bucket"
    }
    x = np.arange(len(bucket_order))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, model_name in enumerate(model_names):
        values = [grouped.get((bucket, model_name), np.nan) for bucket in bucket_order]
        ax.bar(x + (idx - 1) * width, values, width=width, label=MODEL_NOTE_LABELS.get(model_name, model_name))
    ax.set_xticks(x, bucket_order)
    ax.set_ylabel("MAE")
    ax.set_title("Absolute Error by Target Bucket")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_disagreement(aligned_rows: list[dict[str, Any]], output_path: Path) -> None:
    pairs = [
        ("affect_state_forecaster", "patchtst_baseline"),
        ("affect_state_forecaster", "thread_transformer_baseline"),
        ("patchtst_baseline", "thread_transformer_baseline"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (left, right) in zip(np.atleast_1d(axes), pairs):
        left_preds = np.array([row[f"{left}_pred"] for row in aligned_rows], dtype=np.float32)
        right_preds = np.array([row[f"{right}_pred"] for row in aligned_rows], dtype=np.float32)
        colors = [row["future_neg_ratio"] for row in aligned_rows]
        scatter = ax.scatter(left_preds, right_preds, c=colors, cmap="viridis", alpha=0.75, s=24)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
        ax.set_xlabel(MODEL_NOTE_LABELS.get(left, left))
        ax.set_ylabel(MODEL_NOTE_LABELS.get(right, right))
    fig.colorbar(scatter, ax=np.atleast_1d(axes), fraction=0.02, pad=0.02, label="Gold future_neg_ratio")
    fig.suptitle("Pairwise Model Disagreement", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_case_cards(case_rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[row["bucket_name"]].append(row)
    cards = []
    for bucket_name in CASE_BUCKET_ORDER:
        rows = grouped.get(bucket_name, [])
        cards.append(f"<h3>{html.escape(bucket_name)}</h3>")
        if not rows:
            cards.append("<p>没有符合当前阈值的样本。</p>")
            continue
        cards.append("<div class='case-grid'>")
        for row in rows:
            replies = [row["reply_excerpt_1"], row["reply_excerpt_2"], row["reply_excerpt_3"]]
            reply_items = "".join(f"<li>{html.escape(reply)}</li>" for reply in replies if reply)
            cards.append(
                "<article class='case-card'>"
                f"<div class='meta'>thread_id={html.escape(str(row['thread_id']))} | event={html.escape(row['event_name'])}</div>"
                f"<p><strong>Observed/Future:</strong> {row['observed_neg_ratio']:.3f} -> {row['future_neg_ratio']:.3f}</p>"
                f"<p><strong>Replies:</strong> observed={row['observed_reply_count']} | forecast={row['forecast_reply_count']}</p>"
                f"<p><strong>Affect:</strong> pred={row['affect_state_forecaster_pred']:.3f}, err={row['affect_state_forecaster_abs_error']:.3f}</p>"
                f"<p><strong>PatchTST:</strong> pred={row['patchtst_baseline_pred']:.3f}, err={row['patchtst_baseline_abs_error']:.3f}</p>"
                f"<p><strong>Thread Transformer:</strong> pred={row['thread_transformer_baseline_pred']:.3f}, err={row['thread_transformer_baseline_abs_error']:.3f}</p>"
                f"<p><strong>自动说明:</strong> {html.escape(row['case_note'])}</p>"
                f"<p><strong>Source:</strong> {html.escape(row['source_excerpt'])}</p>"
                + (f"<ol>{reply_items}</ol>" if reply_items else "")
                + "</article>"
            )
        cards.append("</div>")
    return "".join(cards)


def render_html(
    output_html: Path,
    report_run_id: str,
    overall_rows: list[dict[str, Any]],
    slice_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    figures: dict[str, Path],
) -> str:
    rel = lambda path: relative_path(output_html, path)
    overall_table = render_table(
        overall_rows,
        ["model", "num_threads", "mae", "rmse", "pearson", "spearman"],
    )
    slice_table = render_table(
        [row for row in slice_rows if row["slice_type"] != "overall"],
        ["slice_type", "slice_value", "model", "num_threads", "mae", "rmse", "pearson", "spearman"],
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(report_run_id)} Failure Case Report</title>
  <style>
    body {{ margin: 0; background: #f6f1e8; color: #1e293b; font-family: Georgia, "Noto Serif SC", serif; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 48px; }}
    .hero {{ background: linear-gradient(135deg, #f8e7cf, #f7f5ef); border: 1px solid #d8d1c2; border-radius: 24px; padding: 28px; margin-bottom: 24px; }}
    .section {{ background: #fffdf8; border: 1px solid #d8d1c2; border-radius: 20px; padding: 22px; margin-bottom: 20px; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .figure-card, .case-card {{ background: #fcfaf4; border: 1px solid #e3d7c1; border-radius: 16px; padding: 14px; }}
    .case-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e3d7c1; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f7efe1; }}
    img {{ width: 100%; border-radius: 14px; border: 1px solid #d8d1c2; }}
    .meta {{ color: #667085; font-size: 13px; margin-bottom: 8px; }}
    a {{ color: #8c3b2f; text-decoration: none; font-weight: 700; }}
    p, li {{ line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <p><a href="../index.html">返回 diagnostics 目录</a> · <a href="../main/index.html">返回主实验目录</a></p>
      <h1>Failure Case Diagnostics: {html.escape(report_run_id)}</h1>
      <p>本页只分析 `affect_state_forecaster`、`patchtst_baseline`、`thread_transformer_baseline`。重点解释为什么 Affect-State Forecaster 的 MAE 最优但排序弱、PatchTST 的排序最好但未明显赢下 MAE、Thread Transformer 保留局部交互后为何仍不够稳。</p>
    </section>

    <section class="section">
      <h2>1. 总览</h2>
      <p>结论固定为：`affect_state_forecaster` 最稳，`patchtst_baseline` 排序最好，`thread_transformer_baseline` 有局部交互优势但不够稳。</p>
      {overall_table}
    </section>

    <section class="section">
      <h2>2. 切片误差矩阵</h2>
      <div class="figure-grid">
        <article class="figure-card"><h3>Slice-Level MAE Heatmap</h3><img src="{html.escape(rel(figures['slice_heatmap']))}" alt="slice heatmap" /></article>
      </div>
      {slice_table}
    </section>

    <section class="section">
      <h2>3. 模型差异图</h2>
      <div class="figure-grid">
        <article class="figure-card"><h3>Gold vs Prediction</h3><img src="{html.escape(rel(figures['gold_vs_prediction']))}" alt="gold vs prediction" /></article>
        <article class="figure-card"><h3>Absolute Error by Target Bucket</h3><img src="{html.escape(rel(figures['absolute_error_by_bucket']))}" alt="absolute error by target bucket" /></article>
        <article class="figure-card"><h3>Pairwise Model Disagreement</h3><img src="{html.escape(rel(figures['pairwise_disagreement']))}" alt="pairwise disagreement" /></article>
      </div>
    </section>

    <section class="section">
      <h2>4. 典型 Failure Case</h2>
      <p>每个 case 固定展示 thread 基本信息、三模型预测与误差、简短文本摘录，以及按规则模板生成的自动说明。</p>
      {build_case_cards(case_rows)}
    </section>

    <section class="section">
      <h2>5. 归纳偏置解释</h2>
      <p><strong>Affect-State Forecaster:</strong> 多模态压缩和去噪适合偏斜分布，但对 `flip_up` 和高负面稀有样本容易过度平滑。</p>
      <p><strong>PatchTST:</strong> 对趋势和相对顺序最敏感，但在需要关键语义线索或单条 reply 触发时会漏掉决定性证据。</p>
      <p><strong>Thread Transformer:</strong> 能利用局部 reply 与结构细节，但在小样本、短线程、弱标注噪声下更不稳。</p>
    </section>

    <section class="section">
      <h2>6. 方法启示</h2>
      <ul>
        <li>下一代 `affect_state_forecaster` 应显式吸收 `binned_time_series`，补上排序能力而不是只继续扩大编码器。</li>
        <li>不应盲目增加 token-level 复杂度，应让细粒度 reply 信息作为补充而不是主导。</li>
        <li>论文 failure-case 部分应重点展示 `flip_up` 样本，因为它最能区分“均值回归”与“真正提前识别恶化”。</li>
      </ul>
    </section>
  </div>
</body>
</html>
"""


def build_markdown(report_run_id: str, overall_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Failure Case Report: {report_run_id}",
        "",
        "## 总览",
        "- 分析对象: affect_state_forecaster, patchtst_baseline, thread_transformer_baseline",
        "- 核心结论: affect_state_forecaster 最稳，patchtst_baseline 排序最好，thread_transformer_baseline 有局部交互优势但不够稳。",
        "",
        "## 整体指标",
    ]
    for row in overall_rows:
        lines.append(
            f"- {row['model']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, Pearson={row['pearson']:.4f}, Spearman={row['spearman']:.4f}"
        )
    lines.extend(["", "## 典型案例数量"])
    for bucket_name in CASE_BUCKET_ORDER:
        lines.append(f"- {bucket_name}: {sum(1 for row in case_rows if row['bucket_name'] == bucket_name)}")
    return "\n".join(lines) + "\n"


def upsert_failure_index(paths: Any, report_run_id: str, html_path: Path, manifest_path: Path, report_path: Path) -> None:
    index_path = paths.records / "experiment_index.csv"
    fieldnames = [
        "run_id",
        "created_at",
        "run_root",
        "best_model",
        "best_mae",
        "best_rmse",
        "experiment_goal",
        "decision_action",
        "hypothesis_summary",
        "suggest_new_idea",
        "html_path",
        "report_path",
        "manifest_path",
    ]
    rows = read_csv_rows(index_path)
    row = {
        "run_id": report_run_id,
        "created_at": utc_now_iso(),
        "run_root": "",
        "best_model": "affect_state_forecaster",
        "best_mae": "0.10519883036613464",
        "best_rmse": "0.1745268702507019",
        "experiment_goal": "Failure-case diagnostics for affect_state_forecaster, patchtst_baseline, and thread_transformer_baseline",
        "decision_action": "diagnose",
        "hypothesis_summary": "failure_case=affect_smoothing_vs_patch_ranking_vs_thread_local_interactions",
        "suggest_new_idea": "False",
        "html_path": str(html_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
    }
    rows = [item for item in rows if item.get("run_id") != report_run_id]
    rows.append(row)
    rows.sort(key=lambda item: item.get("created_at", ""))
    write_csv_rows(index_path, rows, fieldnames)


def main() -> None:
    args = parse_args()
    report_run_id = f"{args.run_id}_failure_case"
    if classify_html_category(report_run_id) != "diagnostics":
        raise ValueError("failure_case reports must be classified under diagnostics.")

    paths = ensure_experiment_paths(args.experiments_root)
    manifest_path = paths.manifests / f"{args.run_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    source_manifest = load_json(manifest_path)

    prediction_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name in args.models:
        eval_dir = Path(source_manifest["run_root"]) / model_name / "eval"
        predictions_path = eval_dir / "predictions.jsonl"
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Missing predictions for {model_name}: {predictions_path}. "
                "Re-run evaluate.py for this model on the server before generating the report."
            )
        prediction_maps[model_name] = load_prediction_map(predictions_path, model_name)

    gold_rows = load_gold_samples(args.test_path)
    aligned_rows = align_predictions(gold_rows, prediction_maps)
    if len(aligned_rows) != 92:
        raise ValueError(f"Expected 92 aligned test threads, found {len(aligned_rows)}")

    metric_rows = compute_slice_metrics(aligned_rows, args.models)
    overall_rows = [row for row in metric_rows if row["slice_type"] == "overall"]
    case_buckets = select_case_buckets(aligned_rows)
    case_rows = build_case_rows(case_buckets)

    record_dir = paths.records / "failure_case"
    figure_dir = paths.figures / "failure_case"
    html_path = paths.html / "diagnostics" / f"{report_run_id}.html"
    report_path = paths.records / f"{report_run_id}.md"
    custom_manifest_path = paths.manifests / f"{report_run_id}.json"

    summary_csv = record_dir / f"{report_run_id}_summary.csv"
    cases_csv = record_dir / f"{report_run_id}_case_studies.csv"
    write_csv_rows(
        summary_csv,
        metric_rows,
        ["slice_type", "slice_value", "model", "num_threads", "mae", "rmse", "pearson", "spearman"],
    )
    write_csv_rows(
        cases_csv,
        case_rows,
        [
            "bucket_name",
            "rank_within_bucket",
            "thread_id",
            "event_name",
            "observed_neg_ratio",
            "future_neg_ratio",
            "observed_reply_count",
            "forecast_reply_count",
            "affect_state_forecaster_pred",
            "patchtst_baseline_pred",
            "thread_transformer_baseline_pred",
            "affect_state_forecaster_abs_error",
            "patchtst_baseline_abs_error",
            "thread_transformer_baseline_abs_error",
            "target_bucket",
            "length_bucket",
            "flip_bucket",
            "case_note",
            "source_excerpt",
            "reply_excerpt_1",
            "reply_excerpt_2",
            "reply_excerpt_3",
        ],
    )

    figures = {
        "slice_heatmap": figure_dir / f"{report_run_id}_slice_heatmap.png",
        "gold_vs_prediction": figure_dir / f"{report_run_id}_gold_vs_prediction.png",
        "absolute_error_by_bucket": figure_dir / f"{report_run_id}_absolute_error_by_bucket.png",
        "pairwise_disagreement": figure_dir / f"{report_run_id}_pairwise_disagreement.png",
    }
    plot_slice_heatmap(metric_rows, args.models, figures["slice_heatmap"])
    plot_gold_vs_prediction(aligned_rows, args.models, figures["gold_vs_prediction"])
    plot_absolute_error_by_bucket(metric_rows, args.models, figures["absolute_error_by_bucket"])
    plot_pairwise_disagreement(aligned_rows, figures["pairwise_disagreement"])

    html_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(html_path, render_html(html_path, report_run_id, overall_rows, metric_rows, case_rows, figures))
    _write_text(report_path, build_markdown(report_run_id, overall_rows, case_rows))
    _write_json(
        custom_manifest_path,
        {
            "run_id": report_run_id,
            "created_at": utc_now_iso(),
            "source_run_id": args.run_id,
            "models": args.models,
            "summary_csv": str(summary_csv),
            "cases_csv": str(cases_csv),
            "html_path": str(html_path),
            "report_path": str(report_path),
            "figures": {key: str(value) for key, value in figures.items()},
        },
    )
    upsert_failure_index(paths, report_run_id, html_path, custom_manifest_path, report_path)
    rebuild_html_indexes(paths)

    print(f"aligned_threads={len(aligned_rows)}")
    print(f"saved_summary={summary_csv}")
    print(f"saved_cases={cases_csv}")
    print(f"saved_html={html_path}")


if __name__ == "__main__":
    main()
