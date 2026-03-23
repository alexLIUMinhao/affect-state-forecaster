from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.pheme_forecast_dataset import _build_binned_time_series
from src.experiment_reporting import ensure_experiment_paths, rebuild_html_indexes
from src.utils.tree import compute_tree_statistics

MODEL_BIAS_ROWS = [
    {
        "model_name": "text_baseline",
        "input_view": "source_text + observed replies concat_texts",
        "ignores": "显式时间顺序、reply 级结构、局部传播轨迹",
        "prefers": "词汇共现、全局语义、事件关键词",
        "benefits": "当事件主题和负面词强耦合时容易受益；对短线程也不敏感",
        "risks": "容易把事件话题当作情绪代理，且难区分同词不同传播形态",
        "fit_event_imbalance": "中",
        "fit_neutral_heavy": "中",
        "need_long_threads": "低",
        "need_clean_structure": "低",
        "label_noise_sensitivity": "中",
    },
    {
        "model_name": "temporal_baseline",
        "input_view": "reply 文本序列",
        "ignores": "显式树拓扑、全局结构摘要",
        "prefers": "局部顺序、early-to-late 的文本演化",
        "benefits": "当早期回复顺序能反映情绪转折时受益",
        "risks": "线程很短或时间噪声大时，序列模式会不稳定",
        "fit_event_imbalance": "中",
        "fit_neutral_heavy": "中",
        "need_long_threads": "中高",
        "need_clean_structure": "低",
        "label_noise_sensitivity": "中高",
    },
    {
        "model_name": "structure_baseline",
        "input_view": "concat_text + 8维树统计",
        "ignores": "细粒度时间过程、reply 级交互细节",
        "prefers": "树深度、分叉、root children、leaf ratio 等全局拓扑摘要",
        "benefits": "在事件不均衡、neutral-heavy 但传播形态仍有差异时更稳",
        "risks": "对局部情绪转折和单条关键回复不敏感",
        "fit_event_imbalance": "高",
        "fit_neutral_heavy": "高",
        "need_long_threads": "中",
        "need_clean_structure": "中高",
        "label_noise_sensitivity": "中低",
    },
    {
        "model_name": "affect_state_forecaster",
        "input_view": "source + temporal + structure，经 affect-state bottleneck",
        "ignores": "没有显式建模更细粒度的未来生成过程",
        "prefers": "先压缩当前群体状态，再预测未来",
        "benefits": "当当前观察窗口中已存在稳定的群体情绪轮廓时受益",
        "risks": "若弱标注噪声大，affect-state 可能成为 noisy summary",
        "fit_event_imbalance": "高",
        "fit_neutral_heavy": "高",
        "need_long_threads": "中",
        "need_clean_structure": "中高",
        "label_noise_sensitivity": "中高",
    },
    {
        "model_name": "patchtst_baseline",
        "input_view": "固定时间桶统计序列",
        "ignores": "单条 reply 文本细节、非桶内的细粒度结构差异",
        "prefers": "局部重复模式、分段时序形状、桶间平滑变化",
        "benefits": "当未来 negativity 可由早期统计轨迹概括时受益",
        "risks": "对非常短线程、强非平稳噪声和极端稀疏样本较敏感",
        "fit_event_imbalance": "中高",
        "fit_neutral_heavy": "高",
        "need_long_threads": "中",
        "need_clean_structure": "低",
        "label_noise_sensitivity": "中",
    },
    {
        "model_name": "timesnet_baseline",
        "input_view": "桶化后的多变量时序特征",
        "ignores": "reply 级文本与树中的离散交互",
        "prefers": "多尺度局部卷积模式、周期和趋势混合",
        "benefits": "当桶化统计中存在稳定尺度结构时受益",
        "risks": "极短线程和事件级分布漂移会削弱卷积式归纳偏置",
        "fit_event_imbalance": "中",
        "fit_neutral_heavy": "中高",
        "need_long_threads": "中",
        "need_clean_structure": "低",
        "label_noise_sensitivity": "中",
    },
    {
        "model_name": "thread_transformer_baseline",
        "input_view": "source token + reply tokens + 结构 side features",
        "ignores": "没有显式全局树统计先验",
        "prefers": "reply 级局部交互、位置与结构偏置、token 细节",
        "benefits": "当关键回复和局部结构模式决定未来时受益",
        "risks": "样本少、线程短、标签偏斜时更容易过拟合 token 级噪声",
        "fit_event_imbalance": "中低",
        "fit_neutral_heavy": "中",
        "need_long_threads": "高",
        "need_clean_structure": "高",
        "label_noise_sensitivity": "高",
    },
]

DISPLAY_TO_SUFFIX = {"30": ("03", "30"), "50": ("05", "50"), "70": ("07", "70")}
COLOR_SCHEMES = ("event_name", "future_majority_sentiment", "future_neg_bucket")
FEATURE_SPACES = (
    "text_view",
    "temporal_view",
    "structure_view",
    "topconf_time_view",
    "thread_token_view",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a data-first analysis HTML report for the PHEME forecasting benchmark.")
    parser.add_argument("--raw_dir", type=Path, default=PROJECT_ROOT / "data/raw/pheme")
    parser.add_argument("--processed_dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--experiments_root", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--main_ratio", type=str, default="50", choices=("30", "50", "70"))
    parser.add_argument("--max_text_features", type=int, default=512)
    return parser.parse_args()


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def import_umap():
    try:
        import umap.umap_ as umap_module
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised on server
        raise RuntimeError(
            "UMAP dependency is missing. Install it with `python -m pip install umap-learn` in the active environment."
        ) from exc
    return umap_module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_ratio_paths(processed_dir: Path, ratio_display: str) -> dict[str, Path]:
    aliases = DISPLAY_TO_SUFFIX[ratio_display]
    names = []
    for suffix in aliases:
        names.extend(
            [
                processed_dir / f"pheme_forecast_ratio_{suffix}.jsonl",
                processed_dir / f"pheme_forecast_ratio_{suffix}_train.jsonl",
                processed_dir / f"pheme_forecast_ratio_{suffix}_val.jsonl",
                processed_dir / f"pheme_forecast_ratio_{suffix}_test.jsonl",
            ]
        )
    found = {}
    for path in names:
        if path.exists():
            key = "all"
            if path.stem.endswith("_train"):
                key = "train"
            elif path.stem.endswith("_val"):
                key = "val"
            elif path.stem.endswith("_test"):
                key = "test"
            found[key] = path
    required = {"all", "train", "val", "test"}
    missing = required - set(found)
    if missing:
        raise FileNotFoundError(f"Missing ratio {ratio_display} files for splits: {sorted(missing)}")
    return found


def parse_timestamp(value: Any) -> float:
    if value in (None, ""):
        return math.inf
    text = str(value).replace("Z", "+00:00")
    try:
        import datetime as _dt

        return _dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return math.inf


def reply_times(replies: list[dict[str, Any]]) -> list[float]:
    values = [parse_timestamp(reply.get("created_at")) for reply in replies]
    return [value for value in values if value != math.inf]


def observed_tree_stats(sample: dict[str, Any]) -> np.ndarray:
    return compute_tree_statistics(
        str(sample["thread_id"]),
        dict(sample.get("conversation_tree", {})),
        list(sample.get("observed_replies", [])),
    )


def temporal_summary_vector(sample: dict[str, Any]) -> np.ndarray:
    observed = list(sample.get("observed_replies", []))
    times = reply_times(observed)
    time_span = max(times) - min(times) if len(times) >= 2 else 0.0
    deltas = np.diff(times) if len(times) >= 2 else np.asarray([0.0])
    direct_to_source = 0.0
    if observed:
        direct_to_source = sum(1.0 for reply in observed if reply.get("parent_id") == sample["thread_id"]) / len(observed)
    return np.asarray(
        [
            len(observed),
            float(sample.get("observed_neg_ratio", 0.0)),
            float(sample.get("observed_neu_ratio", 0.0)),
            float(sample.get("observed_pos_ratio", 0.0)),
            float(time_span),
            float(np.mean(deltas)),
            float(np.std(deltas)),
            direct_to_source,
        ],
        dtype=np.float32,
    )


def thread_token_summary_vector(sample: dict[str, Any]) -> np.ndarray:
    _series, _mask, depths, parent_positions, time_deltas = _build_binned_time_series(
        str(sample["thread_id"]),
        list(sample.get("observed_replies", [])),
        dict(sample.get("conversation_tree", {})),
    )
    if not depths:
        return np.zeros(8, dtype=np.float32)
    parent_arr = np.asarray(parent_positions, dtype=np.float32)
    time_arr = np.asarray(time_deltas, dtype=np.float32)
    direct_ratio = float(np.mean(parent_arr == 0))
    return np.asarray(
        [
            len(depths),
            float(np.mean(depths)),
            float(np.max(depths)),
            float(np.mean(parent_arr)),
            float(np.max(parent_arr)),
            float(np.mean(time_arr)),
            float(np.std(time_arr)),
            direct_ratio,
        ],
        dtype=np.float32,
    )


def topconf_time_vector(sample: dict[str, Any]) -> np.ndarray:
    series, _mask, _depths, _parents, _deltas = _build_binned_time_series(
        str(sample["thread_id"]),
        list(sample.get("observed_replies", [])),
        dict(sample.get("conversation_tree", {})),
    )
    return series.reshape(-1).numpy().astype(np.float32)


def future_neg_bucket(value: float) -> str:
    if value < 0.1:
        return "[0.0,0.1)"
    if value < 0.3:
        return "[0.1,0.3)"
    if value < 0.6:
        return "[0.3,0.6)"
    return "[0.6,1.0]"


def build_sample_frame(samples: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        observed = list(sample.get("observed_replies", []))
        forecast = list(sample.get("forecast_replies", []))
        stats = observed_tree_stats(sample)
        observed_times = reply_times(observed)
        time_span = max(observed_times) - min(observed_times) if len(observed_times) >= 2 else 0.0
        direct_ratio = 0.0
        if observed:
            direct_ratio = sum(1.0 for reply in observed if reply.get("parent_id") == sample["thread_id"]) / len(observed)
        rows.append(
            {
                "thread_id": sample["thread_id"],
                "event_name": sample["event_name"],
                "split": sample["split"],
                "observation_ratio": sample["observation_ratio"],
                "observed_reply_count": len(observed),
                "forecast_reply_count": len(forecast),
                "observed_neg_ratio": sample.get("observed_neg_ratio", 0.0),
                "observed_neu_ratio": sample.get("observed_neu_ratio", 0.0),
                "observed_pos_ratio": sample.get("observed_pos_ratio", 0.0),
                "future_neg_ratio": sample.get("future_neg_ratio", 0.0),
                "future_neu_ratio": sample.get("future_neu_ratio", 0.0),
                "future_pos_ratio": sample.get("future_pos_ratio", 0.0),
                "observed_majority_sentiment": sample.get("observed_majority_sentiment", "neutral"),
                "future_majority_sentiment": sample.get("future_majority_sentiment", "neutral"),
                "future_neg_bucket": future_neg_bucket(float(sample.get("future_neg_ratio", 0.0))),
                "reply_count": float(stats[0]),
                "node_count": float(stats[1]),
                "edge_count": float(stats[2]),
                "root_children": float(stats[3]),
                "avg_depth": float(stats[4]),
                "max_depth": float(stats[5]),
                "branching_factor": float(stats[6]),
                "leaf_ratio": float(stats[7]),
                "observed_time_span_seconds": float(time_span),
                "direct_to_source_ratio": float(direct_ratio),
            }
        )
    return pd.DataFrame(rows)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def summarize_ratios(processed_dir: Path, record_dir: Path) -> tuple[dict[str, dict[str, Path]], dict[str, pd.DataFrame]]:
    ratio_paths = {ratio: resolve_ratio_paths(processed_dir, ratio) for ratio in ("30", "50", "70")}
    ratio_frames: dict[str, pd.DataFrame] = {}
    overview_rows: list[dict[str, Any]] = []
    for ratio, paths in ratio_paths.items():
        samples = load_jsonl(paths["all"])
        frame = build_sample_frame(samples)
        ratio_frames[ratio] = frame
        write_frame(record_dir / f"thread_summary_ratio_{ratio}.csv", frame)
        overview_rows.append(
            {
                "ratio_display": ratio,
                "threads": len(frame),
                "events": frame["event_name"].nunique(),
                "mean_observed_replies": round(frame["observed_reply_count"].mean(), 4),
                "mean_forecast_replies": round(frame["forecast_reply_count"].mean(), 4),
                "mean_observed_neg_ratio": round(frame["observed_neg_ratio"].mean(), 4),
                "mean_future_neg_ratio": round(frame["future_neg_ratio"].mean(), 4),
            }
        )
    write_frame(record_dir / "ratio_overview.csv", pd.DataFrame(overview_rows))
    return ratio_paths, ratio_frames


def dataset_risk_notes(frame: pd.DataFrame) -> list[str]:
    event_counts = frame["event_name"].value_counts()
    notes = []
    if not event_counts.empty and event_counts.max() / max(event_counts.min(), 1) >= 10:
        notes.append("事件分布明显不均衡，头部事件主导总体统计。")
    if (frame["future_majority_sentiment"] == "neutral").mean() >= 0.8:
        notes.append("future_majority_sentiment 明显偏向 neutral，分类任务容易被多数类主导。")
    if event_counts.min() <= 2:
        notes.append("存在极小事件（如仅 2 个 thread 的事件），cross-event 评测的方差会较大。")
    notes.append("benchmark 依赖弱标注词典，reply-level 标签噪声可能直接限制预测上限。")
    return notes


def event_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("event_name")
        .agg(
            thread_count=("thread_id", "count"),
            mean_observed_replies=("observed_reply_count", "mean"),
            mean_forecast_replies=("forecast_reply_count", "mean"),
            mean_future_neg_ratio=("future_neg_ratio", "mean"),
            mean_avg_depth=("avg_depth", "mean"),
            mean_branching_factor=("branching_factor", "mean"),
        )
        .reset_index()
        .sort_values("thread_count", ascending=False)
    )
    return grouped


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("split")
        .agg(
            thread_count=("thread_id", "count"),
            event_count=("event_name", "nunique"),
            mean_future_neg_ratio=("future_neg_ratio", "mean"),
        )
        .reset_index()
    )
    return grouped


def reply_label_summary(labeled_threads: list[dict[str, Any]]) -> pd.DataFrame:
    counts = Counter()
    for thread in labeled_threads:
        for reply in thread.get("replies", []):
            counts[str(reply.get("sentiment_label", "neutral"))] += 1
    total = max(sum(counts.values()), 1)
    rows = [
        {"sentiment_label": label, "count": counts[label], "ratio": counts[label] / total}
        for label in ("negative", "neutral", "positive")
    ]
    return pd.DataFrame(rows)


def build_feature_spaces(main_samples: list[dict[str, Any]], max_text_features: int) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    texts = []
    temporal_vectors = []
    structure_vectors = []
    topconf_vectors = []
    thread_token_vectors = []
    labels: list[dict[str, Any]] = []
    for sample in main_samples:
        observed = list(sample.get("observed_replies", []))
        reply_text = " ".join(str(reply.get("text", "")).strip() for reply in observed if str(reply.get("text", "")).strip())
        texts.append(" ".join(part for part in [str(sample.get("source_text", "")), reply_text] if part).strip())
        temporal_vectors.append(temporal_summary_vector(sample))
        structure_vectors.append(observed_tree_stats(sample))
        topconf_vectors.append(topconf_time_vector(sample))
        thread_token_vectors.append(thread_token_summary_vector(sample))
        labels.append(
            {
                "event_name": sample["event_name"],
                "future_majority_sentiment": sample.get("future_majority_sentiment", "neutral"),
                "future_neg_bucket": future_neg_bucket(float(sample.get("future_neg_ratio", 0.0))),
            }
        )
    tfidf = TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2), min_df=1)
    text_matrix = tfidf.fit_transform(texts)
    feature_spaces = {
        "text_view": text_matrix,
        "temporal_view": np.vstack(temporal_vectors),
        "structure_view": np.vstack(structure_vectors),
        "topconf_time_view": np.vstack(topconf_vectors),
        "thread_token_view": np.vstack(thread_token_vectors),
    }
    return feature_spaces, pd.DataFrame(labels)


def to_dense_embedding_input(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix, dtype=np.float32)
    if dense.ndim == 1:
        dense = dense[:, None]
    if dense.shape[1] > 50:
        reducer = TruncatedSVD(n_components=min(50, dense.shape[1] - 1), random_state=42) if hasattr(matrix, "toarray") else PCA(n_components=50, random_state=42)
        dense = reducer.fit_transform(dense)
    scaler = StandardScaler()
    return scaler.fit_transform(dense)


def compute_embedding(method: str, matrix: Any) -> np.ndarray:
    dense = to_dense_embedding_input(matrix)
    if method == "tsne":
        perplexity = max(5, min(30, dense.shape[0] // 6))
        model = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
        return model.fit_transform(dense)
    if method == "umap":
        umap_module = import_umap()
        model = umap_module.UMAP(n_components=2, random_state=42, n_neighbors=min(15, max(2, dense.shape[0] - 1)), min_dist=0.1)
        return model.fit_transform(dense)
    raise ValueError(f"Unsupported embedding method: {method}")


def categorical_palette(labels: pd.Series) -> tuple[list[str], dict[str, str]]:
    colors = [
        "#8c3b2f",
        "#2f6b8c",
        "#6f7f2f",
        "#8c6b2f",
        "#6b2f8c",
        "#2f8c72",
        "#bf5b17",
        "#7f7f7f",
        "#e7298a",
    ]
    unique = list(dict.fromkeys(str(label) for label in labels))
    return unique, {label: colors[index % len(colors)] for index, label in enumerate(unique)}


def save_embedding_plot(path: Path, coords: np.ndarray, labels: pd.Series, title: str) -> None:
    plt = import_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    unique, palette = categorical_palette(labels)
    for label in unique:
        mask = labels.astype(str) == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.82, c=palette[label], label=label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_histogram(path: Path, values: pd.Series, title: str, xlabel: str, bins: int = 20) -> None:
    plt = import_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values.astype(float), bins=bins, color="#8c3b2f", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt = import_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#2f6b8c")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter(path: Path, x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    plt = import_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x.astype(float), y.astype(float), s=26, alpha=0.75, c="#6f7f2f")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_standard_figures(main_frame: pd.DataFrame, labeled_summary: pd.DataFrame, ratio_overview: pd.DataFrame, figure_dir: Path) -> dict[str, Path]:
    figure_paths = {}
    figure_paths["event_counts"] = figure_dir / "event_thread_counts.png"
    event_counts = main_frame["event_name"].value_counts()
    save_bar(figure_paths["event_counts"], list(event_counts.index), list(event_counts.values), "Event Thread Counts (Ratio 50)", "Threads")

    figure_paths["reply_label_distribution"] = figure_dir / "reply_label_distribution.png"
    save_bar(
        figure_paths["reply_label_distribution"],
        list(labeled_summary["sentiment_label"]),
        list(labeled_summary["count"]),
        "Reply-level Weak Sentiment Distribution",
        "Replies",
    )

    figure_paths["future_neg_hist"] = figure_dir / "future_neg_ratio_hist.png"
    save_histogram(figure_paths["future_neg_hist"], main_frame["future_neg_ratio"], "Future Negativity Distribution (Ratio 50)", "future_neg_ratio")

    figure_paths["observed_vs_future"] = figure_dir / "observed_vs_future_negativity.png"
    save_scatter(
        figure_paths["observed_vs_future"],
        main_frame["observed_neg_ratio"],
        main_frame["future_neg_ratio"],
        "Observed vs Future Negativity",
        "observed_neg_ratio",
        "future_neg_ratio",
    )

    figure_paths["reply_count_vs_future"] = figure_dir / "reply_count_vs_future_negativity.png"
    save_scatter(
        figure_paths["reply_count_vs_future"],
        main_frame["observed_reply_count"],
        main_frame["future_neg_ratio"],
        "Observed Reply Count vs Future Negativity",
        "observed_reply_count",
        "future_neg_ratio",
    )

    figure_paths["depth_vs_future"] = figure_dir / "avg_depth_vs_future_negativity.png"
    save_scatter(
        figure_paths["depth_vs_future"],
        main_frame["avg_depth"],
        main_frame["future_neg_ratio"],
        "Average Depth vs Future Negativity",
        "avg_depth",
        "future_neg_ratio",
    )

    figure_paths["ratio_overview"] = figure_dir / "ratio_overview_future_negativity.png"
    save_bar(
        figure_paths["ratio_overview"],
        [f"{item}/" + str(100 - int(item)) for item in ratio_overview["ratio_display"]],
        list(ratio_overview["mean_future_neg_ratio"]),
        "Mean Future Negativity Across Observation Ratios",
        "mean future_neg_ratio",
    )
    return figure_paths


def build_embedding_figures(feature_spaces: dict[str, Any], label_frame: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for space_name, matrix in feature_spaces.items():
        for method in ("tsne", "umap"):
            coords = compute_embedding(method, matrix)
            for label_name in COLOR_SCHEMES:
                output_path = figure_dir / f"{space_name}_{method}_{label_name}.png"
                title = f"{space_name} {method.upper()} colored by {label_name}"
                save_embedding_plot(output_path, coords, label_frame[label_name], title)
                rows.append(
                    {
                        "feature_space": space_name,
                        "embedding_method": method.upper(),
                        "color_by": label_name,
                        "path": str(output_path),
                    }
                )
    return rows


def render_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>暂无数据。</p>"
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def path_from(output_html: Path, target: Path) -> str:
    return str(target.relative_to(output_html.parent).as_posix()) if target.is_relative_to(output_html.parent) else str(target)


def figure_gallery(output_html: Path, rows: list[dict[str, str]]) -> str:
    cards = []
    for row in rows:
        rel = path_from(output_html, Path(row["path"]))
        cards.append(
            "<article class='figure-card'>"
            f"<h3>{html.escape(row['feature_space'])} / {html.escape(row['embedding_method'])} / {html.escape(row['color_by'])}</h3>"
            f"<img src='{html.escape(rel)}' alt='{html.escape(row['feature_space'])}' />"
            "</article>"
        )
    return "".join(cards)


def model_bias_frame() -> pd.DataFrame:
    return pd.DataFrame(MODEL_BIAS_ROWS)


def raw_data_sample(raw_dir: Path, normalized_threads: list[dict[str, Any]]) -> tuple[str, str]:
    expected_layout = (
        "data/raw/pheme/<event_name>/<thread_id>/\n"
        "├── source-tweets/*.json\n"
        "├── reactions/*.json\n"
        "└── structure.json"
    )
    if raw_dir.exists():
        event_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
        if event_dirs:
            thread_dirs = sorted(path for path in event_dirs[0].iterdir() if path.is_dir())
            if thread_dirs:
                sample_thread = thread_dirs[0]
                sample_desc = {
                    "event_name": event_dirs[0].name,
                    "thread_dir": sample_thread.name,
                    "children": sorted(path.name for path in sample_thread.iterdir()),
                }
                return expected_layout, json.dumps(sample_desc, ensure_ascii=False, indent=2)
    processed = normalized_threads[0] if normalized_threads else {}
    return expected_layout, json.dumps(processed, ensure_ascii=False, indent=2)[:2400]


def workflow_note() -> str:
    return (
        "raw -> normalized threads -> weak-labeled threads -> forecasting benchmark。"
        "其中 prepare_pheme.py 负责时间规范化和 parent_id 修复，label_pheme_sentiment.py 使用弱标注词典，"
        "build_forecasting_benchmark.py 按 event-level split 构造 30/50/70 三个 observation 设定。"
    )


def render_main_html(
    output_html: Path,
    main_frame: pd.DataFrame,
    ratio_overview: pd.DataFrame,
    event_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    bias_frame: pd.DataFrame,
    risk_notes: list[str],
    standard_figures: dict[str, Path],
    embedding_rows: list[dict[str, str]],
    expected_layout: str,
    sample_json: str,
) -> str:
    rel = lambda path: path_from(output_html, path)
    selected_embedding_rows = [row for row in embedding_rows if row["color_by"] in {"event_name", "future_majority_sentiment"}][:10]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>数据优先分析报告</title>
  <style>
    body {{ margin: 0; background: #f6f1e8; color: #1e293b; font-family: Georgia, "Noto Serif SC", serif; }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 18px 48px; }}
    .hero {{ background: linear-gradient(135deg, #f8e7cf, #f7f5ef); border: 1px solid #d8d1c2; border-radius: 24px; padding: 28px; margin-bottom: 24px; }}
    .hero p, .section p, li {{ line-height: 1.6; }}
    .nav a {{ margin-right: 14px; color: #8c3b2f; text-decoration: none; font-weight: 700; }}
    .section {{ background: #fffdf8; border: 1px solid #d8d1c2; border-radius: 20px; padding: 22px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: #fff8ee; border: 1px solid #e3d7c1; border-radius: 16px; padding: 16px; }}
    h1, h2, h3 {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e3d7c1; text-align: left; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f7efe1; }}
    pre {{ background: #231f20; color: #f8f8f2; border-radius: 14px; padding: 16px; overflow-x: auto; white-space: pre-wrap; }}
    img {{ width: 100%; border-radius: 14px; border: 1px solid #d8d1c2; background: #fff; }}
    .figure-card {{ background: #fff8ee; border: 1px solid #e3d7c1; border-radius: 16px; padding: 14px; }}
    .metric {{ font-size: 28px; font-weight: 700; color: #8c3b2f; }}
    .flow {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .flow .step {{ background: #fff8ee; border: 1px solid #e3d7c1; border-radius: 16px; padding: 16px; text-align: center; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <p><a href="../index.html">返回 HTML 总目录</a> | <a href="./feature_spaces.html">查看全部低维图</a> | <a href="./model_bias.html">查看模型偏置矩阵</a></p>
      <h1>数据优先分析报告</h1>
      <p>这份报告先回答数据问题，再解释模型结果。核心目标不是继续堆模型，而是判断当前 benchmark 的统计结构、标签偏置、结构/时间分布，以及 7 类模型各自的归纳性偏置为什么会在这份数据上吃亏或受益。</p>
      <div class="nav">
        <a href="#raw">原始数据</a>
        <a href="#pipeline">预处理与 Benchmark</a>
        <a href="#stats">数据统计</a>
        <a href="#embeddings">低维可视化</a>
        <a href="#bias">模型偏置</a>
      </div>
    </section>

    <section class="section" id="raw">
      <h2>A. 原始数据形式</h2>
      <p>PHEME 原始 thread 的核心组成是 source tweet、reaction tweets 和 structure.json。即使本地没有完整 raw 目录，当前 repo 的处理链路仍然清楚地定义了期望输入和规范化结果。</p>
      <div class="grid">
        <article class="card">
          <h3>期望原始布局</h3>
          <pre>{html.escape(expected_layout)}</pre>
        </article>
        <article class="card">
          <h3>样例 thread / 规范化记录</h3>
          <pre>{html.escape(sample_json)}</pre>
        </article>
      </div>
    </section>

    <section class="section" id="pipeline">
      <h2>B. 预处理与 Benchmark 构建</h2>
      <p>{html.escape(workflow_note())}</p>
      <div class="flow">
        <div class="step"><h3>Raw PHEME</h3><p>source-tweets / reactions / structure.json</p></div>
        <div class="step"><h3>Normalized Threads</h3><p>时间排序、parent_id 修复、统一 thread schema</p></div>
        <div class="step"><h3>Weak Labels</h3><p>lexicon_v1 / lexicon_conservative 三分类 reply 标签</p></div>
        <div class="step"><h3>Forecast Benchmark</h3><p>30/50/70 observation ratio，event-level train/val/test split</p></div>
      </div>
      <p>报告展示统一采用 <strong>30/50/70</strong> 标签，内部同时兼容 `ratio_03/05/07` 与 `ratio_30/50/70` 两套文件命名。</p>
    </section>

    <section class="section" id="stats">
      <h2>C. 数据本体统计与可视化</h2>
      <div class="grid">
        <article class="card"><div class="metric">{len(main_frame)}</div><p>ratio 50 下的 thread 样本数</p></article>
        <article class="card"><div class="metric">{main_frame['event_name'].nunique()}</div><p>事件数</p></article>
        <article class="card"><div class="metric">{main_frame['observed_reply_count'].mean():.2f}</div><p>平均 observed replies</p></article>
        <article class="card"><div class="metric">{main_frame['future_neg_ratio'].mean():.3f}</div><p>平均 future_neg_ratio</p></article>
      </div>
      <h3>当前数据风险</h3>
      <ul>{''.join(f"<li>{html.escape(note)}</li>" for note in risk_notes)}</ul>
      <div class="grid">
        <article class="figure-card"><h3>事件规模</h3><img src="{html.escape(rel(standard_figures['event_counts']))}" alt="event counts" /></article>
        <article class="figure-card"><h3>reply-level 弱标签分布</h3><img src="{html.escape(rel(standard_figures['reply_label_distribution']))}" alt="reply labels" /></article>
        <article class="figure-card"><h3>future_neg_ratio 分布</h3><img src="{html.escape(rel(standard_figures['future_neg_hist']))}" alt="future neg hist" /></article>
        <article class="figure-card"><h3>observed vs future negativity</h3><img src="{html.escape(rel(standard_figures['observed_vs_future']))}" alt="observed future" /></article>
        <article class="figure-card"><h3>reply count vs future negativity</h3><img src="{html.escape(rel(standard_figures['reply_count_vs_future']))}" alt="reply future" /></article>
        <article class="figure-card"><h3>ratio 对比</h3><img src="{html.escape(rel(standard_figures['ratio_overview']))}" alt="ratio overview" /></article>
      </div>
      <h3>事件摘要</h3>
      {render_table(event_frame)}
      <h3>Split 摘要</h3>
      {render_table(split_frame)}
      <h3>Reply 情绪摘要</h3>
      {render_table(labeled_frame)}
      <h3>Ratio 摘要</h3>
      {render_table(ratio_overview)}
    </section>

    <section class="section" id="embeddings">
      <h2>D. 特征可视化</h2>
      <p>下面的低维图展示的是输入空间和归纳偏置空间，而不是训练后 checkpoint latent。固定使用 `t-SNE` 与 `UMAP`，并分别按事件、未来多数类、future_neg_ratio 分桶着色。</p>
      <div class="grid">
        {figure_gallery(output_html, selected_embedding_rows)}
      </div>
      <p><a href="./feature_spaces.html">打开全部低维可视化图集</a></p>
    </section>

    <section class="section" id="bias">
      <h2>E. 7 类模型的归纳性偏置解释</h2>
      <p>这一部分不是重新汇报分数，而是解释不同模型真正“看到”的数据视角，以及在当前事件不均衡、neutral-heavy、弱标注构造的 benchmark 上，各自更容易受益或受损于什么偏置。</p>
      {render_table(bias_frame)}
      <p><a href="./model_bias.html">打开模型偏置矩阵子页</a></p>
    </section>
  </div>
</body>
</html>
"""


def render_feature_spaces_html(output_html: Path, embedding_rows: list[dict[str, str]]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>低维可视化图集</title>
  <style>
    body {{ margin: 0; background: #f6f1e8; color: #1e293b; font-family: Georgia, "Noto Serif SC", serif; }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 18px 48px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .figure-card {{ background: #fffdf8; border: 1px solid #d8d1c2; border-radius: 16px; padding: 14px; }}
    img {{ width: 100%; border-radius: 14px; border: 1px solid #d8d1c2; }}
  </style>
</head>
<body>
  <div class="wrap">
    <p><a href="./index.html">返回数据分析首页</a></p>
    <h1>低维可视化图集</h1>
    <div class="grid">{figure_gallery(output_html, embedding_rows)}</div>
  </div>
</body>
</html>
"""


def render_bias_html(output_html: Path, bias_frame: pd.DataFrame, risk_notes: list[str]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>模型偏置矩阵</title>
  <style>
    body {{ margin: 0; background: #f6f1e8; color: #1e293b; font-family: Georgia, "Noto Serif SC", serif; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 48px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e3d7c1; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f7efe1; }}
  </style>
</head>
<body>
  <div class="wrap">
    <p><a href="./index.html">返回数据分析首页</a></p>
    <h1>模型偏置矩阵</h1>
    <ul>{''.join(f"<li>{html.escape(note)}</li>" for note in risk_notes)}</ul>
    {render_table(bias_frame)}
  </div>
</body>
</html>
"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    exp_paths = ensure_experiment_paths(args.experiments_root)
    record_dir = exp_paths.records / "data_analysis"
    figure_dir = exp_paths.figures / "data_analysis"
    html_dir = exp_paths.html / "data_analysis"
    for path in (record_dir, figure_dir, html_dir):
        path.mkdir(parents=True, exist_ok=True)

    normalized_threads = load_jsonl(args.processed_dir / "pheme_threads.jsonl")
    labeled_threads = load_jsonl(args.processed_dir / "pheme_threads_labeled.jsonl")
    ratio_paths, ratio_frames = summarize_ratios(args.processed_dir, record_dir)
    main_samples = load_jsonl(ratio_paths[args.main_ratio]["all"])
    main_frame = ratio_frames[args.main_ratio]
    labeled_frame = reply_label_summary(labeled_threads)
    event_frame = event_summary(main_frame)
    split_frame = split_summary(main_frame)
    ratio_overview = pd.read_csv(record_dir / "ratio_overview.csv")
    bias_frame = model_bias_frame()
    risk_notes = dataset_risk_notes(main_frame)

    write_frame(record_dir / "event_summary.csv", event_frame)
    write_frame(record_dir / "split_summary.csv", split_frame)
    write_frame(record_dir / "reply_label_summary.csv", labeled_frame)
    write_frame(record_dir / "model_bias_matrix.csv", bias_frame)

    standard_figures = build_standard_figures(main_frame, labeled_frame, ratio_overview, figure_dir)
    feature_spaces, label_frame = build_feature_spaces(main_samples, args.max_text_features)
    embedding_rows = build_embedding_figures(feature_spaces, label_frame, figure_dir)
    write_frame(record_dir / "embedding_figure_index.csv", pd.DataFrame(embedding_rows))

    expected_layout, sample_json = raw_data_sample(args.raw_dir, normalized_threads)
    index_path = html_dir / "index.html"
    feature_path = html_dir / "feature_spaces.html"
    bias_path = html_dir / "model_bias.html"
    write_text(
        index_path,
        render_main_html(
            index_path,
            main_frame,
            ratio_overview,
            event_frame,
            split_frame,
            labeled_frame,
            bias_frame,
            risk_notes,
            standard_figures,
            embedding_rows,
            expected_layout,
            sample_json,
        ),
    )
    write_text(feature_path, render_feature_spaces_html(feature_path, embedding_rows))
    write_text(bias_path, render_bias_html(bias_path, bias_frame, risk_notes))

    manifest = {
        "main_ratio": args.main_ratio,
        "records_dir": str(record_dir),
        "figures_dir": str(figure_dir),
        "html_dir": str(html_dir),
        "thread_count": len(main_frame),
        "event_count": int(main_frame["event_name"].nunique()),
        "embedding_figures": len(embedding_rows),
    }
    write_text(record_dir / "data_analysis_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    rebuild_html_indexes(exp_paths)
    print(f"output_html={index_path}")
    print(f"embedding_figures={len(embedding_rows)}")


if __name__ == "__main__":
    main()
