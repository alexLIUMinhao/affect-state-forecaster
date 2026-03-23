from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.utils.sentiment import SENTIMENT_LABELS, label_to_id, normalize_sentiment_label


@dataclass
class ForecastSample:
    """One forecasting example built from a labeled PHEME thread."""

    thread_id: str
    event_name: str
    split: str
    observation_ratio: float
    source_text: str
    conversation_tree: dict[str, str | None]
    observed_replies: list[dict[str, Any]]
    forecast_replies: list[dict[str, Any]]
    observed_neg_ratio: float
    observed_neu_ratio: float
    observed_pos_ratio: float
    observed_majority_sentiment: str | None
    future_neg_ratio: float
    future_neu_ratio: float
    future_pos_ratio: float
    future_majority_sentiment: str | None


class PHEMEForecastDataset(Dataset):
    """Dataset for thread-level future affect forecasting."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = self._load(self.path)

    def _load(self, path: Path) -> list[ForecastSample]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples: list[ForecastSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                samples.append(
                    ForecastSample(
                        thread_id=str(record["thread_id"]),
                        event_name=str(record["event_name"]),
                        split=str(record.get("split", "train")),
                        observation_ratio=float(record.get("observation_ratio", 0.5)),
                        source_text=str(record.get("source_text", "")),
                        conversation_tree={
                            str(node_id): (str(parent_id) if parent_id is not None else None)
                            for node_id, parent_id in dict(record.get("conversation_tree", {})).items()
                        },
                        observed_replies=list(record.get("observed_replies", [])),
                        forecast_replies=list(record.get("forecast_replies", [])),
                        observed_neg_ratio=float(record.get("observed_neg_ratio", 0.0)),
                        observed_neu_ratio=float(record.get("observed_neu_ratio", 0.0)),
                        observed_pos_ratio=float(record.get("observed_pos_ratio", 0.0)),
                        observed_majority_sentiment=(
                            str(record["observed_majority_sentiment"])
                            if record.get("observed_majority_sentiment") is not None
                            else None
                        ),
                        future_neg_ratio=float(record.get("future_neg_ratio", 0.0)),
                        future_neu_ratio=float(record.get("future_neu_ratio", 0.0)),
                        future_pos_ratio=float(record.get("future_pos_ratio", 0.0)),
                        future_majority_sentiment=(
                            str(record["future_majority_sentiment"])
                            if record.get("future_majority_sentiment") is not None
                            else None
                        ),
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ForecastSample:
        return self.samples[index]


def _reply_texts(replies: list[dict[str, Any]]) -> list[str]:
    return [str(reply.get("text", "")).strip() for reply in replies]


def _concat_text(source_text: str, replies: list[dict[str, Any]]) -> str:
    reply_text = " ".join(text for text in _reply_texts(replies) if text)
    pieces = [source_text.strip(), reply_text.strip()]
    return " ".join(piece for piece in pieces if piece)


def _parse_timestamp(value: Any) -> float:
    if value in (None, ""):
        return float("inf")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    for candidate in (text.replace("Z", "+00:00"), text):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return float("inf")


def _reply_depths(thread_id: str, replies: list[dict[str, Any]], conversation_tree: dict[str, str | None]) -> tuple[list[int], list[int]]:
    reply_ids = [str(reply.get("id", "")) for reply in replies]
    node_positions = {thread_id: 0}
    for index, reply_id in enumerate(reply_ids, start=1):
        node_positions[reply_id] = index

    parent_map: dict[str, str | None] = {thread_id: None}
    for reply in replies:
        reply_id = str(reply.get("id", ""))
        parent_id = reply.get("parent_id")
        if parent_id is None:
            parent_id = conversation_tree.get(reply_id)
        parent_key = str(parent_id) if parent_id is not None and str(parent_id) in node_positions else thread_id
        parent_map[reply_id] = parent_key

    depth_map = {thread_id: 0}
    for reply_id in reply_ids:
        current = reply_id
        depth = 0
        visited: set[str] = set()
        while current in parent_map and parent_map[current] is not None and current not in visited:
            visited.add(current)
            current = str(parent_map[current])
            depth += 1
            if current == thread_id:
                break
        depth_map[reply_id] = depth if depth > 0 else 1

    depths = [depth_map.get(reply_id, 1) for reply_id in reply_ids]
    parent_positions = [node_positions.get(str(parent_map.get(reply_id, thread_id)), 0) for reply_id in reply_ids]
    return depths, parent_positions


def _build_binned_time_series(
    thread_id: str,
    replies: list[dict[str, Any]],
    conversation_tree: dict[str, str | None],
    num_bins: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[float]]:
    feature_dim = 8
    if not replies:
        return (
            torch.zeros(num_bins, feature_dim, dtype=torch.float32),
            torch.zeros(num_bins, dtype=torch.float32),
            [],
            [],
            [],
        )

    timestamps = [_parse_timestamp(reply.get("created_at")) for reply in replies]
    finite_timestamps = [timestamp for timestamp in timestamps if timestamp != float("inf")]
    if finite_timestamps:
        start_time = min(finite_timestamps)
        end_time = max(finite_timestamps)
    else:
        start_time = 0.0
        end_time = float(len(replies) - 1)
        timestamps = [float(index) for index in range(len(replies))]

    span = max(end_time - start_time, 1.0)
    depths, parent_positions = _reply_depths(thread_id, replies, conversation_tree)
    time_deltas = [(timestamp - start_time) / span if timestamp != float("inf") else 1.0 for timestamp in timestamps]
    features = torch.zeros(num_bins, feature_dim, dtype=torch.float32)
    mask = torch.zeros(num_bins, dtype=torch.float32)
    bucket_counts = torch.zeros(num_bins, dtype=torch.float32)
    unique_parents: list[set[int]] = [set() for _ in range(num_bins)]

    for index, reply in enumerate(replies):
        position = min(int(time_deltas[index] * num_bins), num_bins - 1)
        label = normalize_sentiment_label(reply.get("sentiment_label"))
        features[position, 0] += 1.0
        features[position, 1 + SENTIMENT_LABELS.index(label)] += 1.0
        features[position, 4] += float(depths[index])
        features[position, 5] = max(features[position, 5], float(depths[index]))
        features[position, 7] += 1.0 if parent_positions[index] == 0 else 0.0
        bucket_counts[position] += 1.0
        unique_parents[position].add(parent_positions[index])

    for bucket in range(num_bins):
        count = float(bucket_counts[bucket].item())
        if count <= 0.0:
            continue
        mask[bucket] = 1.0
        features[bucket, 1:4] = features[bucket, 1:4] / count
        features[bucket, 4] = features[bucket, 4] / count
        features[bucket, 6] = float(len(unique_parents[bucket]))
        features[bucket, 7] = features[bucket, 7] / count

    return features, mask, depths, parent_positions, time_deltas


def collate_forecast_batch(batch: list[ForecastSample]) -> dict[str, Any]:
    """Collate benchmark samples for unified forecasting models."""

    observed_sequences = [_reply_texts(sample.observed_replies) for sample in batch]
    observed_lengths = [max(len(sequence), 1) for sequence in observed_sequences]
    binned_series: list[torch.Tensor] = []
    binned_masks: list[torch.Tensor] = []
    reply_depths: list[list[int]] = []
    reply_parent_positions: list[list[int]] = []
    reply_time_deltas: list[list[float]] = []

    for sample in batch:
        series, mask, depths, parent_positions, time_deltas = _build_binned_time_series(
            sample.thread_id,
            sample.observed_replies,
            sample.conversation_tree,
        )
        binned_series.append(series)
        binned_masks.append(mask)
        reply_depths.append(depths)
        reply_parent_positions.append(parent_positions)
        reply_time_deltas.append(time_deltas)

    return {
        "thread_ids": [sample.thread_id for sample in batch],
        "event_names": [sample.event_name for sample in batch],
        "splits": [sample.split for sample in batch],
        "observation_ratios": [sample.observation_ratio for sample in batch],
        "source_texts": [sample.source_text for sample in batch],
        "conversation_trees": [sample.conversation_tree for sample in batch],
        "observed_replies": [sample.observed_replies for sample in batch],
        "forecast_replies": [sample.forecast_replies for sample in batch],
        "observed_neg_ratios": torch.tensor([sample.observed_neg_ratio for sample in batch], dtype=torch.float32),
        "observed_neu_ratios": torch.tensor([sample.observed_neu_ratio for sample in batch], dtype=torch.float32),
        "observed_pos_ratios": torch.tensor([sample.observed_pos_ratio for sample in batch], dtype=torch.float32),
        "observed_majority_sentiments": [sample.observed_majority_sentiment for sample in batch],
        "future_majority_sentiments": [sample.future_majority_sentiment for sample in batch],
        "majority_targets": torch.tensor(
            [label_to_id(sample.future_majority_sentiment) for sample in batch],
            dtype=torch.long,
        ),
        "concat_texts": [_concat_text(sample.source_text, sample.observed_replies) for sample in batch],
        "observed_reply_texts": observed_sequences,
        "observed_lengths": torch.tensor(observed_lengths, dtype=torch.long),
        "binned_time_series": torch.stack(binned_series, dim=0),
        "binned_time_series_mask": torch.stack(binned_masks, dim=0),
        "reply_depths": reply_depths,
        "reply_parent_positions": reply_parent_positions,
        "reply_time_deltas": reply_time_deltas,
        "targets": torch.tensor([sample.future_neg_ratio for sample in batch], dtype=torch.float32),
        "future_distribution_targets": torch.tensor(
            [
                [sample.future_neg_ratio, sample.future_neu_ratio, sample.future_pos_ratio]
                for sample in batch
            ],
            dtype=torch.float32,
        ),
    }
