from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.utils.sentiment import label_to_id


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


def collate_forecast_batch(batch: list[ForecastSample]) -> dict[str, Any]:
    """Collate benchmark samples for unified forecasting models."""

    observed_sequences = [_reply_texts(sample.observed_replies) for sample in batch]
    observed_lengths = [max(len(sequence), 1) for sequence in observed_sequences]

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
        "targets": torch.tensor([sample.future_neg_ratio for sample in batch], dtype=torch.float32),
        "future_distribution_targets": torch.tensor(
            [
                [sample.future_neg_ratio, sample.future_neu_ratio, sample.future_pos_ratio]
                for sample in batch
            ],
            dtype=torch.float32,
        ),
    }
