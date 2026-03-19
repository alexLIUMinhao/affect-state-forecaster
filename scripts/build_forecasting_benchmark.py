from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.sentiment import aggregate_sentiment, normalize_sentiment_label as normalize_sentiment_alias

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark construction."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a forecasting benchmark from normalized PHEME threads and write "
            "event-level train/val/test splits."
        )
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/processed/pheme_threads_labeled.jsonl"),
        help="Path to weakly labeled thread JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for forecast benchmark outputs.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="pheme_forecast",
        help="Prefix used for forecast benchmark file names.",
    )
    parser.add_argument(
        "--observation_ratios",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        choices=[0.3, 0.5, 0.7],
        help="Observation window ratios to materialize.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Fraction of events assigned to the training split.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of events assigned to the validation split.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of events assigned to the test split.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure CLI logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def parse_timestamp(value: Any) -> float:
    """Convert a timestamp-like field into a sortable UNIX timestamp."""

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

    LOGGER.warning("Unable to parse reply timestamp %r; placing reply at the end", value)
    return float("inf")


def normalize_sentiment_label(reply: dict[str, Any]) -> str:
    """Read a weak sentiment label from a reply record."""

    raw_label = reply.get("sentiment_label") or reply.get("sentiment") or reply.get("label")
    return normalize_sentiment_alias(raw_label)


def sort_replies_by_time(replies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return replies sorted by timestamp while preserving their tree metadata."""

    return sorted(
        replies,
        key=lambda reply: (
            parse_timestamp(reply.get("created_at")),
            str(reply.get("id", "")),
        ),
    )


def split_replies(
    replies: list[dict[str, Any]], observation_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split replies into observation and forecast windows."""

    if not replies:
        return [], []

    split_index = int(math.floor(len(replies) * observation_ratio))
    split_index = max(1, split_index)
    split_index = min(split_index, len(replies) - 1) if len(replies) > 1 else 1

    observed = replies[:split_index]
    forecast = replies[split_index:]
    return observed, forecast


def aggregate_future_sentiment(forecast_replies: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate forecast-window sentiment ratios and majority label."""

    labels = [normalize_sentiment_label(reply) for reply in forecast_replies]
    summary = aggregate_sentiment(labels)
    return {
        "future_neg_ratio": summary["neg_ratio"],
        "future_neu_ratio": summary["neu_ratio"],
        "future_pos_ratio": summary["pos_ratio"],
        "future_majority_sentiment": summary["majority_sentiment"],
    }


def aggregate_observed_sentiment(observed_replies: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [normalize_sentiment_label(reply) for reply in observed_replies]
    summary = aggregate_sentiment(labels)
    return {
        "observed_neg_ratio": summary["neg_ratio"],
        "observed_neu_ratio": summary["neu_ratio"],
        "observed_pos_ratio": summary["pos_ratio"],
        "observed_majority_sentiment": summary["majority_sentiment"],
    }


def normalize_reply(reply: dict[str, Any]) -> dict[str, Any]:
    """Project a reply into the benchmark schema."""

    return {
        "id": str(reply.get("id", "")),
        "parent_id": str(reply["parent_id"]) if reply.get("parent_id") is not None else None,
        "text": str(reply.get("text", "")),
        "created_at": reply.get("created_at"),
        "sentiment_label": normalize_sentiment_label(reply),
    }


def build_sample(record: dict[str, Any], observation_ratio: float) -> dict[str, Any]:
    """Construct one forecasting sample from a normalized thread record."""

    sorted_replies = sort_replies_by_time(list(record.get("replies", [])))
    observed_replies, forecast_replies = split_replies(sorted_replies, observation_ratio)

    observed_replies = [normalize_reply(reply) for reply in observed_replies]
    forecast_replies = [normalize_reply(reply) for reply in forecast_replies]
    future_targets = aggregate_future_sentiment(forecast_replies)

    return {
        "thread_id": str(record["thread_id"]),
        "event_name": str(record["event_name"]),
        "source_text": str(record.get("source_text", "")),
        "conversation_tree": dict(record.get("conversation_tree", {})),
        "observation_ratio": observation_ratio,
        "observed_replies": observed_replies,
        "forecast_replies": forecast_replies,
        **aggregate_observed_sentiment(observed_replies),
        **future_targets,
    }


def allocate_event_splits(
    event_names: list[str], train_ratio: float, val_ratio: float, test_ratio: float
) -> dict[str, str]:
    """Assign events to train, validation, and test splits deterministically."""

    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    events = sorted(set(event_names))
    if not events:
        return {}

    num_events = len(events)
    train_count = int(math.floor(num_events * train_ratio))
    val_count = int(math.floor(num_events * val_ratio))
    test_count = num_events - train_count - val_count

    if num_events >= 3:
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
        test_count = num_events - train_count - val_count
        if test_count <= 0:
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1
            test_count = 1

    split_map: dict[str, str] = {}
    for index, event_name in enumerate(events):
        if index < train_count:
            split_map[event_name] = "train"
        elif index < train_count + val_count:
            split_map[event_name] = "val"
        else:
            split_map[event_name] = "test"
    return split_map


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of records to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_output_paths(output_path: Path) -> dict[str, Path]:
    """Derive split file paths from the base output path."""

    return {
        "all": output_path,
        "train": output_path.with_name(f"{output_path.stem}_train{output_path.suffix}"),
        "val": output_path.with_name(f"{output_path.stem}_val{output_path.suffix}"),
        "test": output_path.with_name(f"{output_path.stem}_test{output_path.suffix}"),
    }


def main() -> None:
    """CLI entry point for building the forecasting benchmark."""

    args = parse_args()
    configure_logging(args.log_level)

    thread_records = load_jsonl(args.input_path)
    LOGGER.info("Loaded %d normalized threads from %s", len(thread_records), args.input_path)
    event_to_split = allocate_event_splits(
        [str(record["event_name"]) for record in thread_records],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    for ratio in args.observation_ratios:
        samples = [build_sample(record, observation_ratio=ratio) for record in thread_records]
        for sample in samples:
            sample["split"] = event_to_split.get(sample["event_name"], "train")

        ratio_suffix = str(int(ratio * 10)).zfill(2)
        output_path = args.output_dir / f"{args.output_prefix}_ratio_{ratio_suffix}.jsonl"
        paths = split_output_paths(output_path)
        write_jsonl(paths["all"], samples)

        for split_name in ("train", "val", "test"):
            split_records = [sample for sample in samples if sample["split"] == split_name]
            write_jsonl(paths[split_name], split_records)
            LOGGER.info(
                "ratio=%.1f wrote %d %s samples across %d events to %s",
                ratio,
                len(split_records),
                split_name,
                len({sample["event_name"] for sample in split_records}),
                paths[split_name],
            )

        LOGGER.info("ratio=%.1f wrote %d total samples to %s", ratio, len(samples), paths["all"])


if __name__ == "__main__":
    main()
