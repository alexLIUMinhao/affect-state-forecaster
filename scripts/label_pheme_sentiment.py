from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.sentiment import WEAK_LABELERS, weak_label_text


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weakly label PHEME reply sentiments.")
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/processed/pheme_threads.jsonl"),
        help="Normalized thread JSONL produced by prepare_pheme.py.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/pheme_threads_labeled.jsonl"),
        help="Destination JSONL with reply-level sentiment labels.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--labeler",
        type=str,
        default="lexicon_v1",
        choices=list(WEAK_LABELERS),
        help="Weak sentiment labeler used to annotate replies.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def label_thread(record: dict[str, Any], labeler: str) -> dict[str, Any]:
    labeled_replies = []
    for reply in record.get("replies", []):
        labeled_reply = dict(reply)
        labeled_reply["sentiment_label"] = weak_label_text(str(reply.get("text", "")), labeler=labeler)
        labeled_reply["sentiment_labeler"] = labeler
        labeled_replies.append(labeled_reply)

    labeled_record = dict(record)
    labeled_record["replies"] = labeled_replies
    labeled_record["sentiment_labeler"] = labeler
    return labeled_record


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    threads = load_jsonl(args.input_path)
    labeled_threads = [label_thread(record, args.labeler) for record in threads]
    write_jsonl(args.output_path, labeled_threads)
    LOGGER.info("Labeled %d threads with %s and wrote %s", len(labeled_threads), args.labeler, args.output_path)


if __name__ == "__main__":
    main()
