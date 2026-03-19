from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
TWITTER_TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


@dataclass(frozen=True)
class ReplyRecord:
    """Normalized reply entry for a processed PHEME thread."""

    id: str
    parent_id: str | None
    text: str
    created_at: str | None
    timestamp: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PHEME preprocessing script."""

    parser = argparse.ArgumentParser(
        description=(
            "Parse raw PHEME event threads and export normalized JSONL records to "
            "data/processed/pheme_threads.jsonl."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw/pheme"),
        help="Root directory containing raw PHEME events.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/pheme_threads.jsonl"),
        help="Destination JSONL file for normalized threads.",
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
    """Initialize standard CLI logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to load JSON from {path}: {exc}") from exc


def extract_text(tweet: dict[str, Any]) -> str:
    """Return the most useful text field from a tweet payload."""

    extended = tweet.get("extended_tweet")
    if isinstance(extended, dict) and extended.get("full_text"):
        return str(extended["full_text"]).strip()
    if tweet.get("full_text"):
        return str(tweet["full_text"]).strip()
    if tweet.get("text"):
        return str(tweet["text"]).strip()
    return ""


def normalize_created_at(value: Any) -> tuple[str | None, float]:
    """Normalize a tweet timestamp into ISO 8601 and a sortable UNIX timestamp."""

    if value in (None, ""):
        return None, float("inf")

    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        return dt.isoformat(), float(value)

    text = str(value).strip()
    try:
        dt = datetime.strptime(text, TWITTER_TIME_FORMAT)
        return dt.astimezone(timezone.utc).isoformat(), dt.timestamp()
    except ValueError:
        pass

    for candidate in (text.replace("Z", "+00:00"), text):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat(), dt.timestamp()
        except ValueError:
            continue

    LOGGER.warning("Could not parse timestamp: %s", text)
    return text, float("inf")


def extract_tweet_id(tweet: dict[str, Any], fallback: str) -> str:
    """Extract a stable tweet identifier from a payload or file stem."""

    tweet_id = tweet.get("id_str") or tweet.get("id") or fallback
    return str(tweet_id)


def extract_parent_id(tweet: dict[str, Any]) -> str | None:
    """Extract the reply parent identifier if present."""

    parent_id = tweet.get("in_reply_to_status_id_str") or tweet.get("in_reply_to_status_id")
    return str(parent_id) if parent_id not in (None, "") else None


def flatten_tree(tree: Any, parent_id: str | None = None) -> dict[str, str | None]:
    """Traverse a PHEME conversation tree and map each node to its parent."""

    relationships: dict[str, str | None] = {}
    if not isinstance(tree, dict):
        return relationships

    for node_id, children in tree.items():
        node_key = str(node_id)
        if node_key not in relationships:
            relationships[node_key] = parent_id
        relationships.update(flatten_tree(children, parent_id=node_key))
    return relationships


def load_source_tweet(thread_dir: Path) -> tuple[str, dict[str, Any]]:
    """Load the single source tweet for a thread."""

    source_dir = thread_dir / "source-tweets"
    source_files = sorted(source_dir.glob("*.json"))
    if not source_files:
        raise FileNotFoundError(f"No source tweet found in {source_dir}")
    if len(source_files) > 1:
        LOGGER.warning("Multiple source tweets found in %s; using %s", source_dir, source_files[0].name)
    source_path = source_files[0]
    return source_path.stem, load_json(source_path)


def load_replies(thread_dir: Path, relationships: dict[str, str | None], source_id: str) -> list[ReplyRecord]:
    """Load and normalize reply tweets while preserving tree structure."""

    replies_dir = thread_dir / "reactions"
    records: list[ReplyRecord] = []
    if not replies_dir.exists():
        LOGGER.warning("Missing reactions directory in %s", thread_dir)
        return records

    for reply_path in sorted(replies_dir.glob("*.json")):
        payload = load_json(reply_path)
        reply_id = extract_tweet_id(payload, fallback=reply_path.stem)
        created_at, timestamp = normalize_created_at(payload.get("created_at"))
        parent_id = extract_parent_id(payload)
        if parent_id is None:
            parent_id = relationships.get(reply_id)
        if parent_id is None and reply_id in relationships:
            parent_id = relationships[reply_id]
        if parent_id == reply_id:
            parent_id = None
        if parent_id is None and reply_id != source_id:
            parent_id = source_id if reply_id in relationships else None

        records.append(
            ReplyRecord(
                id=reply_id,
                parent_id=parent_id,
                text=extract_text(payload),
                created_at=created_at,
                timestamp=timestamp,
            )
        )

    records.sort(key=lambda item: (item.timestamp, item.id))
    return records


def normalize_thread(thread_dir: Path, event_name: str) -> dict[str, Any]:
    """Convert a raw PHEME thread directory into a normalized JSON-ready record."""

    source_id, source_tweet = load_source_tweet(thread_dir)
    structure_path = thread_dir / "structure.json"
    relationships = flatten_tree(load_json(structure_path)) if structure_path.exists() else {}

    if source_id not in relationships:
        relationships[source_id] = None

    replies = load_replies(thread_dir, relationships=relationships, source_id=source_id)
    source_created_at, _ = normalize_created_at(source_tweet.get("created_at"))

    return {
        "thread_id": source_id,
        "event_name": event_name,
        "source_text": extract_text(source_tweet),
        "source_created_at": source_created_at,
        "conversation_tree": relationships,
        "replies": [
            {
                "id": reply.id,
                "parent_id": reply.parent_id,
                "text": reply.text,
                "created_at": reply.created_at,
            }
            for reply in replies
        ],
    }


def iter_thread_dirs(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all PHEME thread directories under each event directory."""

    threads: list[tuple[str, Path]] = []
    for event_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for thread_dir in sorted(path for path in event_dir.iterdir() if path.is_dir()):
            source_dir = thread_dir / "source-tweets"
            if source_dir.exists():
                threads.append((event_dir.name, thread_dir))
    return threads


def log_input_summary(thread_dirs: list[tuple[str, Path]], input_dir: Path) -> None:
    """Log a compact summary of discovered events and threads."""

    event_names = sorted({event_name for event_name, _ in thread_dirs})
    LOGGER.info("Found %d events under %s", len(event_names), input_dir)
    LOGGER.info("Found %d candidate threads for preprocessing", len(thread_dirs))


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write normalized thread records as JSON Lines."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI entry point for processing the raw PHEME dataset."""

    args = parse_args()
    configure_logging(args.log_level)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    thread_dirs = iter_thread_dirs(args.input_dir)
    if not thread_dirs:
        LOGGER.warning("No PHEME thread directories found under %s", args.input_dir)
        LOGGER.warning("Run python scripts/verify_pheme_layout.py to inspect the raw-data setup.")
        write_jsonl([], args.output_path)
        LOGGER.info("Wrote empty output to %s", args.output_path)
        return

    log_input_summary(thread_dirs, args.input_dir)

    records: list[dict[str, Any]] = []
    for event_name, thread_dir in thread_dirs:
        try:
            record = normalize_thread(thread_dir=thread_dir, event_name=event_name)
            records.append(record)
        except Exception as exc:
            LOGGER.warning("Skipping thread %s due to preprocessing error: %s", thread_dir, exc)

    write_jsonl(records, args.output_path)
    LOGGER.info("Wrote %d normalized threads to %s", len(records), args.output_path)


if __name__ == "__main__":
    main()
