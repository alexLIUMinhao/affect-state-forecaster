from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass
class ThreadCheck:
    """Validation status for one PHEME thread directory."""

    event_name: str
    thread_dir: Path
    warnings: list[str] = field(default_factory=list)
    source_tweet_count: int = 0
    reaction_count: int = 0
    has_structure: bool = False

    @property
    def is_valid(self) -> bool:
        return self.source_tweet_count >= 1


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for layout verification."""

    parser = argparse.ArgumentParser(description="Verify the raw PHEME dataset layout.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw/pheme"),
        help="Root directory containing raw PHEME event folders.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when thread-level warnings are detected.",
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
    """Configure logger for CLI use."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON content, returning None instead of raising on malformed files."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to parse JSON file %s: %s", path, exc)
        return None


def find_thread_dirs(input_dir: Path) -> list[tuple[str, Path]]:
    """Return candidate thread directories under each event folder."""

    thread_dirs: list[tuple[str, Path]] = []
    if not input_dir.exists():
        return thread_dirs

    for event_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for thread_dir in sorted(path for path in event_dir.iterdir() if path.is_dir()):
            thread_dirs.append((event_dir.name, thread_dir))
    return thread_dirs


def check_thread(event_name: str, thread_dir: Path) -> ThreadCheck:
    """Verify the expected files and folders within one thread directory."""

    result = ThreadCheck(event_name=event_name, thread_dir=thread_dir)
    source_dir = thread_dir / "source-tweets"
    reactions_dir = thread_dir / "reactions"
    structure_path = thread_dir / "structure.json"

    if not source_dir.exists():
        result.warnings.append("missing source-tweets directory")
    else:
        source_files = sorted(source_dir.glob("*.json"))
        result.source_tweet_count = len(source_files)
        if result.source_tweet_count == 0:
            result.warnings.append("no source tweet JSON files")
        elif result.source_tweet_count > 1:
            result.warnings.append("multiple source tweet JSON files")
        for source_path in source_files[:1]:
            if safe_load_json(source_path) is None:
                result.warnings.append(f"malformed source tweet JSON: {source_path.name}")

    if not reactions_dir.exists():
        result.warnings.append("missing reactions directory")
    else:
        reaction_files = sorted(reactions_dir.glob("*.json"))
        result.reaction_count = len(reaction_files)
        if result.reaction_count == 0:
            result.warnings.append("no reaction JSON files")
        for reaction_path in reaction_files[:10]:
            if safe_load_json(reaction_path) is None:
                result.warnings.append(f"malformed reaction JSON: {reaction_path.name}")
                break

    result.has_structure = structure_path.exists()
    if not result.has_structure:
        result.warnings.append("missing structure.json")
    elif safe_load_json(structure_path) is None:
        result.warnings.append("malformed structure.json")

    return result


def summarize_checks(checks: list[ThreadCheck]) -> tuple[int, int]:
    """Log a summary and return counts for valid threads and warnings."""

    valid_count = sum(check.is_valid for check in checks)
    warning_count = sum(len(check.warnings) for check in checks)

    LOGGER.info("Checked %d thread directories", len(checks))
    LOGGER.info("Valid threads with source tweets: %d", valid_count)
    LOGGER.info("Total warnings: %d", warning_count)

    for check in checks:
        if check.warnings:
            warning_text = "; ".join(check.warnings)
            LOGGER.warning("[%s/%s] %s", check.event_name, check.thread_dir.name, warning_text)

    return valid_count, warning_count


def main() -> None:
    """CLI entry point for verifying the PHEME raw-data layout."""

    args = parse_args()
    configure_logging(args.log_level)

    if not args.input_dir.exists():
        LOGGER.error("Input directory does not exist: %s", args.input_dir)
        raise SystemExit(1)

    thread_dirs = find_thread_dirs(args.input_dir)
    if not thread_dirs:
        LOGGER.error("No candidate thread directories found under %s", args.input_dir)
        LOGGER.error("Expected layout: data/raw/pheme/<event>/<thread>/source-tweets/*.json")
        raise SystemExit(1)

    checks = [check_thread(event_name, thread_dir) for event_name, thread_dir in thread_dirs]
    valid_count, warning_count = summarize_checks(checks)

    if valid_count == 0:
        LOGGER.error("No usable threads were found. Fix the raw dataset layout before preprocessing.")
        raise SystemExit(1)

    if args.strict and warning_count > 0:
        LOGGER.error("Layout verification completed with warnings under --strict.")
        raise SystemExit(1)

    LOGGER.info("Layout verification completed successfully.")


if __name__ == "__main__":
    main()
