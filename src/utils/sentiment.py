from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


SENTIMENT_LABELS = ("negative", "neutral", "positive")
SENTIMENT_TO_ID = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
ID_TO_SENTIMENT = {index: label for label, index in SENTIMENT_TO_ID.items()}
TOKEN_PATTERN = re.compile(r"\b\w+\b")

# Lightweight weak-supervision lexicons to bootstrap the benchmark.
POSITIVE_WORDS = {
    "accurate",
    "amazing",
    "awesome",
    "best",
    "calm",
    "clear",
    "confirmed",
    "correct",
    "credible",
    "good",
    "great",
    "helpful",
    "hope",
    "important",
    "legit",
    "love",
    "positive",
    "praying",
    "relief",
    "safe",
    "support",
    "true",
    "useful",
    "valid",
}
NEGATIVE_WORDS = {
    "angry",
    "awful",
    "bad",
    "crisis",
    "danger",
    "dead",
    "death",
    "disaster",
    "disgrace",
    "disturbing",
    "fake",
    "fear",
    "frightening",
    "hate",
    "horrible",
    "kill",
    "killed",
    "lies",
    "lying",
    "murder",
    "negative",
    "panic",
    "racist",
    "rumour",
    "scared",
    "shooting",
    "terror",
    "terrible",
    "threat",
    "tragic",
    "untrue",
    "violence",
    "warning",
    "wrong",
}
NEGATION_WORDS = {"no", "not", "never", "none", "hardly", "without", "n't"}


def normalize_sentiment_label(label: str | None) -> str:
    """Normalize label aliases into the canonical 3-way sentiment space."""

    raw = str(label or "neutral").strip().lower()
    alias_map = {
        "neg": "negative",
        "negative": "negative",
        "-1": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "0": "neutral",
        "pos": "positive",
        "positive": "positive",
        "1": "positive",
    }
    return alias_map.get(raw, "neutral")


def label_to_id(label: str | None) -> int:
    return SENTIMENT_TO_ID[normalize_sentiment_label(label)]


def id_to_label(label_id: int) -> str:
    return ID_TO_SENTIMENT[int(label_id)]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def weak_label_text(text: str) -> str:
    """Assign a weak sentiment label using a small lexicon-based scorer."""

    tokens = tokenize(text)
    if not tokens:
        return "neutral"

    score = 0
    for index, token in enumerate(tokens):
        negated = index > 0 and tokens[index - 1] in NEGATION_WORDS
        if token in POSITIVE_WORDS:
            score += -1 if negated else 1
        elif token in NEGATIVE_WORDS:
            score += 1 if negated else -1

    if "!" in text and score != 0:
        score += 1 if score > 0 else -1

    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def aggregate_sentiment(labels: Iterable[str]) -> dict[str, float | str]:
    """Aggregate 3-way sentiment labels into ratios and majority label."""

    normalized = [normalize_sentiment_label(label) for label in labels]
    counts = Counter(normalized)
    total = len(normalized)
    if total == 0:
        return {
            "neg_ratio": 0.0,
            "neu_ratio": 0.0,
            "pos_ratio": 0.0,
            "majority_sentiment": "neutral",
        }

    majority = sorted(
        SENTIMENT_LABELS,
        key=lambda label: (-counts[label], SENTIMENT_LABELS.index(label)),
    )[0]
    return {
        "neg_ratio": counts["negative"] / total,
        "neu_ratio": counts["neutral"] / total,
        "pos_ratio": counts["positive"] / total,
        "majority_sentiment": majority,
    }
