from __future__ import annotations

import hashlib
import re

import torch
from torch import nn


class HashingTextEncoder(nn.Module):
    """Lightweight text encoder based on stable hashed token embeddings."""

    def __init__(self, vocab_size: int = 20000, embedding_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.dropout = nn.Dropout(dropout)
        self.token_pattern = re.compile(r"\b\w+\b")

    def _token_ids(self, text: str) -> list[int]:
        tokens = self.token_pattern.findall(text.lower())
        if not tokens:
            return [0]
        token_ids = []
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            token_ids.append(int.from_bytes(digest, "big") % self.vocab_size)
        return token_ids

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.embedding.weight.device
        all_ids: list[int] = []
        offsets = [0]
        for text in texts:
            token_ids = self._token_ids(text)
            all_ids.extend(token_ids)
            offsets.append(offsets[-1] + len(token_ids))

        token_tensor = torch.tensor(all_ids, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        embeddings = self.embedding(token_tensor, offset_tensor)
        return self.dropout(embeddings)


class TextBaseline(nn.Module):
    """Source-plus-observed-replies text encoder with regression and classification heads."""

    def __init__(self, hidden_dim: int = 128, vocab_size: int = 20000, dropout: float = 0.1):
        super().__init__()
        self.encoder = HashingTextEncoder(vocab_size=vocab_size, embedding_dim=hidden_dim, dropout=dropout)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(texts)
        hidden = self.shared(encoded)
        return {
            "predicted_future_neg_ratio": self.regressor(hidden).squeeze(-1),
            "predicted_future_majority_logits": self.classifier(hidden),
        }
