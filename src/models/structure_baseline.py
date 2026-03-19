from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.text_baseline import HashingTextEncoder
from src.utils.tree import compute_tree_statistics


class StructureBaseline(nn.Module):
    """Lightweight structure-aware regressor using text pooling and tree statistics."""

    def __init__(self, hidden_dim: int = 128, vocab_size: int = 20000, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = HashingTextEncoder(vocab_size=vocab_size, embedding_dim=hidden_dim, dropout=dropout)
        self.tree_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 3)

    def _tree_features(
        self,
        thread_ids: list[str],
        conversation_trees: list[dict[str, str | None]],
        observed_replies: list[list[dict[str, Any]]],
    ) -> torch.Tensor:
        features = [
            compute_tree_statistics(thread_id, conversation_tree, replies)
            for thread_id, conversation_tree, replies in zip(thread_ids, conversation_trees, observed_replies)
        ]
        feature_tensor = torch.from_numpy(np.stack(features, axis=0)).to(self.regressor.weight.device)
        return self.tree_proj(feature_tensor)

    def forward(
        self,
        thread_ids: list[str],
        source_texts: list[str],
        observed_replies: list[list[dict[str, Any]]],
        conversation_trees: list[dict[str, str | None]],
    ) -> dict[str, torch.Tensor]:
        reply_sequences = [
            " ".join(str(reply.get("text", "")).strip() for reply in replies if str(reply.get("text", "")).strip())
            for replies in observed_replies
        ]
        combined_texts = [
            " ".join(part for part in [source_text.strip(), reply_text.strip()] if part)
            for source_text, reply_text in zip(source_texts, reply_sequences)
        ]
        text_features = self.text_encoder(combined_texts)
        tree_features = self._tree_features(thread_ids, conversation_trees, observed_replies)
        hidden = self.shared(torch.cat([text_features, tree_features], dim=1))
        logits = self.classifier(hidden)
        return {
            "predicted_future_neg_ratio": self.regressor(hidden).squeeze(-1),
            "predicted_future_majority_logits": logits,
        }
