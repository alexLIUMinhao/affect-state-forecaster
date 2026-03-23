from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.text_baseline import HashingTextEncoder


class ThreadTransformerBaseline(nn.Module):
    """Pure-torch thread encoder with reply tokens and structural side features."""

    def __init__(
        self,
        hidden_dim: int = 128,
        vocab_size: int = 20000,
        max_replies: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_replies = max_replies
        self.reply_encoder = HashingTextEncoder(vocab_size=vocab_size, embedding_dim=hidden_dim, dropout=dropout)
        self.source_encoder = HashingTextEncoder(vocab_size=vocab_size, embedding_dim=hidden_dim, dropout=dropout)
        self.structure_proj = nn.Linear(4, hidden_dim)
        self.position_embedding = nn.Embedding(max_replies + 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        source_texts: list[str],
        observed_replies: list[list[dict[str, Any]]],
        reply_depths: list[list[int]],
        reply_parent_positions: list[list[int]],
        reply_time_deltas: list[list[float]],
    ) -> dict[str, torch.Tensor]:
        device = self.regressor.weight.device
        batch_size = len(source_texts)
        source_tokens = self.source_encoder(source_texts)
        token_tensor = torch.zeros(batch_size, self.max_replies + 1, source_tokens.size(1), device=device)
        token_mask = torch.ones(batch_size, self.max_replies + 1, dtype=torch.bool, device=device)
        token_tensor[:, 0] = source_tokens
        token_mask[:, 0] = False

        flat_texts: list[str] = []
        flat_positions: list[tuple[int, int]] = []
        structural_rows: list[list[float]] = []
        for batch_index, replies in enumerate(observed_replies):
            limit = min(len(replies), self.max_replies)
            for reply_index in range(limit):
                reply = replies[reply_index]
                flat_texts.append(str(reply.get("text", "")))
                flat_positions.append((batch_index, reply_index + 1))
                depth = float(reply_depths[batch_index][reply_index]) if reply_index < len(reply_depths[batch_index]) else 1.0
                parent_position = float(reply_parent_positions[batch_index][reply_index]) if reply_index < len(reply_parent_positions[batch_index]) else 0.0
                time_delta = float(reply_time_deltas[batch_index][reply_index]) if reply_index < len(reply_time_deltas[batch_index]) else 1.0
                direct_to_source = 1.0 if parent_position == 0.0 else 0.0
                structural_rows.append([depth, parent_position, time_delta, direct_to_source])

        if flat_texts:
            reply_tokens = self.reply_encoder(flat_texts)
            structure_tensor = torch.tensor(structural_rows, dtype=reply_tokens.dtype, device=device)
            reply_tokens = reply_tokens + self.structure_proj(structure_tensor)
            for row_index, (batch_index, token_index) in enumerate(flat_positions):
                token_tensor[batch_index, token_index] = reply_tokens[row_index]
                token_mask[batch_index, token_index] = False

        position_ids = torch.arange(self.max_replies + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        encoded = self.encoder(token_tensor + self.position_embedding(position_ids), src_key_padding_mask=token_mask)
        valid_mask = (~token_mask).float().unsqueeze(-1)
        pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        shared = self.shared(pooled)
        return {
            "predicted_future_neg_ratio": self.regressor(shared).squeeze(-1),
            "predicted_future_majority_logits": self.classifier(shared),
        }
