from __future__ import annotations

import torch
from torch import nn

from src.models.text_baseline import HashingTextEncoder


class TemporalBaseline(nn.Module):
    """Reply-sequence encoder with an LSTM over observed replies."""

    def __init__(
        self,
        hidden_dim: int = 128,
        vocab_size: int = 20000,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.reply_encoder = HashingTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=hidden_dim,
            dropout=dropout,
        )
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, reply_sequences: list[list[str]]) -> dict[str, torch.Tensor]:
        device = self.regressor.weight.device
        batch_size = len(reply_sequences)
        lengths = torch.tensor([max(len(sequence), 1) for sequence in reply_sequences], dtype=torch.long)
        max_steps = int(lengths.max().item()) if batch_size > 0 else 1

        padded = torch.zeros(batch_size, max_steps, self.temporal_encoder.input_size, device=device)
        flat_replies: list[str] = []
        positions: list[tuple[int, int]] = []

        for batch_index, sequence in enumerate(reply_sequences):
            effective_sequence = sequence if sequence else [""]
            for step_index, reply_text in enumerate(effective_sequence):
                flat_replies.append(reply_text)
                positions.append((batch_index, step_index))

        if flat_replies:
            encoded_replies = self.reply_encoder(flat_replies)
            for encoded_reply, (batch_index, step_index) in zip(encoded_replies, positions):
                padded[batch_index, step_index] = encoded_reply

        packed = nn.utils.rnn.pack_padded_sequence(
            padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.temporal_encoder(packed)
        shared = self.shared(hidden[-1])
        return {
            "predicted_future_neg_ratio": self.regressor(shared).squeeze(-1),
            "predicted_future_majority_logits": self.classifier(shared),
        }
