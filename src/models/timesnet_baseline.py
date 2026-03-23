from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _TemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
                for kernel_size in (1, 3, 5)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv_inputs = inputs.transpose(1, 2)
        mixed = sum(layer(conv_inputs) for layer in self.layers) / len(self.layers)
        mixed = mixed.transpose(1, 2)
        return self.norm(inputs + self.dropout(F.gelu(mixed)))


class TimesNetBaseline(nn.Module):
    """Lightweight TimesNet-style baseline over bucketed thread features."""

    def __init__(
        self,
        hidden_dim: int = 128,
        time_series_dim: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(time_series_dim, hidden_dim)
        self.blocks = nn.ModuleList([_TemporalBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, binned_time_series: torch.Tensor, binned_time_series_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.input_proj(binned_time_series)
        for block in self.blocks:
            hidden = block(hidden)
        if binned_time_series_mask is not None:
            pooled = (hidden * binned_time_series_mask.unsqueeze(-1)).sum(dim=1) / binned_time_series_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        shared = self.shared(pooled)
        return {
            "predicted_future_neg_ratio": self.regressor(shared).squeeze(-1),
            "predicted_future_majority_logits": self.classifier(shared),
        }
