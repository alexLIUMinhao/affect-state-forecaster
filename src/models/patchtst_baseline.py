from __future__ import annotations

import torch
from torch import nn


class PatchTSTBaseline(nn.Module):
    """Patch-based time-series baseline over observed thread statistics."""

    def __init__(
        self,
        hidden_dim: int = 128,
        time_series_dim: int = 8,
        patch_len: int = 2,
        stride: int = 1,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_series_dim = time_series_dim
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(time_series_dim * patch_len, hidden_dim)
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

    def _patchify(self, series: torch.Tensor) -> torch.Tensor:
        batch_size, num_bins, feature_dim = series.shape
        if num_bins < self.patch_len:
            padding = torch.zeros(
                batch_size,
                self.patch_len - num_bins,
                feature_dim,
                dtype=series.dtype,
                device=series.device,
            )
            series = torch.cat([series, padding], dim=1)
            num_bins = series.size(1)
        patches = series.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(batch_size, -1, self.patch_len * feature_dim)
        return patches

    def forward(self, binned_time_series: torch.Tensor, binned_time_series_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        patches = self._patchify(binned_time_series)
        encoded = self.encoder(self.patch_proj(patches))
        if binned_time_series_mask is not None:
            patch_count = encoded.size(1)
            patch_mask = torch.ones(encoded.size(0), patch_count, dtype=encoded.dtype, device=encoded.device)
            hidden = (encoded * patch_mask.unsqueeze(-1)).sum(dim=1) / patch_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            hidden = encoded.mean(dim=1)
        shared = self.shared(hidden)
        return {
            "predicted_future_neg_ratio": self.regressor(shared).squeeze(-1),
            "predicted_future_majority_logits": self.classifier(shared),
        }
