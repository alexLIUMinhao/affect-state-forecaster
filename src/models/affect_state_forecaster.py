from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.text_baseline import HashingTextEncoder
from src.utils.tree import compute_tree_statistics


class AffectStateForecaster(nn.Module):
    """Two-stage forecaster with an explicit latent group affect-state bottleneck.

    This differs from a direct end-to-end predictor because it does not map observed
    replies straight to the future negative ratio in one step. Instead, it first
    estimates the current group affect state implied by the observed window, then
    uses that latent state to forecast future negativity. The bottleneck makes the
    intermediate representation inspectable and encourages the model to separate
    "what the group feels now" from "how that state evolves next".
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        affect_state_dim: int = 32,
        vocab_size: int = 20000,
        num_layers: int = 1,
        dropout: float = 0.1,
        disable_temporal: bool = False,
        disable_structure: bool = False,
        disable_affect_state: bool = False,
        fusion_variant: str = "full",
    ):
        super().__init__()
        self.disable_temporal = disable_temporal
        self.disable_structure = disable_structure
        self.disable_affect_state = disable_affect_state
        self.fusion_variant = fusion_variant
        self.reply_encoder = HashingTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=hidden_dim,
            dropout=dropout,
        )
        self.source_encoder = HashingTextEncoder(
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
        self.tree_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.source_scalar_gate = nn.Linear(hidden_dim, 1)
        self.temporal_scalar_gate = nn.Linear(hidden_dim, 1)
        self.structure_scalar_gate = nn.Linear(hidden_dim, 1)
        self.source_vector_gate = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_vector_gate = nn.Linear(hidden_dim, hidden_dim)
        self.structure_vector_gate = nn.Linear(hidden_dim, hidden_dim)
        self.affect_state_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, affect_state_dim),
        )
        self.direct_regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.direct_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.future_regressor = nn.Sequential(
            nn.LayerNorm(affect_state_dim),
            nn.Linear(affect_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.future_classifier = nn.Sequential(
            nn.LayerNorm(affect_state_dim),
            nn.Linear(affect_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def _apply_fusion_gates(
        self,
        source_encoding: torch.Tensor,
        temporal_encoding: torch.Tensor,
        tree_encoding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = source_encoding.size(0)
        device = source_encoding.device
        ones = torch.ones(batch_size, 3, device=device)

        if self.fusion_variant == "full":
            return source_encoding, temporal_encoding, tree_encoding, ones

        if self.fusion_variant == "scalar_gate":
            source_gate = torch.sigmoid(self.source_scalar_gate(source_encoding))
            temporal_gate = torch.sigmoid(self.temporal_scalar_gate(temporal_encoding))
            structure_gate = torch.sigmoid(self.structure_scalar_gate(tree_encoding))
            gate_summary = torch.cat([source_gate, temporal_gate, structure_gate], dim=1)
            return source_encoding * source_gate, temporal_encoding * temporal_gate, tree_encoding * structure_gate, gate_summary

        if self.fusion_variant == "vector_gate":
            source_gate = torch.sigmoid(self.source_vector_gate(source_encoding))
            temporal_gate = torch.sigmoid(self.temporal_vector_gate(temporal_encoding))
            structure_gate = torch.sigmoid(self.structure_vector_gate(tree_encoding))
            gate_summary = torch.stack(
                [source_gate.mean(dim=1), temporal_gate.mean(dim=1), structure_gate.mean(dim=1)],
                dim=1,
            )
            return source_encoding * source_gate, temporal_encoding * temporal_gate, tree_encoding * structure_gate, gate_summary

        if self.fusion_variant == "softmax_router":
            router_scores = torch.cat(
                [
                    self.source_scalar_gate(source_encoding),
                    self.temporal_scalar_gate(temporal_encoding),
                    self.structure_scalar_gate(tree_encoding),
                ],
                dim=1,
            )
            gate_summary = torch.softmax(router_scores, dim=1)
            return (
                source_encoding * gate_summary[:, 0:1],
                temporal_encoding * gate_summary[:, 1:2],
                tree_encoding * gate_summary[:, 2:3],
                gate_summary,
            )

        if self.fusion_variant == "structure_gate_only":
            structure_gate = torch.sigmoid(self.structure_scalar_gate(tree_encoding))
            gate_summary = torch.cat([torch.ones_like(structure_gate), torch.ones_like(structure_gate), structure_gate], dim=1)
            return source_encoding, temporal_encoding, tree_encoding * structure_gate, gate_summary

        if self.fusion_variant == "source_gate_only":
            source_gate = torch.sigmoid(self.source_scalar_gate(source_encoding))
            gate_summary = torch.cat([source_gate, torch.ones_like(source_gate), torch.ones_like(source_gate)], dim=1)
            return source_encoding * source_gate, temporal_encoding, tree_encoding, gate_summary

        if self.fusion_variant == "reply_gate_only":
            temporal_gate = torch.sigmoid(self.temporal_scalar_gate(temporal_encoding))
            gate_summary = torch.cat([torch.ones_like(temporal_gate), temporal_gate, torch.ones_like(temporal_gate)], dim=1)
            return source_encoding, temporal_encoding * temporal_gate, tree_encoding, gate_summary
        raise ValueError(f"Unsupported fusion_variant: {self.fusion_variant}")

    def _encode_reply_sequences(self, reply_sequences: list[list[str]]) -> torch.Tensor:
        device = self.future_regressor[-1].weight.device
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
        return hidden[-1]

    def _encode_tree_features(
        self,
        thread_ids: list[str],
        conversation_trees: list[dict[str, str | None]],
        observed_replies: list[list[dict[str, Any]]],
    ) -> torch.Tensor:
        stats = [
            compute_tree_statistics(thread_id, conversation_tree, replies)
            for thread_id, conversation_tree, replies in zip(thread_ids, conversation_trees, observed_replies)
        ]
        tree_tensor = torch.tensor(np.stack(stats, axis=0), dtype=torch.float32, device=self.future_regressor[-1].weight.device)
        return self.tree_proj(tree_tensor)

    def forward(
        self,
        thread_ids: list[str],
        source_texts: list[str],
        reply_sequences: list[list[str]],
        observed_replies: list[list[dict[str, Any]]],
        conversation_trees: list[dict[str, str | None]],
    ) -> dict[str, torch.Tensor]:
        temporal_encoding = self._encode_reply_sequences(reply_sequences)
        source_encoding = self.source_encoder(source_texts)
        tree_encoding = self._encode_tree_features(thread_ids, conversation_trees, observed_replies)
        if self.disable_temporal:
            temporal_encoding = torch.zeros_like(temporal_encoding)
        if self.disable_structure:
            tree_encoding = torch.zeros_like(tree_encoding)
        source_encoding, temporal_encoding, tree_encoding, gate_summary = self._apply_fusion_gates(
            source_encoding,
            temporal_encoding,
            tree_encoding,
        )
        fused_encoding = self.fusion(torch.cat([source_encoding, temporal_encoding, tree_encoding], dim=1))
        if self.disable_affect_state:
            return {
                "predicted_current_affect_state": fused_encoding,
                "predicted_future_neg_ratio": self.direct_regressor(fused_encoding).squeeze(-1),
                "predicted_future_majority_logits": self.direct_classifier(fused_encoding),
                "fusion_gate_means": gate_summary,
            }
        predicted_current_affect_state = self.affect_state_head(fused_encoding)
        return {
            "predicted_current_affect_state": predicted_current_affect_state,
            "predicted_future_neg_ratio": self.future_regressor(predicted_current_affect_state).squeeze(-1),
            "predicted_future_majority_logits": self.future_classifier(predicted_current_affect_state),
            "fusion_gate_means": gate_summary,
        }
