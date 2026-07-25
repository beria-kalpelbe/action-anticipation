"""A compact temporal Transformer baseline for pre-extracted video features."""

import torch
from torch import Tensor, nn


class TemporalTransformerAnticipator(nn.Module):
    """Causal Transformer encoder following the common model interface."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        length = features.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=features.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.encoder(self.input_projection(features), mask=causal_mask)
        return self.classifier(encoded)
