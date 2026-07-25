"""Modern, self-contained implementation of Rolling-Unrolling LSTM."""

import torch
from torch import Tensor, nn


class RULSTMAnticipator(nn.Module):
    """Predict one action distribution for every observed temporal step.

    Inputs have shape ``[batch, time, feature]``. Outputs have shape
    ``[batch, time, num_classes]``; the final time step is the main prediction.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.dropout = nn.Dropout(dropout)
        self.rolling = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=recurrent_dropout)
        self.unrolling = nn.LSTM(feature_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True, dropout=recurrent_dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        observed = self.dropout(features)
        rolling_states, _ = self.rolling(observed)
        predictions: list[Tensor] = []
        sequence_length = observed.shape[1]
        for step in range(sequence_length):
            # Repeat the current observation for the remaining anticipation steps.
            remaining = sequence_length - step
            future_input = observed[:, step : step + 1].expand(-1, remaining, -1)
            step_hidden = rolling_states[:, step]
            initial_hidden = step_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
            initial_cell = torch.zeros_like(initial_hidden)
            unrolled, _ = self.unrolling(self.dropout(future_input), (initial_hidden, initial_cell))
            predictions.append(self.classifier(unrolled[:, -1]))
        return torch.stack(predictions, dim=1)
