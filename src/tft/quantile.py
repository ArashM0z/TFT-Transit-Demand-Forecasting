"""Quantile (pinball) loss used by TFT."""
from __future__ import annotations

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Pinball loss summed over a fixed set of quantiles.

    Inputs:
        predictions: (batch, horizon, n_quantiles)
        targets:     (batch, horizon) or (batch, horizon, 1)
    """

    def __init__(self, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError("quantiles must lie in (0, 1)")
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == predictions.dim() - 1:
            targets = targets.unsqueeze(-1)
        errors = targets - predictions
        q = self.quantiles.view(1, 1, -1).to(predictions.device)
        loss = torch.maximum(q * errors, (q - 1) * errors)
        return loss.mean()
