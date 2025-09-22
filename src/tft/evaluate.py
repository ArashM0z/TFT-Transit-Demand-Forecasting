"""Evaluation utilities: MAPE, sMAPE, coverage of prediction intervals."""
from __future__ import annotations

import numpy as np
import torch

from tft.model import TemporalFusionTransformer
from tft.quantile import QuantileLoss


def mean_absolute_percentage_error(pred: np.ndarray, target: np.ndarray,
                                   eps: float = 1e-3) -> float:
    return float(np.mean(np.abs(pred - target) / (np.abs(target) + eps))) * 100


def symmetric_mape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-3) -> float:
    denom = (np.abs(pred) + np.abs(target)) / 2 + eps
    return float(np.mean(np.abs(pred - target) / denom)) * 100


def quantile_coverage(lower: np.ndarray, upper: np.ndarray,
                      target: np.ndarray) -> float:
    """Fraction of observations falling within [lower, upper]."""
    return float(np.mean((target >= lower) & (target <= upper)))


@torch.no_grad()
def evaluate(model: TemporalFusionTransformer, loader, device: torch.device
             ) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        static = batch["static"].to(device)
        known = batch["known"].to(device)
        observed = batch["observed"].to(device)
        out = model(static, known, observed)
        preds.append(out["predictions"].cpu().numpy())
        targets.append(batch["target"].numpy())
    p = np.concatenate(preds, axis=0)  # (N, horizon, n_quantiles)
    t = np.concatenate(targets, axis=0)
    n_q = p.shape[-1]
    median = p[..., n_q // 2]
    lower = p[..., 0]
    upper = p[..., -1]
    crit = QuantileLoss(quantiles=tuple(np.linspace(0.1, 0.9, n_q)))
    loss = crit(torch.from_numpy(p), torch.from_numpy(t)).item()
    return {
        "quantile_loss": loss,
        "mape": mean_absolute_percentage_error(median, t),
        "smape": symmetric_mape(median, t),
        "coverage_80": quantile_coverage(lower, upper, t),
    }
