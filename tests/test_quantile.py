import torch

from tft.quantile import QuantileLoss


def test_quantile_loss_zero_when_predictions_equal_targets():
    crit = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
    target = torch.zeros(2, 4)
    pred = torch.zeros(2, 4, 3)
    assert crit(pred, target).item() == 0.0


def test_quantile_loss_penalises_underprediction_more_for_high_quantile():
    crit = QuantileLoss(quantiles=(0.9,))
    target = torch.ones(1, 1)
    under = crit(torch.zeros(1, 1, 1), target).item()
    over = crit(2 * torch.ones(1, 1, 1), target).item()
    assert under > over


def test_quantile_loss_rejects_invalid_quantiles():
    import pytest
    with pytest.raises(ValueError):
        QuantileLoss(quantiles=(0.0, 0.5))
    with pytest.raises(ValueError):
        QuantileLoss(quantiles=(0.5, 1.0))
