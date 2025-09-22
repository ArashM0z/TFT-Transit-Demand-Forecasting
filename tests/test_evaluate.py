import numpy as np

from tft.evaluate import (
    mean_absolute_percentage_error,
    quantile_coverage,
    symmetric_mape,
)


def test_mape_zero_when_perfect():
    assert mean_absolute_percentage_error(np.ones(10), np.ones(10)) == 0.0


def test_smape_symmetric_in_sign_flip():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    assert abs(symmetric_mape(a, b) - symmetric_mape(b, a)) < 1e-9


def test_quantile_coverage_full_interval():
    target = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.5, 1.5, 2.5])
    upper = np.array([1.5, 2.5, 3.5])
    assert quantile_coverage(lower, upper, target) == 1.0


def test_quantile_coverage_empty_interval():
    target = np.array([1.0, 2.0, 3.0])
    lower = np.array([10.0, 10.0, 10.0])
    upper = np.array([11.0, 11.0, 11.0])
    assert quantile_coverage(lower, upper, target) == 0.0
