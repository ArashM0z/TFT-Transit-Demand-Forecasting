"""Temporal Fusion Transformer for transit ridership forecasting.

Reference: Lim, Arik, Loeff, Pfister (2021), "Temporal Fusion Transformers for
interpretable multi-horizon time series forecasting", International Journal of
Forecasting 37(4).
"""

from tft.config import Config, load_config
from tft.model import TemporalFusionTransformer
from tft.quantile import QuantileLoss

__all__ = ["TemporalFusionTransformer", "QuantileLoss", "Config", "load_config"]
__version__ = "0.3.0"
