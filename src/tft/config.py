"""Typed configuration loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    hidden_size: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    n_static_inputs: int = 4
    n_known_inputs: int = 6
    n_observed_inputs: int = 5
    n_targets: int = 1
    encoder_length: int = 168
    decoder_length: int = 24
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class DataConfig:
    csv_path: str = "data/ridership.csv"
    target_col: str = "ridership"
    time_col: str = "timestamp"
    group_col: str = "station_id"
    static_cols: list[str] = field(default_factory=lambda: [
        "station_id", "borough", "line", "is_terminal"
    ])
    known_cols: list[str] = field(default_factory=lambda: [
        "hour", "dow", "month", "is_holiday", "temp_f", "precip_in"
    ])
    observed_cols: list[str] = field(default_factory=lambda: [
        "ridership_lag_1h", "ridership_lag_24h", "ridership_lag_168h",
        "transfers", "delay_count"
    ])
    train_end: str = "2023-12-31"
    val_end: str = "2024-06-30"


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 6
    num_workers: int = 4
    seed: int = 20260514
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"


@dataclass
class MLOpsConfig:
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "tft-transit"
    wandb_project: str = "tft-transit"
    wandb_entity: str | None = None
    register_model: bool = True
    register_threshold_p50_mape: float = 12.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mlops: MLOpsConfig = field(default_factory=MLOpsConfig)


def load_config(path: str | Path) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        mlops=MLOpsConfig(**raw.get("mlops", {})),
    )
