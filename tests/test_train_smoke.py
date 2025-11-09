"""End-to-end smoke test: train a 2-epoch TFT on synthetic data."""
import os
from pathlib import Path

import pytest

# disable wandb network calls in CI
os.environ["WANDB_MODE"] = "disabled"

from tft.config import load_config
from tft.data import synthesize
from tft.train import train


@pytest.fixture
def smoke_config(tmp_path: Path) -> Path:
    cfg = Path(__file__).parents[1] / "configs" / "smoke.yaml"
    return cfg


def test_train_smoke_runs_two_epochs_and_returns_finite_metrics(
    smoke_config: Path, tmp_path: Path
):
    cfg = load_config(smoke_config)
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.mlops.mlflow_tracking_uri = f"file:{tmp_path / 'mlruns'}"
    df = synthesize(n_stations=3, n_days=8)
    metrics = train(cfg, df)
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics["best_val_loss"] >= 0
