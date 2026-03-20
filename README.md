# Temporal Fusion Transformer — Transit Ridership Forecasting

PyTorch implementation of the Temporal Fusion Transformer (Lim et al., 2021,
[arXiv:1912.09363](https://arxiv.org/abs/1912.09363)) for multi-horizon
station-level transit ridership forecasting with quantile prediction
intervals.

## What's in the box

- Full TFT architecture: Variable Selection Networks, Gated Residual Networks,
  LSTM encoder/decoder, interpretable multi-head attention, quantile heads.
- Sliding-window dataset over CSV-backed hourly ridership data with per-group
  scaler fitting on the train portion.
- `mlflow` + `wandb` dual logging, with conditional model registry promotion
  gated on a held-out P50 MAPE threshold.
- Cosine-annealed AdamW, gradient clipping, early stopping, per-epoch
  checkpoint upload.
- Smoke config + synthetic-data generator so the full training loop runs
  end-to-end on a laptop CPU in seconds.

## Layout

```
src/tft/
├── config.py        # dataclass config + YAML loader
├── data.py          # CSV-backed sliding-window dataset (+ synthesizer)
├── model.py         # TFT model (VSN, GRN, IMHA, quantile heads)
├── quantile.py      # pinball loss
├── train.py         # training loop with MLflow + W&B
├── evaluate.py      # MAPE, sMAPE, coverage of prediction intervals
└── cli.py           # tft-train / tft-eval entrypoints
configs/             # default.yaml + smoke.yaml
tests/               # 17+ tests including end-to-end smoke train
```

## Quickstart

```bash
pip install -e ".[dev]"

# Smoke train (synthetic data, 2 epochs, CPU): < 30s
WANDB_MODE=disabled tft-train --config configs/smoke.yaml

# Full train on real CSV: schema documented in src/tft/data.py
tft-train --config configs/default.yaml --data data/ridership.csv
```

## Inference / evaluation

```bash
tft-eval --config configs/default.yaml --checkpoint checkpoints/best.pt
```

Outputs MAPE / sMAPE / 80% prediction-interval coverage on the held-out
test fold.

## MLOps integration

Training writes to MLflow (tracking URI configurable) and Weights & Biases.
A run that beats `mlops.register_threshold_p50_mape` is auto-registered in
the MLflow Model Registry under `mlops.wandb_project`. Disable W&B with
`WANDB_MODE=disabled`.

## Data schema

The expected CSV layout (one row per station × hour) is documented in
[src/tft/data.py](src/tft/data.py). The `synthesize()` helper produces a
schema-conforming sample for development / CI.
