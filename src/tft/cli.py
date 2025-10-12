"""Command-line entrypoints."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from tft.config import load_config
from tft.data import load_csv, synthesize
from tft.evaluate import evaluate
from tft.model import TemporalFusionTransformer
from tft.train import build_loaders, train


def _common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path,
                   help="path to CSV; omit to use synthetic data")
    p.add_argument("--log-level", default="INFO")
    return p


def train_main() -> int:
    args = _common_parser().parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    df = load_csv(args.data, cfg.data) if args.data else synthesize()
    metrics = train(cfg, df)
    for k, v in metrics.items():
        print(f"{k}={v:.4f}")
    return 0


def eval_main() -> int:
    p = _common_parser()
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    df = load_csv(args.data, cfg.data) if args.data else synthesize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalFusionTransformer(
        n_static_inputs=cfg.model.n_static_inputs,
        n_known_inputs=cfg.model.n_known_inputs,
        n_observed_inputs=cfg.model.n_observed_inputs,
        hidden_size=cfg.model.hidden_size,
        n_heads=cfg.model.n_heads,
        encoder_length=cfg.model.encoder_length,
        decoder_length=cfg.model.decoder_length,
        quantiles=cfg.model.quantiles,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    _, _, test_loader = build_loaders(df, cfg)
    metrics = evaluate(model, test_loader, device)
    for k, v in metrics.items():
        print(f"{k}={v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
