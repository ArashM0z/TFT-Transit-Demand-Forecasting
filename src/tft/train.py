"""Training loop with MLflow + W&B dual logging and model registry promotion."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from tft.config import Config
from tft.data import TransitRidershipDataset, time_split
from tft.model import TemporalFusionTransformer
from tft.quantile import QuantileLoss

log = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    _WANDB_AVAILABLE = False


def _init_wandb(cfg: Config, run_name: str) -> "wandb.sdk.wandb_run.Run | None":
    if not _WANDB_AVAILABLE or os.environ.get("WANDB_MODE") == "disabled":
        return None
    return wandb.init(
        project=cfg.mlops.wandb_project,
        entity=cfg.mlops.wandb_entity,
        name=run_name,
        config={
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
        },
        reinit=True,
        resume="allow",
    )


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(df, cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    splits = time_split(df, cfg.data)
    train_ds = TransitRidershipDataset(
        df.iloc[splits.train_idx], cfg.data,
        cfg.model.encoder_length, cfg.model.decoder_length,
        cfg.model.hidden_size,
    ).fit_scalers()
    val_ds = TransitRidershipDataset(
        df.iloc[splits.val_idx], cfg.data,
        cfg.model.encoder_length, cfg.model.decoder_length,
        cfg.model.hidden_size,
    ).apply_scalers(train_ds)
    test_ds = TransitRidershipDataset(
        df.iloc[splits.test_idx], cfg.data,
        cfg.model.encoder_length, cfg.model.decoder_length,
        cfg.model.hidden_size,
    ).apply_scalers(train_ds)
    common = dict(batch_size=cfg.train.batch_size,
                  num_workers=cfg.train.num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **common),
            DataLoader(val_ds, shuffle=False, **common),
            DataLoader(test_ds, shuffle=False, **common))


def epoch_loop(model: TemporalFusionTransformer, loader: DataLoader,
               criterion: QuantileLoss, device: torch.device,
               optimizer: torch.optim.Optimizer | None = None,
               grad_clip: float | None = None) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    n = 0
    p50_errs: list[float] = []

    for batch in loader:
        static = batch["static"].to(device)
        known = batch["known"].to(device)
        observed = batch["observed"].to(device)
        target = batch["target"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(static, known, observed)
            loss = criterion(out["predictions"], target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            with torch.no_grad():
                median = out["predictions"][..., out["predictions"].shape[-1] // 2]
                p50_errs.append(((median - target).abs()
                                / (target.abs() + 1e-3)).mean().item())

        total += loss.item() * static.size(0)
        n += static.size(0)

    return total / max(n, 1), {"p50_mape": float(np.mean(p50_errs)) * 100}


def train(cfg: Config, df) -> dict[str, float]:
    _seed_all(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available()
                          else "cpu")

    mlflow.set_tracking_uri(cfg.mlops.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlops.mlflow_experiment)

    train_loader, val_loader, test_loader = build_loaders(df, cfg)
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

    criterion = QuantileLoss(quantiles=cfg.model.quantiles).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs,
    )

    run_name = f"tft-{int(time.time())}"
    wandb_run = _init_wandb(cfg, run_name)
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    bad = 0
    best_path = Path(cfg.train.checkpoint_dir) / "best.pt"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            **{f"model.{k}": v for k, v in cfg.model.__dict__.items()},
            **{f"train.{k}": v for k, v in cfg.train.__dict__.items()},
        })
        for epoch in range(cfg.train.epochs):
            t0 = time.time()
            train_loss, train_metrics = epoch_loop(
                model, train_loader, criterion, device, optimizer,
                grad_clip=cfg.train.grad_clip,
            )
            val_loss, val_metrics = epoch_loop(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            row = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_p50_mape": train_metrics["p50_mape"],
                "val_p50_mape": val_metrics["p50_mape"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_sec": elapsed,
            }
            mlflow.log_metrics(row, step=epoch)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            log.info("ep=%03d train=%.4f val=%.4f p50_mape=%.2f%% (%.1fs)",
                     epoch, train_loss, val_loss, val_metrics["p50_mape"], elapsed)

            if val_loss < best_val:
                best_val = val_loss
                bad = 0
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
            else:
                bad += 1
                if bad >= cfg.train.early_stopping_patience:
                    log.info("early stopping at epoch %d", epoch)
                    break

        # final test evaluation
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_metrics = epoch_loop(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": test_loss,
                            "test_p50_mape": test_metrics["p50_mape"]})

        # log model
        mlflow.pytorch.log_model(model, artifact_path="model")
        if (cfg.mlops.register_model
                and test_metrics["p50_mape"] <= cfg.mlops.register_threshold_p50_mape):
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=cfg.mlops.wandb_project,
            )
            log.info("registered model under name=%s", cfg.mlops.wandb_project)

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_p50_mape": test_metrics["p50_mape"],
    }
