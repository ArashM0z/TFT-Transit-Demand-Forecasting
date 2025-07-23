"""Transit ridership dataset (CSV-backed, group-aware, sliding-window).

The expected schema follows MTA-style hourly station ridership:
    timestamp, station_id, ridership, hour, dow, month, is_holiday,
    temp_f, precip_in, transfers, delay_count, borough, line, is_terminal,
    ridership_lag_1h, ridership_lag_24h, ridership_lag_168h
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tft.config import DataConfig


@dataclass(frozen=True)
class GroupSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _ensure_dataframe(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    missing = ([cfg.time_col, cfg.group_col, cfg.target_col]
               + cfg.static_cols + cfg.known_cols + cfg.observed_cols)
    missing = [c for c in missing if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns in input data: {missing}")
    df = df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    return df.sort_values([cfg.group_col, cfg.time_col]).reset_index(drop=True)


class TransitRidershipDataset(Dataset):
    """Sliding-window dataset of (encoder, decoder) pairs per group.

    For each group we produce ``T - enc - dec + 1`` samples, advancing by one
    step. The features are normalised per-group using statistics computed on
    the train portion only (set via ``fit_scalers``).
    """

    def __init__(self, df: pd.DataFrame, cfg: DataConfig,
                 encoder_length: int, decoder_length: int,
                 hidden_size: int, indices: np.ndarray | None = None):
        self.cfg = cfg
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.hidden_size = hidden_size
        df = _ensure_dataframe(df, cfg)

        # numeric encode static category cols
        for col in cfg.static_cols:
            if df[col].dtype == object or df[col].dtype.name == "category":
                df[col] = pd.Categorical(df[col]).codes.astype(np.int64)
        self._scaler_means: dict[str, float] = {}
        self._scaler_stds: dict[str, float] = {}
        self.df = df
        self._build_windows(indices)
        # default embeddings: linear projection per feature column
        self._project = torch.nn.Linear(1, hidden_size, bias=True)
        torch.nn.init.xavier_uniform_(self._project.weight)
        torch.nn.init.zeros_(self._project.bias)

    def _build_windows(self, indices: np.ndarray | None) -> None:
        windows: list[tuple[int, int]] = []  # (start, end) row indices per sample
        L = self.encoder_length + self.decoder_length
        for _, group in self.df.groupby(self.cfg.group_col, sort=False):
            idx = group.index.to_numpy()
            for start in range(0, len(idx) - L + 1):
                windows.append((idx[start], idx[start] + L))
        self._windows = np.array(windows)
        if indices is not None:
            self._windows = self._windows[indices]

    def fit_scalers(self) -> "TransitRidershipDataset":
        for col in (self.cfg.known_cols + self.cfg.observed_cols
                    + [self.cfg.target_col]):
            vals = self.df[col].to_numpy(dtype=np.float32)
            mu = float(np.nanmean(vals))
            sigma = float(np.nanstd(vals) + 1e-6)
            self._scaler_means[col] = mu
            self._scaler_stds[col] = sigma
        return self

    def apply_scalers(self, src: "TransitRidershipDataset") -> "TransitRidershipDataset":
        self._scaler_means = dict(src._scaler_means)
        self._scaler_stds = dict(src._scaler_stds)
        return self

    def _normalize(self, col: str, x: np.ndarray) -> np.ndarray:
        mu = self._scaler_means.get(col, 0.0)
        sigma = self._scaler_stds.get(col, 1.0)
        return (x - mu) / sigma

    def __len__(self) -> int:
        return len(self._windows)

    def _project_cols(self, frame: np.ndarray) -> torch.Tensor:
        # frame: (T, n_cols) -> (T, n_cols, hidden_size)
        t = torch.from_numpy(frame.astype(np.float32)).unsqueeze(-1)
        return self._project(t)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        start, end = self._windows[i]
        rows = self.df.iloc[start:end]

        static = rows.iloc[0][self.cfg.static_cols].to_numpy(dtype=np.float32)
        static_t = torch.from_numpy(static).unsqueeze(-1)
        static_emb = self._project(static_t)  # (n_static, hidden)

        known = rows[self.cfg.known_cols].to_numpy(dtype=np.float32)
        for j, col in enumerate(self.cfg.known_cols):
            known[:, j] = self._normalize(col, known[:, j])
        known_emb = self._project_cols(known)  # (L, n_known, hidden)

        observed = rows.iloc[:self.encoder_length][self.cfg.observed_cols] \
            .to_numpy(dtype=np.float32)
        for j, col in enumerate(self.cfg.observed_cols):
            observed[:, j] = self._normalize(col, observed[:, j])
        observed_emb = self._project_cols(observed)  # (enc, n_obs, hidden)

        target = rows.iloc[self.encoder_length:][self.cfg.target_col] \
            .to_numpy(dtype=np.float32)
        target = self._normalize(self.cfg.target_col, target)

        return {
            "static": static_emb,
            "known": known_emb,
            "observed": observed_emb,
            "target": torch.from_numpy(target),
        }


def time_split(df: pd.DataFrame, cfg: DataConfig) -> GroupSplit:
    df = _ensure_dataframe(df, cfg)
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)
    ts = df[cfg.time_col]
    train_idx = np.where(ts <= train_end)[0]
    val_idx = np.where((ts > train_end) & (ts <= val_end))[0]
    test_idx = np.where(ts > val_end)[0]
    return GroupSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def load_csv(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    return _ensure_dataframe(pd.read_csv(path), cfg)


def synthesize(n_stations: int = 6, n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Generate a small synthetic ridership table for tests / smoke runs."""
    rng = np.random.default_rng(seed)
    hours = n_days * 24
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")
    frames = []
    for s in range(n_stations):
        base = 100 + 50 * np.sin(np.arange(hours) * 2 * np.pi / 24)
        noise = rng.normal(0, 10, hours)
        ridership = np.maximum(0, base + noise + 20 * (s % 3)).astype(np.float32)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "station_id": s,
            "borough": s % 5,
            "line": s % 7,
            "is_terminal": int(s % 2 == 0),
            "hour": timestamps.hour.astype(np.float32),
            "dow": timestamps.dayofweek.astype(np.float32),
            "month": timestamps.month.astype(np.float32),
            "is_holiday": np.zeros(hours, dtype=np.float32),
            "temp_f": rng.normal(60, 15, hours).astype(np.float32),
            "precip_in": rng.exponential(0.02, hours).astype(np.float32),
            "transfers": rng.poisson(3, hours).astype(np.float32),
            "delay_count": rng.poisson(0.2, hours).astype(np.float32),
            "ridership": ridership,
            "ridership_lag_1h": np.roll(ridership, 1).astype(np.float32),
            "ridership_lag_24h": np.roll(ridership, 24).astype(np.float32),
            "ridership_lag_168h": np.roll(ridership, 168).astype(np.float32),
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
