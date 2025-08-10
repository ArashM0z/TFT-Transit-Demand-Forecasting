import pandas as pd

from tft.config import DataConfig
from tft.data import TransitRidershipDataset, synthesize, time_split


def test_synthesize_has_required_columns():
    df = synthesize(n_stations=2, n_days=3)
    cfg = DataConfig()
    for col in [cfg.time_col, cfg.group_col, cfg.target_col,
                *cfg.static_cols, *cfg.known_cols, *cfg.observed_cols]:
        assert col in df.columns


def test_time_split_partitions_disjoint_index_sets():
    df = synthesize(n_stations=2, n_days=5)
    cfg = DataConfig(train_end="2024-01-02", val_end="2024-01-04")
    splits = time_split(df, cfg)
    s = set(splits.train_idx) | set(splits.val_idx) | set(splits.test_idx)
    assert len(s) == len(df)
    assert not set(splits.train_idx) & set(splits.val_idx)
    assert not set(splits.val_idx) & set(splits.test_idx)


def test_dataset_produces_correct_tensor_shapes():
    df = synthesize(n_stations=2, n_days=10)
    cfg = DataConfig()
    ds = TransitRidershipDataset(df, cfg, encoder_length=24,
                                 decoder_length=6, hidden_size=8).fit_scalers()
    assert len(ds) > 0
    sample = ds[0]
    assert sample["static"].shape == (4, 8)
    assert sample["known"].shape == (30, 6, 8)
    assert sample["observed"].shape == (24, 5, 8)
    assert sample["target"].shape == (6,)


def test_scalers_normalize_to_unit_variance_approximately():
    df = synthesize(n_stations=3, n_days=20)
    cfg = DataConfig()
    ds = TransitRidershipDataset(df, cfg, encoder_length=24,
                                 decoder_length=6, hidden_size=8).fit_scalers()
    for col in cfg.known_cols:
        assert abs(ds._scaler_stds[col]) > 1e-6


def test_missing_column_raises_keyerror():
    import pytest
    cfg = DataConfig()
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "station_id": [0],
                       "ridership": [100.0]})
    with pytest.raises(KeyError):
        TransitRidershipDataset(df, cfg, encoder_length=24,
                                decoder_length=6, hidden_size=8)
