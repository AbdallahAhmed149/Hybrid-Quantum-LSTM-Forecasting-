"""Data pipeline: CSV -> scaled sliding windows -> DataLoaders."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from config import Config


@dataclass
class DataBundle:
    """Everything the rest of the pipeline needs from the data step."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    y_scaler: MinMaxScaler
    num_features: int
    feature_names: list
    test_dates: pd.Series          # one date per test prediction
    y_test_actual: np.ndarray      # test targets in ORIGINAL units (deg C), shape (N, 1)
    y_test_persistence: np.ndarray # "tomorrow = today" naive forecast for the same days, (N, 1)
    sizes: dict                    # number of samples per split (for logging)
    x_scaler: MinMaxScaler         # fitted on train rows only
    last_window: np.ndarray        # scaled last `seq_length` days, shape (1, seq, F) -> forecast of the NEXT day
    last_date: pd.Timestamp        # last date present in the data


def load_dataframe(cfg: Config) -> pd.DataFrame:
    train_df = pd.read_csv(cfg.train_csv)
    test_df = pd.read_csv(cfg.test_csv)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_sequences(X: np.ndarray, y: np.ndarray, seq_length: int):
    """Window i..i+seq-1 of features  ->  target at day i+seq."""
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i : i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


def _to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_data(cfg: Config) -> DataBundle:
    df = load_dataframe(cfg)

    feature_cols = [c for c in df.columns if c != "date"]
    if not cfg.use_target_as_input:
        feature_cols.remove(cfg.target)

    X_raw = df[feature_cols].values
    y_raw = df[cfg.target].values.reshape(-1, 1)

    # ---- chronological split point (row index of the first test day)
    split_idx = int(len(df) * cfg.train_ratio)

    # ---- scalers are fitted on TRAIN rows only (no information leak)
    x_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_raw[:split_idx])
    y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y_raw[:split_idx])
    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw)

    # ---- build windows over the whole series, then split by TARGET day.
    # Test windows may look back into the end of the train period (that is
    # perfectly legal: it is just history), so no test day is lost.
    X_seq, y_seq = build_sequences(X_scaled, y_scaled, cfg.seq_length)
    target_idx = np.arange(cfg.seq_length, len(df))     # day each window predicts

    train_mask = target_idx < split_idx
    test_mask = ~train_mask

    X_trainval, y_trainval = X_seq[train_mask], y_seq[train_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]

    val_size = int(cfg.val_ratio * len(X_trainval))
    train_size = len(X_trainval) - val_size
    X_train, y_train = X_trainval[:train_size], y_trainval[:train_size]
    X_val, y_val = X_trainval[train_size:], y_trainval[train_size:]

    return DataBundle(
        train_loader=_to_loader(X_train, y_train, cfg.batch_size, shuffle=True),
        val_loader=_to_loader(X_val, y_val, cfg.batch_size, shuffle=False),
        test_loader=_to_loader(X_test, y_test, cfg.batch_size, shuffle=False),
        y_scaler=y_scaler,
        num_features=X_seq.shape[-1],
        feature_names=feature_cols,
        test_dates=df["date"].iloc[target_idx[test_mask]].reset_index(drop=True),
        y_test_actual=y_scaler.inverse_transform(y_test),
        y_test_persistence=y_raw[target_idx[test_mask] - 1],
        x_scaler=x_scaler,
        last_window=X_scaled[-cfg.seq_length:][None].astype(np.float32),
        last_date=df["date"].iloc[-1],
        sizes={"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    )