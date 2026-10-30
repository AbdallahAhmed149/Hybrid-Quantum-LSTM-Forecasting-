"""Training / evaluation engine (no data or model definitions in here)."""
import copy
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn, optim

from config import Config
from dataset import DataBundle
from utils import EarlyStopping


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += criterion(model(x), y).item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def predict(model, loader, y_scaler, device):
    """Returns (y_true, y_pred) both in ORIGINAL units (deg C), shape (N, 1)."""
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.numpy())
    y_pred = y_scaler.inverse_transform(np.concatenate(preds))
    y_true = y_scaler.inverse_transform(np.concatenate(trues))
    return y_true, y_pred


def fit(
    model: nn.Module,
    data: DataBundle,
    cfg: Config,
    device: torch.device,
    checkpoint_path: Path,
    checkpoint_meta: Optional[dict] = None,
    callback: Optional[Callable[[int, dict], None]] = None,
):
    """Train with early stopping + LR scheduling.

    * The best-validation checkpoint is written to `checkpoint_path` every time
      it improves (so a crash never loses progress) and is re-loaded at the end.
    * `callback(epoch, logs)` is called after every epoch - used for W&B.

    Returns (history: dict[list], best_epoch: int, train_seconds: float)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, cfg.patience // 3)
    )
    stopper = EarlyStopping(cfg.patience)

    # scaled-MSE -> deg C: error_C = error_scaled / scale_   (MinMax scaler)
    to_celsius = 1.0 / float(data.y_scaler.scale_[0])

    history = {k: [] for k in
               ["epoch", "train_loss", "val_loss", "train_rmse_c", "val_rmse_c", "lr", "epoch_seconds"]}
    best_epoch = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, data.train_loader, criterion, optimizer, device, cfg.grad_clip)
        val_loss = evaluate_loss(model, data.val_loader, criterion, device)
        scheduler.step(val_loss)

        logs = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse_c": float(np.sqrt(train_loss) * to_celsius),
            "val_rmse_c": float(np.sqrt(val_loss) * to_celsius),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - t0,
        }
        for k, v in logs.items():
            history[k].append(v)

        is_best = stopper.step(val_loss)
        if is_best:
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    **(checkpoint_meta or {}),
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | train {train_loss:.5f} | val {val_loss:.5f} "
            f"| val RMSE {logs['val_rmse_c']:.3f} C | lr {logs['lr']:.1e} "
            f"| {logs['epoch_seconds']:.1f}s{'  *best*' if is_best else ''}"
        )
        if callback is not None:
            callback(epoch, {**logs, "best_val_loss": stopper.best})

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    train_seconds = time.time() - t_start

    # restore the best weights for final evaluation
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return history, best_epoch, train_seconds