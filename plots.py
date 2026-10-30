"""Plot helpers. Figures are saved to disk (and returned) instead of shown,
so the same code works on a laptop, on Colab and on a headless server."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(history: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (scaled)")
    ax.set_title("Training curves")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return fig


def plot_predictions(dates, actual, predicted, path: Path, title: str = "Daily Mean Temperature Forecasting"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, actual, color="#1f77b4", label="Actual Temperature")
    ax.plot(dates, predicted, color="#d62728", label="Predicted Temperature")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return fig