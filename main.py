"""Entry point.

    python main.py                                   # default: QLSTM
    python main.py --model_type hybrid
    python main.py --model_type classical
    python main.py --epochs 5 --hidden_size 16       # any Config field is a flag
    python main.py --no-use_wandb                    # train without Weights & Biases
    python main.py --wandb_promote                   # tag the uploaded model as "production"

Every run gets its own folder:
    metrics/<run_name>/   history.csv, metrics.json, test_predictions.csv, *.png
    models/<run_name>/    model.pt (best weights + config), config.json
and one line is appended to metrics/runs_summary.csv for easy comparison.
"""
import argparse
import dataclasses
import time
from pathlib import Path

import pandas as pd

from config import Config
from dataset import get_data
from engine import fit, predict
from model import build_model
from plots import plot_loss_curves, plot_predictions
from tracking import epoch_callback, init_run, log_training_results
from utils import count_parameters, get_device, regression_metrics, save_json, set_seed


def parse_args() -> Config:
    """Auto-generate one CLI flag per scalar field of Config."""
    cfg = Config()
    parser = argparse.ArgumentParser(description="Hybrid Quantum-Classical temperature forecasting")
    for f in dataclasses.fields(Config):
        default = getattr(cfg, f.name)
        if isinstance(default, Path):
            continue # paths are edited in config.py
        if isinstance(default, bool):
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()

    run_name = cfg.wandb_run_name or f"{cfg.model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_metrics_dir = cfg.metrics_dir / run_name
    run_models_dir = cfg.models_dir / run_name
    run_metrics_dir.mkdir(parents=True, exist_ok=True)
    run_models_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg.to_dict(), run_models_dir / "config.json")

    # ------------------------------------------------------------- data
    data = get_data(cfg)
    print(f"Run: {run_name} | device: {device}")
    print(f"Features ({data.num_features}): {data.feature_names}")
    print(f"Samples: {data.sizes}")

    # ------------------------------------------------------------ model
    model = build_model(cfg, data.num_features).to(device)
    params = count_parameters(model)
    print(f"Model: {cfg.model_type} | parameters: {params['trainable']:,}")

    # ------------------------------------------------------------ W&B
    run = init_run(cfg, run_name, extra_config={"parameters": params, "samples": data.sizes}) if cfg.use_wandb else None

    # ------------------------------------------------------------ train
    checkpoint_meta = {
        "config": cfg.to_dict(),
        "input_size": data.num_features,
        "feature_names": data.feature_names,
    }
    history, best_epoch, train_seconds = fit(
        model, data, cfg, device,
        checkpoint_path=run_models_dir / "model.pt",
        checkpoint_meta=checkpoint_meta,
        callback=epoch_callback(run),
    )

    # --------------------------------------------------------- evaluate
    val_true, val_pred = predict(model, data.val_loader, data.y_scaler, device)
    test_true, test_pred = predict(model, data.test_loader, data.y_scaler, device)

    metrics = {
        "run_name": run_name,
        "model_type": cfg.model_type,
        "parameters": params,
        "best_epoch": best_epoch,
        "epochs_run": len(history["epoch"]),
        "train_seconds": round(train_seconds, 1),
        "samples": data.sizes,
        "val": regression_metrics(val_true, val_pred),
        "test": regression_metrics(test_true, test_pred),
        # "tomorrow = today": the bar any real forecasting model must beat
        "test_persistence_baseline": regression_metrics(data.y_test_actual, data.y_test_persistence),
    }
    save_json(metrics, run_metrics_dir / "metrics.json")

    # ------------------------------------------------------ save results
    pd.DataFrame(history).to_csv(run_metrics_dir / "history.csv", index=False)
    test_df = pd.DataFrame({
        "date": data.test_dates,
        "actual": test_true.ravel(),
        "predicted": test_pred.ravel(),
        "persistence": data.y_test_persistence.ravel(),
    })
    test_df.to_csv(run_metrics_dir / "test_predictions.csv", index=False)

    plot_loss_curves(history, run_metrics_dir / "loss_curve.png")
    plot_predictions(data.test_dates, test_true, test_pred, run_metrics_dir / "predictions.png",
                     title=f"{cfg.model_type.upper()} - Daily Mean Temperature Forecasting")

    # one row per run -> compare all experiments in a single table
    row = pd.DataFrame([{
        "run_name": run_name, "model_type": cfg.model_type,
        "trainable_params": params["trainable"], "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 1),
        **{f"test_{k}": round(v, 4) for k, v in metrics["test"].items()},
        "persistence_r2": round(metrics["test_persistence_baseline"]["r2"], 4),
    }])
    summary_path = cfg.metrics_dir / "runs_summary.csv"
    row.to_csv(summary_path, mode="a", header=not summary_path.exists(), index=False)

    # ------------------------------------------- upload everything to W&B
    log_training_results(
        run, cfg, metrics, test_df,
        plot_paths={"loss_curve": run_metrics_dir / "loss_curve.png",
                    "predictions": run_metrics_dir / "predictions.png"},
        model_dir=run_models_dir, metrics_dir=run_metrics_dir,
    )
    if run is not None:
        print(f"W&B run: {run.url}")
        run.finish()

    # ----------------------------------------------------------- report
    t, p = metrics["test"], metrics["test_persistence_baseline"]
    print("\n================ TEST RESULTS ================")
    print(f"{cfg.model_type:>12}: R2 {t['r2']:.4f} | MAE {t['mae']:.3f} C | RMSE {t['rmse']:.3f} C")
    print(f"{'persistence':>12}: R2 {p['r2']:.4f} | MAE {p['mae']:.3f} C | RMSE {p['rmse']:.3f} C")
    print(f"Saved metrics -> {run_metrics_dir}\nSaved model   -> {run_models_dir}")


if __name__ == "__main__":
    main()