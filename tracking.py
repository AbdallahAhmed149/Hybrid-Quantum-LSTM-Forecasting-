"""Weights & Biases integration (all W&B code lives here).

Design rule: W&B must NEVER break training. Every function degrades to a
warning + no-op if W&B is unavailable, not logged in, or offline.
"""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from config import ROOT, Config


def init_run(cfg: Config, run_name: str, job_type: str = "train", extra_config: Optional[dict] = None):
    """Start a W&B run. Returns the run, or None if W&B can't be used."""
    try:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=run_name,
            job_type=job_type,
            config={**cfg.to_dict(), **(extra_config or {})},
            tags=[cfg.model_type],
            dir=str(ROOT),
        )
        run.define_metric("val_loss", summary="min")
        run.define_metric("val_rmse_c", summary="min")
        return run
    except Exception as e:  # noqa: BLE001
        print(f"[W&B] disabled ({type(e).__name__}: {e}). Training continues without it.")
        return None


def epoch_callback(run) -> Optional[Callable[[int, dict], None]]:
    """Plug into engine.fit(callback=...): logs one point per epoch."""
    if run is None:
        return None

    def _cb(epoch: int, logs: dict) -> None:
        try:
            run.log(logs, step=epoch)
        except Exception as e:  # noqa: BLE001
            print(f"[W&B] log failed: {e}")

    return _cb


def log_training_results(
    run,
    cfg: Config,
    metrics: dict,
    test_df: pd.DataFrame,
    plot_paths: dict,
    model_dir: Path,
    metrics_dir: Path,
) -> None:
    """End-of-training: summary numbers, plots, prediction table, model + metrics artifacts."""
    if run is None:
        return
    try:
        import wandb

        # ---- headline numbers (show up in the runs table, sortable)
        for split in ("val", "test"):
            for k, v in metrics[split].items():
                run.summary[f"{split}/{k}"] = v
        for k, v in metrics["test_persistence_baseline"].items():
            run.summary[f"persistence/{k}"] = v
        run.summary["best_epoch"] = metrics["best_epoch"]
        run.summary["trainable_params"] = metrics["parameters"]["trainable"]
        run.summary["train_seconds"] = metrics["train_seconds"]
        run.summary["beats_persistence"] = bool(
            metrics["test"]["rmse"] < metrics["test_persistence_baseline"]["rmse"]
        )

        # ---- plots + interactive prediction chart + raw table
        run.log({f"plots/{name}": wandb.Image(str(path)) for name, path in plot_paths.items()})
        run.log({
            "test/forecast_chart": wandb.plot.line_series(
                xs=list(range(len(test_df))),
                ys=[test_df["actual"].tolist(), test_df["predicted"].tolist(), test_df["persistence"].tolist()],
                keys=["actual", cfg.model_type, "persistence (yesterday)"],
                title="Test forecast (deg C)",
                xname="test day",
            ),
            "test/predictions": wandb.Table(dataframe=test_df.assign(date=test_df["date"].astype(str))),
        })

        # ---- model artifact (this is what gets promoted to production)
        model_art = wandb.Artifact(
            name=f"model-{cfg.model_type}",
            type="model",
            description=f"{cfg.model_type} temperature forecaster (best-validation weights)",
            metadata={
                "test": metrics["test"],
                "val": metrics["val"],
                "persistence": metrics["test_persistence_baseline"],
                "parameters": metrics["parameters"],
                "best_epoch": metrics["best_epoch"],
                "config": cfg.to_dict(),
            },
        )
        model_art.add_file(str(model_dir / "model.pt"))
        model_art.add_file(str(model_dir / "config.json"))
        aliases = ["production"] if cfg.wandb_promote else None
        logged = run.log_artifact(model_art, aliases=aliases)

        # ---- optional: link into the W&B Model Registry
        if cfg.wandb_registry:
            try:
                logged.wait()
                run.link_artifact(
                    logged,
                    f"wandb-registry-model/{cfg.wandb_registry}",
                    aliases=["production"] if cfg.wandb_promote else None,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[W&B] registry link failed (create the registry collection first?): {e}")

        # ---- every number/plot of the run, as a versioned artifact too
        metrics_art = wandb.Artifact(name=f"metrics-{cfg.model_type}", type="metrics")
        metrics_art.add_dir(str(metrics_dir))
        run.log_artifact(metrics_art)
    except Exception as e:  # noqa: BLE001
        print(f"[W&B] result logging failed: {e}")


def log_inference(run, metrics: dict, forecast: dict, test_df: pd.DataFrame) -> None:
    """Called from predict.py: records what the deployed model is doing."""
    if run is None:
        return
    try:
        import wandb

        for k, v in metrics.items():
            run.summary[f"inference/test_{k}"] = v
        run.summary["inference/next_day_date"] = forecast["date"]
        run.summary["inference/next_day_temp_c"] = forecast["predicted_meantemp_c"]
        run.log({
            "inference/next_day_temp_c": forecast["predicted_meantemp_c"],
            "inference/forecast_chart": wandb.plot.line_series(
                xs=list(range(len(test_df))),
                ys=[test_df["actual"].tolist(), test_df["predicted"].tolist()],
                keys=["actual", "deployed model"],
                title="Deployed model vs actual (deg C)",
                xname="test day",
            ),
        })
    except Exception as e:  # noqa: BLE001
        print(f"[W&B] inference logging failed: {e}")