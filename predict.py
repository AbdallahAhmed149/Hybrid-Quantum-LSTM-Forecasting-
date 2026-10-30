"""Inference with a trained model.

    python predict.py                                  # newest local qlstm run
    python predict.py --model_type hybrid
    python predict.py --run qlstm_20261030_101500      # a specific local run
    python predict.py --from_wandb                     # download the "production" version from W&B
    python predict.py --from_wandb --alias latest
    python predict.py --from_wandb --log_wandb         # ... and log this inference run to W&B

Prints test-set metrics of the loaded model and the forecast for the day AFTER
the last date in the data.
"""
import argparse
import dataclasses
import time
from pathlib import Path

import pandas as pd
import torch

from config import Config
from dataset import get_data
from engine import predict
from model import build_model
from tracking import init_run, log_inference
from utils import get_device, regression_metrics, save_json


def find_local_checkpoint(cfg: Config, model_type: str, run: str) -> Path:
    if run:
        path = cfg.models_dir / run / "model.pt"
    else:
        candidates = sorted(cfg.models_dir.glob(f"{model_type}_*/model.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No trained '{model_type}' model in {cfg.models_dir}. Run main.py first.")
        path = candidates[-1]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def download_from_wandb(base_cfg: Config, model_type: str, alias: str, run) -> Path:
    import wandb

    entity = f"{base_cfg.wandb_entity}/" if base_cfg.wandb_entity else ""
    ref = f"{entity}{base_cfg.wandb_project}/model-{model_type}:{alias}"
    if run is not None:
        artifact = run.use_artifact(ref, type="model")      # records lineage: this run USED that model
    else:
        artifact = wandb.Api().artifact(ref, type="model")
    folder = Path(artifact.download(root=str(base_cfg.models_dir / f"wandb_{model_type}_{alias}")))
    print(f"Downloaded {ref} (version {artifact.version}) -> {folder}")
    return folder / "model.pt"


def cfg_from_checkpoint(ckpt: dict) -> Config:
    """Rebuild the training Config but keep THIS machine's paths."""
    cfg = Config()
    path_fields = {f.name for f in dataclasses.fields(Config) if isinstance(getattr(cfg, f.name), Path)}
    for k, v in ckpt["config"].items():
        if k not in path_fields and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    base = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=base.model_type, choices=["qlstm", "hybrid", "classical"])
    parser.add_argument("--run", default="", help="local run folder name inside models/")
    parser.add_argument("--from_wandb", action="store_true", help="download the model artifact from W&B")
    parser.add_argument("--alias", default="production", help="W&B artifact alias (production / latest / v3 ...)")
    parser.add_argument("--log_wandb", action="store_true", help="log this inference run to W&B")
    args = parser.parse_args()

    device = get_device()
    run = None
    if args.log_wandb or args.from_wandb:
        run = init_run(base, f"inference_{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}", job_type="inference")

    # ------------------------------------------------------- load model
    if args.from_wandb:
        ckpt_path = download_from_wandb(base, args.model_type, args.alias, run)
    else:
        ckpt_path = find_local_checkpoint(base, args.model_type, args.run)
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cfg_from_checkpoint(ckpt)

    model = build_model(cfg, ckpt["input_size"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ------------------------------------------- test-set sanity metrics
    data = get_data(cfg)
    y_true, y_pred = predict(model, data.test_loader, data.y_scaler, device)
    metrics = regression_metrics(y_true, y_pred)
    print(f"[{cfg.model_type}] test R2 {metrics['r2']:.4f} | MAE {metrics['mae']:.3f} C | RMSE {metrics['rmse']:.3f} C")

    # ------------------------------------------------ next-day forecast
    with torch.no_grad():
        scaled = model(torch.tensor(data.last_window, dtype=torch.float32, device=device)).cpu().numpy()
    next_temp = float(data.y_scaler.inverse_transform(scaled)[0, 0])
    next_date = (data.last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    forecast = {"date": next_date, "predicted_meantemp_c": round(next_temp, 2),
                "model_type": cfg.model_type, "checkpoint": str(ckpt_path)}
    print(f"Forecast for {next_date}: {next_temp:.2f} C")

    # ------------------------------------------------------------ save
    out_dir = ckpt_path.parent.name
    save_json({"test_metrics": metrics, "forecast": forecast}, base.metrics_dir / out_dir / "forecast.json")

    test_df = pd.DataFrame({"date": data.test_dates, "actual": y_true.ravel(), "predicted": y_pred.ravel()})
    log_inference(run, metrics, forecast, test_df)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()