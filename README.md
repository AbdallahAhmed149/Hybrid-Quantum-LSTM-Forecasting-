# Hybrid Quantum LSTM Forecasting

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38%2B-6C2E8E)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reproducible research codebase for **one-day-ahead daily mean temperature forecasting** using a hybrid quantum-classical recurrent neural network. The project compares three model families under the same chronological data split:

- **QLSTM**: quantum circuits replace the four recurrent LSTM gates.
- **Hybrid Quantum LSTM**: a classical LSTM produces a latent representation that is encoded into a variational quantum circuit.
- **Classical LSTM**: a non-quantum baseline for controlled comparison.

The default target is `meantemp`, measured in degrees Celsius, using the Daily Delhi Climate train/test CSV convention.

## Abstract and Overview

The pipeline learns a next-day regression function from a rolling window of the previous **10 days**. Numeric features are scaled with `MinMaxScaler(feature_range=(-1, 1))`, with scalers fitted only on the chronological training rows to prevent future-data leakage. The trained model predicts the held-out test period and a forecast for the day after the last available observation.

The engineering focus is reproducibility and experiment traceability: configuration is centralized in `config.py`, seeds are controlled, the best validation checkpoint is restored after early stopping, metrics are written in original temperature units, and optional Weights & Biases integration records configurations, plots, predictions, and model artifacts.

> **Benchmarking note:** This repository does not hard-code accuracy, latency, throughput, or dataset-volume claims. Every run writes measured values to `metrics/<run_name>/metrics.json` and appends a row to `metrics/runs_summary.csv`. Use those artifacts as the source of truth for published results.

## Architecture and Pipeline

```text
DailyDelhiClimateTrain.csv + DailyDelhiClimateTest.csv
                         |
                         v
              Chronological dataframe merge
                         |
                         v
       Train-only MinMax scaling and leakage-safe split
                         |
                         v
      Sliding windows: 10 days of history -> next day
                         |
              +----------+----------+
              |                     |
              v                     v
       Train / validation       Test windows
              |
              v
  QLSTM | Hybrid Quantum LSTM | Classical LSTM
              |
              v
 MSE loss + Adam + gradient clipping + LR reduction
              |
              v
 Best validation checkpoint + early stopping
              |
              v
 Original-unit metrics, plots, CSV predictions, W&B artifacts
              |
              v
 Forecast for the day after the final observed date
```

### Data processing

1. `dataset.py` loads and chronologically sorts the two CSV files.
2. The target column is `meantemp`; all other numeric columns are candidate features unless `use_target_as_input=False`.
3. Feature and target scalers are fitted on training rows only.
4. Sliding windows are built across the full ordered series, then assigned to train/test by the day being predicted. Test windows may use historical observations from the end of the training period, but never future target values.
5. The training portion is split chronologically into training and validation subsets.

### Model and training

The default QLSTM uses a hidden size of **32**, **4 qubits**, **3 strongly entangling quantum layers**, and data re-uploading. Each QLSTM gate has an independent variational circuit. The hybrid model uses a classical LSTM followed by a learned projection into quantum rotation angles and a learned readout of all qubit expectation values.

Training uses Adam, scaled-space MSE, gradient clipping, `ReduceLROnPlateau`, and early stopping. The best validation checkpoint is saved and reloaded before evaluation.

### Inference and observability

`predict.py` loads the newest local checkpoint by default, evaluates it on the test set, and produces a next-day forecast. It can also download a versioned model artifact from Weights & Biases. Each training run stores configuration, history, metrics, predictions, plots, and model weights under run-specific directories.

## Performance Metrics

The following results are the measured outputs from the three completed runs stored in `metrics/`. All test metrics are calculated in original degrees Celsius on the same **316 held-out samples**. The persistence baseline means “tomorrow = today” and is evaluated on exactly the same dates.

| Model | Test R² | Test MAE (°C) | Test RMSE (°C) | Test MAPE | Parameters | Training time | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| **QLSTM** | **0.9426** | **1.2158** | **1.5784** | **5.3699%** | 965 | 1,967.9 s | 45 |
| Hybrid Quantum LSTM | 0.9416 | 1.2318 | 1.5914 | 5.4569% | 5,037 | 49.3 s | 44 |
| Classical LSTM | 0.9395 | 1.2512 | 1.6193 | 5.6102% | 5,001 | 4.7 s | 34 |
| Persistence baseline | 0.9357 | 1.2572 | 1.6703 | 5.5500% | — | — | — |

### Model-by-model interpretation

- **QLSTM** achieved the strongest test performance: R² **0.9426**, MAE **1.2158 °C**, and RMSE **1.5784 °C**. It improved RMSE by approximately **5.50%** versus persistence and used only **965 trainable parameters**, but required **1,967.9 seconds** because the four recurrent gates are simulated with independent variational quantum circuits.
- **Hybrid Quantum LSTM** was the second-best model by test error, with R² **0.9416**, MAE **1.2318 °C**, and RMSE **1.5914 °C**. It improved RMSE by approximately **4.72%** versus persistence and completed in **49.3 seconds**. Its classical LSTM plus quantum projection used **5,037 trainable parameters**.
- **Classical LSTM** was the fastest reference model at **4.7 seconds**. It reached R² **0.9395**, MAE **1.2512 °C**, and RMSE **1.6193 °C**, improving RMSE by approximately **3.05%** versus persistence with **5,001 trainable parameters**.

The results show a clear accuracy/runtime trade-off: QLSTM produced the best measured test metrics and the smallest parameter count, while the classical baseline was approximately **419× faster** than QLSTM. The hybrid model recovered much of the QLSTM accuracy at approximately **40× lower training time** than QLSTM. These are training durations for the recorded runs, not production inference latency or throughput benchmarks.

### Reproducibility and training configuration

All three runs used the same data split: **1,125 training**, **125 validation**, and **316 test** samples. The shared configuration uses a **10-day input sequence**, batch size **16**, maximum **100 epochs**, Adam learning rate **1e-3**, early-stopping patience **15**, and seed **42**. The default QLSTM architecture uses hidden size **32**, **4 qubits**, **3 strongly entangling quantum layers**, and data re-uploading.

The raw experiment artifacts are available at:

- `metrics/classical_20261030_002413/metrics.json`
- `metrics/hybrid_20261030_002437/metrics.json`
- `metrics/qlstm_20261030_002652/metrics.json`
- `metrics/runs_summary.csv`

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.0+** for tensor operations, recurrent models, optimization, and checkpoints
- **PennyLane 0.38+** for differentiable variational quantum circuits and `TorchLayer`
- **pandas / NumPy** for time-series loading and numerical processing
- **scikit-learn** for leakage-safe MinMax scaling and regression metrics
- **Matplotlib** for headless training and forecast plots
- **Weights & Biases** for optional experiment tracking, model artifacts, and inference lineage
- **CPU or CUDA** through PyTorch; quantum simulation defaults to PennyLane `default.qubit`

## Repository Layout

```text
.
├── config.py          # Central paths, data, model, and training configuration
├── dataset.py         # CSV loading, scaling, windows, and DataLoaders
├── model.py           # QLSTM, hybrid, and classical LSTM implementations
├── engine.py          # Training, checkpointing, evaluation, and prediction helpers
├── main.py            # Training/evaluation entry point
├── predict.py         # Local or W&B checkpoint inference
├── tracking.py        # Optional W&B integration
├── utils.py           # Seeds, metrics, device selection, and persistence helpers
├── plots.py           # Loss and prediction plots
├── requirements.txt   # Runtime dependencies
├── data/              # Local input CSVs; not required to be committed
├── metrics/            # Generated run metrics and plots
└── models/             # Generated checkpoints and saved configurations
```

## Getting Started

### 1. Clone and create an environment

```bash
git clone https://github.com/AbdallahAhmed149/Hybrid-Quantum-LSTM-Forecasting-.git
cd Hybrid-Quantum-LSTM-Forecasting-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add the dataset

Create a `data/` directory and place the expected files at:

```text
data/DailyDelhiClimateTrain.csv
data/DailyDelhiClimateTest.csv
```

Each file must contain a parseable `date` column, a numeric `meantemp` target column, and any numeric feature columns you want to use. The paths, target, window size, split ratios, and all hyperparameters can be changed in `config.py` or overridden through CLI flags.

### 3. Configure optional Weights & Biases tracking

Training continues without W&B if it is unavailable or not authenticated. To enable tracking:

```bash
wandb login
# or export WANDB_API_KEY="<your-key>"
python main.py
```

Useful configuration fields include `wandb_project`, `wandb_entity`, `wandb_promote`, and `wandb_registry`. Do not commit credentials or secrets.

### 4. Docker-based environment (optional)

This repository currently does not commit a `Dockerfile`. A disposable Python container can still be used for local experiments:

```bash
docker run --rm -it \
  -v "$PWD":/app \
  -w /app \
  python:3.11-slim bash

# Inside the container:
pip install --upgrade pip
pip install -r requirements.txt
python main.py --no-use_wandb
```

For repeatable team or CI deployments, add a pinned `Dockerfile` and lockfile rather than relying on floating dependency lower bounds.

## Training and Inference

### Train the default QLSTM

```bash
python main.py --no-use_wandb
```

By default, the run uses the QLSTM configuration in `Config`, writes the best checkpoint to `models/<run_name>/model.pt`, and stores metrics and plots in `metrics/<run_name>/`.

### Compare model variants

```bash
python main.py --model_type qlstm --no-use_wandb
python main.py --model_type hybrid --no-use_wandb
python main.py --model_type classical --no-use_wandb
```

### Run a short smoke experiment

```bash
python main.py \
  --model_type classical \
  --epochs 5 \
  --hidden_size 16 \
  --no-use_wandb
```

Every scalar field in `Config` is exposed as a CLI option. Boolean fields support both `--field` and `--no-field` forms, for example `--no-use_target_as_input`.

### Run inference with the newest local checkpoint

```bash
python predict.py --model_type qlstm
```

Load a specific local run:

```bash
python predict.py --model_type hybrid --run hybrid_YYYYMMDD_HHMMSS
```

Load a W&B model artifact by alias:

```bash
python predict.py --model_type qlstm --from_wandb --alias production
```

The command prints test-set R²/MAE/RMSE and the forecast date/value, then saves `forecast.json` inside the corresponding metrics directory.

## Outputs and Reproducibility

A training run produces:

```text
metrics/<run_name>/
├── history.csv
├── metrics.json
├── test_predictions.csv
├── loss_curve.png
├── predictions.png
└── forecast.json             # created by predict.py

models/<run_name>/
├── config.json
└── model.pt                   # best-validation weights and metadata

metrics/runs_summary.csv       # append-only comparison table
```

To reproduce a run, record the exact CLI arguments, input CSV versions, Python/dependency versions, and generated `config.json`. The default seed is `42`, but deterministic results can still vary across hardware and library versions.

## Contributing

1. Create a focused branch from `main`.
2. Keep changes scoped and preserve the chronological, leakage-safe evaluation protocol.
3. Add or update tests and documentation for behavior changes.
4. Run at least a short smoke training command before opening a pull request.
5. Include the model type, configuration, dataset revision, and relevant metrics when reporting experiments.

Please open an issue before large architectural changes so the proposed evaluation protocol and compatibility impact can be discussed.

## License

This project is released under the [MIT License](LICENSE).
