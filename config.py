"""Central configuration for the Hybrid Quantum-Classical forecasting project.

Every hyper-parameter lives here so that (1) main.py stays clean and
(2) the whole config can be dumped to metrics/ and Weights & Biases,
which makes every run reproducible.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # ------------------------------------------------------------ paths
    train_csv: Path = ROOT / "data" / "DailyDelhiClimateTrain.csv"
    test_csv: Path = ROOT / "data" / "DailyDelhiClimateTest.csv"
    metrics_dir: Path = ROOT / "metrics"
    models_dir: Path = ROOT / "models"

    # ------------------------------------------------------------- data
    target: str = "meantemp"
    seq_length: int = 10 # days of history fed to the model
    train_ratio: float = 0.8    
    val_ratio: float = 0.1
    use_target_as_input: bool = True

    # ------------------------------------------------------------ model
    model_type: str = "qlstm"     # "qlstm" (all gates quantum) | "hybrid" (LSTM + quantum layer) | "classical" (baseline)
    hidden_size: int = 32
    lstm_layers: int = 1
    dropout: float = 0.1
    n_qubits: int = 4
    q_layers: int = 3
    reupload: bool = True # data re-uploading 
    q_device: str = "default.qubit"

    # --------------------------------------------------------- training
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 15 # early stopping on validation loss
    seed: int = 42

    # ---------------------------------------------------- Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "hybrid-quantum-classical-ml"
    wandb_run_name: str = ""      
    wandb_entity: str = ""        
    wandb_promote: bool = False   # True -> tag this model version with the "production" alias
    wandb_registry: str = ""  

    def to_dict(self) -> dict:
        """JSON/W&B friendly dict (Paths -> str)."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}