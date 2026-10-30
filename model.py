"""Model definitions.

QLSTM             : every LSTM gate is a Variational Quantum Circuit (full quantum LSTM)
HybridQuantumLSTM : classical LSTM -> Linear -> [quantum circuit] -> Linear
ClassicalLSTM     : LSTM -> MLP head (baseline for a fair comparison)
"""
import numpy as np
import pennylane as qml
import torch
from torch import nn

from config import Config


# --------------------------------------------------------------------------
# Quantum part
# --------------------------------------------------------------------------
def build_quantum_layer(n_qubits: int, q_layers: int, reupload: bool, device_name: str):
    """Wrap a parametrised quantum circuit as a torch layer.

    Input : (batch, n_qubits)  angles
    Output: (batch, n_qubits)  <Z_i> of EVERY qubit (the original script only
            measured qubit 0, throwing away 3/4 of the circuit's output).
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        wires = range(n_qubits)
        if reupload:
            for layer in range(weights.shape[0]):
                qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=wires)
        else:
            qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    weight_shapes = {"weights": (q_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


# --------------------------------------------------------------------------
# Quantum LSTM (QLSTM)  -  Chen & Yoo, "Quantum Long Short-Term Memory"
# --------------------------------------------------------------------------
class QLSTMCell(nn.Module):
    """One LSTM step where each of the 4 gates is its own quantum circuit.

        v_t = [h_{t-1}, x_t]
        y_t = Linear(v_t)                       -> n_qubits angles
        f_t = sigmoid(Linear(VQC_f(y_t)))       forget gate
        i_t = sigmoid(Linear(VQC_i(y_t)))       input gate
        g_t = tanh   (Linear(VQC_g(y_t)))       candidate cell
        o_t = sigmoid(Linear(VQC_o(y_t)))       output gate
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        nq = cfg.n_qubits
        self.to_angles = nn.Linear(input_size + cfg.hidden_size, nq)

        def vqc():
            return build_quantum_layer(nq, cfg.q_layers, cfg.reupload, cfg.q_device)

        # 4 INDEPENDENT circuits (separate trainable weights per gate)
        self.vqc_f, self.vqc_i, self.vqc_g, self.vqc_o = vqc(), vqc(), vqc(), vqc()
        self.out_f = nn.Linear(nq, cfg.hidden_size)
        self.out_i = nn.Linear(nq, cfg.hidden_size)
        self.out_g = nn.Linear(nq, cfg.hidden_size)
        self.out_o = nn.Linear(nq, cfg.hidden_size)

    @staticmethod
    def _run(vqc, angles):
        return vqc(angles.cpu()).float().to(angles.device)

    def forward(self, x_t, state):
        h, c = state
        angles = torch.tanh(self.to_angles(torch.cat([h, x_t], dim=1))) * np.pi
        f = torch.sigmoid(self.out_f(self._run(self.vqc_f, angles)))
        i = torch.sigmoid(self.out_i(self._run(self.vqc_i, angles)))
        g = torch.tanh(self.out_g(self._run(self.vqc_g, angles)))
        o = torch.sigmoid(self.out_o(self._run(self.vqc_o, angles)))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class QLSTM(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.cell = QLSTMCell(input_size, cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x): # x: (batch, seq, features)
        b = x.size(0)
        h = x.new_zeros(b, self.hidden_size)
        c = x.new_zeros(b, self.hidden_size)
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], (h, c))
        return self.head(self.dropout(h))


# --------------------------------------------------------------------------
# Hybrid model
# --------------------------------------------------------------------------
class HybridQuantumLSTM(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.pre_quantum = nn.Linear(cfg.hidden_size, cfg.n_qubits)
        self.qlayer = build_quantum_layer(
            cfg.n_qubits, cfg.q_layers, cfg.reupload, cfg.q_device
        )
        # Learned read-out of all qubit measurements (instead of a single <Z_0>)
        self.post_quantum = nn.Linear(cfg.n_qubits, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) # last time-step
        out = torch.tanh(self.pre_quantum(out)) * np.pi  # angles in (-pi, pi)

        # The simulator runs on CPU; move there and come back.
        device = out.device
        out = self.qlayer(out.cpu()).to(device)
        out = out.float()                             
        return self.post_quantum(out)


# --------------------------------------------------------------------------
# Classical baseline
# --------------------------------------------------------------------------
class ClassicalLSTM(nn.Module):
    def __init__(self, input_size: int, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, cfg.n_qubits),
            nn.Tanh(),
            nn.Linear(cfg.n_qubits, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# --------------------------------------------------------------------------
def build_model(cfg: Config, input_size: int) -> nn.Module:
    if cfg.model_type == "qlstm":
        return QLSTM(input_size, cfg)
    if cfg.model_type == "hybrid":
        return HybridQuantumLSTM(input_size, cfg)
    if cfg.model_type == "classical":
        return ClassicalLSTM(input_size, cfg)
    raise ValueError(f"Unknown model_type: {cfg.model_type!r}")