"""
Step 5 — Hybrid Quantum-Classical Model.

Architecture (PennyLane simulator + PyTorch):
  Input (PCA_COMPONENTS = 16)
    → Linear(16 → N_QUBITS) + ReLU        [classical encoding]
    → scale to [-pi, pi] via tanh
    → AngleEmbedding on N_QUBITS qubits    [quantum feature encoding]
    → VQC_LAYERS × (Rot gates + CNOT ring) [variational quantum circuit]
    → <Z> measurement on each qubit         [N_QUBITS expectation values]
    → Linear(N_QUBITS → 2)                 [classical output head]

Design choices (per the project spec): 8 qubits, 2 shallow variational layers,
default.qubit statevector simulator — kept deliberately compact for stability
and reproducibility rather than circuit depth.

Trains for QUANTUM_EPOCHS with per-epoch train/validation logging and keeps the
BEST-VALIDATION-ACCURACY checkpoint.
"""
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from src.config import (N_QUBITS, VQC_LAYERS, QUANTUM_EPOCHS,
                        QUANTUM_LR, QUANTUM_BATCH, PCA_COMPONENTS, MODELS_DIR)
from src.classical import set_seed

# ── Quantum device + circuit ───────────────────────────────────────────────────

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc(inputs: torch.Tensor, weights: torch.Tensor) -> list:
    """Variational quantum circuit. inputs: (N_QUBITS,), weights: (VQC_LAYERS, N_QUBITS, 3)."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    for layer in range(VQC_LAYERS):
        for q in range(N_QUBITS):
            qml.Rot(weights[layer, q, 0], weights[layer, q, 1],
                    weights[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


# ── Hybrid model ───────────────────────────────────────────────────────────────

class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PCA_COMPONENTS, N_QUBITS),
            nn.ReLU(),
        )
        weight_shapes = {"weights": (VQC_LAYERS, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(vqc, weight_shapes)
        self.classifier = nn.Linear(N_QUBITS, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.pi * torch.tanh(x)         # bound angles to [-pi, pi]
        x = self.qlayer(x)
        return self.classifier(x)


# ── Training ───────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            if train_mode:
                optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            n          += len(yb)
    return total_loss / n, correct / n * 100


def train_quantum(X_train, y_train, X_val, y_val, task: str):
    """Train HybridQNN, keep best-validation checkpoint. Returns (model, history, train_time)."""
    set_seed()
    model     = HybridQNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=QUANTUM_LR)
    criterion = nn.CrossEntropyLoss()

    def make_loader(X, y, bs, shuffle):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=bs, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, QUANTUM_BATCH, True)
    val_loader   = make_loader(X_val,   y_val,   64,            False)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state, best_epoch = -1.0, None, 0

    t0 = time.perf_counter()
    for epoch in range(1, QUANTUM_EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader,   criterion, None)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = copy.deepcopy(model.state_dict())
            best_epoch   = epoch

        print(f"[Quantum/{task}] Epoch {epoch:2d}/{QUANTUM_EPOCHS} — "
              f"train_loss {tr_loss:.4f} acc {tr_acc:5.1f}% | "
              f"val_loss {va_loss:.4f} acc {va_acc:5.1f}%"
              + ("  <-- best" if epoch == best_epoch else ""))

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    print(f"[Quantum/{task}] Done in {train_time:.1f}s — "
          f"best val acc {best_val_acc:.1f}% at epoch {best_epoch}")

    torch.save(best_state, os.path.join(MODELS_DIR, f"quantum_{task}.pt"))
    history["best_epoch"]   = best_epoch
    history["best_val_acc"] = best_val_acc
    return model, history, train_time


def predict_quantum(model, X_test):
    """Returns (predictions, P(class=1), inference_time)."""
    X_t    = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=64, shuffle=False)
    model.eval()
    preds, probs = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb)
            preds.append(out.argmax(1).numpy())
            probs.append(torch.softmax(out, dim=1)[:, 1].numpy())
    return np.concatenate(preds), np.concatenate(probs), time.perf_counter() - t0
