"""
Step 4 — Classical Baselines.

1. SVM           — RBF kernel, on PCA features
2. Random Forest — on PCA features
3. EEGNet        — depthwise-separable CNN on raw EEG time-series

EEGNet trains for EEGNET_EPOCHS with per-epoch train/validation logging and
keeps the BEST-VALIDATION-ACCURACY checkpoint (not the final epoch).

All predict_* functions return (predictions, positive-class scores, time) so
ROC-AUC can be computed downstream.
"""
import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.config import (SVM_C, SVM_GAMMA, RF_TREES, RANDOM_SEED,
                        EEGNET_EPOCHS, EEGNET_LR, EEGNET_BATCH,
                        N_EEG_CHANNELS, MODELS_DIR)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── SVM ────────────────────────────────────────────────────────────────────────

def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    clf = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, random_state=RANDOM_SEED)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"[SVM] Trained in {train_time:.2f}s")
    return clf, train_time


def predict_svm(clf, X_test: np.ndarray):
    t0 = time.perf_counter()
    preds  = clf.predict(X_test)
    scores = clf.decision_function(X_test)      # signed distance → ROC-AUC score
    return preds, scores, time.perf_counter() - t0


# ── Random Forest ──────────────────────────────────────────────────────────────

def train_rf(X_train: np.ndarray, y_train: np.ndarray):
    clf = RandomForestClassifier(n_estimators=RF_TREES,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"[RF] Trained in {train_time:.2f}s")
    return clf, train_time


def predict_rf(clf, X_test: np.ndarray):
    t0 = time.perf_counter()
    preds  = clf.predict(X_test)
    scores = clf.predict_proba(X_test)[:, 1]    # P(class = 1) → ROC-AUC score
    return preds, scores, time.perf_counter() - t0


# ── EEGNet (PyTorch) ───────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """Compact depthwise-separable CNN for EEG (Lawhern et al., 2018)."""
    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2):
        super().__init__()
        F2 = F1 * D
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )
        reduced = (n_samples // 4) // 8
        self.fc = nn.Linear(F2 * reduced, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        return self.fc(x)


def _loader(X_raw, y, batch_size, shuffle, drop_last=False):
    # drop_last avoids a size-1 final batch, which would break BatchNorm in
    # training mode ("Expected more than 1 value per channel").
    X_t = torch.tensor(X_raw[:, np.newaxis, :, :], dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)


def _run_epoch(model, loader, criterion, optimizer=None):
    """One pass. If optimizer is given → train; else → evaluate. Returns (loss, acc)."""
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


def train_eegnet(X_train, y_train, X_val, y_val, task: str):
    """Train EEGNet, keep best-validation-accuracy checkpoint. Returns (model, history, train_time)."""
    set_seed()
    n_samples = X_train.shape[2]
    model     = EEGNet(n_channels=N_EEG_CHANNELS, n_samples=n_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=EEGNET_LR)
    criterion = nn.CrossEntropyLoss()

    train_loader = _loader(X_train, y_train, EEGNET_BATCH, shuffle=True,
                           drop_last=True)
    val_loader   = _loader(X_val,   y_val,   64,           shuffle=False)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state, best_epoch = -1.0, None, 0

    t0 = time.perf_counter()
    for epoch in range(1, EEGNET_EPOCHS + 1):
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

        print(f"[EEGNet/{task}] Epoch {epoch:2d}/{EEGNET_EPOCHS} — "
              f"train_loss {tr_loss:.4f} acc {tr_acc:5.1f}% | "
              f"val_loss {va_loss:.4f} acc {va_acc:5.1f}%"
              + ("  <-- best" if epoch == best_epoch else ""))

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)            # restore best checkpoint
    print(f"[EEGNet/{task}] Done in {train_time:.1f}s — "
          f"best val acc {best_val_acc:.1f}% at epoch {best_epoch}")

    torch.save(best_state, os.path.join(MODELS_DIR, f"eegnet_{task}.pt"))
    history["best_epoch"]   = best_epoch
    history["best_val_acc"] = best_val_acc
    return model, history, train_time


def predict_eegnet(model, X_test):
    """Returns (predictions, P(class=1), inference_time)."""
    loader = _loader(X_test, np.zeros(len(X_test), dtype=int),
                     batch_size=64, shuffle=False)
    model.eval()
    preds, probs = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb)
            preds.append(out.argmax(1).numpy())
            probs.append(torch.softmax(out, dim=1)[:, 1].numpy())
    return np.concatenate(preds), np.concatenate(probs), time.perf_counter() - t0
