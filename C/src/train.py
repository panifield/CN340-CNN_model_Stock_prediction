"""การเทรน การทำนาย และ metrics สำหรับงาน C."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .config import BATCH_SIZE, EPOCHS, LR, PATIENCE, SEED, WEIGHT_DECAY
from .model import SpectroCNNANN

DEVICE = torch.device(os.environ.get("TASKC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
    return torch.sigmoid(logits).cpu().numpy()


def train_one(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray,
              epochs: int = EPOCHS, patience: int = PATIENCE, seed: int = SEED,
              lr: float = LR, weight_decay: float = WEIGHT_DECAY,
              batch_size: int = BATCH_SIZE):
    """เทรนโดย early stop จาก validation loss; ไม่ใช้ test ระหว่างเทรน."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpectroCNNANN(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr, ytr = torch.from_numpy(X_train).to(DEVICE), torch.from_numpy(y_train).to(DEVICE)
    Xva, yva = torch.from_numpy(X_valid).to(DEVICE), torch.from_numpy(y_valid).to(DEVICE)
    history = {"train_loss": [], "valid_loss": [], "valid_acc": []}
    best_loss, best_state, stale_epochs = np.inf, None, 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(Xtr), device=DEVICE)
        for start in range(0, len(permutation), batch_size):
            index = permutation[start:start + batch_size]
            if len(index) < 2:  # BatchNorm ต้องมีอย่างน้อยสองตัวอย่าง
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr[index]), ytr[index])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(Xtr), ytr).item()
            valid_logits = model(Xva)
            valid_loss = loss_fn(valid_logits, yva).item()
            valid_acc = ((valid_logits >= 0) == yva.bool()).float().mean().item()
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        if valid_loss < best_loss - 1e-5:
            best_loss, stale_epochs = valid_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> tuple[dict, np.ndarray]:
    """คำนวณ metrics โดย ROC-AUC เป็น None เมื่อ split มีเพียงคลาสเดียว."""
    predictions = (probabilities >= threshold).astype(int)
    auc = None if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, probabilities))
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix[[down,up] true x predicted]": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    return metrics, predictions
