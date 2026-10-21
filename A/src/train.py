"""การเทรนและประเมินผลโมเดล"""
import os

import numpy as np
import torch
import torch.nn as nn

from .config import BATCH_SIZE, EPOCHS, LR, PATIENCE, WEIGHT_DECAY
from .model import SpectroCNNANN


# ใช้ GPU ถ้ามี (ตั้ง env TASKA_DEVICE=cpu เพื่อบังคับใช้ CPU)
DEVICE = torch.device(os.environ.get("TASKA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))


def _t(a):
    return None if a is None else torch.from_numpy(a).to(DEVICE)


def predict_proba(model, X, E=None):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(_t(X), _t(E))).cpu().numpy()


def train_one(Xtr, ytr, Xva, yva, seed, Etr=None, Eva=None, arch="current",
              noise_std=0.0, init_cnn=None, freeze_cnn=False, model_kwargs=None,
              lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE):
    """noise_std > 0 : augmentation ใส่ Gaussian noise ให้ input ตอนเทรน
    init_cnn       : state_dict ของ CNN ที่ pretrain มาแล้ว (transfer learning)
    freeze_cnn     : ไม่อัปเดต CNN เทรนแค่ ANN
    lr, weight_decay, batch_size : ค่า default ตาม config (ปรับได้สำหรับ hyperparameter search)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_extra = 0 if Etr is None else Etr.shape[1]
    model = SpectroCNNANN(Xtr.shape[1], n_extra=n_extra, arch=arch, **(model_kwargs or {})).to(DEVICE)
    if init_cnn is not None:
        model.cnn.load_state_dict(init_cnn)
    if freeze_cnn:
        for prm in model.cnn.parameters():
            prm.requires_grad = False
    opt = torch.optim.Adam([prm for prm in model.parameters() if prm.requires_grad],
                           lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t, Etr_t = _t(Xtr), _t(ytr), _t(Etr)
    Xva_t, yva_t, Eva_t = _t(Xva), _t(yva), _t(Eva)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, wait = np.inf, None, 0
    for _ in range(EPOCHS):
        model.train()
        if freeze_cnn:
            model.cnn.eval()  # BatchNorm ของ CNN ที่ freeze ต้องไม่อัปเดตสถิติ
        perm = torch.randperm(len(Xtr_t), device=DEVICE)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:  # BatchNorm ต้องการ batch >= 2
                continue
            opt.zero_grad()
            e = None if Etr_t is None else Etr_t[idx]
            xb = Xtr_t[idx]
            if noise_std > 0:
                xb = xb + noise_std * torch.randn_like(xb)
            loss = loss_fn(model(xb, e), ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(Xtr_t, Etr_t), ytr_t).item()
            va_logit = model(Xva_t, Eva_t)
            va_loss = loss_fn(va_logit, yva_t).item()
            va_acc = ((va_logit > 0).float() == yva_t).float().mean().item()
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)

        if va_loss < best_loss - 1e-4:
            best_loss, wait = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, hist
