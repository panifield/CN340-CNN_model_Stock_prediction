"""
train.py — loop เทรน PyTorch แบบ regression + การ scale target (SPEC ข้อ 5.1, 4.2)
=================================================================================
ไฟล์นี้ทำอะไร
    1. `fit()`      : เทรนโมเดลหนึ่งตัวพร้อม early stopping บนชุด val ที่แยกตามเวลา (15% ท้ายของ train)
    2. `predict()`  : ทำนายแบบแบ่ง chunk (ประหยัดหน่วยความจำ)
    3. `run_fold()` : เทรน + ทำนายช่วงประเมินของ fold หนึ่ง เฉลี่ยหลาย seed ตามที่สเปกกำหนด

ทำไมต้องมี — จุดที่พลาดง่ายและถูกบังคับไว้ที่นี่
    * early stopping ใช้ **เฉพาะ val ที่เป็น 15% ท้ายของ train** ห้ามแตะช่วงประเมินเด็ดขาด (ข้อ 4.2, 11)
    * y ถูก standardize ด้วย **สถิติของ train** เท่านั้น แล้วแปลงกลับตอนทำนาย
      (ไม่งั้น δ ของ Huber = 1 จะใหญ่กว่า return รายวัน ~100 เท่า → Huber กลายเป็น MSE ทั้งหมด)
    * CPU · 1 thread · deterministic → รันซ้ำได้ผลเดิมทุกบิต
      (ported from task_b/phase2/train_torch.py, branch task_b)
"""

import numpy as np
import torch
import torch.nn.functional as F

from src import models
from src.settings import SEARCH_SEEDS

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
# ค่า subnormal ทำให้ CPU ช้าลงหลายเท่าโดยไม่เปลี่ยนผลอย่างมีนัยสำคัญ -- ปัดเป็น 0
torch.set_flush_denormal(True)


def loss_fn(cfg):
    """ชื่อ loss ใน config → ฟังก์ชัน (pred, target) → scalar · คิดบน y ที่ standardize แล้ว"""
    if cfg.loss == "mae":
        return F.l1_loss
    if cfg.loss == "mse":
        return F.mse_loss
    if cfg.loss == "huber":
        delta = cfg.huber_delta
        return lambda p, t: F.huber_loss(p, t, delta=delta)
    raise KeyError(cfg.loss)


class TargetScaler:
    """
    standardize y ด้วยสถิติของ train เท่านั้น (ไม่จัดกึ่งกลางก็ได้ แต่ที่นี่จัด เพราะ Huber สนใจระยะจากศูนย์)
    y_scaled = (y − mean) / std   ·   แปลงกลับ : y = y_scaled · std + mean
    """

    def fit(self, y):
        self.mean = float(np.mean(y))
        self.std = float(np.std(y))
        if not self.std > 0:
            raise ValueError("std ของ y ใน train เป็น 0")
        return self

    def transform(self, y):
        return (np.asarray(y, dtype=np.float32) - self.mean) / self.std

    def inverse(self, z):
        return np.asarray(z, dtype=np.float64) * self.std + self.mean


def _tensors(arrays):
    return [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in arrays]


def fit(cfg, inputs_fit, y_fit, inputs_val, y_val, *, n_extra=0, seed=0, min_delta=1e-4, factory=None):
    """
    เทรนโมเดลหนึ่งตัว คืน (model ที่โหลด state ดีที่สุดแล้ว, history)

    inputs_* : list ของ ndarray ที่เรียงตรงกับ y (ภาพ [, extra feature])
    y_*      : ndarray ที่ **standardize แล้ว**
    early stopping : หยุดเมื่อ val loss ไม่ดีขึ้นเกิน min_delta ติดกัน `cfg.patience` epoch
    factory : (ไม่บังคับ) callable ไม่มีอาร์กิวเมนต์ → โมเดลที่ต้องการเอง (ใช้กับ ANN ขั้นที่สองของสองขั้น) · None = models.build(cfg)
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = factory() if factory is not None else models.build(cfg, n_extra)
    crit = loss_fn(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xf, Xv = _tensors(inputs_fit), _tensors(inputs_val)
    yf = torch.from_numpy(np.asarray(y_fit, dtype=np.float32))
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
    n = len(yf)

    best, best_state, wait, best_epoch = np.inf, None, 0, 0
    hist = {"train_loss": [], "val_loss": []}
    for _ in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            if len(idx) < 2:                               # BatchNorm ต้องการอย่างน้อย 2 แถว
                continue
            xb = [x[idx] for x in Xf]
            if cfg.noise_std > 0 and cfg.ann_input != "hybrid_nocnn":      # augmentation เฉพาะ "ภาพ" (input ตัวแรก) — control ไม่มีภาพ (B25)
                xb[0] = xb[0] + cfg.noise_std * torch.randn(xb[0].shape, generator=gen)
            opt.zero_grad()
            crit(model(*xb), yf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hist["train_loss"].append(float(crit(model(*Xf), yf)))
            v = float(crit(model(*Xv), yv))
        hist["val_loss"].append(v)
        if v < best - min_delta:
            best, wait = v, 0
            best_epoch = len(hist["val_loss"])                  # นับจาก 1
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist["best_val"] = best
    hist["best_epoch"] = best_epoch                  # epoch ที่ val ดีที่สุด (state ที่ถูกเก็บไว้ใช้ทำนาย)
    hist["epochs"] = len(hist["val_loss"])           # epoch ที่หยุดจริง
    hist["stopped_early"] = hist["epochs"] < cfg.max_epochs
    return model, hist


def predict(model, inputs, chunk=1024):
    model.eval()
    xs = _tensors(inputs)
    out = []
    with torch.no_grad():
        for i in range(0, len(xs[0]), chunk):
            out.append(model(*[x[i:i + chunk] for x in xs]).numpy())
    return np.concatenate(out).astype(np.float64)


def _inputs(cfg, part, extra):
    """ประกอบ input list ตาม ann_input ของ config"""
    if cfg.ann_input == "spec":
        return [part]
    if cfg.ann_input == "hybrid_nocnn":
        return [extra]                                  # ไม่มี CNN → รับ 29 feature อย่างเดียว
    return [part, extra]


def run_fold(cfg, prepared, *, seeds=SEARCH_SEEDS, extra=None, y_override=None, return_hist=False,
             return_seed_preds=False, return_val_pred=False):
    """
    เทรนและทำนายช่วงประเมินของ fold หนึ่ง (หุ้นเดียว) — เฉลี่ยคำทำนายข้าม seed

    prepared   : ผลของ src.data.fold_inputs
    extra      : dict {"fit","val","eval"} ของ 29 feature ที่ standardize แล้ว (เฉพาะ hybrid)
    y_override : dict {"fit","val"} ถ้าต้องการเปลี่ยน target (ใช้กับ control ในข้อ 5.0)
    คืน pred_eval (หน่วย y จริง · เฉลี่ยข้าม seed) · ถ้าขอเพิ่มจะคืน tuple ตามลำดับ:
        return_hist=True       → + [history ของทุก seed]
        return_seed_preds=True → + [คำทำนายของแต่ละ seed เดี่ยว ๆ (หน่วย y จริง)] — ไว้วัด noise ข้าม seed (DECISIONS B17)
        return_val_pred=True   → + (คำทำนายบนส่วน early-stop เฉลี่ยข้าม seed, y ของส่วนนั้น) — ไว้เลือก c ของ shrinkage (B18)
    """
    y_fit = prepared["y_fit"] if y_override is None else y_override["fit"]
    y_val = prepared["y_val"] if y_override is None else y_override["val"]
    scaler = TargetScaler().fit(y_fit)
    n_extra = 0 if extra is None else extra["fit"].shape[1]

    preds, hists, val_preds = [], [], []
    for s in seeds:
        model, hist = fit(cfg,
                          _inputs(cfg, prepared["X_fit"], None if extra is None else extra["fit"]),
                          scaler.transform(y_fit),
                          _inputs(cfg, prepared["X_val"], None if extra is None else extra["val"]),
                          scaler.transform(y_val),
                          n_extra=n_extra, seed=s)
        preds.append(predict(model, _inputs(cfg, prepared["X_eval"], None if extra is None else extra["eval"])))
        if return_val_pred:
            val_preds.append(predict(model, _inputs(cfg, prepared["X_val"], None if extra is None else extra["val"])))
        hists.append(hist)
    pred = scaler.inverse(np.mean(preds, axis=0))
    out = [pred]
    if return_hist:
        out.append(hists)
    if return_seed_preds:
        out.append([scaler.inverse(p) for p in preds])
    if return_val_pred:
        out.append((scaler.inverse(np.mean(val_preds, axis=0)), np.asarray(y_val, dtype=float)))
    return out[0] if len(out) == 1 else tuple(out)


def standardized_extra(ds, prepared):
    """
    29 feature เดิมของ fold นี้ standardize ด้วยสถิติของแถว fit เท่านั้น (ใช้กับ hybrid)
    คืน dict {"fit","val","eval"} ของ ndarray float32
    """
    F29 = ds.feat29.to_numpy(dtype=np.float64)
    fit = F29[prepared["pos_fit"]]
    mean, std = fit.mean(axis=0), np.maximum(fit.std(axis=0), 1e-8)
    return {k: ((F29[prepared[f"pos_{k}"]] - mean) / std).astype(np.float32) for k in ("fit", "val", "eval")}
