"""
twostage.py — วิธีเทรนแบบ "สองขั้น" พร้อม cross-fitting (SPEC ข้อ 5.4 · DECISIONS B32)
=======================================================================================
ไฟล์นี้ทำอะไร
    ขั้น 1  เทรน CNN (spectrogram อย่างเดียว) ทำนาย y แบบ cross-fitting:
            แบ่งแถว train เป็น K = 4 ช่วงต่อเนื่อง · CNN_j เทรนบน 3 ช่วงที่เหลือ แล้วทำนายช่วง j = **คำทำนาย out-of-fold (OOF)**
            แถวประเมินได้ค่าเฉลี่ยของ CNN ทั้ง 4 ตัว (ไม่เทรนตัวที่ 5)
    ขั้น 2  เทรน ANN แยกบน feature OOF (สเกลเดียว) — ถ้าเป็น hybrid ต่อด้วย 29 feature เดิม + linear skip

ทำไมต้องใช้ OOF
    ถ้าใช้ feature ที่ CNN สกัดจากแถวที่มันเพิ่งเทรนมา ANN จะเชื่อ feature เกินจริง (CNN จำ y ของแถวนั้นไปแล้ว)
    OOF บังคับให้ feature ของแต่ละแถวมาจาก CNN ที่ไม่เคยเห็น y ของแถวนั้น

ทำไม "CNN feature" เป็น scalar ไม่ใช่เวกเตอร์ 16 ตัว (B32)
    เวกเตอร์ feature จาก CNN คนละตัว (4 ตัวของ cross-fitting) อยู่คนละ "ระบบพิกัด" ใช้ร่วมกันหรือเฉลี่ยกันไม่ได้
    คำทำนาย scalar หน่วย return เทียบกันได้ข้าม CNN (วิธีเดียวกับ task_b/phase2/crossfit.py : cnn_oof)
"""

from dataclasses import replace

import numpy as np

from src import data, models, train
from src.settings import EARLY_STOP_FRAC, SEARCH_SEEDS

K_BLOCKS = 4


def block_slices(n, k=K_BLOCKS):
    """แบ่งดัชนี 0..n−1 เป็น k ช่วงต่อเนื่องตามเวลา"""
    return np.array_split(np.arange(n), k)


def oof_features(cfg_spec, X_tr, y_tr, X_ev, seed, k=K_BLOCKS):
    """
    cross-fitting ของขั้น 1 · คืน (oof (n,), eval_feature (n_eval,)) ในหน่วย y จริง
    CNN_j เทรนบนแถวนอกช่วง j (early stopping ด้วย 15% ท้ายตามเวลาของแถวเหล่านั้น) → ไม่เคยเห็น y ของช่วง j เลย
    """
    n = len(y_tr)
    oof = np.full(n, np.nan)
    ev_preds = []
    for blk in block_slices(n, k):
        train_idx = np.setdiff1d(np.arange(n), blk)                    # เรียงตามเวลาอยู่แล้ว
        fit_m, val_m = data.early_stop_split(len(train_idx), EARLY_STOP_FRAC)
        fi, vi = train_idx[fit_m], train_idx[val_m]
        sc = train.TargetScaler().fit(y_tr[fi])
        model, _ = train.fit(cfg_spec, [X_tr[fi]], sc.transform(y_tr[fi]), [X_tr[vi]], sc.transform(y_tr[vi]), seed=seed)
        oof[blk] = sc.inverse(train.predict(model, [X_tr[blk]]))
        ev_preds.append(sc.inverse(train.predict(model, [X_ev])))
    if not np.isfinite(oof).all():
        raise AssertionError("OOF feature มีค่าไม่ finite")
    return oof, np.mean(ev_preds, axis=0)


def run_fold_two_stage(cfg, prepared, *, seeds=SEARCH_SEEDS, extra=None, y_override=None):
    """
    เทรนและทำนายช่วงประเมินของ fold หนึ่งแบบสองขั้น · อินเทอร์เฟซเหมือน `train.run_fold` ที่ขอ hist/seed_preds/val_pred ครบ
    คืน (pred_eval, [hist ของขั้น ANN ต่อ seed], [pred ต่อ seed], (val_pred, y_val ของส่วน early-stop ของขั้น ANN))

    prepared : ผลของ data.fold_inputs · extra : dict {"fit","val","eval"} ของ 29 feature (เฉพาะ hybrid) · ANN ขั้น 2 ไม่มี noise augmentation
    """
    cfg_spec = replace(cfg, ann_input="spec", train_mode="e2e")        # ขั้น 1 = CNN spectrogram อย่างเดียวเสมอ
    cfg_ann = replace(cfg, noise_std=0.0)                              # ขั้น 2 input ไม่ใช่ภาพ → ไม่ใส่ noise
    y_fit = prepared["y_fit"] if y_override is None else y_override["fit"]
    y_val = prepared["y_val"] if y_override is None else y_override["val"]
    X_tr = np.concatenate([prepared["X_fit"], prepared["X_val"]])
    y_tr = np.concatenate([y_fit, y_val]).astype(float)
    e_tr = None if extra is None else np.concatenate([extra["fit"], extra["val"]])
    fit_m, val_m = data.early_stop_split(len(y_tr), EARLY_STOP_FRAC)   # 15% ท้ายตามเวลาของ train ทั้งหมด

    preds, hists, val_preds = [], [], []
    for s in seeds:
        oof, ev = oof_features(cfg_spec, X_tr, y_tr, prepared["X_eval"], s)
        mu, sd = float(oof.mean()), max(float(oof.std()), 1e-12)       # standardize ด้วยสถิติของแถว train (OOF) เท่านั้น
        f_tr = ((oof - mu) / sd)[:, None].astype(np.float32)
        f_ev = ((ev - mu) / sd)[:, None].astype(np.float32)
        if e_tr is None:
            x_tr, x_ev = f_tr, f_ev
        else:
            x_tr, x_ev = np.hstack([f_tr, e_tr]), np.hstack([f_ev, extra["eval"]])
        n_in = x_tr.shape[1]
        sc = train.TargetScaler().fit(y_tr[fit_m])
        model, hist = train.fit(cfg_ann, [x_tr[fit_m]], sc.transform(y_tr[fit_m]), [x_tr[val_m]], sc.transform(y_tr[val_m]),
                                seed=s, factory=lambda: models.FeatureANN(n_in, cfg.ann_hidden, cfg.dropout, skip=n_in > 1))
        preds.append(sc.inverse(train.predict(model, [x_ev])))
        val_preds.append(sc.inverse(train.predict(model, [x_tr[val_m]])))
        hists.append(hist)
    return np.mean(preds, axis=0), hists, preds, (np.mean(val_preds, axis=0), y_tr[val_m])
