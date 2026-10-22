"""
sign_control.py — ทดสอบว่า "ภาพ" เก็บเครื่องหมายของ return ไว้หรือไม่ (DECISIONS B24)
=====================================================================================
ไฟล์นี้ทำอะไร
    target = ret_close ของ "วันสุดท้ายในหน้าต่าง" (มีเครื่องหมาย) = ln(C_{t−1} / C_{t−2})
    ซึ่งเป็นฟังก์ชันของ input ล้วน (ไม่ใช้ข้อมูลวัน t เลย) จึงเรียนรู้ได้ถ้าภาพเก็บเครื่องหมายไว้

ทำไมต้องมี — ทฤษฎีที่ต้องการพิสูจน์
    log-power = log1p(|Z|²/m) ใช้ขนาดของ STFT อย่างเดียว และ |STFT(x)| = |STFT(−x)| ทุกจุด
    → ภาพของหน้าต่าง x กับ −x เหมือนกันทุกพิกเซล → CNN บอกไม่ได้ว่า return ล่าสุดเป็นบวกหรือลบ → R² ≈ 0
    complex เก็บ Re/Im (รวม phase) จึงควรเรียนเครื่องหมายได้ ถ้า R² ต่ำด้วย แปลว่าข้อมูลหายระหว่างทางที่ไหนสักแห่ง
    → `probe_r2` ใช้ linear probe (ridge) ตรวจที่ 3 จุด: หลัง STFT · หลัง arcsinh/log1p · หลัง normalize (ที่ CNN เห็นจริง)
    แยกทดสอบ "ทุก frame" กับ "frame สุดท้ายอย่างเดียว"

*** ไฟล์นี้รายงานเท่านั้น ไม่แก้ pipeline (ผู้ใช้สั่ง) ***
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import controls, data, train
from src.configs import BASE, variant
from src.settings import SEARCH_SEEDS, VAL_END
from src.spectrogram import raw_channels

STAGES = ("stft", "compressed", "normalized")          # หลัง STFT · หลัง arcsinh/log1p · หลัง normalize
FRAME_SETS = ("all", "last")                            # ทุก frame · frame สุดท้ายอย่างเดียว
PROBE_ALPHAS = np.logspace(-2, 5, 15)

# 4 config ของ sign control (DECISIONS B24)
SIGN_CONFIGS = [
    ("logpower_w128", BASE),
    ("complex_w128", variant("complex_w128", spec_type="complex", scale_mode="fixed")),
    ("logpower_w64", variant("logpower_w64", window=64)),
    ("complex_w64", variant("complex_w64", spec_type="complex", scale_mode="fixed", window=64)),
]


def last_return_target(ds, positions):
    """
    ret_close ของวันสุดท้ายในหน้าต่างของแถวที่ตำแหน่ง `positions` = ln(C_{p−1}/C_{p−2})
    หน้าต่างของวัน p คือแถว p−window … p−1 → แถวสุดท้ายคือ p−1 · ใช้ข้อมูลถึงวัน p−1 เท่านั้น
    """
    ret = raw_channels(ds.prices)["ret_close"].to_numpy(dtype=float)
    out = ret[np.asarray(positions, dtype=int) - 1]
    if not np.isfinite(out).all():
        raise ValueError("target ของ sign control มีค่าไม่ finite")
    return out


def _r2_corr(y, pred):
    return controls.r2(y, pred), float(np.corrcoef(pred, y)[0, 1]) if np.ptp(pred) > 0 else float("nan")


def probe_r2(ds, cfg, fold):
    """
    linear probe : ridge (ปรับ α ด้วย LOO บนแถว train) ทำนาย target จากภาพ ณ แต่ละขั้น
    คืน list ของ dict {stage, frames, R2, corr, n_features}
      stft       ค่าดิบหลัง STFT (Re/Im หรือ |Z|²)
      compressed หลัง arcsinh (complex) / log1p(·/m) (log-power) — ยังไม่ normalize
      normalized หลัง standardize ทีละแถบความถี่ = ภาพที่ CNN เห็นจริง
    """
    prep = data.fold_inputs(ds, cfg, fold)
    tf = prep["transform"]
    pos_tr = np.concatenate([prep["pos_fit"], prep["pos_val"]])
    pos_ev = prep["pos_eval"]
    y_tr, y_ev = last_return_target(ds, pos_tr), last_return_target(ds, pos_ev)
    raw_tr, _ = ds.raw_images(cfg, pos_tr)
    raw_ev, _ = ds.raw_images(cfg, pos_ev)
    views = {"stft": (raw_tr, raw_ev), "compressed": (tf.compress(raw_tr), tf.compress(raw_ev)),
             "normalized": (tf.transform(raw_tr), tf.transform(raw_ev))}
    out = []
    for stage in STAGES:
        a, b = views[stage]
        for frames in FRAME_SETS:
            fa = a.reshape(len(a), -1) if frames == "all" else a[..., -1].reshape(len(a), -1)
            fb = b.reshape(len(b), -1) if frames == "all" else b[..., -1].reshape(len(b), -1)
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=PROBE_ALPHAS)).fit(fa, y_tr)
            r2, corr = _r2_corr(y_ev, model.predict(fb))
            out.append({"stage": stage, "frames": frames, "R2": r2, "corr": corr, "n_features": fa.shape[1]})
    return out


def sign_unit(args):
    """
    หนึ่ง unit : (ชื่อ config, config, หุ้น, fold) → (แถวผลของ CNN, list ผลของ probe)
    เทรน CNN ให้ทำนาย target (มีเครื่องหมาย) แล้วประเมินบนช่วงประเมิน · 3 seed เฉลี่ย
    """
    label, cfg, ticker, fold_k, seeds, zone_end = args
    t0 = time.time()
    ds = data.build_dataset(ticker, zone_end)
    fold = next(f for f in data.make_folds(ds.dates) if f.k == fold_k)
    prep = data.fold_inputs(ds, cfg, fold)
    y_over = {k: last_return_target(ds, prep[f"pos_{k}"]) for k in ("fit", "val")}
    y_eval = last_return_target(ds, prep["pos_eval"])
    pred = train.run_fold(cfg, prep, seeds=seeds, y_override=y_over)
    r2, corr = _r2_corr(y_eval, pred)
    row = {"config": label, "ticker": ticker, "fold": fold_k, "n": len(y_eval), "R2": r2, "corr": corr,
           "pred_std_ratio": float(np.std(pred) / np.std(y_eval)), "seconds": round(time.time() - t0, 1)}
    probes = [{"config": label, "ticker": ticker, "fold": fold_k, **p} for p in probe_r2(ds, cfg, fold)]
    return row, probes


def summarize_expectation(rows):
    """
    ตัดสินคาดการณ์ของ B24 ตาม "วิธีอ่านผล" ที่ประกาศไว้ก่อนรัน
      log-power R² ≈ 0 : R² เฉลี่ย 6 evaluation ≤ 0.05 และ |corr เฉลี่ย| < 0.10
      complex R² ต่ำ   : R² เฉลี่ย < 0.3
    """
    df = pd.DataFrame(rows)
    out = []
    for label, g in df.groupby("config", sort=False):
        r2, corr = float(g["R2"].mean()), float(g["corr"].mean())
        kind = "logpower" if label.startswith("logpower") else "complex"
        if kind == "logpower":
            ok = bool(r2 <= 0.05 and abs(corr) < 0.10)
            verdict = "ยืนยันทฤษฎี (R² ≈ 0)" if ok else "**ขัดกับทฤษฎี** (R² > 0.05 หรือ |corr| ≥ 0.10)"
        else:
            low = bool(r2 < 0.3)
            verdict = "**R² ต่ำ (< 0.3) → ต้องตรวจว่าข้อมูลหายตรงไหน**" if low else "เรียนเครื่องหมายได้ (R² ≥ 0.3)"
        out.append({"config": label, "ชนิด": kind, "R² เฉลี่ย": r2, "corr เฉลี่ย": corr, "สรุป": verdict})
    return out
