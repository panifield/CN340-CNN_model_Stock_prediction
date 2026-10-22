"""
baselines.py — ตัวเทียบทั้งหมด: Naive, Mean Return, และโมเดลเดิม MLP / RF / XGBoost (SPEC ข้อ 2.3, 4.3)
===================================================================================================
ไฟล์นี้ทำอะไร: รันโมเดลเดิมบน **29 feature เดิม** ด้วยพารามิเตอร์เดิมที่จูนไว้แล้ว (ไม่จูนใหม่)
ภายใต้ protocol เดียวกับ CNN — fold เดียวกัน วันประเมินชุดเดียวกัน

ทำไมต้องมี: ตัวเลขของรอบก่อนรันคนละ fold กัน จึงเอามาวางข้างกันไม่ได้ · ต้องรันใหม่บนวันเดียวกันเท่านั้น
ถึงจะพูดได้ว่า "CNN ดีกว่า/แย่กว่าโมเดลเดิม"

พารามิเตอร์มาจาก `legacy/config.py` (`CONFIG_TAG = "phase1c-tuned"`) โดยตรง — ห้าม hardcode ซ้ำที่นี่
โมเดลถูกห่อด้วย TransformedTargetRegressor + StandardScaler เหมือน legacy (return มีขนาด ~0.01
ถ้าไม่ scale target ANN จะ converge ช้ามาก) · tree ไม่ต้อง scale feature แต่ ANN ต้อง — Pipeline ของ legacy จัดการให้แล้ว
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from src import data, evaluate, paths

paths.use_legacy()
from models import get_regressors            # noqa: E402  (legacy -- ANN/RF/XGB พร้อมพารามิเตอร์เดิม)

LEGACY_MODEL_NAMES = ["ANN (MLP)", "Random Forest", "XGBoost"]
SIMPLE_NAMES = ["Naive", "Mean Return"]
ALL_NAMES = SIMPLE_NAMES + LEGACY_MODEL_NAMES


def predict_simple(name, y_train, n_eval):
    if name == "Naive":
        return np.zeros(n_eval)                       # ŷ = 0 → ราคาปิดพรุ่งนี้ = ราคาปิดวันนี้
    if name == "Mean Return":
        return np.full(n_eval, float(np.mean(y_train)))
    raise KeyError(name)


def run_fold(ds, fold, *, include_legacy=True):
    """
    รันตัวเทียบทุกตัวบน fold หนึ่งของหุ้นหนึ่ง → {ชื่อโมเดล: ค่าทำนายของช่วงประเมิน}

    *** แถว train ของตัวเทียบ ***
    ใช้ทุกแถวก่อนช่วงประเมินที่ feature 29 ตัวครบ (ไม่ตัด 15% ออก)
    ต่างจาก CNN ที่กัน 15% ท้ายไว้หยุดเทรน เพราะโมเดลเดิมไม่ได้ใช้ early stopping
    (ANN ของ legacy ตั้ง early_stopping=False, max_iter=3000) — ดู DECISIONS ข้อ B11
    """
    ok = ds.ok29
    tr = np.arange(fold.train_stop)[ok[:fold.train_stop]]
    ev = fold.eval_pos
    assert ok[ev].all() and tr[-1] < ev[0]

    X = ds.feat29.to_numpy(dtype=float)
    y = ds.y
    out = {name: predict_simple(name, y[tr], len(ev)) for name in SIMPLE_NAMES}
    if include_legacy:
        for name, model in get_regressors().items():
            wrapped = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
            wrapped.fit(X[tr], y[tr])
            out[name] = np.asarray(wrapped.predict(X[ev]), dtype=float)
    return out, tr, ev


def evaluate_fold(ds, fold, *, include_legacy=True):
    """คืน (list ของแถว metric, list ของ DataFrame คำทำนาย) ของตัวเทียบทุกตัวใน fold นี้"""
    preds, _tr, ev = run_fold(ds, fold, include_legacy=include_legacy)
    rows, frames = [], []
    y_eval, prev = ds.y[ev], ds.prev_close[ev]
    for name, p in preds.items():
        rows.append(evaluate.evaluation_row(name, "legacy-phase1c-tuned", ds.ticker, fold.k, y_eval, p, prev))
        frames.append(evaluate.predictions_frame(ds.dates[ev], ds.ticker, fold.k, y_eval, p, prev,
                                                 model=name, config="legacy-phase1c-tuned"))
    return rows, frames


def run_all(zone_end=None, include_legacy=True, verbose=True):
    """รันตัวเทียบทุกตัว ทุกหุ้น ทุก fold → (DataFrame metric, DataFrame คำทำนาย)"""
    import pandas as pd
    from src.settings import TICKERS, VAL_END
    zone_end = VAL_END if zone_end is None else zone_end
    rows, frames = [], []
    for t in TICKERS:
        ds = data.build_dataset(t, zone_end)
        for fold in data.make_folds(ds.dates):
            r, f = evaluate_fold(ds, fold, include_legacy=include_legacy)
            rows += r
            frames += f
            if verbose:
                for x in r:
                    print(f"  {t:6s} fold {fold.k} {x['model']:14s} relMAE {x['relMAE']:.4f} "
                          f"MAE_baht {x['MAE_baht']:.4f} StdRatio {x['StdRatio']:.3f} IC {x['IC']:+.3f}")
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
