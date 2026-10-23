"""
evaluate.py — ตัววัดทั้งหมด + Diebold–Mariano + การรวมคะแนนข้าม evaluation (SPEC ข้อ 4.2, 6)
==========================================================================================
ไฟล์นี้ทำอะไร
    1. `metrics(y, pred, prev_close)` : ตัววัดหนึ่ง evaluation (หุ้นหนึ่งตัว × fold หนึ่ง)
    2. `score_config(evals)`          : รวม 6 evaluation (2 หุ้น × 3 fold) เป็นคะแนนเดียว + guard
    3. `dm_test(a, b)`                : Diebold–Mariano เทียบสองโมเดลบนวันเดียวกัน

ตัววัด (SPEC ข้อ 6 · DECISIONS ข้อ B5 และ B16)
    รายงานหลัก  MAE_baht, RMSE_baht, **relMAE** (= MAE_return / MAE_return ของ Naive) และ relMAE_baht — รายงานทุกตาราง
    รอง        MAE_return, R2_OOS, direction accuracy (ไม่นับวันที่ y = 0), StdRatio (เป็นธง ไม่ตัดสิทธิ์), Bias
    **เกณฑ์เลือก config (หลัง B16) = IC (Pearson) เฉลี่ยของ 6 evaluation** · Spearman รายงานคู่กัน
        IC = corr(ŷ, y) — วัด "สัญญาณ" โดยไม่ขึ้นกับขนาดคำทำนาย จึงไม่ให้รางวัลกับการทำนายใกล้ 0
        (89% ของช่องว่างระหว่างโมเดลกับ Naive มาจากวัน y = 0 ซึ่ง relMAE จึงลำเอียงเข้าหา Naive)

ทำไมต้องมี: ตัววัดถูกประกาศก่อนเห็นผล (ข้อ 11 ห้ามเปลี่ยนทีหลัง) · คิดที่เดียวทั้งโครงการ
Naive ของโหมด next_day คือ ŷ = 0 (พรุ่งนี้ราคาเท่าวันนี้)
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, spearmanr

from src.settings import GUARD_MIN_STDRATIO, N_FOLDS

CONST_PRED_RATIO = 1e-8      # B39: std(ŷ)/std(y) ต่ำกว่านี้ (หรือ ptp(ŷ) = 0) = "คำทำนายคงที่" → IC นิยามไม่ได้

METRIC_COLS = ["MAE_baht", "RMSE_baht", "relMAE", "relMAE_baht", "MAE_return", "RMSE_return",
               "R2_OOS", "DirAcc", "IC", "IC_spearman", "StdRatio", "Bias", "naive_MAE_return", "n"]


def naive_prediction(y):
    """Naive (random walk) : ทำนายว่า return = 0 → ราคาปิดพรุ่งนี้ = ราคาปิดวันนี้"""
    return np.zeros(len(y), dtype=float)


def mean_return_prediction(y_train, n_eval):
    """Mean Return : ทำนายด้วยค่าเฉลี่ย return ของ train"""
    return np.full(n_eval, float(np.mean(y_train)))


def is_constant_prediction(y, pred):
    """B39: คำทำนายคงที่ = ptp(ŷ) = 0 หรือ std(ŷ)/std(y) < CONST_PRED_RATIO (ต่างกันแค่เศษปัดเศษก็นับ)"""
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.ptp(pred) == 0:
        return True
    sy = float(np.std(y))
    return bool(sy > 0 and float(np.std(pred)) / sy < CONST_PRED_RATIO)


def metrics(y, pred, prev_close, const_ic_zero=False):
    """
    หนึ่ง evaluation → dict ของตัววัดทุกตัว
    y, pred เป็น "return" · prev_close ใช้แปลงเป็นบาท : ราคา = prev_close · (1 + return)

    const_ic_zero (เฉพาะ Stage 6 · DECISIONS B39 · ค่าเริ่มต้นปิด = พฤติกรรมเดิมทุกประการ):
        ถ้าคำทำนายคงที่ (is_constant_prediction) → IC Pearson = IC Spearman := 0 แทน NaN · เพิ่มคอลัมน์ `collapsed`
        (ไม่แตะตัววัดอื่น เช่น relMAE · StdRatio)
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    prev = np.asarray(prev_close, dtype=float)
    if not np.isfinite(pred).all():
        return {"pred_finite": False, **{c: np.nan for c in METRIC_COLS}}

    err = pred - y
    err_baht = err * prev                        # (prev·(1+pred)) − (prev·(1+y)) = prev·(pred − y)
    naive_mae = float(np.mean(np.abs(y)))
    naive_mae_baht = float(np.mean(np.abs(y) * prev))
    if not naive_mae > 0:
        raise ValueError("MAE ของ Naive = 0 → relMAE นิยามไม่ได้ (ห้ามเติม epsilon)")

    moved = y != 0                               # วันที่ราคาไม่เปลี่ยนไม่นับในความแม่นของทิศทาง
    varies = np.ptp(y) > 0 and np.ptp(pred) > 0
    out = {
        "pred_finite": True, "n": len(y),
        "MAE_baht": float(np.mean(np.abs(err_baht))),
        "RMSE_baht": float(np.sqrt(np.mean(err_baht ** 2))),
        "MAE_return": float(np.mean(np.abs(err))),
        "RMSE_return": float(np.sqrt(np.mean(err ** 2))),
        "relMAE": float(np.mean(np.abs(err)) / naive_mae),
        "relMAE_baht": float(np.mean(np.abs(err_baht)) / naive_mae_baht),
        "R2_OOS": float(1 - np.mean(err ** 2) / np.mean(y ** 2)),        # เทียบกับ Naive ไม่ใช่ค่าเฉลี่ย
        "DirAcc": float(np.mean(np.sign(pred[moved]) == np.sign(y[moved]))) if moved.any() else np.nan,
        "IC": float(np.corrcoef(pred, y)[0, 1]) if varies else np.nan,
        "IC_spearman": float(spearmanr(pred, y).statistic) if varies else np.nan,
        "StdRatio": float(np.std(pred) / np.std(y)) if np.ptp(y) > 0 else np.nan,
        "Bias": float(np.mean(err)),
        "naive_MAE_return": naive_mae,
    }
    if const_ic_zero:
        out["collapsed"] = is_constant_prediction(y, pred)
        if out["collapsed"]:
            out["IC"] = 0.0
            out["IC_spearman"] = 0.0
    return out


def evaluation_row(model, cfg_name, ticker, fold_k, y, pred, prev_close, *, const_ic_zero=False, **tags):
    return {"model": model, "config": cfg_name, "ticker": ticker, "fold": fold_k,
            **tags, **metrics(y, pred, prev_close, const_ic_zero=const_ic_zero)}


def score_config(evals, n_expected=2 * N_FOLDS):
    """
    รวมผลของ config เดียว (ต้องครบ 6 evaluation)

    mean_IC          = IC (Pearson) เฉลี่ยของ 6 evaluation ← **เกณฑ์เลือก config** (DECISIONS B16)
    mean_relMAE      = relMAE (return) เฉลี่ย ← ตัววัดหลักที่รายงาน (ไม่ใช้เลือก)
    stdratio_ok      = StdRatio finite และ ≥ 0.05 ทุก evaluation ← **ธง** ไม่ตัดสิทธิ์ (เดิมชื่อ passed_guard)
    ic_seed_std      = std (ddof = 1) ข้าม seed ของ "IC เฉลี่ย 6 evaluation" ที่คิดจากคำทำนายของแต่ละ seed เดี่ยว ๆ
                       (ต้องมีคอลัมน์ ic_s0, ic_s1, … ใน evals · ใช้ประมาณ noise ของ seed — DECISIONS B17)
    """
    g = pd.DataFrame(evals)
    if len(g) != n_expected:
        raise RuntimeError(f"evaluation ไม่ครบ ({len(g)} จาก {n_expected}) -- ถือเป็นบั๊ก")
    valid = bool(g["pred_finite"].all()) and bool(np.isfinite(g["relMAE"]).all())
    sr = g["StdRatio"].to_numpy(dtype=float)
    out = {"n_evals": len(g), "valid": valid,
           "stdratio_ok": bool(valid and np.isfinite(sr).all() and (sr >= GUARD_MIN_STDRATIO).all()),
           "n_evals_beat_naive": int((g["relMAE"] < 1).sum()),
           "n_evals_ic_positive": int((g["IC"] > 0).sum())}
    seed_cols = sorted((c for c in g.columns if c.startswith("ic_s") and c[4:].isdigit()), key=lambda c: int(c[4:]))
    if seed_cols:
        by_seed = [float(g[c].mean()) for c in seed_cols]               # IC เฉลี่ย 6 evaluation ของ seed แต่ละตัว
        out["ic_by_seed"] = ";".join(f"{v:.5f}" for v in by_seed)
        out["ic_seed_std"] = float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else np.nan
    for c in ("relMAE", "relMAE_baht", "MAE_baht", "RMSE_baht", "MAE_return", "R2_OOS",
              "DirAcc", "IC", "IC_spearman", "StdRatio"):
        out[f"mean_{c}"] = float(g[c].mean())
    out["min_StdRatio"] = float(np.nanmin(sr)) if np.isfinite(sr).any() else np.nan
    if "collapsed" in g.columns:                                          # B39 (Stage 6): จำนวนคำทำนายคงที่ (IC := 0)
        out["n_collapsed_evals"] = int(g["collapsed"].astype(bool).sum())
        cs = [c for c in g.columns if c.startswith("collapsed_s") and c[11:].isdigit()]
        out["n_collapsed_seed_runs"] = int(sum(g[c].astype(bool).sum() for c in cs))
    if "relMAE_shrunk" in g.columns:                                      # หลังปรับขนาดด้วย c (B18 · รายงานเพิ่ม ไม่ใช้เลือก)
        out["mean_relMAE_shrunk"] = float(g["relMAE_shrunk"].mean())
        out["mean_relMAE_baht_shrunk"] = float(g["relMAE_baht_shrunk"].mean())
        out["mean_MAE_baht_shrunk"] = float(g["MAE_baht_shrunk"].mean())
        out["n_evals_beat_naive_shrunk"] = int((g["relMAE_shrunk"] < 1).sum())
        out["mean_c"] = float(g["c_chosen"].mean())
        out["n_c_zero"] = int((g["c_chosen"] == 0).sum())
    return out


# ---------------------------------------------------------------
# Diebold–Mariano (ported from task_b/phase2/stats.py, branch task_b)
# ---------------------------------------------------------------
def hac_mean_test(d):
    """
    ทดสอบว่าค่าเฉลี่ยของอนุกรม d ต่างจาก 0 ไหม โดยใช้ความแปรปรวนแบบ HAC (Bartlett)
    จำเป็นเพราะ error รายวันมีสหสัมพันธ์ข้ามวัน ถ้าใช้ t-test ธรรมดาจะได้ p เล็กเกินจริง
    คืน (mean, se, stat, p สองด้าน, bandwidth)
    """
    d = np.asarray(d, dtype=float)
    T = len(d)
    mean = float(d.mean())
    u = d - mean
    m = int(np.floor(4 * (T / 100) ** (2 / 9)))
    lrv = float(np.dot(u, u) / T)
    for j in range(1, m + 1):
        lrv += 2 * (1 - j / (m + 1)) * float(np.dot(u[j:], u[:-j]) / T)
    se = float(np.sqrt(max(lrv, 1e-300) / T))
    stat = mean / se
    return mean, se, float(stat), float(2 * (1 - norm.cdf(abs(stat)))), m


def dm_test(preds_a, preds_b):
    """
    Diebold–Mariano : โมเดล A ชนะ B อย่างมีนัยสำคัญไหม
    preds_* : DataFrame คอลัมน์ date, ticker, fold, y, pred (ต้องเป็นวันชุดเดียวกัน)
    unit    : d = |error ของ A| − |error ของ B|  (ลบ = A ดีกว่า)
    รวมสองหุ้นเป็นอนุกรมรายวันด้วยการเฉลี่ย d ในวันเดียวกัน แล้วทดสอบด้วย HAC
    """
    key = ["date", "ticker", "fold"]
    m = preds_a[key + ["y", "pred"]].merge(preds_b[key + ["y", "pred"]], on=key, suffixes=("_a", "_b"))
    if len(m) != len(preds_a) or len(m) != len(preds_b):
        raise ValueError(f"แถวของสองโมเดลไม่ตรงกัน ({len(preds_a)}, {len(preds_b)}, join {len(m)})")
    if not np.allclose(m["y_a"], m["y_b"]):
        raise ValueError("y ของสองโมเดลไม่ตรงกัน")
    m["d"] = (m["pred_a"] - m["y_a"]).abs() - (m["pred_b"] - m["y_b"]).abs()

    daily = m.groupby("date")["d"].mean().sort_index().to_numpy()
    mean, se, stat, p, bw = hac_mean_test(daily)
    ev = m.groupby(["ticker", "fold"])["d"].mean()
    nz = ev[ev != 0]
    n_better = int((nz < 0).sum())
    return {"n_days": len(daily), "mean_d": mean, "se": se, "DM": stat, "p_DM": p, "bandwidth": bw,
            "n_eval": len(nz), "n_eval_A_better": n_better,
            "p_sign": float(binomtest(n_better, len(nz), 0.5).pvalue) if len(nz) else np.nan}


def predictions_frame(dates, ticker, fold_k, y, pred, prev_close, **tags):
    """ตารางคำทำนายรายวัน — ใช้ทำ DM test และกราฟ"""
    return pd.DataFrame({**tags, "date": dates, "ticker": ticker, "fold": fold_k,
                         "y": np.asarray(y, dtype=float), "pred": np.asarray(pred, dtype=float),
                         "prev_close": np.asarray(prev_close, dtype=float)})
