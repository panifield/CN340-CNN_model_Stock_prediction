"""
metrics_1600.py — metric / NaN policy / economic diagnostics ของ Phase 1D (Mode A)
==================================================================================
นิยามทั้งหมดล็อกไว้ใน PHASE1D_PLAN.md ก่อนรัน · ห้ามวางตัวเลข Phase 1D ในตารางเดียวกับ daily model

NaN / Inf / zero-denominator policy:
  - Naive MAE = 0 ใน evaluation ใด -> relMAE นิยามไม่ได้ -> RankingUndefined (หยุด ranking ทั้งหมด)
  - config ต้องมีครบ 6 evaluations และ relMAE finite ทุกอัน (ห้าม nanmean)
  - prediction non-finite แม้แถวเดียว -> config invalid
  - std(y_true) = 0 -> StdRatio = NaN -> guard ไม่ผ่าน (ไม่แทนด้วย 0)
  - MSE_Naive = 0 -> R2_OOS = NaN
  - truth หรือ prediction คงที่ -> Rho = NaN
  - truth คงที่ -> R2_return = NaN (ใช้ policy เดียวทุก model/baseline)
"""

import numpy as np

from trading_costs import cost_round_trip, tick_size

N_EVALS = 6                 # 2 หุ้น x 3 folds
GUARD_MIN_STDRATIO = 0.05


class RankingUndefined(RuntimeError):
    """relMAE นิยามไม่ได้ (Naive MAE = 0) -- หยุด ranking ทั้งหมด ห้ามเติม epsilon/ข้าม fold"""


class IncompleteEvaluations(RuntimeError):
    """config มี evaluation ไม่ครบ 6 หรือมีค่า non-finite -- ถือเป็น bug -> STOP"""


def _arr(x):
    return np.asarray(x, dtype=float)


def main_metrics(y_true, y_pred, c15):
    """MAE/RMSE/R2 ในหน่วย return + MAE/RMSE หน่วยบาท (คำนวณจาก C15)"""
    y, p, c = _arr(y_true), _arr(y_pred), _arr(c15)
    err = p - y
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "MAE_return": float(np.mean(np.abs(err))),
        "RMSE_return": float(np.sqrt(np.mean(err ** 2))),
        "R2_return": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
        "MAE_baht": float(np.mean(np.abs(err) * c)),
        "RMSE_baht": float(np.sqrt(np.mean((err * c) ** 2))),
    }


def naive_mae(y_true):
    """Naive: ŷ = 0 (close_bar16 = C15)"""
    return float(np.mean(np.abs(_arr(y_true))))


def rel_mae(mae_model, mae_naive):
    if not mae_naive > 0:
        raise RankingUndefined(f"Naive MAE = {mae_naive} -> relMAE นิยามไม่ได้ -- หยุด ranking")
    return mae_model / mae_naive


def r2_oos(y_true, y_pred):
    """1 − MSE/MSE_Naive (Naive = 0) · MSE_Naive = 0 -> NaN"""
    y, p = _arr(y_true), _arr(y_pred)
    mse_naive = np.mean(y ** 2)
    if mse_naive == 0:
        return np.nan
    return float(1 - np.mean((p - y) ** 2) / mse_naive)


def shape(y_true, y_pred):
    """StdRatio / Bias / Rho + สัดส่วนวันที่ y = 0 (diagnostic -- StdRatio ใช้เป็น guard เท่านั้น)"""
    y, p = _arr(y_true), _arr(y_pred)
    true_varies = np.ptp(y) > 0
    pred_varies = np.ptp(p) > 0
    return {
        "StdRatio": float(p.std() / y.std()) if true_varies else np.nan,
        "Bias": float(np.mean(p - y)),
        "Rho": float(np.corrcoef(p, y)[0, 1]) if (true_varies and pred_varies) else np.nan,
        "frac_y_zero": float(np.mean(y == 0)),
    }


def passes_guard(stdratios):
    """StdRatio finite และ >= 0.05 ทุก evaluation (NaN = ไม่ผ่าน)"""
    s = _arr(stdratios)
    return bool(len(s) == N_EVALS and np.isfinite(s).all() and (s >= GUARD_MIN_STDRATIO).all())


def primary_score(rel_maes):
    """mean relMAE ของ 6 evaluation · ขาด/non-finite -> IncompleteEvaluations"""
    r = _arr(rel_maes)
    if len(r) != N_EVALS:
        raise IncompleteEvaluations(f"ต้องมี {N_EVALS} evaluations ได้ {len(r)}")
    if not np.isfinite(r).all():
        raise IncompleteEvaluations(f"relMAE non-finite: {r}")
    return float(r.mean())


def economic_diagnostics(y_true, y_pred, c15, cost=None):
    """
    two-sided magnitude diagnostic -- ไม่ใช่ backtest และไม่ใช่จำนวนโอกาส long-only
    return เทียบกับ cost (สัดส่วน) · บาทเทียบกับ tick (บาท) -- ห้ามเทียบ return กับ tick ตรง ๆ
    ไม่คำนวณ PnL ที่ราคา C15 (ราคาที่ซื้อได้จริงเกิดหลัง prediction timestamp)
    """
    y, p, c = _arr(y_true), _arr(y_pred), _arr(c15)
    cost = cost_round_trip() if cost is None else cost
    ticks = np.array([tick_size(x) for x in c])
    actual_mag, pred_mag = np.abs(y), np.abs(p)
    actual_baht = actual_mag * c
    signal = pred_mag > cost
    n_signal = int(signal.sum())
    if n_signal:
        ys, ps = y[signal], p[signal]
        hit = float(np.mean((np.sign(ps) == np.sign(ys)) & (ys != 0)))   # y = 0 ไม่นับ hit
    else:
        hit = np.nan
    return {
        "frac_actual_ge_cost": float(np.mean(actual_mag >= cost)),
        "frac_actual_ge_1tick": float(np.mean(actual_baht >= ticks)),
        "frac_pred_gt_cost": float(np.mean(signal)),
        "signal_days": n_signal,
        "sign_hit_rate_on_signal": hit,
        "signal_days_y_zero": int(np.sum(signal & (y == 0))),
        "mean_actual_move_baht": float(np.mean(actual_baht)),
        "mean_pred_move_baht": float(np.mean(pred_mag * c)),
    }
