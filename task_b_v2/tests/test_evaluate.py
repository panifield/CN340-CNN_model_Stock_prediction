"""
test_evaluate.py — ตัววัดต้องให้ค่าที่รู้คำตอบล่วงหน้าได้ถูกต้อง (SPEC ข้อ 6)
===========================================================================
ไฟล์นี้ทำอะไร: ป้อนกรณีที่คำนวณมือได้ (ทำนายสมบูรณ์แบบ · ทำนายเท่า Naive · ทำนายกลับทิศ)
แล้วตรวจว่า relMAE / R²_OOS / DirAcc / StdRatio ออกมาตามที่ควรเป็น
ทำไมต้องมี: ตัววัดคือสิ่งที่ตัดสินทุกอย่างในงานนี้ ถ้ามันผิดเราจะเลือก config ผิดโดยไม่มีทางรู้
"""

import numpy as np
import pandas as pd
import pytest

from src import evaluate

rng = np.random.default_rng(0)
Y = rng.normal(0, 0.012, 300)
PREV = np.full(300, 200.0)


def test_naive_gives_relmae_exactly_one():
    m = evaluate.metrics(Y, evaluate.naive_prediction(Y), PREV)
    assert m["relMAE"] == pytest.approx(1.0)
    assert m["relMAE_baht"] == pytest.approx(1.0)
    assert m["R2_OOS"] == pytest.approx(0.0)
    assert m["StdRatio"] == 0.0


def test_perfect_prediction():
    m = evaluate.metrics(Y, Y, PREV)
    assert m["relMAE"] == pytest.approx(0.0, abs=1e-12)
    assert m["R2_OOS"] == pytest.approx(1.0)
    assert m["DirAcc"] == pytest.approx(1.0)
    assert m["IC"] == pytest.approx(1.0) and m["StdRatio"] == pytest.approx(1.0)


def test_sign_flipped_prediction_is_worse_than_naive():
    m = evaluate.metrics(Y, -Y, PREV)
    assert m["relMAE"] == pytest.approx(2.0)          # |−y − y| = 2|y|
    assert m["DirAcc"] == pytest.approx(0.0)
    assert m["IC"] == pytest.approx(-1.0)


def test_baht_metrics_use_prev_close():
    prev = np.linspace(100, 300, len(Y))
    m = evaluate.metrics(Y, np.zeros_like(Y), prev)
    assert m["MAE_baht"] == pytest.approx(float(np.mean(np.abs(Y) * prev)))
    assert m["MAE_baht"] != pytest.approx(m["MAE_return"] * prev.mean()), "ต้องถ่วงรายวัน ไม่ใช่คูณค่าเฉลี่ย"


def test_direction_accuracy_skips_zero_return_days():
    y = np.array([0.01, -0.01, 0.0, 0.0, 0.02])
    pred = np.array([0.01, -0.01, 0.5, -0.5, -0.02])      # 2 ถูก 1 ผิด (2 วันที่ y = 0 ไม่นับ)
    assert evaluate.metrics(y, pred, np.full(5, 100.0))["DirAcc"] == pytest.approx(2 / 3)


def test_non_finite_prediction_is_rejected():
    pred = np.zeros_like(Y)
    pred[3] = np.nan
    assert evaluate.metrics(Y, pred, PREV)["pred_finite"] is False


def test_zero_naive_mae_raises():
    with pytest.raises(ValueError):
        evaluate.metrics(np.zeros(10), np.zeros(10), np.full(10, 100.0))


# ---------------------------------------------------------------
# การรวมคะแนนและ guard
# ---------------------------------------------------------------
def _evals(relmae, stdratio):
    return [{"pred_finite": True, "relMAE": r, "relMAE_baht": r, "MAE_baht": 1.0, "RMSE_baht": 1.0,
             "MAE_return": 0.01, "R2_OOS": 0.0, "DirAcc": 0.5, "IC": 0.01, "IC_spearman": 0.01,
             "StdRatio": s} for r, s in zip(relmae, stdratio)]


def test_score_reports_mean_relmae_and_mean_ic_of_six_evaluations():
    s = evaluate.score_config(_evals([1.0, 0.99, 1.01, 1.0, 0.98, 1.02], [0.2] * 6))
    assert s["mean_relMAE"] == pytest.approx(1.0)
    assert s["mean_IC"] == pytest.approx(0.01)                           # _evals ตั้ง IC = 0.01 ทุก evaluation
    # ชนะ Naive = relMAE < 1 เท่านั้น (0.99 และ 0.98) · relMAE = 1.0 คือเสมอ ไม่นับว่าชนะ
    assert s["n_evals"] == 6 and s["stdratio_ok"] and s["n_evals_beat_naive"] == 2


def test_stdratio_flag_is_raised_when_any_evaluation_is_below_0_05():
    """หลัง B16 เป็นแค่ธง: StdRatio ต่ำ → stdratio_ok = False แต่ config ยัง valid (ไม่ถูกตัดสิทธิ์)"""
    s = evaluate.score_config(_evals([1.0] * 6, [0.2, 0.2, 0.049, 0.2, 0.2, 0.2]))
    assert s["valid"] and not s["stdratio_ok"]


def test_incomplete_evaluations_raise():
    with pytest.raises(RuntimeError):
        evaluate.score_config(_evals([1.0] * 5, [0.2] * 5))


# ---------------------------------------------------------------
# Diebold–Mariano
# ---------------------------------------------------------------
def _frame(pred, seed=0):
    n = len(Y)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.concat([evaluate.predictions_frame(dates, t, 1, Y, pred, PREV) for t in ("KBANK", "ADVANC")],
                     ignore_index=True)


def test_dm_detects_a_clearly_better_model():
    r = evaluate.dm_test(_frame(Y * 0.99), _frame(np.zeros_like(Y)))   # A เกือบสมบูรณ์แบบ · B = Naive
    assert r["mean_d"] < 0 and r["p_DM"] < 0.01 and r["n_eval_A_better"] == r["n_eval"]


def test_dm_on_identical_models_is_not_significant():
    r = evaluate.dm_test(_frame(np.zeros_like(Y)), _frame(np.zeros_like(Y)))
    assert r["mean_d"] == pytest.approx(0.0) and np.isnan(r["p_sign"])


def test_dm_requires_matching_rows():
    a = _frame(np.zeros_like(Y))
    with pytest.raises(ValueError):
        evaluate.dm_test(a, a.iloc[:-5])
