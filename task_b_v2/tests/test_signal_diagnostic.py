"""
test_signal_diagnostic.py — diagnostic ต้องคำนวณถูกต้อง ไม่เทรนโมเดล และติดป้ายห้ามใช้เลือก (DECISIONS B26)
=========================================================================================================
ไฟล์นี้ทำอะไร: ตรวจสูตร ACF / IC / ธง "คงที่" / permutation null ด้วยข้อมูลที่รู้คำตอบ
ทำไมต้องมี: ตัวเลขของ diagnostic จะถูกใช้ตีความสมมติฐานของผู้ใช้ (return lag → IC บวก) ถ้าสูตรผิดจะสรุปผิดทาง
"""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src import paths, signal_diagnostic as sd


# ---------------------------------------------------------------
# ACF
# ---------------------------------------------------------------
def test_acf_hand_computed_case():
    # x = [1, −1, 1, −1] : mean 0 · Σd² = 4 · lag1 = (−1−1−1)/4 = −0.75 · lag2 = (1+1)/4 = 0.5 · lag3 = −1/4
    np.testing.assert_allclose(sd.acf([1, -1, 1, -1], max_lag=3), [-0.75, 0.5, -0.25])


def test_acf_recovers_ar1_coefficient_and_white_noise_is_near_zero():
    rng = np.random.default_rng(0)
    e = rng.normal(size=20000)
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = 0.5 * x[t - 1] + e[t]
    a = sd.acf(x)
    assert a[0] == pytest.approx(0.5, abs=0.03) and a[1] == pytest.approx(0.25, abs=0.03)
    assert np.all(np.abs(sd.acf(e)) < 0.03)


def test_acf_ignores_nan_at_the_start():
    x = np.array([np.nan, 1.0, -1.0, 1.0, -1.0])
    np.testing.assert_allclose(sd.acf(x, 1), sd.acf([1, -1, 1, -1], 1))


# ---------------------------------------------------------------
# IC
# ---------------------------------------------------------------
def test_pearson_and_spearman_matrices_match_reference_implementations():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    X[:, 4] = rng.integers(0, 2, 300)                       # dummy ค่า 0/1 (เหมือน dow) ต้องมีค่าซ้ำเยอะ
    y = X[:, 0] * 0.3 + rng.normal(size=300)
    pe, sp = sd.pearson_matrix(X, y), sd.spearman_matrix(X, y)
    for j in range(5):
        assert pe[j] == pytest.approx(np.corrcoef(X[:, j], y)[0, 1])
        assert sp[j] == pytest.approx(spearmanr(X[:, j], y).statistic)


def test_constant_column_gives_nan_not_an_error():
    X = np.column_stack([np.ones(50), np.arange(50.0)])
    assert np.isnan(sd.pearson_matrix(X, np.arange(50.0)))[0] and sd.pearson_matrix(X, np.arange(50.0))[1] == pytest.approx(1.0)


def _long(values):
    """values : {feature: [ic ของ 6 evaluation]} → ตารางยาวแบบที่ summarize_features รับ"""
    rows = []
    for f, ics in values.items():
        for (t, k), ic in zip(sd.EVALS, ics):
            rows.append({"ticker": t, "fold": k, "feature": f, "pearson": ic, "spearman": ic / 2})
    return pd.DataFrame(rows)


def test_consistency_flag_needs_five_of_six_same_sign_and_sorts_descending():
    s = sd.summarize_features(_long({
        "pos6": [0.1] * 6, "pos5": [0.1, 0.1, 0.1, 0.1, 0.1, -0.05], "pos4": [0.1, 0.1, 0.1, 0.1, -0.05, -0.05],
        "neg5": [-0.1, -0.1, -0.1, -0.1, -0.1, 0.02], "mixed": [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]}))
    assert list(s.index) == sorted(s.index, key=lambda f: -s.loc[f, "mean_pearson"]), "ต้องเรียงจากมากไปน้อย"
    assert s["consistent"].to_dict() == {"pos6": True, "pos5": True, "pos4": False, "neg5": True, "mixed": False}
    assert s["sign"].to_dict() == {"pos6": "+", "pos5": "+", "pos4": "", "neg5": "−", "mixed": ""}
    assert s.loc["pos6", "n_pos"] == 6 and s.loc["neg5", "n_neg"] == 5
    assert s.loc["pos6", "mean_spearman"] == pytest.approx(0.05)


def test_chance_probability_matches_the_hand_calculation():
    assert sd.chance_consistency_probability() == pytest.approx(14 / 64)                 # 2 × (6 + 1) / 64
    assert sd.chance_consistency_probability(n_eval=6, min_same=6) == pytest.approx(2 / 64)


# ---------------------------------------------------------------
# permutation null
# ---------------------------------------------------------------
def _fake(n_feat=29, n=266, seed=0):
    rng = np.random.default_rng(seed)
    dsets, folds = {}, [SimpleNamespace(k=k, eval_pos=np.arange(n * (k - 1), n * k)) for k in (1, 2, 3)]
    for t in ("KBANK", "ADVANC"):
        dsets[t] = SimpleNamespace(feat29=pd.DataFrame(rng.normal(size=(3 * n, n_feat))), y=rng.normal(size=3 * n))
    return dsets, folds


def test_permutation_null_is_reproducible_and_has_the_chance_level_mean():
    dsets, folds = _fake()
    a = sd.permutation_null(dsets, folds, n_perm=600, seed=5)
    b = sd.permutation_null(dsets, folds, n_perm=600, seed=5)
    np.testing.assert_array_equal(a["n_consistent"], b["n_consistent"])
    expected = sd.chance_consistency_probability() * 29                                  # ≈ 6.34
    assert a["n_consistent"].mean() == pytest.approx(expected, abs=0.5)
    assert a["n_consistent"].shape == (600,) and a["n_features"] == 29
    assert np.all(a["max_abs_mean_ic"] >= a["max_signed_mean_ic"])


def test_permutation_null_detects_a_planted_signal_as_far_above_null():
    """ถ้า feature หนึ่งมีสัญญาณจริงในทุก evaluation IC เฉลี่ยของมันต้องสูงกว่า P95 ของ null มาก"""
    dsets, folds = _fake()
    for t in dsets:
        dsets[t].y = dsets[t].y + 0.5 * dsets[t].feat29[0].to_numpy()                    # ฝังสัญญาณใน feature 0
    long = sd.feature_ic_by_eval(
        {t: SimpleNamespace(feat29=dsets[t].feat29.set_axis([f"f{j}" for j in range(29)], axis=1),
                            y=dsets[t].y, ok29=np.ones(len(dsets[t].y), bool)) for t in dsets}, folds)
    s = sd.summarize_features(long)
    null = sd.null_summary(sd.permutation_null(dsets, folds, n_perm=400))
    assert s.index[0] == "f0" and s.loc["f0", "mean_pearson"] > 3 * null["max_signed_mean_ic"]["p95"]


# ---------------------------------------------------------------
# เทียบกับโมเดล
# ---------------------------------------------------------------
def test_compare_models_reports_signed_best_and_absolute_best_with_sign_applied():
    s = sd.summarize_features(_long({"up": [0.02] * 6, "down": [-0.08] * 6}))
    models = pd.DataFrame([{"model": "RF", "ticker": t, "fold": k, "pearson": 0.1, "spearman": 0.2} for t, k in sd.EVALS])
    c = sd.compare_models(s, models).set_index("รายการ")
    signed = c.loc[c.index[0]]
    assert "up" in c.index[0] and signed["Pearson เฉลี่ย"] == pytest.approx(0.02)
    absolute = c.loc[c.index[1]]
    assert "down" in c.index[1] and absolute["Pearson เฉลี่ย"] == pytest.approx(0.08), "ใช้ feature 'down' กลับเครื่องหมายตามของมัน"
    assert c.loc["RF", "Pearson เฉลี่ย"] == pytest.approx(0.1) and c.loc["RF", "IC>0"] == 6


# ---------------------------------------------------------------
# ข้อกำหนดของ B26
# ---------------------------------------------------------------
def test_module_never_imports_torch_or_training_code():
    """diagnostic ต้องไม่เทรนโมเดล : import แล้วต้องไม่มี torch และ src.train อยู่ในระบบ"""
    code = "import sys, src.signal_diagnostic; print('torch' in sys.modules, 'src.train' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=paths.TASK_DIR)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.split() == ["False", "False"]


def test_label_text_is_the_one_the_user_asked_for():
    assert sd.LABEL == "diagnostic — ห้ามใช้เลือก feature หรือ config"


@pytest.mark.skipif(not (sd.OUT_DIR / "diagnostic_summary.md").exists(), reason="ยังไม่ได้รัน diagnostic")
def test_every_output_file_carries_the_label():
    md = (sd.OUT_DIR / "diagnostic_summary.md").read_text(encoding="utf-8")
    assert md.count(sd.LABEL) >= 2
    assert sd.LABEL in (sd.OUT_DIR / "permutation_null.json").read_text(encoding="utf-8")
