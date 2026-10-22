"""
test_placebo.py — การนับผู้ชนะ, กติกา B21, เกณฑ์ B22 และการสลับ y ของ placebo ต้องถูกต้องตามที่ประกาศไว้
=====================================================================================================
ไฟล์นี้ทำอะไร: สร้างผล placebo ปลอมที่รู้คำตอบ แล้วตรวจว่า `placebo.analyze` นับผู้ชนะ (ทั้งแบบระดับปัจจัยและระดับ config)
คิดเกณฑ์ B22 และตัดสินตามกติกา B21 ได้ตรงตามนิยามใน DECISIONS B21/B23
ทำไมต้องมี: placebo ตัดสินว่า Stage 1 ใช้ config รวมหรือ base · ถ้าการนับผิดแม้นิดเดียว ผลของทั้ง Stage 1 ผิดตาม
"""

import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from src import controls, data, placebo, runner
from src.configs import BASE
from src.run_stage1 import FACTORS, all_configs

NAMES = [c.name for c in all_configs()]                       # 9 config (base อยู่แรก)
D = 0.02                                                      # σ ข้าม seed ที่ตั้งให้ทุก config → noise ของ stage = 0.02


def make_evals(ic_by_shuffle, sigma=D):
    """
    ic_by_shuffle : {shuffle: {config: mean_IC}} · config ที่ไม่ระบุ IC = 0
    ทุก evaluation ของ config ได้ IC = mean_IC · IC รายseed = mean_IC + (−σ, 0, +σ) → σ ข้าม seed = sigma พอดี
    """
    rows = []
    for s, table in ic_by_shuffle.items():
        for name in NAMES:
            ic = table.get(name, 0.0)
            for t in ("KBANK", "ADVANC"):
                for k in (1, 2, 3):
                    rows.append({"pred_finite": True, "config": name, "shuffle": s, "ticker": t, "fold": k,
                                 "relMAE": 1.0, "relMAE_baht": 1.0, "MAE_baht": 1.0, "RMSE_baht": 1.0,
                                 "MAE_return": 0.01, "R2_OOS": 0.0, "DirAcc": 0.5, "IC": ic, "IC_spearman": 0.0,
                                 "StdRatio": 0.1, "ic_s0": ic - sigma, "ic_s1": ic, "ic_s2": ic + sigma})
    return pd.DataFrame(rows)


def run(ic_by_shuffle):
    return placebo.analyze(make_evals(ic_by_shuffle), NAMES, FACTORS, shuffle_seeds=tuple(ic_by_shuffle))


def test_noise_is_the_pooled_seed_sigma():
    r = run({0: {}, 1: {}, 2: {}})
    assert r["per_shuffle"]["noise"].tolist() == pytest.approx([D, D, D], rel=1e-9)


def test_pure_noise_with_no_candidate_beating_gives_zero_winners_and_combined():
    r = run({0: {"win64": 0.015}, 1: {"ch3": -0.03}, 2: {}})                # ทุกตัวไม่เกิน noise 0.02
    assert r["per_shuffle"]["n_factor_winners"].tolist() == [0, 0, 0]
    assert r["summary"]["mean_n_factor_winners"] == 0 and r["summary"]["decision_primary"] == "combined"


def test_counts_factor_winners_and_candidates_separately():
    r = run({0: {"win64": 0.05},                                    # 1 ปัจจัย · 1 config
             1: {"win64": 0.05, "ch3": 0.05},                       # 2 ปัจจัย · 2 config
             2: {"norm_global": 0.05, "norm_image": 0.05}})         # 1 ปัจจัย · 2 config (ปัจจัย normalize เดียวกัน)
    assert r["per_shuffle"]["n_factor_winners"].tolist() == [1, 2, 1]
    assert r["per_shuffle"]["n_candidates_beating"].tolist() == [1, 2, 2]
    assert r["summary"]["mean_n_factor_winners"] == pytest.approx(4 / 3)
    assert r["summary"]["mean_n_candidates_beating"] == pytest.approx(5 / 3)


def test_rule_b21_boundary_is_inclusive_at_one():
    assert placebo.decide(1.0) == "base" and placebo.decide(0.999) == "combined" and placebo.decide(2.5) == "base"


def test_primary_and_secondary_can_disagree_and_it_is_flagged():
    # ปัจจัยเดียว (normalize) มี 2 config ชนะใน 2 จาก 3 shuffle → ระดับปัจจัยเฉลี่ย 0.67 (<1) แต่ระดับ config เฉลี่ย 1.33 (≥1)
    r = run({0: {"norm_global": 0.05, "norm_image": 0.05}, 1: {"norm_global": 0.05, "norm_image": 0.05}, 2: {}})
    s = r["summary"]
    assert s["mean_n_factor_winners"] == pytest.approx(2 / 3) and s["mean_n_candidates_beating"] == pytest.approx(4 / 3)
    assert s["decision_primary"] == "combined" and s["decision_secondary"] == "base"
    assert s["primary_and_secondary_agree"] is False


def test_b22_threshold_is_the_max_delta_over_candidates_and_shuffles():
    r = run({0: {"win64": 0.01, "ch4": -0.02}, 1: {"complex": 0.037}, 2: {"norm_image": 0.02}})
    assert r["summary"]["b22_threshold"] == pytest.approx(0.037)
    assert r["summary"]["n_delta_values"] == 21, "7 ผู้สมัคร × 3 shuffle"
    assert r["summary"]["max_delta_per_shuffle"] == pytest.approx([0.01, 0.037, 0.02])


def test_delta_is_relative_to_base_of_the_same_shuffle():
    r = run({0: {"base": 0.03, "win64": 0.05}, 1: {"base": -0.01, "win64": 0.00}, 2: {}})
    d = r["deltas"].set_index(["shuffle", "config"])["delta_vs_base"]
    assert d[(0, "win64")] == pytest.approx(0.02) and d[(1, "win64")] == pytest.approx(0.01)


def test_detrend_is_never_a_candidate_but_counts_in_noise():
    r = run({0: {"detrend": 0.5}, 1: {}, 2: {}})
    assert "detrend" not in set(r["deltas"]["config"]), "detrend เป็นตัวเทียบ ไม่ใช่ผู้สมัคร (B19/B23)"
    assert r["per_shuffle"]["n_factor_winners"][0] == 0
    # IC สูงผิดปกติของ detrend ต้องไม่ดัน b22_threshold (ไม่อยู่ในกลุ่มที่ใช้หา ΔIC สูงสุด)
    assert r["summary"]["b22_threshold"] == pytest.approx(0.0)


def test_candidates_are_exactly_the_seven_options():
    opts = [c for o in FACTORS.values() for c in o]
    assert sorted(opts) == sorted(["win64", "win256", "complex", "norm_global", "norm_image", "ch3", "ch4"])


# ---------------------------------------------------------------
# การสลับ y ต้องเหมือน negative control ของ C3 ทุกประการ
# ---------------------------------------------------------------
def _prepared():
    return {"y_fit": np.arange(50, dtype=float), "y_val": np.arange(100, 110, dtype=float)}


def test_shuffle_is_deterministic_permutes_fit_and_val_separately():
    a, b = controls.shuffled_targets(_prepared(), 0), controls.shuffled_targets(_prepared(), 0)
    np.testing.assert_array_equal(a["fit"], b["fit"])
    np.testing.assert_array_equal(np.sort(a["fit"]), np.arange(50.0))                 # เป็น permutation ของ fit เท่านั้น
    np.testing.assert_array_equal(np.sort(a["val"]), np.arange(100.0, 110.0))         # val ไม่ปนกับ fit
    assert not np.array_equal(a["fit"], np.arange(50.0)), "ต้องถูกสลับจริง"
    c = controls.shuffled_targets(_prepared(), 1)
    assert not np.array_equal(a["fit"], c["fit"]), "คนละ shuffle seed ต้องได้การสลับต่างกัน"


def test_shuffle_matches_the_original_negative_control_recipe():
    """สูตรเดิมของ C3: rng = default_rng(seed); {k: rng.permutation(y_k) for k in (fit, val)}"""
    rng = np.random.default_rng(2)
    manual = {k: rng.permutation(_prepared()[f"y_{k}"]) for k in ("fit", "val")}
    got = controls.shuffled_targets(_prepared(), 2)
    np.testing.assert_array_equal(got["fit"], manual["fit"])
    np.testing.assert_array_equal(got["val"], manual["val"])


# ---------------------------------------------------------------
# ตรวจความสอดคล้องกับ negative control ของ C3
# ---------------------------------------------------------------
def _neg_frame():
    return pd.DataFrame({"control": "negative", "ticker": ["KBANK"] * 3 + ["ADVANC"] * 3, "fold": [1, 2, 3] * 2,
                         "n": 266, "relMAE": [1.01, 1.02, 1.03, 1.04, 1.05, 1.06],
                         "IC": [0.001, -0.002, 0.003, -0.004, 0.005, -0.006]})


def _mine(shift=0.0):
    n = _neg_frame()
    return pd.DataFrame({"config": "base", "shuffle": 0, "ticker": n["ticker"], "fold": n["fold"],
                         "IC": n["IC"] + shift, "relMAE": n["relMAE"]})


def test_consistency_check_detects_identical_and_different(tmp_path):
    f = tmp_path / "neg.csv"
    _neg_frame().to_csv(f, index=False)
    ok, _ = placebo.check_against_negative_control(_mine(), f)
    assert ok is True
    bad, note = placebo.check_against_negative_control(_mine(shift=1e-12), f)
    assert bad is False and "e-" in note
    assert placebo.check_against_negative_control(_mine(), tmp_path / "missing.csv")[0] is None


# ---------------------------------------------------------------
# หน่วยงานจริง (เทรนเล็ก ๆ) : cache ต้องคืนค่าเดิม และ shuffle ต่างกันให้ผลต่างกัน
# ---------------------------------------------------------------
@pytest.mark.slow
def test_placebo_unit_cache_and_shuffle_sensitivity(tmp_path):
    cfg = replace(BASE, name="tiny", max_epochs=3, patience=3)
    cache = tmp_path / "cache"
    a = placebo.placebo_unit((cfg, "KBANK", 3, 0, (0, 1), "2025-02-25", cache))
    b = placebo.placebo_unit((cfg, "KBANK", 3, 0, (0, 1), "2025-02-25", cache))
    assert a["from_cache"] is False and b["from_cache"] is True and a["IC"] == b["IC"]
    c = placebo.placebo_unit((cfg, "KBANK", 3, 1, (0, 1), "2025-02-25", None))
    assert c["IC"] != a["IC"], "คนละ shuffle ต้องให้คำทำนายต่างกัน"
    assert {"ic_s0", "ic_s1", "shuffle", "relMAE_s0"} <= set(a)


@pytest.mark.slow
def test_placebo_unit_equals_the_c3_negative_control_path():
    """placebo (shuffle 0) ต้องให้ IC/relMAE เท่ากับ controls.run_negative (seed 0) ทุกบิต — ขั้นตอนเดียวกัน"""
    cfg = replace(BASE, name="tiny", max_epochs=3, patience=3)
    ds = data.build_dataset("KBANK", "2025-02-25")
    fold = data.make_folds(ds.dates)[2]
    neg = controls.run_negative(ds, cfg, fold, (0, 1), seed=0)
    pl = placebo.placebo_unit((cfg, "KBANK", 3, 0, (0, 1), "2025-02-25", None))
    assert pl["IC"] == neg["IC"] and pl["relMAE"] == neg["relMAE"]
