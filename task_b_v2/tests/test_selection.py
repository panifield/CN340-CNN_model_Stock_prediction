"""
test_selection.py — กติกาเลือกผู้ชนะด้วย IC + noise ของ seed ต้องทำงานตาม B16/B17 ทุกมุม
======================================================================================
ไฟล์นี้ทำอะไร: ป้อนตารางคะแนนปลอมที่รู้คำตอบ แล้วตรวจว่า
    * ผู้ชนะต้องชนะ base เกิน noise เท่านั้น (ต่างน้อยกว่า noise → คงค่า base)
    * ผลต่างติดลบมาก ๆ ก็ไม่ใช่ผู้ชนะ · เสมอพอดี noise ไม่ชนะ (ต้อง "เกิน")
    * noise รวมคิดจาก σ ของทุก config แบบ root-mean-square
    * กติกา B14: config รวมที่ IC ต่ำกว่า base แม้นิดเดียวต้องถูกจับว่าแพ้
    * score_config เก็บ IC รายseed และคำนวณ σ ข้าม seed ถูกต้อง
ทำไมต้องมี: ฟังก์ชันนี้ตัดสินทุก Stage ถ้ามันเอนเอียงแม้นิดเดียว ผลทั้งโครงการเอนเอียงตาม
"""

import numpy as np
import pandas as pd
import pytest

from src import evaluate, selection


def table(rows):
    return pd.DataFrame(rows, columns=["config", "mean_IC", "ic_seed_std"])


# ---------------------------------------------------------------
# noise รวม
# ---------------------------------------------------------------
def test_pooled_noise_is_root_mean_square_of_sigmas():
    t = table([("base", 0.0, 0.010), ("a", 0.0, 0.030), ("b", 0.0, 0.020)])
    assert selection.pooled_seed_noise(t) == pytest.approx(np.sqrt((0.01 ** 2 + 0.03 ** 2 + 0.02 ** 2) / 3))


def test_pooled_noise_skips_nan_sigmas():
    t = table([("base", 0.0, 0.02), ("a", 0.0, np.nan)])
    assert selection.pooled_seed_noise(t) == pytest.approx(0.02)
    assert np.isnan(selection.pooled_seed_noise(table([("base", 0.0, np.nan)])))


def test_sampling_se_reference_value():
    assert selection.sampling_se_reference() == pytest.approx(0.0250, abs=1e-3)


# ---------------------------------------------------------------
# ผู้ชนะของปัจจัย
# ---------------------------------------------------------------
NOISE = 0.02


def test_candidate_within_noise_keeps_base():
    t = table([("base", 0.010, 0.02), ("a", 0.025, 0.02)])                    # ต่าง +0.015 < 0.02
    winner, cmp = selection.pick_factor_winner(t, "base", ["a"], NOISE)
    assert winner == "base" and not cmp.iloc[0]["beats_base"]


def test_candidate_beyond_noise_wins():
    t = table([("base", 0.010, 0.02), ("a", 0.040, 0.02)])                    # ต่าง +0.030 > 0.02
    assert selection.pick_factor_winner(t, "base", ["a"], NOISE)[0] == "a"


def test_difference_exactly_equal_to_noise_does_not_win():
    t = table([("base", 0.0, 0.02), ("a", 0.020, 0.02)])
    assert selection.pick_factor_winner(t, "base", ["a"], 0.020)[0] == "base", "ต้อง \"เกิน\" noise ไม่ใช่เท่ากับ"


def test_much_worse_candidate_never_wins():
    t = table([("base", 0.050, 0.02), ("a", -0.100, 0.02)])
    assert selection.pick_factor_winner(t, "base", ["a"], NOISE)[0] == "base"


def test_best_of_several_beyond_noise_wins():
    t = table([("base", 0.0, 0.02), ("a", 0.030, 0.02), ("b", 0.080, 0.02), ("c", 0.025, 0.02)])
    winner, cmp = selection.pick_factor_winner(t, "base", ["a", "b", "c"], NOISE)
    assert winner == "b"
    assert list(cmp["config"]) == ["b", "a", "c"], "ตารางเรียงตาม IC จากมากไปน้อย"


def test_nan_ic_candidate_cannot_win():
    t = table([("base", 0.0, 0.02), ("a", np.nan, 0.02), ("b", 0.05, 0.02)])
    winner, cmp = selection.pick_factor_winner(t, "base", ["a", "b"], NOISE)
    assert winner == "b" and "NaN" in cmp.set_index("config").loc["a", "note"]


def test_nan_noise_means_nobody_wins():
    t = table([("base", 0.0, np.nan), ("a", 0.9, np.nan)])
    assert selection.pick_factor_winner(t, "base", ["a"], float("nan"))[0] == "base"


def test_unknown_base_raises():
    with pytest.raises(KeyError):
        selection.compare_to_base(table([("a", 0.0, 0.0)]), "base", ["a"], NOISE)


# ---------------------------------------------------------------
# B14 : config รวมแพ้ base
# ---------------------------------------------------------------
def test_combined_with_lower_ic_loses_even_by_a_hair():
    t = table([("base", 0.0500, 0.02), ("combined", 0.0499, 0.02)])
    loses, d = selection.combined_loses_to_base(t, "combined", "base")
    assert loses and d == pytest.approx(-0.0001)


def test_combined_with_higher_or_equal_ic_does_not_lose():
    assert not selection.combined_loses_to_base(table([("base", 0.05, 0), ("combined", 0.06, 0)]), "combined", "base")[0]
    assert not selection.combined_loses_to_base(table([("base", 0.05, 0), ("combined", 0.05, 0)]), "combined", "base")[0]


def test_combined_with_nan_ic_counts_as_losing():
    assert selection.combined_loses_to_base(table([("base", 0.05, 0), ("combined", np.nan, 0)]), "combined", "base")[0]


def test_missing_seed_noise_is_caught():
    with pytest.raises(RuntimeError):
        selection.assert_scores_have_seed_noise(pd.DataFrame({"config": ["base"], "mean_IC": [0.0]}))
    selection.assert_scores_have_seed_noise(table([("base", 0.0, 0.02)]))       # ไม่ error


# ---------------------------------------------------------------
# score_config : IC รายseed
# ---------------------------------------------------------------
def _evals(ic_by_seed_per_eval, ic_mean):
    """ic_by_seed_per_eval : list ยาว 6 ของ list [ic_s0, ic_s1, ic_s2] ต่อ evaluation"""
    rows = []
    for k, seeds in enumerate(ic_by_seed_per_eval):
        r = {"pred_finite": True, "relMAE": 1.0, "relMAE_baht": 1.0, "MAE_baht": 1.0, "RMSE_baht": 1.0,
             "MAE_return": 0.01, "R2_OOS": 0.0, "DirAcc": 0.5, "IC": ic_mean[k], "IC_spearman": 0.0, "StdRatio": 0.2}
        r.update({f"ic_s{i}": v for i, v in enumerate(seeds)})
        rows.append(r)
    return rows


def test_score_config_computes_seed_std_of_mean_ic():
    per_eval = [[0.10, 0.00, 0.05]] * 3 + [[0.20, 0.10, 0.15]] * 3             # IC เฉลี่ย 6 eval: seed0 .15 · seed1 .05 · seed2 .10
    s = evaluate.score_config(_evals(per_eval, [0.05] * 6))
    assert [float(x) for x in s["ic_by_seed"].split(";")] == pytest.approx([0.15, 0.05, 0.10])
    assert s["ic_seed_std"] == pytest.approx(np.std([0.15, 0.05, 0.10], ddof=1)) == pytest.approx(0.05)
    assert s["mean_IC"] == pytest.approx(0.05)


def test_score_config_without_seed_columns_has_no_seed_std():
    ev = _evals([[0.0, 0.0, 0.0]] * 6, [0.0] * 6)
    for r in ev:
        for c in ("ic_s0", "ic_s1", "ic_s2"):
            r.pop(c)
    assert "ic_seed_std" not in evaluate.score_config(ev)


def test_stdratio_flag_is_reported_but_does_not_invalidate():
    ev = _evals([[0.0, 0.0, 0.0]] * 6, [0.0] * 6)
    ev[2]["StdRatio"] = 0.01
    s = evaluate.score_config(ev)
    assert s["valid"] and not s["stdratio_ok"], "StdRatio ต่ำ = ธง ไม่ใช่การตัดสิทธิ์ (valid ยังเป็น True)"


# ---------------------------------------------------------------
# เกณฑ์ B22 (floor) : ต้องผ่านทั้ง noise และ floor
# ---------------------------------------------------------------
def test_floor_blocks_a_candidate_that_beats_noise_but_not_the_floor():
    t = table([("base", 0.0, 0.02), ("a", 0.04, 0.02)])                         # Δ = +0.04 > noise 0.02 แต่ < floor 0.06
    assert selection.pick_factor_winner(t, "base", ["a"], NOISE)[0] == "a", "ไม่มี floor → ชนะตามกติกา Stage 1"
    w, cmp = selection.pick_factor_winner(t, "base", ["a"], NOISE, floor=0.06)
    assert w == "base" and cmp.iloc[0]["threshold"] == pytest.approx(0.06) and not cmp.iloc[0]["beats_base"]


def test_candidate_must_beat_both_thresholds_and_the_larger_one_applies():
    t = table([("base", 0.0, 0.02), ("a", 0.07, 0.02)])
    assert selection.pick_factor_winner(t, "base", ["a"], NOISE, floor=0.0597)[0] == "a"              # 0.07 > 0.0597 > 0.02
    assert selection.pick_factor_winner(t, "base", ["a"], noise=0.08, floor=0.0597)[0] == "base"       # noise ใหญ่กว่า floor → ใช้ noise


def test_difference_equal_to_floor_does_not_win():
    t = table([("base", 0.0, 0.02), ("a", 0.0597, 0.02)])
    assert selection.pick_factor_winner(t, "base", ["a"], NOISE, floor=0.0597)[0] == "base"


def test_floor_none_keeps_the_stage1_behaviour_and_exposes_nan_floor():
    t = table([("base", 0.0, 0.02), ("a", 0.05, 0.02)])
    cmp = selection.compare_to_base(t, "base", ["a"], NOISE)
    assert cmp.iloc[0]["beats_base"] and np.isnan(cmp.iloc[0]["floor"]) and cmp.iloc[0]["threshold"] == pytest.approx(NOISE)
