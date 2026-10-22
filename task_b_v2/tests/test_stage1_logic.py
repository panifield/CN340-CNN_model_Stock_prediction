"""
test_stage1_logic.py — รายการ config ของ Stage 1, การตัดสินคาดการณ์ล่วงหน้า, และการตรวจรันซ้ำ ต้องถูกต้อง
==========================================================================================================
ไฟล์นี้ทำอะไร
    * ยืนยันว่า config ของ Stage 1 ต่างจาก base "ทีละปัจจัย" จริง (ไม่เผลอเปลี่ยนสองอย่างพร้อมกัน)
    * ยืนยันว่าการตัดสินคาดการณ์ P1/P2 ใช้เกณฑ์ที่ประกาศไว้ใน DECISIONS ส่วน C เป๊ะ ๆ
    * ยืนยันว่าการเทียบ base กับ C3 จับความต่างแม้เล็กน้อยได้
ทำไมต้องมี: Stage 1 รันครั้งเดียวจริง ถ้า config ผิดหรือเกณฑ์ตัดสินเพี้ยนจะรู้ตอนเห็นผลแล้ว ซึ่งสายเกินไป
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import run_stage1, selection
from src.configs import BASE


def changed_fields(cfg):
    a, b = dataclasses.asdict(BASE), dataclasses.asdict(cfg)
    return {k for k in a if a[k] != b[k] and k != "name"}


def test_stage1_has_nine_unique_configs_with_base_first():
    cfgs = run_stage1.all_configs()
    names = [c.name for c in cfgs]
    assert len(names) == len(set(names)) == 9 and names[0] == "base"
    assert set(names) == {"base", "win64", "win256", "complex", "norm_global", "norm_image", "ch3", "ch4", "detrend"}


@pytest.mark.parametrize("name,expected", [
    ("win64", {"window"}),
    ("win256", {"window", "nperseg", "hop"}),                 # SPEC: 256/32/4 เป็นชุดเดียวกัน
    ("complex", {"spec_type", "scale_mode"}),                 # complex ใช้สเกลคงที่ 1e-3 (B10)
    ("norm_global", {"normalize"}),
    ("norm_image", {"normalize"}),
    ("ch3", {"channels"}),
    ("ch4", {"channels"}),
    ("detrend", {"channels"}),
])
def test_each_option_changes_only_its_own_factor(name, expected):
    cfg = next(c for c in run_stage1.all_configs() if c.name == name)
    assert changed_fields(cfg) == expected


def test_detrend_is_a_comparator_not_a_candidate():
    """B19: detrend ไม่อยู่ในปัจจัยใด จึงไม่มีทางเป็นผู้ชนะหรือเข้า config รวม"""
    assert "detrend" in run_stage1.COMPARATORS
    assert all("detrend" not in opts for opts in run_stage1.FACTORS.values())


def test_complex_option_uses_fixed_scale_and_complex_type():
    c = next(c for c in run_stage1.all_configs() if c.name == "complex")
    assert c.spec_type == "complex" and c.scale_mode == "fixed" and c.fixed_scale == pytest.approx(1e-3)
    assert c.image_shape == (2, 9, 65)


def test_combined_config_applies_every_winner_change():
    from src.configs import variant
    changes = {}
    for factor, w in {"window": "win64", "channel": "ch4", "type": "base", "normalize": "base"}.items():
        if w != "base":
            changes.update(run_stage1.FACTORS[factor][w])
    comb = variant("combined", **changes)
    assert changed_fields(comb) == {"window", "channels"} and comb.window == 64 and comb.channels == "ret4"
    assert comb.image_shape == (4, 9, 33)


# ---------------------------------------------------------------
# ตัดสินคาดการณ์ที่ประกาศไว้ (DECISIONS ส่วน C)
# ---------------------------------------------------------------
def _setup(deltas, ic_complex=None, noise=0.02):
    """สร้างตารางคะแนน + ตารางเทียบ โดยกำหนด ΔIC ของแต่ละ config ที่เกี่ยวข้อง (base IC = 0.01)"""
    base_ic = 0.01
    rows = [{"config": "base", "mean_IC": base_ic, "ic_seed_std": 0.02}]
    for name, d in deltas.items():
        rows.append({"config": name, "mean_IC": base_ic + d, "ic_seed_std": 0.02})
    scores = pd.DataFrame(rows)
    if ic_complex is not None:
        scores.loc[scores["config"] == "complex", "mean_IC"] = ic_complex
    cmp_tables = [selection.compare_to_base(scores, "base", [n for n in deltas], noise)]
    return scores, cmp_tables, noise


ALL_P2 = {n: 0.0 for n in run_stage1.P2_CONFIGS}


def test_p1_true_only_when_complex_beats_base_by_more_than_noise_and_ic_positive():
    scores, tables, noise = _setup({"complex": 0.05, **ALL_P2})                    # +0.05 > 0.02 และ IC = 0.06 > 0
    assert "เป็นจริง" in run_stage1.judge_predictions(tables, scores, noise)[0]["สรุป"]


def test_p1_not_supported_when_within_noise():
    scores, tables, noise = _setup({"complex": 0.015, **ALL_P2})                   # +0.015 < 0.02
    assert "ไม่สนับสนุน" in run_stage1.judge_predictions(tables, scores, noise)[0]["สรุป"]


def test_p1_not_supported_when_ic_not_positive_even_if_beats_base():
    base_neg = {"complex": 0.05}
    scores, tables, noise = _setup({**base_neg, **ALL_P2}, ic_complex=-0.001)      # IC ติดลบ แม้เหนือ base
    # compare_to_base อ่าน IC จาก scores แล้ว ดังนั้นต้องสร้างตารางเทียบใหม่หลังปรับ IC
    tables = [selection.compare_to_base(scores, "base", list(base_neg) + list(ALL_P2), noise)]
    out = run_stage1.judge_predictions(tables, scores, noise)[0]["สรุป"]
    assert "ไม่สนับสนุน" in out or "เป็นจริง" not in out


def test_p2_not_contradicted_when_no_other_variant_beats_base():
    scores, tables, noise = _setup({"complex": 0.0, **ALL_P2})
    assert "ไม่ขัด" in run_stage1.judge_predictions(tables, scores, noise)[1]["สรุป"]


def test_p2_contradicted_and_names_the_breaking_variant():
    scores, tables, noise = _setup({"complex": 0.0, **{**ALL_P2, "ch4": 0.06}})
    verdict = run_stage1.judge_predictions(tables, scores, noise)[1]["สรุป"]
    assert "ขัด" in verdict and "ch4" in verdict and "win64" not in verdict


# ---------------------------------------------------------------
# ตรวจการรันซ้ำเทียบ C3
# ---------------------------------------------------------------
def _pred_frame(shift=0.0):
    n = 6
    return pd.DataFrame({"config": "base", "ticker": ["KBANK"] * n, "fold": [1] * n,
                         "date": pd.date_range("2024-01-01", periods=n).astype(str),
                         "pred": np.linspace(-0.001, 0.001, n) + shift})


def test_match_check_detects_identical_and_different(tmp_path):
    c3 = tmp_path / "predictions.csv"
    _pred_frame().to_csv(c3, index=False)
    same, _ = run_stage1.check_matches_c3(_pred_frame(), c3_path=c3)
    assert same is True
    diff, note = run_stage1.check_matches_c3(_pred_frame(shift=1e-12), c3_path=c3)
    assert diff is False and "e-" in note, "ต่างแม้ 1e-12 ก็ต้องจับได้ (เทสต์นี้ตรวจความเท่ากันทุกบิต)"


def test_match_check_without_c3_file_is_inconclusive(tmp_path):
    assert run_stage1.check_matches_c3(_pred_frame(), c3_path=tmp_path / "nope.csv")[0] is None
