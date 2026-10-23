"""
test_stage6.py — การออกแบบ HPO (Stage 6) และกติกาตัดสินต้องทำงานตรงตามที่ประกาศใน DECISIONS B38
==========================================================================================
ไฟล์นี้ทำอะไร
    1. การออกแบบ 24 config: ซ้ำรันได้เหมือนเดิม · สมดุล 8/8/8 (noise_std 6/6/6/6) · ไม่ซ้ำกันเอง/ไม่ซ้ำ base · hop ผ่านกติกา ·
       ถ้าไม่ผ่านต้อง "สุ่มใหม่ทั้งชุดด้วย seed ถัดไป" และบันทึกจำนวนครั้ง · configs.csv ตรงกับที่โค้ดสร้างทุกตัวอักษร
    2. A8: nperseg/hop ต่างกันได้ภาพ shape ต่างกันจริงและไม่ดึง cache ของกัน · สเกลภาพ/สถิติ normalize fit ต่อ config จากแถว fit เท่านั้น ·
       น้ำหนักวันสุดท้ายของ Hann ไม่ใกล้ 0 ทุกคู่ · y ที่สลับเหมือนกันทุก config ในรอบเดียวกัน
    3. กติกา A5/A6 ด้วยผลปลอมที่รู้คำตอบ: T_s/T_hpo · เกณฑ์ชนะทีละข้อ (1)–(4) · ตรวจเฉพาะอันดับ 1
ทำไมต้องมี: ผลของ Stage 6 ตัดสินว่า track 1 ใช้ config ไหนใน final · ถ้ากติกาเพี้ยนแม้ข้อเดียวต้องรู้ก่อนรัน ไม่ใช่หลังเห็นผล
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import hpo, models, paths, runner, run_stage6
from src.configs import BASE, variant


# ---------------------------------------------------------------
# 1. การออกแบบ
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def design():
    return hpo.design_configs()


def test_design_is_deterministic(design):
    configs, info = design
    configs2, info2 = hpo.design_configs()
    assert [c for c in configs] == [c for c in configs2] and info == info2
    assert len(configs) == 25 and configs[0] == BASE and configs[0].name == "base"
    assert [c.name for c in configs[1:]] == [f"hpo{i:02d}" for i in range(1, 25)]
    assert info["seed_first"] == hpo.DESIGN_SEED == 38


def test_design_is_balanced(design):
    configs, _ = design
    counts = hpo.level_counts(configs)
    for p, levels in hpo.GRID.items():
        expect = 24 // len(levels)                                  # 3 ระดับ → 8 · 4 ระดับ (noise_std) → 6
        assert sorted(counts[p].values()) == [expect] * len(levels), p
    assert sorted(counts["noise_std"].values()) == [6, 6, 6, 6]


def test_design_has_no_duplicates_and_respects_hop_rule(design):
    configs, _ = design
    keys = [hpo.config_key_without_name(c) for c in configs]
    assert len(set(keys)) == 25                                     # ไม่ซ้ำกันเอง ไม่ซ้ำ base
    for c in configs:
        assert 1 <= c.hop < c.nperseg and 2 * c.hop <= c.nperseg
        assert c.image_shape == (1, c.nperseg // 2 + 1, 128 // c.hop + 1)
    # ทุกอย่างที่ไม่ได้ค้นหาต้องเท่า base (ไม่เปลี่ยนสิ่งที่ Stage 1–5 ตัดสินแล้ว)
    for c in configs:
        for f in ("channels", "window", "spec_type", "scale_mode", "normalize", "arch", "head", "cnn_feat", "ann_input",
                  "train_mode", "loss", "huber_delta", "max_epochs", "patience"):
            assert getattr(c, f) == getattr(BASE, f), (c.name, f)


def test_violations_detect_each_rule():
    ok = hpo._draw(38)
    assert hpo.violations(ok) == []
    bad_hop = [dict(ok[0], nperseg=8, hop=8)] + ok[1:]
    assert any("hop" in r for r in hpo.violations(bad_hop))
    half = [dict(ok[0], nperseg=8, hop=5)] + ok[1:]                  # hop < nperseg แต่ซ้อนกันไม่ถึง 50%
    assert any("hop" in r for r in hpo.violations(half))
    assert any("ซ้ำกับ base" in r for r in hpo.violations([hpo.base_params()] + ok[1:]))
    assert any("ซ้ำกับ hpo01" in r for r in hpo.violations([ok[0], ok[0]] + ok[2:]))


def test_redraw_uses_next_seed_and_records_attempts(monkeypatch):
    real = hpo._draw
    calls = []

    def fake(seed):
        calls.append(seed)
        rows = real(seed)
        return [rows[0]] * len(rows) if seed in (38, 39) else rows          # สองครั้งแรกซ้ำกันหมด

    monkeypatch.setattr(hpo, "_draw", fake)
    configs, info = hpo.design_configs()
    assert calls == [38, 39, 40]
    assert info["attempts"] == 3 and info["redraws"] == 2 and info["seed_used"] == 40
    assert [r["seed"] for r in info["redraw_reasons"]] == [38, 39]
    assert configs == [BASE] + [replace(BASE, name=f"hpo{i:02d}", **r) for i, r in enumerate(real(40), 1)]


def test_configs_csv_matches_code_exactly():
    configs, csv, log = run_stage6.design_files()
    f = run_stage6.OUT / "configs.csv"
    assert f.exists(), "ยังไม่ได้รัน `python -m src.run_stage6 design`"
    assert f.read_text(encoding="utf-8") == csv
    assert (run_stage6.OUT / "design_log.json").read_text(encoding="utf-8") == log
    t = pd.read_csv(f)
    assert len(t) == 25 and list(t["name"][:2]) == ["base", "hpo01"]
    assert {"image_shape", "n_params"} <= set(t.columns)
    for c, row in zip(configs, t.itertuples()):
        assert row.n_params == models.count_params(models.build(c, 0)) and row.image_shape == "x".join(map(str, c.image_shape))
    assert t.loc[0, "n_params"] == 10081                              # base = 10,081 พารามิเตอร์ (Stage 3)


def test_design_file_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    out = tmp_path / "stage6_hpo"
    run_stage6.write_design(out)
    first = (out / "configs.csv").read_text(encoding="utf-8")
    run_stage6.write_design(out)                                       # เหมือนเดิม → ผ่าน
    (out / "configs.csv").write_text(first.replace("hpo01", "hpoXX"), encoding="utf-8")
    with pytest.raises(run_stage6.StopRun):
        run_stage6.write_design(out)                                   # ถูกแก้ → ต้องหยุด ไม่เขียนทับ
    assert "hpoXX" in (out / "configs.csv").read_text(encoding="utf-8")


def test_b22_floor_constant_matches_decision_file():
    assert run_stage6.b22_floor() == hpo.B22_FLOOR == 0.0597


# ---------------------------------------------------------------
# 2. A8 — สายงานของภาพและ cache
# ---------------------------------------------------------------
def test_nperseg_and_hop_change_image_shape_and_cache_key(design):
    configs, _ = design
    r = run_stage6.check_images_and_keys(configs)
    assert r["ok"], r
    assert r["n_pairs"] == 9 and r["pairs_have_distinct_shapes"]


def test_last_day_weight_is_not_near_zero_for_every_pair(design):
    configs, _ = design
    ww = run_stage6.check_window_weights(configs)
    assert len(ww) == 9 and ww["ok"].all()
    assert ww["last_day_weight"].min() >= hpo.MIN_LAST_DAY_WEIGHT


def test_fit_statistics_come_from_fit_rows_for_extreme_configs():
    cfgs = [variant("small", nperseg=8, hop=4), variant("big", nperseg=32, hop=1), BASE]
    df, ok = run_stage6.check_fit_statistics(cfgs, cases=(("KBANK", 2),))
    assert ok, df.to_string()
    assert len(df) == 3 and df["scale"].nunique() == 3                # แต่ละ config fit สเกลของตัวเอง ไม่ใช้ร่วมกัน


def test_shuffled_y_is_identical_across_configs(design):
    configs, _ = design
    assert run_stage6.check_shuffle_identical_across_configs(configs[:6])


@pytest.mark.slow
def test_same_name_configs_with_different_nperseg_hop_do_not_share_cache(tmp_path):
    r = run_stage6.check_cache_isolation(tmp_path)
    assert r["ok"], r


@pytest.mark.slow
def test_units_subset_runs_only_requested_evaluations(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "EXPERIMENT_LOG", tmp_path / "experiment_log.csv")
    tiny = replace(BASE, name="tiny", max_epochs=3, patience=3)
    evals, scores, _ = runner.run_configs([tiny], tmp_path / "s", seeds=(0, 1), units=[("KBANK", 1)], verbose=False)
    assert len(evals) == 1 and evals.loc[0, "ticker"] == "KBANK" and evals.loc[0, "fold"] == 1
    assert int(scores.loc[scores.index[0], "n_evals"]) == 1


def test_announced_files_must_be_committed(monkeypatch):
    class Out:
        returncode, stdout, stderr = 0, "?? results/stage6_hpo/configs.csv\n", ""
    monkeypatch.setattr(run_stage6.subprocess, "run", lambda *a, **k: Out)
    with pytest.raises(run_stage6.StopRun):
        run_stage6.ensure_announced_committed()
    Out.stdout = ""
    run_stage6.ensure_announced_committed()                           # สะอาด → ผ่าน


# ---------------------------------------------------------------
# 3. กติกา A5 / A6 ด้วยผลปลอมที่รู้คำตอบ
# ---------------------------------------------------------------
NAMES = [c.name for c in hpo.design_configs()[0]]
CANDS = NAMES[1:]


def fake_evals(ic, shuffle=None, n_eval=6):
    """ic: {config: [IC ของแต่ละ evaluation] หรือเลขตัวเดียว} · config ที่ไม่ระบุ = 0"""
    rows = []
    for n in NAMES:
        v = ic.get(n, 0.0)
        v = list(v) if isinstance(v, (list, tuple)) else [v] * n_eval
        for i, x in enumerate(v):
            row = {"config": n, "ticker": "KBANK" if i < 3 else "ADVANC", "fold": i % 3 + 1, "IC": x, "relMAE": 1.0}
            if shuffle is not None:
                row["shuffle"] = shuffle
            rows.append(row)
    return pd.DataFrame(rows)


def fake_scores(ic, relmae=None):
    relmae = relmae or {}
    return pd.DataFrame([{"config": n, "mean_IC": float(np.mean(ic.get(n, 0.0))), "mean_relMAE": relmae.get(n, 1.0)} for n in NAMES])


def test_placebo_stats_t_s_and_t_hpo():
    rounds = {100: {"base": 0.01, "hpo03": 0.05, "hpo07": 0.02},        # T = 0.04
              101: {"base": -0.02, "hpo10": 0.03},                       # T = 0.05
              102: {"base": 0.03, "hpo01": 0.02}}                        # T = max(−0.01, −0.03 ...) = 0 − 0.03 = −0.03 (ผู้สมัครอื่น 0 − 0.03)
    ev = pd.concat([fake_evals(t, shuffle=s) for s, t in rounds.items()], ignore_index=True)
    st = hpo.placebo_stats(ev, NAMES)
    assert st["t_s"][100] == pytest.approx(0.04) and st["t_s"][101] == pytest.approx(0.05)
    assert st["t_s"][102] == pytest.approx(-0.01)                         # hpo01 0.02 − base 0.03
    assert st["t_hpo"] == pytest.approx(0.05)
    assert st["delta"].shape == (24, 3) and "base" not in st["delta"].index


def test_placebo_stats_stops_on_nan():
    ev = fake_evals({"hpo05": [np.nan] + [0.0] * 5}, shuffle=100)
    with pytest.raises(RuntimeError):
        hpo.placebo_stats(ev, NAMES)


GOOD = {"base": 0.00, "hpo09": [0.10, 0.12, 0.11, 0.09, 0.13, 0.11]}       # ผู้สมัครอันดับ 1 ชนะชัด ทั้ง 6 evaluation


def judge(ic=None, confirm_ic=None, relmae=None, noise=0.02, t_hpo=0.04, confirm=True):
    ic = ic or GOOD
    confirm_ic = confirm_ic or GOOD
    return hpo.judge_winner(fake_scores(ic, relmae), fake_evals(ic),
                            fake_scores(confirm_ic) if confirm else None, fake_evals(confirm_ic) if confirm else None,
                            noise=noise, t_hpo=t_hpo)


def test_winner_when_every_criterion_passes():
    j = judge()
    assert j["top"] == "hpo09" and j["has_winner"] and j["c1"] and j["c2"] and j["c3"] and j["c4"]
    assert j["thr1"] == pytest.approx(0.0597) and j["thr3"] == pytest.approx(0.0597)       # max(0.02, 0.0597, 0.04)


def test_threshold_is_the_max_of_noise_floor_and_t_hpo():
    assert judge(t_hpo=0.09)["thr1"] == pytest.approx(0.09) and judge(noise=0.12)["thr1"] == pytest.approx(0.12)
    j = judge(t_hpo=0.09)
    assert j["c1"] and j["delta_ic"] == pytest.approx(0.11)               # ΔIC 0.11 > 0.09 · thr3 ยังเป็น 0.0597 (ไม่รวม T_hpo)
    assert j["thr3"] == pytest.approx(0.0597)
    assert not judge(t_hpo=0.12)["c1"] and not judge(t_hpo=0.12)["has_winner"]


def test_criterion_2_uses_new_seeds_and_same_threshold():
    weak = {"base": 0.0, "hpo09": 0.03}
    j = judge(confirm_ic=weak)
    assert j["c1"] and not j["c2"] and not j["has_winner"] and j["delta_ic_confirm"] == pytest.approx(0.03)
    assert not judge(confirm=False)["c2"]                                 # ไม่ได้ยืนยัน = ไม่ผ่าน


def test_criterion_3_ic_positive_and_trimmed_mean():
    # หนึ่ง evaluation ติดลบได้ (5/6) แต่ 2 ติดลบไม่ได้
    one_neg = {"base": 0.0, "hpo09": [0.30, 0.30, 0.30, 0.30, 0.30, -0.01]}
    assert judge(one_neg)["c3"] and judge(one_neg)["n_ic_positive"] == 5
    two_neg = {"base": 0.0, "hpo09": [0.40, 0.40, 0.40, 0.40, -0.01, -0.01]}
    assert not judge(two_neg)["c3"]
    # ΔIC ที่เหลือหลังตัด evaluation สูงสุดต้อง > max(noise, 0.0597): ค่าเฉลี่ยรวมสูงเพราะ evaluation เดียว
    driven = {"base": 0.0, "hpo09": [0.60, 0.01, 0.01, 0.01, 0.01, 0.01]}
    j = judge(driven)
    assert j["c1"] and not j["c3"] and j["max_delta_eval"] == "KBANK-1"
    assert j["delta_mean_trimmed"] == pytest.approx(0.01) and not j["has_winner"]


def test_criterion_4_relmae_not_higher_than_base():
    assert judge(relmae={"base": 1.0100, "hpo09": 1.0100})["c4"]          # เท่ากัน = ไม่สูงกว่า
    j = judge(relmae={"base": 1.0100, "hpo09": 1.0101})
    assert not j["c4"] and not j["has_winner"]


def test_only_rank_one_candidate_is_checked():
    ic = {"base": 0.0, "hpo09": [0.03] * 6, "hpo11": [0.20] * 6}           # hpo11 ผ่านทุกข้อ แต่ hpo09 ไม่ใช่อันดับ 1 — ไม่เกี่ยว
    assert judge(ic)["top"] == "hpo11"
    ic2 = {"base": 0.0, "hpo09": [0.08] * 6, "hpo11": [0.07] * 6}           # อันดับ 1 = hpo09 · ถ้าตก ห้ามลอง hpo11 แม้ hpo11 จะผ่านเกณฑ์อื่น
    j = judge(ic2, t_hpo=0.09)
    assert j["top"] == "hpo09" and not j["has_winner"]


def test_base_overall_top_means_no_winner():
    ic = {"base": 0.10, "hpo02": [0.05] * 6}
    j = judge(ic)
    assert j["base_is_overall_top"] and j["top"] == "hpo02" and j["delta_ic"] < 0 and not j["has_winner"]


def test_epoch_cap_counts_and_level_means():
    h = {"base|KBANK|fold1": [{"stopped_early": True}, {"stopped_early": False}, {"stopped_early": True}],
         "base|ADVANC|fold1": [{"stopped_early": True}] * 3, "hpo01|KBANK|fold1": [{"stopped_early": False}] * 3}
    assert hpo.epoch_cap_counts(h) == {"base": (1, 6), "hpo01": (3, 3)}
    table = hpo.configs_table(hpo.design_configs()[0])
    ic = pd.Series({n: (1.0 if n == "base" else float(i)) for i, n in enumerate(NAMES)})
    lv = hpo.level_means(table, ic)
    assert lv["n_config"].groupby(lv["param"]).sum().eq(24).all()          # แต่ละปัจจัยครอบ 24 ผู้สมัครพอดี (ไม่นับ base)
    for p, grp in lv.groupby("param", sort=False):                         # ค่าเฉลี่ยถ่วงน้ำหนักของระดับ = ค่าเฉลี่ยรวมของ 24 ผู้สมัคร
        assert np.average(grp["mean_IC"], weights=grp["n_config"]) == pytest.approx(np.mean([float(i) for i in range(1, 25)]))
