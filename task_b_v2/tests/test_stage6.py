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


# ---------------------------------------------------------------
# 4. B39 — คำทำนายคงที่ → IC := 0 (เฉพาะ Stage 6 · ค่าเริ่มต้นปิด)
# ---------------------------------------------------------------
from src import evaluate, placebo                                      # noqa: E402

RNG = np.random.default_rng(5)
Y = RNG.normal(0, 0.01, 200)
PREV = np.full(200, 100.0)


def test_default_metrics_are_unchanged_for_constant_predictions():
    m = evaluate.metrics(Y, np.full(200, 0.001), PREV)
    assert np.isnan(m["IC"]) and np.isnan(m["IC_spearman"]) and "collapsed" not in m


@pytest.mark.parametrize("pred", [np.full(200, 0.001),                              # ptp = 0
                                  np.full(200, 0.001) + 1e-20 * RNG.normal(size=200),  # ต่างกันแค่เศษปัดเศษ (StdRatio ~ 1e-18)
                                  np.zeros(200)])
def test_constant_prediction_gets_ic_zero_with_flag(pred):
    m = evaluate.metrics(Y, pred, PREV, const_ic_zero=True)
    assert m["collapsed"] is True and m["IC"] == 0.0 and m["IC_spearman"] == 0.0
    assert np.isfinite(m["relMAE"]) and m["StdRatio"] < 1e-8                        # ตัววัดอื่นไม่ถูกแตะ


def test_varying_prediction_is_not_collapsed_and_ic_is_unchanged():
    pred = 0.3 * Y + RNG.normal(0, 0.01, 200)
    a, b = evaluate.metrics(Y, pred, PREV), evaluate.metrics(Y, pred, PREV, const_ic_zero=True)
    assert b["collapsed"] is False and a["IC"] == b["IC"] and a["IC_spearman"] == b["IC_spearman"] and a["IC"] != 0
    # ขอบของ tolerance: อัตราส่วน 2e-8 (> 1e-8) ยังไม่ถือว่าคงที่ · 5e-9 (< 1e-8) ถือว่าคงที่
    base = np.full(200, 0.001)
    above = base + 2e-8 * np.std(Y) * (Y - Y.mean()) / np.std(Y)
    below = base + 5e-9 * np.std(Y) * (Y - Y.mean()) / np.std(Y)
    assert evaluate.metrics(Y, above, PREV, const_ic_zero=True)["collapsed"] is False
    assert evaluate.metrics(Y, below, PREV, const_ic_zero=True)["collapsed"] is True


def test_naive_zero_prediction_still_works_when_flag_off():
    assert evaluate.metrics(Y, np.zeros(200), PREV)["relMAE"] == pytest.approx(1.0)


def collapsed_evals(flags_by_eval):
    """flags_by_eval: list (6) ของ (collapsed ระดับ evaluation, [collapsed ต่อ seed 0..2])"""
    rows = []
    for i, (c, seeds) in enumerate(flags_by_eval):
        rows.append({"config": "x", "ticker": "KBANK" if i < 3 else "ADVANC", "fold": i % 3 + 1, "collapsed": c,
                     **{f"collapsed_s{k}": s for k, s in enumerate(seeds)}})
    return pd.DataFrame(rows)


def test_collapse_counts_by_config_and_level():
    ev = pd.concat([collapsed_evals([(True, [True] * 3), (False, [True, False, False])] + [(False, [False] * 3)] * 4),
                    collapsed_evals([(False, [False] * 3)] * 6).assign(config="y")], ignore_index=True)
    c = hpo.collapse_counts(ev)
    assert c.loc["x"].to_dict() == {"seed_runs": 4, "seed_runs_total": 18, "evals": 1, "evals_total": 6}
    assert c.loc["y"].to_dict() == {"seed_runs": 0, "seed_runs_total": 18, "evals": 0, "evals_total": 6}
    with pytest.raises(RuntimeError):                                              # ไม่ได้รันด้วย flag → หยุด ไม่เดา
        hpo.collapse_counts(ev.drop(columns=["collapsed"]))


def test_common_levels_finds_shared_parameter_values():
    table = hpo.configs_table(hpo.design_configs()[0])
    t = table.set_index("name")
    share = [n for n in CANDS if str(t.loc[n, "weight_decay"]) == "0.01"][:3]       # สมมุติ 3 ตัวนี้ collapse (ล้วน wd 1e-2)
    total = pd.Series({n: 0 for n in NAMES}, dtype=float)
    for n in share:
        total[n] = 10
    common, top = hpo.common_levels(table, total, top_k=5)
    assert sorted(top) == sorted(share)                                            # ตัวที่ count = 0 ไม่ถูกนับเป็น "collapse บ่อย"
    assert common.set_index("param").loc["weight_decay", "พบใน"] == f"{len(share)}/{len(share)}"


def test_score_config_counts_collapse_only_when_columns_exist():
    from src.evaluate import score_config
    rows = []
    for i in range(6):
        rows.append({"pred_finite": True, "relMAE": 1.0, "StdRatio": 0.1, "IC": 0.0 if i == 0 else 0.02, "IC_spearman": 0.0,
                     "relMAE_baht": 1.0, "MAE_baht": 1.0, "RMSE_baht": 1.0, "MAE_return": 0.01, "R2_OOS": 0.0, "DirAcc": 0.5,
                     "ic_s0": 0.0, "ic_s1": 0.02, "ic_s2": 0.02, "collapsed": i == 0,
                     "collapsed_s0": i == 0, "collapsed_s1": False, "collapsed_s2": i < 2})
    s = score_config(rows)
    assert s["n_collapsed_evals"] == 1 and s["n_collapsed_seed_runs"] == 3
    for r in rows:
        r.pop("collapsed"), r.pop("collapsed_s0"), r.pop("collapsed_s1"), r.pop("collapsed_s2")
    assert "n_collapsed_evals" not in score_config(rows)                           # ไม่มีคอลัมน์ = ผลเหมือนเดิมทุกประการ


BASE_COLS = run_stage6.BASE_CHECK_COLS


def base_rows(override=None):
    rows = []
    for t, k in (("KBANK", 1), ("KBANK", 2)):
        r = {"config": "base", "ticker": t, "fold": k, **{c: 0.5 + k for c in BASE_COLS}}
        r["IC_shrunk"] = np.nan if k == 2 else 0.1                                 # c = 0 → NaN ทั้งสองฝั่ง (ที่มาของบั๊กเดิม)
        rows.append(r)
    df = pd.DataFrame(rows)
    for (idx, col), v in (override or {}).items():
        df.loc[idx, col] = v
    return df


def test_base_check_treats_nan_equal_to_nan_but_not_to_finite(tmp_path):
    ref = tmp_path / "ref.csv"
    base_rows().to_csv(ref, index=False)
    ok, d = run_stage6.check_base_matches_stage3(base_rows(), ref_path=ref)
    assert ok and max(d.values()) == 0.0                                           # NaN = NaN (เดิมรายงานว่าไม่ตรง)
    ok, d = run_stage6.check_base_matches_stage3(base_rows({(1, "IC_shrunk"): 0.2}), ref_path=ref)
    assert not ok and d["IC_shrunk"] == np.inf                                     # NaN vs finite = ไม่ตรง
    ok, d = run_stage6.check_base_matches_stage3(base_rows({(0, "IC"): 1.5 + 1e-12}), ref_path=ref)
    assert not ok and d["IC"] > 0                                                  # ต่างแม้เศษเดียว = ไม่ตรง


@pytest.mark.slow
def test_flag_is_plumbed_through_run_unit_and_placebo_unit(tmp_path):
    tiny = replace(BASE, name="tiny", max_epochs=3, patience=3)
    off, _, _ = runner.run_unit(tiny, "KBANK", 3, seeds=(0, 1), cache_dir=tmp_path)
    on, _, _ = runner.run_unit(tiny, "KBANK", 3, seeds=(0, 1), cache_dir=tmp_path, const_ic_zero=True)
    assert off["from_cache"] is False and on["from_cache"] is False                # คีย์ cache ของสองแบบต่างกัน
    assert "collapsed" not in off and {"collapsed", "collapsed_s0", "collapsed_s1"} <= set(on)
    if not on["collapsed"]:
        assert on["IC"] == off["IC"]                                               # ไม่ collapse → ผลเท่าเดิมทุกบิต
    p_off = placebo.placebo_unit((tiny, "KBANK", 3, 0, (0, 1), "2025-02-25", None))
    p_on = placebo.placebo_unit((tiny, "KBANK", 3, 0, (0, 1), "2025-02-25", None, True))
    assert "collapsed" not in p_off and {"collapsed", "collapsed_s0", "collapsed_s1"} <= set(p_on)


# ---------------------------------------------------------------
# 5. รายงานต้อง render ได้กับ 25 config จริง (ไม่ให้พังหลังรันเสร็จ 2.5 ชม.)
# ---------------------------------------------------------------
def synthetic_ctx(tmp_path, monkeypatch, n_rounds=5):
    rng = np.random.default_rng(1)
    configs = hpo.design_configs()[0]
    shuffles = tuple(range(100, 100 + n_rounds))

    def eval_rows(cfg, seeds, shuffle=None, collapse_p=0.1):
        out = []
        for t in ("KBANK", "ADVANC"):
            for k in (1, 2, 3):
                col = rng.random(len(seeds)) < collapse_p
                ics = np.where(col, 0.0, rng.normal(0, 0.03, len(seeds)))
                row = {"model": "m", "config": cfg.name, "ticker": t, "fold": k, "pred_finite": True, "n": 266, "relMAE": 1.0 + rng.normal(0, 0.003),
                       "relMAE_baht": 1.0, "MAE_baht": 1.0, "RMSE_baht": 1.0, "MAE_return": 0.01, "R2_OOS": 0.0, "DirAcc": 0.5,
                       "IC": float(ics.mean()), "IC_spearman": 0.0, "StdRatio": 0.02, "seconds": 5.0, "collapsed": bool(col.all()),
                       "c_chosen": 0.1, "relMAE_shrunk": 1.0, "relMAE_baht_shrunk": 1.0, "MAE_baht_shrunk": 1.0,
                       **{f"ic_s{s}": float(v) for s, v in zip(seeds, ics)}, **{f"collapsed_s{s}": bool(c) for s, c in zip(seeds, col)}}
                if shuffle is not None:
                    row.update(shuffle=shuffle, n_hit_cap=0, epochs_mean=30.0)
                out.append(row)
        return out

    def score(cfgs, seeds):
        ev = pd.DataFrame([r for c in cfgs for r in eval_rows(c, seeds)])
        sc = pd.DataFrame([{"config": c.name, **evaluate.score_config(ev[ev["config"] == c.name].to_dict("records"))} for c in cfgs])
        return ev, sc

    ev, sc = score(configs, (0, 1, 2))
    ev_p = pd.DataFrame([r for s in shuffles for c in configs for r in eval_rows(c, (0, 1, 2), shuffle=s, collapse_p=0.3)])
    top = str(sc[sc["config"] != "base"].sort_values("mean_IC").iloc[-1]["config"])
    c_ev, c_sc = score([BASE, next(c for c in configs if c.name == top)], (10, 11, 12))
    hist = {f"{c.name}|{t}|fold{k}": [{"stopped_early": True}] * 3 for c in configs for t in ("KBANK", "ADVANC") for k in (1, 2, 3)}
    ps = hpo.placebo_stats(ev_p, [c.name for c in configs])
    noise = float(np.sqrt(np.nanmean(sc["ic_seed_std"].to_numpy(float) ** 2)))
    judge = hpo.judge_winner(sc, ev, c_sc, c_ev, noise, ps["t_hpo"])
    monkeypatch.setattr(runner, "EXPERIMENT_LOG", tmp_path / "experiment_log.csv")
    return {"configs": configs, "table": hpo.configs_table(configs), "scores": sc, "evals": ev, "histories": hist, "ev_p": ev_p, "pstat": ps,
            "c_scores": c_sc, "c_evals": c_ev, "c_hist": hist, "judge": judge, "noise": noise, "shuffles": shuffles, "smoke": False,
            "units": None, "out": tmp_path,
            "timing": {"search_wall_min": 1.0, "placebo_wall_min": 2.0, "confirm_wall_min": 0.5}}


def test_full_report_renders_with_25_configs(tmp_path, monkeypatch):
    ctx = synthetic_ctx(tmp_path, monkeypatch)
    run_stage6.write_report(ctx)
    md = (tmp_path / "stage6_summary.md").read_text(encoding="utf-8")
    for section in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7. Collapse", "### 7.1", "### 7.2", "### 7.3"):
        assert section in md, section
    assert "[บังคับ]" in md and md.count("hpo24") >= 2 and "T_hpo" in md
    assert (tmp_path / "stage6_ic_vs_placebo.png").stat().st_size > 5000
    csv = pd.read_csv(tmp_path / "collapse_by_config.csv")
    assert len(csv) == 25 and csv["search_seed_runs_total"].eq(18).all() and csv["placebo_seed_runs_total"].eq(90).all()
    dec = json.loads((tmp_path / "stage6_decision.json").read_text(encoding="utf-8"))
    assert dec["const_ic_zero"] is True and dec["collapse"]["search"]["seed_runs_total"] == 450 and dec["collapse"]["placebo"]["seed_runs_total"] == 2250
