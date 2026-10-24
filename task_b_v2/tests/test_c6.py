"""
test_c6.py — โค้ด C6 (src/run_c6.py) ต้องทำตาม LOCK.md หัวข้อ 9 โดยไม่แตะข้อมูลหลัง 2025-02-25 ระหว่างทดสอบ (DECISIONS B41)
===========================================================================================================================
ไฟล์นี้ทำอะไร (ทุก test ใช้ข้อมูลสังเคราะห์ หรือโซน dev ≤ 2025-02-25 เท่านั้น)
    1. track ทุกตัวตรงกับตารางใน LOCK.md หัวข้อ 5 · รายการงานครบ (final · replay · S1) · S1 มีแค่ track 1 + RF
    2. Split → Fold: train จบก่อนวันแรกของช่วงประเมินพอดี · S1 เริ่ม train ที่ 2021-01-01 · ช่วงประเมินต่อกัน
    3. leak: เปลี่ยนราคาตั้งแต่วัน t → input (ภาพ · 29 feature) และสถิติที่ fit ของแถว ≤ t ไม่เปลี่ยน (y ของวัน t เปลี่ยน)
    4. consistency: จำลอง pipeline final บน dev (เทรนถึงก่อน fold 3 → ทาย fold 3 · seeds 0–2) → คำทำนาย/c ตรง Stage 3 ทุกบิต
    5. F1 bootstrap · F2 · ตาราง log replay · ตัววัด/ธงคำทำนายคงที่ · gate (git ปลอม) · ไฟล์ผลไม่นับ cache/
ทำไมต้องมี: C6 รันได้ครั้งเดียว · บั๊กที่เจอหลังรัน = ต้องประกาศรอบใหม่ (LOCK.md ข้อ (c)) จึงต้องจับให้ได้ก่อน commit
"""

import json
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import data, paths, run_c6 as C
from src.configs import BASE
from src.settings import EARLY_STOP_FRAC, VAL_END

TICKER = "KBANK"


@pytest.fixture(scope="module")
def ds():
    return data.build_dataset(TICKER, VAL_END)            # dev เท่านั้น


# ---------------------------------------------------------------
# 1. track และรายการงาน
# ---------------------------------------------------------------
def lock_section5():
    """ตาราง config ใน LOCK.md หัวข้อ 5 → {field: [ค่าของ track 1, 2, 3, 4, control]} (อ่านอย่างเดียว)"""
    text = (paths.TASK_DIR / "LOCK.md").read_text(encoding="utf-8")
    sec = text[text.index("## 5."):text.index("## 6.")]
    out = {}
    for line in sec.splitlines():
        m = re.match(r"\| `(\w+)` \| (.+) \|$", line.strip())
        if m:
            out[m.group(1)] = [v.strip().strip("`") for v in m.group(2).split(" | ")]
    return out


def test_tracks_match_lock_md_section5_every_field():
    table = lock_section5()
    assert len(table) >= 20
    keys = ["track1", "track2", "track3", "track4", "control"]
    for field, vals in table.items():
        for key, want in zip(keys, vals):
            v = getattr(C.TRACKS[key], field)
            got = "−".join(map(str, v)) if isinstance(v, tuple) else str(v)
            assert got == want, (key, field, got, want)
    assert C.TRACKS["track1"] is BASE


def test_plan_units_covers_lock_section9():
    tasks = C.plan_units()
    nn = {(ph, key, t) for kind, ph, key, t, _ in tasks if kind == "nn"}
    assert {(ph, k) for ph, k, _ in nn} == {(ph, k) for ph in ("final", "replay") for k in C.TRACKS} | {("s1", "track1")}
    leg = [(ph, tuple(key)) for kind, ph, key, t, _ in tasks if kind == "legacy"]
    assert ("s1", ("Random Forest",)) in leg and ("final", tuple(C.LEGACY)) in leg and ("replay", tuple(C.LEGACY)) in leg
    assert len(tasks) == 2 * 5 * 2 + 2 + 3 * 2 and all(s == tuple(range(10)) for kind, *_, s in tasks if kind == "nn")


def test_splits_are_the_locked_periods():
    f, r, s = C.SPLITS["final"], C.SPLITS["replay"], C.SPLITS["s1"]
    assert (f.eval_start, f.eval_end, f.train_end, f.train_start) == ("2025-02-26", "2026-08-28", "2025-02-25", None)
    assert (r.eval_start, r.eval_end, r.train_end) == ("2026-09-28", "2026-10-22", "2026-09-25")
    assert (s.train_start, s.eval_start, s.eval_end, s.train_end) == ("2021-01-01", "2025-02-26", "2026-08-28", "2025-02-25")


# ---------------------------------------------------------------
# 2. Split → Fold (บน dev)
# ---------------------------------------------------------------
def test_dev_split_reproduces_fold3(ds):
    f3 = data.make_folds(ds.dates)[-1]
    fold = C.make_fold(ds, C.dev_split(ds))
    np.testing.assert_array_equal(fold.eval_pos, f3.eval_pos)
    assert fold.train_stop == f3.train_stop and fold.train_start == 0
    usable = ds.usable(BASE)
    np.testing.assert_array_equal(fold.train_pos(usable), f3.train_pos(usable))


def test_train_start_and_end_are_honoured(ds):
    sp = replace(C.dev_split(ds), name="s1_dev", train_start="2021-01-01")
    fold = C.make_fold(ds, sp)
    tr = fold.train_pos(ds.usable(BASE))
    assert ds.dates[tr[0]] >= pd.Timestamp("2021-01-01") and ds.dates[tr[0] - 1] < pd.Timestamp("2021-01-01")
    assert tr[-1] == fold.train_stop - 1 < fold.eval_pos[0]
    with pytest.raises(C.StopRun):
        C.make_fold(ds, replace(sp, train_end="2020-01-02"))             # วันสุดท้ายของ train ไม่ตรงที่คาด → หยุด


# ---------------------------------------------------------------
# 3. leak
# ---------------------------------------------------------------
@pytest.mark.parametrize("key", ["track1", "track2", "track4"])
def test_future_prices_do_not_change_inputs_of_day_t(ds, key):
    from tests.test_leak import _perturbed_prices
    sp = C.dev_split(ds)
    cfg = C.TRACKS[key]
    fold = C.make_fold(ds, sp)
    pos = int(fold.eval_pos[40])                                        # วัน t ในช่วงประเมิน
    alt = data.dataset_from_prices(TICKER, _perturbed_prices(ds.prices, pos, seed=3), VAL_END)
    a, b = data.fold_inputs(ds, cfg, fold), data.fold_inputs(alt, cfg, C.make_fold(alt, sp))
    k = 40                                                              # แถวที่ 0..40 ของช่วงประเมิน = วัน ≤ t
    np.testing.assert_array_equal(a["X_eval"][:k + 1], b["X_eval"][:k + 1])
    np.testing.assert_array_equal(a["X_fit"], b["X_fit"])
    assert a["transform"].scale == b["transform"].scale
    assert not np.isclose(a["y_eval"][k], b["y_eval"][k]), "y ของวัน t ต้องเปลี่ยน — ไม่งั้น test ไม่ได้ทดสอบอะไร"
    if cfg.needs_features:
        from src import train
        ea, eb = train.standardized_extra(ds, a), train.standardized_extra(alt, b)
        np.testing.assert_array_equal(ea["eval"][:k + 1], eb["eval"][:k + 1])
        np.testing.assert_array_equal(ea["fit"], eb["fit"])


# ---------------------------------------------------------------
# 4. consistency กับ Stage 3 (ทุกบิต)
# ---------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("key,config", [("track1", "base"), ("track2", "hybrid")])
def test_simulated_final_on_dev_matches_stage3_bitwise(ds, key, config):
    frame, info = C.nn_unit(C.dev_split(ds), key, TICKER, (0, 1, 2))
    ref = pd.read_csv(paths.RESULTS_DIR / "stage3_input" / "predictions.csv", float_precision="round_trip", parse_dates=["date"])
    ref = ref[(ref["config"] == config) & (ref["ticker"] == TICKER) & (ref["fold"] == 3)].sort_values("date")
    got = frame.sort_values("date")
    np.testing.assert_array_equal(got["date"].to_numpy(), ref["date"].to_numpy())
    np.testing.assert_array_equal(got["pred"].to_numpy(), ref["pred"].to_numpy())
    np.testing.assert_array_equal(got["pred_shrunk"].to_numpy(), ref["pred_shrunk"].to_numpy())
    ev = pd.read_csv(paths.RESULTS_DIR / "stage3_input" / "evals.csv", float_precision="round_trip")
    assert info["c"] == float(ev[(ev["config"] == config) & (ev["ticker"] == TICKER) & (ev["fold"] == 3)]["c_chosen"].iloc[0])
    assert info["train_last"] < VAL_END and info["n_hit_epoch_cap"] == 0


@pytest.mark.slow
def test_legacy_unit_on_dev_matches_c3_comparators(ds):
    frame, _ = C.legacy_unit(C.dev_split(ds), TICKER, ["Naive", "Mean Return", "Random Forest"])
    ref = pd.read_csv(paths.RESULTS_DIR / "stage0_base" / "comparators_predictions.csv", float_precision="round_trip", parse_dates=["date"])
    for name in ("Naive", "Mean Return", "Random Forest"):
        a = frame[frame["model"] == name].sort_values("date")["pred"].to_numpy()
        b = ref[(ref["model"] == name) & (ref["ticker"] == TICKER) & (ref["fold"] == 3)].sort_values("date")["pred"].to_numpy()
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------
# 5. F1 · F2 · log · ตัววัด · gate (ข้อมูลสังเคราะห์)
# ---------------------------------------------------------------
def synth_preds(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2030-01-01", periods=n)
    rows = []
    for t in ("KBANK", "ADVANC"):
        y = rng.normal(0, 0.01, n)
        y[::7] = 0.0
        for model, p in (("track1", 0.3 * y + rng.normal(0, 0.005, n)), ("track2", 0.5 * y + rng.normal(0, 0.005, n)),
                         ("control", 0.1 * y + rng.normal(0, 0.005, n)), ("track4", rng.normal(0, 0.005, n)),
                         ("Naive", np.zeros(n))):
            rows.append(pd.DataFrame({"phase": "final", "model": model, "date": dates, "ticker": t, "fold": 0, "y": y,
                                      "pred": p, "pred_shrunk": 0.5 * p, "prev_close": 100.0, "close": 100 * (1 + y),
                                      "c": 0.5 if model != "Naive" else np.nan}))
    return pd.concat(rows, ignore_index=True)


def test_bootstrap_matches_definition():
    rng = np.random.default_rng(1)
    y = rng.normal(size=95)
    a = y + rng.normal(size=95)
    b = rng.normal(size=95)
    same = C.bootstrap_delta_ic(y, a, a, reps=200)
    assert same["delta_IC"] == 0 and same["ci_lo"] == 0 and same["ci_hi"] == 0 and not same["supported"]
    r = C.bootstrap_delta_ic(y, a, b, reps=300)
    assert r["n_days"] == 95 and r["block"] == 20 and r["seed"] == 0 and r["ci_lo"] <= r["delta_IC"] <= r["ci_hi"]
    assert r == C.bootstrap_delta_ic(y, a, b, reps=300)                   # seed คงที่ → ผลเดิมทุกบิต
    # คำนวณมือตามนิยาม (95 วัน → บล็อก 20, 20, 20, 20, 15 · สุ่มบล็อกแบบมีคืน 5 บล็อกต่อรอบ)
    blocks = [np.arange(i, min(i + 20, 95)) for i in range(0, 95, 20)]
    assert [len(x) for x in blocks] == [20, 20, 20, 20, 15]
    g = np.random.default_rng(0)
    d = []
    for _ in range(300):
        idx = np.concatenate([blocks[j] for j in g.integers(0, 5, 5)])
        d.append(C.ic(y[idx], a[idx]) - C.ic(y[idx], b[idx]))
    assert r["ci_lo"] == pytest.approx(np.percentile(d, 2.5)) and r["ci_hi"] == pytest.approx(np.percentile(d, 97.5))


def test_f1_and_f2_tables():
    p = synth_preds()
    f1 = C.f1_table(p)
    assert list(f1["ticker"]) == ["KBANK", "ADVANC"] and (f1["delta_IC"] > 0).all()
    m = C.metrics_table(p)
    f2 = C.f2_table(p, m)
    assert list(f2["ticker"]) == ["KBANK", "ADVANC", "รวม"]
    k = m.set_index(["model", "ticker"])
    assert f2.loc[0, "relMAE_track1"] == pytest.approx(k.loc[("track1", "KBANK"), "relMAE"])
    assert f2.loc[2, "relMAE_track1"] == pytest.approx((k.loc[("track1", "KBANK"), "relMAE"] + k.loc[("track1", "ADVANC"), "relMAE"]) / 2)
    e = lambda mod: (p[p["model"] == mod]["pred"] - p[p["model"] == mod]["y"]).abs().mean()  # noqa: E731
    assert f2.loc[2, "MAE_ratio_track1_over_track4"] == pytest.approx(e("track1") / e("track4"))


def test_metrics_flag_constant_predictions():
    p = synth_preds()
    m = C.metrics_table(p).set_index(["model", "ticker"])
    nv = m.loc[("Naive", "KBANK")]
    assert bool(nv["collapsed"]) and nv["IC"] == 0.0 and nv["IC_spearman"] == 0.0            # B41: คงที่ → IC := 0 + ธง
    assert nv["relMAE"] == pytest.approx(1.0) and bool(nv["collapsed_shrunk"])
    assert not bool(m.loc[("track1", "KBANK"), "collapsed"]) and m.loc[("track1", "KBANK"), "IC"] != 0


def test_constant_prediction_rule_in_f1_and_ic():
    """F1 / bootstrap ใช้กติกาเดียวกัน: คำทำนายคงที่ (รวมต่างแค่เศษปัดเศษ) → IC = 0 · ติดธงในตาราง F1"""
    y = np.random.default_rng(2).normal(0, 0.01, 50)
    assert C.ic(y, np.zeros(50)) == 0.0 and C.ic(y, np.full(50, 1e-3) + 1e-20 * y) == 0.0
    assert C.ic(np.zeros(50), y) != C.ic(np.zeros(50), y)                                   # y คงที่ → NaN
    p = synth_preds()
    p.loc[p["model"] == "control", "pred"] = 0.0
    f1 = C.f1_table(p)
    assert f1["collapsed_b"].all() and not f1["collapsed_a"].any()
    assert (f1["IC_b"] == 0).all() and np.allclose(f1["delta_IC"], f1["IC_a"])
    m = C.metrics_table(p)
    f2 = C.f2_table(p, m)
    assert not f2["collapsed_track1"].any() and not f2["collapsed_track4"].any()


def test_replay_log_columns_direction_and_error():
    p = synth_preds(n=17).assign(phase="replay")
    log = C.replay_log(p, "KBANK")
    assert list(log.columns) == ["วันที่", "โมเดล", "ราคาปิดวันก่อน", "ราคาที่ทาย", "ราคาจริง", "error บาท", "ทิศถูกหรือไม่", "ป้าย"]
    assert len(log) == 17 * 5 and (log["ป้าย"] == "ทำนายย้อนหลังแบบ walk-forward ไม่ใช่ log สด").all()
    g = p[(p["ticker"] == "KBANK")].sort_values(["date", "model"])
    np.testing.assert_allclose(log["error บาท"], (100 * (g["pred"] - g["y"])).round(4), atol=1e-9)
    zero = (g["y"] == 0).to_numpy()
    assert (log["ทิศถูกหรือไม่"].to_numpy()[zero] == "—").all()
    assert set(log["ทิศถูกหรือไม่"].to_numpy()[~zero]) <= {"ถูก", "ผิด"}
    s = C.replay_mae_summary(p)
    n = s.set_index(["ticker", "model"]).loc[("KBANK", "Naive")]
    assert n["n_direction_correct"] == 0 and n["n_direction_counted"] == int((~zero).sum() / 5) and bool(n["collapsed"])


class FakeGit:
    """จำลอง git สำหรับ gate: status · ls-files · log · rev-parse"""

    def __init__(self, status="", tracked=(C.LOCK_MD, C.LOCK_1600), lock_commit="a" * 40, commits_1600=("b" * 40,)):
        self.status, self.tracked, self.lock_commit, self.commits_1600 = status, set(tracked), lock_commit, commits_1600

    def __call__(self, *args):
        a = [x for x in args if x not in ("-C",)]
        if a[0] == "rev-parse" and "--show-toplevel" in a:
            return 0, "ROOT"
        a = a[1:] if a[0] == "ROOT" else a
        if a[0] == "status":
            return 0, self.status
        if a[0] == "ls-files":
            return (0, a[-1]) if a[-1] in self.tracked else (1, "")
        if a[0] == "log":
            return (0, self.lock_commit) if a[-1] == C.LOCK_MD else (0, "\n".join(self.commits_1600))
        if a[0] == "rev-parse":
            return 0, "c" * 40
        raise AssertionError(a)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    task = tmp_path / "task_b_v2"
    (task / "mode1600").mkdir(parents=True)
    (task / "mode1600" / "LOCK_1600.md").write_text("x\nNEXTDAY_LOCK_COMMIT: " + "a" * 40 + "\n", encoding="utf-8")
    monkeypatch.setattr(paths, "TASK_DIR", task)
    out = {k: tmp_path / "results" / k for k in ("final", "replay", "s1")}
    return out


def test_gate_passes_when_clean_and_locked(fake_repo, monkeypatch):
    monkeypatch.setattr(C, "_git", FakeGit())
    g = C.gate(fake_repo)
    assert g["lock_commit"] == "a" * 40 and g["head_commit"] == "c" * 40 and g["run_time"]


@pytest.mark.parametrize("fake,msg", [(FakeGit(status=" M task_b_v2/src/x.py"), "git status ไม่ว่าง"),
                                      (FakeGit(tracked=(C.LOCK_MD,)), "ยังไม่ถูก commit"),
                                      (FakeGit(commits_1600=("b" * 40, "d" * 40)), "ถูกแตะโดย 2 commit"),
                                      (FakeGit(lock_commit="e" * 40), "LOCK.md ถูกแก้หลัง LOCK_1600 อ้าง")])
def test_gate_refuses(fake_repo, monkeypatch, fake, msg):
    monkeypatch.setattr(C, "_git", fake)
    with pytest.raises(C.StopRun, match=msg):
        C.gate(fake_repo)


def test_gate_refuses_existing_results_but_ignores_cache(fake_repo, monkeypatch):
    monkeypatch.setattr(C, "_git", FakeGit())
    (fake_repo["replay"] / "cache").mkdir(parents=True)
    (fake_repo["replay"] / "cache" / "x.pkl").write_bytes(b"0")
    C.gate(fake_repo)                                                     # cache อย่างเดียว → ผ่าน
    (fake_repo["final"]).mkdir(parents=True)
    (fake_repo["final"] / "metrics.csv").write_text("x", encoding="utf-8")
    with pytest.raises(C.StopRun, match="มีผลอยู่แล้ว"):
        C.gate(fake_repo)


def test_run_refuses_before_loading_any_data(monkeypatch):
    """cmd_run ต้องเรียก gate เป็นคำสั่งแรก — gate ไม่ผ่านแล้วต้องไม่มีการโหลดข้อมูล/เทรน"""
    calls = []
    monkeypatch.setattr(C, "gate", lambda *a, **k: (_ for _ in ()).throw(C.StopRun("no")))
    monkeypatch.setattr(C, "run_units", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(data, "build_dataset", lambda *a, **k: calls.append(2))
    with pytest.raises(C.StopRun):
        C.cmd_run(type("A", (), {"jobs": 1})())
    assert calls == []
