"""
test_runner1600.py — wrapper ของโหมด 16:00 (PLAN ข้อ c, e, f, g, h) · เทรนด้วย **ข้อมูลสังเคราะห์เท่านั้น**
======================================================================================================
* track_configs: track 1 = BASE จาก src · ช่องว่าง → ปฏิเสธ · หลัก = hybrid · control = hybrid_nocnn (ค่าอื่นเท่ากัน)
* wrapper ส่ง n_extra = 4 · หลักมี CNN + linear skip จาก 4 feature · control ไม่มี CNN
* leak: เปลี่ยนทุกอย่างหลัง 15:00 ของวัน t (แท่ง 16, แท่งวันหลัง, รายวันแถว ≥ t) → คำทำนายของวัน t ไม่เปลี่ยน ทุก track
* walk-forward: train < วันแรกของบล็อก · บล็อก 20 วันทำการต่อกัน · replay พก provenance
* F3 bootstrap / summarize / DM บนคำทำนายปลอม (ไม่เทรน)
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

from mode1600 import features1600 as FE
from mode1600 import locked1600 as L
from mode1600 import runner1600 as R
from synth1600 import make_bars, make_daily, tiny_track1

SEEDS1 = (0,)


@pytest.fixture(scope="module")
def syn():
    daily = make_daily()
    bars = make_bars(daily)
    table = FE.attach_target(FE.feature_table(bars, daily), daily)
    return daily, bars, table


@pytest.fixture(scope="module")
def cfgs():
    return L.track_configs(tiny_track1())


def _lab(table):
    return table.index[table["labeled"].to_numpy(dtype=bool)]


# ------------------------------------------------------------------ config
def test_track1_is_base_object_from_src(monkeypatch):
    """M22: track 1 = BASE ที่ import จาก src/configs.py (วัตถุเดียวกัน ไม่ได้พิมพ์ค่าเอง) · ช่องว่าง → ปฏิเสธ"""
    from src import configs
    assert L.TRACK1_CONFIG is configs.BASE
    assert L.track_configs()[L.MAIN].ann_input == "hybrid"
    monkeypatch.setattr(L, "TRACK1_CONFIG", None)
    with pytest.raises(RuntimeError, match="TRACK1_CONFIG"):
        L.track_configs()


def test_track_configs_only_change_ann_input(cfgs):
    t1 = tiny_track1()
    assert cfgs[L.MAIN].ann_input == "hybrid" and cfgs[L.CONTROL].ann_input == "hybrid_nocnn"
    for c in cfgs.values():
        assert replace(c, name=t1.name, ann_input=t1.ann_input) == t1
    with pytest.raises(ValueError):
        L.track_configs(replace(t1, train_mode="two_stage"))
    with pytest.raises(ValueError):
        L.track_configs(replace(t1, ann_input="hybrid"))


# ------------------------------------------------------------------ โมเดลที่ wrapper สร้าง
def test_n_extra_4_main_has_cnn_and_skip_control_has_no_cnn(syn, cfgs, monkeypatch):
    from src import models
    daily, _, table = syn
    built = []
    orig = models.build

    def spy(cfg, n_extra=0):
        m = orig(cfg, n_extra)
        built.append((cfg.name, n_extra, m))
        return m

    monkeypatch.setattr(models, "build", spy)
    lab = _lab(table)
    R.run_split("SYN", daily, table, cfgs, train_before=lab[150], eval_dates=lab[150:155], zone_end=lab[154], seeds=SEEDS1)
    by = {name: (n, m) for name, n, m in built}
    n_main, main = by[L.MAIN]
    n_ctrl, ctrl = by[L.CONTROL]
    assert n_main == 4 and n_ctrl == 4
    assert main.use_extra and main.skip.in_features == 4
    assert any(isinstance(x, nn.Conv2d) for x in main.modules())
    assert not any(isinstance(x, (nn.Conv1d, nn.Conv2d)) for x in ctrl.modules())
    assert not hasattr(ctrl, "cnn")
    assert ctrl.skip.in_features == 4


# ------------------------------------------------------------------ leak ระดับคำทำนาย (PLAN ข้อ h 1–3)
def _scale_after(daily, bars, d, rng):
    """ทุกอย่างหลัง 15:00 ของวัน d เปลี่ยน (สเกลราคาแบบสุ่มต่อแถว ให้ OHLC ยังสมเหตุผล · volume สุ่ม)"""
    dl, b = daily.copy(), bars.copy()
    md = dl.index >= d
    f = rng.uniform(0.5, 2.0, int(md.sum()))
    dl.loc[md, ["Open", "High", "Low", "Close"]] = dl.loc[md, ["Open", "High", "Low", "Close"]].mul(f, axis=0)
    dl.loc[md, "Volume"] = rng.uniform(1e5, 1e7, int(md.sum()))
    mb = ((b["date"] == d) & (b["hour"] >= 16)) | (b["date"] > d)
    g = rng.uniform(0.5, 2.0, int(mb.sum()))
    b.loc[mb, ["Open", "High", "Low", "Close", "AdjClose"]] = b.loc[mb, ["Open", "High", "Low", "Close", "AdjClose"]].mul(g, axis=0)
    b.loc[mb, "Volume"] = rng.uniform(1e4, 1e7, int(mb.sum()))
    return dl, b


@pytest.mark.parametrize("k", [120, 170])
def test_leak_predictions_unchanged_after_1500(syn, cfgs, k):
    daily, bars, table = syn
    lab = _lab(table)
    d, ev = lab[k], lab[k:k + 6]
    base, _ = R.run_split("SYN", daily, table, cfgs, train_before=d, eval_dates=ev, zone_end=ev[-1], seeds=SEEDS1)
    dl, b = _scale_after(daily, bars, d, np.random.default_rng(k))
    t2 = FE.attach_target(FE.feature_table(b, dl), dl)
    assert _lab(t2)[k] == d
    pois, _ = R.run_split("SYN", dl, t2, cfgs, train_before=d, eval_dates=_lab(t2)[k:k + 6], zone_end=ev[-1], seeds=SEEDS1)
    a = base[base["date"] == d].set_index("model")
    p = pois[pois["date"] == d].set_index("model")
    assert set(a.index) == set(L.ALL_TRACKS)
    for col in ("pred", "pred_shrunk", "prev_close"):
        pd.testing.assert_series_equal(a[col], p.loc[a.index, col])
    assert a.loc[L.MAIN, "y"] != p.loc[L.MAIN, "y"]          # target ของวัน t เปลี่ยน (ยืนยันว่าวางยาจริง)


# ------------------------------------------------------------------ walk-forward / replay
@pytest.mark.parametrize("n_extra_days", [0, 1, 7, 19])
def test_wf_blocks_contiguous_20_days_remainder_merged(n_extra_days):
    """M20 (ข): บล็อกละ 20 วัน · เศษรวมเข้าบล็อกสุดท้าย (ไม่มีบล็อกสั้นกว่า 20)"""
    days = pd.bdate_range("2024-10-01", periods=5 * L.RETRAIN_EVERY + n_extra_days)
    bl = R.wf_blocks(days, start=days[0], end=days[-1])
    assert bl["start"].iloc[0] == days[0] and bl["end"].iloc[-1] == days[-1]
    sizes = [((days >= r.start) & (days <= r.end)).sum() for r in bl.itertuples()]
    assert len(bl) == 5 and all(s == L.RETRAIN_EVERY for s in sizes[:-1]) and sizes[-1] == L.RETRAIN_EVERY + n_extra_days
    assert sum(sizes) == len(days)


def test_wf_blocks_real_calendar_23_blocks_last_21():
    """ปฏิทินจริง (ไฟล์รายวัน · นับวันเท่านั้น): 461 วัน → 23 บล็อก · บล็อกท้าย 2026-07-30 → 2026-08-28 (21 วัน)"""
    from mode1600 import data1600 as D
    cal = D.load_daily("KBANK").index
    bl = R.wf_blocks(cal)
    days = cal[(cal >= pd.Timestamp(L.WF_START)) & (cal <= pd.Timestamp(L.WF_END))]
    assert len(days) == 461 and len(bl) == 23
    last = bl.iloc[-1]
    assert str(last["end"].date()) == L.WF_END and ((days >= last["start"]) & (days <= last["end"])).sum() == 21


def test_walk_forward_train_strictly_before_block(syn, cfgs):
    daily, _, table = syn
    lab = _lab(table)
    blocks = R.wf_blocks(daily.index, start=lab[140], end=lab[179], every=20)      # 40 วัน → 2 บล็อก
    preds, info = R.run_walk_forward({"SYN": (daily, table)}, cfgs, blocks, seeds=SEEDS1)
    assert len(info) == 2
    for r in info.itertuples():
        assert pd.Timestamp(r.train_last) < pd.Timestamp(r.eval_first)
    assert set(preds["model"]) == set(L.ALL_TRACKS)
    assert preds.groupby("model").size().nunique() == 1


def test_replay_carries_provenance_and_same_training(syn, cfgs):
    daily, _, table = syn
    lab = _lab(table)
    train_end, rep = lab[160], lab[161:166]
    ti = table.loc[rep].copy()
    ti["bar_source"] = "final_x"
    tii = table.loc[rep[:3]].copy()
    tii["bar_source"] = "snapshot_1600_1630@16:22"
    hist = table[table.index <= train_end]
    p, info = R.run_replay("SYN", daily, hist, {"i": ti, "ii": tii}, cfgs, seeds=SEEDS1,
                           train_end=train_end, start=rep[0], zone_end=rep[-1])
    assert set(p["tag"]) == {"replay_i", "replay_ii"}
    assert (p[p["tag"] == "replay_ii"]["bar_source"] == "snapshot_1600_1630@16:22").all()
    a = p[(p["tag"] == "replay_i") & p["date"].isin(rep[:3])].set_index(["model", "date"])["pred"]
    b = p[p["tag"] == "replay_ii"].set_index(["model", "date"])["pred"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())          # input เดียวกัน → โมเดลเดียวกัน → คำทำนายเดียวกัน
    assert (info["train_last"] == str(train_end.date())).all()


# ------------------------------------------------------------------ ตัววัด / F3 (ไม่เทรน)
def _fake_preds(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-10-01", periods=n)
    frames = []
    for t in ("A", "B"):
        y = rng.normal(0, 0.003, n)
        y[rng.random(n) < 0.4] = 0.0
        preds = {L.MAIN: 0.5 * y + rng.normal(0, 0.002, n), L.CONTROL: rng.normal(0, 0.002, n),
                 L.RF: rng.normal(0, 0.001, n), L.NAIVE: np.zeros(n), L.MEAN: np.full(n, 1e-4)}
        for m, p in preds.items():
            frames.append(pd.DataFrame({"date": dates, "ticker": t, "fold": np.arange(n) // 20 + 1, "y": y, "pred": p,
                                        "prev_close": 100.0, "model": m, "tag": "walk_forward", "pred_shrunk": 0.5 * p}))
    return pd.concat(frames, ignore_index=True)


def test_bootstrap_deterministic_and_sane():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    a, b = y + rng.normal(size=200), rng.normal(size=200)
    r1 = R.bootstrap_delta_ic(y, a, b, reps=500)
    assert r1 == R.bootstrap_delta_ic(y, a, b, reps=500)
    assert r1["supported"] and r1["ci_lo"] > 0
    same = R.bootstrap_delta_ic(y, a, a, reps=200)
    assert same["delta_IC"] == 0 and same["ci_lo"] == 0 and same["ci_hi"] == 0 and not same["supported"]
    assert R.ic(y, np.full(200, 0.1)) == 0.0                                  # คำทำนายคงที่ → IC := 0 (B39)


def test_summarize_and_f3_shapes():
    p = _fake_preds()
    s = R.summarize(p)
    assert set(s["period"]) == {"all", "2024-10-01..2025-02-25", "2025-02-26..2026-08-28"}
    allp = s[s["period"] == "all"]
    assert len(allp) == 2 * len(L.ALL_TRACKS)
    assert allp[allp["model"] == L.NAIVE]["relMAE"].eq(1.0).all()
    assert allp[allp["model"] != L.NAIVE]["p_DM"].notna().all()
    assert allp[allp["model"].isin(L.NN_TRACKS)]["relMAE_shrunk"].notna().all()
    f = R.f3(p, reps=300)
    assert set(f[f["period"] == "all"]["ticker"]) == {"A", "B"}
    assert {"delta_IC", "ci_lo", "ci_hi", "supported"} <= set(f.columns)
