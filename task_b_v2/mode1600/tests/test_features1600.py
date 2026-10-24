"""
test_features1600.py — feature 4 ตัว / target / replay ของโหมด 16:00 (PLAN ข้อ a, d, h)
====================================================================================
* นิยาม feature ตรงสูตร (คิดมือบนข้อมูลสังเคราะห์)
* leak: แท่ง 16:00 / แถว t ของไฟล์รายวัน / ทุกอย่างหลัง 15:00 ของวัน t → feature ของวัน t (และก่อนหน้า) ไม่เปลี่ยน
* vol15: ตัวหารใช้แค่วันก่อนหน้า · วันไม่ครบตัดทั้งวัน ไม่ ffill
* ข้อมูลจริง (feature เท่านั้น ไม่มี y): feature จาก snapshot = feature จากไฟล์ล็อก บนวันที่แท่งตรงกัน · กฎเลือก snapshot ของ (ii)
"""

import numpy as np
import pandas as pd
import pytest

from mode1600 import data1600 as D
from mode1600 import features1600 as FE
from mode1600.locked1600 import FEATURES, VOL_LOOKBACK
from synth1600 import make_bars, make_daily

COLS = list(FEATURES) + ["C15", "prev_close", "vol_base", "usable"]


@pytest.fixture(scope="module")
def syn():
    daily = make_daily()
    bars = make_bars(daily)
    return daily, bars


def _day(bars, i):
    return sorted(bars["date"].unique())[i]


def test_feature_formulas(syn):
    daily, bars = syn
    t = FE.feature_table(bars, daily)
    d = _day(bars, 60)
    b = bars[bars["date"] == d].set_index("hour")
    prev = daily["Close"][daily.index < d].iloc[-1]
    hrs = [10, 11, 12, 14, 15]
    assert t.loc[d, "r15"] == pytest.approx(b.loc[15, "Close"] / prev - 1)
    assert t.loc[d, "gap"] == pytest.approx(b.loc[10, "Open"] / prev - 1)
    assert t.loc[d, "range15"] == pytest.approx((b.loc[hrs, "High"].max() - b.loc[hrs, "Low"].min()) / prev)
    days = sorted(bars["date"].unique())
    k = days.index(d)
    vs = lambda x: bars[(bars["date"] == x) & bars["hour"].isin([11, 12, 14, 15])]["Volume"].sum()   # noqa: E731
    base = np.mean([vs(x) for x in days[k - VOL_LOOKBACK:k]])
    assert t.loc[d, "vol15"] == pytest.approx(np.log(vs(d) / base))


def test_bar13_and_v10_not_used(syn):
    daily, bars = syn
    p = bars.copy()
    p.loc[p["hour"] == 13, ["High", "Low", "Volume"]] = [1e6, 1e-6, 1e12]
    p.loc[p["hour"] == 10, "Volume"] = 7e9
    pd.testing.assert_frame_equal(FE.feature_table(bars, daily)[COLS], FE.feature_table(p, daily)[COLS])


def test_warmup_and_usable(syn):
    daily, bars = syn
    t = FE.feature_table(bars, daily)
    assert not t["usable"].iloc[:VOL_LOOKBACK].any()
    assert t["usable"].iloc[VOL_LOOKBACK:].all()


def test_incomplete_day_dropped_not_filled(syn):
    daily, bars = syn
    d = _day(bars, 50)
    p = bars[~((bars["date"] == d) & (bars["hour"] == 14))]
    t = FE.feature_table(p, daily)
    assert not t.loc[d, "usable"] and t.loc[[d], list(FEATURES)].isna().all(axis=None)
    # วันถัดไป: ตัวหาร vol15 ข้ามวันที่ถูกตัด (20 eligible day ก่อนหน้า) ไม่ใช่ ffill
    days = sorted(p["date"].unique())
    nxt = days[days.index(d) + 1]
    elig = [x for x in days[:days.index(nxt)] if x != d][-VOL_LOOKBACK:]
    vs = lambda x: p[(p["date"] == x) & p["hour"].isin([11, 12, 14, 15])]["Volume"].sum()   # noqa: E731
    assert t.loc[nxt, "vol_base"] == pytest.approx(np.mean([vs(x) for x in elig]))


# ------------------------------------------------------------------ leak (PLAN ข้อ h 1–3, 6)
def _poison_after_15(bars, daily, d, rng):
    """แท่ง ≥ 16 ของวัน d + ทุกแท่งของวัน > d + ทุกแถวรายวัน ≥ d → สุ่มใหม่"""
    b, dl = bars.copy(), daily.copy()
    m = ((b["date"] == d) & (b["hour"] >= 16)) | (b["date"] > d)
    b.loc[m, ["Open", "High", "Low", "Close", "AdjClose"]] = rng.uniform(1, 1000, (int(m.sum()), 5))
    b.loc[m, "Volume"] = rng.uniform(0, 1e8, int(m.sum()))
    md = dl.index >= d
    dl.loc[md, ["Open", "High", "Low", "Close"]] = rng.uniform(1, 1000, (int(md.sum()), 4))
    return b, dl


def test_leak_bar16(syn):
    daily, bars = syn
    p = bars.copy()
    p.loc[p["hour"] == 16, ["Open", "High", "Low", "Close", "Volume"]] = -5.0
    pd.testing.assert_frame_equal(FE.feature_table(bars, daily)[COLS], FE.feature_table(p, daily)[COLS])


def test_leak_daily_row_t(syn):
    daily, bars = syn
    d = _day(bars, 100)
    dl = daily.copy()
    dl.loc[dl.index >= d, "Close"] = 1e6
    a, b = FE.feature_table(bars, daily), FE.feature_table(bars, dl)
    pd.testing.assert_frame_equal(a.loc[:d, COLS], b.loc[:d, COLS])
    # target ของวัน t เปลี่ยนได้ (เป็น target ไม่ใช่ feature)
    assert FE.attach_target(a, daily).loc[d, "y"] != FE.attach_target(b, dl).loc[d, "y"]


def test_leak_everything_after_15(syn):
    daily, bars = syn
    rng = np.random.default_rng(3)
    for i in (30, 100, 200):
        d = _day(bars, i)
        b, dl = _poison_after_15(bars, daily, d, rng)
        pd.testing.assert_frame_equal(FE.feature_table(bars, daily).loc[:d, COLS], FE.feature_table(b, dl).loc[:d, COLS])


def test_vol_denominator_uses_only_previous_days(syn):
    daily, bars = syn
    d = _day(bars, 80)
    p = bars.copy()
    p.loc[(p["date"] == d) & (p["hour"] == 12), "Volume"] *= 50
    a, b = FE.feature_table(bars, daily), FE.feature_table(p, daily)
    assert a.loc[d, "vol_base"] == b.loc[d, "vol_base"]
    assert a.loc[d, "vol15"] != b.loc[d, "vol15"]


# ------------------------------------------------------------------ replay (PLAN ข้อ d, h4)
def test_splice_uses_history_on_overlap(syn):
    daily, bars = syn
    days = sorted(bars["date"].unique())
    hist = bars[bars["date"] <= days[150]]
    snap = bars[bars["date"] >= days[120]].copy()
    snap.loc[snap["date"] <= days[150], "Close"] = -1.0          # ฉบับต่างบนช่วงทับ → ต้องไม่ถูกใช้
    s = FE.splice(hist, snap)
    pd.testing.assert_frame_equal(s[s["date"] <= days[150]].reset_index(drop=True), hist.reset_index(drop=True))
    assert (s[s["date"] > days[150]]["Close"] > 0).all()


def test_snapshot_rule_latest_in_window(syn):
    daily, bars = syn
    d = _day(bars, 210)
    snaps = pd.DataFrame({"path": ["a", "b", "c", "e"],
                          "downloaded_at": pd.to_datetime([f"{d.date()} 15:59:00+07:00", f"{d.date()} 16:19:00+07:00",
                                                           f"{d.date()} 16:22:00+07:00", f"{d.date()} 16:42:00+07:00"])})
    full = bars[bars["date"] <= d]
    sb = {"a": full, "b": full, "c": full, "e": full}
    assert FE.select_snapshot_ii(snaps, d, sb)["path"] == "c"
    sb["c"] = full[~((full["date"] == d) & (full["hour"] == 15))]          # ล่าสุดไม่มีแท่ง 15 → ใช้ตัวก่อนหน้า
    assert FE.select_snapshot_ii(snaps, d, sb)["path"] == "b"
    assert FE.select_snapshot_ii(snaps.iloc[[0, 3]], d, sb) is None


needs_team = pytest.mark.skipif(not D.HIST_DIR.exists(), reason="ไม่มี repo ทีมในเครื่อง")


@needs_team
@pytest.mark.parametrize("ticker", D.TICKERS)
def test_real_snapshot_features_equal_history_where_bars_equal(ticker):
    """
    ข้อมูลจริง · feature เท่านั้น (ไม่คำนวณ y): ทุก snapshot — แทนแท่งของไฟล์ล็อกด้วยแท่งของ snapshot บนช่วงที่ทับ
    แล้ว feature ต้องเท่ากับของไฟล์ล็อก ในทุกวันที่แท่งของวันนั้นและ 20 eligible day ก่อนหน้าตรงกันทุกค่า
    """
    hist, daily = D.load_history(ticker), D.load_daily(ticker)
    ref = FE.feature_table(hist, daily)
    n_checked = 0
    for p in D.snapshot_index(ticker)["path"]:
        snap = D.load_snapshot(p)
        days = sorted(set(snap["date"]) & set(hist["date"]))
        mixed = pd.concat([hist[~hist["date"].isin(days)], snap[snap["date"].isin(days)]]).sort_values("ts")
        got = FE.feature_table(mixed, daily)
        a = D.day_table(hist).loc[days]
        b = D.day_table(snap).loc[days]
        same = (a.drop(columns=["missing"]).fillna(-1) == b.drop(columns=["missing"]).fillna(-1)).all(axis=1)
        elig = ref.index[ref["complete"]]
        for d in days:
            i = elig.get_loc(d) if d in elig else None
            if i is None or i < VOL_LOOKBACK:
                continue
            window = elig[i - VOL_LOOKBACK:i + 1]
            if all(same.get(x, True) for x in window):
                pd.testing.assert_series_equal(got.loc[d, COLS], ref.loc[d, COLS], check_names=False)
                n_checked += 1
    assert n_checked > 250       # สำเนาใน data/intraday/ มี 11 snapshot ต่อหุ้น (repo ทีมมี 15) → วันที่เทียบได้ลดจาก >300 เหลือ ~280 (B41)


@needs_team
def test_real_snapshot_rule_picks_1622_on_0930():
    """กฎ (ii) บนข้อมูลจริง: 2026-09-30 มี 16:19 และ 16:22 → ต้องได้ 16:22 (M13) · ตรวจแค่ชื่อไฟล์ ไม่ดูค่าใด ๆ"""
    snaps = D.snapshot_index("KBANK")
    sb = {p: D.load_snapshot(p) for p in snaps["path"]}
    r = FE.select_snapshot_ii(snaps, "2026-09-30", sb)
    assert r["downloaded_at"].strftime("%H:%M") == "16:22"
    assert FE.select_snapshot_ii(snaps, "2026-10-21", sb) is None
