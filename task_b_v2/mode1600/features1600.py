"""
features1600.py — feature 4 ตัว + target + eligible day + ตาราง replay (i)/(ii) ของโหมด 16:00 (PLAN ข้อ a, b, d)
=========================================================================================================
ไฟล์นี้ทำอะไร
    1. `feature_table(bars, daily)` : 1 แถวต่อวันที่มีแท่ง → r15, gap, range15, vol15 + C15, ฐานราคาปิด (t−1), ธง usable
    2. `attach_target(table, daily)` : y = Close_official(t)/C15(t) − 1 (อ่านแถว t ของไฟล์รายวัน **เป็น target เท่านั้น**)
    3. `history_table(ticker)`      : ตารางของไฟล์ล็อก 730d (ใช้เทรน/ประเมิน walk-forward)
    4. `replay_table_i(ticker)`     : replay ทาง (i) — แท่งจาก snapshot 2026-10-23 03:55 (ผลหลัก 17 วัน · M13)
       `replay_table_ii(ticker)`    : replay ทาง (ii) — แท่งที่รู้จริง ณ 16:00–16:30 (snapshot ล่าสุดในช่วงนั้น · ผลเสริม)
       ทั้งสองตารางมีคอลัมน์ provenance: bar_source · snapshot_file · snapshot_downloaded_at · snapshot_sha256

ทำไมต้องมี — ความ causal ถูกบังคับที่นี่ไฟล์เดียว
    * แท่ง: อ่านผ่าน `data1600.day_table` ซึ่งทิ้งแท่ง ≥ 16:00 เป็นคำสั่งแรก → feature ไม่มีทางเห็นแท่ง 16
    * รายวัน: ฐานราคา (t−1) หาด้วย `searchsorted(t) − 1` = แถวสุดท้ายที่วันที่ **< t** → ไม่มีทางเห็นแถว t
    * vol15: ตัวหาร = ค่าเฉลี่ยของ 20 eligible day **ก่อนหน้า** (shift 1 ก่อน rolling) · วันไม่ครบไม่เข้าหน้าต่าง · ไม่ ffill
    leak test อยู่ใน `tests/test_features1600.py`

นิยาม (PLAN ข้อ a · M3, M4, M5)
    r15     = C15 / Close_official(t−1) − 1
    gap     = O10 / Close_official(t−1) − 1
    range15 = (max H ของแท่ง 10,11,12,14,15 − min L ของแท่งเดียวกัน) / Close_official(t−1)      (ไม่รวมแท่ง 13)
    vol15   = ln( (V11+V12+V14+V15) / ค่าเฉลี่ยของปริมาณเดียวกันใน 20 eligible day ก่อนหน้า )   (ไม่ใช้ V10, V13)
    t−1     = แถวก่อนหน้าในไฟล์รายวัน (วันทำการก่อนหน้าจริง · D2 ของทีม)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mode1600 import data1600 as D
from mode1600.locked1600 import (FEATURES, RANGE_HOURS, REPLAY_END, REPLAY_FINAL_SNAPSHOT, REPLAY_START,
                                 SNAPSHOT_WINDOW, VOL_HOURS, VOL_LOOKBACK)

FINAL_SOURCE = "final_20261023_0355"


def feature_table(bars, daily):
    """
    bars  : แท่งรูปแบบ `data1600.read_bars` (แท่ง 16 อยู่ในนี้ได้ — จะถูกทิ้งก่อนใช้)
    daily : ราคารายวัน (index = วันที่ · คอลัมน์ Close)
    คืน DataFrame (index = วันที่ของแท่ง) คอลัมน์:
        complete · prev_date · prev_close · O10 · C15 · r15 · gap · range15 · vsum · vol_base · vol15 · usable
    usable = แท่งบังคับครบ · มีฐานราคา t−1 · มีประวัติ volume 20 eligible day · feature ครบและ finite
    """
    dt = D.day_table(bars)                                         # แท่ง 10..15 เท่านั้น
    dates = dt.index
    didx = daily.index
    prev_pos = didx.searchsorted(dates, side="left") - 1           # แถวสุดท้ายที่วันที่ < t
    has_prev = prev_pos >= 0
    close = daily["Close"].to_numpy(dtype=float)
    prev_close = np.where(has_prev, close[np.clip(prev_pos, 0, None)], np.nan)
    prev_date = pd.DatetimeIndex(np.where(has_prev, didx[np.clip(prev_pos, 0, None)], pd.NaT))
    assert (prev_date[has_prev] < dates[has_prev]).all(), "ฐานราคาต้องมาจากวันก่อน t เท่านั้น"

    complete = dt["complete"].to_numpy(dtype=bool)
    hi = dt[[f"H{h}" for h in RANGE_HOURS]].max(axis=1, skipna=False).to_numpy(dtype=float)
    lo = dt[[f"L{h}" for h in RANGE_HOURS]].min(axis=1, skipna=False).to_numpy(dtype=float)
    vsum = dt[[f"V{h}" for h in VOL_HOURS]].sum(axis=1, min_count=len(VOL_HOURS)).to_numpy(dtype=float)

    # ตัวหารของ vol15: เฉลี่ย 20 eligible day ก่อนหน้า (shift 1 = ไม่รวมวันนี้) · วันไม่ครบไม่อยู่ในอนุกรมนี้
    s = pd.Series(vsum[complete], index=dates[complete])
    base = s.shift(1).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean().reindex(dates).to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "complete": complete,
            "prev_date": prev_date,
            "prev_close": prev_close,
            "O10": dt["O10"].to_numpy(dtype=float),
            "C15": dt["C15"].to_numpy(dtype=float),
            "r15": dt["C15"].to_numpy(dtype=float) / prev_close - 1,
            "gap": dt["O10"].to_numpy(dtype=float) / prev_close - 1,
            "range15": (hi - lo) / prev_close,
            "vsum": vsum,
            "vol_base": base,
            "vol15": np.log(vsum / base),
        }, index=dates)
    feats = out[list(FEATURES)].to_numpy(dtype=float)
    out["usable"] = complete & has_prev & np.isfinite(feats).all(axis=1) & (vsum > 0)
    out.loc[~complete, list(FEATURES)] = np.nan                    # วันไม่ครบ: ไม่มี feature (ไม่ ffill)
    return out


def attach_target(table, daily, c15_col="C15"):
    """
    เพิ่ม close (= Close_official(t)) · y = close/C15 − 1 · labeled = usable และมี y
    ใช้แถว t ของไฟล์รายวันเป็น **target เท่านั้น** (ไม่ถูกส่งต่อเป็น feature)
    """
    out = table.copy()
    out["close"] = daily["Close"].reindex(out.index).to_numpy(dtype=float)
    out["y"] = out["close"] / out[c15_col] - 1
    out["labeled"] = out["usable"] & np.isfinite(out["y"].to_numpy(dtype=float))
    return out


def history_table(ticker):
    """ไฟล์ล็อก 730d (≤ 2026-09-25) + ราคารายวันของ next_day → ตารางพร้อม target (bar_source = hist_730d)"""
    t = attach_target(feature_table(D.load_history(ticker), D.load_daily(ticker)), D.load_daily(ticker))
    t["bar_source"] = "hist_730d"
    return t


# ------------------------------------------------------------------ replay
def replay_dates(daily):
    """วันทำการของ replay จากปฏิทินไฟล์รายวัน (17 วัน)"""
    return daily.index[(daily.index >= pd.Timestamp(REPLAY_START)) & (daily.index <= pd.Timestamp(REPLAY_END))]


def splice(hist, snap, upto=None):
    """
    ต่อแท่ง: ไฟล์ล็อกสำหรับวัน ≤ วันสุดท้ายของไฟล์ล็อก + snapshot สำหรับวันหลังจากนั้น (≤ upto ถ้าระบุ)
    (วันที่ทับกันใช้ไฟล์ล็อก = ฉบับเดียวกับข้อมูลเทรน · PLAN ข้อ d)
    """
    last = hist["date"].max()
    s = snap[snap["date"] > last]
    if upto is not None:
        s = s[s["date"] <= pd.Timestamp(upto)]
    return pd.concat([hist, s], ignore_index=True).sort_values("ts").reset_index(drop=True)


def _snapshot_meta(path, downloaded_at, sha):
    return {"snapshot_file": Path(path).parent.name + "/" + Path(path).name,
            "snapshot_downloaded_at": str(downloaded_at), "snapshot_sha256": sha}


def replay_table_i(ticker, hist=None, daily=None, snaps=None, snap_bars=None):
    """
    ทาง (i) · ผลหลัก: แท่งของทุกวัน replay จาก snapshot เดียว 2026-10-23 03:55 (ฉบับเดียวกับไฟล์ล็อกบนช่วงที่ทับ)
    คืนตารางเฉพาะวัน replay + target + provenance
    """
    hist = D.load_history(ticker) if hist is None else hist
    daily = D.load_daily(ticker) if daily is None else daily
    snaps = D.snapshot_index(ticker) if snaps is None else snaps
    rel = REPLAY_FINAL_SNAPSHOT.format(ticker=ticker)
    row = snaps[snaps["path"].map(lambda p: Path(p).parent.name + "/" + Path(p).name) == rel]
    if len(row) != 1:
        raise FileNotFoundError(f"ไม่พบ snapshot ของทาง (i): {rel}")
    r = row.iloc[0]
    sb = D.load_snapshot(r["path"]) if snap_bars is None else snap_bars[r["path"]]
    t = attach_target(feature_table(splice(hist, sb), daily), daily)
    t = t.reindex(replay_dates(daily))
    t["bar_source"] = FINAL_SOURCE
    for k, v in _snapshot_meta(r["path"], r["downloaded_at"], r["meta_sha"]).items():
        t[k] = v
    t["labeled"] = t["labeled"].eq(True)
    return t


def select_snapshot_ii(snaps, day, snap_bars):
    """
    กฎของทาง (ii) (M13 · กฎทีม): snapshot **ล่าสุด** ที่ดาวน์โหลดในช่วง 16:00–16:30 ของวันนั้น และมีแท่งบังคับของวันนั้นครบ
    คืนแถวของ snapshot_index หรือ None
    """
    day = pd.Timestamp(day)
    a = day + pd.Timedelta(SNAPSHOT_WINDOW[0] + ":00")
    b = day + pd.Timedelta(SNAPSHOT_WINDOW[1] + ":00")
    dl = snaps["downloaded_at"].map(lambda x: x.tz_localize(None) if pd.notna(x) else pd.NaT)
    cand = snaps[(dl >= a) & (dl <= b)].sort_values("downloaded_at", ascending=False)
    for _, r in cand.iterrows():
        dt = D.day_table(snap_bars[r["path"]])
        if day in dt.index and bool(dt.loc[day, "complete"]):
            return r
    return None


def replay_table_ii(ticker, hist=None, daily=None, snaps=None, snap_bars=None, final_table=None):
    """
    ทาง (ii) · ผลเสริม: วัน d ใช้แท่งจาก snapshot ที่เลือกตามกฎ (วันหลังไฟล์ล็อกถึง d มาจาก snapshot นั้น = สิ่งที่ระบบเห็นจริง)
    target หลักของ (ii) = Close/C15_snapshot − 1 (สิ่งที่ระบบรู้จริง · R2 ข้อ 18) · เพิ่ม c15_final / y_final_base เป็นตารางเสริม
    วันที่ไม่มี snapshot ในช่วง → แถวอยู่ แต่ labeled = False และ bar_source = "none"
    """
    hist = D.load_history(ticker) if hist is None else hist
    daily = D.load_daily(ticker) if daily is None else daily
    snaps = D.snapshot_index(ticker) if snaps is None else snaps
    snap_bars = {p: D.load_snapshot(p) for p in snaps["path"]} if snap_bars is None else snap_bars
    if final_table is None:
        final_table = replay_table_i(ticker, hist, daily, snaps, snap_bars)
    rows = []
    for day in replay_dates(daily):
        r = select_snapshot_ii(snaps, day, snap_bars)
        if r is None:
            rows.append(pd.DataFrame({"bar_source": ["none"], "labeled": [False], "usable": [False]}, index=[day]))
            continue
        ft = attach_target(feature_table(splice(hist, snap_bars[r["path"]], upto=day), daily), daily)
        one = ft.loc[[day]].copy()
        dl = r["downloaded_at"]
        one["bar_source"] = f"snapshot_1600_1630@{dl.strftime('%H:%M')}"
        for k, v in _snapshot_meta(r["path"], dl, r["meta_sha"]).items():
            one[k] = v
        rows.append(one)
    t = pd.concat(rows)
    t.index.name = None
    t["c15_final"] = final_table["C15"].reindex(t.index)
    t["y_final_base"] = t["close"] / t["c15_final"] - 1 if "close" in t else np.nan
    t["labeled"] = t["labeled"].eq(True)
    return t
