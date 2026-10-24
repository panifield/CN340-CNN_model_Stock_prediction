"""
test_data1600.py — ทดสอบตัวโหลดข้อมูลโหมด 16:00 (mode1600/data1600.py)
=====================================================================
สิ่งที่ต้องจริงเสมอ
    * แท่ง 16:00 ไม่ถูกคืนจากฟังก์ชันใดเลย และการเปลี่ยนค่าแท่ง 16:00 ไม่เปลี่ยนผลลัพธ์
    * ไฟล์ที่ SHA ไม่ตรง / timestamp ไม่ใช่ +07:00 ถูกปฏิเสธ
    * วันไม่ครบถูกติดธง ไม่ ffill · daily_before ไม่คืนแถว t
    * (ข้อมูลจริง ถ้ามี repo ทีมในเครื่อง) SHA ตรง · ตัวโหลดรายวันให้ผลเท่ากับ src.data.load_prices
"""

import numpy as np
import pandas as pd
import pytest

from mode1600 import data1600 as D

HEADER = "Datetime,Adj Close,Close,High,Low,Open,Volume\n"


def _bars_csv(path, days=("2024-01-03", "2024-01-04"), hours=(10, 11, 12, 13, 14, 15, 16), offset="+07:00", skip=()):
    """เขียนไฟล์แท่งสังเคราะห์รูปแบบเดียวกับ Yahoo · skip = {(day, hour)} ที่ไม่ต้องเขียน"""
    rng = np.random.default_rng(0)
    lines = [HEADER]
    for d in days:
        for h in hours:
            if (d, h) in skip:
                continue
            o = 100 + rng.integers(0, 10)
            c = o + rng.integers(-2, 3)
            lines.append(f"{d} {h:02d}:00:00{offset},{c},{c},{max(o, c) + 1},{min(o, c) - 1},{o},{rng.integers(0, 10**6)}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _poison_bar16(bars):
    out = bars.copy()
    m = out["hour"] == 16
    out.loc[m, ["Open", "High", "Low", "Close", "AdjClose"]] = 9999.0
    out.loc[m, "Volume"] = 123456789
    return out


# ------------------------------------------------------------------ ข้อมูลสังเคราะห์
def test_reads_and_keeps_bkk_time(tmp_path):
    b = D.read_bars(_bars_csv(tmp_path / "x.csv"))
    assert (b["ts"].map(lambda x: x.utcoffset()) == pd.Timedelta(hours=7)).all()
    assert sorted(b["hour"].unique()) == [10, 11, 12, 13, 14, 15, 16]
    assert b["date"].dt.tz is None


def test_rejects_other_timezone(tmp_path):
    with pytest.raises(ValueError, match=r"\+07:00"):
        D.read_bars(_bars_csv(tmp_path / "x.csv", offset="+00:00"))


def test_rejects_sha_mismatch(tmp_path):
    p = _bars_csv(tmp_path / "x.csv")
    with pytest.raises(ValueError, match="SHA-256"):
        D.read_bars(p, expected_sha="0" * 64)
    assert D.read_bars(p, expected_sha=D.sha256_file(p)) is not None


def test_day_table_never_has_bar16_and_ignores_its_values(tmp_path):
    b = D.read_bars(_bars_csv(tmp_path / "x.csv"))
    t1 = D.day_table(b)
    t2 = D.day_table(_poison_bar16(b))
    assert not [c for c in t1.columns if c.endswith("16")]
    pd.testing.assert_frame_equal(t1, t2)
    assert t1["complete"].all()


def test_known_at_1600_only_bars_10_to_15_of_that_day(tmp_path):
    b = D.read_bars(_bars_csv(tmp_path / "x.csv"))
    k = D.known_at_1600(b, "2024-01-04")
    assert set(k["hour"]) == {10, 11, 12, 13, 14, 15}
    assert (k["date"] == pd.Timestamp("2024-01-04")).all()
    pd.testing.assert_frame_equal(k, D.known_at_1600(_poison_bar16(b), "2024-01-04"))


def test_changing_anything_from_bar16_on_does_not_change_known_bars(tmp_path):
    """แท่ง 16 และแท่งของวันถัดไปทั้งหมด (ข้อมูลหลัง 16:00 ของวัน t) เปลี่ยนแล้ว แท่งที่รู้ ณ 16:00 ของวัน t ต้องเท่าเดิม"""
    b = D.read_bars(_bars_csv(tmp_path / "x.csv"))
    after = (b["date"] > pd.Timestamp("2024-01-03")) | (b["hour"] >= 16)
    p = b.copy()
    p.loc[after, ["Open", "High", "Low", "Close", "Volume"]] = -1.0
    pd.testing.assert_frame_equal(D.known_at_1600(b, "2024-01-03"), D.known_at_1600(p, "2024-01-03"))
    pd.testing.assert_frame_equal(D.day_table(b).loc[["2024-01-03"]], D.day_table(p).loc[["2024-01-03"]])


def test_incomplete_day_flagged_not_filled(tmp_path):
    b = D.read_bars(_bars_csv(tmp_path / "x.csv", skip={("2024-01-04", 14), ("2024-01-03", 13)}))
    t = D.day_table(b)
    assert bool(t.loc["2024-01-03", "complete"])            # แท่ง 13 ไม่บังคับ
    assert np.isnan(t.loc["2024-01-03", "C13"])
    assert not bool(t.loc["2024-01-04", "complete"])
    assert t.loc["2024-01-04", "missing"] == "14"
    assert np.isnan(t.loc["2024-01-04", "C14"])              # ไม่ ffill


def test_day_with_only_bar16_is_absent(tmp_path):
    b = D.read_bars(_bars_csv(tmp_path / "x.csv", skip={("2024-01-04", h) for h in (10, 11, 12, 13, 14, 15)}))
    assert list(D.day_table(b).index) == [pd.Timestamp("2024-01-03")]


def test_daily_before_excludes_day_t():
    daily = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    out = D.daily_before(daily, "2024-01-03")
    assert list(out.index) == [pd.Timestamp("2024-01-02")]


# ------------------------------------------------------------------ ข้อมูลจริง (ข้ามถ้าไม่มี repo ทีม)
needs_team = pytest.mark.skipif(not D.HIST_DIR.exists(), reason="ไม่มี repo ทีมในเครื่อง")


@needs_team
@pytest.mark.parametrize("ticker", D.TICKERS)
def test_real_history_sha_and_hours(ticker):
    h = D.load_history(ticker)
    assert set(h["hour"]) <= {10, 11, 12, 13, 14, 15, 16}
    t = D.day_table(h)
    assert not [c for c in t.columns if c.endswith("16")]
    pd.testing.assert_frame_equal(t, D.day_table(_poison_bar16(h)))


@needs_team
@pytest.mark.parametrize("ticker", D.TICKERS)
def test_real_snapshots_sha_ok(ticker):
    idx = D.snapshot_index(ticker)
    assert len(idx) > 0 and idx["sha_ok"].all()
    assert idx["downloaded_at"].notna().all()


@pytest.mark.parametrize("ticker", D.TICKERS)
def test_daily_loader_matches_next_day_loader(ticker):
    """ราคารายวันต้องเป็นไฟล์และค่าชุดเดียวกับ next_day (src.data.load_prices · import อ่านอย่างเดียว)"""
    from src import data as src_data
    ours = D.load_daily(ticker)
    ref = src_data.load_prices(ticker)
    pd.testing.assert_frame_equal(ours, ref, check_freq=False)
