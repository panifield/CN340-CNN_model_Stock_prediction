"""
data1600.py — โหลดข้อมูลของโหมด 16:00 (อ่านอย่างเดียว · ตรวจ SHA-256 ทุกไฟล์)
==========================================================================
ไฟล์นี้ทำอะไร
    1. `load_history(ticker)`   : แท่ง 1 ชั่วโมงของ Yahoo จากไฟล์ล็อกของทีม (`*_1h_730d.csv` · สำเนาใน `data/intraday/`) + ตรวจ SHA ตาม README ทีม
    2. `snapshot_index(ticker)` / `load_snapshot(path)` : snapshot สดใน `raw_data_intraday_live/<วันที่>/`
                                  + ตรวจ SHA ตาม `.meta.json` + เวลาที่ดาวน์โหลด
    3. `day_table(bars)`        : ตารางแท่งต่อวัน (1 แถวต่อวัน) ของแท่ง 10..15 **เท่านั้น**
    4. `known_at_1600(bars, d)` : แท่งของวัน d ที่ "รู้แล้ว ณ 16:00" = แท่ง 10..15 เท่านั้น
    5. `load_daily(ticker)`     : ราคารายวันไฟล์เดียวกับ next_day (`task_b_v2/data/`) + ตรวจ SHA ตาม LOCK.md
       `daily_before(daily, d)` : แถวรายวันที่วันที่ < d เท่านั้น (ห้ามอ่านแถว t เป็น feature)

ทำไมต้องมี
    * **ที่เดียวที่รู้เรื่องเวลาตัด 16:00** — label ของแท่ง = เวลา "เริ่ม" ของแท่ง (ยืนยันใน AUDIT.md ข้อ 1)
      แท่ง 15:00 = 15:00–16:00 จบพอดี 16:00 · แท่ง 16:00 = 16:00–ปิดตลาด ยังไม่เกิด ณ 16:00
      → ทุกฟังก์ชันที่คืน "แท่ง" ตัดแท่งที่เริ่ม ≥ 16:00 ทิ้งเป็นคำสั่งแรก (ไม่มีทางคืนแท่ง 16:00)
    * ไฟล์ของทีมห้ามแก้ — โมดูลนี้อ่านอย่างเดียวและปฏิเสธไฟล์ที่ SHA ไม่ตรง

ข้อตกลงที่ตามทีม (`task_b/INTRADAY_FINDINGS.md`, `PHASE1D_PLAN_R2.md`)
    * แท่งบังคับ = 10, 11, 12, 14, 15 · แท่ง 13 ไม่บังคับ (มีไม่สม่ำเสมอ) · วันไม่ครบ = ตัดทั้งวัน ห้าม ffill
    * Volume ของแท่ง 10:00 ห้ามใช้ (เป็น 0 ใน ~3/4 ของวัน) — การเลือกคอลัมน์อยู่ที่ตัวสร้าง feature ไม่ใช่ที่นี่
"""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

MODE_DIR = Path(__file__).resolve().parent
TASK_DIR = MODE_DIR.parent                                   # task_b_v2/
# ค่าเริ่มต้น = สำเนา byte-for-byte ใน repo นี้ `task_b_v2/data/intraday/` (โครงโฟลเดอร์เดียวกับ task_b/ ของทีม · ที่มาและ SHA ใน data/SOURCES.md)
# ทางเลือก: ชี้ไป clone ของ repo ทีมด้วย env MODE1600_TEAM_TASK_B (เช่น .../StockPrediction2/task_b) — SHA ต้องตรงเหมือนเดิม
DEFAULT_INTRADAY_DIR = TASK_DIR / "data" / "intraday"
TEAM_TASK_B = Path(os.environ.get("MODE1600_TEAM_TASK_B", DEFAULT_INTRADAY_DIR))
HIST_DIR = TEAM_TASK_B / "raw_data_intraday"
LIVE_DIR = TEAM_TASK_B / "raw_data_intraday_live"
DAILY_DIR = TASK_DIR / "data"

TICKERS = ("KBANK", "ADVANC")
TZ_OFFSET = "+07:00"                                         # เวลาไทย (Asia/Bangkok ไม่มี DST)

# SHA-256 ตาม task_b/raw_data_intraday/README.md ของทีม
HIST_SHA = {
    "KBANK": "b192ad2c0bc3ae2a75b732752369c56b7dd27d043900eb6e8fb2d5fb08912060",
    "ADVANC": "2097bb53ce1c6ad49023f432128b39363187f4fdaebcb12bbd5ee7e5a295e79c",
}
# SHA-256 ตาม task_b_v2/LOCK.md หัวข้อ 1 (ไฟล์เดียวกับ next_day)
DAILY_SHA = {
    "KBANK": "97fa5be2b1bfc2afd9da49f74f70360adb2412ca84d90c397874f4540a278082",
    "ADVANC": "4a245577ea0178c5edecf28c3e747f06059841f4ebd49e2a830213ebc8387c1d",
}

REQUIRED_HOURS = (10, 11, 12, 14, 15)                        # แท่งที่ต้องมีครบ (ตามทีม)
KNOWN_HOURS = (10, 11, 12, 13, 14, 15)                       # แท่งที่จบไม่เกิน 16:00 (13 อาจไม่มี)
LAST_KNOWN_HOUR = 15                                         # แท่งสุดท้ายที่รู้ ณ 16:00
PRICE_COLS = ("Open", "High", "Low", "Close", "Volume")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha(path, expected):
    """ปฏิเสธไฟล์ที่ SHA-256 ไม่ตรง (ไฟล์ของทีมห้ามแก้ — ถ้าไม่ตรงแปลว่าข้อมูลเปลี่ยน ต้องหยุดถามผู้ใช้)"""
    got = sha256_file(path)
    if got != expected:
        raise ValueError(f"SHA-256 ของ {Path(path).name} ไม่ตรง: ได้ {got[:16]}… คาด {expected[:16]}…")
    return got


def read_bars(path, expected_sha=None):
    """
    อ่านไฟล์แท่ง 1h ของ Yahoo → DataFrame เรียงเวลา
    คอลัมน์: ts (tz +07:00), date (วันที่ไม่มี tz), hour, Open, High, Low, Close, AdjClose, Volume
    ตรวจ: ทุก timestamp ลงท้าย +07:00 · นาที = 0 · ไม่มี timestamp ซ้ำ
    """
    if expected_sha is not None:
        check_sha(path, expected_sha)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"Datetime": str})
    bad = ~raw["Datetime"].str.endswith(TZ_OFFSET)
    if bad.any():
        raise ValueError(f"{Path(path).name}: มี timestamp ที่ไม่ใช่ {TZ_OFFSET} {int(bad.sum())} แถว")
    ts = pd.to_datetime(raw["Datetime"], format="%Y-%m-%d %H:%M:%S%z")
    if (ts.dt.minute != 0).any() or (ts.dt.second != 0).any():
        raise ValueError(f"{Path(path).name}: มีแท่งที่ไม่ได้เริ่มต้นชั่วโมงพอดี")
    df = pd.DataFrame({
        "ts": ts,
        "date": ts.dt.tz_localize(None).dt.normalize(),
        "hour": ts.dt.hour.astype(int),
        "Open": pd.to_numeric(raw["Open"]), "High": pd.to_numeric(raw["High"]),
        "Low": pd.to_numeric(raw["Low"]), "Close": pd.to_numeric(raw["Close"]),
        "AdjClose": pd.to_numeric(raw["Adj Close"]), "Volume": pd.to_numeric(raw["Volume"]),
    }).sort_values("ts").reset_index(drop=True)
    if df["ts"].duplicated().any():
        raise ValueError(f"{Path(path).name}: มี timestamp ซ้ำ")
    return df


def history_path(ticker):
    return HIST_DIR / f"{ticker}_BK_1h_730d.csv"


def load_history(ticker):
    """ไฟล์ล็อกของทีม 2023-10-02 → 2026-09-25 (ตรวจ SHA)"""
    return read_bars(history_path(ticker), HIST_SHA[ticker])


def snapshot_index(ticker, live_dir=None):
    """
    รายการ snapshot สดของหุ้นหนึ่งตัว เรียงตามเวลาดาวน์โหลด
    คอลัมน์: path, folder, downloaded_at (tz +07:00 จาก meta), sha_ok (SHA ของไฟล์ตรง meta)
    """
    live_dir = LIVE_DIR if live_dir is None else Path(live_dir)
    rows = []
    for p in sorted(live_dir.glob(f"*/{ticker}_BK_1h_*.csv")):
        meta_p = p.with_name(p.name + ".meta.json")
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        rows.append({"path": p, "folder": p.parent.name,
                     "downloaded_at": pd.Timestamp(meta["downloaded_at"]) if "downloaded_at" in meta else pd.NaT,
                     "meta_sha": meta.get("sha256"),
                     "sha_ok": meta.get("sha256") == sha256_file(p) if meta else False})
    idx = pd.DataFrame(rows, columns=["path", "folder", "downloaded_at", "meta_sha", "sha_ok"])
    return idx.sort_values("downloaded_at").reset_index(drop=True)


def load_snapshot(path):
    """อ่าน snapshot หนึ่งไฟล์ + ตรวจ SHA ตาม .meta.json (ไม่มี meta = ปฏิเสธ)"""
    path = Path(path)
    meta = json.loads(path.with_name(path.name + ".meta.json").read_text(encoding="utf-8"))
    return read_bars(path, meta["sha256"])


def _drop_after_cutoff(bars):
    """คำสั่งแรกของทุกฟังก์ชันที่คืนแท่ง: ทิ้งแท่งที่เริ่ม ≥ 16:00 (และก่อน 10:00 ถ้ามี)"""
    return bars[bars["hour"].isin(KNOWN_HOURS)]


def known_at_1600(bars, date):
    """แท่งของวัน `date` ที่รู้แล้ว ณ 16:00 = แท่ง 10..15 เท่านั้น (แท่ง 16:00 ไม่ถูกคืนในทุกกรณี)"""
    bars = _drop_after_cutoff(bars)
    out = bars[bars["date"] == pd.Timestamp(date)].copy()
    assert out["hour"].max() <= LAST_KNOWN_HOUR if len(out) else True
    return out


def day_table(bars):
    """
    ตารางแท่งต่อวันของแท่ง 10..15 (1 แถวต่อวัน · index = date)
    คอลัมน์: O10..O15, H10..H15, L10..L15, C10..C15, V10..V15 (แท่งที่ไม่มี = NaN · ไม่ ffill)
             n_bars (จำนวนแท่ง 10..15 ที่มี) · complete (มีแท่งบังคับ 10,11,12,14,15 ครบ) · missing (รายชื่อแท่งบังคับที่ขาด)
    """
    bars = _drop_after_cutoff(bars)
    wide = bars.pivot(index="date", columns="hour", values=list(PRICE_COLS))
    out = pd.DataFrame(index=wide.index)
    for col, short in zip(PRICE_COLS, "OHLCV"):
        for h in KNOWN_HOURS:
            out[f"{short}{h}"] = wide[(col, h)] if (col, h) in wide.columns else float("nan")
    present = out[[f"C{h}" for h in KNOWN_HOURS]].notna()
    out["n_bars"] = present.sum(axis=1)
    req = out[[f"C{h}" for h in REQUIRED_HOURS]].notna()
    out["complete"] = req.all(axis=1)
    out["missing"] = req.apply(lambda r: ",".join(str(h) for h, ok in zip(REQUIRED_HOURS, r) if not ok), axis=1)
    assert not any(c.endswith("16") for c in out.columns), "day_table ต้องไม่มีแท่ง 16:00"
    return out


def daily_path(ticker):
    return DAILY_DIR / f"{ticker}_10Y_Cleaned.csv"


def load_daily(ticker, check=True):
    """
    ราคารายวันไฟล์เดียวกับ next_day → DataFrame Open/High/Low/Close/Volume (index = วันที่)
    อ่านแบบเดียวกับ src.data.load_prices (test ยืนยันว่าเท่ากัน) แต่ตรวจ SHA ตาม LOCK.md ด้วย
    """
    p = daily_path(ticker)
    if check:
        check_sha(p, DAILY_SHA[ticker])
    raw = pd.read_csv(p)
    df = pd.DataFrame({"Open": pd.to_numeric(raw["Open"]), "High": pd.to_numeric(raw["High"]),
                       "Low": pd.to_numeric(raw["Low"]), "Close": pd.to_numeric(raw["Price"]),
                       "Volume": pd.to_numeric(raw["Vol. ('000)"]) * 1000})
    df.index = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df[df["Close"] > 0]


def daily_before(daily, date):
    """แถวรายวันที่วันที่ < date เท่านั้น — ทางเดียวที่ตัวสร้าง feature ควรเห็นข้อมูลรายวัน (ห้ามเห็นแถว t)"""
    return daily[daily.index < pd.Timestamp(date)]
