"""
daily_close.py — ราคาปิดทางการรายวันสำหรับ Phase 1D รอบ 2 (PHASE1D_PLAN_R2.md ข้อ 3)
====================================================================================

ใช้เป็น (ก) target  y(t) = ราคาปิดทางการ(t) / C15(t) − 1
        (ข) ฐานของ feature gap = O10(t) / ราคาปิดทางการของ "วันทำการก่อนหน้า" − 1

แหล่งข้อมูล (ตัวเลือก B ที่ผู้ใช้เลือก 2026-10-17):
  ช่วง dev  (<= LOCKED_END)  : raw_data_daily/<TICKER>_1d_locked.csv · ล็อก SHA-256 เหมือน raw_data_intraday/
                               สร้างจาก raw_data/ ครั้งเดียว (tools/make_locked_daily.py)
                               เหตุผล: Yahoo 1d ที่ดึงสด 2026-10-17 มีแถวเสียปี 2024 (Volume=0, ราคาปิดค้าง)
  ช่วง live (>  LOCKED_END)  : snapshot Yahoo 1d (interval=1d, auto_adjust=False) เก็บ byte-for-byte + meta + SHA-256
                               raw_data_daily_live/<YYYYMMDD>/<KBANK_BK>_1d_<HHMMSS>.csv(+.meta.json)
                               ค่าที่ทับกับไฟล์ล็อกใช้ค่าจากไฟล์ล็อก และรายงานแถวที่ไม่ตรง (ไม่ใช่ error)
  แถวที่ Volume <= 0 หรือ Close ว่าง ไม่ถูกใช้ (แถวผี/ยังไม่ backfill ของ Yahoo)

ไม่ import data_loader / ไม่อ่าน raw_data/ ตอนรัน (tools/make_locked_daily.py อ่านครั้งเดียวตอนสร้างไฟล์ล็อก)
"""

import hashlib
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from intraday_1600 import BANGKOK, BASE_DIR

DAILY_DIR = BASE_DIR / "raw_data_daily"
LIVE_DAILY_DIR = BASE_DIR / "raw_data_daily_live"
HOLIDAYS_PATH = BASE_DIR / "set_holidays.txt"
LOCKED_END = pd.Timestamp("2026-09-25")          # วันสุดท้ายของไฟล์ 1h ที่ล็อก = วันสุดท้ายของ dev data
ALLOWED_SOURCES = {"yahoo"}

# SHA-256 ของไฟล์ที่ล็อกไว้ (ตรงกับ raw_data_daily/README.md)
EXPECTED_SHA256 = {
    "KBANK.BK":  "ab244d8ff4f3b3c6dff3329aa4d5e8ae94981eeb57bc5cd64473deebe664ec31",
    "ADVANC.BK": "641dab045cb5f5141bbdefdf41adc066ba4686a6d27d76ab895443a51a0ca800",
}


class DailyCloseError(RuntimeError):
    pass


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _ts(s):
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        raise DailyCloseError(f"เวลา {s!r} ต้องมี timezone")
    return t.tz_convert(BANGKOK)


def locked_path(ticker):
    return DAILY_DIR / f"{ticker.replace('.', '_')}_1d_locked.csv"


# ---------------------------------------------------------------
# ไฟล์ล็อก (dev)
# ---------------------------------------------------------------
def _parse_locked(raw_bytes):
    from io import BytesIO
    df = pd.read_csv(BytesIO(raw_bytes), parse_dates=["Date"], index_col="Date")
    if list(df.columns) != ["Close", "Volume"]:
        raise DailyCloseError(f"คอลัมน์ไฟล์ล็อกต้องเป็น Close,Volume (ได้ {list(df.columns)})")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise DailyCloseError("ไฟล์ล็อก: วันที่ต้องเรียงและไม่ซ้ำ")
    if df.index.max() > LOCKED_END:
        raise DailyCloseError(f"ไฟล์ล็อกมีวันที่เกิน {LOCKED_END.date()}")
    return df


def load_locked(ticker, expected_sha256=None):
    """อ่านไฟล์ล็อก + ตรวจ SHA-256 (ไม่ตรง = หยุด) · คืน DataFrame index=วันที่ คอลัมน์ Close, Volume"""
    path = locked_path(ticker)
    raw = path.read_bytes()
    want = EXPECTED_SHA256.get(ticker) if expected_sha256 is None else expected_sha256
    got = _sha(raw)
    if want and got != want:
        raise DailyCloseError(f"{path.name}: SHA-256 {got} ไม่ตรงค่าที่ล็อกไว้ {want}")
    return _parse_locked(raw)


# ---------------------------------------------------------------
# snapshot (live)
# ---------------------------------------------------------------
def usable(df):
    """แถวที่ใช้ได้: Close ไม่ว่าง > 0 และ Volume > 0 (แถวผี Volume=0 ของ Yahoo ไม่ใช้)"""
    ok = df["Close"].notna() & (df["Close"] > 0) & df["Volume"].fillna(0).gt(0)
    return df[ok]


def prepare_snapshot(df):
    """CSV ของ Yahoo 1d -> DataFrame index=วันที่ (naive, เที่ยงคืน) · ตรวจรูปแบบ"""
    need = ["Date", "Close", "Volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DailyCloseError(f"snapshot ขาดคอลัมน์ {missing}")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    if getattr(out["Date"].dt, "tz", None) is not None:
        out["Date"] = out["Date"].dt.tz_localize(None)
    out["Date"] = out["Date"].dt.normalize()
    if out["Date"].duplicated().any():
        raise DailyCloseError("snapshot มีวันที่ซ้ำ")
    return out.set_index("Date").sort_index()[["Close", "Volume"]]


def save_snapshot(src, ticker, source, downloaded_at=None, root=LIVE_DAILY_DIR):
    """คัดลอกไฟล์ byte-for-byte เข้า raw_data_daily_live/ + เขียน meta · คืน path"""
    src = Path(src)
    if ticker not in EXPECTED_SHA256:
        raise DailyCloseError(f"ไม่รู้จัก ticker {ticker}")
    source = source.strip().lower()
    downloaded_at = _ts(downloaded_at) if downloaded_at else pd.Timestamp(datetime.now(BANGKOK))
    raw = src.read_bytes()
    prepare_snapshot(pd.read_csv(src, encoding="utf-8-sig"))        # parse ได้จริงก่อนเก็บ
    d = Path(root) / downloaded_at.strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{ticker.replace('.', '_')}_1d_{downloaded_at.strftime('%H%M%S')}.csv"
    if dst.exists():
        raise DailyCloseError(f"{dst} มีอยู่แล้ว -- ห้ามเขียนทับ snapshot")
    shutil.copyfile(src, dst)
    meta = {"ticker": ticker, "source": source,
            "downloaded_at": downloaded_at.isoformat(timespec="seconds"),
            "sha256": _sha(raw), "original_name": src.name}
    Path(str(dst) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return dst


def load_snapshot(path):
    """คืน (DataFrame, meta) · ตรวจว่า SHA ของไฟล์ตรง meta"""
    path = Path(path)
    mp = Path(str(path) + ".meta.json")
    if not mp.exists():
        raise DailyCloseError(f"ไม่พบ meta ของ {path.name}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if _sha(path.read_bytes()) != meta["sha256"]:
        raise DailyCloseError(f"{path.name}: SHA-256 ไม่ตรง meta -- ไฟล์ถูกแก้หลังบันทึก")
    meta["downloaded_at"] = _ts(meta["downloaded_at"])
    meta["path"] = path
    return prepare_snapshot(pd.read_csv(path, encoding="utf-8-sig")), meta


def find_snapshots(ticker, root=LIVE_DAILY_DIR):
    """snapshot ทั้งหมดของหุ้นนี้ เรียงตามเวลาดาวน์โหลด"""
    stem = ticker.replace(".", "_")
    out = [load_snapshot(p) for p in sorted(Path(root).glob(f"*/{stem}_1d_*.csv"))]
    return sorted(out, key=lambda x: x[1]["downloaded_at"])


def latest_snapshot_for(ticker, target, root=LIVE_DAILY_DIR):
    """snapshot ล่าสุดที่ดาวน์โหลดในวัน target (สำหรับทำนาย)"""
    target = pd.Timestamp(target).normalize()
    same_day = [s for s in find_snapshots(ticker, root)
                if s[1]["downloaded_at"].tz_localize(None).normalize() == target]
    if not same_day:
        raise DailyCloseError(f"ไม่พบ snapshot ราคาปิดรายวันของ {ticker} ที่ดาวน์โหลดในวัน {target.date()}")
    return same_day[-1]


def validate_for_prediction(meta, target):
    """snapshot ที่ใช้ทำนาย: แหล่งถูก + ดาวน์โหลดในวันเดียวกับ target"""
    if meta["source"] not in ALLOWED_SOURCES:
        raise DailyCloseError(f"แหล่ง '{meta['source']}' ไม่ใช่แหล่งที่อนุญาต {sorted(ALLOWED_SOURCES)}")
    if meta["downloaded_at"].tz_localize(None).normalize() != pd.Timestamp(target).normalize():
        raise DailyCloseError(f"snapshot ดาวน์โหลดคนละวันกับ target ({meta['downloaded_at'].date()})")


# ---------------------------------------------------------------
# merge + guard
# ---------------------------------------------------------------
def merge_daily(locked, snap):
    """
    รวมไฟล์ล็อก + snapshot · วันที่ทับกันใช้ค่าจากไฟล์ล็อก
    คืน (Series ราคาปิดที่ใช้ได้ index=วันที่, mismatches DataFrame, จำนวนแถว snapshot ที่ถูกทิ้ง)
    mismatches = วันที่ทับกันแต่ Close ไม่เท่ากัน (รายงานเท่านั้น)
    """
    snap_ok = usable(snap)
    dropped = int(len(snap) - len(snap_ok))
    both = locked[["Close"]].join(snap_ok[["Close"]], how="inner", lsuffix="_lock", rsuffix="_snap")
    mism = both[both["Close_lock"] != both["Close_snap"]]
    extra = snap_ok[snap_ok.index > locked.index.max()]
    close = pd.concat([locked["Close"], extra["Close"]]).sort_index()
    return close, mism, dropped


def read_holidays(path=HOLIDAYS_PATH):
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("confirmed_through:"):
            out.add(pd.Timestamp(line))
    return out


def previous_trading_day(target, holidays):
    d = pd.Timestamp(target).normalize() - timedelta(days=1)
    while d.dayofweek > 4 or d in holidays:
        d -= timedelta(days=1)
    return d


def require_previous_close(close, target, holidays=None):
    """ราคาปิดของ "วันทำการก่อนหน้า" ต้องมีในข้อมูล ไม่งั้นหยุด (ห้ามทำนายด้วยข้อมูลเก่าเงียบ ๆ)"""
    holidays = read_holidays() if holidays is None else holidays
    prev = previous_trading_day(target, holidays)
    if prev not in close.index:
        last = close.index[close.index < pd.Timestamp(target).normalize()].max()
        raise DailyCloseError(
            f"ไม่มีราคาปิดทางการของวันทำการก่อนหน้า {prev.date()} (ข้อมูลล่าสุดก่อน target = "
            f"{last.date() if pd.notna(last) else 'ไม่มี'}) -- ไม่ทำนาย")
    return prev


def daily_close_for_prediction(ticker, target, snapshot=None, check_previous=True, holidays=None):
    """
    ราคาปิดรายวันที่ใช้ทำนาย target: ไฟล์ล็อก (+ snapshot ถ้ามี)
    คืน (Series ราคาปิด, info) · info = sha256, source, path, mismatches, dropped_rows, last_date
    snapshot = (DataFrame, meta) จาก load_snapshot · ไม่ส่ง = ใช้ไฟล์ล็อกอย่างเดียว (replay ย้อนหลังใน dev)
    check_previous: บังคับว่าต้องมีราคาปิดของวันทำการก่อนหน้า target (live / target เกินไฟล์ล็อก)
    ราคาปิดของวัน target เองและวันหลังจากนั้น "ไม่ถูกตัดที่นี่" -- ผู้เรียก (build_1600_live_feature) ตัดเอง
    """
    locked = load_locked(ticker)
    if snapshot is None:
        close, mism, dropped, info = locked["Close"], locked.iloc[0:0], 0, {
            "sha256": EXPECTED_SHA256.get(ticker, ""), "source": "locked", "path": str(locked_path(ticker))}
    else:
        snap_df, meta = snapshot
        validate_for_prediction(meta, target)
        close, mism, dropped = merge_daily(locked, snap_df)
        info = {"sha256": meta["sha256"], "source": meta["source"], "path": str(meta["path"])}
    if check_previous:
        require_previous_close(close, target, holidays)
    prior = close[close.index < pd.Timestamp(target).normalize()]
    info.update(mismatches=len(mism), dropped_rows=dropped,
                last_date=str(prior.index.max().date()) if len(prior) else "")
    return close, info
