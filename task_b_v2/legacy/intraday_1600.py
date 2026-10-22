"""
intraday_1600.py — Phase 1D dataset / feature / target builder (Mode A)
=======================================================================
ณ 16:00 ของวัน t ทำนาย close_bar16 ของวัน t -- นิยามทั้งหมดอยู่ใน PHASE1D_PLAN.md

    y(t) = C16(t) / C15(t) − 1        predicted_close_bar16 = C15(t) × (1 + ŷ)

*** แหล่งข้อมูล: raw_data_intraday/*.csv (Yahoo) เท่านั้น ***
- ห้าม import data_loader / ห้ามอ่าน raw_data/ (Investing) -- ห้ามผสมสองแหล่ง
- ไม่อ่านแท่ง 13:00 เลย (ไม่สม่ำเสมอ) · ไม่อ่าน Volume ของแท่ง 10:00
- feature ของวัน t ห้ามอ่านค่าใด ๆ ของแท่ง 16:00 วัน t (นั่นคือ target)
- วันที่ขาดแท่งบังคับ -> ตัดทั้งวัน ห้าม ffill
- ไม่มี historical test: ข้อมูลทั้งหมดคือ development ที่เคย probe แล้ว

Time semantics (Asia/Bangkok +07:00) -- working assumption, ยังไม่ยืนยัน:
    timestamp = เวลา "เริ่ม" ของแท่ง · แท่ง 15:00 = 15:00→16:00 · แท่ง 16:00 = 16:00→ปิดตลาด
"""

import hashlib
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INTRADAY_DIR = BASE_DIR / "raw_data_intraday"
BANGKOK = timezone(timedelta(hours=7))

# SHA-256 ของไฟล์ที่ล็อกไว้ (ตรงกับ raw_data_intraday/README.md)
EXPECTED_SHA256 = {
    "KBANK.BK":  "b192ad2c0bc3ae2a75b732752369c56b7dd27d043900eb6e8fb2d5fb08912060",
    "ADVANC.BK": "2097bb53ce1c6ad49023f432128b39363187f4fdaebcb12bbd5ee7e5a295e79c",
}

FEATURE_HOURS = (10, 11, 12, 14, 15)      # แท่งบังคับของวัน live (ไม่มี 13)
TARGET_HOUR = 16
LABELED_HOURS = FEATURE_HOURS + (TARGET_HOUR,)
WARMUP_ROWS = 20                           # จาก relvol (20 วันก่อนหน้า) ครอบคลุม lag 5 แล้ว
RELVOL_WINDOW = 20
TARGET_LAGS = (1, 2, 3, 5)

FEATURE_COLS = (
    ["ret_10_11", "ret_11_12", "ret_12_14", "ret_14_15", "ret_open10_15"]   # returns (5)
    + ["range", "pos", "gap"]                                              # state (3)
    + ["share12", "share14", "share15", "relvol"]                          # volume (4)
    + [f"y_lag{k}" for k in TARGET_LAGS]                                   # target lags (4)
    + [f"dow_{i}" for i in range(5)]                                       # calendar (5)
)
assert len(FEATURE_COLS) == 21


# ---------------------------------------------------------------
# อ่านข้อมูล
# ---------------------------------------------------------------
def intraday_path(ticker):
    return INTRADAY_DIR / f"{ticker.replace('.', '_')}_1h_730d.csv"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bars(path, expected_sha256=None):
    """อ่านไฟล์ Yahoo รายชั่วโมง (utf-8-sig รองรับ BOM) · ตรวจ SHA ถ้าให้ค่าที่คาดไว้"""
    path = Path(path)
    if expected_sha256 is not None:
        got = sha256_file(path)
        if got != expected_sha256:
            raise RuntimeError(f"{path.name}: SHA-256 {got} ไม่ตรงค่าที่ล็อกไว้ {expected_sha256}")
    return prepare_bars(pd.read_csv(path, encoding="utf-8-sig"))


def load_ticker_bars(ticker):
    """ไฟล์ที่ล็อก SHA ใน raw_data_intraday/ -- แหล่งเดียวของ Phase 1D"""
    return read_bars(intraday_path(ticker), EXPECTED_SHA256[ticker])


def prepare_bars(df):
    """เพิ่ม ts (tz +07:00), date, hour · ตรวจว่าเป็นแท่งต้นชั่วโมงและไม่ซ้ำ"""
    df = df.copy()
    ts = pd.to_datetime(df["Datetime"])
    if ts.dt.tz is None:
        raise ValueError("Datetime ต้องมี timezone (+07:00)")
    ts = ts.dt.tz_convert(BANGKOK)
    if (ts.dt.minute != 0).any() or (ts.dt.second != 0).any():
        raise ValueError("พบแท่งที่ไม่ได้เริ่มต้นชั่วโมง")
    df["ts"] = ts
    df["date"] = pd.to_datetime(ts.dt.date)
    df["hour"] = ts.dt.hour
    if df.duplicated(["date", "hour"]).any():
        raise ValueError("พบแท่งซ้ำ (date, hour)")
    return df.sort_values("ts").reset_index(drop=True)


def _day_table(bars, hours=LABELED_HOURS, daily=None):
    """
    ตาราง 1 แถวต่อวัน: O/H/L/C/V ของแท่งที่ใช้เท่านั้น (hours · รอบ 1 = 10, 11, 12, 14, 15, 16)
    แท่ง 13 ไม่ถูกหยิบเข้ามาเลย · V10 ไม่ถูกหยิบ
    รอบ 2 (daily != None): hours = FEATURE_HOURS (ไม่อ่านแท่ง 16 เลย) + คอลัมน์
        CD     = ราคาปิดทางการของวันนั้น (ไม่มี = NaN)
        CDprev = ราคาปิดทางการของแถวก่อนหน้าใน daily ที่วันที่ < วันนั้น (= วันทำการก่อนหน้า)
    """
    use = bars[bars["hour"].isin(hours)]
    out = {}
    for field, short in (("Open", "O"), ("High", "H"), ("Low", "L"), ("Close", "C"),
                         ("Volume", "V")):
        p = use.pivot(index="date", columns="hour", values=field)
        for h in hours:
            if short == "V" and h == 10:
                continue                         # ห้ามอ่าน Volume แท่ง 10:00
            if short in ("O", "H", "L") and h == TARGET_HOUR:
                continue                         # ไม่มีใครต้องใช้ O/H/L ของแท่ง 16
            out[f"{short}{h}"] = p[h] if h in p.columns else np.nan
    table = pd.DataFrame(out).sort_index()
    if daily is not None:
        daily = daily.sort_index()
        table["CD"] = daily.reindex(table.index).to_numpy()
        pos = daily.index.searchsorted(table.index, side="left")      # แถวแรกที่ >= วันนั้น
        vals = daily.to_numpy(dtype=float)
        table["CDprev"] = (np.where(pos > 0, vals[np.maximum(pos - 1, 0)], np.nan)
                           if len(vals) else np.nan)
    return table


# ---------------------------------------------------------------
# features -- นิยามเดียวใช้ทั้ง historical และ live
# ---------------------------------------------------------------
def _features(seq):
    """
    seq = ตาราง day_table ของ "ลำดับวัน" ที่ใช้คำนวณ lag / rolling / t_prev
          = eligible labeled days (เรียงตามเวลา) + วัน live ต่อท้าย (ถ้ามี, C16 / CD = NaN)
          มีคอลัมน์ CD = รอบ 2 (target = ราคาปิดทางการ · gap ใช้ CDprev) · ไม่มี = รอบ 1
    lag / rolling / t_prev นับตามลำดับแถวนี้ (ข้ามวันที่ไม่ eligible)
    ค่าของแถว i ใช้ข้อมูลแถว i (ไม่รวม C16) และแถวก่อนหน้าเท่านั้น
    """
    f = pd.DataFrame(index=seq.index)
    C10, C11, C12, C14, C15 = (seq[f"C{h}"] for h in (10, 11, 12, 14, 15))
    f["ret_10_11"] = C11 / C10 - 1
    f["ret_11_12"] = C12 / C11 - 1
    f["ret_12_14"] = C14 / C12 - 1
    f["ret_14_15"] = C15 / C14 - 1
    f["ret_open10_15"] = C15 / seq["O10"] - 1

    hi = seq[[f"H{h}" for h in FEATURE_HOURS]].max(axis=1)
    lo = seq[[f"L{h}" for h in FEATURE_HOURS]].min(axis=1)
    width = hi - lo
    f["range"] = width / C15
    f["pos"] = np.where(width == 0, 0.5, (C15 - lo) / width.where(width != 0, 1.0))
    if "CD" in seq.columns:                                    # รอบ 2: ราคาปิดทางการของวันทำการก่อนหน้า
        f["gap"] = seq["O10"] / seq["CDprev"] - 1
    else:                                                      # รอบ 1
        f["gap"] = seq["O10"] / seq["C16"].shift(1) - 1       # C16 ของ t_prev

    vcum = seq["V11"] + seq["V12"] + seq["V14"] + seq["V15"]
    zero = seq.index[vcum == 0]
    assert len(zero) == 0, f"Vcum = 0 ในวัน {[str(d.date()) for d in zero]}"
    f["share12"] = seq["V12"] / vcum
    f["share14"] = seq["V14"] / vcum
    f["share15"] = seq["V15"] / vcum
    prev_mean = vcum.shift(1).rolling(RELVOL_WINDOW, min_periods=RELVOL_WINDOW).mean()
    f["relvol"] = np.log(vcum / prev_mean)

    y = (seq["CD"] if "CD" in seq.columns else seq["C16"]) / C15 - 1   # target ของแต่ละแถว
    for k in TARGET_LAGS:
        f[f"y_lag{k}"] = y.shift(k)

    dow = seq.index.dayofweek
    for i in range(5):
        f[f"dow_{i}"] = (dow == i).astype(float)
    return f[FEATURE_COLS], y


def _business_gap_dates(index):
    """วันที่ที่ t_prev ห่างเกิน 1 วันทำการ (จ.-ศ.) -- รายงานเท่านั้น ไม่ตัดทิ้ง"""
    d = index.values.astype("datetime64[D]")
    gaps = np.busday_count(d[:-1], d[1:])
    return [str(index[i + 1].date()) for i in np.where(gaps > 1)[0]]


def build_dataset(bars, daily_close=None):
    """
    historical labeled dataset ของหุ้นหนึ่งตัว
    daily_close = None   : รอบ 1 (target = C16/C15 − 1 · ต้องมีแท่ง 16)
    daily_close = Series : รอบ 2 (target = ราคาปิดทางการ/C15 − 1 · ไม่อ่านแท่ง 16 · index = วันที่ ราคาปิดที่ใช้ได้เท่านั้น)
    คืน (X, y, info, report)
        X      : 21 features · index = วันที่ (หลังตัด warm-up และแถว NaN/Inf)
        y      : target y(t)
        info   : C15 (ใช้แปลงเป็นบาท / เทียบ tick)
        report : eligible days, วันที่ถูกตัด + เหตุผล, warm-up, t_prev gaps, NaN rows
    """
    r2 = daily_close is not None
    hours = FEATURE_HOURS if r2 else LABELED_HOURS
    table = _day_table(bars, hours, daily_close)
    all_days = sorted(bars["date"].unique())
    needed = [f"C{h}" for h in hours] + (["CD"] if r2 else [])
    eligible = table[needed].notna().all(axis=1)
    present = bars.groupby("date")["hour"].apply(set)
    dropped = {}
    for d in all_days:
        if eligible.get(d, False):
            continue
        miss = sorted(set(hours) - present[d])
        if r2 and not miss:
            miss = ["daily_close"]
        dropped[str(pd.Timestamp(d).date())] = miss
    seq = table[eligible]

    X, y = _features(seq)
    t_prev_gaps = _business_gap_dates(seq.index)

    X, y, c15 = X.iloc[WARMUP_ROWS:], y.iloc[WARMUP_ROWS:], seq["C15"].iloc[WARMUP_ROWS:]
    finite = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
    bad = [str(d.date()) for d in X.index[~finite]]
    X, y, c15 = X[finite], y[finite], c15[finite]

    report = {
        "n_days_in_file": len(all_days),
        "n_eligible_labeled": int(eligible.sum()),
        "dropped_days": dropped,                        # วันที่ -> แท่งที่ขาด
        "warmup_dropped": [str(d.date()) for d in seq.index[:WARMUP_ROWS]],
        "nan_inf_dropped": bad,
        "n_rows": len(X),
        "t_prev_gap_dates": t_prev_gaps,
    }
    return X, y, pd.DataFrame({"C15": c15}), report


# ---------------------------------------------------------------
# time semantics + causal check
# ---------------------------------------------------------------
def time_fields(target_date):
    d = pd.Timestamp(target_date).date()
    at = lambda h: pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=h,  # noqa: E731
                                tz=BANGKOK)
    return {
        "last_feature_bar_start": at(15),
        "feature_window_end": at(16),            # = data_cutoff_timestamp
        "target_bar_start": at(TARGET_HOUR),
    }


def check_causal(feature_bar_starts, tf):
    """ทุก feature bar_start < target_bar_start และ feature_window_end <= target_bar_start"""
    starts = pd.DatetimeIndex(feature_bar_starts)
    if not (starts < tf["target_bar_start"]).all():
        raise AssertionError("มี feature bar ที่เริ่มไม่ก่อน target_bar_start")
    if not tf["feature_window_end"] <= tf["target_bar_start"]:
        raise AssertionError("feature_window_end เกิน target_bar_start")
    last_end = starts.max() + pd.Timedelta(hours=1)
    if last_end > tf["feature_window_end"]:
        raise AssertionError("feature bar สุดท้ายจบหลัง feature_window_end")


# ---------------------------------------------------------------
# live
# ---------------------------------------------------------------
def build_1600_live_feature(bars, target_date, daily_close=None):
    """
    สร้าง X ของวัน target_date ณ 16:00 + ชุดเทรน (dry-run เท่านั้นจนกว่า human freeze)

    1. ตัดทุกวันหลัง target_date ทิ้ง
    2. ตัดแท่งของ target_date ที่ timestamp >= 16:00 ทิ้ง
    3. ต้องมีแท่ง 10, 11, 12, 14, 15 ของ target_date ไม่งั้น raise
    4. features ด้วยนิยามเดียวกับ historical (ลำดับ = eligible labeled days ก่อนหน้า + วันนี้)
    5. ชุดเทรน = labeled rows หลังตัด warm-up / invalid ที่วันที่ < target_date

    รอบ 2 (daily_close = Series ราคาปิดทางการ): ตัดราคาปิดของวัน target และวันหลังจากนั้นทิ้งเสมอ (index < target)
    ก่อนสร้าง feature / label · gap ใช้ราคาปิดของวันก่อนหน้า target · ไม่อ่านแท่ง 16 ของวันใดเลย
    """
    target = pd.Timestamp(target_date).normalize()
    r2 = daily_close is not None
    hours = FEATURE_HOURS if r2 else LABELED_HOURS
    daily_hist = daily_close[daily_close.index < target] if r2 else None
    if r2 and daily_hist.empty:
        raise ValueError(f"{target.date()}: ไม่มีราคาปิดทางการก่อน target เลย -- สร้าง live feature ไม่ได้")
    tf = time_fields(target)
    b = bars[bars["date"] <= target]
    b = b[~((b["date"] == target) & (b["ts"] >= tf["feature_window_end"]))]

    today = b[b["date"] == target]
    have = set(today["hour"])
    missing = [h for h in FEATURE_HOURS if h not in have]
    if missing:
        raise ValueError(f"{target.date()}: ขาดแท่งบังคับ {missing} -- สร้าง live feature ไม่ได้")
    check_causal(today.loc[today["hour"].isin(FEATURE_HOURS), "ts"], tf)

    hist = b[b["date"] < target]
    X_hist, y_hist, info_hist, report = build_dataset(hist, daily_hist)

    table = _day_table(b, hours, daily_hist)
    needed = [f"C{h}" for h in hours] + (["CD"] if r2 else [])
    labeled = table[(table.index < target) & table[needed].notna().all(axis=1)]
    seq = pd.concat([labeled, table.loc[[target]]])
    X_all, _ = _features(seq)
    X_live = X_all.loc[[target]]
    if not np.isfinite(X_live.to_numpy()).all():
        raise ValueError(f"{target.date()}: live feature มี NaN/Inf "
                         f"{X_live.columns[~np.isfinite(X_live.to_numpy()[0])].tolist()}")
    out = {
        "X_live": X_live,
        "close_bar15": float(table.loc[target, "C15"]),
        **tf,
        "X_train": X_hist,
        "y_train": y_hist,
        "C15_train": info_hist["C15"],
        "train_days": len(X_hist),
        "report": report,
    }
    if r2:
        out["daily_prev_close"] = float(table.loc[target, "CDprev"])
    return out
