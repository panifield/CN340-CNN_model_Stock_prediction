"""
ตรวจ data leakage ของ A2 (ทำนาย ณ 16:00)

Usage:
    python A2/tests/check_leak.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from intraday import (CUTOFFS, START_DATE, STOCKS, build_dataset, load_daily, load_hourly,  # noqa: E402
                      load_raw_daily, load_yahoo_daily)

ROOT = Path(__file__).resolve().parent.parent.parent
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def recompute_features(h):
    h = h.copy()
    h["ret"] = np.log(h["Close"] / h["Close"].shift(1))
    h["range"] = np.log(h["High"] / h["Low"])
    h["vol"] = np.log1p(h["Volume"]).diff()
    return h


def perturb(h, d, mask_h, mask_d, rng):
    h, d = h.copy(), d.copy()
    h["Volume"] = h["Volume"].astype(float)
    for c in ["Open", "High", "Low", "Close"]:
        h.loc[mask_h, c] *= rng.uniform(0.9, 1.1, mask_h.sum())
    h.loc[mask_h, "High"] = h.loc[mask_h, ["Open", "High", "Low", "Close"]].max(axis=1)
    h.loc[mask_h, "Low"] = h.loc[mask_h, ["Open", "High", "Low", "Close"]].min(axis=1)
    h.loc[mask_h, "Volume"] *= rng.uniform(0.5, 2, mask_h.sum())
    d.loc[mask_d, "Close"] = d.loc[mask_d, "Close"] + rng.choice([-1.0, 1.0], mask_d.sum())
    d["parity"] = np.floor(d["Close"]).astype(int) % 2
    return recompute_features(h), d


def leak_test(stock, when, n_probe=12):
    """when = เวลาที่ทำนาย ("16:00" หรือ "16:30")"""
    bar = CUTOFFS[when]
    hh, mm = map(int, when.split(":"))
    last_bar_h = int(bar[:2])
    h, d = load_hourly(stock), load_daily(stock)
    X0, E0, y0, dates, _ = build_dataset(stock, h=h, d=d, cutoff_bar=bar)
    rng = np.random.default_rng(0)
    leaks = label_changes = sens = 0
    for k in rng.choice(len(X0), n_probe, replace=False):
        t = dates[k]
        # ข้อมูลหลัง 16:00 ของวัน t (แท่ง 16:00 + วันถัดไป) และราคาปิดวัน t เป็นต้นไป
        mask_h = (h["Datetime"] > t + pd.Timedelta(hours=last_bar_h))  # ทุกแท่งหลังแท่งสุดท้ายที่ใช้ได้
        mask_d = d["Date"] >= t
        h1, d1 = perturb(h, d, mask_h, mask_d, rng)
        X1, E1, y1, dates1, _ = build_dataset(stock, start=t, end=t + pd.Timedelta(days=1), h=h1, d=d1,
                                              cutoff_bar=bar)
        leaks += not (np.array_equal(X0[k], X1[0]) and np.array_equal(E0[k], E1[0]))
        label_changes += y0[k] != y1[0]
        # ตัวควบคุม: แก้แท่ง 15:00 ของวัน t (ข้อมูลที่ใช้ได้) -> input ต้องเปลี่ยน
        mask_last = h["Datetime"] == t + pd.Timedelta(hours=last_bar_h)
        h2, d2 = perturb(h, d, mask_last, d["Date"] < d["Date"].min(), rng)
        X2, E2, *_ = build_dataset(stock, start=t, end=t + pd.Timedelta(days=1), h=h2, d=d2, cutoff_bar=bar)
        sens += not (np.array_equal(X0[k], X2[0]) and np.array_equal(E0[k], E2[0]))
    check(f"[{when}] input ไม่เปลี่ยนเมื่อแก้ข้อมูลหลัง {hh:02d}:{mm:02d} / ราคาปิดวัน t", leaks == 0,
          f"{leaks}/{n_probe} เปลี่ยน")
    check(f"[{when}] (ตัวควบคุม) การแก้ราคาปิดวัน t ทำให้ label เปลี่ยน", label_changes > 0,
          f"{label_changes}/{n_probe}")
    check(f"[{when}] (ตัวควบคุม) แก้แท่งสุดท้ายที่ใช้ได้ ({bar}) แล้ว input เปลี่ยน", sens == n_probe,
          f"{sens}/{n_probe}")


def daily_source_test(stock):
    """Yahoo (ใช้หลัง raw_data สิ้นสุด) ต้องตรงกับ raw_data ในช่วงที่ทับกัน -> เชื่อถือได้"""
    y, r = load_yahoo_daily(stock), load_raw_daily(stock)
    m = y.merge(r[["Date", "Close"]], on="Date", suffixes=("_y", "_r"))
    m = m[m["Date"] >= START_DATE]
    bad = m[(m["Close_y"] - m["Close_r"]).abs() > 1e-6]
    check("ราคาปิด Yahoo (ตัด Volume=0) ตรงกับ raw_data ตั้งแต่ START_DATE", len(bad) == 0,
          f"ไม่ตรง {len(bad)}/{len(m)} วัน" + (f": {bad['Date'].dt.date.tolist()}" if len(bad) else ""))
    d = load_daily(stock)
    check("label มาจาก raw_data ทุกวันที่ raw_data มี", bool(
        (d.merge(r[["Date", "Close"]], on="Date", suffixes=("", "_r")).eval("Close == Close_r")).all()))


def main():
    for s in STOCKS:
        print(f"\n===== {s} =====")
        daily_source_test(s)
        for when in CUTOFFS:
            leak_test(s, when)
    print("\nสรุป:", "ผ่านทั้งหมด" if not failures else f"ไม่ผ่าน {len(failures)} ข้อ: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
