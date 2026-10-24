"""
synth1600.py — ข้อมูลสังเคราะห์สำหรับ test ของโหมด 16:00 (ไม่ใช่ข้อมูลจริง · test ที่เทรนใช้ข้อมูลนี้เท่านั้น)
==========================================================================================================
make_daily()  : ราคารายวัน OHLCV (random walk) รูปแบบเดียวกับ data1600.load_daily / src.data.load_prices
make_bars()   : แท่ง 1h วันละ 10..16 (13 สุ่มหาย) รูปแบบเดียวกับ data1600.read_bars ของ n วันท้ายของปฏิทินรายวัน
tiny_track1() : config "track 1" ขนาดเล็ก (window 32 · nperseg 8 · 3 epoch) — แทนช่องว่าง TRACK1_CONFIG ใน test เท่านั้น
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from src.configs import BASE


def make_daily(n=420, start="2022-01-03", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, name="Date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)


def make_bars(daily, n_days=220, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for d in daily.index[-n_days:]:
        o = float(daily.loc[d, "Open"])
        for h in (10, 11, 12, 13, 14, 15, 16):
            if h == 13 and rng.random() < 0.5:
                continue
            c = o * (1 + rng.normal(0, 0.003))
            rows.append({"ts": pd.Timestamp(f"{d.date()} {h:02d}:00:00+07:00"), "date": d, "hour": h,
                         "Open": o, "High": max(o, c) * 1.001, "Low": min(o, c) * 0.999, "Close": c, "AdjClose": c,
                         "Volume": 0.0 if (h == 10 and rng.random() < 0.7) else float(rng.integers(10_000, 2_000_000))})
            o = c
    return pd.DataFrame(rows)


def tiny_track1():
    return replace(BASE, name="tiny_track1", window=32, nperseg=8, hop=2, max_epochs=3, patience=2)
