#!/usr/bin/env python3
"""Sanity checks สำหรับ temporal split และการสร้าง sample ของงาน C."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from src.config import SPLIT, WINDOW  # noqa: E402
from src.data import build_dataset, load_stock, split_and_normalize  # noqa: E402


def check_stock(stock: str) -> None:
    df = load_stock(stock, years=3)
    X, y, dates = build_dataset(df)
    parts, _, mean, std = split_and_normalize(X, y, dates, SPLIT)

    assert df["Date"].is_monotonic_increasing
    assert not df["Date"].duplicated().any()
    assert len(X) == len(df) - WINDOW
    # ทุก label ต้องตรงกับ target ที่ load_stock สร้างไว้
    assert np.array_equal(y, df["target"].iloc[WINDOW:].to_numpy(dtype=np.float32))

    # sample i maps to row t=i+WINDOW: input ends at t-1 and target is Close(t+1) > Close(t).
    # ไม่ใช้ sample สุดท้ายตรงนี้ เพราะ load_stock ตัดแถว Close(t+1) สุดท้ายออกแล้ว
    # (แต่ label ของมันถูกตรวจจาก df["target"] ใน assertion ด้านบน)
    for index in (0, len(X) // 2, len(X) - 2):
        t = index + WINDOW
        expected = float(df["Close"].iloc[t + 1] > df["Close"].iloc[t])
        assert y[index] == expected
        assert df["Date"].iloc[t - 1] < dates[index]

    train_dates = parts["train"][2]
    valid_dates = parts["valid"][2]
    test_dates = parts["test"][2]
    assert train_dates[-1] < valid_dates[0] < test_dates[0]
    # หนึ่ง sample ถูกเว้นที่รอยต่อแต่ละด้าน: label ของชุดก่อนจึงไม่แตะชุดถัดไป
    date_index = {date: i for i, date in enumerate(dates)}
    assert date_index[valid_dates[0]] - date_index[train_dates[-1]] == 2
    assert date_index[test_dates[0]] - date_index[valid_dates[-1]] == 2
    raw_train = X[:len(train_dates)]
    assert np.allclose(mean, raw_train.mean(axis=(0, 2, 3), keepdims=True))
    assert np.allclose(std, raw_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6)
    print(f"PASS {stock}: {len(X)} samples; {train_dates[0].date()} -> {test_dates[-1].date()}")


if __name__ == "__main__":
    for stock_name in ("ADVANC", "KBANK"):
        check_stock(stock_name)
