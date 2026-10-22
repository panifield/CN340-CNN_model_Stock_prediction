"""
test_data_sources.py — ไฟล์ข้อมูลใน data/ ต้องตรงกับที่บันทึกไว้ใน data/SOURCES.md
=================================================================================
ไฟล์นี้ทำอะไร: ตรวจ SHA-256 / จำนวนแถว / วันแรก-วันสุดท้าย ของ data/*.csv เทียบกับ SOURCES.md และตรวจความสะอาดของข้อมูล
ทำไมต้องมี: ถ้าข้อมูลถูกแก้เงียบ ๆ (เช่น refresh ไฟล์แล้วลืมอัปเดต SOURCES.md) ผลทุกตัวจะเทียบกันไม่ได้
"""
import hashlib
import re

import pandas as pd
import pytest

from src import paths
from src.settings import TICKERS, VAL_END

SOURCES = (paths.DATA_DIR / "SOURCES.md").read_text(encoding="utf-8")


def _load(t):
    return pd.read_csv(paths.DATA_DIR / f"{t}_10Y_Cleaned.csv")


@pytest.mark.parametrize("ticker", TICKERS)
def test_sha256_matches_sources_md(ticker):
    sha = hashlib.sha256((paths.DATA_DIR / f"{ticker}_10Y_Cleaned.csv").read_bytes()).hexdigest()
    assert sha in SOURCES, f"SHA-256 ของ {ticker} ไม่อยู่ใน SOURCES.md -- อัปเดตไฟล์ SOURCES.md ด้วย"


@pytest.mark.parametrize("ticker", TICKERS)
def test_dates_sorted_unique_no_nulls(ticker):
    df = _load(ticker)
    d = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    assert d.is_monotonic_increasing and d.is_unique
    assert int(df.isna().sum().sum()) == 0
    assert list(df.columns[:6]) == ["Date", "Price", "Open", "High", "Low", "Vol. ('000)"]
    assert (df[["Price", "Open", "High", "Low"]] > 0).all().all()


def test_both_tickers_share_the_same_calendar():
    a, b = _load(TICKERS[0])["Date"], _load(TICKERS[1])["Date"]
    assert a.equals(b), "สองหุ้นต้องมีวันซื้อขายชุดเดียวกัน (ใช้วันประเมินชุดเดียวกัน)"


def test_dev_zone_has_expected_rows():
    d = pd.to_datetime(_load(TICKERS[0])["Date"], format="%m/%d/%Y")
    assert int((d <= pd.Timestamp(VAL_END)).sum()) == 2071
    assert d[d <= pd.Timestamp(VAL_END)].iloc[-1] == pd.Timestamp(VAL_END), "วันสุดท้ายของ dev ต้องเป็น VAL_END พอดี"
