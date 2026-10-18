"""
ดาวน์โหลดข้อมูลรายชั่วโมงและรายวันจาก Yahoo Finance -> A2/data/

หมายเหตุ: Yahoo ให้ข้อมูล 1h ย้อนหลังได้ ~730 วัน (5m/15m ได้แค่ ~60 วัน จึงใช้ 1h)
         การรันใหม่จะได้ช่วงเวลาเลื่อนไปตามวันที่รัน

Usage:
    python A2/download_data.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"


def flat(df):
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for s in ["KBANK", "ADVANC"]:
        h = flat(yf.download(f"{s}.BK", period="730d", interval="1h", progress=False, auto_adjust=False))
        h.index = h.index.tz_convert("Asia/Bangkok").tz_localize(None)
        h.index.name = "Datetime"
        d = flat(yf.download(f"{s}.BK", start="2023-09-01", interval="1d", progress=False, auto_adjust=False))
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d.index.name = "Date"
        h.to_csv(DATA_DIR / f"{s}_1h_yahoo.csv")
        d.to_csv(DATA_DIR / f"{s}_1d_yahoo.csv")
        print(f"{s}: 1h {len(h)} แท่ง ({h.index.min()} - {h.index.max()}), 1d {len(d)} วัน")


if __name__ == "__main__":
    main()
