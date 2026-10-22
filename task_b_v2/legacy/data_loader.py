"""
data_loader.py
==============
โหลดข้อมูลราคาหุ้นจากไฟล์ investing.com ใน raw_data/ (A1)

- แหล่งข้อมูลมีแหล่งเดียว ไม่มี fallback -- ไฟล์หาย = error ทันที
- พิมพ์ลายนิ้วมือ (sha256) ของไฟล์ทุกครั้งที่โหลด (E5)
  เพื่อยืนยันว่า "รันโค้ดเดิม" ใช้ข้อมูลเดิมจริง

รูปแบบไฟล์ investing.com:
    Date,Price,Open,High,Low,Vol. ('000),Change %
    08/26/2016,198.0,195.0,199.0,195.0,6790.0,1.54%
"""

import hashlib
from pathlib import Path

import pandas as pd

from config import DATA_SOURCE, RAW_DATA_DIR, RAW_DATA_SUFFIX

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

INVESTING_DATE_FORMAT = "%m/%d/%Y"
INVESTING_VOLUME_COL = "Vol. ('000)"          # หน่วยพันหุ้น -> x1000


def raw_data_path(ticker):
    """KBANK.BK -> raw_data/KBANK_10Y_Cleaned.csv"""
    return RAW_DATA_DIR / (ticker.replace(".BK", "") + RAW_DATA_SUFFIX)


def dataset_fingerprint(path):
    """
    พิมพ์ลายนิ้วมือของไฟล์ข้อมูล เพื่อยืนยันว่า 'รันโค้ดเดิม' ใช้ข้อมูลเดิมจริง

    คืน SHA-256 เต็ม 64 hex (ค่านี้ลง dataset_sha256 ของ prediction log)
    พิมพ์แค่ 16 ตัวแรกให้อ่านง่าย · แถวเก่าใน prediction_log.csv ที่มี 16 ตัว = legacy
    """
    raw = Path(path).read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(path)
    print(f"[data] {Path(path).name}  rows={len(df)}  sha256={h[:16]}...")
    return h


def load_from_investing(ticker, verbose=True):
    """อ่านไฟล์ csv จาก investing.com แล้วแปลงเป็น Open/High/Low/Close/Volume"""
    path = raw_data_path(ticker)
    # ไฟล์ intraday เป็นแหล่งข้อมูลอื่น (Yahoo) ห้ามหลุดเข้า pipeline รายวัน (§6.5)
    if "_1h" in path.name or "intraday" in str(path):
        raise ValueError(
            f"{path.name} เป็นข้อมูล intraday จากคนละแหล่ง "
            "ห้ามโหลดเข้า pipeline รายวัน"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ {path}\n"
            f"DATA_SOURCE = 'investing' จะไม่ดึงข้อมูลจาก Yahoo มาแทนให้"
        )

    if verbose:
        print(f"[data] อ่านจาก raw_data: {path.name}")
        dataset_fingerprint(path)

    raw = pd.read_csv(path)
    needed = ["Date", "Price", "Open", "High", "Low", INVESTING_VOLUME_COL]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} ไม่มีคอลัมน์ {missing} "
                         f"(มี {list(raw.columns)})")

    # Price -> Close, Vol.('000) x 1000 -> Volume, ตัด Change % ทิ้ง
    df = pd.DataFrame({
        "Open": pd.to_numeric(raw["Open"]),
        "High": pd.to_numeric(raw["High"]),
        "Low": pd.to_numeric(raw["Low"]),
        "Close": pd.to_numeric(raw["Price"]),
        "Volume": pd.to_numeric(raw[INVESTING_VOLUME_COL]) * 1000,
    })
    df.index = pd.to_datetime(raw["Date"], format=INVESTING_DATE_FORMAT)
    df.index.name = "Date"
    return df[REQUIRED_COLS]


def load_stock(ticker, verbose=True):
    """
    ฟังก์ชันหลักที่ไฟล์อื่นเรียกใช้
    คืน DataFrame คอลัมน์ Open/High/Low/Close/Volume index เป็นวันที่
    """
    if DATA_SOURCE != "investing":
        raise ValueError(f"DATA_SOURCE = {DATA_SOURCE!r} ไม่รองรับ "
                         f"(รองรับแค่ 'investing')")

    df = load_from_investing(ticker, verbose=verbose)
    df = clean(df, verbose=verbose)
    if verbose:
        print(f"[data] {ticker}: {len(df)} แถว "
              f"({df.index[0].date()} ถึง {df.index[-1].date()})")
    return df


def clean(df, verbose=True):
    """ทำความสะอาดข้อมูลเบื้องต้น"""
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    before = len(df)
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    if verbose and len(df) < before:
        print(f"[data] ตัดแถวเสีย {before - len(df)} แถว")

    # วันที่ Volume = 0 มักเป็นวันหยุด/ข้อมูลผิด
    zero_vol = (df["Volume"] == 0).sum()
    if verbose and zero_vol > 0:
        print(f"[data] เตือน: มี {zero_vol} วันที่ Volume = 0")

    return df
