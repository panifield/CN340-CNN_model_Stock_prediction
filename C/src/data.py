"""โหลดข้อมูล สร้าง feature/label และสร้าง spectrogram โดยไม่มี future leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import stft

from .config import CHANNELS, DATA_DIR, NOVERLAP, NPERSEG, WINDOW


def load_stock(stock: str, years: int) -> pd.DataFrame:
    """คืนข้อมูลย้อนหลัง `years` ปี พร้อม feature และ next-day target.

    target ของแถว t คือ 1 เมื่อ Close(t+1) > Close(t), มิฉะนั้นเป็น 0.
    แถวสุดท้ายถูกตัดออก เพราะยังไม่มีราคาปิดของวันถัดไป.
    """
    path = DATA_DIR / f"{stock}_10Y_Cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบข้อมูลหุ้น: {path}")

    df = pd.read_csv(path).rename(columns={"Price": "Close", "Vol. ('000)": "Volume"})
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} ไม่มีคอลัมน์: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="raise")
    df = df.drop_duplicates(subset="Date", keep="last").sort_values("Date").reset_index(drop=True)
    if (df[["Open", "High", "Low", "Close", "Volume"]] <= 0).any().any():
        raise ValueError(f"{path.name} มีราคา/ปริมาณที่ไม่เป็นบวก จึงคำนวณ log feature ไม่ได้")

    previous_close = df["Close"].shift(1)
    df["ret_close"] = np.log(df["Close"] / previous_close)
    df["gap_open"] = np.log(df["Open"] / previous_close)
    df["range_hl"] = np.log(df["High"] / df["Low"])
    df["chg_vol"] = np.log(df["Volume"] / df["Volume"].shift(1))
    df = df.dropna(subset=CHANNELS).reset_index(drop=True)

    cutoff = df["Date"].max() - pd.DateOffset(years=years)
    df = df.loc[df["Date"] >= cutoff].copy().reset_index(drop=True)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(np.float32)
    return df.iloc[:-1].reset_index(drop=True)


def stft_logpower(signal: np.ndarray, nperseg: int = NPERSEG,
                  noverlap: int = NOVERLAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """แปลง 1D signal เป็น log-power STFT โดย padding เฉพาะในอดีต window เดียวกัน."""
    frequencies, frames, coefficients = stft(
        signal, fs=1.0, nperseg=nperseg, noverlap=noverlap, window="hann",
        boundary="zeros", padded=False, detrend="constant",
    )
    return frequencies, frames, np.log1p(np.abs(coefficients) ** 2)


def to_spectrogram(window_features: np.ndarray, nperseg: int = NPERSEG,
                   noverlap: int = NOVERLAP) -> np.ndarray:
    """(window, channels) -> (channels, frequency, time)"""
    specs = [stft_logpower(window_features[:, i], nperseg, noverlap)[2]
             for i in range(window_features.shape[1])]
    return np.stack(specs).astype(np.float32)


def build_dataset(df: pd.DataFrame, window: int = WINDOW,
                  channels: list[str] = CHANNELS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """สร้าง sample ที่ label วัน t แต่ใช้ feature ถึงวัน t-1 เท่านั้น."""
    if len(df) <= window:
        raise ValueError(f"ข้อมูลมี {len(df)} แถว น้อยเกินกว่าสำหรับ window={window}")
    features = df[channels].to_numpy(dtype=np.float64)
    X, y, dates = [], [], []
    for t in range(window, len(df)):
        X.append(to_spectrogram(features[t - window:t]))
        y.append(df["target"].iloc[t])
        dates.append(df["Date"].iloc[t])
    return np.stack(X), np.asarray(y, dtype=np.float32), np.asarray(dates)


def split_and_normalize(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                        split: tuple[float, float, float], embargo: int = 1):
    """แบ่งตามเวลาแบบ purged split และ normalize ด้วยสถิติของ train เท่านั้น.

    label ของ sample วัน t ใช้ Close(t+1) จึงเว้น `embargo` sample ที่รอยต่อ
    train/valid และ valid/test เพื่อไม่ให้ label ของชุดก่อนอ่านราคาของชุดถัดไป.
    """
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"split ต้องรวมเป็น 1.0, ได้ {split}")
    usable = len(X) - 2 * embargo
    if usable <= 0:
        raise ValueError(f"ตัวอย่าง {len(X)} ไม่พอสำหรับ embargo={embargo}")
    n_train, n_valid = int(usable * split[0]), int(usable * split[1])
    slices = {
        "train": slice(0, n_train),
        "valid": slice(n_train + embargo, n_train + embargo + n_valid),
        "test": slice(n_train + 2 * embargo + n_valid, len(X)),
    }
    if any(len(y[s]) == 0 for s in slices.values()):
        raise ValueError(f"ตัวอย่าง {len(X)} ไม่พอสำหรับ temporal split {split}")

    train = X[slices["train"]]
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    X_normalized = ((X - mean) / std).astype(np.float32)
    parts = {
        name: (X_normalized[sl], y[sl], dates[sl])
        for name, sl in slices.items()
    }
    return parts, slices, mean, std
