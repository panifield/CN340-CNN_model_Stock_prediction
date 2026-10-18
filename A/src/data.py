"""โหลดข้อมูล, สร้าง feature รายวัน และแปลงเป็น spectrogram"""
import numpy as np
import pandas as pd
from scipy.signal import stft

from .config import CHANNELS, DATA_DIR, NOVERLAP, NPERSEG, START_DATE, WINDOW

# feature เสริมที่ป้อนเข้า ANN โดยตรง (ไม่ผ่าน CNN) -- ทุกตัวมาจากวัน t-1 หรือก่อนหน้า
EXTRA_FEATURES = ["prev_parity", "prev_int_chg_odd", "prev_frac", "prev_parity2"]


def load_stock(stock: str, start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
    """end (ไม่รวม) ใช้เฉพาะการทดลองย้อนหลังปีอื่น; ค่า default = ถึงข้อมูลล่าสุด"""
    df = pd.read_csv(DATA_DIR / f"{stock}_10Y_Cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.rename(columns={"Price": "Close", "Vol. ('000)": "Volume"})
    df = df[df["Date"] >= start]
    if end is not None:
        df = df[df["Date"] < end]
    df = df.sort_values("Date").reset_index(drop=True)

    prev_close = df["Close"].shift(1)
    df["ret_close"] = np.log(df["Close"] / prev_close)
    df["gap_open"] = np.log(df["Open"] / prev_close)
    df["range_hl"] = np.log(df["High"] / df["Low"])
    df["chg_vol"] = np.log(df["Volume"] / df["Volume"].shift(1))
    # 0 = คู่, 1 = คี่ (ใช้ส่วนจำนวนเต็ม เพราะ KBANK มีราคา .5)
    df["parity"] = np.floor(df["Close"]).astype(int) % 2
    # channel ที่เกี่ยวกับคู่/คี่โดยตรง (ใช้ในการทดลอง)
    df["parity_sig"] = 2.0 * df["parity"] - 1.0               # +1 คี่ / -1 คู่
    df["int_chg"] = np.floor(df["Close"]).diff()               # ส่วนต่างราคาเต็มบาท
    return df.iloc[1:].reset_index(drop=True)  # แถวแรกไม่มี prev_close


def stft_logpower(x: np.ndarray, nperseg: int = NPERSEG, noverlap: int = NOVERLAP):
    """1-D signal -> (freqs, frame_centers, log-power (F, T))

    boundary="even" สะท้อนข้อมูลที่ขอบ (ใช้แค่ข้อมูลในหน้าต่างเดียวกัน ไม่ใช่อนาคต)
    ทำให้ frame สุดท้ายอยู่ตรงกับวันล่าสุด -- ถ้าไม่ pad, Hann window จะให้น้ำหนัก
    วัน t-1 เหลือแค่ ~0.5% ของวันกลางหน้าต่าง ทำให้โมเดลแทบไม่เห็นข้อมูลเมื่อวาน"""
    f, t, z = stft(x, fs=1.0, nperseg=nperseg, noverlap=noverlap, window="hann",
                   boundary="even", padded=False, detrend="constant")
    return f, t, np.log1p(np.abs(z) ** 2 * 1e5)


def to_spectrogram(window_feats: np.ndarray, nperseg: int = NPERSEG,
                   noverlap: int = NOVERLAP) -> np.ndarray:
    """window_feats: (window, C) -> (C, F, T) log-power spectrogram"""
    specs = [stft_logpower(window_feats[:, c], nperseg, noverlap)[2]
             for c in range(window_feats.shape[1])]
    return np.stack(specs).astype(np.float32)


def build_dataset(df: pd.DataFrame, window: int = WINDOW, channels=CHANNELS,
                  nperseg: int = NPERSEG, noverlap: int = NOVERLAP):
    """sample ของวัน t ใช้ข้อมูล t-window .. t-1 เท่านั้น (ไม่มี leakage)"""
    feats = df[list(channels)].to_numpy(dtype=np.float64)
    X, y, dates = [], [], []
    for t in range(window, len(df)):
        X.append(to_spectrogram(feats[t - window:t], nperseg, noverlap))
        y.append(df["parity"].iloc[t])
        dates.append(df["Date"].iloc[t])
    return np.stack(X), np.array(y, dtype=np.float32), np.array(dates)


def build_extras(df: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """feature เสริมของ sample วัน t (เรียงตรงกับ build_dataset) จากวัน t-1, t-2"""
    close, parity, int_chg = df["Close"].to_numpy(), df["parity"].to_numpy(), df["int_chg"].to_numpy()
    rows = []
    for t in range(window, len(df)):
        rows.append([parity[t - 1],
                     abs(int_chg[t - 1]) % 2,        # เมื่อวานคู่/คี่สลับไหม
                     close[t - 1] % 1 * 2,           # มีเศษ .5 ไหม (KBANK)
                     parity[t - 2]])
    return np.array(rows, dtype=np.float32)


def build_round_features(df: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """price clustering: one-hot เลขท้ายราคาปิดเมื่อวาน (10) + ระยะห่างจากเลขลงท้าย 0 (0..1)"""
    last = np.floor(df["Close"].to_numpy()).astype(int) % 10
    rows = []
    for t in range(window, len(df)):
        d = last[t - 1]
        rows.append(np.r_[np.eye(10)[d], min(d, 10 - d) / 5])
    return np.array(rows, dtype=np.float32)


def build_latest_input(df: pd.DataFrame) -> np.ndarray:
    """spectrogram ของ WINDOW วันล่าสุด -> ใช้ทำนายวันซื้อขายถัดไป"""
    feats = df[CHANNELS].to_numpy(dtype=np.float64)
    return to_spectrogram(feats[-WINDOW:])[None]


def split_and_normalize(X, y, split):
    """แบ่งตามเวลา แล้ว standardize ต่อ channel ด้วยสถิติจาก train เท่านั้น"""
    n = len(X)
    n_tr, n_va = int(n * split[0]), int(n * split[1])
    sl = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, n)}
    mu = X[sl["train"]].mean(axis=(0, 2, 3), keepdims=True)
    sd = X[sl["train"]].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    parts = {k: (Xn[s], y[s]) for k, s in sl.items()}
    return parts, sl, mu, sd
