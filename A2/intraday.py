"""
Task A2: ทำนายราคาปิดคู่/คี่ของวัน t ตอน 16:00 ของวัน t (ก่อนตลาดปิด) ด้วยข้อมูลรายชั่วโมง

ข้อมูล    : data/{STOCK}_1h_yahoo.csv (แท่ง 1 ชั่วโมง, เวลาไทย, label = เวลา "เริ่ม" ของแท่ง)
            data/{STOCK}_1d_yahoo.csv (ราคาปิดทางการรายวัน -> ใช้ทำ label)
เวลาตัด   : 16:00 -> ใช้ได้เฉพาะแท่งที่จบก่อน 16:00 = แท่ง 10:00 ... 15:00 (15:00-16:00)
            แท่ง 16:00 (16:00-16:30) และช่วงปิดตลาด ATC ห้ามใช้
Label     : floor(ราคาปิดวัน t) % 2   (0 = คู่, 1 = คี่)
Input CNN : spectrogram ของ N_BARS แท่งชั่วโมงล่าสุด ณ 16:00 (ข้ามวันได้)
Input ANN : feature ของราคา ณ 16:00 (คู่/คี่ตอนนี้, ระยะห่างจาก open/high/low ของวัน ฯลฯ)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest  # noqa: F401  (ใช้ใน main/tests)

A2_DIR = Path(__file__).resolve().parent
DATA_DIR = A2_DIR / "data"
STOCKS = ["KBANK", "ADVANC"]
START_DATE = "2025-09-07"          # 7 ก.ย. 2568
CUTOFF_BAR = "15:00"               # แท่งสุดท้ายที่ใช้ได้ ณ 16:00
# เวลาทำนาย -> แท่งสุดท้ายที่ใช้ได้ (16:30 = ใช้แท่ง 16:00-16:30 ก่อนช่วง ATC)
CUTOFFS = {"16:00": "15:00", "16:30": "16:00"}
N_BARS = 36                        # ~6 วันทำการ
NPERSEG, NOVERLAP = 8, 6
CHANNELS = ["ret", "range", "vol"]
EXTRA_NAMES = ["parity_now", "half_now", "tick_is_half", "ticks_from_open", "ticks_from_high",
               "ticks_from_low", "prev_close_parity", "ret_last_bar"]


def tick_size(p):
    """ช่วงราคาตามกฎ SET"""
    for hi, t in [(2, 0.01), (5, 0.02), (10, 0.05), (25, 0.1), (100, 0.25), (200, 0.5), (400, 1.0)]:
        if p < hi:
            return t
    return 2.0


def load_hourly(stock):
    h = pd.read_csv(DATA_DIR / f"{stock}_1h_yahoo.csv", parse_dates=["Datetime"])
    h = h.dropna(subset=["Close"]).sort_values("Datetime").reset_index(drop=True)
    h["day"] = h["Datetime"].dt.normalize()
    h["hm"] = h["Datetime"].dt.strftime("%H:%M")
    h["ret"] = np.log(h["Close"] / h["Close"].shift(1))
    h["range"] = np.log(h["High"] / h["Low"])
    h["vol"] = np.log1p(h["Volume"]).diff()
    return h.iloc[1:].reset_index(drop=True)


def load_yahoo_daily(stock):
    """Yahoo รายวัน -- ตัดแถวที่ Volume = 0 (Yahoo ใส่ค่าหลอกไว้ เช่น 2025-10-07)"""
    y = pd.read_csv(DATA_DIR / f"{stock}_1d_yahoo.csv", parse_dates=["Date"])
    return y[(y["Volume"] > 0) & y["Close"].notna()]


def load_raw_daily(stock):
    r = pd.read_csv(A2_DIR.parent / "raw_data" / f"{stock}_10Y_Cleaned.csv")
    r["Date"] = pd.to_datetime(r["Date"], format="%m/%d/%Y")
    return r.rename(columns={"Price": "Close"})


def load_daily(stock):
    """ราคาปิดทางการ: ใช้ raw_data (ข้อมูลของโจทย์) เป็นหลัก, ต่อด้วย Yahoo หลังวันสุดท้ายของ raw_data"""
    r = load_raw_daily(stock)[["Date", "Close"]]
    y = load_yahoo_daily(stock)
    y = y.loc[y["Date"] > r["Date"].max(), ["Date", "Close"]]
    d = pd.concat([r, y]).sort_values("Date").reset_index(drop=True)
    d["parity"] = np.floor(d["Close"]).astype(int) % 2
    return d


def stft_logpower(x):
    from scipy.signal import stft
    _, _, z = stft(x, fs=1.0, nperseg=NPERSEG, noverlap=NOVERLAP, window="hann",
                   boundary="even", padded=False, detrend="constant")
    return np.log1p(np.abs(z) ** 2 * 1e5)


def build_dataset(stock, start=START_DATE, end=None, h=None, d=None, cutoff_bar=CUTOFF_BAR):
    """คืน X (n, C, F, T), E (n, n_extra), y (n,), dates, p16 (ราคา ณ 16:00)"""
    h = load_hourly(stock) if h is None else h
    d = load_daily(stock) if d is None else d
    feats = h[CHANNELS].to_numpy(dtype=np.float64)
    prev_parity = dict(zip(d["Date"], d["parity"].shift(1)))
    X, E, y, dates, p16s = [], [], [], [], []
    for _, row in d[d["Date"] >= start].iterrows():
        if end is not None and row["Date"] >= pd.Timestamp(end):
            break
        day = h[(h["day"] == row["Date"]) & (h["hm"] <= cutoff_bar)]
        if day.empty or day["hm"].iloc[-1] != cutoff_bar or pd.isna(prev_parity.get(row["Date"])):
            continue  # ไม่มีแท่ง 15:00 (ข้อมูลขาด) -> ข้าม
        i = day.index[-1]
        if i + 1 < N_BARS:
            continue
        p16 = day["Close"].iloc[-1]
        tk = tick_size(p16)
        X.append(np.stack([stft_logpower(feats[i - N_BARS + 1:i + 1, c]) for c in range(len(CHANNELS))]))
        E.append([np.floor(p16) % 2,
                  (p16 % 1 >= 0.5) * 1.0,
                  float(tk < 1),
                  np.clip((p16 - day["Open"].iloc[0]) / tk, -10, 10) / 10,
                  np.clip((p16 - day["High"].max()) / tk, -10, 0) / 10,
                  np.clip((p16 - day["Low"].min()) / tk, 0, 10) / 10,
                  prev_parity[row["Date"]],
                  np.clip(feats[i, 0] * 100, -3, 3)])
        y.append(row["parity"])
        dates.append(row["Date"])
        p16s.append(p16)
    return (np.stack(X).astype(np.float32), np.array(E, dtype=np.float32),
            np.array(y, dtype=np.float32), np.array(dates), np.array(p16s))
