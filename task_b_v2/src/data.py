"""
data.py — โหลดราคา, สร้าง target / feature 29 ตัว, แบ่งโซนและ fold (SPEC ข้อ 3.1 และ 4)
======================================================================================
ไฟล์นี้ทำอะไร
    1. `load_prices(ticker)`  : อ่าน data/<TICKER>_10Y_Cleaned.csv → DataFrame Open/High/Low/Close/Volume
    2. `build_dataset(...)`   : **ตัดข้อมูลตามโซนก่อน** แล้วจึงสร้าง target (y = Close(t)/Close(t−1) − 1)
                                และ 29 feature เดิมด้วย legacy/features.py (อ่านอย่างเดียว)
    3. `make_folds(...)`      : walk-forward 3 fold แบบ expanding · fold ละ 266 วันทำการ จบที่ 2025-02-25

ทำไมต้องมี
    * **ที่เดียวที่รู้เรื่อง "วันที่"** — ความ causal และการตัดโซนถูกบังคับที่นี่ไฟล์เดียว
      (`build_dataset` ตัด `prices.loc[:zone_end]` เป็นคำสั่งแรกก่อนคำนวณอะไรทั้งนั้น แล้ว assert)
    * ใช้ feature เดิมผ่าน legacy โดยตรง ไม่เขียนใหม่ → ตัวเทียบ (MLP/RF/XGB) ได้ feature ชุดเดียวกับรอบก่อนแน่นอน

*** ข้อห้าม (SPEC ข้อ 11) : ระหว่างออกแบบ zone_end ต้องเป็น VAL_END = 2025-02-25 เท่านั้น ***
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import paths
from src.configs import Config
from src.settings import EARLY_STOP_FRAC, FOLD_DAYS, N_FOLDS, TICKERS, VAL_END
from src.spectrogram import ImageTransform, build_images, channel_matrix

paths.use_legacy()                     # ต้องเรียกก่อน import โมดูลของ legacy
from features import build_features     # noqa: E402  (legacy -- อ่านอย่างเดียว)
from targets import build_targets       # noqa: E402

DATE_FORMAT = "%m/%d/%Y"                        # รูปแบบวันที่ของไฟล์ investing.com
VOLUME_COL = "Vol. ('000)"                      # หน่วยพันหุ้น → ×1000
N_LEGACY_FEATURES = 29                          # 24 market (shift(1) แล้ว) + 5 day-of-week


def load_prices(ticker, verbose=False):
    """
    อ่านไฟล์ราคาของ data/ → DataFrame Open/High/Low/Close/Volume (index = วันที่, เรียงเวลา)

    (ported from task_b/legacy/data_loader.py : load_from_investing + clean, branch task_b)
    เขียนเองเพราะ legacy อ่านจาก legacy/raw_data/ ซึ่งเราไม่ได้ copy มา (ข้อมูลอยู่ที่ data/ แหล่งเดียว)
    `tests/test_data_loader_matches_legacy.py` ยืนยันว่าให้ผลเท่ากับ legacy ทุกค่าเมื่อชี้ไปไฟล์เดียวกัน
    """
    path = paths.DATA_DIR / f"{ticker}_10Y_Cleaned.csv"
    raw = pd.read_csv(path)
    missing = [c for c in ["Date", "Price", "Open", "High", "Low", VOLUME_COL] if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} ไม่มีคอลัมน์ {missing}")
    df = pd.DataFrame({
        "Open": pd.to_numeric(raw["Open"]),
        "High": pd.to_numeric(raw["High"]),
        "Low": pd.to_numeric(raw["Low"]),
        "Close": pd.to_numeric(raw["Price"]),          # investing.com เรียกราคาปิดว่า Price
        "Volume": pd.to_numeric(raw[VOLUME_COL]) * 1000,
    })
    df.index = pd.to_datetime(raw["Date"], format=DATE_FORMAT)
    df.index.name = "Date"
    # clean แบบเดียวกับ legacy: ตัดวันซ้ำ, เรียงเวลา, ตัดแถวที่ Close ว่างหรือ <= 0
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    if verbose:
        print(f"[data] {ticker}: {len(df)} แถว ({df.index[0].date()} → {df.index[-1].date()})")
    return df


@dataclass(frozen=True)
class Dataset:
    """
    ข้อมูลของหุ้นหนึ่งตัวในโซนหนึ่ง — ทุก array/Series เรียงตาม `dates` ชุดเดียวกัน

    dates      วันทำการทั้งหมดในโซน (รวมแถวอุ่นเครื่องต้นชุด เพื่อให้ "ตำแหน่งแถว" คงที่ทุก config)
    y          target : Close(t)/Close(t−1) − 1       (NaN ที่แถวแรก)
    prev_close ราคาปิดวัน t−1 — ใช้แปลง return → ราคา (รู้ค่าตั้งแต่ก่อนทำนาย จึงไม่ leak)
    close      ราคาปิดวัน t (ความจริงที่ต้องทำนาย)
    feat29     29 feature เดิมจาก legacy (market shift(1) แล้ว + day-of-week)
    ok29       แถวที่ y และ feature 29 ตัวครบ (ไม่มี NaN) — แถวอุ่นเครื่องต้นชุดเป็น False
    """

    ticker: str
    zone_end: str
    prices: pd.DataFrame
    dates: pd.DatetimeIndex
    y: np.ndarray
    prev_close: np.ndarray
    close: np.ndarray
    feat29: pd.DataFrame
    ok29: np.ndarray

    def __len__(self):
        return len(self.dates)

    def pos_of(self, date):
        return int(self.dates.get_loc(pd.Timestamp(date)))

    def raw_images(self, cfg: Config, positions):
        """
        **ค่าดิบ** ของภาพ (ยังไม่บีบอัด ยังไม่ normalize) ของวันที่ตำแหน่ง `positions`
        คืน (X, has) — การแปลงเป็นภาพจริงต้องใช้ ImageTransform ที่ fit จาก train ของ fold (ดู fold_inputs)
        """
        arr, first = channel_matrix(self.prices, cfg)
        return build_images(arr, first, positions, cfg)

    def usable(self, cfg: Config):
        """
        bool array : แถวที่ใช้เทรน/ประเมินได้สำหรับ cfg นี้
        = y ไม่ NaN · มีหน้าต่างภาพครบ · (ถ้าเป็น hybrid) feature 29 ตัวครบ
        """
        arr, first = channel_matrix(self.prices, cfg)
        pos = np.arange(len(self))
        has_img = pos - cfg.window >= first
        ok = np.isfinite(self.y) & has_img
        if cfg.needs_features:                 # hybrid และ hybrid-noCNN ใช้แถวชุดเดียวกัน (B25 ข้อ 2)
            ok &= self.ok29
        return ok


def dataset_from_prices(ticker, prices, zone_end, verbose=False):
    """
    OHLCV DataFrame → Dataset · **ตัดข้อมูลหลัง zone_end ออกก่อนทำอย่างอื่นทั้งหมด**

    ลำดับสำคัญ (เหมือน legacy/tune.py : load_dev_data)
        ราคา → ตัดที่ zone_end → build_features / build_targets → assert
    ถ้าสลับลำดับ (สร้าง feature ก่อนตัด) ค่า rolling/ewm ที่ขอบจะเห็นข้อมูลอนาคต = leak

    แยกออกมาจาก `build_dataset` เพื่อให้ leak test ป้อนราคาที่ถูกแก้เข้ามาได้ โดยใช้เส้นทางเดียวกันเป๊ะ ๆ
    """
    end = pd.Timestamp(zone_end)
    prices = prices.loc[:end].copy()
    assert prices.index[-1] <= end, "ตัดโซนไม่สำเร็จ — หยุดทันที"

    feat = build_features(prices, verbose=False)            # legacy: market shift(1) + dow
    assert feat.shape[1] == N_LEGACY_FEATURES, f"feature เดิมต้องมี {N_LEGACY_FEATURES} ตัว ได้ {feat.shape[1]}"
    tg = build_targets(prices, verbose=False)               # legacy: y_return = Close/prev_close − 1

    y = tg["y_return"].to_numpy(dtype=float)
    ok29 = np.isfinite(y) & feat.notna().all(axis=1).to_numpy()
    ds = Dataset(ticker=ticker, zone_end=str(end.date()), prices=prices, dates=prices.index,
                 y=y, prev_close=tg["prev_close"].to_numpy(dtype=float),
                 close=tg["close"].to_numpy(dtype=float), feat29=feat, ok29=ok29)
    assert ds.dates[-1] <= end
    if verbose:
        print(f"[data] {ticker}: {len(ds)} แถว · feature+target ครบ {int(ok29.sum())} แถว "
              f"(อุ่นเครื่องต้นชุด {int((~ok29).sum())} แถว)")
    return ds


def build_dataset(ticker, zone_end=VAL_END, verbose=False):
    """อ่านราคาจาก data/ แล้วสร้าง Dataset ของโซนที่กำหนด (ทางเข้าหลักที่ไฟล์อื่นเรียกใช้)"""
    return dataset_from_prices(ticker, load_prices(ticker, verbose=verbose), zone_end, verbose)


def build_all(zone_end=VAL_END, verbose=False):
    """Dataset ของทั้งสองหุ้น (เทรนแยกกัน แต่ใช้ปฏิทินวันประเมินชุดเดียวกัน)"""
    return {t: build_dataset(t, zone_end, verbose) for t in TICKERS}


@dataclass(frozen=True)
class Fold:
    """
    หนึ่ง evaluation window ของ walk-forward
    k            หมายเลข fold (1..N_FOLDS)
    eval_pos     ตำแหน่งแถวของช่วงประเมิน (ความยาว FOLD_DAYS)
    train_stop   train = ทุกแถวที่ตำแหน่ง < train_stop (= eval_pos[0]) → expanding window
    """

    k: int
    eval_pos: np.ndarray
    train_stop: int

    def train_pos(self, usable):
        """ตำแหน่งแถวที่ใช้เทรนได้จริง = แถวก่อนช่วงประเมิน ∩ แถวที่ข้อมูลครบของ config นั้น"""
        pos = np.arange(self.train_stop)
        return pos[usable[:self.train_stop]]


def make_folds(dates, n_folds=N_FOLDS, fold_days=FOLD_DAYS):
    """
    3 fold แบบ expanding · fold ละ `fold_days` วันทำการ ต่อกันและ **จบที่วันสุดท้ายของโซนพอดี**
    (DECISIONS ข้อ B6: ใช้ 3×266 = 798 วันท้ายของ dev)

    ตำแหน่งช่วงประเมินถูกกำหนดจาก "ปฏิทินของโซน" ไม่ใช่จากแถวที่ config ใช้ได้
    → ทุก config และทุกโมเดล (รวมตัวเทียบ) ประเมินบนวันชุดเดียวกันเสมอ ตาม SPEC ข้อ 4.2
    window ที่ยาวขึ้นจะตัดเฉพาะฝั่ง train ไม่กระทบช่วงประเมิน
    """
    n = len(dates)
    start = n - n_folds * fold_days
    if start <= 0:
        raise ValueError(f"โซนมี {n} วัน ไม่พอสำหรับ {n_folds} fold × {fold_days} วัน")
    folds = []
    for k in range(1, n_folds + 1):
        a = start + (k - 1) * fold_days
        folds.append(Fold(k=k, eval_pos=np.arange(a, a + fold_days), train_stop=a))
    assert folds[-1].eval_pos[-1] == n - 1, "fold สุดท้ายต้องจบที่วันสุดท้ายของโซน"
    for prev, nxt in zip(folds, folds[1:]):
        assert prev.eval_pos[-1] + 1 == nxt.eval_pos[0], "ช่วงประเมินต้องต่อกันไม่ซ้อนทับ"
    return folds


def fold_summary(ds, folds, cfg):
    """ตารางสรุปแต่ละ fold (จำนวนแถว train/eval และช่วงวันที่) — ไว้พิมพ์ให้ผู้ใช้ตรวจ"""
    usable = ds.usable(cfg)
    rows = []
    for f in folds:
        tr = f.train_pos(usable)
        ev = f.eval_pos
        assert usable[ev].all(), f"fold {f.k}: มีวันประเมินที่ข้อมูลไม่ครบ — หยุดและถามผู้ใช้"
        assert tr[-1] < ev[0], "fold ข้ามเวลา"
        rows.append({"ticker": ds.ticker, "fold": f.k, "train_rows": len(tr),
                     "train_first": str(ds.dates[tr[0]].date()), "train_last": str(ds.dates[tr[-1]].date()),
                     "eval_rows": len(ev), "eval_first": str(ds.dates[ev[0]].date()),
                     "eval_last": str(ds.dates[ev[-1]].date())})
    return pd.DataFrame(rows)


def fold_inputs(ds, cfg, fold, frac=EARLY_STOP_FRAC):
    """
    เตรียม input ของหุ้นหนึ่งตัวใน fold หนึ่ง — **ที่เดียวที่ประกอบ train/val/eval เข้าด้วยกัน**

    ลำดับที่ห้ามสลับ
        1. หาแถว train (ก่อนช่วงประเมิน) และแถว eval จากปฏิทินของโซน
        2. แบ่ง train เป็น fit (85% แรก) / val (15% ท้าย) ตามเวลา
        3. **fit ImageTransform จากแถว fit เท่านั้น** (DECISIONS ข้อ B8, B9)
        4. ค่อย transform ทั้ง train และ eval ด้วยสถิติชุดนั้น

    คืน dict : X_fit/X_val/X_eval (ภาพพร้อมใช้), y_fit/y_val/y_eval, pos_*, prev_close_eval, transform
    """
    usable = ds.usable(cfg)
    tr = fold.train_pos(usable)
    ev = fold.eval_pos
    assert usable[ev].all(), f"fold {fold.k}: วันประเมินต้องใช้ได้ทุกวัน"
    assert tr[-1] < ev[0], "fold ข้ามเวลา"

    fit_m, val_m = early_stop_split(len(tr), frac)
    raw_tr, has_tr = ds.raw_images(cfg, tr)
    raw_ev, has_ev = ds.raw_images(cfg, ev)
    assert has_tr.all() and has_ev.all(), "แถวที่เลือกมาต้องมีหน้าต่างครบทุกแถว"

    tf = ImageTransform(cfg).fit(raw_tr[fit_m])
    return {"pos_fit": tr[fit_m], "pos_val": tr[val_m], "pos_eval": ev,
            "X_fit": tf.transform(raw_tr[fit_m]), "X_val": tf.transform(raw_tr[val_m]),
            "X_eval": tf.transform(raw_ev),
            "y_fit": ds.y[tr[fit_m]], "y_val": ds.y[tr[val_m]], "y_eval": ds.y[ev],
            "prev_close_eval": ds.prev_close[ev], "close_eval": ds.close[ev],
            "dates_eval": ds.dates[ev], "transform": tf}


def early_stop_split(n_train_rows, frac):
    """
    แบ่ง train ของ fold เป็น (fit, val) โดย val = `frac` ท้ายสุดตามเวลา (SPEC ข้อ 4.2)
    คืน (mask_fit, mask_val) · ห้ามใช้ช่วงประเมินมาหยุดเทรนเด็ดขาด
    """
    n_val = max(1, int(round(n_train_rows * frac)))
    if n_val >= n_train_rows:
        raise ValueError("train น้อยเกินไปสำหรับ early stopping")
    fit = np.zeros(n_train_rows, dtype=bool)
    fit[: n_train_rows - n_val] = True
    return fit, ~fit
