"""
spectrogram.py — แปลงหน้าต่างอนุกรมเวลาเป็น "ภาพ" ด้วย STFT (SPEC ข้อ 5.1–5.2)
==============================================================================
ไฟล์นี้ทำอะไร
    1. `raw_channels(prices)` : สร้างช่องสัญญาณดิบจาก OHLCV (ret_close, range_hl, chg_vol, gap_open, detrend_ma20)
    2. `raw_coeffs(window, cfg)` : หน้าต่าง (L, C) → ค่าดิบของ STFT (C_img, F, T)
    4. `ImageTransform` : บีบอัดช่วงค่า + normalize ด้วยสถิติที่ fit จาก train ของ fold เท่านั้น
    3. `build_images(ch, positions, cfg)` : ทำภาพของหลายวันพร้อมกัน + บอกว่าวันไหนหน้าต่างไม่ครบ

ทำไมต้องมี — ทำไมตั้งค่า STFT แบบนี้ (เหตุผลจาก SPEC ข้อ 2.4)
    * `boundary="even"`, `padded=False` : ถ้าไม่เติมขอบ หน้าต่าง Hann จะให้น้ำหนักวันล่าสุด = 0
      (วัดแล้วที่ window 128 / nperseg 16: วันล่าสุดได้น้ำหนัก 0%, วันก่อนหน้า 0.7%) → ข้อมูลวันล่าสุดหายไปเฉย ๆ
      การสะท้อนขอบใช้แค่ตัวเลขที่อยู่ในหน้าต่าง (อดีตทั้งหมด) จึงไม่ leak
    * `detrend="constant"` : หักค่าเฉลี่ยของแต่ละ segment ออก (ตัดองค์ประกอบความถี่ 0 ที่ไม่ให้ข้อมูล)
    * ชนิด complex เก็บ real + imag แยกช่อง เพราะ log-power ของ x กับ −x เท่ากันทุกจุด → บอกทิศทางไม่ได้
      (ported from A/experiments/methods.py : t_stft_complex และ task_b/phase2/spectrogram.py, branch task_b)

*** ไฟล์นี้ไม่รู้จักวันที่/ตลาด — รับ array แล้วคืนภาพ ***
ความ causal (แถววันที่ t ใช้ข้อมูลถึง t−1 เท่านั้น) เป็นหน้าที่ของ src/data.py ที่เลือกช่วงหน้าต่างให้
"""

import numpy as np
import pandas as pd
from scipy.signal import stft

from src.configs import CHANNEL_SETS

# ค่าที่ตายตัวทุกการทดลอง (SPEC ข้อ 5.1) · fs = 1 (1 ตัวอย่าง = 1 วันทำการ)
STFT_KW = dict(fs=1.0, window="hann", detrend="constant", boundary="even", padded=False)

COMPLEX_SCALE = 1e3        # arcsinh(z·1e3) (ported from A/experiments/methods.py)


def raw_channels(prices):
    """
    OHLCV DataFrame → DataFrame ของช่องดิบทุกชนิดที่ Stage 1 ใช้เลือก (ยังไม่ตัด NaN)

    ret_close     ln(C_t / C_{t−1})          ผลตอบแทนรายวัน (ช่องหลักของ base config)
    range_hl      ln(H_t / L_t)              ความกว้างของแท่งวันนั้น
    chg_vol       ln(V_t / V_{t−1})          การเปลี่ยนของปริมาณซื้อขาย
    gap_open      ln(O_t / C_{t−1})          ช่องว่างราคาเปิด
    detrend_ma20  C_t / MA20_t − 1           ราคาที่หัก trend (ตัวเทียบของ Stage 1)

    ทุกช่องเป็นอัตราส่วน/ลอการิทึม → stationary · ไม่ใช้ราคาดิบ (เหตุผลเดียวกับ legacy/features.py)
    แถวแรกเป็น NaN เพราะไม่มีค่าของวันก่อนหน้า (detrend_ma20 เป็น NaN 19 แถวแรก)
    """
    c, o, h, l, v = (prices[k] for k in ("Close", "Open", "High", "Low", "Volume"))
    prev = c.shift(1)
    ch = pd.DataFrame(index=prices.index)
    ch["ret_close"] = np.log(c / prev)
    ch["range_hl"] = np.log(h / l)
    ch["chg_vol"] = np.log(v / v.shift(1))
    ch["gap_open"] = np.log(o / prev)
    ch["detrend_ma20"] = c / c.rolling(20).mean() - 1.0
    return ch


def channel_matrix(prices, cfg):
    """คืน (ndarray (n, C) ของช่องที่ cfg เลือก, ตำแหน่งแถวแรกที่ทุกช่องมีค่า)"""
    names = CHANNEL_SETS[cfg.channels]
    ch = raw_channels(prices)[names]
    arr = ch.to_numpy(dtype=np.float64)
    finite = np.isfinite(arr).all(axis=1)
    if not finite.any():
        raise ValueError("ไม่มีแถวใดที่ทุกช่องมีค่า finite")
    first = int(np.argmax(finite))
    if not finite[first:].all():
        bad = ch.index[first:][~finite[first:]]
        raise ValueError(f"มีค่าไม่ finite กลางชุดข้อมูล (เช่น Volume = 0 → ln(0)) ที่ {list(bad[:5])} — หยุดและถามผู้ใช้")
    return arr, first


def raw_coeffs(window, cfg):
    """
    หน้าต่าง (L, C) → **ค่าดิบ** ก่อนบีบอัดช่วงค่า (C_img, F, T)
      logpower : |Z|²                       1 ช่องต่อช่องดิบ
      complex  : Re Z, Im Z                 2 ช่องต่อช่องดิบ (เก็บ phase ไว้)
      cnn1d_raw: หน้าต่างดิบ (C, L)          ไม่ทำ STFT

    แยก "ค่าดิบ" ออกจาก "การแปลง" เพราะตัวบีบอัดต้องใช้สเกลที่ fit จาก train ของ fold เท่านั้น
    (ดู ImageTransform และ DECISIONS ข้อ B8/B9)
    """
    if cfg.arch == "cnn1d_raw":
        return window.T.astype(np.float32)
    out = []
    for c in range(window.shape[1]):
        _, _, z = stft(window[:, c], nperseg=cfg.nperseg, noverlap=cfg.noverlap, **STFT_KW)
        if cfg.spec_type == "logpower":
            out.append(np.abs(z) ** 2)
        else:
            out += [z.real, z.imag]
    raw = np.stack(out).astype(np.float32)
    if raw.shape != cfg.image_shape:
        raise AssertionError(f"ภาพที่ได้ {raw.shape} ไม่ตรงกับที่คำนวณไว้ {cfg.image_shape}")
    return raw


def build_images(arr, first_valid, positions, cfg):
    """
    ทำค่าดิบของหลายวันพร้อมกัน

    arr         : ndarray (n, C) ของช่องดิบ (จาก channel_matrix)
    first_valid : ตำแหน่งแถวแรกที่ทุกช่องมีค่า
    positions   : ตำแหน่งแถวของวันที่ต้องการ (วัน t)
    คืน (X, has) โดย
      X   : (len(positions),) + cfg.image_shape — แถวที่หน้าต่างไม่ครบเป็น NaN
      has : bool array บอกว่าวันนั้นมีหน้าต่างครบไหม

    *** หัวใจของความ causal : หน้าต่างของวัน t คือแถว [t−window, t−1] ***
    ช่องทุกช่องของแถว t−1 คำนวณจากราคาถึงวัน t−1 เท่านั้น → ไม่มีข้อมูลของวัน t หลุดเข้าภาพ
    """
    positions = np.asarray(positions, dtype=int)
    has = positions - cfg.window >= first_valid
    X = np.full((len(positions),) + cfg.image_shape, np.nan, dtype=np.float32)
    for i, (p, ok) in enumerate(zip(positions, has)):
        if ok:
            X[i] = raw_coeffs(arr[p - cfg.window:p], cfg)      # แถว t−window … t−1
    return X, has


class ImageTransform:
    """
    แปลง "ค่าดิบ" เป็นภาพที่ CNN เห็นจริง · **fit จากแถว fit (85% แรกของ train ใน fold) เท่านั้น** แยกต่อหุ้น

    ขั้นที่ 1 บีบอัดช่วงค่า (ค่าดิบมีหางยาวมาก)
        logpower  log1p(|Z|² / s)
        complex   arcsinh(Re Z / s), arcsinh(Im Z / s)
        cnn1d_raw ไม่บีบอัด (เป็น return อยู่แล้ว)
      s มาจาก `scale_mode`:
        "fit_median" → median(|ค่าดิบ|) ของแถว fit    ← ค่าตั้งต้น (DECISIONS ข้อ B9)
        "fixed"      → cfg.fixed_scale               ← complex ที่ port มาใช้ 1e-3 (= arcsinh(z·1e3))

      *** ทำไมต้องหารด้วยสเกลก่อน *** log1p(x) ≈ x เมื่อ x เล็กมาก · |Z|² จริงอยู่ราว 1e-5
      ถ้าไม่หาร log1p จะเป็นการแปลงเชิงเส้น (วัดได้ corr = 1.000000) คือไม่ได้บีบอัดอะไรเลย

    ขั้นที่ 2 normalize ตาม cfg.normalize
        "band"   ทีละแถบความถี่ (สถิติแยกต่อ channel × frequency)   ← ค่าตั้งต้น
        "global" สถิติตัวเดียวทั้งภาพ
        "image"  แต่ละภาพ normalize ด้วยสถิติของตัวเอง (ไม่ต้อง fit)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.scale = 1.0
        self.mean = 0.0
        self.std = 1.0

    # ---- ขั้นที่ 1 ----
    def compress(self, raw):
        """ขั้นบีบอัดช่วงค่าอย่างเดียว (ยังไม่ normalize) — เปิดเป็นสาธารณะไว้ให้ linear probe ตรวจว่าข้อมูลหายตรงไหน (DECISIONS B24)"""
        return self._compress(raw)

    def _compress(self, raw):
        if self.cfg.arch == "cnn1d_raw":
            return raw
        if self.cfg.spec_type == "logpower":
            return np.log1p(raw / self.scale)
        return np.arcsinh(raw / self.scale)

    @staticmethod
    def _band_axes(x):
        """
        แกนที่ยุบเพื่อคิดสถิติ "ทีละแถบความถี่" : ยุบแกนแถว (0) และแกนเวลา (แกนสุดท้าย)
        (N, C, F, T) → เหลือสถิติต่อ (C, F) · (N, C, L) ของ cnn1d_raw → เหลือสถิติต่อ (C,)
        """
        return (0, x.ndim - 1)

    def fit(self, raw_fit):
        """raw_fit : (n_fit,) + image_shape — ต้องเป็นแถว fit ของ fold นั้นเท่านั้น (ห้ามมีแถว eval ปน)"""
        if raw_fit.ndim != len(self.cfg.image_shape) + 1:
            raise ValueError(f"raw_fit ต้องมีรูป (n,) + {self.cfg.image_shape} ได้ {raw_fit.shape}")
        if not np.isfinite(raw_fit).all():
            raise ValueError("raw_fit มีค่าไม่ finite -- ต้องกรองแถวที่หน้าต่างไม่ครบออกก่อน")
        if self.cfg.arch != "cnn1d_raw":
            if self.cfg.scale_mode == "fit_median":
                m = float(np.median(np.abs(raw_fit)))
                if not m > 0:
                    raise ValueError("median ของค่าดิบเป็น 0 -- สเกลใช้ไม่ได้")
                self.scale = m
            else:
                self.scale = float(self.cfg.fixed_scale)
        c = self._compress(raw_fit)
        if self.cfg.normalize == "image":
            self.mean, self.std = 0.0, 1.0                   # คิดตอน transform จากภาพแต่ละภาพเอง
        else:
            axes = self._band_axes(c) if self.cfg.normalize == "band" else tuple(range(c.ndim))
            self.mean = c.mean(axis=axes, keepdims=True)     # รูป (1, C, F, 1) → broadcast ได้ทั้ง batch
            self.std = np.maximum(c.std(axis=axes, keepdims=True), 1e-8)
        return self

    def transform(self, raw):
        """raw : (n,) + image_shape → ภาพพร้อมเข้า CNN (float32)"""
        c = self._compress(raw)
        if self.cfg.normalize == "image":
            axes = tuple(range(1, c.ndim))                   # สถิติของภาพแต่ละภาพเอง
            mean = c.mean(axis=axes, keepdims=True)
            std = np.maximum(c.std(axis=axes, keepdims=True), 1e-8)
        else:
            mean, std = self.mean, self.std
        return ((c - mean) / std).astype(np.float32)

    def fit_transform(self, raw_fit):
        return self.fit(raw_fit).transform(raw_fit)

    def describe(self):
        return {"scale_mode": self.cfg.scale_mode, "scale": float(self.scale),
                "normalize": self.cfg.normalize,
                "n_stats": 1 if np.isscalar(self.mean) else int(np.asarray(self.mean).size)}


def hann_tail_weights(nperseg):
    """น้ำหนักของ "ตัวอย่างท้ายสุดในหนึ่ง segment" ตามหน้าต่าง Hann (สัดส่วนของค่าสูงสุด) — ตัวเลขดิบก่อนรวมทุก segment"""
    from scipy.signal import get_window
    w = get_window("hann", nperseg, fftbins=True)
    return float(w[-1] / w.max()), float(w[-2] / w.max())


def window_weights(cfg, boundary="even"):
    """
    "วันแต่ละวันในหน้าต่างถูกใช้มากแค่ไหน" — ไว้เป็นหลักฐานว่าทำไมต้องเติมขอบ (SPEC ข้อ 2.4)

    วิธีวัด: STFT (รวม detrend="constant") เป็น **ตัวดำเนินการเชิงเส้น** ของสัญญาณ
    จึงยิง impulse (เลข 1 ตัวเดียว) ที่วันที่ j แล้ววัดขนาดของผลลัพธ์ ‖Z‖ ได้ตรง ๆ
    = ส่วนที่วันที่ j มีผลต่อภาพทั้งภาพ · คืน array ความยาว window ที่ normalize ให้ค่าสูงสุด = 1

    ผลที่ควรเห็น: boundary=None → วันท้าย ๆ ของหน้าต่าง (เมื่อวาน ซึ่งสำคัญสุด) ได้น้ำหนักเกือบ 0
    """
    kw = dict(STFT_KW)
    kw["boundary"] = boundary
    w = np.zeros(cfg.window)
    for j in range(cfg.window):
        x = np.zeros(cfg.window)
        x[j] = 1.0
        _, _, z = stft(x, nperseg=cfg.nperseg, noverlap=cfg.noverlap, **kw)
        w[j] = np.linalg.norm(z)
    return w / w.max()
