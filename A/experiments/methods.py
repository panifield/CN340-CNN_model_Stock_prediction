"""
วิธีแปลงอนุกรมเวลาเป็นภาพแบบอื่น ๆ (นอกจาก STFT spectrogram) สำหรับการทดลอง

ทุก transform รับ window_feats (window, C) ซึ่งเป็นข้อมูลวัน t-window .. t-1 เท่านั้น
และ normalize/pad ภายในหน้าต่างของตัวเอง -> ไม่ใช้ข้อมูลอนาคต
"""
import numpy as np
from scipy.signal import stft

from src.data import to_spectrogram


def t_stft(w, nperseg=10, noverlap=8):
    """log-power spectrogram (แบบ main pipeline)"""
    return to_spectrogram(w, nperseg, noverlap)


def t_stft_complex(w, nperseg=10, noverlap=8):
    """STFT แบบเก็บ phase: ส่วนจริง + ส่วนจินตภาพ (2C channel)
    spectrogram ปกติทิ้ง phase ทำให้ไม่รู้ทิศทางขึ้น/ลง แบบนี้เก็บไว้ครบ"""
    out = []
    for c in range(w.shape[1]):
        _, _, z = stft(w[:, c], fs=1.0, nperseg=nperseg, noverlap=noverlap, window="hann",
                       boundary="even", padded=False, detrend="constant")
        out += [np.arcsinh(z.real * 1e3), np.arcsinh(z.imag * 1e3)]
    return np.stack(out).astype(np.float32)


CWT_SCALES = (1.5, 2, 3, 4, 6, 8, 11, 16)


def t_cwt(w, scales=CWT_SCALES, omega0=5.0):
    """Scalogram ด้วย Morlet wavelet: ความละเอียดเวลาเต็มทุกวัน (C, n_scales, window)"""
    out = []
    for c in range(w.shape[1]):
        x = w[:, c] - w[:, c].mean()
        rows = []
        for s in scales:
            tt = np.arange(-int(3 * s), int(3 * s) + 1)
            wav = np.exp(1j * omega0 * tt / s) * np.exp(-0.5 * (tt / s) ** 2) / np.sqrt(s)
            full = np.convolve(x, wav[::-1].conj(), mode="full")
            k0 = (len(wav) - 1) // 2  # ตัดให้ยาวเท่า x และตรงกลาง (เหมือน mode="same")
            rows.append(np.abs(full[k0:k0 + len(x)]) ** 2)
        out.append(np.log1p(np.array(rows) * 1e5))
    return np.stack(out).astype(np.float32)


def t_gaf(w, channels):
    """Gramian Angular Summation Field (C, window, window)
    ret_close -> แปลงเป็นเส้นราคา (cumsum) ก่อน เพราะ GAF นิยมใช้กับ price path"""
    out = []
    for c, name in enumerate(channels):
        x = np.cumsum(w[:, c]) if name == "ret_close" else w[:, c]
        lo, hi = x.min(), x.max()
        x = np.zeros_like(x) if hi - lo < 1e-12 else 2 * (x - lo) / (hi - lo) - 1
        phi = np.arccos(np.clip(x, -1, 1))
        out.append(np.cos(phi[:, None] + phi[None, :]))
    return np.stack(out).astype(np.float32)


def t_raw(w):
    """ไม่แปลงเป็นภาพ: ข้อมูลดิบ (C, 1, window) -> CNN ทำงานเหมือน 1D-CNN"""
    return w.T[:, None, :].astype(np.float32)


def build_images(df, window, channels, transform, nperseg=10, noverlap=8):
    feats = df[list(channels)].to_numpy(dtype=np.float64)
    fn = {
        "stft": lambda w: t_stft(w, nperseg, noverlap),
        "stft_complex": lambda w: t_stft_complex(w, nperseg, noverlap),
        "cwt": t_cwt,
        "gaf": lambda w: t_gaf(w, channels),
        "raw": t_raw,
    }[transform]
    X, y, d = [], [], []
    for t in range(window, len(df)):
        X.append(fn(feats[t - window:t]))
        y.append(df["parity"].iloc[t])
        d.append(df["Date"].iloc[t])
    return np.stack(X), np.array(y, dtype=np.float32), np.array(d)


def tail_features(df, window, channels, k=5):
    """feature ตาราง: ค่า k วันล่าสุดของแต่ละ channel (สำหรับ gradient boosting)"""
    feats = df[list(channels)].to_numpy(dtype=np.float64)
    return np.array([feats[t - k:t].ravel() for t in range(window, len(df))], dtype=np.float32)


def aux_labels(df, window):
    """label ช่วยสำหรับ pretrain: |return วัน t| (ใช้เป็น label ของชุด train เท่านั้น)"""
    r = df["ret_close"].abs().to_numpy()
    return np.array([r[t] for t in range(window, len(df))], dtype=np.float64)
