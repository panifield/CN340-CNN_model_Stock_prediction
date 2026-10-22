"""
test_leak.py — ข้อสอบข้อสำคัญที่สุด: ภาพและ feature ของแถว t ต้องไม่รู้จักราคาของวัน t (SPEC ข้อ 5.0)
================================================================================================
ไฟล์นี้ทำอะไร
    สุ่มเปลี่ยนราคาของวัน t และทุกวันหลังจากนั้น แล้วสร้าง dataset ใหม่ด้วยเส้นทางเดิม
    → ภาพ, 29 feature และ prev_close ของแถว 0..t ต้องเหมือนเดิมทุกบิต
    ส่วน y ของแถว t **ต้องเปลี่ยน** (เพราะ target คือ Close(t) จริง ๆ) — ถ้าไม่เปลี่ยนคือเทสต์ไม่ได้ทดสอบอะไร

ทำไมต้องมี
    ข้อผิดพลาดที่พบบ่อยที่สุดของงานทำนายราคาคือ feature ของวัน t เผลอใช้ราคาวัน t (เช่นลืม shift(1))
    ซึ่งทำให้ผลดูดีเกินจริงมากและจับด้วยตาไม่ได้ · เทสต์นี้จับได้ทันทีและจับได้ทุกช่องทาง
    (ตรวจทั้ง legacy features, ช่องดิบ, และ STFT ของทุก config ที่จะใช้ใน Stage 1)
"""

import numpy as np
import pytest
from dataclasses import replace

from src import data
from src.configs import BASE, variant

TICKER = "KBANK"
ZONE_END = "2025-02-25"

# config ที่ต้องผ่าน leak test ทุกตัว (ครอบทุกตัวเลือกของ Stage 1–2 ที่เปลี่ยนวิธีทำ input)
CONFIGS = [
    BASE,
    variant("win64", window=64, nperseg=16, hop=2),
    variant("win256", window=256, nperseg=32, hop=4),
    variant("complex", spec_type="complex"),
    variant("ch4", channels="ret4"),
    variant("detrend", channels="detrend"),
    variant("raw1d", arch="cnn1d_raw"),
]


@pytest.fixture(scope="module")
def base_ds():
    return data.build_dataset(TICKER, ZONE_END)


def _perturbed_prices(prices, pos, seed=0):
    """คูณ OHLC และ Volume ของแถว pos ขึ้นไปด้วยตัวเลขสุ่ม (คนละตัวทุกแถว/ทุกคอลัมน์)"""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    block = out.iloc[pos:]
    out.iloc[pos:] = block.to_numpy() * rng.uniform(1.05, 1.60, block.shape)
    return out


@pytest.mark.parametrize("day_from_end", [1, 50, 400])
def test_future_prices_do_not_change_past_inputs(base_ds, day_from_end):
    pos = len(base_ds) - day_from_end                    # แถวที่จะเริ่มเปลี่ยนราคา
    alt = data.dataset_from_prices(TICKER, _perturbed_prices(base_ds.prices, pos), ZONE_END)

    # 1) ราคาเปลี่ยนจริง (กันเทสต์ที่ "ผ่านเพราะไม่ได้เปลี่ยนอะไร")
    assert not np.allclose(base_ds.close[pos:], alt.close[pos:]), "ราคาไม่ได้ถูกเปลี่ยน — เทสต์นี้ไม่ได้ทดสอบอะไร"
    # 2) target ของแถว t ต้องเปลี่ยน (y ใช้ Close(t) ตามนิยาม)
    assert not np.isclose(base_ds.y[pos], alt.y[pos]), "y ของวันที่เปลี่ยนราคาต้องเปลี่ยน"

    # 3) แต่ทุกอย่างที่เป็น "input ของแถว 0..t" ต้องเหมือนเดิมทุกบิต
    np.testing.assert_array_equal(base_ds.prev_close[: pos + 1], alt.prev_close[: pos + 1])
    np.testing.assert_array_equal(base_ds.y[:pos], alt.y[:pos])
    a = base_ds.feat29.iloc[: pos + 1].to_numpy()
    b = alt.feat29.iloc[: pos + 1].to_numpy()
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_images_of_past_rows_do_not_change(base_ds, cfg):
    pos = len(base_ds) - 30
    alt = data.dataset_from_prices(TICKER, _perturbed_prices(base_ds.prices, pos, seed=1), ZONE_END)
    look = np.arange(pos - 5, pos + 1)                   # แถว t−5 … t

    xa, ha = base_ds.raw_images(cfg, look)
    xb, hb = alt.raw_images(cfg, look)
    assert ha.all() and hb.all(), "แถวที่ทดสอบต้องมีหน้าต่างครบ"
    np.testing.assert_array_equal(xa, xb, err_msg=f"LEAK! ภาพของ {cfg.name} เปลี่ยนเมื่อราคาอนาคตเปลี่ยน")

    # และภาพของแถว t+1 ต้องเปลี่ยน (หน้าต่างของมันกินแถว t ซึ่งถูกแก้) — ยืนยันว่าภาพไหวตามข้อมูลจริง
    xa2, _ = base_ds.raw_images(cfg, [pos + 1])
    xb2, _ = alt.raw_images(cfg, [pos + 1])
    assert not np.allclose(xa2, xb2), "ภาพของวันถัดไปควรเปลี่ยน — ถ้าไม่เปลี่ยนแปลว่าหน้าต่างไม่ได้ใช้แถว t−1"


def test_window_uses_only_rows_before_t(base_ds):
    """
    ตรวจตรง ๆ ว่าหน้าต่างของวัน t คือแถว t−window … t−1 : ภาพของวัน t ที่คำนวณจาก Dataset
    ต้องเท่ากับภาพที่สร้างมือจากช่องดิบช่วงนั้น (ซึ่งเห็นเฉพาะราคาถึงวัน t−1)
    """
    from src.spectrogram import channel_matrix, raw_coeffs
    arr, _ = channel_matrix(base_ds.prices, BASE)
    pos = len(base_ds) - 10
    manual = raw_coeffs(arr[pos - BASE.window:pos], BASE)
    got, has = base_ds.raw_images(BASE, [pos])
    assert has[0]
    np.testing.assert_array_equal(got[0], manual)


def test_rows_without_full_window_are_marked_unusable(base_ds):
    """แถวต้นชุดที่หน้าต่างไม่ครบต้องถูกตัดออก (ค่าเป็น NaN และ usable = False)"""
    X, has = base_ds.raw_images(BASE, np.arange(0, BASE.window + 5))
    assert not has[: BASE.window + 1].any(), "แถวที่ไม่มีหน้าต่างครบต้องถูกทำเครื่องหมายว่าไม่มีภาพ"
    assert np.isnan(X[0]).all()
    assert has[-1], "แถวที่หน้าต่างครบแล้วต้องมีภาพ"
    assert not base_ds.usable(BASE)[: BASE.window + 1].any()
