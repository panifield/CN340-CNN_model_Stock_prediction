"""
test_data_loader_matches_legacy.py — ตัวอ่านไฟล์ของเราต้องให้ผลเท่ากับ legacy ทุกค่า
===================================================================================
ไฟล์นี้ทำอะไร: ชี้ `legacy/data_loader.py` ให้อ่านไฟล์เดียวกันใน data/ (ด้วย monkeypatch ไม่แก้ไฟล์ legacy)
แล้วเทียบผลกับ `src.data.load_prices` ทุกคอลัมน์ทุกแถว
ทำไมต้องมี: เราเขียนตัวอ่านเอง (เพราะ legacy อ่านจาก legacy/raw_data/ ที่เราไม่ copy มา)
ถ้าสองทางให้ค่าไม่เท่ากัน ตัวเทียบ MLP/RF/XGB จะไม่ได้ข้อมูลชุดเดียวกับ CNN = เทียบกันไม่ได้
"""

import pandas as pd

from src import data, paths
from src.settings import TICKERS

paths.use_legacy()
import data_loader  # noqa: E402  (legacy)


def test_load_prices_matches_legacy_loader(monkeypatch):
    # legacy หา path จาก RAW_DATA_DIR / (ticker ตัด ".BK" + RAW_DATA_SUFFIX)
    monkeypatch.setattr(data_loader, "RAW_DATA_DIR", paths.DATA_DIR)
    for t in TICKERS:
        mine = data.load_prices(t)
        theirs = data_loader.load_stock(f"{t}.BK", verbose=False)
        pd.testing.assert_frame_equal(mine, theirs, check_exact=True)


def test_legacy_features_on_our_prices_have_no_nan_after_warmup():
    """แถวที่ NaN ต้องอยู่ต้นชุดติดกันเท่านั้น (ไม่มีรูโหว่กลางชุด เช่นตัวหารเป็นศูนย์)"""
    for t in TICKERS:
        ds = data.build_dataset(t)
        bad = ~ds.ok29
        n_head = 0
        while n_head < len(bad) and bad[n_head]:
            n_head += 1
        assert bad.sum() == n_head, f"{t}: มี NaN กลางชุด {int(bad.sum() - n_head)} แถว — ตรวจ features.py"
        # 21 แถว = ตัวที่ต้องการข้อมูลย้อนหลังมากสุดคือ volatility_20d :
        #   return รายวันเริ่มมีค่าที่แถว 1 → rolling(20) ครบที่แถว 20 → shift(1) ของ market feature → แถว 21
        assert n_head == 21, f"{t}: คาดว่าแถวอุ่นเครื่อง 21 แถว (volatility_20d) ได้ {n_head}"
