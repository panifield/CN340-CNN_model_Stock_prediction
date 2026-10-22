"""
splits.py
=========
แบ่งข้อมูล train / validation / test

*** ห้ามใช้ train_test_split(shuffle=True) เด็ดขาด ***

ข้อมูลหุ้นเป็น time series ถ้า shuffle จะเกิดสถานการณ์
"เอาข้อมูลปี 2025 ไปเทรน แล้วกลับมาทำนายปี 2018"
ซึ่งเป็นไปไม่ได้ในโลกจริง และทำให้ผลดูดีเกินจริงมาก

แบ่งตามวันที่ที่ freeze ไว้ใน config.SPLIT_BY_DATE เท่านั้น (0.1)
    [-------- Train --------][-- Val --][-- Test --]
    2016-09-26      2023-09-01   2025-02-25   2026-08-28

โหมด --dev ไม่สร้าง test partition เลย (0.3)
"""

import numpy as np
import pandas as pd

from config import SPLIT_BY_DATE


def prepare_xy(X, y, extra=None, verbose=True):
    """
    จัด X และ y ให้ index ตรงกัน แล้วตัดแถวที่มี NaN ออก (A2)

    *** ฟังก์ชันนี้ต้องถูกใช้โดยทุกเส้นทางที่เตรียมข้อมูลเข้าโมเดล ***
    ทั้ง main.py (historical) และ predict_live.py (live)
    ห้ามเขียน logic การเตรียมข้อมูลขึ้นมาใหม่ที่อื่นเด็ดขาด
    ถ้าเขียนใหม่แล้วลืมตัดแถว warm-up บั๊ก A2 จะกลับมาเงียบๆ
    เฉพาะในเส้นทางนั้น โดยที่ pipeline หลักยังถูกอยู่ = จับไม่ได้

    NaN ช่วงต้นชุดเกิดจาก rolling indicator ที่ยังมีข้อมูลย้อนหลังไม่ครบ
    จึงเป็นค่าที่ "ยังนิยามไม่ได้" ไม่ใช่ missing value ทั่วไป
    แม้ SimpleImputer จะ fit จาก train เท่านั้น การเติม median ก็ไม่ได้ทำให้
    indicator นั้นมีความหมายขึ้นมา -> ตัดแถวทิ้งแทน
    """
    idx = X.index.intersection(y.dropna().index)
    X = X.loc[idx]
    y = y.loc[idx]

    ok = X.notna().all(axis=1)            # มี NaN แม้แต่ตัวเดียว = ตัดทิ้ง
    n_drop = int((~ok).sum())
    if n_drop and verbose:
        pos = np.where(~ok.values)[0]
        run = 0
        while run < len(pos) and pos[run] == run:
            run += 1
        if run == n_drop:
            print(f"[prepare] ตัด {n_drop} แถวอุ่นเครื่องต้นชุด (ต่อเนื่องกัน)")
        else:
            print(f"[prepare] !! เตือน: มี {n_drop - run} แถวที่ NaN อยู่กลางชุด "
                  f"-- ตรวจ features.py ว่ามีตัวหารเป็นศูนย์ไหม")
    X, y = X[ok], y[ok]

    if extra is not None:
        extra = extra.loc[X.index]
        return X, y, extra
    return X, y


def chronological_split(X, y, verbose=True, name="", include_test=True):
    """
    แบ่งตามเวลา คืน dict ของ (X, y) แต่ละชุด
    include_test=False -> คืนแค่ train / val (ไม่มี key "test")
    """
    assert X.index.equals(y.index), "index ของ X และ y ไม่ตรงกัน"
    assert X.index.is_monotonic_increasing, "index ต้องเรียงตามเวลา"

    train_end = pd.Timestamp(SPLIT_BY_DATE["train_end"])
    val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
    i_train = int((X.index <= train_end).sum())
    i_val = int((X.index <= val_end).sum())

    parts = {
        "train": (X.iloc[:i_train], y.iloc[:i_train]),
        "val":   (X.iloc[i_train:i_val], y.iloc[i_train:i_val]),
    }
    if include_test:
        parts["test"] = (X.iloc[i_val:], y.iloc[i_val:])

    for k, (Xp, _) in parts.items():
        assert len(Xp) > 0, f"ชุด {k} ว่างเปล่า"

    if verbose:
        print(f"[split] {name}")
        for k, (Xp, _) in parts.items():
            print(f"        {k:5s}: {len(Xp):5d} แถว  "
                  f"({Xp.index[0].date()} -> {Xp.index[-1].date()})")

    # ตรวจว่าไม่มีวันซ้อนทับกัน
    assert parts["train"][0].index[-1] < parts["val"][0].index[0]
    if include_test:
        assert parts["val"][0].index[-1] < parts["test"][0].index[0]

    return parts


def walk_forward_splits(X, n_splits=5, min_train=250):
    """
    (เฟส 2) Walk-forward validation
    ใช้ตอนอยากได้ผลที่น่าเชื่อถือกว่า split เดียว
    ต้องส่ง X ที่เป็น train+val เท่านั้นเข้ามา ห้ามรวม test

    fold 1: train[0:200]   test[200:250]
    fold 2: train[0:250]   test[250:300]
    ...
    """
    n = len(X)
    fold_size = (n - min_train) // n_splits
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            break
        yield (np.arange(0, train_end), np.arange(train_end, test_end))
