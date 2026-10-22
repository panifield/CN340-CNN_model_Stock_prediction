"""
targets.py — งาน B (ราคาปิด / return)
=====================================
สร้าง target ของงาน B เท่านั้น: return -> ทำนายราคาปิด

*** ทำไมงาน B ต้องทำนาย return ไม่ใช่ราคาดิบ ? ***

1. Tree-based models ไม่สามารถ extrapolate trend ออกนอก feature range
   ได้ตามธรรมชาติ
   Random Forest ถูกจำกัดด้วยค่าเฉลี่ยใน leaf ส่วน boosted trees อย่าง
   XGBoost แม้ไม่ถูก bound แบบเดียวกัน (เพราะเป็นผลรวมของหลายต้น)
   แต่ยังมีข้อจำกัดในการ extrapolate นอก distribution ที่เห็นระหว่าง train
   ถ้าเทรนช่วงราคา 100-150 แล้ว test ช่วง 150-200 กราฟจะแบนผิดปกติ

2. ราคาดิบเป็น non-stationary (มี trend)
   ส่วน return เป็น stationary -> โมเดลเรียนรู้ได้ถูกต้องกว่า

3. ถ้าทำนายราคาดิบจะได้ R² ~0.99 ซึ่งหลอกมาก
   เพราะโมเดลแค่เรียนรู้ว่า "พรุ่งนี้ ≈ วันนี้"

พอทำนาย return เสร็จ ค่อยแปลงกลับเป็นราคาด้วย
    Close_pred(t) = Close(t-1) * (1 + return_pred)
ซึ่ง Close(t-1) เรารู้อยู่แล้ว ณ เวลาทำนาย -> ไม่ leak
"""

import numpy as np
import pandas as pd


def build_targets(df, verbose=True):
    """
    คืน DataFrame ที่มี target งาน B + คอลัมน์ช่วยเหลือ
    ทุกแถว t อ้างอิงราคาปิดของวัน t
    """
    close = df["Close"]
    prev_close = close.shift(1)

    t = pd.DataFrame(index=df.index)

    # คอลัมน์ช่วยเหลือ (ไม่ใช่ target แต่ต้องใช้ตอนแปลงผลกลับ)
    t["close"] = close
    t["prev_close"] = prev_close

    # ---------- งาน B : ผลตอบแทน ----------
    t["y_return"] = close / prev_close - 1

    if verbose:
        print(f"[targets] สร้าง target งาน B (return) เรียบร้อย")

    return t
