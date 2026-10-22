"""
diagnostics.py — งาน B (ราคาปิด / return)
==========================================
วิเคราะห์ข้อมูลก่อนเทรน  <-- ส่วนที่จะทำให้รายงานดูแข็งแรง

*** รันบน train เท่านั้น (A3) ***
การวิเคราะห์ข้อมูลก่อนเทรนต้องใช้ข้อมูลที่โมเดลมีสิทธิ์เห็นเท่านั้น
main.py เรียกหลัง split และส่งแถวของ train มาเท่านั้น

มี 2 การวิเคราะห์:
  1. return_analysis - ขนาดของ return และราคา
  2. leak_check      - ตรวจหา feature ที่น่าสงสัยว่า leak
"""

import numpy as np
import pandas as pd

from config import LEAK_CORR_THRESHOLD


def return_analysis(targets, verbose=True):
    """วิเคราะห์งาน B: ขนาดของ return และราคา"""
    r = targets["y_return"].dropna()
    close = targets["close"].dropna()

    daily_std_baht = float((r.std() * close.mean()))

    stats = {
        "ret_mean": float(r.mean()),
        "ret_std": float(r.std()),
        "daily_move_baht": daily_std_baht,
    }

    if verbose:
        print("\n--- การวิเคราะห์งาน B (ราคา) ---")
        print(f"Return เฉลี่ย  : {r.mean()*100:.4f}% ต่อวัน")
        print(f"Return SD      : {r.std()*100:.4f}% ต่อวัน")
        print(f"= ประมาณ {daily_std_baht:.2f} บาท/วัน")
        print(f"Naive baseline (ทำนาย return = 0) "
              f"จะได้ MAE ราว {r.abs().mean()*100:.4f}%")

    return stats


def leak_check(X, y_return, verbose=True):
    """
    ตรวจหา feature ที่น่าสงสัยว่า leak

    วิธี: ดู correlation ระหว่าง feature กับ target return
    ถ้า correlate กับ "return" สูงเกินไป -> น่าสงสัย
    เพราะ return เป็นสิ่งที่ทำนายยากมาก
    """
    suspicious = []

    common = X.index.intersection(y_return.dropna().index)
    Xc = X.loc[common]
    yr = y_return.loc[common]

    if verbose:
        print("\n--- ตรวจสอบ Data Leakage ---")

    for col in Xc.columns:
        s = Xc[col]
        if s.nunique() <= 1:
            continue
        c = abs(s.corr(yr))
        c = c if pd.notna(c) else 0
        if c > LEAK_CORR_THRESHOLD:
            suspicious.append((col, c))

    if verbose:
        if suspicious:
            print(f"!! พบ {len(suspicious)} feature น่าสงสัย "
                  f"(|corr| > {LEAK_CORR_THRESHOLD}):")
            for col, c in sorted(suspicious, key=lambda x: -x[1]):
                print(f"   {col:35s} |corr| = {c:.3f}")
            print("   ตรวจสอบว่า feature พวกนี้ shift แล้วจริงหรือไม่")
        else:
            print(f"ผ่าน: ไม่มี feature ไหน correlate กับ return "
                  f"เกิน {LEAK_CORR_THRESHOLD}")

    return suspicious


def run_all_diagnostics(targets, X, verbose=True):
    """
    รันการวิเคราะห์ทั้งหมดรวดเดียว
    targets / X ต้องเป็นแถวของ train เท่านั้น
    """
    print("\n" + "#" * 78)
    print("#  การวิเคราะห์ข้อมูลก่อนเทรน (train เท่านั้น -- เอาผลส่วนนี้ใส่รายงานด้วย)")
    print("#" * 78)
    print(f"[diagnostics] ใช้ {len(X)} แถว "
          f"({X.index[0].date()} -> {X.index[-1].date()})")

    out = {"n_rows": len(X)}
    out["return"] = return_analysis(targets, verbose)
    out["leak"] = leak_check(X, targets["y_return"], verbose)
    return out
