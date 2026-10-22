"""
baselines.py — งาน B (ราคาปิด / return)
=======================================
Baseline ของงาน B  <-- ส่วนที่สำคัญที่สุดของโปรเจกต์นี้

ถ้ารายงานบอกว่า "XGBoost ได้ MAE 1.08 ดีที่สุด"
แต่ไม่บอกว่า naive baseline ได้ 0.95
อาจารย์ถามคำถามเดียวก็จบ

Baseline ที่ต้องมี:
  - naive        : ทำนายว่าพรุ่งนี้ = วันนี้ (return = 0)  <-- คู่แข่งตัวจริง
  - mean return  : ทำนายด้วยค่าเฉลี่ย return ของ train
"""

import numpy as np


def baseline_naive_return(y_test):
    """
    Random walk: ทำนายว่าพรุ่งนี้ราคาเท่าวันนี้ -> return = 0
    ตัวนี้คือคู่แข่งหลักของงานทำนายราคา

    ถ้า MAE แพ้ naive ยังไม่มีหลักฐานจาก metric นี้ว่า ML ให้ประโยชน์
    เหนือ baseline -- ต้องพิจารณาความมีนัยสำคัญ (DM test) และมูลค่า
    เชิงเศรษฐศาสตร์เพิ่มเติม (เฟส 2)
    """
    return np.zeros(len(y_test))


def baseline_mean_return(y_train, y_test):
    """ทำนายด้วยค่าเฉลี่ย return ของ train set"""
    return np.full(len(y_test), y_train.mean())


def get_regression_baselines(y_train, y_test):
    """คืน dict {ชื่อ: prediction array}"""
    return {
        "Baseline: Naive (RW)": baseline_naive_return(y_test),
        "Baseline: Mean Return": baseline_mean_return(y_train, y_test),
    }
