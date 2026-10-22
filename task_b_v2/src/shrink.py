"""
shrink.py — ปรับขนาดคำทำนายด้วยตัวคูณ c (SPEC ข้อ 5.6 หลังแก้ · DECISIONS B16 ข้อ 2, B18, B33)
===========================================================================================
ไฟล์นี้ทำอะไร
    เลือก c ∈ {0, 0.1, …, 1} จาก "ส่วน early-stop" (15% ท้ายของ train) ของแต่ละ evaluation ด้วยเกณฑ์ลด MAE_return
    แล้วคูณคำทำนายของช่วงประเมินด้วย c · เสมอกันเลือก c เล็กกว่า

ทำไมต้องมี
    งานนี้คาดว่าสัญญาณอ่อนมาก (SPEC ข้อ 12) โมเดลที่ "ไม่กล้าทำนาย" ถูกต้องแล้ว การหดคำทำนายเข้าหา 0 ด้วย c ที่เลือกจากข้อมูลนอกช่วงประเมิน
    คือวิธีที่ไม่ลำเอียงที่จะบอกว่า "สัญญาณที่เรียนมาช่วยนอก train ไหม" — ถ้า c = 0 บ่อย = ไม่ช่วย
    c มีผลต่อ relMAE/MAE ที่รายงานเท่านั้น **ไม่มีผลต่อ IC** (เมื่อ c > 0) จึงไม่กระทบการเลือกผู้ชนะที่ใช้ IC

ข้อควรรู้ (B18): ส่วน early-stop ถูกใช้เลือก epoch ไปแล้ว คำทำนายบนส่วนนี้จึงดูดีเกินจริงเล็กน้อย → c มีแนวโน้มสูงกว่าที่ควร
"""

import numpy as np

C_GRID = tuple(round(0.1 * i, 1) for i in range(11))          # 0.0, 0.1, …, 1.0


def choose_c(y_val, pred_val, grid=C_GRID):
    """c ที่ทำให้ MAE ของ (c · pred_val) เทียบ y_val ต่ำสุด · เสมอกัน → c เล็กกว่า (argmin คืนตัวแรกของกริดที่เรียงจากน้อยไปมาก)"""
    y = np.asarray(y_val, dtype=float)
    p = np.asarray(pred_val, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y_val กับ pred_val ต้องยาวเท่ากันและไม่ว่าง")
    maes = [float(np.mean(np.abs(c * p - y))) for c in grid]
    return float(grid[int(np.argmin(maes))])


def apply(pred, c):
    return c * np.asarray(pred, dtype=float)
