"""
settings.py — ค่าคงที่ของ protocol ที่ SPEC ข้อ 4 กำหนด (ห้ามแก้หลังเห็นผล)
==========================================================================
ไฟล์นี้ทำอะไร: เก็บ "ตัวเลขที่ตกลงกันแล้ว" ไว้ที่เดียว -- วันตัดโซน, จำนวน fold, ขนาด fold, seed, เกณฑ์ guard
ทำไมต้องมี: ถ้ากระจายอยู่หลายไฟล์ แล้วมีใครแก้ที่เดียว protocol จะเพี้ยนโดยไม่รู้ตัว
(ส่วนค่า hyperparameter ของโมเดลอยู่ใน src/config.py ซึ่งเป็นสิ่งที่ทดลองเปลี่ยนได้)
"""

TICKERS = ["KBANK", "ADVANC"]           # เทรนแยกต่อหุ้น (D3) · ไฟล์ data/<ticker>_10Y_Cleaned.csv

# ---- โซนข้อมูล (SPEC ข้อ 4.1) ----
VAL_END = "2025-02-25"                  # dev: ต้นข้อมูล -> วันนี้ · ห้ามใช้ข้อมูลหลังวันนี้ระหว่างออกแบบ
FINAL_START = "2025-02-26"              # final: ประเมินครั้งเดียวหลัง LOCK
FINAL_END = "2026-08-28"
# replay: จำลองทำนายรายวันช่วงเก็บ log · **17 วันทำการ** (DECISIONS B36 · ตรงกับช่วงเก็บ log ให้คะแนนครบทันที)
# วันหยุด SET ที่อยู่ในช่วงนี้ (ยืนยันจาก set_holidays.txt ของ repo ทีม + ข้อมูลจริง): 10-13 วันนวมินทรมหาราช · 10-16 วันหยุดพิเศษที่ SET ประกาศเพิ่ม
REPLAY_START = "2026-09-28"
REPLAY_END = "2026-10-22"
REPLAY_TRAIN_END = "2026-09-25"         # โมเดล replay เทรนด้วยข้อมูลถึงวันนี้

# ---- walk-forward ใน dev (SPEC ข้อ 4.2) ----
N_FOLDS = 3                             # expanding window 3 fold
FOLD_DAYS = 266                         # แต่ละ fold ประเมิน 266 วันทำการ ต่อกันจนถึง VAL_END
EARLY_STOP_FRAC = 0.15                  # early stopping = 15% ท้ายของ train ใน fold นั้น (ไม่ใช้ช่วงประเมิน)

# ---- seed ----
SEARCH_SEEDS = (0, 1, 2)                # ระหว่างค้นหา: เฉลี่ยคำทำนาย 3 seeds
REFIT_SEEDS = tuple(range(10))          # ผู้ชนะ refit 10 seeds
HPO_SAMPLER_SEED = 0                    # seed ของการสุ่ม config ใน HPO (ข้อ 5.6)

# ---- ธง StdRatio (SPEC ข้อ 4.2 หลังแก้ตาม DECISIONS B16) ----
# เดิมเป็น guard ที่ตัดสิทธิ์ config · ตอนนี้ **เป็นแค่ธงรายงาน** ไม่ใช้ตัดสิทธิ์
# (StdRatio ต่ำกว่านี้ = คำทำนายแทบคงที่ แต่โมเดลที่ดีที่สุดเท่าที่เป็นไปได้ก็อาจได้ StdRatio = ρ < 0.05 — SPEC ข้อ 12.1)
GUARD_MIN_STDRATIO = 0.05

# ---- sanity controls (SPEC ข้อ 5.0) ----
POS_CONTROL_MIN_R2 = 0.3
NEG_CONTROL_MIN_RELMAE = 0.99
