"""
locked1600.py — ค่าที่ล็อกของโหมด 16:00 (ที่เดียวในโหมดนี้ · ห้ามแก้หลัง commit LOCK_1600.md)
=========================================================================================
ไฟล์นี้ทำอะไร: เก็บ protocol ของ PLAN_DRAFT.md ข้อ c/f และ **ช่อง config track 1** ไว้ที่เดียว
    `TRACK1_CONFIG`  = config track 1 ที่ล็อกจาก next_day (LOCK.md) · **ยังเป็น None จนกว่า Stage 6 จบ**
                       ห้าม hardcode config ที่อื่น — ทุกไฟล์เรียก `track_configs()` ซึ่งปฏิเสธถ้ายังว่าง
ทำไมต้องมี: ถ้าตัวเลขกระจายหลายไฟล์ แก้ที่เดียวแล้ว protocol เพี้ยนโดยไม่รู้ตัว (เหมือน src/settings.py ของ next_day)
"""

from dataclasses import replace

from src.configs import BASE

# ---- config track 1 ที่ล็อกจาก next_day (M22) ----
# Stage 6 ไม่มีผู้ชนะ → track 1 = base (LOCK.md หัวข้อ 5 · B40 · ผู้ใช้ยืนยัน 2026-10-24)
# **อ้าง BASE จาก src/configs.py โดยตรง ห้ามพิมพ์ค่าเอง** · test ตรวจว่าทุก field ตรงกับตาราง Track 1 ใน LOCK.md หัวข้อ 5
TRACK1_CONFIG = BASE
NEXTDAY_LOCK = "LOCK.md"                # LOCK ของ next_day (task_b_v2/LOCK.md) · LOCK_1600 บันทึก commit hash ของไฟล์นี้ (M23)

TICKERS = ("KBANK", "ADVANC")

# ---- walk-forward (PLAN ข้อ c · M10, M17) ----
WF_START = "2024-10-01"                 # วันแรกของช่วงประเมิน
WF_END = "2026-08-28"                   # วันสุดท้ายของช่วงประเมิน
SUBPERIODS = (("2024-10-01", "2025-02-25"), ("2025-02-26", "2026-08-28"))    # รายงานแยก (descriptive)
RETRAIN_EVERY = 20                      # เทรนใหม่ทุก 20 วันทำการ · ขอบบล็อกจากปฏิทินไฟล์รายวัน (ร่วมสองหุ้น)
                                        # จำนวนบล็อก = floor(วัน / 20) · เศษรวมเข้าบล็อกสุดท้าย (M20 ข: 461 วัน → 23 บล็อก · บล็อกท้าย 21 วัน)
SEEDS = tuple(range(10))                # 10 seeds เฉลี่ยคำทำนาย
EARLY_STOP_FRAC = 0.15                  # 15% ท้ายของ train = early stop + เลือก c

# ---- replay (PLAN ข้อ c, d · M13) ----
REPLAY_TRAIN_END = "2026-09-25"         # เทรนด้วยแถว ≤ วันนี้
REPLAY_START = "2026-09-28"
REPLAY_END = "2026-10-22"
REPLAY_FINAL_SNAPSHOT = "20261023/{ticker}_BK_1h_035532.csv"   # ทาง (i) · ผลหลัก
SNAPSHOT_WINDOW = ("16:00", "16:30")    # ทาง (ii) · เลือก snapshot "ล่าสุด" ที่ดาวน์โหลดในช่วงนี้ของวันนั้น (กฎทีม)

# ---- feature (PLAN ข้อ a · M4, M5) ----
FEATURES = ("r15", "gap", "range15", "vol15")
RANGE_HOURS = (10, 11, 12, 14, 15)      # ไม่รวมแท่ง 13 (M4)
VOL_HOURS = (11, 12, 14, 15)            # ไม่ใช้ V10 (ทีม) และ V13
VOL_LOOKBACK = 20                       # 20 eligible day ก่อนหน้า (M5)

# ---- F3 (PLAN ข้อ f) ----
BOOT_BLOCK = 20
BOOT_REPS = 10_000
BOOT_SEED = 0

# ---- ชื่อ track ----
MAIN, CONTROL, RF, NAIVE, MEAN = "t1600_cnn_ann", "t1600_ann_only", "Random Forest", "Naive", "Mean Return"
NN_TRACKS = (MAIN, CONTROL)
ALL_TRACKS = (MAIN, CONTROL, RF, NAIVE, MEAN)


def track_configs(track1=None):
    """
    config ของ track หลักและ control จาก config track 1 ที่ล็อก (M16: ไม่แก้ src · คง linear skip ทั้งสอง)
      หลัก   = track 1 + `ann_input="hybrid"`        → [CNN feature + 4 feature] + linear skip จาก 4 feature
      control = track 1 + `ann_input="hybrid_nocnn"`  → ANN เดียวกัน ไม่มี CNN (แบบ B25)
    track1 : ส่งมาเองได้ (ใช้ใน test) · None = ใช้ `TRACK1_CONFIG` ซึ่งต้องไม่ว่าง
    """
    cfg = TRACK1_CONFIG if track1 is None else track1
    if cfg is None:
        raise RuntimeError("TRACK1_CONFIG ยังว่าง — รอ Stage 6 จบและ LOCK.md ของ next_day ถูก commit (PLAN ข้อ e)")
    if cfg.train_mode != "e2e" or cfg.ann_input != "spec":
        raise ValueError(f"track 1 ที่ล็อกต้องเป็น e2e + spectrogram อย่างเดียว ได้ {cfg.train_mode}/{cfg.ann_input} — หยุดถามผู้ใช้")
    return {MAIN: replace(cfg, name=MAIN, ann_input="hybrid"),
            CONTROL: replace(cfg, name=CONTROL, ann_input="hybrid_nocnn")}
