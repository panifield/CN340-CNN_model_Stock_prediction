"""
config.py — งาน B (ราคาปิด / return)
====================================
ไฟล์ตั้งค่าของงาน B เท่านั้น แก้ที่นี่ไม่กระทบ task_a / task_c
"""

from pathlib import Path

# ---------------------------------------------------------------
# 0) Path — anchor จากโฟลเดอร์ของไฟล์นี้ (E2)
# ---------------------------------------------------------------
# ทำให้ `python task_b/main.py` กับ `cd task_b && python main.py`
# อ่าน/เขียนไฟล์ที่เดียวกันเสมอ
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
OUTPUT_DIR = BASE_DIR / "results"
LOCK_PATH = BASE_DIR / "PRE_TEST_LOCK.md"


# ---------------------------------------------------------------
# 1) ข้อมูลหุ้น (A1)
# ---------------------------------------------------------------
TICKERS = ["KBANK.BK", "ADVANC.BK"]

# ข้อมูลอ่านจาก raw_data/ เท่านั้น ไม่มีโหมด fallback
# ถ้าไฟล์หายต้อง error ทันที ไม่ใช่ได้ข้อมูลจากแหล่งอื่นมาแทนโดยไม่รู้ตัว
# แถวใหม่เข้า raw_data/ ได้ทางเดียว: tools/append_investing.py (ไฟล์ Investing หรือ Yahoo จาก
# tools/fetch_yahoo_daily.py ที่ตรวจ OHLC เป๊ะแล้ว) -- ชื่อ "investing" คือรูปแบบไฟล์
DATA_SOURCE = "investing"
RAW_DATA_SUFFIX = "_10Y_Cleaned.csv"


# ---------------------------------------------------------------
# 2) การแบ่งข้อมูล — freeze ด้วยวันที่ (0.1) ห้าม shuffle
# ---------------------------------------------------------------
# วันที่ชุดนี้ทำให้ train/val/test มีขนาดเท่าเวอร์ชันก่อนหน้า (1,688 / 362 / 362)
# จึงเทียบผลได้ว่าอะไรเปลี่ยนเพราะการแก้บั๊ก ไม่ใช่เพราะ split เลื่อน
# (ถ้าจะเปลี่ยนวัน ต้องเขียนเหตุผลไว้ตรงนี้)
SPLIT_BY_DATE = {
    "train_end": "2023-09-01",   # train: 2016-09-26 -> 2023-09-01   1,688 วัน
    "val_end":   "2025-02-25",   # val:   2023-09-04 -> 2025-02-25     362 วัน
}                                # test:  2025-02-26 -> 2026-08-28     362 วัน
# ขอบปลายของ historical test (ไม่ได้อยู่ใน SPLIT_BY_DATE เพราะเดิม test = "ทุกแถวหลัง val_end")
# เมื่อ append ข้อมูลใหม่ต่อท้าย raw_data/ แถวเหล่านั้นต้องไม่ไหลเข้า historical test เงียบ ๆ
# -> main.py โหมดเปิด test ตัดข้อมูลที่วันนี้ก่อนสร้าง feature (historical test = 362 แถวเดิม)
HISTORICAL_TEST_END = "2026-08-28"
TRAIN_RATIO = None               # ไม่ใช้แล้ว -- boundary มาจากวันที่เท่านั้น
VAL_RATIO = None


# ---------------------------------------------------------------
# 3) Feature engineering — เฉพาะงาน B
# ---------------------------------------------------------------
LAG_DAYS = [1, 2, 3, 5, 10]        # ย้อนหลังกี่วัน
MA_WINDOWS = [5, 10, 20]           # เส้นค่าเฉลี่ย
VOL_WINDOWS = [5, 20]              # ความผันผวน
RSI_PERIOD = 14

# ใส่ feature วันในสัปดาห์ไหม
USE_DAY_OF_WEEK = True

# งาน B ไม่ใช้ feature parity ย้อนหลัง (เฉพาะงาน A เท่านั้น)


# ---------------------------------------------------------------
# 4) โมเดล (C2)
# ---------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------
# *** ทั้ง 3 โมเดลด้านล่าง "จูนแล้ว" ด้วย protocol เดียวกัน (Phase 1C §4, ข้อ 11) ***
# ที่มา: tune.py รันวันที่ 2026-09-27 ตามแผนที่ commit ไว้ก่อนใน TUNING_PLAN.md
#        expanding window 3 folds x 2 หุ้น บน train+val (ไม่แตะ test)
#        17 configs ต่อโมเดลเท่ากัน (half-life 5 + hyperparameter 12)
#        ค่าชุดเดียวใช้ทั้ง KBANK และ ADVANC
#        จัดอันดับด้วย mean MAE_return ในบรรดา config ที่ผ่าน validity guard StdRatio >= 0.05
#        ผลเต็ม: results/tuning_scores.csv, tuning_stage*_*.csv, tuning_summary.md
# ค่า Phase 1 เดิม (ก่อนจูน) เขียนกำกับไว้ในแต่ละตัว -- เป็นจุดเริ่มของ stage 1
# ---------------------------------------------------------------

# ค่าเดิมของ v1 คือ (64,32) alpha=1e-3 ซึ่ง overfit หนัก
# (วัดใน rebuild นี้ ก่อนเปลี่ยนค่า: val R2_return = -2.39 KBANK / -2.37 ADVANC)
# จึงลดความจุและเพิ่ม L2:
#   (16,8) บน 29 features = ราว 625 พารามิเตอร์ ต่อข้อมูล 1,688 แถว
#   = ราว 2.7 แถวต่อพารามิเตอร์ ซึ่งยังตึง -> ต้องมี L2 ที่มีน้ำหนักจริง
#
#   alpha = 1.0 เป็นค่า L2 ที่กำหนดแบบ conservative ให้แรงกว่า default ของ sklearn
#   (1e-4) อย่างชัดเจน เพื่อจำกัดความซับซ้อนของ ANN ในเฟส 1
#
# ANN -- จูนแล้ว (ข้อ 11) · Phase 1 เดิม: hidden (16, 8), alpha 1.0
# ที่มา: results/tuning_stage2_params.csv รันวันที่ 2026-09-27 · 12 configs (เท่ากับ RF และ XGB)
#        ผู้ชนะ (3 seeds): alpha 7.0, hidden (8, 4) = 0.0092176323 mean MAE_return
#        ค่าที่แพ้ที่ใกล้ที่สุด: alpha 7.0, hidden (16, 8) = 0.0092171125 (ต่ำกว่า -5.2e-07
#        แต่ต่างไม่ถึง 1e-5 -> กฎเสมอเลือก hidden เล็กกว่า)
#        refit 10 seeds = 0.0091994160 (รายงานเท่านั้น) · min StdRatio 10 seeds = 0.0636
ANN_PARAMS = {
    "hidden_layer_sizes": (8, 4),
    "activation": "relu",
    "alpha": 7.0,
    "solver": "adam",
    "learning_rate_init": 1e-3,
    # เดิม 500 -- ANN ชน ceiling ทั้ง 20/20 fit (10 seeds x 2 หุ้น) โดย loss
    # ยังลดต่ออีก 0.1-1.8% ทุก 50 iterations แปลว่า optimizer ยังเดินอยู่จริง
    # ทดสอบด้วย ceiling 3000 แล้วทุก seed หยุดเองที่ 502-869 iterations
    # => 500 เป็น ceiling ที่ต่ำเกินไป ไม่ใช่ว่าโมเดล converge แล้ว
    #
    # การแก้นี้ *ไม่ใช่* การจูนบน validation -- เหตุผลมาจาก training convergence
    # ล้วนๆ (ConvergenceWarning + loss curve) ไม่ได้ดูค่า MAE ของ val เลย
    "max_iter": 3000,
    # ปิด early_stopping: MLPRegressor จะ shuffle แบ่ง val ของตัวเองออกจาก train
    # ซึ่งขัดกับหลัก "ห้าม shuffle" ของ time series
    "early_stopping": False,
    "n_iter_no_change": 20,
    "random_state": RANDOM_STATE,   # ถูกแทนด้วยแต่ละค่าใน ANN_SEEDS ตอนเทรนจริง
}

# seed averaging ไม่ใช่การจูน -- เป็นส่วนหนึ่งของ "นิยามโมเดล"
# ผลของ MLP ขึ้นกับการสุ่ม weight เริ่มต้น การรายงาน seed เดียว = เลือกผลที่ถูกใจได้
ANN_SEEDS = list(range(10))

# ต้นไม้ลึกจำ noise ได้ง่าย -> จำกัดความลึกและบังคับให้ leaf ใหญ่พอ
# (ไม่ใช่ค่า default ของ sklearn ซึ่งเป็น max_depth=None, min_samples_leaf=1
#  ซึ่งปล่อยให้ต้นไม้โตจนจำ training set ได้หมด)
#
# RF -- จูนแล้ว (ข้อ 11) · Phase 1 เดิม: depth 8, leaf 20, max_features 1.0 (default)
# ที่มา: results/tuning_stage2_params.csv รันวันที่ 2026-09-27 · 12 configs
#        ผู้ชนะ: depth 4, leaf 50, max_features 0.5 = 0.0091666015 mean MAE_return
#        ค่าที่แพ้ที่ใกล้ที่สุด: depth 4, leaf 20, max_features 0.5 = 0.0091671180 (+5.2e-07)
#        (depth 4 ทุกตัวเสมอกัน -> กฎเสมอเลือก depth ตื้นสุด แล้ว score ต่ำสุด)
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "min_samples_leaf": 50,
    "max_features": 0.5,
    # n_jobs=1 (เดิม -1) 2026-09-29: n_jobs=-1 ทำให้ลำดับรวมผลแบบขนานไม่คงที่
    # จริง -- ทดสอบรัน main.py --dev ซ้ำ 5 ครั้งด้วยข้อมูลเดียวกัน เจอ diff ที่
    # แถวนี้ 2 ใน 4 ครั้ง (ดู KNOWN_ISSUES.md) ทำให้ regression-check guard ของ
    # daily_next_day.py (git diff results/*_val.csv ต้องว่าง) false-positive
    # ได้แบบสุ่ม แม้ข้อมูลไม่เปลี่ยนเลย -- n_jobs=1 การันตี reproducible เป๊ะ
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
}

# depth ตื้น + lr ต่ำ + subsample = ค่ามาตรฐานสาย conservative สำหรับ tabular ที่ noise สูง
#
# XGB -- จูนแล้ว (ข้อ 11) · Phase 1 เดิม: depth 3, lr 0.05, subsample 0.8
# ที่มา: results/tuning_stage2_params.csv รันวันที่ 2026-09-27 · 12 configs
#        ผู้ชนะ: depth 2, lr 0.01, subsample 0.8 = 0.0091660517 mean MAE_return
#        ค่าที่แพ้ที่ใกล้ที่สุด: depth 2, lr 0.01, subsample 1.0 = 0.0091789446 (+1.29e-05)
#        min StdRatio ของผู้ชนะ = 0.0569 (ผ่าน guard 0.05 แบบเฉียด)
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    # n_jobs=1 (เดิม -1) 2026-09-29: เหตุผลเดียวกับ RF_PARAMS ด้านบน --
    # ทดสอบซ้ำเจอ diff ของแถวนี้เองด้วย (~1e-7, ใหญ่กว่าที่ models.py เคย
    # เขียนไว้ว่า "~1e-18" มาก)
    "n_jobs": 1,
}


# ---------------------------------------------------------------
# 5) การตรวจสอบ Data Leakage
# ---------------------------------------------------------------
# ถ้า feature ตัวไหนมี |correlation| กับ target return เกินค่านี้
# แปลว่าน่าจะ leak -> โปรแกรมจะเตือน
LEAK_CORR_THRESHOLD = 0.50


# ---------------------------------------------------------------
# 6) Lock ก่อนเปิด test (0.5)
# ---------------------------------------------------------------
# PRE_TEST_LOCK.md ต้องมีทุกหัวข้อนี้และมีค่าหลัง ":" จริง
# *** ห้ามสร้าง PRE_TEST_LOCK.md ตลอดเฟส 1 ***
REQUIRED_LOCK_FIELDS = [
    "DATASET:", "TRAIN_END:", "VAL_END:", "FEATURE_SET:",
    "MODELS:", "HYPERPARAMS:", "PRIMARY_METRIC:",
    "BASELINES:", "EXPECTED:", "LOCK_DATE:",
]



# ---------------------------------------------------------------
# 7) ป้ายกำกับชุดค่าโมเดลที่ใช้อยู่ -- บันทึกลง prediction_log ทุกแถว (§1.5)
# ต้องเปลี่ยนค่านี้ทุกครั้งที่แก้ ANN_PARAMS / RF_PARAMS / XGB_PARAMS
# หรือ RECENCY_HALF_LIFE เพื่อให้แยกได้ว่าแถวไหนมาจากโมเดลชุดไหน
# ---------------------------------------------------------------
CONFIG_TAG = "phase1c-tuned"     # เดิม "phase1b-untuned" -- เปลี่ยนเพราะจูนแล้ว (§4.9)

if not CONFIG_TAG.strip():
    raise ValueError("config.CONFIG_TAG ต้องไม่ว่าง")


# ---------------------------------------------------------------
# 8) Recency weighting (feedback ข้อ 7) -- §3
# ---------------------------------------------------------------
# หน่วย = จำนวนแถว (วันทำการ) ที่น้ำหนักลดลงครึ่งหนึ่ง
# None = ไม่ถ่วงน้ำหนัก (พฤติกรรมเดิมทุกอย่าง)
#
# เป็น dict "ต่อโมเดล" เพราะ §4 stage 1 เลือก half-life แยกแต่ละโมเดล
# (ใช้ค่าเดียวกันทั้ง KBANK และ ADVANC -- เหมือน ANN_PARAMS/RF_PARAMS/XGB_PARAMS)
# key ต้องตรงกับชื่อใน models.get_regressors() ทุกตัวอักษร
# ค่าทั้งหมดถูกเลือกด้วย §4 stage 1 (results/tuning_stage1_halflife.csv, 2026-09-27)
# grid [None, 1000, 500, 250, 125] -> None ชนะทั้ง 3 โมเดล
#   ANN None 0.0098041 vs 500 0.0098395 (+3.5e-05)
#   RF  None 0.0091967 vs 1000 0.0092051 (+8.5e-06 < 1e-5 -> เสมอ, กฎเลือกถ่วงน้อยกว่า = None)
#   XGB None 0.0094121 vs 1000 0.0094812 (+6.9e-05)
# => recency weighting ไม่ช่วยภายใต้ protocol นี้ -- โค้ดยังอยู่ แต่ปิดตามผลจูน
RECENCY_HALF_LIFE = {
    "ANN (MLP)":     None,
    "Random Forest": None,
    "XGBoost":       None,
}


# ---------------------------------------------------------------
# ต้นทุนการซื้อขายบน SET (feedback ข้อ 10 / Phase 1C §5.2)
# ---------------------------------------------------------------
# *** ทุกค่าด้านล่างต้องใส่ลิงก์อ้างอิงกำกับก่อนส่งรายงาน (feedback ข้อ 4) ***
# อัตราค่าคอมมิชชันต่างกันตามโบรกเกอร์และประเภทบัญชี
# ค่าที่ใส่ไว้เป็นอัตราออนไลน์รายย่อยทั่วไป ไม่ใช่ค่าที่ใช้ได้กับทุกคน
# ถ้าหาอัตราจริงของโบรกเกอร์ที่อ้างอิงไม่ได้ ให้รายงานเป็นช่วง ไม่ใช่ค่าเดียว
# ใช้กับตาราง diagnostic ที่สามเท่านั้น -- ห้ามใช้เลือกโมเดล
SET_COSTS = {
    "commission_per_side": 0.0015,   # 0.15% -- ต้องอ้างอิงโบรกเกอร์
    "trading_fee":         0.00005,  # ค่าธรรมเนียมตลาด
    "clearing_fee":        0.00001,
    "regulatory_fee":      0.00001,
    "vat":                 0.07,     # VAT คิดบนค่าธรรมเนียมทั้งหมด
}

# ขั้นราคาต่ำสุด (tick size) ของหุ้น -- ต้องอ้างอิงประกาศของ SET
# ราคาที่ขยับได้น้อยกว่า 1 tick คือขยับไม่ได้จริงในตลาด
# (lo, hi, tick): lo <= price < hi · hi = None คือไม่มีเพดาน
SET_TICK_BANDS = [
    (0,    2,    0.01), (2,   5,   0.02), (5,   10,  0.05),
    (10,   25,   0.10), (25,  100, 0.25), (100, 200, 0.50),
    (200,  400,  1.00), (400, None, 2.00),
]
