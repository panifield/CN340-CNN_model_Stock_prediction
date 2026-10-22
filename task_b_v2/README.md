# task_b_v2 — Spectrogram → CNN → ANN ทำนายราคาปิด KBANK / ADVANC

งานรอบที่ 2 ของ Task B (CN340) · ทำตาม [`TASK_B_V2_SPEC.md`](TASK_B_V2_SPEC.md) · branch `task_b_v2` (แยกจาก `main`)

> **สถานะ: กำลังทำ — ผ่าน C0 (environment + ข้อมูล), C1 (dataset + leak test), C2 (โครงโมเดล + control) · C3 · Stage 1 (ใช้ base ต่อ) · Stage 2–5 (ไม่มีผู้ชนะ ยกเว้น hybrid ที่ผ่านเพราะ 29 feature) · ร่าง LOCK.md — หยุดที่ C5 รอผู้ใช้** · ผลทุกชิ้นก่อน LOCK เป็น development / walk-forward ไม่ใช่ test

## ไอเดียสั้น ๆ
1. เอา return รายวันย้อนหลัง (หน้าต่าง 128 วัน) ไปแปลงเป็น **spectrogram** (ภาพเวลา × ความถี่ ด้วย STFT)
   (บีบอัดด้วย `log1p(|Z|²/m)` แล้ว standardize ทีละแถบความถี่ — m และสถิติมาจาก train ของ fold นั้นเท่านั้น)
2. ส่งภาพเข้า **CNN** ให้สกัด feature 16 ตัว
3. ส่ง feature เข้า **ANN** ให้ทำนาย return ของวันถัดไป แล้วแปลงเป็นราคา `Close_pred(t) = Close(t−1) × (1 + ŷ)`
4. เทียบกับ Naive (ราคาเท่าเมื่อวาน) และโมเดลเดิม (MLP / Random Forest / XGBoost บน 29 feature)

## โครงโฟลเดอร์
| path | คืออะไร |
|---|---|
| `TASK_B_V2_SPEC.md` | คำสั่งงาน (ห้ามแก้) |
| `DECISIONS.md` | บันทึกการตัดสินใจทุกข้อ |
| `LOCK.md` | config ที่ล็อกก่อนประเมิน final (ยังไม่มี — สร้างที่ C5) |
| `data/` | ราคารายวัน + `SOURCES.md` (SHA-256, ช่วงวันที่) |
| `legacy/` | โมเดลเดิม **อ่านอย่างเดียว** (snapshot จาก branch `task_b`, ตรวจ SHA-256 ใน `tests/`) |
| `src/` | โค้ดหลัก (เป็น package — ดูหมายเหตุด้านล่าง) |
| `tests/` | leak test, shape test, control test, legacy snapshot |
| `results/stage*/` | ตารางผลแต่ละ stage (แต่ละ stage เขียนโฟลเดอร์ของตัวเอง ไม่เขียนทับของเก่า) |
| `results/experiment_log.csv` | ทุก config ที่ลอง (รวมที่แพ้) |

## วิธีรัน
ต้องอยู่ที่โฟลเดอร์ `task_b_v2/` และเรียกแบบ module เสมอ (ห้าม `python src/xxx.py` เพราะชื่อไฟล์ชนกับ legacy)

```bash
cd task_b_v2
pip install -r requirements.txt          # pin ตรงกับ legacy: numpy 2.1.1, pandas 2.3.3, scikit-learn 1.9.1, xgboost 3.4.1, torch 2.8.0 (CPU)
python -m src.check_env                  # C0: เวอร์ชัน library + ลายนิ้วมือข้อมูล
python -m src.check_data                 # C1: ขนาด dataset, fold, ขนาดภาพ, รูปตัวอย่าง
python -m src.check_controls             # C2: shape ทุกชั้น + positive/negative control (~5 นาที)
python -m src.run_c3 --jobs 6            # C3: base config เทียบ Naive/Mean/MLP/RF/XGB (~4 นาที)
python -m src.run_stage1 --jobs 6        # C4: Stage 1 การแทนข้อมูล (~10 นาที)
python -m src.run_placebo --jobs 6       # placebo test ของ Stage 1 + sign control (~28 นาที)
python -m src.signal_diagnostic          # diagnostic สัญญาณ (ไม่เทรน · ห้ามใช้เลือก feature/config)
python -m src.run_stage2 --jobs 6        # C4: Stage 2 โครงสร้าง (~2 นาที)
python -m src.run_stage345 3 --jobs 6    # Stage 3 input ของ ANN + placebo คู่ hybrid/control (~6 นาที)
python -m src.run_stage345 4 --jobs 6    # Stage 4 สองขั้น + cross-fitting (~5 นาที · ใส่ --estimate-only เพื่อวัดเวลาก่อน)
python -m src.run_stage345 5 --jobs 6    # Stage 5 loss (~3 นาที)
python -m src.daring_curve               # daring curve จากทุก config ที่รันแล้ว (รวม placebo)
python -m src.draft_lock                 # ร่าง LOCK.md จากผลจริง (C5)
python -m pytest                         # ทดสอบทั้งหมด (~1 นาที)
python -m pytest -m "not slow"           # ข้ามเทสต์ที่ต้องเทรนจริง
```

## กติกาที่ต้องจำ
- ใช้ข้อมูลถึง **2025-02-25** เท่านั้นระหว่างออกแบบ (ตัดทิ้งก่อนสร้าง feature / ภาพ แล้ว assert)
- ห้ามใช้ช่วงประเมินทำ early stopping / normalize / เลือก seed
- ห้ามแก้ config หลังเห็นผล final / replay
- Claude Code ไม่ commit/push — ผู้ใช้เป็นคน commit (3 commit ตามข้อ 9.1 ของสเปก)
