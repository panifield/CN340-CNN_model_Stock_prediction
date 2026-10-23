# SPEC — Task B v2 : Spectrogram → CNN → ANN ทำนายราคาปิด KBANK / ADVANC

> เอกสารนี้คือ "คำสั่งงาน" สำหรับ Claude Code · ผู้ใช้เป็นคนตัดสินใจ Claude Code เป็นคนเขียนโค้ด
> ทุกการตัดสินใจในเอกสารนี้ผู้ใช้เลือกแล้ว — **ห้ามเปลี่ยนเองโดยไม่ถาม**
> ผลทุกชิ้นก่อนข้อ 4.4 (LOCK) เป็น development / walk-forward — ไม่ใช่ test

---

## 0. กติกาการทำงาน (สำคัญที่สุด อ่านก่อน)

1. **ผู้ใช้ต้องเข้าใจงาน** — ทุกไฟล์ที่สร้างต้องมี docstring ภาษาไทยอธิบายว่า "ไฟล์นี้ทำอะไร ทำไมต้องมี" ·
   ทุกครั้งที่เสร็จ checkpoint (ข้อ 9) ให้หยุด สรุปเป็นภาษาไทยง่าย ๆ ว่าทำอะไรไป ผลเป็นอย่างไร แล้วรอผู้ใช้ตอบ
2. **ห้ามตัดสินใจแทนผู้ใช้** — เจอเรื่องที่สเปกไม่ได้ระบุ หรือผลไม่ตรงที่คาด → หยุดและถาม พร้อมเสนอทางเลือก + ข้อดีข้อเสีย + คำแนะนำ
3. **บันทึกการตัดสินใจ** ทุกข้อใน `DECISIONS.md` (วันที่ · เรื่อง · ทางเลือก · เลือกอะไร · เพราะอะไร · ใครตัดสิน)
4. **บันทึกทุก config ที่ลอง** ใน `results/experiment_log.csv` (รวมที่แพ้) — จำนวนครั้งที่ลองต้องรายงานได้
5. **Claude Code ห้าม commit / push เอง — ผู้ใช้เป็นคน commit** ตามแผน 3 commit ในข้อ 9.1 · สื่อสารภาษาไทย · โค้ด/ชื่อไฟล์/คอลัมน์ภาษาอังกฤษ
6. **ทำแบบง่ายก่อน** — โค้ดต้องอ่านเข้าใจได้สำหรับนักศึกษาที่เพิ่งเรียน CNN (CS231n lecture 5–6) ·
   ไม่ต้องทำ framework ซับซ้อน · ไฟล์ละหน้าที่เดียว

## 1. เป้าหมาย

- ทำนายราคาปิดของ KBANK และ ADVANC ด้วย **Spectrogram → CNN สกัด feature → ANN ทำนาย** (โจทย์อาจารย์)
- CNN→ANN เป็น **โมเดลหลัก** · โมเดลเดิม (MLP / Random Forest / XGBoost บน 29 feature) และ Naive เป็น **ตัวเทียบ**
- 2 โหมด: **next_day (ทำนายก่อน 09:00)** ทำก่อน → **16:00 (ก่อนปิด 16:30)** ทำทีหลัง
- แยกโมเดลต่อหุ้น (เทรน KBANK กับ ADVANC แยกกัน)
- ส่งงานภายในวันเสาร์ 24 ต.ค. 2026 · พรีเซนต์ความคืบหน้าวันจันทร์ 26 ต.ค. 2026

## 2. ที่ทำงาน ข้อมูล และของที่ port มา

### 2.1 repo / branch / folder
- repo: `CN340-CNN_model_Stock_prediction` (StockCNN) · สร้าง branch ใหม่ **`task_b_v2` จาก `main`** (ไม่ใช่จาก branch `task_b` เดิม)
- เขียนไฟล์ได้เฉพาะใน **`task_b_v2/`** · ห้ามแก้ `A/`, `A2/`, `raw_data/` ที่ root, `README.md` ที่ root
- branch `task_b` เดิม (phase2) = **อ่านอ้างอิงได้ ห้ามแก้** (ผู้ใช้ commit ผลรอบก่อนไว้แล้ว) · ถ้าเอาโค้ดมาใช้ให้ copy แล้วใส่คอมเมนต์ที่มา
  เช่น `# ported from task_b/phase2/spectrogram.py (branch task_b)` หรือ `# ported from A/experiments/methods.py`
- คำสั่งเริ่มงาน (ทำแล้วใน C0 · ถ้ายังไม่ได้ทำในเครื่องผู้ใช้ให้ทำตามนี้):
  ```bash
  git switch main && git switch -c task_b_v2          # ฐาน = main commit e6c3ff0 (มีแค่ A/, A2/, raw_data/, README.md)
  git show task_b:task_b/phase2/spectrogram.py        # ดูไฟล์ของ branch เดิมโดยไม่ checkout
  git show task_b:<path> > task_b_v2/legacy/<file>     # copy แบบ byte-for-byte (ไม่แปลง line ending)
  ```
- ระวังชื่อ: repo นี้ใช้ `task_b` (ขีดล่าง) · repo `CN340_Stock_Prediction` ของทีมมี `task-b` (ขีดกลาง) คนละอันกัน

### 2.0 Environment
- ต้องมี torch (CPU พอ) + xgboost · ใช้ pin ตาม `task_b/requirements-cnn.txt` ของ branch เดิม
  (numpy 2.1.1, pandas 2.3.3, scikit-learn 1.9.1, xgboost 3.4.1, torch 2.8.0 CPU) เพื่อให้ legacy ให้ผลตรงเดิม
- **เทรนบนเครื่องผู้ใช้** (sandbox บน cloud ที่ใช้ทำ C0 ติดตั้ง torch ไม่ได้ เพราะ egress proxy ปฏิเสธ PyPI)
- torch: `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`, ตั้ง seed ทุกตัว (python/numpy/torch)

### 2.2 ข้อมูล
- ราคารายวัน: copy `task_b/raw_data/KBANK_10Y_Cleaned.csv` และ `ADVANC_10Y_Cleaned.csv`
  จาก repo `panifield/CN340_Stock_Prediction` branch `main` **หลังอัปเดตของวันที่ 22 ต.ค. 2026**
  (raw_data ที่ root ของ StockCNN จบที่ 2026-08-28 ซึ่งสั้นเกินไปสำหรับ replay)
  → วางที่ `task_b_v2/data/` · บันทึก SHA-256, วันแรก/วันสุดท้าย, จำนวนแถว ใน `data/SOURCES.md`
- **ฉบับที่ใช้จริง (refresh 2026-10-23 ก่อน commit 2 · DECISIONS B35/B36)** — copy จาก commit `6350406` (2026-10-23 04:31 +07:00)
  ตรวจแล้ว: เรียงเวลา ไม่ซ้ำ ไม่มีค่าว่าง · ปฏิทินสองหุ้นชุดเดียวกัน · SHA-256 เต็มและขั้นตอนการตรวจอยู่ใน `data/SOURCES.md`

  | ไฟล์ | แถว | ช่วง | SHA-256 (16 ตัวแรก) |
  |---|---|---|---|
  | KBANK_10Y_Cleaned.csv | 2,470 | 2016-08-26 → 2026-10-22 | `97fa5be2b1bfc2af` |
  | ADVANC_10Y_Cleaned.csv | 2,470 | 2016-08-26 → 2026-10-22 | `4a245577ea0178c5` |

  ตรวจตอน refresh: **2,467 แถวแรกตรงกับฉบับก่อน (จบ 2026-10-19) ทุกไบต์** และเพิ่มมา 3 แถว (10-20, 10-21, 10-22)
  · โซน dev ยังได้ 2,071 แถวเท่าเดิม → ผลทุก Stage ก่อน LOCK ไม่เปลี่ยน
  · ฉบับที่เคยบันทึกไว้ตอน C0: 2,468 แถว จบ 2026-10-20 (commit `cb6b7be`, SHA `d9623e23bc73ae1d` / `8fa4d580916515e3`) — ไม่ได้ใช้ เพราะ clone ในเครื่องผู้ใช้ยังไม่มี commit นั้นตอน C0
- รายชั่วโมง (โหมด 16:00 เท่านั้น): `task_b/raw_data_intraday/*_1h_730d.csv` จาก repo เดียวกัน
  (ครอบคลุม 2023-10-02 → 2026-09-25) + snapshot รายวันใน `raw_data_intraday_live/` สำหรับช่วง replay
- คอลัมน์: `Date, Price(=Close), Open, High, Low, Vol. ('000)` · รูปแบบวันที่ `%m/%d/%Y`

### 2.3 โมเดลเดิม (ตัวเทียบ)
- copy `task_b/legacy/` จาก branch `task_b` (snapshot commit `e49bbdd`, `CONFIG_TAG = "phase1c-tuned"` —
  ตรวจแล้วว่าพารามิเตอร์ตรงกับ main ปัจจุบัน) มาใช้ **แบบอ่านอย่างเดียว** เพื่อสร้าง 29 feature และโมเดล MLP/RF/XGB
- ใช้ hyperparameter เดิมตามที่จูนไว้ **ไม่จูนใหม่** แต่ต้อง **รันใหม่ภายใต้ protocol ของข้อ 4** เพื่อให้เทียบกับ CNN บนวันเดียวกัน

### 2.4 สิ่งที่ port จากงานเดิม (เหตุผลอยู่ในวงเล็บ)
| ของ | ที่มา | ทำไม |
|---|---|---|
| STFT `boundary="even", padded=False` | `A/`, `task_b/phase2/spectrogram.py` | ถ้าไม่ pad ขอบ หน้าต่าง Hann ให้น้ำหนักวันล่าสุด = 0 (ตรวจแล้ว: window 128, nperseg 16 → วันล่าสุดได้น้ำหนัก 0%, วันก่อนหน้า 0.7%) · การสะท้อนขอบใช้แค่ข้อมูลในอดีต ไม่ leak |
| spectrogram แบบ complex (real+imag, `arcsinh(z·1e3)`) | `A/experiments/methods.py` | log-power ของ x กับ −x เท่ากันทุกจุด → บอกทิศทางไม่ได้ · complex เก็บ phase ไว้ |
| positive / negative control | `task_b/phase2` (control_*) | พิสูจน์ว่า pipeline เรียนได้จริงและไม่ leak (ข้อ 5.0) |
| early stopping ด้วย 15% ท้ายของ train ใน fold | `task_b/phase2/settings.py` | ห้ามใช้ช่วงประเมินผลมาหยุดเทรน |
| cross-fitting (OOF) สำหรับแบบสองขั้น | `task_b/phase2/crossfit.py` | ข้อ 5.4 |
| linear skip ใน hybrid | `task_b/phase2/cnn.py` (Hybrid), A2 `cnn_ann_skip_mlp` | ให้โมเดลอย่างน้อยทำได้เท่าเชิงเส้นของ feature เดิม |
| Gaussian noise augmentation | `task_b/phase2/settings.py` (`cnn_noise`) | กัน overfit |
| Diebold–Mariano test | `task_b/phase2/stats.py` | ทดสอบว่าชนะ Naive อย่างมีนัยสำคัญไหม (ตัววัดรอง) |
| torch deterministic, CPU, 1 thread | `task_b/phase2/train_torch.py` | รันซ้ำได้ผลเดิม |

**ไม่ port:** factorial 2×2, ระบบ pre-registration ทั้งชุด, pooled 2 หุ้น — ซับซ้อนเกินไปสำหรับรอบนี้

## 3. นิยามโจทย์

### 3.1 next_day (09:00)
- แถวของวัน t: input ใช้ข้อมูลถึงวัน **t−1 เท่านั้น** · target `y = Close(t)/Close(t−1) − 1`
- ราคาที่ทำนาย `Close_pred(t) = Close(t−1) × (1 + ŷ)` · Naive: `ŷ = 0`

### 3.2 16:00 (Tier 2)
- target เหมือน Phase 1D รอบ 2 ของทีม: `y = Close_official(t)/C15(t) − 1` (C15 = ราคาปิดแท่ง 15:00 ของวัน t) · Naive: ราคาปิด = C15
- input: spectrogram รายวันชุดเดียวกับ next_day (ถึง t−1) + feature ของวันนี้ส่งเข้า ANN ตรง ๆ:
  `C15/Close(t−1) − 1`, `Open(t)/Close(t−1) − 1`, ช่วง high–low ถึงแท่ง 15:00, volume สะสมถึงแท่ง 15:00 เทียบค่าเฉลี่ย
- **ไม่ทำ** spectrogram จากแท่งรายชั่วโมง (มีแค่ ~730 วัน และเวลาไม่สม่ำเสมอเพราะพักเที่ยง/ข้ามคืน)
- ข้อจำกัด: ข้อมูลรายชั่วโมงเริ่ม 2023-10-02 → ถ้ายึดโซน dev ≤ 2025-02-25 จะเหลือแค่ ~340 วัน ซึ่งน้อยเกินไป
  ทางที่แนะนำ (ต้องให้ผู้ใช้ยืนยันก่อนเริ่มโหมดนี้): walk-forward ตลอดช่วง 2023-10 → 2026-08-28 แบบเดียวกับ Phase 1D
  โดย **ใช้ config ที่ล็อกจาก next_day ไม่จูนใหม่** แล้วติดป้ายว่า development / walk-forward + ทำ replay ช่วงเก็บ log

## 4. Protocol การประเมิน (ห้ามละเมิด)

### 4.1 โซน
| โซน | ช่วง | ใช้ทำอะไร |
|---|---|---|
| dev | ต้นข้อมูล → **2025-02-25** | ออกแบบ เลือก config จูน — ทุกอย่างในข้อ 5 |
| final | 2025-02-26 → 2026-08-28 | ประเมิน **ครั้งเดียว** หลัง LOCK |
| replay | 2026-09-28 → **2026-10-22** (17 วันทำการ) ✱ | จำลองการทำนายรายวันช่วงเก็บ log |

ระหว่างทำข้อ 5 ต้องตัดข้อมูลหลัง 2025-02-25 ทิ้ง **ก่อน** สร้าง feature/ภาพ แล้ว assert

### 4.2 walk-forward ใน dev
- 3 folds แบบ expanding: แต่ละ fold ประเมิน 266 วันทำการ ต่อกันจนถึง 2025-02-25 · train = ทุกแถวก่อนช่วงประเมิน
- **ทุก config ต้องประเมินบนวันชุดเดียวกัน** (window ยาวทำให้แถวแรก ๆ หาย → ตัดเฉพาะฝั่ง train ห้ามให้ช่วงประเมินต่างกัน)
- early stopping / scaler / normalization stats ใช้ **train ของ fold นั้นเท่านั้น** (early stopping = 15% ท้ายของ train)
- คะแนนเลือก config (Stage 1–4) = **ค่าเฉลี่ย IC (Pearson) ของ 6 evaluations** (2 หุ้น × 3 folds) · config เดียวใช้ทั้ง 2 หุ้น (แต่เทรนแยก)
  · ถ้าต่างจาก base **น้อยกว่า noise ของ seed** ให้คงค่า base · รายงาน Spearman IC คู่กัน
  · noise ของ seed = sqrt(ค่าเฉลี่ยของ σ² ข้ามทุก config ใน stage) โดย σ ของ config = std (ddof = 1) ข้าม seed ของ "IC เฉลี่ย 6 evaluations" ที่คิดจากคำทำนายของแต่ละ seed เดี่ยว ๆ (DECISIONS ข้อ B17)
- **relMAE (หน่วย return และบาท) ยังเป็นตัววัดหลักที่รายงานในทุกตาราง** · StdRatio = std(ŷ)/std(y) รายงานเป็น **ธง** (< 0.05 = ธงเตือนว่าคำทำนายแทบคงที่) **ไม่ใช้ตัดสิทธิ์**

> ✱ **แก้ข้อ 4.2 เมื่อ 2026-10-22 02:22 (ผู้ใช้ตัดสิน · DECISIONS ข้อ B16)** — เดิมเลือกด้วย mean relMAE และมี guard StdRatio ≥ 0.05 ตัดสิทธิ์ config
> เปลี่ยน **หลังเห็นผล C3 (base และตัวเทียบ) แต่ก่อนเห็นผล Stage 1 ใด ๆ** เหตุผล: (1) 89% ของช่องว่างระหว่างโมเดลกับ Naive มาจากวัน y = 0 (ราคาไม่เปลี่ยน 12–22% ของวัน) → relMAE ให้รางวัลกับการทำนายใกล้ 0 ไม่ได้วัดสัญญาณ · (2) guard 0.05 ขัดกับข้อ 12.1: โมเดลที่ดีที่สุดได้ StdRatio = ρ ซึ่งอาจ < 0.05
> base ที่ C3 ผ่าน guard เดิมแค่ 2/6 evaluation

> ✱ **เพิ่มข้อ 4.2 เมื่อ 2026-10-22 02:47 (ผู้ใช้ตัดสิน · DECISIONS ข้อ B21, B22)** — หลังเห็นผล Stage 1 (ก่อนเห็น Stage 2):
> (ก) **Stage 1** ตัดสินด้วย placebo: รัน 3 shuffle (seed 0, 1, 2) × 9 config ใช้กติกา B16/B17 เดิม · ถ้าจำนวนผู้ชนะเฉลี่ยต่อ shuffle ≥ 1 → ใช้ base ต่อ · ถ้า < 1 → ใช้ config รวม
> (ข) **Stage 2–5** ผู้ชนะต้องมี ΔIC > เปอร์เซ็นไทล์ 95 ของ "ΔIC สูงสุดต่อ shuffle" จาก placebo (ถ้า 3 shuffle น้อยเกินไปให้ใช้ค่าสูงสุดของ ΔIC ทุกตัวใน placebo แทน) **และ** ผ่านกติกา B16/B17 เดิมด้วย
- ระหว่างค้นหาใช้ 3 seeds เฉลี่ยคำทำนาย · ผู้ชนะ refit 10 seeds

### 4.3 ตัวเทียบ (รันทุก protocol เดียวกัน)
Naive · Mean Return · MLP / RF / XGB เดิม (29 feature, พารามิเตอร์เดิม)

### 4.4 LOCK แล้วประเมิน
1. เขียน `LOCK.md`: config ผู้ชนะของแต่ละ track (ดูข้อ 7) + จำนวน config ที่ลองทั้งหมด + SHA ของข้อมูล + commit
2. ให้ผู้ใช้อ่านและยืนยัน → ผู้ใช้ commit (commit ที่ 2 ของข้อ 9.1) → รอผู้ใช้แจ้งว่า commit แล้วจึงทำข้อ 3
3. final: เทรนบน dev ทั้งหมด (≤ 2025-02-25) ครั้งเดียว ทำนายทั้งช่วง final
4. replay: เทรนบนข้อมูลถึง 2026-09-25 ครั้งเดียว ทำนาย 2026-09-28 → **2026-10-22** (input ของแต่ละวันใช้ข้อมูลถึงวันก่อนหน้าเท่านั้น)

> ✱ **ช่วง replay สุดท้ายที่ใช้: 2026-09-28 → 2026-10-22 = 17 วันทำการ** (แก้ 2026-10-23 · ผู้ใช้กำหนด · DECISIONS ข้อ **B36**)
> เหตุผล: ข้อมูลถึง 2026-10-22 เข้า repo ทีมแล้ว (commit `6350406`) และ refresh ลง `data/` ก่อน commit 2 แล้ว → ช่วงนี้ **ตรงกับช่วงเก็บ log พอดีและให้คะแนนครบได้ทันที**
> (ระหว่างทางเคยกำหนดเป็น → 2026-10-21 = 16 วันทำการ ตอนที่ยังไม่มีข้อมูลวันที่ 22 — ยกเลิกแล้วด้วย B36)
> ยืนยันจำนวนวันจากข้อมูลจริง (17 แถวในทั้งสองไฟล์) + `set_holidays.txt` ของ repo ทีม: วันหยุด SET ในช่วงนี้คือ 10-13 (วันนวมินทรมหาราช) · 10-16 (วันหยุดพิเศษที่ SET ประกาศเพิ่ม)
5. รายงานต้องเขียนว่า: "ช่วง final เคยถูกใช้ประเมินโมเดลรุ่นก่อนของทีมมาแล้ว แต่ไม่เคยใช้ออกแบบหรือเลือก config ของ CNN รุ่นนี้" และ "replay มี 17 วัน ใช้สรุปทางสถิติไม่ได้"
6. **ห้ามกลับไปแก้ config หลังเห็นผล final/replay** — ถ้าจะแก้ต้องเป็นรอบใหม่ที่ประกาศชัด

## 5. Pipeline และการทดลอง

### 5.0 Sanity (ต้องผ่านก่อนทำอย่างอื่น)
- **leak test**: เปลี่ยนราคาวัน t (และหลังจากนั้น) แบบสุ่ม → ภาพและ feature ของแถว t ต้องไม่เปลี่ยน
- **positive control**: target = std ของ 5 return ล่าสุดในหน้าต่างของภาพเอง (เป็นฟังก์ชันของ input จึงต้องเรียนได้) → ต้องได้ R² ≥ 0.3 · ถ้าไม่ได้แปลว่า pipeline มีบั๊ก
- **negative control**: สลับ y ใน train แบบสุ่ม → relMAE ต้อง ≥ 0.99 · ถ้าต่ำกว่าแปลว่ามีอะไรรั่ว
- พิมพ์ shape ทุกชั้น + จำนวนพารามิเตอร์ (คำนวณมือเทียบด้วยสูตร L5: `(W − K + 2P)/S + 1`, params = `Cin·K·K·Cout + Cout`)

### 5.1 Base config (จุดเริ่มต้น ทุกการทดลองเปลี่ยนจากตรงนี้ทีละอย่าง)
| ส่วน | ค่า |
|---|---|
| channel | 1: `ret_close = ln(C_t/C_{t−1})` |
| window / nperseg / hop | 128 / 16 / 2 → ภาพ 9 × 65 |
| STFT | Hann, `detrend="constant"`, `boundary="even"`, `padded=False`, fs = 1 |
| ชนิด | log-power: `log1p(|Z|²/m)` โดย m = median ของ \|ค่าดิบ\| จากส่วน fit (85% แรก) ของ train ใน fold นั้น แยกต่อหุ้น **(แก้ 2026-10-22 — ดูหมายเหตุ ✱)** |
| normalize | standardize ทีละแถบความถี่ ด้วยสถิติจากส่วน fit (85% แรก) ของ train ใน fold นั้น |
| CNN | 2D: Conv(C→8, 3×3, pad 1)–BN–ReLU–MaxPool 2×2 → Conv(8→16, 3×3, pad 1)–BN–ReLU–MaxPool 2×2 → Flatten → Linear(→16)–ReLU = **CNN feature 16 ตัว** |
| ANN | [CNN feature 16] → Linear(32)–ReLU–Dropout 0.3 → Linear(1) |
| วิธีเทรน | end-to-end |
| loss | Huber (δ = 1 บน y ที่ standardize ด้วยสถิติ train) |
| optimizer | Adam lr 1e-3, weight decay 1e-3, batch 32, ≤ 200 epochs, patience 25, noise aug σ = 0.3 |

> ✱ **แก้ไขข้อ 5.1 เมื่อ 2026-10-22 (ก่อนเทรนโมเดลใด ๆ · ยังไม่เห็นผลของโมเดลเลย)** — ผู้ใช้อนุมัติ · บันทึกใน `DECISIONS.md` ข้อ B9
> สูตรเดิมเขียนว่า `log1p(|Z|²)` ซึ่ง **ไม่ได้บีบอัดแบบ log เลย**: |Z|² จริงอยู่ราว 1e-5 และ `log1p(x) ≈ x` เมื่อ x เล็ก
> วัดจากข้อมูลจริงได้ corr(log1p(x), x) = **1.000000** พอดี (ทั้ง KBANK และ ADVANC) → ภาพที่ได้คือ power ดิบที่หางยาว (p99/median ≈ 17–30)
> แก้เป็นหารด้วยสเกลที่ประมาณจากข้อมูล train ก่อน: `log1p(|Z|²/m)` → หางสั้นลงเหลือ p99/median ≈ 4.2–4.9
> m มาจาก **ส่วน fit (85% แรก) ของ train ใน fold นั้นเท่านั้น แยกต่อหุ้น** (ชุดสถิติเดียวกับที่ใช้ normalize ทีละแถบความถี่) จึงไม่เห็นช่วงประเมิน
> ตรวจ `complex` (`arcsinh(z·1e3)`) ด้วยวิธีเดียวกันแล้ว: บีบอัดจริง (corr 0.92–0.95, หาง 6–7.6 → 2.5) **จึงไม่แก้**

### 5.2 Stage 1 — การแทนข้อมูล (เปลี่ยนทีละปัจจัย เลือกผู้ชนะแล้วใช้ต่อ)
| ปัจจัย | ตัวเลือก |
|---|---|
| window / nperseg / hop | 64/16/2 · **128/16/2** · 256/32/4 |
| ชนิด | **log-power** · complex |
| normalize | **ทีละแถบความถี่** · global · ต่อภาพ (per-image) |
| channel | **1 (ret)** · 3 (ret + `ln(H/L)` + `ln(V_t/V_{t−1})`) · 4 (+ `ln(O_t/C_{t−1})`) |
| (ตัวเทียบ) | ราคาที่หัก trend `C/MA20 − 1` เป็น channel เดียว |

> ✱ **กติกาเพิ่มข้อ 5.2 (2026-10-22 · ผู้ใช้สั่ง · DECISIONS ข้อ B14):** เมื่อจบ Stage 1 ให้รันอีก 1 config ที่ **รวมผู้ชนะของทุกปัจจัย**
> (window · ชนิด · normalize · channel) เข้าด้วยกัน แล้วเทียบกับ base ด้วยคะแนนเดียวกัน (mean relMAE ของ 6 evaluations)
> **ถ้า config รวมแพ้ base (คะแนนสูงกว่า) ให้หยุดและถามผู้ใช้ ห้ามเลือกต่อเอง** — เพราะปัจจัยที่ชนะทีละตัวอาจไม่ได้ผลเมื่อรวมกัน
> (และเพราะชนะทีละปัจจัยด้วยความต่างที่เล็กกว่า noise ของ seed คือการเลือกจาก noise)

### 5.3 Stage 2 — โครงสร้าง
| ตัวเลือก | รายละเอียด |
|---|---|
| ก. 2D CNN | ตาม base |
| ข. 1D CNN ตามแกนเวลา | มองแถบความถี่ F แถว (× channel) เป็น channel: Conv1d(C·F→8, k=3)–BN–ReLU–MaxPool 2 → Conv1d(8→16, k=3)–BN–ReLU–MaxPool 2 → Flatten → Linear(→16) **(แก้ 2026-10-22: filter (8,16) แทน (16,16) — ดูหมายเหตุ ✱ ใต้ตาราง)** |
| ค. 1D CNN บน return ดิบ (ตัวเทียบ) | ไม่ทำ STFT · input = หน้าต่าง return ดิบ (C × window) |
| หัว CNN | Flatten (base) · Global Average Pooling |
| pretrained (Tier 2) | ResNet18 ImageNet แบบ freeze: ทำภาพให้เป็น 3 channel + resize 224×224 → ดึง feature 512 ตัวครั้งเดียว → ANN · ต้องใช้ torchvision และโหลด weights |

> ✱ **แก้ข้อ 5.3 (2026-10-22 · ผู้ใช้อนุมัติ · DECISIONS ข้อ B13):** ตัวเลือก ข. ใช้จำนวน filter **(8, 16)** เท่ากับ 2D CNN ของ base
> แทน (16, 16) ที่เขียนไว้เดิม เพื่อให้เทียบ 1D กับ 2D ได้ยุติธรรม (ต่างกันที่ "มิติของ convolution" อย่างเดียว ไม่ใช่ความจุของโมเดล)
> ตัวเลือก ค. (1D บน return ดิบ) สเปกไม่ได้ระบุ filter จึงใช้ (8, 16) ของ base เช่นกัน

### 5.4 Stage 3–4 — input ของ ANN และวิธีเทรน
- input ANN: Spectrogram อย่างเดียว · **hybrid** = [CNN feature 16 + 29 feature เดิม (standardize)] + linear skip จาก 29 feature
- วิธีเทรน: **end-to-end** · **สองขั้น** = (1) เทรน CNN ทำนาย y (2) freeze แล้วดึง CNN feature (3) เทรน ANN แยก
  - สองขั้นต้องใช้ **cross-fitting**: แบ่ง train เป็น 4 ช่วงต่อเนื่อง เทรน CNN บน 3 ช่วงแล้วสกัด feature ของช่วงที่เหลือ (OOF)
    เพราะถ้าใช้ feature ที่ CNN สกัดจากแถวที่มันเพิ่งเทรนมา ANN จะเชื่อ feature เกินจริง
- (ทางเลือก) ป้อน CNN feature ที่ได้ให้ XGBoost เทียบด้วย
- **hybrid-noCNN (control · เพิ่ม 2026-10-22 09:06 · ผู้ใช้สั่ง · DECISIONS ข้อ B25):** ANN ตัวเดียวกับ hybrid ทุกอย่าง (หัว ANN, linear skip จาก 29 feature, loss, optimizer, early stopping, seeds, การเทรน) แต่ **ไม่มี CNN** — input ของ ANN = 29 feature เดิม (standardize) อย่างเดียว
  - เป็น **control ไม่ใช่ตัวเลือก** (ไม่นับในเกณฑ์ B22) · **คำถามหลักของ Stage 3 = hybrid เทียบกับ hybrid-noCNN** (ไม่ใช่เทียบกับ MLP ของ sklearn)
  - ประกาศก่อนรัน: noise augmentation ใช้กับภาพเท่านั้นจึงไม่มีในตัว control · เทรน/ประเมินบนแถวชุดเดียวกับ hybrid
  - ✱ **นิยามสองขั้น (2026-10-23 00:08 · B32):** "CNN feature" ของสองขั้น = คำทำนาย (scalar) ของ CNN แบบ out-of-fold ไม่ใช่เวกเตอร์ 16 ตัว (เวกเตอร์จาก CNN คนละตัวใน cross-fitting เทียบกันไม่ได้) · ตัวเลือกเกณฑ์ hybrid vs hybrid-noCNN = placebo 5 shuffle (B28) · เพิ่ม P3b (B30)

### 5.5 Stage 5 — loss
MSE · MAE · **Huber** (ผู้ใช้เอียงไปทาง Huber · ถ้าคะแนนต่างกันน้อยกว่า noise ของ seed ให้เลือก Huber)

### 5.6 Stage 6 — Hyperparameter Optimization

> ✱ **ข้อ 5.6 ถูกข้ามในรอบนี้ (2026-10-23 00:08 · ผู้ใช้สั่ง · DECISIONS ข้อ B29):** ไม่รัน HPO 30 config · ให้สร้าง daring curve จากทุก config ที่รันไปแล้ว (รวม placebo) แทน · ส่วน shrinkage (ปรับขนาดด้วย c) ใช้ต่อใน Stage 5 และตอน final ตามข้อ B33
random search 30 config บนโครงที่ชนะ: lr {3e-4, 1e-3, 3e-3}, weight decay {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}, dropout {0, 0.1, 0.3, 0.5},
filters {(4,8), (8,16), (16,32)}, ANN hidden {16, 32, 64}, noise σ {0, 0.3}, patience {10, 25, 50} · seed ของการสุ่ม config = 0

**ต้องรู้ก่อนจูน (หลักฐานจากรอบก่อน ข้อ 12.1):** การทำนายใกล้ Naive ไม่ได้เกิดจาก hyperparameter ผิด แต่เป็นพฤติกรรมที่ถูกต้องเมื่อสัญญาณในข้อมูลน้อย
→ HPO รอบนี้มีไว้ **ตรวจสอบและแสดงให้เห็น** ไม่ใช่หวังให้ชนะ Naive · ห้ามเพิ่มงบค้นหาเพื่อ "หาจนกว่าจะชนะ"
- ช่วงค่า weight decay ต้องกว้างพอให้เห็นทั้งสองฝั่ง ถ้าผู้ชนะอยู่ที่ขอบของช่วง ให้รายงานผู้ใช้ (รอบก่อน alpha = 7 ซึ่งเป็นค่าสูงสุดของ grid)
- **daring curve** (รูปสำหรับสไลด์): จากทุก config ที่ลอง พล็อตแกน x = StdRatio, แกน y = relMAE · ใส่เส้นทฤษฎีของตารางข้อ 12.1 · ทำอีกรูปคู่กันที่แกน y = IC (เพราะ IC คือเกณฑ์เลือก)
- **การเลือกผู้ชนะ (Stage 5 และ HPO)**: เลือกด้วย mean IC (Pearson) ของ 6 evaluations เหมือน Stage 1–4 (ข้อ 4.2) แล้วจึง **ปรับขนาดคำทำนายด้วย c**
- **shrinkage (ปรับขนาด)**: นำคำทำนายของผู้ชนะคูณ c ∈ {0, 0.1, …, 1} โดยเลือก c จากส่วน early-stop (15% ท้ายของ train) ของแต่ละ fold · เกณฑ์ = ลด MAE_return บนส่วนนั้น เลือกแยกต่อ evaluation เสมอกันเลือก c เล็กกว่า (DECISIONS ข้อ B18) · ถ้า c ที่ดีที่สุดคือ 0 ส่วนใหญ่ = สัญญาณที่โมเดลเรียนมาไม่ช่วยนอก train · c ไม่กระทบ IC จึงไม่กระทบการเลือกผู้ชนะ มีผลต่อ relMAE/MAE ที่รายงานเท่านั้น

> ✱ **แก้ข้อ 5.6 เมื่อ 2026-10-22 02:22 (DECISIONS ข้อ B16, B18)** — เดิม c เป็นแค่ "control" ตอนนี้เป็นขั้นปรับขนาดของผู้ชนะ
- ห้ามบังคับให้โมเดล "กล้า" ขึ้นแบบประดิษฐ์ (เช่นขยายคำทำนาย หรือลงโทษ StdRatio ต่ำ) — จะได้ MAE แย่ลงแน่นอน

### 5.7 Tier 3 (ถ้ามีเวลา / งานต่อยอด)
- feature ใหม่: return ของดัชนี SET (`^SET.BK` จาก Yahoo — ตรวจว่ามีข้อมูลครบก่อน), วัน XD ของทั้งสองหุ้น
- Wavelet (CWT scalogram) — มีโค้ดใน `A/experiments/methods.py : t_cwt`
- ~~วิเคราะห์การปัดคำทำนายเป็น tick~~ → **ย้ายไป Tier 2 แล้ว (2026-10-22 · DECISIONS ข้อ B20)** ดูหัวข้อ 5.8

### 5.8 วิเคราะห์การปัดคำทำนายเป็น tick (Tier 2 · ย้ายมาจาก 5.7)
ปัดราคาที่ทำนายเป็นขั้นราคา (tick) แล้วดูว่า MAE / relMAE เปลี่ยนไหม · **ต้องใช้ตาราง tick ของ SET ตามระดับราคา ไม่ใช่ค่าเดียว**:

| ราคา (บาท) | < 2 | 2–5 | 5–10 | 10–25 | 25–100 | 100–200 | 200–400 | ≥ 400 |
|---|---|---|---|---|---|---|---|---|
| tick (บาท) | 0.01 | 0.02 | 0.05 | 0.10 | 0.25 | **0.50** | **1.00** | 2.00 |

แหล่ง: `legacy/config.py : SET_TICK_BANDS` และ `legacy/trading_costs.py : tick_size` · ต้องเทียบกับตารางทางการของ SET อีกครั้งตอนทำจริง
ข้อสังเกตจากข้อมูลวันประเมิน: KBANK ราคา 119–170.5 → tick 0.50 ทั้งหมด · ADVANC ราคา 184.5–304 → tick 1.00 ถึง 84% และ 0.50 อีก 16%
(ข้อความเดิม "ราคา 200–400 บาท tick = 1 บาท" จึงผิดสำหรับ KBANK ทั้งตัว) · รายละเอียดที่ต้องตัดสินตอนทำ: ใช้ tick ของราคาที่ทำนาย หรือของราคาปิดวันก่อน

## 6. ตัววัด
- **หลัก:** MAE (บาท), RMSE (บาท), **relMAE = MAE/MAE_Naive** (< 1 = ชนะ Naive)
- **รอง:** MAE_return, R²_OOS, direction accuracy (ไม่นับวันที่ return = 0), StdRatio, Diebold–Mariano vs Naive (บอก p-value และจำนวน evaluation ที่ชนะ)
- **IC (information coefficient)** = corr(ŷ, y) และ Spearman — วัด "สัญญาณ" โดยไม่ขึ้นกับขนาดของคำทำนาย
  (MAE ของ return รายวันถูก noise ครอบเกือบทั้งหมด สัญญาณเล็กแต่จริงจะเห็นใน IC ก่อน MAE) · รายงานเป็นค่าเฉลี่ย 6 evaluations + จำนวนที่เป็นบวก

## 7. Track ที่ต้องได้ผลออกมา (สำหรับตารางวันจันทร์)
1. **CNN→ANN หลัก** (spectrogram อย่างเดียว · ผู้ชนะ Stage 1–6)
2. **CNN→ANN hybrid** (+29 feature)
3. CNN สองขั้น (ผู้ชนะ)
4. ค. 1D CNN บน return ดิบ (ตัวเทียบ "spectrogram ช่วยไหม")
5. ตัวเทียบ: Naive, Mean, MLP/RF/XGB เดิม

## 8. ลำดับงานและ Tier
| Tier | งาน | เป้าวัน |
|---|---|---|
| 1 | ข้อ 2 setup → 5.0 sanity → 5.1 base → Stage 1–2 (ก, ข, ค) → 5.4 hybrid/สองขั้น → 5.5 loss → LOCK → final + replay (next_day) | พฤ.–ศ. |
| 2 | ~~Stage 6 HPO~~ (ข้าม) · daring curve จากทุก config ที่รันแล้ว (รวม placebo) | ศ.–ส. |
| 3 | **หลังพรีเซนต์:** pretrained ResNet18 · โหมด 16:00 · ปัดคำทำนายเป็น tick (ข้อ 5.8) · ข้อ 5.7 (feature ใหม่ · Wavelet) | หลังพรีเซนต์ |

> ✱ **ปรับแผนเมื่อ 2026-10-23 00:08 (ผู้ใช้สั่ง · DECISIONS ข้อ B29)** เพราะเวลาจำกัด: รัน Stage 3 → 4 → 5 ต่อกัน (ข้าม Stage 4 ถ้าประเมินแล้วเกิน 30 นาที) · ข้าม HPO · ResNet18 และโหมด 16:00 ย้ายเป็นงานหลังพรีเซนต์ · หลังจบ Stage 5 ร่าง LOCK.md แล้วหยุดรอผู้ใช้ที่ C5

ก่อนเริ่ม Stage 1 ให้วัดเวลาเทรน 1 config (6 evaluations × 3 seeds) แล้วรายงานผู้ใช้ เพื่อประเมินว่างบเวลาพอไหม

## 9. Checkpoint (หยุดรอผู้ใช้ทุกข้อ)
| # | ต้องแสดง |
|---|---|
| C0 | environment (เวอร์ชัน python/torch/scipy) · SHA และช่วงวันที่ของข้อมูล |
| C1 | shape ของ dataset · ภาพตัวอย่าง 3 วัน · ผล leak test |
| C2 | ผล positive/negative control |
| C3 | ผล base config เทียบตัวเทียบ · loss curve train/early-stop ของ 1 fold · เวลาที่ใช้ |
| C4 | ตารางผลของแต่ละ Stage + ผู้ชนะที่จะใช้ต่อ (ให้ผู้ใช้ยืนยันก่อนไป Stage ถัดไป) |
| C5 | `LOCK.md` ร่าง |
| C6 | ผล final + replay |
| C7 | รูปสำหรับสไลด์ (ข้อ 10) |

### 9.1 แผน commit (ผู้ใช้ทำเอง · 3 commit)
| # | เมื่อไหร่ | รวมอะไร | ข้อความ commit ที่เสนอ |
|---|---|---|---|
| 1 | หลัง C4 (จบ Stage สุดท้ายที่ทำ) | ทุกอย่างก่อน LOCK: SPEC, data, src, tests, DECISIONS.md, results ของ dev ทั้งหมด, experiment_log.csv | `task_b_v2: development pipeline and walk-forward results (dev <= 2025-02-25)` |
| 2 | หลัง C5 | **`LOCK.md` อย่างเดียว** | `task_b_v2: lock configs before final evaluation` |
| 3 | หลัง C6–C7 | ผล final + replay + figures + README ที่อัปเดต + SHA ข้อมูลใหม่ใน SOURCES.md | `task_b_v2: final and replay results, figures` |

กติกา:
- **commit 2 ต้องเกิดก่อนรัน final/replay** — Claude Code ห้ามรันขั้น C6 จนกว่าผู้ใช้จะบอกว่า commit `LOCK.md` แล้ว และ working tree สะอาด (`git status` ว่าง)
- ทุก checkpoint ยังต้องหยุดรอผู้ใช้เหมือนเดิม แม้จะไม่ commit
- ที่จุด commit ให้ Claude Code เสนอรายการไฟล์ + ข้อความ commit แล้วรอ · ห้ามรัน `git add`/`git commit`/`git push` เอง
- ระหว่าง commit 1 ยังไม่มีจุดย้อนกลับ → ห้ามลบหรือเขียนทับผลของ stage ก่อนหน้า (แต่ละ stage เขียนลงโฟลเดอร์ `results/stage*/` ของตัวเอง)
- ไฟล์ใหญ่/ชั่วคราว (cache, checkpoint ของโมเดล `.pt`) ใส่ใน `task_b_v2/.gitignore` ไม่ต้อง commit

## 10. Output
```
task_b_v2/
  README.md          อธิบายงาน + วิธีรัน (ภาษาไทย)
  DECISIONS.md       บันทึกการตัดสินใจ
  LOCK.md            config ที่ล็อกก่อนประเมิน final
  data/              ข้อมูล + SOURCES.md
  legacy/            โมเดลเดิม (อ่านอย่างเดียว)
  src/               data.py, spectrogram.py, models.py, train.py, evaluate.py, controls.py, ...
  tests/             leak test, shape test, control test
  results/
    experiment_log.csv
    stage*/          ตารางผลแต่ละ stage
    final/, replay/
    figures/
```
รูปสำหรับสไลด์: (1) แผนภาพ pipeline (2) ภาพ spectrogram ตัวอย่าง (price vs return) (3) แผนภาพโครง CNN + จำนวนพารามิเตอร์
(4) ตารางผล track ทั้งหมดเทียบ Naive (5) กราฟราคาทำนาย vs จริงช่วง replay (6) cumulative absolute error เทียบ Naive

## 11. ข้อห้าม
- ห้ามใช้ข้อมูลหลัง 2025-02-25 ระหว่างออกแบบ · ห้ามใช้ช่วงประเมินทำ early stopping / normalize / เลือก seed
- ห้ามเปลี่ยนนิยาม target / protocol / ตัววัดหลังเห็นผล
- ห้ามแตะ pipeline เก็บ log ของ `CN340_Stock_Prediction` · ห้ามเขียน `prediction_log*`
- ห้ามสลับลำดับเวลา (augmentation แบบกลับเวลา / shuffle ข้ามช่วงประเมิน) — shuffle batch ภายใน train ได้
- ห้ามเปลี่ยนเวอร์ชัน library ของ legacy เอง

## 12. ข้อมูลอ้างอิงจากรอบก่อน (ไว้ตั้งความคาดหวัง)
branch `task_b` เดิม (phase2, walk-forward ≤ 2025-02-25) พบว่า:
- next_day: ไม่มีโมเดลไหนชนะ Naive อย่างมีนัยสำคัญ (relMAE ดีสุด ≈ 0.9997–1.001) · การเพิ่ม CNN feature ไม่ต่างอย่างมีนัยสำคัญ · CNN อย่างเดียวแพ้ Naive อย่างมีนัยสำคัญ (0/6 evaluation)
- 16:00: ทุกโมเดลแพ้ Naive (relMAE 1.004–1.05)
- งาน A ของเพื่อน (คู่/คี่) ก็พบว่า spectrogram ไม่เพิ่มข้อมูลเกินกว่ากฎง่าย ๆ

### 12.1 ทำไมโมเดล "ไม่กล้าทำนาย" (วิเคราะห์ 2026-10-22 จากผลรอบก่อน)
1. **ยิ่งกล้า ยิ่งแย่** — จูน MLP ใน `CN340_Stock_Prediction/task_b/results/tuning_stage2_params.csv`:
   alpha 0.3 → StdRatio 0.33–0.61, MAE 0.00979–0.01070 · alpha 7.0 → StdRatio 0.11–0.12, MAE 0.00921 (ดีสุด และเป็นค่าสูงสุดของ grid)
2. **คูณ 0 ดีที่สุด** — shrinkage control ของ phase2: c = 0 (= Naive) ชนะ 13 ใน 18 กรณี
3. **ช่วงของ hyperparameter แทบไม่เปลี่ยนผล** — 12 config ต่อโมเดลใน phase2: relMAE ของ RF 0.9992–1.0010, ANN 1.0027–1.0070 · feature selection ช่วย 0.60% น้อยกว่าที่การสลับ y แบบสุ่มได้ (~2%)
4. **ทฤษฎี** — โมเดลที่ดีที่สุดเท่าที่เป็นไปได้ เมื่อสัญญาณมีความสัมพันธ์ ρ กับ return จะได้ StdRatio = ρ และ (จำลอง 2 ล้านจุด, noise หางยาว t₄):

   | ρ | relMAE ดีที่สุด | StdRatio | R² |
   |---|---|---|---|
   | 0.02 | 0.9997 | 0.02 | 0.0004 |
   | 0.05 | 0.9981 | 0.05 | 0.0025 |
   | 0.10 | 0.9925 | 0.10 | 0.0101 |
   | 0.20 | 0.9703 | 0.20 | 0.0403 |

   ρ ≈ 0.05 ถือว่าดีมากสำหรับ return รายวัน → relMAE ≈ 0.998 และ StdRatio ≈ 0.05 คือสิ่งที่ "โมเดลที่ดี" ควรได้ · StdRatio ที่เห็นในรอบก่อน (0.06–0.11) จึงสอดคล้องกับสัญญาณที่อ่อนมาก
5. **สรุป** — ตัวจำกัดคือ "ข้อมูลที่ input มี" ไม่ใช่ hyperparameter · ทางที่จะทำให้ดีขึ้นจริงคือ input ที่มีข้อมูลใหม่ (ข้อ 5.7) หรือโหมด 16:00 ที่รู้ราคาของวันนั้นแล้ว

สิ่งที่รอบนี้ **ต่างจากรอบก่อน**: window ยาวกว่าและมีหลายขนาด (เดิม 30 วันตายตัว) · 1D CNN ตามแกนเวลา · Flatten เทียบ GAP (เก็บตำแหน่งเวลา) · แยกต่อหุ้น (เดิม pooled) · ทดลอง normalization · pretrained ResNet18 · 1D CNN บน return ดิบเป็นตัวเทียบ · มีช่วง final + replay หลัง LOCK

## 13. บันทึกการตัดสินใจเริ่มต้น (ให้ copy ไปเป็นแถวแรกของ `DECISIONS.md`)

| วันที่ | เรื่อง | เลือก | เหตุผล | ผู้ตัดสิน |
|---|---|---|---|---|
| 2026-10-22 | branch | `task_b_v2` จาก `main` | main สะอาด · `task_b` ใช้อ่านอ้างอิงผ่าน `git show` | ผู้ใช้ |
| 2026-10-22 | D1 | CNN→ANN หลัก, โมเดลเดิมเป็นตัวเทียบ | — | ผู้ใช้ |
| 2026-10-22 | D2 | 09:00 ก่อน แล้ว 16:00 | ข้อมูลรายวัน 10 ปีพอให้ CNN เรียน | ผู้ใช้ |
| 2026-10-22 | D3 | แยกโมเดลต่อหุ้น · config เดียวใช้ทั้งสองหุ้น | ลด overfit ตอนเลือก config | ผู้ใช้ |
| 2026-10-22 | D4–D7 | ทดลองหลายแบบ เลือกจาก validation (ข้อ 5.2) | — | ผู้ใช้ |
| 2026-10-22 | D8 | STFT ก่อน · Wavelet ภายหลัง | — | ผู้ใช้ |
| 2026-10-22 | D9 | เทรนเอง (ก, ข) + ค เป็นตัวเทียบ + ResNet18 pretrained (Tier 2) · ไม่ทำ ง | — | ผู้ใช้ |
| 2026-10-22 | D10 | spectrogram ล้วน + hybrid · feature ใหม่ Tier 3 · HPO ข้อ 5.6 | — | ผู้ใช้ |
| 2026-10-22 | D11 | end-to-end + สองขั้น (cross-fitting) | — | ผู้ใช้ |
| 2026-10-22 | D12–D14 | ตามข้อ 5.1 · loss ลองทั้ง MSE/MAE/Huber (เอียง Huber) · PyTorch | — | ผู้ใช้ |
| 2026-10-22 | D15–D16 | โฟลเดอร์/branch แยก · protocol ข้อ 4 | — | ผู้ใช้ |
| 2026-10-22 | HPO | ทำเพื่อตรวจสอบ ไม่ใช่เพื่อชนะ · งบ 30 config ตายตัว + daring curve + shrinkage | หลักฐานข้อ 12.1 | ผู้ใช้ (รอยืนยัน) |
