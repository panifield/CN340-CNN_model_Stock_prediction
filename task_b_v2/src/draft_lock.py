"""
draft_lock.py — ร่าง LOCK.md จากผลที่มีอยู่จริง (SPEC ข้อ 4.4 · ข้อ 9.1 commit 2 · Checkpoint C5)
==============================================================================================
ไฟล์นี้ทำอะไร: อ่านผลตัดสินของทุก Stage (json) + experiment_log + SOURCES.md + นิยาม config ในโค้ด แล้วเขียน task_b_v2/LOCK.md
ทำไมต้องมี: LOCK.md คือ "คำสัญญา" ก่อนประเมิน final — ตารางค่า config สร้างจากโค้ดตรง ๆ (dataclasses.asdict) ไม่พิมพ์มือ กันค่าผิด
ร่างนี้ **รอผู้ใช้ตรวจและยืนยันที่ C5** ก่อน commit 2 (commit LOCK.md อย่างเดียว) · Claude ห้ามรัน git เอง

รัน:  python -m src.draft_lock
"""

import dataclasses
import json
import re
from datetime import datetime

import pandas as pd

from src import paths
from src.configs import BASE, variant
from src.run_stage345 import HYBRID, NOCNN, STAGE4_CFGS

R = paths.RESULTS_DIR
FIELDS = ["channels", "window", "nperseg", "hop", "spec_type", "scale_mode", "fixed_scale", "normalize", "arch", "head", "filters",
          "cnn_feat", "ann_hidden", "dropout", "ann_input", "train_mode", "loss", "huber_delta", "lr", "weight_decay", "batch",
          "max_epochs", "patience", "noise_std"]


def jload(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def config_table():
    tracks = [("Track 1 · CNN→ANN หลัก (spectrogram อย่างเดียว)", BASE), ("Track 2 · hybrid (+29 feature)", HYBRID),
              ("Track 3 · CNN สองขั้น", STAGE4_CFGS[0]), ("Track 4 · ค. 1D บน return ดิบ (ตัวเทียบ)", variant("cnn1d_raw", arch="cnn1d_raw")),
              ("Control · hybrid-noCNN", NOCNN)]
    head = "| field | " + " | ".join(t for t, _ in tracks) + " |"
    rows = [head, "|---|" + "---|" * len(tracks)]
    for f in FIELDS:
        vals = []
        for _, c in tracks:
            v = getattr(c, f)
            vals.append("−".join(map(str, v)) if isinstance(v, tuple) else str(v))
        rows.append(f"| `{f}` | " + " | ".join(f"`{v}`" for v in vals) + " |")
    return "\n".join(rows)


def main():
    src = (paths.DATA_DIR / "SOURCES.md").read_text(encoding="utf-8")
    shas = re.findall(r"\| `(\w+_10Y_Cleaned\.csv)` \| ([\d,]+) \| ([\d\-]+ → [\d\-]+) \| [\d,]+ \| `([0-9a-f]{64})` \|", src)
    log = pd.read_csv(R / "experiment_log.csv")
    s1, s2, s3 = jload("stage1_placebo/stage1_final_decision.json"), jload("stage2_structure/stage2_decision.json"), jload("stage3_input/stage3_decision.json")
    s4, s5 = jload("stage4_twostage/stage4_decision.json"), jload("stage5_loss/stage5_decision.json")
    d1 = jload("stage1_representation/stage1_decision.json")
    n_cfg = len(log)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    two_h = s4["two_stage_hybrid_vs_control"]

    md = f"""# LOCK — config ที่ล็อกก่อนประเมิน final (ร่าง · รอผู้ใช้ยืนยันที่ C5)

> **สถานะ: ร่าง** สร้างอัตโนมัติจากผลที่มีอยู่จริงเมื่อ {stamp} (`python -m src.draft_lock`) · ยังไม่ commit
> **ลำดับที่ใช้จริง (DECISIONS B35 · ปรับจากสเปกข้อ 9.1):** (1) ผู้ใช้ commit 1 (ทุกอย่างยกเว้นไฟล์นี้) → (2) ผู้ใช้ `git pull` repo ทีม แล้วแจ้ง Claude →
> (3) Claude refresh `data/` + อัปเดต `data/SOURCES.md` + ใส่ SHA ใหม่ในไฟล์นี้ แล้ว **หยุด** → (4) ผู้ใช้ตรวจไฟล์นี้แล้ว commit 2 = **LOCK.md + data ที่ refresh + SOURCES.md**
> → (5) ผู้ใช้แจ้ง Claude จึงจะรัน final/replay (C6) ได้
> เหตุผลที่ย้าย data เข้า commit 2: การ refresh ข้อมูลหลัง commit 2 จะทำให้ working tree ไม่สะอาด ซึ่งขัดกับ gate ของสเปกเอง · สาระของ gate ไม่เปลี่ยน — **ห้ามรัน final/replay ก่อนผู้ใช้ commit LOCK และแจ้ง**
> **หลัง commit นี้ ห้ามกลับไปแก้ config เพราะเห็นผล final/replay** (SPEC ข้อ 4.4 ข้อ 6)
> ผลทุกชิ้นที่นำมาประกอบ LOCK นี้เป็น development / walk-forward ในโซน dev ≤ 2025-02-25 — ไม่ใช่ test

## 1. ข้อมูล (SHA-256 จาก `data/SOURCES.md`)

| ไฟล์ | แถว | ช่วง | SHA-256 |
|---|---|---|---|
""" + "\n".join(f"| `{f}` | {n} | {rg} | `{sha}` |" for f, n, rg, sha in shas) + f"""

- ข้อมูลชุดนี้จบ **2026-10-19** · ก่อนทำ replay Claude จะ refresh ไฟล์จาก repo ทีม + ตรวจว่า **2,467 แถวแรกตรงกันทุกแถว** (DECISIONS B1) แล้วอัปเดต SOURCES.md และตาราง SHA ข้างบนนี้ (เป็นส่วนของ **commit 2** ตาม B35)
- dev (≤ 2025-02-25) ไม่ได้รับผลกระทบจากการ refresh
- **SHA หลัง refresh (Claude เติมในขั้นที่ 3 ของลำดับข้างบน):** `__________`

## 2. Protocol (ห้ามเปลี่ยน)

- โซน: dev ≤ **2025-02-25** · final **2025-02-26 → 2026-08-28** (ประเมินครั้งเดียว) · replay **2026-09-28 → 2026-10-21** (**16 วันทำการ** · เทรนด้วยข้อมูลถึง 2026-09-25)
  - replay จบ 10-21 เพราะข้อมูลของวันที่ 22 ต.ค. จะเข้า repo ทีมเช้าวันจันทร์ 26 ต.ค. (23 ต.ค. วันปิยมหาราช + เสาร์อาทิตย์) ซึ่งช้ากว่ากำหนดส่ง (B35)
  - วันหยุด SET ในช่วงนี้ (ยืนยันจาก `set_holidays.txt` ของ repo ทีม): 10-13 วันนวมินทรมหาราช · 10-16 วันหยุดพิเศษที่ SET ประกาศเพิ่ม · 10-23 วันปิยมหาราช
- target next_day: `y = Close(t)/Close(t−1) − 1` · ราคาที่ทำนาย `Close(t−1)·(1+ŷ)` · Naive: ŷ = 0
- walk-forward ใน dev: 3 fold expanding × 266 วัน ต่อกันจบ 2025-02-25 (B6) · early stopping = 15% ท้ายของ train · scaler/normalize/สเกลภาพ fit จากส่วน fit (85% แรก) เท่านั้น (B8, B9)
- ระหว่างค้นหา: 3 seeds เฉลี่ยคำทำนาย (0, 1, 2) · **final/replay: refit 10 seeds (0–9)** ของแต่ละ track **และของ control hybrid-noCNN** (ผู้ใช้ยืนยันแล้ว · B35)
- ตัววัด: relMAE หน่วย return (ตัววัดหลักที่รายงาน) และหน่วยบาท · MAE/RMSE บาท · IC Pearson+Spearman · DirAcc · StdRatio (ธง) · Diebold–Mariano vs Naive (B5, B16)
- **ปรับขนาดด้วย c** (B16 ข้อ 2, B18, B33): c ∈ {{0, 0.1, …, 1}} เลือกจากส่วน early-stop ต่อ evaluation เพื่อลด MAE_return · รายงานทั้งก่อนและหลังปรับ (ตัวเลขก่อนปรับคือฐานของการเลือกและการเทียบ)
- โมเดลเดิม (MLP / RF / XGB บน 29 feature · พารามิเตอร์ `phase1c-tuned` ไม่จูนใหม่) รันซ้ำภายใต้ protocol เดียวกัน (B11)

## 3. กติกาเลือกผู้ชนะที่ใช้จริง

1. **Stage 1–4**: mean IC (Pearson) ของ 6 evaluation · ต้องชนะ base เกิน noise ของ seed (B16, B17)
2. **Stage 1** ตัดสินด้วย placebo 3 shuffle: ผู้ชนะเฉลี่ย 2.33 ต่อ shuffle ≥ 1 → **ใช้ base ต่อ** (B21 · ผู้ใช้ยืนยัน)
3. **Stage 2–5**: ต้องมี ΔIC > max(noise, **{s1['b22_threshold_stage2_to_5']:+.4f}**) เทียบ config อ้างอิงของ track ตัวเอง (B22)
4. **hybrid vs hybrid-noCNN (P3b)**: เกณฑ์ = ค่าสูงสุดของ ΔIC จาก placebo 5 shuffle = **{s3['threshold_pair']:+.4f}** (B28)
5. relMAE รายงานทุกตาราง · StdRatio เป็นธง ไม่ตัดสิทธิ์ (B16)

## 4. ผลการเลือกต่อ Stage (ผู้ชนะที่ใช้ใน LOCK)

| Stage | ผู้สมัคร | noise | ผลตามกติกา | config ที่ใช้ต่อ |
|---|---|---|---|---|
| C3 base | — | — | — | base (SPEC ข้อ 5.1 + แก้ B9) |
| 1 การแทนข้อมูล | win64 · win256 · complex · norm_global · norm_image · ch3 · ch4 (ตัวเทียบ: detrend) | {d1['noise_pooled']:.4f} | กติกา B16/B17 แนะนำ win64+norm_global+ch3 แต่ placebo (B21) ชี้ว่าเลือกจาก noise → **ใช้ base** | **base** |
| 2 โครงสร้าง | cnn1d_time · gap (ตัวเทียบ: cnn1d_raw) | {s2['noise_pooled']:.4f} | ไม่มีผู้ชนะ (ΔIC สูงสุด +0.0405 < {s2['b22_floor']:.4f}) | **base** |
| 3 input ของ ANN | hybrid (control: hybrid-noCNN) | {s3['noise']:.4f} | hybrid เทียบ base: ΔIC {s3['delta_hybrid_vs_base']:+.4f} (ผ่าน B22) แต่เกิดจาก 29 feature · **ไม่แทน track 1 (B31)** | track 1 = base · **track 2 = hybrid** |
| 4 สองขั้น | two_stage_spec (vs base) · two_stage_hybrid (vs hybrid) | {s4['noise']:.4f} | ไม่มีผู้ชนะ (ΔIC +0.0060 และ −0.0067) | track 3 = two_stage_spec (รายงานตามสเปกข้อ 7) |
| 5 loss | MSE · MAE (ทั้งสอง track) | {s5['noise']:.4f} | ไม่มีผู้ชนะ (ΔIC สูงสุด +0.0376 และ +0.0102) | **Huber** |
| 6 HPO | — | — | **ข้าม** (B29) | — |

## 5. config ของแต่ละ track (สร้างจากโค้ดโดยตรง)

""" + config_table() + f"""

- ค่า field ที่ไม่ปรากฏในตาราง: `name` เป็นแค่ป้ายชื่อ · `logpower`: สเกลภาพ `s` = median(|ค่าดิบ|) ของส่วน fit ต่อหุ้นต่อ fold (B9) · `complex`: สเกลคงที่ 1e-3 (B10)
- `noise_std` ใช้กับ "ภาพ" เท่านั้น: **ไม่มีผลกับ Control hybrid-noCNN (B25) และกับ ANN ขั้นที่ 2 ของ Track 3 (B32)** เพราะ input ไม่ใช่ภาพ
- Track 3: ขั้น 1 = CNN (spectrogram อย่างเดียว) cross-fitting 4 ช่วง · ขั้น 2 = ANN บน [คำทำนาย OOF] (B32) · Track 5 = Naive · Mean Return · MLP · RF · XGBoost (`legacy/`)
- แยกโมเดลต่อหุ้น: KBANK กับ ADVANC เทรนแยกกัน · config ชุดเดียวใช้ทั้งสองหุ้น (D3)
- **Control hybrid-noCNN ถูกรันใน final/replay ด้วย** (refit 10 seeds) เพราะคาดการณ์ F1 ต้องใช้ (B35) — เป็น control ไม่ใช่ track ที่แข่ง

## 6. จำนวนที่ลอง (ต้องรายงานได้ · SPEC ข้อ 0.4)

- **config CNN ที่ต่างกันที่ลองในโซน dev = {n_cfg}** (`results/experiment_log.csv`) — base 1 · Stage 1: 8 + config รวม 1 · Stage 2: 3 · Stage 3: 2 · Stage 4: 2 · Stage 5: 4
- control/ตรวจสอบที่ไม่นับเป็น config ที่ลอง: positive control (C2) · negative control (C3, 1 และ 3 seeds) · placebo Stage 1 (9 config × 3 shuffle = 27 ชุด) · placebo Stage 3 (2 config × 5 shuffle = 10 ชุด) · sign control (4 config) · diagnostic สัญญาณ (ไม่เทรน)
- การเลือกในแต่ละ Stage เลือกจาก 2–7 ผู้สมัคร · ผลที่ผ่านเกณฑ์ใน Stage ใด ๆ มีแค่ hybrid (เพราะ 29 feature)

## 7. คาดการณ์ที่ประกาศไว้ก่อนรัน และสถานะ (ห้ามแก้ — DECISIONS ส่วน C)

| ข้อ | เนื้อหา | ผล |
|---|---|---|
| P1 | complex ควรได้ IC บวกเหนือ base เกิน noise | **ไม่สนับสนุน** (Stage 1: +0.0200 < 0.0214 ก้ำกึ่ง) |
| P2 | log-power ตัวอื่นไม่ควรช่วย | **ขัด** ตามเกณฑ์ (5/6 ตัวเหนือ base เกิน noise) แต่ placebo ชี้ว่าระดับนั้นเกิดจาก noise ได้ |
| P3 | hybrid สูงกว่า base เกิน noise และ > 0 | **เป็นจริง** (ΔIC {s3['delta_hybrid_vs_base']:+.4f}) |
| P3b | ΔIC(hybrid − hybrid-noCNN) > max(noise, {s3['threshold_pair']:.4f}) | **เป็นจริงตามกติกา แต่หวุดหวิดและไม่นิ่ง** (ΔIC {s3['delta_hybrid_vs_nocnn']:+.4f} เกินเกณฑ์ {s3['threshold_pair']:.4f} เพียง {s3['delta_hybrid_vs_nocnn'] - s3['threshold_pair']:+.4f}) |

**ข้อควรระวังของ P3b (ผู้ใช้สั่งให้ระบุ · B35):** เกณฑ์ "ค่าสูงสุดของ ΔIC จาก 5 shuffle" **มีอัตราผลบวกปลอม ≈ 1/6 (p ≈ 0.17)** —
ภายใต้สมมติฐานว่าไม่มีสัญญาณ ค่าจริงกับค่าจาก 5 shuffle สลับกันได้ (exchangeable) โอกาสที่ค่าจริงจะสูงสุดใน 6 ค่าจึงเป็น 1/6
ดังนั้น "ผ่าน P3b" ครั้งเดียวที่ระยะห่าง {s3['delta_hybrid_vs_nocnn'] - s3['threshold_pair']:+.4f} **ไม่ใช่หลักฐานที่หนักแน่น** ·
ประกอบกับผลขับโดย evaluation เดียว (K2 +0.100 · ตัดออกแล้วเหลือเฉลี่ย +0.007) และ ΔIC ราย seed +0.020 / −0.029 / +0.055 → ต้องเขียนในรายงานว่า **ยังไม่มีหลักฐานที่นิ่งว่า CNN เพิ่มข้อมูลเหนือ 29 feature**

### คาดการณ์สำหรับช่วง final — ประกาศก่อนรัน (ห้ามแก้หลังเห็นผล · B35)

| ข้อ | เนื้อหา | เกณฑ์อ่านผล |
|---|---|---|
| **F1** | IC(hybrid) − IC(hybrid-noCNN) บนช่วง final **> 0 แยกหุ้น** + 95% CI แบบ block bootstrap (block = 20 วัน) | "สนับสนุน" ในหุ้นนั้น = ΔIC > 0 **และ** ขอบล่างของ CI > 0 |
| **F2** | relMAE ของ **track 1** เทียบ **Naive** และเทียบ **track 4** (1D บน return ดิบ) บนช่วง final | รายงานค่าตรง ๆ: relMAE < 1 = ชนะ Naive · MAE(track 1)/MAE(track 4) < 1 = spectrogram ดีกว่า return ดิบ · แนบ Diebold–Mariano |

นิยามที่ระบุพร้อมกัน (ผู้ใช้ค้านได้ก่อนรัน): IC = Pearson corr(ŷ, y) บนวัน final ทั้งหมดของหุ้นนั้น ·
block bootstrap = แบ่งวัน final ตามเวลาเป็นบล็อกติดกันบล็อกละ 20 วัน สุ่มบล็อกแบบมีคืน (seed 0, 10,000 รอบ) คิด ΔIC ใหม่ทุกรอบจากคู่ (ŷ_hybrid, ŷ_noCNN, y) ของวันเดียวกัน → CI แบบ percentile 2.5/97.5 ·
F2 รายงานทั้งรวมสองหุ้นและแยกหุ้น

## 8. ข้อที่ผู้ใช้ยืนยันแล้ว ({stamp})

1. **Track 3** ใช้ `two_stage_spec` (ANN บน OOF) — Stage 4 ไม่มีผู้ชนะ แต่สเปกข้อ 7 ต้องการ track นี้ในตารางสุดท้าย (รายงานตามจริงไม่ว่าชนะหรือไม่) — **ยืนยัน**
2. **ตัวเลขหลักในสไลด์**: relMAE ก่อนปรับขนาดเป็นฐาน และแสดงหลังปรับขนาดคู่กัน — **ยืนยัน**
3. **การเขียนผล P3b**: ตามหัวข้อ 7 (รวมอัตราผลบวกปลอม ≈ 1/6) · two_stage_hybrid เทียบ control = {two_h:+.4f} (< {s3['threshold_pair']:.4f}) — **ยืนยัน**
4. **final/replay refit 10 seeds** ทุก track **และ control hybrid-noCNN** — **ยืนยัน**
5. ข้อมูลต้อง refresh ก่อน replay (หัวข้อ 1) · โหมด 16:00 / ResNet18 / HPO → หลังพรีเซนต์ (B29) · **tick analysis = งานเสริมหลัง C6** (ใช้คำทำนายที่ได้แล้ว ไม่เทรนใหม่) ถ้ามีเวลา (B35)

## 9. ขั้นตอน C6 (หลังผู้ใช้ commit LOCK.md และ `git status` ว่าง)

1. **final**: เทรนทุก track **และ control hybrid-noCNN** บน dev ทั้งหมด (≤ 2025-02-25) ครั้งเดียว (10 seeds) · ทำนาย 2025-02-26 → 2026-08-28 · เลือก c จากส่วน early-stop ของการเทรนนั้น · ตัดสิน **F1 / F2** ตามหัวข้อ 7
2. **replay**: เทรนด้วยข้อมูลถึง 2026-09-25 ครั้งเดียว · ทำนาย 2026-09-28 → **2026-10-21** รายวัน (16 วันทำการ · input ใช้ข้อมูลถึงวันก่อนหน้า)
3. รายงานต้องเขียนว่า "ช่วง final เคยถูกใช้ประเมินโมเดลรุ่นก่อนของทีมมาแล้ว แต่ไม่เคยใช้ออกแบบหรือเลือก config ของ CNN รุ่นนี้" และ "replay มี ~17 วัน ใช้สรุปทางสถิติไม่ได้"
4. ห้ามแก้ config หลังเห็นผล · ถ้าจะแก้ต้องเป็นรอบใหม่ที่ประกาศชัด
5. **หลัง C6 (งานเสริม ถ้ามีเวลา):** วิเคราะห์การปัดคำทำนายเป็น tick โดยใช้คำทำนายที่ได้แล้ว **ไม่เทรนใหม่** · ใช้ตาราง tick ตามระดับราคา (ข้อ 5.8 ของสเปก · B20)

## 10. ช่องสำหรับผู้ใช้ (เติมหลัง commit)

ตามลำดับที่ปรับใน B35:

- commit ที่ 1 — ทุกอย่างยกเว้น `LOCK.md` · ข้อความ `task_b_v2: development pipeline and walk-forward results (dev <= 2025-02-25)`: `__________`
- commit ที่ 2 — **`LOCK.md` + `data/*.csv` ที่ refresh + `data/SOURCES.md`** · ข้อความที่เสนอ `task_b_v2: lock configs and refresh data before final evaluation`: `__________`
- วันที่ผู้ใช้ยืนยัน LOCK: `__________`
- commit ที่ 3 (หลัง C6–C7) — ผล final + replay + figures + README: `__________`
"""
    out = paths.TASK_DIR / "LOCK.md"
    out.write_text(md, encoding="utf-8")
    print(f"เขียน {out} ({len(md):,} ตัวอักษร) · config ที่ลอง {n_cfg}")


if __name__ == "__main__":
    main()
