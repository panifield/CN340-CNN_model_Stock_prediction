"""
draft_lock.py — ร่าง LOCK.md จากผลที่มีอยู่จริง (SPEC ข้อ 4.4 · ข้อ 9.1 commit 2 · Checkpoint C5)
==============================================================================================
ไฟล์นี้ทำอะไร: อ่านผลตัดสินของทุก Stage (json) + experiment_log + SOURCES.md + นิยาม config ในโค้ด แล้วเขียน task_b_v2/LOCK.md
ทำไมต้องมี: LOCK.md คือ "คำสัญญา" ก่อนประเมิน final — ตารางค่า config สร้างจากโค้ดตรง ๆ (dataclasses.asdict) ไม่พิมพ์มือ กันค่าผิด
ร่างนี้ **รอผู้ใช้ตรวจและยืนยันที่ C5** ก่อน commit 2 (ไฟล์ที่ต้องใส่ใน commit 2 อยู่ในหัวข้อ 10 ของ LOCK.md) · Claude ห้ามรัน git เอง
ผล Stage 6 (HPO · B38–B40) อ่านจาก results/stage6_hpo/stage6_decision.json — ไฟล์นี้เขียนสำหรับผล "ไม่มีผู้ชนะ" เท่านั้น (ถ้ามีผู้ชนะจะหยุด ไม่เขียนเงียบ ๆ)

รัน:  python -m src.draft_lock
"""

import dataclasses
import json
import re

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
    s6 = jload("stage6_hpo/stage6_decision.json")
    j6 = s6["judge"]
    if s6["has_winner"] or s6["smoke"]:
        raise SystemExit("ผล Stage 6 ไม่ใช่ 'ไม่มีผู้ชนะ' (หรือเป็น dry-run) — draft_lock.py เขียนสำหรับกรณีไม่มีผู้ชนะเท่านั้น หยุดและถามผู้ใช้ (A7)")
    n_cfg = len(log)
    two_h = s4["two_stage_hybrid_vs_control"]

    md = f"""# LOCK — config ที่ล็อกก่อนประเมิน final

> **สถานะ: ล็อกแล้ว** — การ commit ไฟล์นี้คือการยืนยันของผู้ใช้ · ห้ามแก้ไฟล์นี้หลัง commit
> **ลำดับที่ใช้จริง (DECISIONS B35 · ปรับจากสเปกข้อ 9.1):** (1) ผู้ใช้ commit 1 (ทุกอย่างยกเว้นไฟล์นี้) → (2) ผู้ใช้ `git pull` repo ทีม แล้วแจ้ง Claude →
> (3) Claude refresh `data/` + อัปเดต `data/SOURCES.md` + ใส่ SHA ใหม่ในไฟล์นี้ แล้ว **หยุด** ← **ทำเสร็จแล้ว 2026-10-23** →
> (3b) **Stage 6 (HPO ก่อน LOCK · B38–B40)** รันครบและผู้ใช้ยืนยันผลแล้ว (ไม่มีผู้ชนะ · track 1 = base) ← **ทำเสร็จแล้ว 2026-10-24** → (4) ผู้ใช้ตรวจไฟล์นี้แล้ว commit 2 = **รายการในหัวข้อ 10 (ไม่รวม `mode1600/`)**
> → (5) ผู้ใช้ (หรือ session ของโหมด 16:00) commit **`LOCK_1600.md`** ด้วย → (6) ผู้ใช้แจ้ง Claude จึงจะรัน final/replay (C6) ได้ (หัวข้อ 9 ข้อ (d))
> เหตุผลที่ย้าย data เข้า commit 2: การ refresh ข้อมูลหลัง commit 2 จะทำให้ working tree ไม่สะอาด ซึ่งขัดกับ gate ของสเปกเอง · สาระของ gate ไม่เปลี่ยน — **ห้ามรัน final/replay ก่อนผู้ใช้ commit LOCK และแจ้ง**
> **หลัง commit นี้ ห้ามกลับไปแก้ config เพราะเห็นผล final/replay** (SPEC ข้อ 4.4 ข้อ 6)
> ผลทุกชิ้นที่นำมาประกอบ LOCK นี้เป็น development / walk-forward ในโซน dev ≤ 2025-02-25 — ไม่ใช่ test

## 1. ข้อมูล (SHA-256 จาก `data/SOURCES.md`)

| ไฟล์ | แถว | ช่วง | SHA-256 |
|---|---|---|---|
""" + "\n".join(f"| `{f}` | {n} | {rg} | `{sha}` |" for f, n, rg, sha in shas) + f"""

- **refresh แล้วเมื่อ 2026-10-23** (ขั้นที่ 3 ของลำดับข้างบน) จาก clone `CN340/StockPrediction2` HEAD `c87a7f26a2c346f128714f41dbc402172042b218`
  · commit ที่แก้ไฟล์ข้อมูล `6350406b972048aae14312b419a034348a85c9a2` ("data: task_b raw_data through 2026-10-22 (yahoo)")
  · วิธี copy `git cat-file blob HEAD:task_b/raw_data/<file>` · ยืนยัน `git hash-object --no-filters` = blob id ทั้งสองไฟล์
- การตรวจตาม **DECISIONS B1**: **2,467 แถวแรกตรงกับฉบับก่อน refresh (จบ 2026-10-19) ทุกไบต์** · เพิ่มมา **3 แถว** (10-20, 10-21, 10-22) → รวม **2,470 แถว**
- dev (≤ 2025-02-25) **ไม่ได้รับผลกระทบ**: ยังได้ 2,071 แถว วันสุดท้าย 2025-02-25 พอดี · final ยังได้ 362 แถว · ผลทุก Stage ก่อน LOCK ไม่เปลี่ยน (ยืนยันด้วย `python -m pytest`)
- SHA ก่อน refresh (ไว้ย้อนดู): `dce93acf5a9544bda63b6ee80ed709a287bc83f7a088723e6ad5823899baca1a` (KBANK) · `6d3398ba6537bdf5702cda5558157ab41c5b582c8af959493656f6f308eb6625` (ADVANC)

## 2. Protocol (ห้ามเปลี่ยน)

- โซน: dev ≤ **2025-02-25** · final **2025-02-26 → 2026-08-28** (ประเมินครั้งเดียว) · replay **2026-09-28 → 2026-10-22** (**17 วันทำการ** · เทรนด้วยข้อมูลถึง 2026-09-25)
  - replay จบ **10-22** เพราะข้อมูลถึงวันนั้นเข้า repo ทีมแล้ว (commit `6350406`) และ refresh ลง `data/` แล้ว → ช่วงนี้ **ตรงกับช่วงเก็บ log พอดีและให้คะแนนครบได้ทันที** (**B36**)
  - วันหยุด SET ในช่วงนี้ (ยืนยันจาก `set_holidays.txt` ของ repo ทีม + นับจากข้อมูลจริง 17 แถว): 10-13 วันนวมินทรมหาราช · 10-16 วันหยุดพิเศษที่ SET ประกาศเพิ่ม
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
6. **Stage 6 (HPO ของ track 1 · B38, B39)**: ตรวจเฉพาะผู้สมัครที่ mean IC สูงสุด 1 ตัว ต้องผ่านครบ (1) ΔIC เทียบ base > max(noise, 0.0597, T_hpo) โดย T_hpo = ค่าสูงสุดข้าม 5 รอบ placebo ของ (max ข้ามผู้สมัครของ IC_ผู้สมัคร − IC_base ในรอบนั้น) ·
   (2) **[บังคับ]** ยืนยันด้วย seeds 10–12 ΔIC เกินเกณฑ์ข้อ (1) ตัวเดิม · (3) IC > 0 ≥ 5/6 และหลังตัด evaluation ที่ ΔIC สูงสุด ΔIC เฉลี่ยที่เหลือ > max(noise, 0.0597) · (4) relMAE (return) ไม่สูงกว่า base ·
   คำทำนายคงที่ (std(ŷ)/std(y) < 1e-8) → IC := 0 ทุกส่วน (B39)

## 4. ผลการเลือกต่อ Stage (ผู้ชนะที่ใช้ใน LOCK)

| Stage | ผู้สมัคร | noise | ผลตามกติกา | config ที่ใช้ต่อ |
|---|---|---|---|---|
| C3 base | — | — | — | base (SPEC ข้อ 5.1 + แก้ B9) |
| 1 การแทนข้อมูล | win64 · win256 · complex · norm_global · norm_image · ch3 · ch4 (ตัวเทียบ: detrend) | {d1['noise_pooled']:.4f} | กติกา B16/B17 แนะนำ win64+norm_global+ch3 แต่ placebo (B21) ชี้ว่าเลือกจาก noise → **ใช้ base** | **base** |
| 2 โครงสร้าง | cnn1d_time · gap (ตัวเทียบ: cnn1d_raw) | {s2['noise_pooled']:.4f} | ไม่มีผู้ชนะ (ΔIC สูงสุด +0.0405 < {s2['b22_floor']:.4f}) | **base** |
| 3 input ของ ANN | hybrid (control: hybrid-noCNN) | {s3['noise']:.4f} | hybrid เทียบ base: ΔIC {s3['delta_hybrid_vs_base']:+.4f} (ผ่าน B22) แต่เกิดจาก 29 feature · **ไม่แทน track 1 (B31)** | track 1 = base · **track 2 = hybrid** |
| 4 สองขั้น | two_stage_spec (vs base) · two_stage_hybrid (vs hybrid) | {s4['noise']:.4f} | ไม่มีผู้ชนะ (ΔIC +0.0060 และ −0.0067) | track 3 = two_stage_spec (รายงานตามสเปกข้อ 7) |
| 5 loss | MSE · MAE (ทั้งสอง track) | {s5['noise']:.4f} | ไม่มีผู้ชนะ (ΔIC สูงสุด +0.0376 และ +0.0102) | **Huber** |
| 6 HPO (track 1) | 24 ผู้สมัครสุ่มสมดุล (seed 38) + base = 25 config · lr · batch · dropout · weight_decay · filters · ann_hidden · nperseg · hop · noise_std | {s6['noise']:.4f} (pooled 25 config) | placebo ของการค้นหาทั้งชุด 5 รอบ (seed 100–104): **T_hpo = {j6['t_hpo']:+.4f}** → เกณฑ์ข้อ (1) = max({j6['noise']:.4f}, {j6['floor']:.4f}, {j6['t_hpo']:.4f}) = **{j6['thr1']:.4f}** · อันดับ 1 = {j6['top']}: ΔIC {j6['delta_ic']:+.4f} **ไม่ผ่านข้อ (1)** · ยืนยัน seeds 10–12 ΔIC {j6['delta_ic_confirm']:+.4f} **ไม่ผ่านข้อ (2 · บังคับ)** · ข้อ (3) ไม่ผ่าน (ΔIC เฉลี่ยหลังตัดเหลือ {j6['delta_mean_trimmed']:+.4f} < {j6['thr3']:.4f}) · ข้อ (4) ผ่าน → **ไม่มีผู้ชนะ** (B38 · B39 · B40) · รอบค้นหาแรกหยุดตาม B3 เพราะคำทำนายคงที่ทำให้ IC เป็น NaN → ผลรอบนั้นเก็บใน `results/stage6_hpo/aborted_run1/` **ไม่ใช้ตัดสิน** | **base** (ไม่มี track ใดถูกจูน) |

## 5. config ของแต่ละ track (สร้างจากโค้ดโดยตรง)

""" + config_table() + f"""

- ค่า field ที่ไม่ปรากฏในตาราง: `name` เป็นแค่ป้ายชื่อ · `logpower`: สเกลภาพ `s` = median(|ค่าดิบ|) ของส่วน fit ต่อหุ้นต่อ fold (B9) · `complex`: สเกลคงที่ 1e-3 (B10)
- `noise_std` ใช้กับ "ภาพ" เท่านั้น: **ไม่มีผลกับ Control hybrid-noCNN (B25) และกับ ANN ขั้นที่ 2 ของ Track 3 (B32)** เพราะ input ไม่ใช่ภาพ
- Track 3: ขั้น 1 = CNN (spectrogram อย่างเดียว) cross-fitting 4 ช่วง · ขั้น 2 = ANN บน [คำทำนาย OOF] (B32) · Track 5 = Naive · Mean Return · MLP · RF · XGBoost (`legacy/`)
- แยกโมเดลต่อหุ้น: KBANK กับ ADVANC เทรนแยกกัน · config ชุดเดียวใช้ทั้งสองหุ้น (D3)
- **ไม่มี track ใดถูกจูน (A7 ของ B38):** Stage 6 ไม่มีผู้ชนะ → Track 1 = base เหมือนเดิม จึงไม่มีความไม่สมมาตรระหว่าง track ที่จูนกับไม่จูน · ไม่ต้องรัน "track 1-base" เพิ่มใน final และ F2 เทียบ track 1 (= base) กับ Naive และ track 4 ตามเดิม
- **Control hybrid-noCNN ถูกรันใน final/replay ด้วย** (refit 10 seeds) เพราะคาดการณ์ F1 ต้องใช้ (B35) — เป็น control ไม่ใช่ track ที่แข่ง

## 6. จำนวนที่ลอง (ต้องรายงานได้ · SPEC ข้อ 0.4)

- **config CNN ที่ต่างกันที่ลองในโซน dev = {n_cfg}** (`results/experiment_log.csv`) — base 1 · Stage 1: 8 + config รวม 1 · Stage 2: 3 · Stage 3: 2 · Stage 4: 2 · Stage 5: 4 · **Stage 6: 24**
- control/ตรวจสอบที่ไม่นับเป็น config ที่ลอง: positive control (C2) · negative control (C3, 1 และ 3 seeds) · placebo Stage 1 (9 config × 3 shuffle = 27 ชุด) · placebo Stage 3 (2 config × 5 shuffle = 10 ชุด) · sign control (4 config) · diagnostic สัญญาณ (ไม่เทรน) ·
  **placebo Stage 6 (25 config × 5 รอบ (seed 100–104) = 125 ชุด)** · **ยืนยัน seed ใหม่ของ Stage 6 (seeds 10–12 · base + {j6['top']})** · รอบค้นหาแรกของ Stage 6 ที่หยุดตาม B3 (`aborted_run1/` · ไม่ใช้ตัดสิน)
- การเลือกในแต่ละ Stage เลือกจาก 2–7 ผู้สมัคร (Stage 6: 24 ผู้สมัคร ซึ่งเลือกค่าสูงสุดจาก 24 ค่าจึงต้องมีเกณฑ์จาก placebo) · ผลที่ผ่านเกณฑ์ใน Stage ใด ๆ (รวม Stage 6) มีแค่ hybrid (เพราะ 29 feature)

## 7. คาดการณ์ที่ประกาศไว้ก่อนรัน และสถานะ (ห้ามแก้ — DECISIONS ส่วน C)

| ข้อ | เนื้อหา | ผล |
|---|---|---|
| P1 | complex ควรได้ IC บวกเหนือ base เกิน noise | **ไม่สนับสนุน** (Stage 1: +0.0200 < 0.0214 ก้ำกึ่ง) |
| P2 | log-power ตัวอื่นไม่ควรช่วย | **ขัด** ตามเกณฑ์ (5/6 ตัวเหนือ base เกิน noise) แต่ placebo ชี้ว่าระดับนั้นเกิดจาก noise ได้ |
| P3 | hybrid สูงกว่า base เกิน noise และ > 0 | **เป็นจริง** (ΔIC {s3['delta_hybrid_vs_base']:+.4f}) |
| P3b | ΔIC(hybrid − hybrid-noCNN) > max(noise, {s3['threshold_pair']:.4f}) | **เป็นจริงตามกติกา แต่หวุดหวิดและไม่นิ่ง** (ΔIC {s3['delta_hybrid_vs_nocnn']:+.4f} เกินเกณฑ์ {s3['threshold_pair']:.4f} เพียง {s3['delta_hybrid_vs_nocnn'] - s3['threshold_pair']:+.4f}) |
| P6 | ไม่มี config ใดผ่านเกณฑ์ชนะของ Stage 6 (ประกาศก่อนรัน · B38) | **เป็นจริง** ({j6['top']} ΔIC {j6['delta_ic']:+.4f} < {j6['thr1']:.4f} · ยืนยัน seeds 10–12 ΔIC {j6['delta_ic_confirm']:+.4f} · ผ่านเฉพาะข้อ (4)) |

**ข้อสังเกตของ P6 (descriptive · ห้ามใช้เปลี่ยน config · B40):** {j6['top']} ได้ IC +0.0959 ใน placebo รอบ 100 (y ถูกสลับ) สูงกว่า IC จริงของมันเอง (+0.0607) ·
base ขยับ +0.047 ระหว่างสองชุด seed (−0.0096 → +0.0373) · collapse (คำทำนายคงที่) กระจุกที่ hop 1 / lr 0.003 / dropout 0.5 · ใช้เวลาจริง 4.0 ชม. — รายละเอียดใน DECISIONS ส่วน C "ผลของ P6"

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

### S1 — การทดสอบความไวต่อช่วงข้อมูลเทรน (ประกาศก่อนรัน 2026-10-23 · ผู้ใช้สั่ง · **B37**)

> **descriptive เท่านั้น — ห้ามใช้ผลของ S1 เปลี่ยน config หลักหรือ track ใด ๆ** · ไม่ใช่การเลือก config และไม่เข้าเกณฑ์ตัดสินในหัวข้อ 3
> คำถามที่ S1 ตอบ: **"ข้อมูลตั้งแต่ 2016 เก่าเกินไปสำหรับทำนายวันนี้ไหม"** — ตอบด้วยตัวเลข

| รายการ | ค่าที่กำหนดไว้ก่อนรัน |
|---|---|
| โมเดลที่ทดสอบ | **track 1** (CNN→ANN spectrogram อย่างเดียว · **10 seeds**) และ **Random Forest** (29 feature · พารามิเตอร์ `phase1c-tuned` ไม่จูนใหม่) |
| ช่วงเทรนแบบสั้น | **2021-01-01 → 2025-02-25** |
| ช่วงเทรนแบบเต็ม (ตัวเทียบ) | ต้นข้อมูล → 2025-02-25 (= ที่ใช้ใน final ปกติ) |
| ช่วงประเมิน | **ช่วง final** (2025-02-26 → 2026-08-28) ชุดวันเดียวกันทั้งสองแบบ |
| early stopping / normalize / scaler | **15% ท้าย / 85% แรกของช่วงเทรนที่สั้นลงนั้น** (ไม่ใช่ของช่วงเต็ม) · fit แยกต่อหุ้นเหมือนเดิม |
| ที่รายงาน | **relMAE (หน่วย return และบาท) และ IC (Pearson + Spearman) แยกหุ้น** ของทั้งสองแบบ + ผลต่าง |
| ที่ไม่ทำ | ไม่ค้นหาวันเริ่มที่ดีที่สุด (ไม่ลองหลายวันเริ่มแล้วเลือกอันที่ดีที่สุด) · ไม่แตะ track 2–5 · ไม่เปลี่ยน LOCK เพราะผลนี้ |

เหตุผลที่ต้องประกาศก่อนรัน: ถ้าเห็นผลก่อนแล้วค่อยตั้งกติกา จะกลายเป็นการเลือกช่วงข้อมูลจากผลประเมิน ซึ่งเป็นสิ่งที่ protocol ทั้งหมดของงานนี้กันไว้

## 8. ข้อที่ผู้ใช้ยืนยันแล้ว (ข้อ 1–5: 2026-10-23 01:56 · ข้อ 6–8: 2026-10-23 · ข้อ 9–14: 2026-10-24 · B40)

1. **Track 3** ใช้ `two_stage_spec` (ANN บน OOF) — Stage 4 ไม่มีผู้ชนะ แต่สเปกข้อ 7 ต้องการ track นี้ในตารางสุดท้าย (รายงานตามจริงไม่ว่าชนะหรือไม่) — **ยืนยัน**
2. **ตัวเลขหลักในสไลด์**: relMAE ก่อนปรับขนาดเป็นฐาน และแสดงหลังปรับขนาดคู่กัน — **ยืนยัน**
3. **การเขียนผล P3b**: ตามหัวข้อ 7 (รวมอัตราผลบวกปลอม ≈ 1/6) · two_stage_hybrid เทียบ control = {two_h:+.4f} (< {s3['threshold_pair']:.4f}) — **ยืนยัน**
4. **final/replay refit 10 seeds** ทุก track **และ control hybrid-noCNN** — **ยืนยัน**
5. ข้อมูลต้อง refresh ก่อน replay (หัวข้อ 1) · ResNet18 / HPO → หลังพรีเซนต์ (B29) · **tick analysis = งานเสริมหลัง C6** (ใช้คำทำนายที่ได้แล้ว ไม่เทรนใหม่) ถ้ามีเวลา (B35)
6. **replay = 2026-09-28 → 2026-10-22 (17 วันทำการ)** ตรงกับช่วงเก็บ log — **ยืนยัน** (B36 · แทนที่ 16 วันทำการของ B35)
7. **S1** (ความไวต่อช่วงข้อมูลเทรน · หัวข้อ 7) เป็น **descriptive เท่านั้น** ห้ามใช้เปลี่ยน config หลัก — **ยืนยัน** (B37)
8. **โหมด 16:00** ทำ **หลัง C6 ของ next_day** ตามสเปกข้อ 3.2 และจะมี **`LOCK_1600.md` แยกของตัวเอง** — ไฟล์นี้ (`LOCK.md`) เป็นของ next_day เท่านั้น (ผู้ใช้แจ้ง 2026-10-23)
9. **ผล Stage 6**: ไม่มีผู้ชนะ · **track 1 = base** · **P6 เป็นจริง** (หัวข้อ 4 และ 7) — **ยืนยัน** (B40)
10. **S1 ใช้ track 1 ตามที่ล็อก (= base)** · **S2 ไม่ทำ** (S2 = ประเมิน spectrogram 3 ช่อง (ch3) บนช่วง final แบบ descriptive · ผู้ใช้ตัดสินไม่ทำ (เวลาจำกัด · Stage 1 + placebo ชี้แล้วว่าแยกจาก noise ไม่ได้)) — **ยืนยัน** (B40)
11. **รูปแบบ log ของ replay**: ตารางรายวันแยกหุ้น + สรุป MAE ต่อโมเดล ติดป้าย "ทำนายย้อนหลังแบบ walk-forward ไม่ใช่ log สด" (หัวข้อ 9 ข้อ (a)) — **ยืนยัน**
12. **กติกาเมื่อพบบั๊กระหว่าง C6**: หยุดและรายงาน ห้ามแก้แล้วรันใหม่เงียบ ๆ (หัวข้อ 9 ข้อ (c)) — **ยืนยัน**
13. **ลำดับ**: C6 เริ่มได้เมื่อผู้ใช้ commit ทั้ง `LOCK.md` และ `LOCK_1600.md` แล้วเท่านั้น เพราะช่วงประเมินของโหมด 16:00 ทับกับ final ของ next_day (หัวข้อ 9 ข้อ (d)) ·
    **การรันจริงของโหมด 16:00 เริ่มได้ทันทีหลัง C6 (final + replay + S1) เสร็จ ไม่ต้องรอ C7** โดยมี `LOCK.md` และ `LOCK_1600.md` ถูก commit ก่อนเริ่ม C6 และผ่าน gate ของ 16:00 — **ยืนยัน**
14. **commit 2 ไม่รวม `task_b_v2/mode1600/`** (ของอีก session · จะ commit พร้อม `LOCK_1600.md`) — **ยืนยัน**

## 9. ขั้นตอน C6 (หลังผู้ใช้ commit **LOCK.md และ LOCK_1600.md** และ `git status` ว่าง)

**เงื่อนไขเริ่ม C6 — ข้อ (d):** C6 เริ่มได้เมื่อผู้ใช้ commit ทั้ง `LOCK.md` และ `LOCK_1600.md` แล้วเท่านั้น (`LOCK_1600.md` อยู่ใน `mode1600/` ของอีก session) เพราะช่วงประเมินของโหมด 16:00 (2024-10-01 → 2026-08-28) **ทับกับช่วง final ของ next_day** (2025-02-26 → 2026-08-28) —
ถ้ารัน final ก่อนล็อกโหมด 16:00 จะเห็นผลของวันเดียวกันก่อนตั้ง config ของโหมด 16:00 · **การรันจริงของโหมด 16:00 เริ่มได้ทันทีหลัง C6 (final + replay + S1) เสร็จ ไม่ต้องรอ C7** ·
เงื่อนไขคือ `LOCK.md` และ `LOCK_1600.md` ถูก commit ก่อนเริ่ม C6 และผ่าน gate ของ 16:00 (ผู้ใช้กำหนด · B40)

โค้ดของ C6 เขียนหลัง commit นี้ (ณ เวลา commit ยังไม่มี) · ต้องทำตามหัวข้อ 9 ทุกข้อ · ใช้ฟังก์ชันเทรน/ประเมินชุดเดิมของ Stage 0–6 · ห้ามแตะข้อมูล ≥ 2025-02-26 ระหว่างเขียนและทดสอบ · ต้อง commit ก่อนรัน

1. **final**: เทรนทุก track **และ control hybrid-noCNN** บน dev ทั้งหมด (≤ 2025-02-25) ครั้งเดียว (10 seeds) · ทำนาย 2025-02-26 → 2026-08-28 · เลือก c จากส่วน early-stop ของการเทรนนั้น · ตัดสิน **F1 / F2** ตามหัวข้อ 7 · **track 1 = base** (Stage 6 ไม่มีผู้ชนะ)
2. **replay**: เทรนด้วยข้อมูลถึง 2026-09-25 ครั้งเดียว · ทำนาย 2026-09-28 → **2026-10-22** รายวัน (**17 วันทำการ** · input ใช้ข้อมูลถึงวันก่อนหน้า · B36)
   - **ข้อ (a) ตาราง log รายวัน (B40):** replay ต้องออกตารางรายวัน **แยกหุ้น** คอลัมน์: วันที่ · โมเดล · ราคาปิดวันก่อน · ราคาที่ทาย · ราคาจริง · error บาท · ทิศถูกหรือไม่ — พร้อม **สรุป MAE ต่อโมเดล** ·
     ติดป้ายกำกับทุกตารางว่า **"ทำนายย้อนหลังแบบ walk-forward ไม่ใช่ log สด"** ·
     นิยามที่ระบุพร้อมกัน (ผู้ใช้ค้านได้ก่อน commit 2): "ทิศถูกหรือไม่" = เครื่องหมายของ (ราคาที่ทาย − ราคาปิดวันก่อน) ตรงกับของ (ราคาจริง − ราคาปิดวันก่อน) · วันที่ราคาไม่เปลี่ยน (y = 0) แสดง "—" และไม่นับ (ตามนิยาม DirAcc เดิม) ·
     "โมเดล" = ทุกโมเดลที่รัน replay (track 1–4 · control hybrid-noCNN · Naive · Mean Return · MLP · RF · XGBoost)
3. **S1** (หัวข้อ 7 · B37) — **ข้อ (b):** เทรน **track 1 ตามที่ล็อก (= base)** (10 seeds) และ Random Forest ด้วยข้อมูล **2021-01-01 → 2025-02-25** แล้วประเมินบนช่วง final ชุดวันเดียวกัน · รายงาน relMAE + IC แยกหุ้นเทียบกับแบบใช้ข้อมูลทั้งหมด · **descriptive เท่านั้น** · **S2 ไม่ทำ** (S2 = ประเมิน spectrogram 3 ช่อง (ch3) บนช่วง final แบบ descriptive · ผู้ใช้ตัดสินไม่ทำ (เวลาจำกัด · Stage 1 + placebo ชี้แล้วว่าแยกจาก noise ไม่ได้))
4. รายงานต้องเขียนว่า "ช่วง final เคยถูกใช้ประเมินโมเดลรุ่นก่อนของทีมมาแล้ว แต่ไม่เคยใช้ออกแบบหรือเลือก config ของ CNN รุ่นนี้" และ "replay มี 17 วัน ใช้สรุปทางสถิติไม่ได้"
5. ห้ามแก้ config หลังเห็นผล · ถ้าจะแก้ต้องเป็นรอบใหม่ที่ประกาศชัด
   - **ข้อ (c) กติกาเมื่อพบบั๊กระหว่าง C6 (B40):** **หยุดและรายงาน** ห้ามแก้แล้วรันใหม่เงียบ ๆ · การแก้หลังเห็นผลต้องประกาศเป็น **รอบใหม่** (บันทึกใน DECISIONS ก่อนรัน) และ **รายงานทั้งผลก่อนแก้และหลังแก้** เคียงกัน
6. **หลัง C6 (งานเสริม ถ้ามีเวลา):** วิเคราะห์การปัดคำทำนายเป็น tick โดยใช้คำทำนายที่ได้แล้ว **ไม่เทรนใหม่** · ใช้ตาราง tick ตามระดับราคา (ข้อ 5.8 ของสเปก · B20)
7. **หลัง C6 (final + replay + S1) เสร็จ:** การรันจริงของโหมด **16:00** ตามสเปกข้อ 3.2 พร้อม **`LOCK_1600.md` แยก** เริ่มได้ทันที **ไม่ต้องรอ C7** ·
   เงื่อนไข: `LOCK.md` และ `LOCK_1600.md` ถูก commit **ก่อนเริ่ม C6** (ข้อ (d)) และผ่าน gate ของ 16:00 (ผู้ใช้กำหนด · B40) · ยังไม่เริ่มตอนนี้

## 10. ช่องสำหรับผู้ใช้ (เติมหลัง commit)

> ช่องในหัวข้อนี้จะไม่ถูกเติม — การแก้ LOCK.md หลัง commit ทำให้ gate ของโหมด 16:00 ปฏิเสธ · commit hash และวันที่ยืนยันบันทึกใน DECISIONS.md (B41) แทน

ตามลำดับที่ปรับใน B35:

- commit ที่ 1 — ทุกอย่างยกเว้น `LOCK.md` · ข้อความ `task_b_v2: development pipeline and walk-forward results (dev <= 2025-02-25)`: `__________`
- commit ที่ 2 — **ต้องใส่ให้ครบทุกรายการด้านล่าง (10 รายการ) มิฉะนั้น `git status` จะไม่ว่างและ gate ของหัวข้อ 9 จะติด** · **ห้ามรวม `task_b_v2/mode1600/`** (ของอีก session · commit พร้อม `LOCK_1600.md`):

  | ไฟล์ | ทำไมอยู่ใน commit นี้ |
  |---|---|
  | `task_b_v2/LOCK.md` | ไฟล์ LOCK เอง (ยัง untracked) |
  | `task_b_v2/data/KBANK_10Y_Cleaned.csv` | refresh ถึง 2026-10-22 (+3 แถว) |
  | `task_b_v2/data/ADVANC_10Y_Cleaned.csv` | refresh ถึง 2026-10-22 (+3 แถว) |
  | `task_b_v2/data/SOURCES.md` | SHA ใหม่ + บันทึกการ refresh |
  | `task_b_v2/src/settings.py` | `REPLAY_END` → 2026-10-22 (B36) |
  | `task_b_v2/src/draft_lock.py` | ต้นแบบของไฟล์นี้ — แก้ให้ตรงกับ B36–B40 (LOCK.md สร้างจากไฟล์นี้ ห้ามแก้ LOCK.md ด้วยมือ) |
  | `task_b_v2/TASK_B_V2_SPEC.md` | ข้อ 2.2 (ข้อมูลใหม่) + ข้อ 4.1/4.4 (replay 17 วัน · B36) |
  | `task_b_v2/DECISIONS.md` | B40 + ผลของ P6 (B36–B39 อยู่ใน commit ก่อนหน้าแล้ว บางส่วนเพิ่งเติมในรอบนี้) |
  | `task_b_v2/results/experiment_log.csv` | ต่อท้าย 24 แถวของ Stage 6 → รวม **45** config |
  | `task_b_v2/results/stage6_hpo/` | ผลค้นหา/placebo/ยืนยัน/รายงาน/รูปของ Stage 6 (ไฟล์ใหม่ที่ยัง untracked · cache ถูก .gitignore กันไว้) |

  คำสั่งที่เสนอ (ผู้ใช้รันเอง — Claude ไม่รัน):

  ```bash
  git add task_b_v2/LOCK.md task_b_v2/data task_b_v2/src/settings.py task_b_v2/src/draft_lock.py task_b_v2/TASK_B_V2_SPEC.md task_b_v2/DECISIONS.md task_b_v2/results/experiment_log.csv task_b_v2/results/stage6_hpo
  git commit -m "task_b_v2: lock configs after Stage 6 (no winner, B38-B40), refresh data to 2026-10-22, replay 17 days (B36), declare S1 (B37)"
  ```

  ช่อง commit hash: `__________`
- commit ของโหมด 16:00 (`mode1600/` + `LOCK_1600.md` · อีก session): `__________` — **ต้องเสร็จก่อนเริ่ม C6**
- วันที่ผู้ใช้ยืนยัน LOCK: `__________`
- commit ที่ 3 (หลัง C6–C7) — ผล final + replay + figures + README: `__________`
"""
    out = paths.TASK_DIR / "LOCK.md"
    out.write_text(md, encoding="utf-8")
    print(f"เขียน {out} ({len(md):,} ตัวอักษร) · config ที่ลอง {n_cfg}")


if __name__ == "__main__":
    main()
