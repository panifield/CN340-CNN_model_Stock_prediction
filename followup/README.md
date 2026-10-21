# Follow-up: 5 การวิเคราะห์เพิ่มเติม

| # | งาน | สคริปต์ | ผล (ไฟล์) |
|---|---|---|---|
| 1 | ROC + AUC | `followup/roc_threshold.py` | `results/roc_curves.png`, `results/auc_by_subset.png`, `results/roc_threshold.md` |
| 2 | Threshold sweep | `followup/roc_threshold.py` | `results/threshold_sweep.png`, `results/threshold_table.csv` |
| 3 | Label แบบ x2 | `A/experiments/search.py --label x2` + `followup/x2_label.py` | `results/x2_label.png`, `results/x2_label.md` |
| 4 | Pretrain 2016-2023 -> fine-tune | `A/experiments/search.py --only pretrain8y_ft,pretrain8y_frozen` | `A/experiments/results/2025-09_now/report.md` |
| 5 | Optuna (HyperOps) | `followup/optuna_search.py --trials 50` | `results/optuna/optuna.png`, `results/optuna/report.md` |

## 1. ROC + AUC (ค่า P(odd) แบบ out-of-sample ที่มีอยู่แล้ว ไม่เทรนใหม่)

| แหล่ง | โมเดล | n | AUC | 95% CI | p |
|---|---|---|---|---|---|
| A ทดสอบซ้ำ 9 ปี | hybrid_extras | 2016 | 0.506 | 0.481-0.531 | 0.32 |
| A ทดสอบซ้ำ 9 ปี | base | 2016 | 0.505 | 0.479-0.532 | 0.34 |
| A เทรน 10 ปี | hybrid_extras | 472 | 0.530 | 0.478-0.582 | 0.13 |
| A2 16:00 | cnn_ann (เดิม) | 982 | 0.504 | 0.466-0.540 | 0.42 |
| A2 16:00 | cnn_ann_skip_mlp | 982 | **0.544** | 0.507-0.581 | 0.009 |
| A2 16:30 | cnn_ann_skip_mlp | 984 | **0.586** | 0.550-0.621 | 2e-6 |

แยกช่วงราคา (A2 cnn_ann_skip_mlp): **tick 0.5 บาท AUC 0.606 (16:00) / 0.646 (16:30)**,
tick 1 บาท 0.466 / 0.513 (เท่าการเดา) -- ยืนยันผลเดิมจากอีกมุม: A ทายไม่ได้, A2 ทายได้เฉพาะช่วงราคา 0.5 บาท

## 2. Threshold sweep 0.30-0.70

- มองย้อนหลัง threshold ที่ดีที่สุดเพิ่ม accuracy ได้เล็กน้อย (เช่น A2 16:00: 0.535 -> 0.544 ที่ 0.53)
- ทดสอบแบบไม่โกง (เลือก threshold บนครึ่งแรกตามเวลา -> วัดครึ่งหลัง): ไม่มีกรณีไหนดีกว่า 0.5 อย่างมีนัยสำคัญ
  - A ทดสอบซ้ำ: 0.463-0.471 (แย่กว่าใช้ 0.5)
  - A2 16:30: 0.533 vs 0.561 ที่ 0.5 -- การเลือก threshold กลับทำให้แย่ลง
- สรุป: ใช้ 0.5 ต่อไป การเลือก threshold จากข้อมูลย้อนหลังเป็น overfitting อีกแบบหนึ่ง

## 3. Label แบบ x2: `round(close*2) % 2`

- label นี้ = "ราคาปิดลงท้าย .5 หรือไม่" มีความหมายเฉพาะตอน tick 0.5 บาท
- ADVANC ปี 2023-2026 ราคา >= 200 (tick 1 บาท) -> label เป็น 0 เกือบ/ทุกวัน (คี่ 0-3%)
- accuracy ดูสูง (64-87%) แต่ **ต่ำกว่าการทายคลาสใหญ่เสมอทุกกรณี** (hybrid - majority = -0.4 ถึง -14 จุด)
- วันที่ tick 0.5 (label สมดุล 51% คี่, n=1334): hybrid 0.516 vs majority 0.519 -> ทายไม่ได้เหมือนกัน
- สรุป: label x2 ไม่ได้ทำให้ทายง่ายขึ้น แค่ทำให้คลาสไม่สมดุล (ต้องดู baseline เสมอ)

## 4. Pretrain (2016-2023) -> fine-tune บนข้อมูลล่าสุด

ของเดิมใน 34 วิธี (`pretrain_hybrid`) เป็นแค่ self-supervised บนข้อมูลปีเดียว -- รอบนี้ทำใหม่ตามโจทย์:
เทรนโมเดลคู่/คี่บนข้อมูลปี 2559-2566 ของหุ้นเดียวกัน แล้วเอา CNN มาใช้ต่อ (ข้อมูล pretrain จบก่อนช่วงทดสอบ ~2 ปี)

| วิธี | walk-forward | holdout |
|---|---|---|
| pretrain8y_ft (fine-tune ทั้งโมเดล lr 1e-4) | 0.525 | 0.469 |
| pretrain8y_frozen (freeze CNN เทรนแค่ ANN) | 0.537 | 0.484 |
| base (เทรนปีเดียว) | 0.525 | 0.516 |
| เทรนรวม 10 ปีทีเดียว (hybrid_extras) | 0.528 (ทั้งช่วง 472 วัน) | |

-> ไม่ดีขึ้น: ข้อมูลเก่าไม่มี pattern คู่/คี่ที่ถ่ายทอดมาได้

## 5. Optuna 50 trials (TPE) -- ค้นบน walk-forward selection เท่านั้น

ค้น: transform (stft/complex/cwt), window, nperseg, channel 6 ตัว, extras, rounds, arch, noise, lr, weight decay,
batch size, bottleneck, skip

| | selection (ใช้เลือก) | holdout 2026 | ทดสอบซ้ำ 9 ปี (n=2016) |
|---|---|---|---|
| Optuna best (#42) | 0.556 (p=0.089) | **0.469** | **0.500** |
| base | 0.525 | 0.516 | 0.504 |
| hybrid_extras | 0.575 | 0.609 | 0.505 |

- best trial ไม่ผ่าน Bonferroni (p 0.089 > 0.05/50 = 0.001)
- ทุก trial อยู่ในแถบ 95% ของการเดา; ค่าดีสุดตกลงเหลือ 47-50% บนข้อมูลใหม่ = **หาดวง** ตามที่คาดไว้
- ความสำคัญ hyperparameter (fANOVA) กระจาย ไม่มีตัวไหนโดดเด่น (batch_size 0.20, weight_decay 0.19) -> สัญญาณของ noise

## ข้อสรุปรวม

ทั้ง 5 วิธียืนยันผลเดิม: **Task A ทายไม่ได้ (AUC ~0.50 ทุกมุม)**, **A2 ทายได้จริงเฉพาะวันที่ช่วงราคา 0.5 บาท**
(AUC 0.61-0.65) และการปรับ threshold / label / pretrain / hyperparameter ไม่ได้ทำให้ดีขึ้นบนข้อมูลที่ไม่เคยเห็น
