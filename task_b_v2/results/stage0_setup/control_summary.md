# C2 — โครงโมเดลและ control

base config · seeds = (0,) · โซน dev ≤ 2025-02-25

## โครงโมเดลและจำนวนพารามิเตอร์

### ก. 2D CNN (base)

input `(1, 9, 65)` · พารามิเตอร์รวม **10,081**

| ชั้น | shape ที่ออก | พารามิเตอร์ |
|---|---|---|
| input (ภาพ) | (1, 9, 65) | 0 |
| Conv2d | (8, 9, 65) | 80 |
| BatchNorm2d | (8, 9, 65) | 16 |
| ReLU | (8, 9, 65) | 0 |
| MaxPool2d | (8, 4, 32) | 0 |
| Conv2d | (16, 4, 32) | 1,168 |
| BatchNorm2d | (16, 4, 32) | 32 |
| ReLU | (16, 4, 32) | 0 |
| MaxPool2d | (16, 2, 16) | 0 |
| Flatten | (512,) | 0 |
| Linear | (16,) | 8,208 |
| ReLU | (16,) | 0 |
| = CNN feature | (16,) | 0 |
| Linear | (32,) | 544 |
| ReLU | (32,) | 0 |
| Dropout | (32,) | 0 |
| Linear | (1,) | 33 |

### ก. 2D CNN + hybrid (29 feature)

input `(1, 9, 65)` · พารามิเตอร์รวม **11,039**

| ชั้น | shape ที่ออก | พารามิเตอร์ |
|---|---|---|
| input (ภาพ) | (1, 9, 65) | 0 |
| Conv2d | (8, 9, 65) | 80 |
| BatchNorm2d | (8, 9, 65) | 16 |
| ReLU | (8, 9, 65) | 0 |
| MaxPool2d | (8, 4, 32) | 0 |
| Conv2d | (16, 4, 32) | 1,168 |
| BatchNorm2d | (16, 4, 32) | 32 |
| ReLU | (16, 4, 32) | 0 |
| MaxPool2d | (16, 2, 16) | 0 |
| Flatten | (512,) | 0 |
| Linear | (16,) | 8,208 |
| ReLU | (16,) | 0 |
| = CNN feature | (16,) | 0 |
| concat 29 feature เดิม (29) | (45,) | 0 |
| Linear | (32,) | 1,472 |
| ReLU | (32,) | 0 |
| Dropout | (32,) | 0 |
| Linear | (1,) | 33 |
| linear skip (+) | (1,) | 30 |

### ก. 2D CNN + GAP head

input `(1, 9, 65)` · พารามิเตอร์รวม **2,145**

| ชั้น | shape ที่ออก | พารามิเตอร์ |
|---|---|---|
| input (ภาพ) | (1, 9, 65) | 0 |
| Conv2d | (8, 9, 65) | 80 |
| BatchNorm2d | (8, 9, 65) | 16 |
| ReLU | (8, 9, 65) | 0 |
| MaxPool2d | (8, 4, 32) | 0 |
| Conv2d | (16, 4, 32) | 1,168 |
| BatchNorm2d | (16, 4, 32) | 32 |
| ReLU | (16, 4, 32) | 0 |
| MaxPool2d | (16, 2, 16) | 0 |
| GlobalAvgPool | (16,) | 0 |
| Flatten | (16,) | 0 |
| Linear | (16,) | 272 |
| ReLU | (16,) | 0 |
| = CNN feature | (16,) | 0 |
| Linear | (32,) | 544 |
| ReLU | (32,) | 0 |
| Dropout | (32,) | 0 |
| Linear | (1,) | 33 |

### ข. 1D CNN ตามแกนเวลา

input `(1, 9, 65)` · พารามิเตอร์รวม **5,985**

| ชั้น | shape ที่ออก | พารามิเตอร์ |
|---|---|---|
| input (ภาพ) | (1, 9, 65) | 0 |
| reshape (cnn1d_time) | (9, 65) | 0 |
| Conv1d | (16, 65) | 448 |
| BatchNorm1d | (16, 65) | 32 |
| ReLU | (16, 65) | 0 |
| MaxPool1d | (16, 32) | 0 |
| Conv1d | (16, 32) | 784 |
| BatchNorm1d | (16, 32) | 32 |
| ReLU | (16, 32) | 0 |
| MaxPool1d | (16, 16) | 0 |
| Flatten | (256,) | 0 |
| Linear | (16,) | 4,112 |
| ReLU | (16,) | 0 |
| = CNN feature | (16,) | 0 |
| Linear | (32,) | 544 |
| ReLU | (32,) | 0 |
| Dropout | (32,) | 0 |
| Linear | (1,) | 33 |

### ค. 1D CNN บน return ดิบ

input `(1, 128)` · พารามิเตอร์รวม **9,697**

| ชั้น | shape ที่ออก | พารามิเตอร์ |
|---|---|---|
| input (ภาพ) | (1, 128) | 0 |
| Conv1d | (16, 128) | 64 |
| BatchNorm1d | (16, 128) | 32 |
| ReLU | (16, 128) | 0 |
| MaxPool1d | (16, 64) | 0 |
| Conv1d | (16, 64) | 784 |
| BatchNorm1d | (16, 64) | 32 |
| ReLU | (16, 64) | 0 |
| MaxPool1d | (16, 32) | 0 |
| Flatten | (512,) | 0 |
| Linear | (16,) | 8,208 |
| ReLU | (16,) | 0 |
| = CNN feature | (16,) | 0 |
| Linear | (32,) | 544 |
| ReLU | (32,) | 0 |
| Dropout | (32,) | 0 |
| Linear | (1,) | 33 |

### ตรวจด้วยสูตรมือ (CS231n L5)

`out = (W − K + 2P)/S + 1` · `params = Cin·K·K·Cout + Cout`

| รายการ | สูตรให้ | ค่าจริง | ตรงกัน |
|---|---|---|---|
| conv1 3×3 pad 1 ไม่เปลี่ยนขนาด | 9 | 9 | ใช่ |
| maxpool 2×2 ลดครึ่ง (แกนความถี่) | 4 | 4 | ใช่ |
| maxpool 2×2 ลดครึ่ง (แกนเวลา) | 32 | 32 | ใช่ |
| พารามิเตอร์ conv1 = 1·3·3·8 + 8 | 80 | 80 | ใช่ |
| พารามิเตอร์ conv2 = 8·3·3·16 + 16 | 1168 | 1168 | ใช่ |
| ขนาดหลัง flatten = 16·2·16 | 512 | 512 | ใช่ |
| พารามิเตอร์ Linear(512→16) = 512·16 + 16 | 8208 | 8208 | ใช่ |

## positive control

| control | ticker | fold | n | R2 | corr | pred_std_ratio | passed |
|---|---|---|---|---|---|---|---|
| positive | KBANK | 1 | 266 | 0.8622 | 0.9297 | 0.8975 | True |
| positive | KBANK | 2 | 266 | 0.9001 | 0.9536 | 0.8617 | True |
| positive | KBANK | 3 | 266 | 0.8787 | 0.9414 | 0.8588 | True |
| positive | ADVANC | 1 | 266 | 0.8793 | 0.9443 | 0.8345 | True |
| positive | ADVANC | 2 | 266 | 0.928 | 0.9636 | 0.9748 | True |
| positive | ADVANC | 3 | 266 | 0.8788 | 0.9561 | 0.8041 | True |

**ผ่าน** — ผ่าน 6/6 evaluation · ค่าเฉลี่ย 0.8879 · แย่สุด 0.8622 · เกณฑ์ 0.3

## negative control

| control | ticker | fold | n | relMAE | IC | StdRatio | passed |
|---|---|---|---|---|---|---|---|
| negative | KBANK | 1 | 266 | 1.0214 | -0.0035 | 0.111 | True |
| negative | KBANK | 2 | 266 | 1.0524 | -0.0493 | 0.2138 | True |
| negative | KBANK | 3 | 266 | 1.0086 | 0.0011 | 0.0123 | True |
| negative | ADVANC | 1 | 266 | 1.0101 | 0.0103 | 0.0528 | True |
| negative | ADVANC | 2 | 266 | 1.0259 | -0.0752 | 0.0531 | True |
| negative | ADVANC | 3 | 266 | 1.0064 | -0.0074 | 0.0079 | True |

**ผ่าน** — ผ่าน 6/6 evaluation · ค่าเฉลี่ย 1.0208 · แย่สุด 1.0064 · เกณฑ์ 0.99

## เวลาที่ใช้ (SPEC ข้อ 8)

เฉลี่ย 24 วินาทีต่อ evaluation ที่ 1 seed → **1 config เต็ม (6 evaluation × 3 seeds) ≈ 7 นาที** (เครื่องผู้ใช้ · CPU 1 thread)

| control | ticker | fold | seconds |
|---|---|---|---|
| positive | KBANK | 1 | 23.1 |
| positive | KBANK | 2 | 36.2 |
| positive | KBANK | 3 | 29.2 |
| positive | ADVANC | 1 | 17.7 |
| positive | ADVANC | 2 | 25.3 |
| positive | ADVANC | 3 | 58.7 |
| negative | KBANK | 1 | 13.4 |
| negative | KBANK | 2 | 21.2 |
| negative | KBANK | 3 | 13.9 |
| negative | ADVANC | 1 | 10.4 |
| negative | ADVANC | 2 | 18.8 |
| negative | ADVANC | 3 | 18.2 |

