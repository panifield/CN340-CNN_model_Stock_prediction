# Task C — Flow และรายละเอียดโมเดล CNN-ANN

เอกสารนี้อธิบาย workflow ที่ใช้งานจริงใน `C/src/` สำหรับทำนายทิศทางราคาปิดของหุ้นในวันถัดไป (`UP` / `DOWN`) รวมถึงตัวเลขของ layer, filter, kernel, activation, วิธีเทรน และเหตุผลที่เลือกใช้

> ค่าทั้งหมดเป็นค่าตั้งต้นปัจจุบันใน `C/src/config.py` และ `C/src/model.py` ไม่ใช่ค่าที่พิสูจน์แล้วว่าดีที่สุด การเปลี่ยนค่าใด ๆ ต้องเลือกด้วย Validation ก่อนเปิด Test

## 1. ภาพรวม

```text
OHLCV รายวันย้อนหลัง 3 ปี
        ↓
สร้าง feature 4 ตัว + label UP/DOWN
        ↓
ย้อนหลัง 64 วันต่อ 1 sample
        ↓
STFT ของแต่ละ feature → spectrogram 4-channel (4 × 9 × 33)
        ↓
Purged temporal split: Train / Valid / Test = 70 / 15 / 15
        ↓
Normalize ด้วยสถิติ Train เท่านั้น
        ↓
CNN: Conv(4→8) → Conv(8→16)
        ↓
ANN: 256 → 16 → 1 logit
        ↓
Sigmoid ตอนทำนาย → P(UP)
```

## 2. ข้อมูลเข้าและ target

### 2.1 แหล่งข้อมูล

อ่านไฟล์ `raw_data/ADVANC_10Y_Cleaned.csv` หรือ `raw_data/KBANK_10Y_Cleaned.csv` แล้วเลือกช่วง **3 ปีล่าสุด** (`YEARS = 3`) โดยใช้คอลัมน์:

```text
Date, Open, High, Low, Close, Volume
```

ก่อนสร้าง feature โค้ดจะเรียงตาม `Date`, ตัดวันที่ซ้ำ (เก็บแถวสุดท้าย) และตรวจว่าราคา/volume เป็นบวก เพราะ feature ใช้ log

### 2.2 Feature 4 channels

ทุก feature ถูกคำนวณรายวัน ดังนี้

| Channel | ชื่อในโค้ด | สูตร | เหตุผล |
|---|---|---|---|
| 1 | `ret_close` | `log(Close(t) / Close(t-1))` | วัดผลตอบแทนราคา โดยลดผลของระดับราคาที่ต่างกันระหว่างหุ้น |
| 2 | `gap_open` | `log(Open(t) / Close(t-1))` | เก็บข้อมูล gap ระหว่างปิดเมื่อวานกับเปิดวันนี้ |
| 3 | `range_hl` | `log(High(t) / Low(t))` | วัด intraday range หรือความผันผวนภายในวัน |
| 4 | `chg_vol` | `log(Volume(t) / Volume(t-1))` | วัดการเปลี่ยนของปริมาณซื้อขาย |

การใช้ log ratio ทำให้ข้อมูลเป็นสัดส่วนการเปลี่ยนแปลง ไม่ใช่ระดับราคาดิบ จึงช่วยให้ scale เสถียรกว่าเมื่อราคาเปลี่ยนตามเวลา

### 2.3 Label

```text
y(t) = 1  (UP)    เมื่อ Close(t+1) > Close(t)
y(t) = 0  (DOWN)  เมื่อ Close(t+1) ≤ Close(t)
```

แถวสุดท้ายของข้อมูลถูกตัดออก เพราะไม่มี `Close(t+1)` สำหรับสร้าง label

### 2.4 หนึ่ง sample ใช้ข้อมูลอะไร

สำหรับ sample ของวัน `t`:

```text
Input  X(t) = feature ของวัน t-64, ..., t-1   (รวม 64 วัน)
Target y(t) = ทิศทาง Close(t) → Close(t+1)
```

ดังนั้น input ไม่มี feature ของวัน `t` หรือวันหลังจากนั้น จึงไม่ใช้ข้อมูลอนาคตเป็น input

## 3. Spectrogram input

แต่ละ feature 1D ที่ยาว 64 วันถูกแปลงด้วย Short-Time Fourier Transform (STFT)

| ค่า | ตัวเลข | ความหมาย / เหตุผล |
|---|---:|---|
| Rolling window | 64 วัน | มีบริบทประมาณ 3 เดือนซื้อขาย แต่ยังเหลือตัวอย่างพอสำหรับข้อมูล 3 ปี |
| `nperseg` | 16 | STFT มองสัญญาณครั้งละ 16 วัน เพื่อแยก pattern ความถี่ระยะสั้น |
| `noverlap` | 14 | หน้าต่างติดกันเลื่อนครั้งละ 2 วัน จึงเห็นการเปลี่ยนแปลงตามเวลาได้ละเอียด |
| Window function | Hann | ลด spectral leakage ที่ขอบของแต่ละ frame |
| Power scale | `log1p(|STFT|²)` | ทำให้ power ที่มีช่วงกว้างไม่ครอบงำค่าเล็ก |

ผลของ 1 channel มี shape **9 frequency bins × 33 time bins**. เมื่อนำ 4 feature มา stack กัน input หนึ่งตัวอย่างจึงมี shape:

```text
(channels, frequency, time) = (4, 9, 33)
```

เมื่อเทรนแบบ batch 32 shape คือ:

```text
(batch, channels, frequency, time) = (32, 4, 9, 33)
```

## 4. การแบ่งข้อมูลและการป้องกัน leakage

แบ่งตามลำดับเวลา ไม่สุ่มข้อมูล:

```text
Train 70% → เว้น 1 sample → Validation 15% → เว้น 1 sample → Test 15%
```

การเว้น 1 sample (`embargo = 1`) สำคัญเพราะ label ของวัน `t` ใช้ `Close(t+1)`. จึงป้องกันไม่ให้ label สุดท้ายของ Train ใช้ราคาที่อยู่ใน Validation และไม่ให้ label สุดท้ายของ Validation ใช้ราคาที่อยู่ใน Test

การ normalize ทำหลัง split โดยหา mean และ standard deviation แยกตาม channel จาก **Train เท่านั้น**:

```text
X_normalized = (X - mean_train) / (std_train + 1e-6)
```

Validation ใช้สำหรับจูนและ early stopping; Test ถูกสงวนไว้สำหรับประเมิน checkpoint ที่เลือกแล้วด้วย `--mode test`

## 5. โครงสร้าง CNN และ ANN

โมเดลชื่อ `SpectroCNNANN` ใน `C/src/model.py`. มี CNN 2 convolution blocks ตามด้วย ANN classifier

### 5.1 ตาราง layer และขนาด output

| ลำดับ | Layer | Hyperparameter | Output shape ต่อ 1 sample | บทบาท |
|---:|---|---|---|---|
| 0 | Input | 4 channels | `4 × 9 × 33` | spectrogram ของ 4 feature |
| 1 | Conv2D #1 | `4 → 8` filters, kernel `3 × 2`, padding `(1, 0)` | `8 × 9 × 32` | หา pattern พื้นฐานข้าม channels/frequency/time |
| 2 | BatchNorm2D | 8 channels | `8 × 9 × 32` | ทำ distribution ของ activation เสถียรขึ้น |
| 3 | ReLU | - | `8 × 9 × 32` | เพิ่ม non-linearity |
| 4 | MaxPool2D | pool `2 × 2` | `8 × 4 × 16` | ลดขนาดและเก็บ activation เด่น |
| 5 | Conv2D #2 | `8 → 16` filters, kernel `3 × 2`, padding `(1, 0)` | `16 × 4 × 15` | รวม pattern พื้นฐานให้ซับซ้อนขึ้น |
| 6 | BatchNorm2D | 16 channels | `16 × 4 × 15` | ทำ activation เสถียร |
| 7 | ReLU | - | `16 × 4 × 15` | เพิ่ม non-linearity |
| 8 | AdaptiveAvgPool2D | output `4 × 4` | `16 × 4 × 4` | ทำขนาด feature map คงที่ก่อนเข้า ANN |
| 9 | Flatten | - | `256` | แปลง feature map เป็น vector (`16×4×4`) |
| 10 | Linear / Dense #1 | `256 → 16` | `16` | เรียนรู้การรวม CNN features |
| 11 | ReLU | - | `16` | เพิ่ม non-linearity ให้ ANN |
| 12 | Dropout | `p = 0.5` | `16` | ปิด neuron แบบสุ่มเฉพาะตอน train เพื่อลด overfitting |
| 13 | Linear / Dense #2 | `16 → 1` | `1` logit | คะแนนดิบสำหรับคลาส UP |

### 5.2 Filter และ kernel

#### Conv2D #1

```python
nn.Conv2d(4, 8, kernel_size=(3, 2), padding=(1, 0))
```

- มี **8 filters**
- filter หนึ่งตัวมี weight shape `(4, 3, 2)` = 4 input channels × 3 frequency bins × 2 time bins
- filter แต่ละตัวจึงเรียนรู้ pattern จาก feature ทั้ง 4 ตัวพร้อมกันในบริเวณเล็กของ spectrogram
- จำนวน parameter: `(4 × 3 × 2 + 1 bias) × 8 = 200`

#### Conv2D #2

```python
nn.Conv2d(8, 16, kernel_size=(3, 2), padding=(1, 0))
```

- มี **16 filters**
- filter หนึ่งตัวมี weight shape `(8, 3, 2)` เพราะรับ 8 feature maps จาก Conv #1
- จำนวน parameter: `(8 × 3 × 2 + 1 bias) × 16 = 784`
- Layer นี้ไม่ได้มอง return หรือ volume ดิบโดยตรงแล้ว แต่รวม pattern ที่ Conv #1 สกัดไว้

#### เหตุผลของ kernel `3 × 2`

แกนสูงของ spectrogram คือ frequency และแกนกว้างคือ time ดังนั้น kernel นี้มอง 3 frequency bins และ 2 time bins ในครั้งเดียว

- `3` บน frequency: เห็นความสัมพันธ์ของความถี่ที่อยู่ติดกัน
- `2` บน time: เห็นการเปลี่ยนแปลงระหว่างช่วงเวลาติดกัน โดยไม่ขยาย receptive field เร็วเกินไป
- input มีขนาดเล็กเพียง `9 × 33` และ Train มีประมาณ 463 samples จึงใช้ kernel เล็กและ CNN ตื้นเพื่อลดจำนวน parameter/ความเสี่ยง overfitting

`padding=(1, 0)` เติมด้าน frequency 1 bin แต่ไม่เติมด้าน time: ความสูงจึงคงไว้ ขณะที่ time ลดลง 1 bin ในแต่ละ convolution

### 5.3 ANN และจำนวน parameter

```python
nn.Linear(16 * 4 * 4, 16)  # 256 → 16
nn.ReLU()
nn.Dropout(0.5)
nn.Linear(16, 1)           # 16 → 1
```

| Layer | Parameter |
|---|---:|
| Conv #1 | 200 |
| BatchNorm #1 (gamma, beta) | 16 |
| Conv #2 | 784 |
| BatchNorm #2 (gamma, beta) | 32 |
| Dense `256 → 16` | 4,112 |
| Dense `16 → 1` | 17 |
| **รวม trainable parameters** | **5,161** |

`AdaptiveAvgPool2D`, `ReLU`, `MaxPool2D`, `Dropout`, `Flatten` ไม่มี trainable parameter

เหตุผลที่ ANN เล็ก (`256 → 16 → 1`) และ Dropout สูง (`0.5`) คือจำนวนตัวอย่าง Train มีไม่มาก จึงต้องลดความสามารถของโมเดลเพื่อช่วยลด overfitting

## 6. Activation functions และ output probability

| จุด | Function | เหตุผล |
|---|---|---|
| หลัง Conv #1 และ Conv #2 | `ReLU(x) = max(0, x)` | ทำให้ CNN เรียนรู้ความสัมพันธ์ non-linear, คำนวณง่าย และช่วยลดปัญหา vanishing gradient เทียบกับ sigmoid/tanh ใน hidden layers |
| หลัง Dense #1 | ReLU | ให้ ANN เรียนรู้การรวม feature แบบ non-linear |
| ก่อน loss | **ไม่มี sigmoid ใน model** | model ส่งออก `logit` ดิบ 1 ค่า เพื่อใช้ loss ที่เสถียรกว่า |
| ตอนทำนาย | `sigmoid(logit)` | แปลง logit เป็น `P(UP)` อยู่ในช่วง 0 ถึง 1 |

ตัวอย่าง:

```text
logit = 0      → sigmoid = 0.50
logit = 1.386  → sigmoid ≈ 0.80
logit = -1.386 → sigmoid ≈ 0.20
```

เกณฑ์ default คือ `P(UP) ≥ 0.50` ทาย UP; น้อยกว่า 0.50 ทาย DOWN. เกณฑ์นี้เปลี่ยน Accuracy/Precision/Recall ได้ แต่ไม่เปลี่ยน ROC-AUC

## 7. Training

| รายการ | ค่าที่ใช้ | เหตุผล |
|---|---:|---|
| Loss | `BCEWithLogitsLoss` | เหมาะกับ binary classification และรวม sigmoid กับ binary cross-entropy แบบ numerical stable |
| Optimizer | Adam | ปรับ learning rate ตามพารามิเตอร์ ช่วยเริ่มต้นทดลองกับข้อมูลขนาดเล็กได้สะดวก |
| Learning rate | `1e-3` | ค่าเริ่มต้นมาตรฐานของ Adam; ควรลอง `1e-4`, `3e-4` ใน validation หากไม่เสถียร |
| Weight decay | `1e-4` | L2 regularization เพื่อลด weight ใหญ่เกินและช่วยลด overfitting |
| Batch size | 32 | สมดุลระหว่างความเสถียรของ gradient และขนาดข้อมูล |
| Max epochs | 50 | เป็นเพดาน ไม่จำเป็นต้องรันครบ |
| Early stopping patience | 10 epochs | หยุดหาก validation loss ไม่ดีขึ้นต่อเนื่อง 10 epochs |
| Improvement threshold | `1e-5` | ต้องดีขึ้นเกินค่านี้จึงนับว่า validation loss ดีขึ้น |
| Seed | 42 | ทำให้ผลรันซ้ำได้ใกล้เคียงเดิม |

ในแต่ละ epoch:

```text
สุ่มลำดับ Train samples
    ↓
แบ่งเป็น mini-batch 32
    ↓
forward → BCEWithLogitsLoss → backpropagation → Adam update
    ↓
คำนวณ train loss และ validation loss/accuracy
    ↓
เก็บ model state ที่ validation loss ต่ำที่สุด
    ↓
หยุดเมื่อ patience ครบ แล้วคืน state ที่ดีที่สุด
```

BatchNorm ต้องการ batch อย่างน้อย 2 ตัวอย่าง โค้ดจึงข้าม batch สุดท้ายถ้ามีเพียง 1 ตัวอย่าง

ใช้ GPU อัตโนมัติถ้ามี; ตั้ง `TASKC_DEVICE=cpu` เพื่อบังคับ CPU

## 8. Validation, baseline และ Test

### 8.1 Validation สำหรับการจูน

คำสั่ง:

```bash
python3 C/main.py --stock ADVANC --mode train-valid --epochs 50
```

ใช้ Train สำหรับ update weight และ Validation สำหรับ early stopping/เปรียบเทียบ hyperparameter เท่านั้น ผลที่บันทึกได้แก่ validation metrics, prediction รายวัน, learning history, confusion matrix, accuracy-by-epoch และ ROC curve

Baseline ที่รายงานคือ **majority-class baseline**: หาคลาส UP/DOWN ที่พบมากที่สุดใน Train แล้วทายคลาสนั้นทุกวันใน Validation. CNN ควรชนะ baseline นี้ก่อนจึงมีเหตุผลที่จะนำไปทดสอบ Test

### 8.2 Test final evaluation

หลังเลือก architecture และ hyperparameter โดยไม่ดู Test แล้วเท่านั้น:

```bash
python3 C/main.py --stock ADVANC --mode test
```

คำสั่งนี้โหลด `outputs/ADVANC/model/model.pt` และคำนวณ Test metrics ครั้งสุดท้าย ไม่เทรนเพิ่ม

### 8.3 Metrics

| Metric | ความหมาย |
|---|---|
| Accuracy | สัดส่วนที่ทาย UP/DOWN ถูกต้องที่ threshold 0.5 |
| Precision | ในวันที่โมเดลทาย UP มีสัดส่วน UP จริงเท่าไร |
| Recall | ในวันที่ UP จริง โมเดลจับได้เท่าไร |
| F1 | ค่าเฉลี่ย harmonic ของ precision และ recall |
| ROC-AUC | คุณภาพการเรียงลำดับ probability ข้ามทุก threshold; 0.5 ใกล้เคียงการสุ่ม |
| Confusion matrix | จำนวน DOWN/UP จริงเทียบกับค่าที่ทำนาย |

## 9. จุดที่ควรจูนต่อ

ผล Validation เป็นเกณฑ์เดียวในการเลือก ไม่ใช้ Test ระหว่างจูน ตัวอย่างค่าที่ทดลองได้:

```text
filters:       (4→8, 8→16), (4→16, 16→32)
kernel size:   (3,2), (3,3), (2,3)
dense units:   8, 16, 32
dropout:       0.4, 0.5, 0.6
learning rate: 1e-4, 3e-4, 1e-3
weight decay:  0, 1e-4, 1e-3
```

เนื่องจากข้อมูลมีน้อย ควรรันหลาย seed และเทียบค่าเฉลี่ย/ส่วนเบี่ยงเบนของ Validation AUC มากกว่าตัดสินจาก seed เดียว หากจูนหลายแบบแล้วยังได้ AUC ใกล้ 0.5 หรือแพ้ majority baseline ควรรายงานว่าชุดข้อมูลและ feature นี้ยังไม่มี predictive signal ที่น่าเชื่อถือสำหรับ next-day direction
