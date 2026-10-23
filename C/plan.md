# Task: Stock Price Up/Down Prediction with 3-Year Temporal Split

## 1. วัตถุประสงค์
สร้าง task ใหม่สำหรับการพัฒนาระบบทำนายทิศทางราคาหุ้นวันถัดไปแบบ Binary Classification (ขึ้น / ลง) โดยอ้างอิงแนวคิดจากแผนงานและ task อื่น ๆ ที่มีอยู่แล้ว แต่ไม่แตะ pipeline เก่า/โค้ดเดิม

เป้าหมายหลัก:
- ทำนายว่าราคาปิดของหุ้นในวันถัดไปจะขึ้นหรือลง
- ใช้ข้อมูลราคาหุ้นรายวันและปริมาณการซื้อขาย
- ประยุกต์ใช้ Spectrogram + CNN + ANN
- แยกข้อมูลเป็น Train / Valid / Test ตามเวลาจริง 70/15/15
- ใช้ข้อมูลประมาณ 3 ปีเพื่อให้เริ่มต้นได้เร็วและง่ายต่อการทดลอง

อ้างอิงจาก:
- [stock_prediction_project_plan.md](stock_prediction_project_plan.md)
- task อื่น ๆ ในโฟลเดอร์ C ที่เกี่ยวกับแนวทางการทำนายขึ้น/ลง

---

## 2. ขอบเขตงาน

### 2.1 ข้อมูลนำเข้า
- ข้อมูลราคาหุ้นรายวัน: Open, High, Low, Close, Volume
- เลือกช่วงข้อมูลประมาณ 3 ปีที่เหมาะสมที่สุดจากชุดข้อมูลที่มีอยู่
- ต้องเรียงข้อมูลตามวันที่ก่อนทำ split

### 2.2 Feature ที่ใช้
1. Close log-return
2. Open gap
3. High-Low range
4. Volume log-change

### 2.3 Target label
- 1 = UP เมื่อ Close(t+1) > Close(t)
- 0 = DOWN เมื่อ Close(t+1) <= Close(t)

### 2.4 Metrics และประเมินผล
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

---

## 3. Data Split Strategy
ใช้ split แบบลำดับเวลาโดยไม่มีการสุ่มข้อมูล:

- Train: 70%
- Valid: 15%
- Test: 15%

กติกา:
- ต้องเรียงตามวันที่ก่อน split
- ไม่ใช้ข้อมูลจากอนาคตในช่วง train
- validation ใช้สำหรับ tuning
- test ใช้สำหรับประเมินผลสุดท้าย

### ตัวอย่างแผนการใช้ 3 ปี (2023 - 2026)
ถ้าใช้ข้อมูล 3 ปีเต็ม:
- 70% = ส่วนแรกของ 3 ปี
- 15% = ส่วนถัดไป
- 15% = ส่วนสุดท้าย

ทั้งหมดต้องเป็นช่วงต่อเนื่องตามเวลา ไม่แยกแบบสุ่ม

---

## 4. Task Breakdown

### Task 1: เลือกช่วงข้อมูลและทำความสะอาด
- [x] เลือกช่วงข้อมูล 3 ปีล่าสุดที่ถูกต้องและมีความครบถ้วน
- [x] ตรวจสอบวันที่และลำดับเวลา
- [x] ตรวจสอบ missing value / duplicated rows
- [x] จัดเรียงข้อมูลตามวันเวลา

### Task 2: สร้าง Feature และ Target
- [x] คำนวณ Close log-return
- [x] คำนวณ Open gap
- [x] คำนวณ High-Low range
- [x] คำนวณ Volume log-change
- [x] สร้าง label ขึ้น/ลงสำหรับวันถัดไป

### Task 3: ทำ Train / Valid / Test Split
- [x] แบ่งข้อมูลเป็น 70/15/15 ตามลำดับเวลา
- [x] ตรวจสอบว่าไม่มี leakage ระหว่าง train-valid-test
- [x] เก็บ index หรือ date range ของแต่ละชุด

### Task 4: สร้าง Spectrogram
- [x] ใช้ rolling window 64 วัน
- [x] แปลงข้อมูลแต่ละ feature เป็นสัญญาณ 1D
- [x] ใช้ STFT ด้วย nperseg = 16
- [x] ตั้ง noverlap = 14
- [x] ปรับสเกลด้วย log1p(power spectrum)
- [x] รวมเป็น tensor 4-channel

### Task 5: Implement Model
- [x] สร้าง CNN feature extractor
- [x] ใช้ shallow CNN 2 blocks
- [x] ใช้ kernel 3 x 2
- [x] ใส่ BatchNormalization และ MaxPooling2D
- [x] Flatten และต่อ ANN
- [x] Dense 64 พร้อม Dropout 0.4
- [x] Output layer ใช้ Sigmoid ตอนแปลง logit เป็น probability

### Task 6: Training และ Validation
- [x] ตั้ง BCEWithLogitsLoss สำหรับ binary classification
- [x] ใช้ Adam, learning rate 1e-3
- [x] Train บน train split
- [x] ใช้ valid split สำหรับ early stopping
- [x] บันทึก learning curve เพื่อตรวจ overfitting

### Task 7: Evaluation และ Reporting
- [x] ประเมินผลบน test split
- [x] บันทึก Accuracy, Precision, Recall, F1, ROC-AUC
- [x] สร้าง confusion matrix
- [x] สรุปผลใน report.md ที่สร้างอัตโนมัติ

---

## 5. Deliverables
- Script สำหรับเลือกข้อมูล 3 ปี
- Script สำหรับสร้าง feature และ label
- Script สำหรับ split 70/15/15 ตามเวลา
- Script สำหรับสร้าง spectrogram
- โมเดล CNN-ANN สำหรับทำนายขึ้น/ลง
- Metrics summary และ report

---

## 6. Success Criteria
- มี workflow ที่ทำงานได้จริงสำหรับปัญหาทำนายราคา ขึ้น/ลง
- ไม่มี data leakage ระหว่าง train, valid, test
- ได้ผลลัพธ์ที่สามารถประเมินโดยชัดเจนจาก test split
- แนวทางนี้สามารถต่อยอดหรือเชื่อมเข้ากับโครงการหลักได้ต่อไป

---

## 7. Notes
สิ่งนี้คือ task ใหม่ที่แยกออกมาเพื่อให้เริ่มพัฒนาได้โดยไม่ต้องแก้โค้ดเดิมหรือ pipeline หลัก ใช้แนวทางจาก task/plan เดิมเป็นฐาน แต่ปรับให้ตรงกับข้อกำหนดใหม่: 3 ปี + 70/15/15 + temporal split.

ถ้าต้องการ สามารถต่อยอดต่อได้เป็น:
- issue list ตาม phase
- checklist สำหรับคนทำงาน
- schedule / owner / due date
