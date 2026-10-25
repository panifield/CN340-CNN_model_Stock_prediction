# C — Stock Price Up/Down Prediction

โปรเจกต์นี้ทำนายทิศทางราคาปิดของวันถัดไป:

- `1 (UP)` เมื่อ `Close(t+1) > Close(t)`
- `0 (DOWN)` เมื่อ `Close(t+1) <= Close(t)`

ใช้ข้อมูลย้อนหลัง 3 ปี, feature 4 ช่อง (`close return`, `open gap`, `high-low range`, `volume change`), แปลงเป็น spectrogram 4-channel แล้วเข้า CNN + ANN. ข้อมูลถูกแบ่งตามเวลาเป็น Train/Validation/Test = 70/15/15 ของตัวอย่างที่ใช้ได้ โดยเว้น 1 sample ที่รอยต่อแต่ละชุด เพราะ label ใช้ราคาวันถัดไป; normalization ใช้สถิติจาก train เท่านั้น

## โครงสร้าง

```text
C/
├── main.py                 # จุดเริ่มต้นการรัน
├── src/
│   ├── config.py            # การตั้งค่า
│   ├── data.py              # data, features, labels, STFT และ temporal split
│   ├── model.py             # CNN + ANN
│   ├── train.py             # train, early stopping และ metrics
│   ├── pipeline.py          # workflow และการบันทึกผล
│   └── visualize.py         # กราฟการเทรน/spectrogram/confusion matrix
└── outputs/<STOCK>/
    ├── model/model.pt
    ├── results/validation_summary.json
    ├── results/validation_report.md
    ├── results/validation_predictions.csv
    ├── results/training_history.json
    ├── results/test_summary.json
    ├── results/test_predictions.csv
    ├── figures/
    └── graphs/
        ├── validation_accuracy_by_epoch.png
        └── validation_roc_curve.png
```

## วิธีรัน

จาก root ของ repository:

```bash
# 1) เทรนและจูนด้วย train + validation เท่านั้น (default; ยังไม่วัด test)
python3 C/main.py --stock ADVANC --mode train-valid --epochs 50

# 2) หลังเลือกการตั้งค่าที่พอใจแล้ว จึงวัด test จาก checkpoint ที่บันทึกไว้
python3 C/main.py --stock ADVANC --mode test

# ทางลัด: รันทั้งสองขั้นตอนต่อเนื่องกัน
python3 C/main.py --stock KBANK --mode full --years 3 --epochs 50
```

ต้องมี `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` และไฟล์ `raw_data/<STOCK>_10Y_Cleaned.csv`.

```bash
python3 -m pip install -r C/requirements.txt
python3 C/tests/check_pipeline.py  # ตรวจ target, split และ train-only normalization
```

ตั้ง `TASKC_DEVICE=cpu` หากต้องการบังคับใช้ CPU. ผลสำหรับจูนอยู่ที่ `outputs/<STOCK>/results/validation_summary.json`, `validation_report.md`, `validation_predictions.csv` และ `training_history.json`; หลังสั่ง `--mode test` ผลสุดท้ายอยู่ที่ `test_summary.json`. โมเดลไม่ใช้ test split ระหว่าง training หรือ early stopping.
