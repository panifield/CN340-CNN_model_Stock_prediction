# CN340-CNN_model_Stock_prediction

ทำนายว่าราคาปิดของหุ้น KBANK และ ADVANC จะเป็น **เลขคู่หรือคี่** (ใช้ส่วนจำนวนเต็มของราคา)
ด้วยการแปลงข้อมูลราคาเป็น **spectrogram → CNN สกัด feature → ANN ทำนาย**

## โครงสร้าง

| โฟลเดอร์ | งาน |
|---|---|
| `raw_data/` | ราคารายวัน 10 ปีของ KBANK, ADVANC |
| `A/` | ทำนายวัน t ด้วยข้อมูลถึงวัน t-1 (รายวัน) |
| `A/experiments/` | เทียบ 34 วิธี, ยืนยันผลบนข้อมูล 9 ปี, ทดสอบ price clustering แบบ pre-registered |
| `A/tests/` | ตรวจ data leakage และ sanity check |
| `A2/` | ทำนายวัน t ตอน 16:00 (ก่อนตลาดปิด) ด้วยข้อมูลรายชั่วโมง |

## วิธีรัน

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib yfinance

python A/main.py                      # เทรน + สร้างภาพ spectrogram (A)
python A/tests/check_pipeline.py      # ตรวจ data leakage (A)
python A/experiments/search.py        # เทียบหลายวิธี (A)

python A2/download_data.py            # ดาวน์โหลดข้อมูลรายชั่วโมงจาก Yahoo Finance
python A2/tests/check_leak.py         # ตรวจ data leakage (A2)
python A2/main.py --extended          # เทรน/ประเมิน A2 แบบ walk-forward
```

ใช้ GPU อัตโนมัติถ้ามี (ตั้ง `TASKA_DEVICE=cpu` เพื่อบังคับใช้ CPU)
หมายเหตุ: Yahoo ให้ข้อมูลรายชั่วโมงย้อนหลัง ~730 วัน การดาวน์โหลดใหม่จะได้ช่วงเวลาเลื่อนไปตามวันที่รัน

## ผลสรุป

**A (ข้อมูลถึงวัน t-1):** ทำนายไม่ได้ดีกว่าการเดา
- ลอง 34 วิธี (ขนาด CNN, spectrogram หลายแบบ, wavelet, GAF, transfer learning, ensemble ฯลฯ) ได้ ~50%
- วิธีที่ดีที่สุดในปี 2568 (57.5%) เหลือ 50.5% เมื่อทดสอบบนข้อมูล 9 ปีก่อนหน้า
- แม้รู้ราคาเปิดของวัน t ก็ยังทายได้แค่ ~50%

**A2 (ณ 16:00, walk-forward ต.ค. 2567 – ต.ค. 2569, 982 วัน):**

| วิธี | ทั้งหมด | ราคา < 200 (ช่วงราคา 0.5 บาท) | ราคา ≥ 200 |
|---|---|---|---|
| ทายตามคู่/คี่ของราคา ณ 16:00 | **55.9%** (p=0.0001) | **62.6%** | 51.4% |
| CNN + ANN (skip connection) | 53.5% (p=0.016) | 61.1% | 48.4% |

- ทายได้จริงเฉพาะเมื่อช่วงราคาเป็น 0.5 บาท (กลไกของตลาด) -- ราคา ≥ 200 บาทยังทายไม่ได้
- spectrogram ไม่ได้เพิ่มข้อมูลเกินกว่าคู่/คี่ของราคา ณ 16:00

รายละเอียด: `A/experiments/results/*/report.md`, `A/experiments/clustering/results.md`, `A2/outputs/extended/report.md`
