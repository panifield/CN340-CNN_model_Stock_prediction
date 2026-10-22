"""
src — โค้ดหลักของ task_b_v2 (Spectrogram -> CNN -> ANN ทำนายราคาปิด KBANK / ADVANC)
================================================================================
ทำไมต้องเป็น package: ชื่อไฟล์ที่ SPEC กำหนด (models.py, evaluate.py) ซ้ำกับโมดูลระดับบนสุดของ legacy
(legacy import กันเองด้วยชื่อเปล่า เช่น `import models`) -- ถ้าเป็น package จะเรียกเป็น `src.models` จึงไม่ชนกัน

วิธีรันทุกอย่าง: ต้องอยู่ที่โฟลเดอร์ task_b_v2/ แล้วเรียกแบบ module เสมอ เช่น
    python -m src.check_env
    python -m pytest
(ห้ามรัน `python src/xxx.py` ตรง ๆ เพราะ sys.path จะเอา src/ ขึ้นก่อนแล้วชนกับ legacy)
"""
