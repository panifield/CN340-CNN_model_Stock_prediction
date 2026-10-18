"""ค่าตั้งค่าทั้งหมดของ Task A"""
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent      # .../A
ROOT = TASK_DIR.parent                                 # โฟลเดอร์โปรเจกต์
DATA_DIR = ROOT / "raw_data"
OUT_DIR = TASK_DIR / "outputs"

STOCKS = ["KBANK", "ADVANC"]
START_DATE = "2025-09-07"  # 7 ก.ย. 2568

# spectrogram
WINDOW = 30          # จำนวนวันย้อนหลังที่ใช้สร้าง spectrogram
NPERSEG = 10         # ความยาวหน้าต่าง STFT
NOVERLAP = 8
CHANNELS = ["ret_close", "gap_open", "range_hl", "chg_vol"]
CHANNEL_LABELS = {
    "ret_close": "Close log-return",
    "gap_open": "Open gap",
    "range_hl": "High-Low range",
    "chg_vol": "Volume log-change",
}

# training
SPLIT = (0.70, 0.15, 0.15)   # train / val / test (ตามเวลา)
EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-3
N_SEEDS = 5
