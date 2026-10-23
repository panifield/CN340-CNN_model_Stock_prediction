"""ค่าตั้งต้นของงาน C: next-day stock up/down classification."""
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
ROOT = TASK_DIR.parent
DATA_DIR = ROOT / "raw_data"
OUT_DIR = TASK_DIR / "outputs"

STOCKS = ["ADVANC", "KBANK"]
YEARS = 3

# Input ของตัวอย่างในวัน t คือข้อมูลตั้งแต่ t-WINDOW ถึง t-1 เท่านั้น
WINDOW = 64
NPERSEG = 16
NOVERLAP = 14
CHANNELS = ["ret_close", "gap_open", "range_hl", "chg_vol"]
CHANNEL_LABELS = {
    "ret_close": "Close log-return",
    "gap_open": "Open gap",
    "range_hl": "High-Low range",
    "chg_vol": "Volume log-change",
}

SPLIT = (0.70, 0.15, 0.15)
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
