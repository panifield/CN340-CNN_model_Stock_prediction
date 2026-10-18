"""
Task A: ทำนายราคาปิด "คู่/คี่" ด้วย Spectrogram -> CNN (สกัด feature) -> ANN (classifier)

Usage:
    python A/main.py                     # เทรน + สร้างภาพ ทั้ง KBANK และ ADVANC
    python A/main.py --stock KBANK
    python A/main.py --no-per-day        # ไม่บันทึกภาพ spectrogram รายวัน
"""
import argparse
import json

from src.config import N_SEEDS, OUT_DIR, STOCKS
from src.pipeline import run_stock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", choices=STOCKS + ["ALL"], default="ALL")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--no-per-day", action="store_true", help="ไม่บันทึกภาพ spectrogram รายวัน")
    args = ap.parse_args()

    stocks = STOCKS if args.stock == "ALL" else [args.stock]
    results = [run_stock(s, list(range(args.seeds)), per_day_figs=not args.no_per_day)
               for s in stocks]
    with open(OUT_DIR / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
