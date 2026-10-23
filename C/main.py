#!/usr/bin/env python3
"""คำสั่งรันโปรเจกต์ C: ทำนายทิศทางราคาปิดวันถัดไป."""
from __future__ import annotations

import argparse
import json

from src.config import EPOCHS, STOCKS, YEARS
from src.pipeline import run_stock, run_test, run_train_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="3-year temporal stock up/down prediction")
    parser.add_argument("--stock", choices=STOCKS, default="ADVANC", help="หุ้นที่ต้องการทดลอง")
    parser.add_argument("--years", type=int, default=YEARS, help="จำนวนปีย้อนหลัง (default: 3)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="จำนวน epoch สูงสุด")
    parser.add_argument("--mode", choices=["train-valid", "test", "full"], default="train-valid",
                        help="train-valid=จูนโดยไม่ประเมิน test, test=วัด checkpoint, full=ทำทั้งสอง")
    args = parser.parse_args()
    if args.years < 1 or args.epochs < 1:
        parser.error("--years และ --epochs ต้องมีค่าอย่างน้อย 1")
    if args.mode == "train-valid":
        result = run_train_valid(args.stock, years=args.years, epochs=args.epochs)
    elif args.mode == "test":
        result = run_test(args.stock, years=args.years)
    else:
        result = run_stock(args.stock, years=args.years, epochs=args.epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
