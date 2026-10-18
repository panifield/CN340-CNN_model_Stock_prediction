"""
สร้างเฉพาะภาพ spectrogram (ไม่เทรนโมเดล)

Usage:
    python A/show_spectrograms.py
    python A/show_spectrograms.py --stock ADVANC --no-per-day
"""
import argparse

from src.config import OUT_DIR, STOCKS
from src.data import build_dataset, load_stock
from src.visualize import make_spectrogram_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", choices=STOCKS + ["ALL"], default="ALL")
    ap.add_argument("--no-per-day", action="store_true", help="ไม่บันทึกภาพ spectrogram รายวัน")
    args = ap.parse_args()

    for stock in (STOCKS if args.stock == "ALL" else [args.stock]):
        df = load_stock(stock)
        X, y, dates = build_dataset(df)
        fig_dir = OUT_DIR / stock / "figures"
        make_spectrogram_figures(df, X, y, dates, stock, fig_dir, per_day=not args.no_per_day)
        print(f"{stock}: saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
