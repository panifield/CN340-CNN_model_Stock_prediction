"""
check_controls.py — โครงโมเดล + positive/negative control (Checkpoint C2)
========================================================================
ไฟล์นี้ทำอะไร
    1. พิมพ์ shape ทุกชั้นและจำนวนพารามิเตอร์ของทุกโครง พร้อม "ตรวจด้วยสูตรมือ" (SPEC ข้อ 5.0)
    2. รัน positive control : target = std ของ 5 return ล่าสุดในหน้าต่างของภาพเอง → ต้องได้ R² ≥ 0.3
    3. รัน negative control : สลับ y ใน train แบบสุ่ม → relMAE ต้อง ≥ 0.99
    4. วัดเวลาเทรนจริง เพื่อประเมินงบเวลาของ Stage 1–6 (SPEC ข้อ 8)
    5. เขียนผลลง results/stage0_setup/controls.csv และ control_summary.md

ทำไมต้องมี: ถ้า control ไม่ผ่าน ผลทุกตัวหลังจากนี้ตีความไม่ได้ · ต้องหยุดและแก้ก่อน

รัน:  python -m src.check_controls            (1 seed · เร็วพอสำหรับ sanity check)
      python -m src.check_controls --seeds 3  (3 seed เหมือนตอนค้นหา config จริง)
"""

import argparse
import time

import numpy as np
import pandas as pd

from src import controls, data, models, paths
from src.check_data import md_table
from src.configs import BASE, variant
from src.settings import TICKERS, VAL_END

ARCH_CONFIGS = [
    ("ก. 2D CNN (base)", BASE, 0),
    ("ก. 2D CNN + hybrid (29 feature)", variant("hybrid", ann_input="hybrid"), 29),
    ("ก. 2D CNN + GAP head", variant("gap", head="gap"), 0),
    # filter (8, 16) เท่ากับ base ทั้ง ข. และ ค. (DECISIONS ข้อ B13 · แก้สเปกข้อ 5.3 แล้ว)
    ("ข. 1D CNN ตามแกนเวลา", variant("cnn1d_time", arch="cnn1d_time"), 0),
    ("ค. 1D CNN บน return ดิบ", variant("cnn1d_raw", arch="cnn1d_raw"), 0),
]


def print_layer_reports(md):
    print("== โครงโมเดลและจำนวนพารามิเตอร์ ==")
    md += ["## โครงโมเดลและจำนวนพารามิเตอร์", ""]
    for title, cfg, n_extra in ARCH_CONFIGS:
        rows, total = models.layer_report(cfg, n_extra)
        print(f"\n  {title} · input {cfg.image_shape} · พารามิเตอร์รวม {total:,}")
        for r in rows:
            print(f"    {r['layer']:30s} {str(r['out_shape']):16s} {r['params']:>8,}")
        md += [f"### {title}", "", f"input `{cfg.image_shape}` · พารามิเตอร์รวม **{total:,}**", "",
               md_table(pd.DataFrame([{"ชั้น": r["layer"], "shape ที่ออก": str(r["out_shape"]),
                                       "พารามิเตอร์": f"{r['params']:,}"} for r in rows])), ""]

    # ตรวจด้วยสูตรของ CS231n L5 : out = (W − K + 2P)/S + 1 · params = Cin·K·K·Cout + Cout
    c, f, t = BASE.image_shape
    checks = [
        ("conv1 3×3 pad 1 ไม่เปลี่ยนขนาด", models.conv_output_size(f, 3, 1, 1), f),
        ("maxpool 2×2 ลดครึ่ง (แกนความถี่)", models.conv_output_size(f, 2, 0, 2), 4),
        ("maxpool 2×2 ลดครึ่ง (แกนเวลา)", models.conv_output_size(t, 2, 0, 2), 32),
        ("พารามิเตอร์ conv1 = 1·3·3·8 + 8", models.conv_param_count(1, 3, 8), 80),
        ("พารามิเตอร์ conv2 = 8·3·3·16 + 16", models.conv_param_count(8, 3, 16), 1168),
        ("ขนาดหลัง flatten = 16·2·16", 16 * 2 * 16, 512),
        ("พารามิเตอร์ Linear(512→16) = 512·16 + 16", 512 * 16 + 16, 8208),
    ]
    print("\n  ตรวจด้วยสูตรมือ (CS231n L5):")
    md += ["### ตรวจด้วยสูตรมือ (CS231n L5)", "",
           "`out = (W − K + 2P)/S + 1` · `params = Cin·K·K·Cout + Cout`", "",
           "| รายการ | สูตรให้ | ค่าจริง | ตรงกัน |", "|---|---|---|---|"]
    for label, got, want in checks:
        ok = got == want
        print(f"    {'OK ' if ok else 'ผิด'} {label:42s} {got} (คาด {want})")
        md.append(f"| {label} | {got} | {want} | {'ใช่' if ok else '**ไม่**'} |")
        assert ok, f"สูตรไม่ตรงกับขนาดจริง: {label}"
    md.append("")
    return md


def run_controls(seeds, md):
    out_rows, timings = [], []
    ds_all = data.build_all(VAL_END)
    folds = data.make_folds(ds_all[TICKERS[0]].dates)

    for kind, runner in (("positive", controls.run_positive), ("negative", controls.run_negative)):
        print(f"\n== {kind} control ({len(seeds)} seed ต่อ evaluation) ==")
        rows = []
        for t in TICKERS:
            for fold in folds:
                t0 = time.time()
                r = runner(ds_all[t], BASE, fold, seeds)
                dt = time.time() - t0
                timings.append({"control": kind, "ticker": t, "fold": fold.k, "seconds": dt})
                rows.append(r)
                shown = {k: (f"{v:.4f}" if isinstance(v, float) else v)
                         for k, v in r.items() if k not in ("control", "ticker", "fold", "n")}
                print(f"  {t:6s} fold {fold.k}: {shown}  ({dt:.0f}s)")
        v = controls.verdict(rows, kind)
        print(f"  → {kind}: ผ่าน {v['n_passed']}/{v['n_eval']} · เฉลี่ย {v['mean']:.4f} "
              f"· แย่สุด {v['min']:.4f} · เกณฑ์ {v['threshold']} → {'ผ่าน' if v['passed'] else 'ไม่ผ่าน'}")
        out_rows += rows
        md += [f"## {kind} control", "", md_table(pd.DataFrame(rows).round(4)), "",
               f"**{'ผ่าน' if v['passed'] else 'ไม่ผ่าน'}** — ผ่าน {v['n_passed']}/{v['n_eval']} evaluation · "
               f"ค่าเฉลี่ย {v['mean']:.4f} · แย่สุด {v['min']:.4f} · เกณฑ์ {v['threshold']}", ""]
    return out_rows, timings, md


def report_timing(timings, seeds, md):
    tdf = pd.DataFrame(timings)
    per_eval = tdf["seconds"].mean()
    one_config = per_eval * 6 / len(seeds) * 3          # 6 evaluation × 3 seeds ตามสเปกข้อ 4.2
    print(f"\n== เวลา ==")
    print(f"  เฉลี่ย {per_eval:.0f}s ต่อ evaluation ({len(seeds)} seed) "
          f"→ 1 config เต็ม (6 evaluation × 3 seeds) ≈ {one_config / 60:.0f} นาที")
    print(f"  รวมเวลาที่ใช้ใน control ทั้งหมด {tdf['seconds'].sum() / 60:.0f} นาที")
    md += ["## เวลาที่ใช้ (SPEC ข้อ 8)", "",
           f"เฉลี่ย {per_eval:.0f} วินาทีต่อ evaluation ที่ {len(seeds)} seed → "
           f"**1 config เต็ม (6 evaluation × 3 seeds) ≈ {one_config / 60:.0f} นาที** (เครื่องผู้ใช้ · CPU 1 thread)", "",
           md_table(tdf.round(1)), ""]
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="จำนวน seed ต่อ evaluation (ค่าตั้งต้น 1)")
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    out = paths.assert_output_dir(paths.RESULTS_DIR / "stage0_setup")
    out.mkdir(parents=True, exist_ok=True)
    md = ["# C2 — โครงโมเดลและ control", "",
          f"base config · seeds = {seeds} · โซน dev ≤ {VAL_END}", ""]
    md = print_layer_reports(md)
    rows, timings, md = run_controls(seeds, md)
    md = report_timing(timings, seeds, md)

    pd.DataFrame(rows).to_csv(out / "controls.csv", index=False)
    (out / "control_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nเขียน {out / 'controls.csv'} และ {out / 'control_summary.md'}")


if __name__ == "__main__":
    main()
