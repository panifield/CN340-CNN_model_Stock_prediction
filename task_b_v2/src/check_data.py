"""
check_data.py — สรุป dataset, fold, หลักฐานเรื่องขอบหน้าต่าง และวาดภาพตัวอย่าง (Checkpoint C1)
=========================================================================================
ไฟล์นี้ทำอะไร
    1. สร้าง dataset ของทั้งสองหุ้นในโซน dev (≤ 2025-02-25) แล้วพิมพ์ขนาดทุกชิ้น
    2. พิมพ์ตาราง fold (3 fold × 266 วัน) ว่าช่วง train/eval ของแต่ละ fold เป็นวันที่อะไรถึงอะไร
    3. พิมพ์ขนาดภาพของทุก config ที่จะใช้ใน Stage 1 (เทียบกับสูตร) + หลักฐานว่าทำไมต้องเติมขอบหน้าต่าง
    4. วาดภาพตัวอย่าง 3 วัน และภาพเทียบ "ราคา vs return" ลง results/figures/
    5. เขียนสรุปลง results/stage0_setup/dataset_summary.md และ folds.csv

ทำไมต้องมี: ให้ผู้ใช้ตรวจตัวเลขทั้งหมดก่อนเริ่มเทรน (ถ้าโครงข้อมูลผิด ผลทุกตัวหลังจากนี้ไม่มีความหมาย)
รัน:  python -m src.check_data
"""

import numpy as np
import pandas as pd

from src import data, figures, paths
from src.configs import BASE, variant
from src.settings import EARLY_STOP_FRAC, TICKERS, VAL_END
from src.spectrogram import hann_tail_weights, window_weights

# config ของ Stage 1 ที่ต้องรายงานขนาดภาพ (ยังไม่เทรน -- แค่ตรวจขนาด)
STAGE1_CONFIGS = [
    BASE,
    variant("win64", window=64),
    variant("win256", window=256, nperseg=32, hop=4),
    variant("complex", spec_type="complex"),
    variant("ch3", channels="ret3"),
    variant("ch4", channels="ret4"),
    variant("detrend", channels="detrend"),
    variant("raw1d", arch="cnn1d_raw"),
]


def _compression(raw, fn, s):
    """คืน (corr กับค่าก่อนแปลง, p99/median ของ |ค่าหลังแปลง|) — ใช้วัดว่าตัวบีบอัดทำงานจริงไหม"""
    v = fn(raw / s)
    av = np.abs(v)
    return float(np.corrcoef(v, raw)[0, 1]), float(np.percentile(av, 99) / np.median(av))


def compression_report(ds, every=10):
    """ตารางเปรียบเทียบตัวบีบอัดของ logpower และ complex บนค่าดิบจริงของหุ้นตัวนั้น (ทุก `every` วันในโซน dev)"""
    rows = []
    for spec_type in ("logpower", "complex"):
        cfg = variant(f"cmp_{spec_type}", spec_type=spec_type)
        raw, has = ds.raw_images(cfg, np.flatnonzero(ds.usable(cfg))[::every])
        raw = raw[has].ravel().astype(np.float64)
        med = float(np.median(np.abs(raw)))
        if spec_type == "logpower":
            cand = [("logpower  log1p(|Z|²)  [สเปกเดิม]", np.log1p, 1.0),
                    ("logpower  log1p(|Z|²·1e5) [งานเดิม]", np.log1p, 1e-5),
                    ("logpower  log1p(|Z|²/m)  [ที่ใช้]", np.log1p, med)]
        else:
            cand = [("complex   arcsinh(z)     [ไม่สเกล]", np.arcsinh, 1.0),
                    ("complex   arcsinh(z·1e3) [ที่ใช้]", np.arcsinh, 1e-3),
                    ("complex   arcsinh(z/m)   [ทางเลือก]", np.arcsinh, med)]
        for label, fn, s in cand:
            r, ratio = _compression(raw, fn, s)
            rows.append({"ชนิด": label, "s": s, "corr": r, "p99/median": ratio})
    return rows


def md_table(df):
    """DataFrame → ตาราง markdown (เขียนเองเพื่อไม่ต้องลงไลบรารี tabulate เพิ่ม)"""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def main():
    out = paths.assert_output_dir(paths.RESULTS_DIR / "stage0_setup")
    out.mkdir(parents=True, exist_ok=True)
    md = ["# C1 — dataset, fold และภาพตัวอย่าง", "",
          f"โซน dev: ต้นข้อมูล → **{VAL_END}** (ตัดก่อนสร้าง feature/ภาพทุกครั้ง)", ""]

    print("== dataset (โซน dev) ==")
    ds_all = data.build_all(VAL_END)
    md += ["## ขนาดของ dataset", "", "| หุ้น | แถวในโซน | ช่วงวันที่ | แถวอุ่นเครื่อง (29 feature) | แถวใช้ได้ (base) |",
           "|---|---|---|---|---|"]
    for t in TICKERS:
        ds = ds_all[t]
        u = ds.usable(BASE)
        print(f"  {t}: {len(ds)} แถว {ds.dates[0].date()} → {ds.dates[-1].date()} · feat29 ครบ {int(ds.ok29.sum())} "
              f"· ใช้ได้กับ base config {int(u.sum())} แถว (ตัดเพราะหน้าต่าง {BASE.window} วัน)")
        print(f"      y shape {ds.y.shape} · feat29 shape {ds.feat29.shape} · prices shape {ds.prices.shape}")
        md.append(f"| {t} | {len(ds)} | {ds.dates[0].date()} → {ds.dates[-1].date()} | "
                  f"{int((~ds.ok29).sum())} | {int(u.sum())} |")

    print("\n== รายชื่อ 29 feature เดิม (จาก legacy/features.py) ==")
    cols = list(ds_all[TICKERS[0]].feat29.columns)
    for i in range(0, len(cols), 4):
        print("   " + ", ".join(cols[i:i + 4]))
    md += ["", f"29 feature เดิม: `{'`, `'.join(cols)}`", ""]

    print("\n== fold (walk-forward 3 fold × 266 วัน · จบที่ 2025-02-25) ==")
    folds = data.make_folds(ds_all[TICKERS[0]].dates)
    tables = [data.fold_summary(ds_all[t], folds, BASE) for t in TICKERS]
    fold_df = pd.concat(tables, ignore_index=True)
    print(fold_df.to_string(index=False))
    n_fit = len(folds[0].train_pos(ds_all[TICKERS[0]].usable(BASE)))
    print(f"  early stopping: ใช้ {EARLY_STOP_FRAC:.0%} ท้ายของ train (fold 1 ของ {TICKERS[0]}: "
          f"fit {n_fit - max(1, round(n_fit * EARLY_STOP_FRAC))} / val {max(1, round(n_fit * EARLY_STOP_FRAC))} แถว)")
    fold_df.to_csv(out / "folds.csv", index=False)
    md += ["## fold (base config)", "", md_table(fold_df), "",
           f"early stopping ใช้ {EARLY_STOP_FRAC:.0%} ท้ายของ train ใน fold นั้น (ไม่แตะช่วงประเมิน)", ""]

    print("\n== ขนาดภาพของแต่ละ config (Stage 1) ==")
    md += ["## ขนาดภาพของแต่ละ config", "", "| config | ช่อง | window/nperseg/hop | ชนิด | ขนาดภาพ (C×F×T) | จุดข้อมูลต่อภาพ |",
           "|---|---|---|---|---|---|"]
    ds0 = ds_all[TICKERS[0]]
    for cfg in STAGE1_CONFIGS:
        shape = cfg.image_shape
        X, has = ds0.raw_images(cfg, [len(ds0) - 1])
        assert has[0] and X.shape[1:] == shape and np.isfinite(X[0]).all()
        print(f"  {cfg.name:9s} ช่อง {cfg.channels:8s} {cfg.window:3d}/{cfg.nperseg:2d}/{cfg.hop} "
              f"{cfg.spec_type:9s} → {shape}  = {int(np.prod(shape)):5d} จุด")
        md.append(f"| {cfg.name} | {cfg.channels} ({cfg.n_raw_channels}) | "
                  f"{cfg.window}/{cfg.nperseg}/{cfg.hop} | {cfg.spec_type} | {'×'.join(map(str, shape))} | "
                  f"{int(np.prod(shape))} |")

    print(f"\n== ทำไมต้องเติมขอบหน้าต่าง (boundary='even') ==")
    print("  วัดด้วยการยิง impulse ที่แต่ละวันในหน้าต่าง แล้วดูขนาดของผลที่ออกมาในภาพ (STFT เป็นเชิงเส้น)")
    md += ["## ทำไมต้องเติมขอบหน้าต่าง (`boundary=\"even\"`)", "",
           "วัดด้วยการยิง impulse ที่แต่ละวันในหน้าต่างแล้วดูขนาดของผลในภาพ (STFT เป็นตัวดำเนินการเชิงเส้น "
           "ค่านี้จึงเป็นอิทธิพลของวันนั้นต่อภาพทั้งภาพ · normalize ให้วันที่ถูกใช้มากสุด = 1.000)", "",
           "| boundary | เมื่อวาน (t−1) | t−2 | t−3 | วันกลางหน้าต่าง |", "|---|---|---|---|---|"]
    for boundary in ("even", None):
        w = window_weights(BASE, boundary)
        print(f"  boundary={str(boundary):6s}: น้ำหนักของเมื่อวาน (t−1) = {w[-1]:.4f} · t−2 = {w[-2]:.4f} "
              f"· t−3 = {w[-3]:.4f} · วันกลางหน้าต่าง = {w[len(w) // 2]:.4f}")
        md.append(f"| `{boundary}` | {w[-1]:.4f} | {w[-2]:.4f} | {w[-3]:.4f} | {w[len(w) // 2]:.4f} |")
    w_last, w_prev = hann_tail_weights(BASE.nperseg)
    print(f"  (ตัวเลขดิบ: หน้าต่าง Hann ยาว {BASE.nperseg} ให้น้ำหนักตัวอย่างท้ายสุดของ segment {w_last:.4f} "
          f"· ตัวก่อนหน้า {w_prev:.4f} ของค่าสูงสุด)")
    print("  → ถ้าไม่เติมขอบ ข้อมูลของ 'เมื่อวาน' (วันที่สำคัญที่สุด) ถูกใช้น้อยกว่าเติมขอบ ~6.5 เท่า")
    md += ["", f"ตัวเลขดิบของหน้าต่าง Hann ยาว {BASE.nperseg}: ตัวอย่างท้ายสุดของแต่ละ segment ได้น้ำหนัก "
           f"{w_last:.4f} และตัวก่อนหน้า {w_prev:.4f} ของค่าสูงสุด · "
           "ที่ตารางข้างบนยังได้ 0.11 ไม่ใช่ 0.04 เพราะ `detrend=\"constant\"` หักค่าเฉลี่ยของ segment ออก "
           "ทำให้ค่าของวันนั้นมีผลผ่านค่าเฉลี่ยด้วย", "",
           "สรุป: ถ้าไม่เติมขอบ ข้อมูลของเมื่อวาน (วันที่สำคัญที่สุดสำหรับการทำนายวันพรุ่งนี้) ถูกใช้น้อยกว่าเติมขอบประมาณ 6.5 เท่า · "
           "การสะท้อนขอบใช้แค่ตัวเลขที่อยู่ในหน้าต่าง (อดีตทั้งหมด) จึงไม่ leak", ""]

    print("\n== ตัวบีบอัดช่วงค่าบีบจริงแค่ไหน (DECISIONS ข้อ B9, B10) ==")
    md += ["## ตัวบีบอัดช่วงค่าบีบจริงแค่ไหน", "",
           "`corr` = สหสัมพันธ์ระหว่างค่าหลังแปลงกับค่าก่อนแปลง (1.000000 = การแปลงเป็นเชิงเส้น คือไม่ได้บีบอัดเลย) · "
           "`p99/median` = ความยาวหางของการแจกแจงหลังแปลง (เล็กลง = บีบอัดได้ผล)", ""]
    for t in TICKERS:
        md += [f"### {t}", "", "| ชนิด | สเกล s | corr | p99/median |", "|---|---|---|---|"]
        print(f"  -- {t} --")
        for row in compression_report(ds_all[t]):
            print(f"    {row['ชนิด']:28s} s = {row['s']:>10.3e} · corr = {row['corr']:.6f} "
                  f"· p99/median = {row['p99/median']:6.2f}")
            md.append(f"| {row['ชนิด']} | {row['s']:.3e} | {row['corr']:.6f} | {row['p99/median']:.2f} |")
        md.append("")
    md += ["**สรุป** — `logpower` : สูตรเดิมของสเปกข้อ 5.1 (`log1p(|Z|²)`, s = 1) ได้ corr = 1.000000 พอดี "
           "คือ log1p ไม่ได้บีบอัดอะไรเลย เพราะ |Z|² จริงอยู่ราว 1e-5 ซึ่ง `log1p(x) ≈ x` · "
           "แก้เป็น `log1p(|Z|²/m)` โดย m = median ของส่วน fit ใน train ของ fold นั้น (แยกต่อหุ้น) แล้วบีบอัดได้จริง", "",
           "`complex` : `arcsinh(z·1e3)` ที่ port มาบีบอัดอยู่แล้ว (corr ≈ 0.92–0.95, หางสั้นลงจาก ~7 เหลือ ~2.5) "
           "จึงไม่ต้องแก้ · ค่าคงที่ 1e3 บังเอิญใกล้กับ 1/median ที่ fit ได้ (ต่างกันไม่ถึง 1.7 เท่า)", ""]

    print("\n== รูป ==")
    figs = []
    for t in TICKERS:
        figs.append(figures.plot_sample_spectrograms(ds_all[t], BASE))
    figs.append(figures.plot_price_vs_return_channel(ds_all[TICKERS[0]], BASE))
    for f in figs:
        print(f"  เขียน {f}")
    md += ["## รูป", ""] + [f"- `{f.relative_to(paths.TASK_DIR).as_posix()}`" for f in figs] + [""]

    (out / "dataset_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nเขียน {out / 'dataset_summary.md'} และ {out / 'folds.csv'}")


if __name__ == "__main__":
    main()
