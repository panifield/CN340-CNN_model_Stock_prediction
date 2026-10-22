"""
run_c3.py — base config เทียบตัวเทียบทั้งหมด บนวันชุดเดียวกัน (Checkpoint C3)
===========================================================================
ไฟล์นี้ทำอะไร
    1. รัน base config (CNN→ANN, 3 seeds เฉลี่ย) ครบ 6 evaluation (2 หุ้น × 3 fold) แบบขนาน `--jobs N`
    2. รันตัวเทียบ: Naive · Mean Return · MLP / RF / XGB เดิม (29 feature, พารามิเตอร์เดิม) บน fold เดียวกัน
    3. รัน negative control ซ้ำด้วย 3 seeds (ครั้งก่อนใช้ 1 seed) เพื่อให้เทียบกับ base ได้ยุติธรรม
    4. สร้างตารางต่อ evaluation: relMAE หน่วย return และบาท · IC · StdRatio · DirAcc · epoch ที่หยุด
    5. วาด loss curve ของ train กับ early-stop ทั้ง 6 evaluation พร้อมระบุ epoch ที่หยุด
    6. เขียนทุกอย่างลง results/stage0_base/ และพิมพ์สรุปบนหน้าจอ

ทำไมต้องมี: ต้องรู้ว่า "base config ทำอะไรได้" ก่อนเริ่มทดลองปัจจัยอื่น — โดยเฉพาะ 3 เรื่อง
    (ก) มันชนะ Naive ไหม  (ข) แยกจาก negative control (โมเดลที่เรียนจาก noise ล้วน) ได้ไหม
    (ค) มัน "ทำนายค่าคงที่" (StdRatio < 0.05) หรือไม่ ซึ่งจะทำให้ผ่าน guard ไม่ได้ (ข้อ 4.2)

รัน:  python -m src.run_c3 --jobs 6
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import baselines, controls, data, evaluate, figures, paths, runner
from src.check_data import md_table
from src.configs import BASE
from src.settings import GUARD_MIN_STDRATIO, SEARCH_SEEDS, TICKERS, VAL_END

STAGE = paths.RESULTS_DIR / "stage0_base"
CNN_LABEL = "CNN→ANN (base)"
MODEL_ORDER = ["Naive", "Mean Return", "ANN (MLP)", "Random Forest", "XGBoost", CNN_LABEL]
EVALS = [(t, k) for t in TICKERS for k in (1, 2, 3)]           # K1 K2 K3 A1 A2 A3
EVAL_LABELS = [f"{t[0]}{k}" for t, k in EVALS]


# ---------------------------------------------------------------
# การรัน
# ---------------------------------------------------------------
def run_negative_3seeds(jobs, seeds):
    """negative control ซ้ำด้วยจำนวน seed เท่ากับ base (ครั้งก่อน 1 seed ที่ results/stage0_setup/controls.csv)"""
    tasks = [(BASE, t, k, tuple(seeds), VAL_END) for t, k in EVALS]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(controls.run_negative_unit, tasks))
    else:
        rows = [controls.run_negative_unit(a) for a in tasks]
    return pd.DataFrame(rows)


def run_everything(jobs, seeds):
    STAGE.mkdir(parents=True, exist_ok=True)
    timing = {}

    t0 = time.time()
    evals_cnn, scores, preds_cnn = runner.run_configs([BASE], STAGE, seeds=seeds, jobs=jobs)
    timing["base"] = time.time() - t0
    timing["base_serial_equivalent"] = float(evals_cnn["seconds"].sum())      # ผลรวมเวลาของ unit = เวลาถ้ารันทีละตัว
    with open(STAGE / "histories.json", encoding="utf-8") as f:
        histories = json.load(f)

    print("\n[C3] ตัวเทียบ (Naive · Mean · MLP · RF · XGB บน 29 feature เดิม)")
    t0 = time.time()
    evals_cmp, preds_cmp = baselines.run_all(verbose=True)
    timing["comparators"] = time.time() - t0

    print(f"\n[C3] negative control ซ้ำด้วย {len(seeds)} seeds")
    t0 = time.time()
    neg3 = run_negative_3seeds(jobs, seeds)
    timing["negative_3seeds"] = time.time() - t0
    return evals_cnn, scores, preds_cnn, histories, evals_cmp, preds_cmp, neg3, timing


# ---------------------------------------------------------------
# ตาราง
# ---------------------------------------------------------------
def fmt(v, kind):
    if pd.isna(v):
        return "—"
    return {"r4": f"{v:.4f}", "r3": f"{v:.3f}", "s3": f"{v:+.3f}", "b3": f"{v:.3f}"}[kind]


def all_models_evals(evals_cnn, evals_cmp):
    c = evals_cnn.copy()
    c["model"] = CNN_LABEL
    return pd.concat([evals_cmp, c], ignore_index=True)


def pivot_table(all_evals, metric, kind, with_summary=None):
    """แถว = โมเดล · คอลัมน์ = 6 evaluation + ค่าเฉลี่ย (+ จำนวนที่ชนะ Naive ถ้าเป็น relMAE)"""
    rows = []
    for m in MODEL_ORDER:
        g = all_evals[all_evals["model"] == m].set_index(["ticker", "fold"])
        vals = [g.loc[e, metric] for e in EVALS]
        row = {"โมเดล": m, **{lab: fmt(v, kind) for lab, v in zip(EVAL_LABELS, vals)}, "เฉลี่ย": fmt(pd.Series(vals, dtype=float).mean(), kind)}      # pandas ข้าม NaN ให้โดยไม่เตือน
        if with_summary == "beat":
            row["ชนะ Naive"] = f"{int(np.sum(np.array(vals) < 1))}/6"
        elif with_summary == "pos":
            row["IC>0"] = f"{int(np.sum(np.array(vals) > 0))}/6"
        rows.append(row)
    return pd.DataFrame(rows)


def base_detail_table(evals_cnn, neg1, neg3):
    """ตารางของ base แยกตาม 6 evaluation + เทียบ negative control (ข้อ 1 และ 3 ที่ผู้ใช้ขอ)"""
    g = evals_cnn.set_index(["ticker", "fold"])
    n1 = neg1.set_index(["ticker", "fold"])
    n3 = neg3.set_index(["ticker", "fold"])
    rows, d3 = [], []
    for (t, k), lab in zip(EVALS, EVAL_LABELS):
        r = g.loc[(t, k)]
        diff = r["relMAE"] - n3.loc[(t, k), "relMAE"]
        d3.append(diff)
        rows.append({"eval": f"{lab} ({t} fold {k})", "relMAE (return)": fmt(r["relMAE"], "r4"),
                     "relMAE (บาท)": fmt(r["relMAE_baht"], "r4"), "IC": fmt(r["IC"], "s3"),
                     "Spearman": fmt(r["IC_spearman"], "s3"), "StdRatio": fmt(r["StdRatio"], "r3"),
                     "DirAcc": fmt(r["DirAcc"], "r3"), "MAE (บาท)": fmt(r["MAE_baht"], "r4"),
                     "epoch หยุด/ดีสุด (เฉลี่ย 3 seed)": f"{r['epochs_mean']:.0f} / {r['best_epoch_mean']:.0f}",
                     "negative 1 seed": fmt(n1.loc[(t, k), "relMAE"], "r4"),
                     "negative 3 seeds": fmt(n3.loc[(t, k), "relMAE"], "r4"),
                     "base − neg(3)": f"{diff:+.4f}"})
    mean = {"eval": "**เฉลี่ย**", "relMAE (return)": fmt(evals_cnn["relMAE"].mean(), "r4"),
            "relMAE (บาท)": fmt(evals_cnn["relMAE_baht"].mean(), "r4"), "IC": fmt(evals_cnn["IC"].mean(), "s3"),
            "Spearman": fmt(evals_cnn["IC_spearman"].mean(), "s3"), "StdRatio": fmt(evals_cnn["StdRatio"].mean(), "r3"),
            "DirAcc": fmt(evals_cnn["DirAcc"].mean(), "r3"), "MAE (บาท)": fmt(evals_cnn["MAE_baht"].mean(), "r4"),
            "epoch หยุด/ดีสุด (เฉลี่ย 3 seed)": f"{evals_cnn['epochs_mean'].mean():.0f} / {evals_cnn['best_epoch_mean'].mean():.0f}",
            "negative 1 seed": fmt(neg1["relMAE"].mean(), "r4"), "negative 3 seeds": fmt(neg3["relMAE"].mean(), "r4"),
            "base − neg(3)": f"{np.mean(d3):+.4f}"}
    rows.append(mean)
    return pd.DataFrame(rows), d3


def summary_table(all_evals, preds_all):
    """สรุปต่อโมเดล + Diebold–Mariano เทียบ Naive (ตัววัดรอง ข้อ 6)"""
    naive = preds_all[preds_all["model"] == "Naive"]
    rows = []
    for m in MODEL_ORDER:
        g = all_evals[all_evals["model"] == m]
        sr = g["StdRatio"].to_numpy(dtype=float)
        row = {"โมเดล": m, "relMAE (return)": fmt(g["relMAE"].mean(), "r4"), "relMAE (บาท)": fmt(g["relMAE_baht"].mean(), "r4"),
               "MAE (บาท)": fmt(g["MAE_baht"].mean(), "r4"), "RMSE (บาท)": fmt(g["RMSE_baht"].mean(), "r4"),
               "ชนะ Naive": f"{int((g['relMAE'] < 1).sum())}/6", "IC": fmt(g["IC"].mean(), "s3"),
               "IC>0": f"{int((g['IC'] > 0).sum())}/6" if g["IC"].notna().any() else "—",
               "DirAcc": fmt(g["DirAcc"].mean(), "r3"), "StdRatio ต่ำสุด": fmt(np.nanmin(sr), "r3"),
               "StdRatio ≥ 0.05 ทุก eval": "ใช่" if np.isfinite(sr).all() and (sr >= GUARD_MIN_STDRATIO).all() else "ไม่"}
        if m == "Naive":
            row.update({"DM vs Naive": "—", "p (DM)": "—"})
        else:
            pa = preds_all[preds_all["model"] == m]
            dm = evaluate.dm_test(pa, naive)
            row.update({"DM vs Naive": f"{dm['DM']:+.2f}", "p (DM)": f"{dm['p_DM']:.3f}"})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6, help="จำนวน process ที่รันขนาน (ผู้ใช้สั่ง 6)")
    ap.add_argument("--seeds", type=int, default=len(SEARCH_SEEDS), help="จำนวน seed ที่เฉลี่ย (ค่าตั้งต้น 3 ตามข้อ 4.2)")
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    paths.assert_output_dir(STAGE)

    (evals_cnn, scores, preds_cnn, histories, evals_cmp, preds_cmp, neg3, timing) = run_everything(args.jobs, seeds)
    neg1 = pd.read_csv(paths.RESULTS_DIR / "stage0_setup" / "controls.csv")
    neg1 = neg1[neg1["control"] == "negative"]

    all_evals = all_models_evals(evals_cnn, evals_cmp)
    pc = preds_cnn.copy()
    pc["model"] = CNN_LABEL
    preds_all = pd.concat([preds_cmp, pc], ignore_index=True)

    all_evals.to_csv(STAGE / "c3_all_models_evals.csv", index=False)
    evals_cmp.to_csv(STAGE / "comparators_evals.csv", index=False)
    preds_cmp.to_csv(STAGE / "comparators_predictions.csv", index=False)
    neg3.to_csv(STAGE / "negative_control_3seeds.csv", index=False)
    fig_path = figures.plot_loss_curves(histories, BASE.name, name="c3_loss_curves_base.png")

    detail, d3 = base_detail_table(evals_cnn, neg1, neg3)
    summ = summary_table(all_evals, preds_all)

    md = ["# C3 — base config เทียบตัวเทียบ (development / walk-forward · โซน dev ≤ " + VAL_END + ")", "",
          f"base = CNN→ANN ตามสเปกข้อ 5.1 · เฉลี่ยคำทำนาย {len(seeds)} seeds · วันประเมินชุดเดียวกันทุกโมเดล "
          "(3 fold × 266 วัน ทั้งสองหุ้น) · **ไม่ใช่ผล test**", "",
          "ชื่อ evaluation: K = KBANK · A = ADVANC · เลข = fold (K1 = KBANK fold 1)", "",
          "## 1. base แยกตาม 6 evaluation และเทียบ negative control", "", md_table(detail), "",
          f"ชนะ negative control (base − neg(3) < 0) : {int(np.sum(np.array(d3) < 0))}/6 evaluation · "
          "relMAE < 1 คือชนะ Naive", "",
          "## 2. สรุปต่อโมเดล", "", md_table(summ), "",
          "DM = Diebold–Mariano (HAC) เทียบ Naive บนวันเดียวกัน · DM ลบ = ดีกว่า Naive · "
          f"ธง StdRatio ≥ {GUARD_MIN_STDRATIO} ทุก evaluation (หลัง B16 เป็นธงรายงาน ไม่ตัดสิทธิ์)", ""]
    for title, metric, kind, summ_kind in [("3. relMAE หน่วย return (คะแนนเลือก config — DECISIONS ข้อ B5)", "relMAE", "r4", "beat"),
                                           ("4. relMAE หน่วยบาท (รายงานคู่กัน)", "relMAE_baht", "r4", "beat"),
                                           ("5. IC = corr(ŷ, y)", "IC", "s3", "pos"),
                                           ("6. IC แบบ Spearman", "IC_spearman", "s3", "pos"),
                                           ("7. StdRatio = std(ŷ)/std(y)", "StdRatio", "r3", None),
                                           ("8. Direction accuracy (ไม่นับวันที่ y = 0)", "DirAcc", "r3", None),
                                           ("9. MAE หน่วยบาท", "MAE_baht", "r4", None)]:
        md += [f"## {title}", "", md_table(pivot_table(all_evals, metric, kind, summ_kind)), ""]
    md += ["## 10. loss curve และ epoch ที่หยุด", "",
           f"รูป: `{fig_path.relative_to(paths.TASK_DIR).as_posix()}` · เส้นประ = epoch ที่ val ดีที่สุด (state ที่ใช้ทำนาย) · "
           "early stopping ใช้ 15% ท้ายของ train ของ fold นั้น ไม่แตะช่วงประเมิน", "",
           md_table(pd.DataFrame([{"eval": f"{lab}", **{f"seed {i}": f"หยุด {h['epochs']} / ดีสุด {h['best_epoch']}"
                                                         for i, h in enumerate(histories[f"{BASE.name}|{t}|fold{k}"])}}
                                  for (t, k), lab in zip(EVALS, EVAL_LABELS)])), "",
           "## 11. เวลา", "",
           f"- base ({len(seeds)} seeds · 6 evaluation) แบบขนาน {args.jobs} process: **{timing['base'] / 60:.1f} นาที** "
           f"(ผลรวมเวลาของ unit = {timing['base_serial_equivalent'] / 60:.1f} นาที ≈ เวลาถ้ารันทีละตัว)",
           f"- ตัวเทียบ MLP/RF/XGB (รันทีละตัว): {timing['comparators'] / 60:.1f} นาที",
           f"- negative control {len(seeds)} seeds ({args.jobs} process): {timing['negative_3seeds'] / 60:.1f} นาที", ""]
    (STAGE / "c3_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- พิมพ์สรุปบนหน้าจอ ----
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100 + "\n1. base แยกตาม 6 evaluation + negative control\n" + "=" * 100)
    print(detail.to_string(index=False))
    print(f"\nชนะ negative control (3 seeds): {int(np.sum(np.array(d3) < 0))}/6 evaluation")
    print("\n" + "=" * 100 + "\n2. สรุปต่อโมเดล\n" + "=" * 100)
    print(summ.to_string(index=False))
    print("\n" + "=" * 100 + "\nเวลา\n" + "=" * 100)
    print(f"  base {args.jobs} process: {timing['base'] / 60:.1f} นาที (ผลรวม unit {timing['base_serial_equivalent'] / 60:.1f} นาที)")
    print(f"  ตัวเทียบ {timing['comparators'] / 60:.1f} นาที · negative 3 seeds {timing['negative_3seeds'] / 60:.1f} นาที")
    print(f"\nเขียน {STAGE}\\c3_summary.md และ {fig_path}")


if __name__ == "__main__":
    main()
