"""
run_stage2.py — Stage 2 โครงสร้าง CNN: เปลี่ยนจาก base ทีละปัจจัย เลือกผู้ชนะด้วย IC + เกณฑ์ B22 (Checkpoint C4 ภาคต่อ)
=======================================================================================================================
ไฟล์นี้ทำอะไร (SPEC ข้อ 5.3 + DECISIONS B13, B16, B17, B21, B22, B27)
    base ของ Stage 2 = **base** (Stage 1 ใช้ base ต่อตาม B21)
    config:
        base         ก. 2D CNN · หัว Flatten                      (จุดอ้างอิง)
        cnn1d_time   ข. 1D CNN ตามแกนเวลา · filter (8, 16)         (ปัจจัย arch · ผู้สมัคร)   B13
        gap          2D CNN · หัว Global Average Pooling            (ปัจจัย head · ผู้สมัคร)
        cnn1d_raw    ค. 1D CNN บน return ดิบ (ไม่ทำ STFT)           (ตัวเทียบ "spectrogram ช่วยไหม" · ไม่ร่วมเป็นผู้ชนะ)

    กติกาผู้ชนะ (B22): ΔIC เทียบ base > **max(noise ของ stage, เกณฑ์ B22 = ค่าสูงสุดของ ΔIC ใน placebo)** — ต้องผ่านทั้งสอง
    ไม่มีผู้ชนะ → คงค่า base · ถ้าทั้งสองปัจจัยมีผู้ชนะ จะรันตัวรวมแล้วเทียบ base แบบ B14 (แพ้ → หยุดถามผู้ใช้)

*** สคริปต์นี้ "แนะนำ" ผู้ชนะเท่านั้น และ **หยุดที่ C4** ไม่ไป Stage 3 เอง (ผู้ใช้สั่ง) ***

รัน:  python -m src.run_stage2 --jobs 6
      python -m src.run_stage2 --smoke --jobs 6   (epoch น้อยมาก เขียนไป results/_smoke_stage2/ ตรวจสายงานเท่านั้น)
"""

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src import models, paths, runner, selection
from src.check_data import md_table
from src.configs import BASE, variant
from src.run_stage1 import check_matches_c3, fmt, pivot_by_config
from src.settings import SEARCH_SEEDS, VAL_END

# ปัจจัย → {ชื่อ config ผู้สมัคร: สิ่งที่เปลี่ยนจาก BASE}
FACTORS = {
    "arch": {"cnn1d_time": dict(arch="cnn1d_time")},          # filter (8, 16) เป็นค่า base อยู่แล้ว (B13)
    "head": {"gap": dict(head="gap")},
}
COMPARATORS = {"cnn1d_raw": dict(arch="cnn1d_raw")}            # ตัวเทียบ ไม่ร่วมเป็นผู้ชนะ (B27)

STAGE = paths.RESULTS_DIR / "stage2_structure"
COMBINED = paths.RESULTS_DIR / "stage2_structure_combined"
B22_FILE = paths.RESULTS_DIR / "stage1_placebo" / "stage1_final_decision.json"


def all_configs(shrink=None):
    cfgs = [BASE]
    for opts in FACTORS.values():
        cfgs += [variant(n, **ch) for n, ch in opts.items()]
    cfgs += [variant(n, **ch) for n, ch in COMPARATORS.items()]
    return [replace(c, **shrink) for c in cfgs] if shrink else cfgs


def read_b22_threshold(path=B22_FILE):
    """เกณฑ์ B22 จากไฟล์ผลตัดสินของ placebo (ไม่ hardcode — ค่าต้องมาจากผลที่ประกาศไว้)"""
    d = json.loads(path.read_text(encoding="utf-8"))
    return float(d["b22_threshold_stage2_to_5"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=len(SEARCH_SEEDS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    global STAGE, COMBINED
    shrink = None
    if args.smoke:
        STAGE = paths.RESULTS_DIR / "_smoke_stage2" / "stage2_structure"
        COMBINED = paths.RESULTS_DIR / "_smoke_stage2" / "stage2_structure_combined"
        runner.EXPERIMENT_LOG = paths.RESULTS_DIR / "_smoke_stage2" / "experiment_log.csv"
        shrink = dict(max_epochs=3, patience=3)
        print("[smoke] ตัวเลขไม่มีความหมาย — ตรวจแค่ว่าสายงานทำงานครบ")
    paths.assert_output_dir(STAGE)
    t_start = time.time()
    floor = read_b22_threshold()

    # ---------------- 1. รัน 4 config ----------------
    configs = all_configs(shrink)
    evals, scores, preds = runner.run_configs(configs, STAGE, seeds=seeds, jobs=args.jobs)
    selection.assert_scores_have_seed_noise(scores)
    noise = selection.pooled_seed_noise(scores)
    se_ref = selection.sampling_se_reference()
    print(f"\n[stage2] noise ของ seed (pooled 4 config) = {noise:.4f} · เกณฑ์ B22 = {floor:+.4f} → เกณฑ์ที่ใช้ = {max(noise, floor):+.4f}")

    # ---------------- 2. ผู้ชนะต่อปัจจัย ----------------
    winners, cmp_tables, changes = {}, [], {}
    for factor, options in FACTORS.items():
        w, tbl = selection.pick_factor_winner(scores, "base", list(options), noise, floor=floor)
        tbl.insert(0, "factor", factor)
        winners[factor] = w
        cmp_tables.append(tbl)
        if w != "base":
            changes.update(options[w])
    comp_tbl = selection.compare_to_base(scores, "base", list(COMPARATORS), noise, floor=floor)

    # ---------------- 3. ตัวรวม (เฉพาะเมื่อทั้งสองปัจจัยมีผู้ชนะ) ----------------
    both_win = all(w != "base" for w in winners.values())
    scores_all, evals_all = scores, evals
    stop_and_ask, d_ic = False, float("nan")
    if both_win:
        comb_cfg = variant("combined", **changes)
        if shrink:
            comb_cfg = replace(comb_cfg, **shrink)
        paths.assert_output_dir(COMBINED)
        print(f"\n[stage2] ทั้งสองปัจจัยมีผู้ชนะ → รันตัวรวม {changes}")
        ev_c, sc_c, _ = runner.run_configs([comb_cfg], COMBINED, seeds=seeds, jobs=args.jobs)
        scores_all = pd.concat([scores, sc_c], ignore_index=True)
        evals_all = pd.concat([evals, ev_c], ignore_index=True)
        stop_and_ask, d_ic = selection.combined_loses_to_base(scores_all, "combined", "base")
        combined_note = f"ตัวรวมมี ΔIC {d_ic:+.4f} เทียบ base → " + ("**แพ้ base — ต้องหยุดถามผู้ใช้**" if stop_and_ask else "ไม่แพ้ base")
    else:
        combined_note = ("ไม่ต้องรันตัวรวม — มีผู้ชนะไม่ครบทั้งสองปัจจัย (" +
                         ", ".join(f"{f} → {w}" for f, w in winners.items()) + ")")
    print(f"[stage2] {combined_note}")

    # ---------------- 4. ตรวจการรันซ้ำ ----------------
    same, same_note = check_matches_c3(preds)
    print(f"[stage2] base ใน Stage 2 เทียบกับ C3: {'ตรงกันทุกบิต' if same else ('ไม่ตรง' if same is False else '—')} ({same_note})")

    # ---------------- 5. รายงาน ----------------
    order = ["base"] + [n for o in FACTORS.values() for n in o] + list(COMPARATORS) + (["combined"] if both_win else [])
    sc = scores_all.set_index("config")
    cands = {n for o in FACTORS.values() for n in o}
    cmp_all = pd.concat(cmp_tables + [comp_tbl]).set_index("config")
    n_log = len(pd.read_csv(runner.EXPERIMENT_LOG)) if runner.EXPERIMENT_LOG.exists() else 0
    params = {c.name: models.layer_report(c)[1] for c in configs}

    def role(n):
        return next((f for f, o in FACTORS.items() if n in o), "ตัวเทียบ" if n in COMPARATORS else ("รวม" if n == "combined" else "base"))

    sel_rows, rep_rows = [], []
    for n in order:
        r = sc.loc[n]
        verdict = ""
        if n in cands:
            verdict = "**ชนะ**" if bool(cmp_all.loc[n, "beats_base"]) else "ไม่ผ่านเกณฑ์"
        sel_rows.append({"config": n, "บทบาท": role(n), "พารามิเตอร์": f"{params[n]:,}" if n in params else "—",
                         "mean IC": fmt(float(r["mean_IC"]), "s4"),
                         "ΔIC vs base": "—" if n == "base" else fmt(float(r["mean_IC"] - sc.loc["base", "mean_IC"]), "s4"),
                         "σ seed": fmt(float(r["ic_seed_std"]), "r4"), "IC ราย seed": r["ic_by_seed"].replace(";", " / "),
                         "Spearman": fmt(float(r["mean_IC_spearman"]), "s4"), "IC>0": f"{int(r['n_evals_ic_positive'])}/6", "ผล": verdict})
        rep_rows.append({"config": n, "relMAE (return)": fmt(float(r["mean_relMAE"]), "r4"), "relMAE (บาท)": fmt(float(r["mean_relMAE_baht"]), "r4"),
                         "MAE (บาท)": fmt(float(r["mean_MAE_baht"]), "r4"), "ชนะ Naive": f"{int(r['n_evals_beat_naive'])}/6",
                         "DirAcc": fmt(float(r["mean_DirAcc"]), "r3"), "StdRatio เฉลี่ย": fmt(float(r["mean_StdRatio"]), "r3"),
                         "StdRatio ต่ำสุด": fmt(float(r["min_StdRatio"]), "r3"),
                         "ธง StdRatio ≥ 0.05": "ผ่าน" if bool(r["stdratio_ok"]) else "ต่ำกว่า"})
    win_rows = [{"ปัจจัย": f, "ผู้ชนะ (แนะนำ)": w,
                 "เหตุผล": ("ΔIC เกินทั้ง noise และเกณฑ์ B22" if w != "base" else "ไม่มีตัวเลือกใดเกิน max(noise, B22) → คงค่า base")}
                for f, w in winners.items()]
    raw_d = float(sc.loc["cnn1d_raw", "mean_IC"] - sc.loc["base", "mean_IC"])
    raw_note = (f"ตัวเทียบ `cnn1d_raw` (ไม่ใช้ spectrogram): IC {fmt(float(sc.loc['cnn1d_raw', 'mean_IC']), 's4')} "
                f"(ต่างจาก base {fmt(raw_d, 's4')}) · relMAE {fmt(float(sc.loc['cnn1d_raw', 'mean_relMAE']), 'r4')} "
                f"เทียบ base {fmt(float(sc.loc['base', 'mean_relMAE']), 'r4')} — "
                + ("เกินเกณฑ์ที่ใช้" if raw_d > max(noise, floor) else "ไม่เกินเกณฑ์ (ต่างจาก base ภายใน noise/เกณฑ์ B22)")
                + " · รายงานเท่านั้น ไม่ร่วมเป็นผู้ชนะ (B27)")
    elapsed = time.time() - t_start

    md = ["# C4 (Stage 2) — โครงสร้าง CNN · development / walk-forward (โซน dev ≤ " + VAL_END + ")", "",
          ("> **ผลของโหมด --smoke — ตัวเลขไม่มีความหมาย**" if args.smoke else
           "> ผลทุกตัวเป็น development / walk-forward ไม่ใช่ test · **หยุดที่ C4** ตามคำสั่ง ไม่ไป Stage 3"), "",
          "## กติกาที่ใช้ (DECISIONS B16, B17, B22, B27)", "",
          "- base ของ Stage 2 = base (Stage 1 ใช้ base ต่อตาม B21)",
          f"- เลือกด้วย mean IC (Pearson) ของ 6 evaluation · ผู้ชนะต้องมี ΔIC > **max(noise, เกณฑ์ B22)** = max({noise:.4f}, {floor:.4f}) = **{max(noise, floor):.4f}**",
          f"- noise ของ seed (pooled ข้าม {len(configs)} config) = {noise:.4f} · เกณฑ์ B22 = {floor:+.4f} (ผู้ใช้ยืนยัน · conservative สำหรับ Stage ที่มีผู้สมัครน้อยกว่า 7 — Stage นี้มี 2) · "
          f"(อ้างอิง noise จากการสุ่มวัน ≈ {se_ref:.4f})",
          "- relMAE (return และบาท) รายงานทุกตาราง · StdRatio เป็นธงรายงาน ไม่ตัดสิทธิ์", "",
          "## 1. ตารางการเลือก", "", md_table(pd.DataFrame(sel_rows)), "",
          "## 2. ผู้ชนะที่แนะนำของแต่ละปัจจัย", "", md_table(pd.DataFrame(win_rows)), "",
          raw_note, "", "ตัวรวม: " + combined_note, "",
          "## 3. ตัววัดหลักที่รายงาน (relMAE หน่วย return และบาท) และธง StdRatio", "", md_table(pd.DataFrame(rep_rows)), "",
          "## 4. IC (Pearson) แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "IC", "s3", order)), "",
          "## 5. IC แบบ Spearman แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "IC_spearman", "s3", order)), "",
          "## 6. relMAE หน่วย return แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "relMAE", "r4", order)), "",
          "## 7. relMAE หน่วยบาท แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "relMAE_baht", "r4", order)), "",
          "## 8. StdRatio แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "StdRatio", "r3", order)), "",
          "## 9. Direction accuracy แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "DirAcc", "r3", order)), "",
          "## 10. ตรวจการรันซ้ำและการนับ config", "",
          f"- base ใน Stage 2 เทียบกับ base ใน C3: **{'ตรงกันทุกบิต' if same else ('ไม่ตรง' if same is False else '—')}** ({same_note})",
          f"- จำนวน config ที่ต่างกันที่ลองแล้วใน experiment_log.csv: **{n_log}**",
          f"- เวลา: {elapsed / 60:.1f} นาที (jobs {args.jobs})", ""]
    (STAGE / "stage2_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    decision = {"status": "แนะนำเท่านั้น — หยุดที่ C4 รอผู้ใช้ยืนยัน", "smoke": bool(args.smoke), "base": "base",
                "noise_pooled": noise, "b22_floor": floor, "threshold_used": max(noise, floor), "winners": winners,
                "combined_changes": changes if both_win else {}, "combined_run": bool(both_win),
                "combined_loses_to_base": bool(stop_and_ask), "base_matches_c3": same, "n_configs_logged": n_log,
                "comparator_cnn1d_raw_delta_ic": raw_d, "seeds": list(seeds)}
    (STAGE / "stage2_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 110 + "\nตารางการเลือก (mean IC · เทียบเกณฑ์ max(noise, B22))\n" + "=" * 110)
    print(pd.DataFrame(sel_rows).to_string(index=False))
    print("\n" + "=" * 110 + "\nตัววัดหลักที่รายงาน\n" + "=" * 110)
    print(pd.DataFrame(rep_rows).to_string(index=False))
    print("\n" + "=" * 110 + "\nผู้ชนะที่แนะนำ\n" + "=" * 110)
    print(pd.DataFrame(win_rows).to_string(index=False))
    print(f"\n{raw_note}\nตัวรวม: {combined_note}")
    print(f"\nconfig ที่ต่างกันที่ลองแล้ว (experiment_log): {n_log} · เวลา {elapsed / 60:.1f} นาที")
    print(f"เขียน {STAGE / 'stage2_summary.md'}")
    if stop_and_ask:
        print("\n!!! ตัวรวมแพ้ base → ต้องหยุดถามผู้ใช้ (กติกา B14/B27) !!!")


if __name__ == "__main__":
    main()
