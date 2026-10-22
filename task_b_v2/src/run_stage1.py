"""
run_stage1.py — Stage 1 การแทนข้อมูล: เปลี่ยนจาก base ทีละปัจจัย เลือกผู้ชนะด้วย IC แล้วรัน config รวม (Checkpoint C4)
=================================================================================================================
ไฟล์นี้ทำอะไร (SPEC ข้อ 5.2 + DECISIONS B14, B16, B17, B19)
    1. รัน base + ตัวเลือกของ 4 ปัจจัย + ตัวเทียบ detrend  (9 config × 2 หุ้น × 3 fold × 3 seeds)
         window    64/16/2 · 256/32/4
         ชนิด       complex
         normalize global · per-image
         channel   3 ช่อง (ret3) · 4 ช่อง (ret4)
         ตัวเทียบ   ช่องราคาหัก trend (detrend) — รายงานคะแนนเท่านั้น ไม่ร่วมเป็นผู้ชนะ (B19)
    2. คิด noise ของ seed จาก IC รายseed แล้วเลือกผู้ชนะของแต่ละปัจจัยด้วย `src/selection.py`
         ชนะ base ได้ก็ต่อเมื่อ IC สูงกว่า base **เกิน noise** · ไม่งั้นคงค่า base
    3. รัน config ที่รวมผู้ชนะทุกปัจจัย (B14) แล้วเทียบกับ base — **ถ้าแพ้ base ให้หยุดถามผู้ใช้**
    4. ตรวจว่าผลของ base ใน Stage 1 ตรงกับ base ใน C3 ทุกบิต (พิสูจน์ว่ารันซ้ำได้ผลเดิม)
    5. ตรวจคาดการณ์ที่ผู้ใช้ประกาศไว้ก่อนรัน (DECISIONS ส่วน C: P1, P2)
    6. เขียนผลลง results/stage1_representation/ และ results/stage1_representation_combined/

*** สคริปต์นี้ "แนะนำ" ผู้ชนะเท่านั้น ไม่เดินต่อไป Stage 2 เอง — ผู้ใช้ต้องยืนยันที่ C4 ก่อน (SPEC ข้อ 9) ***

รัน:  python -m src.run_stage1 --jobs 6
      python -m src.run_stage1 --smoke --jobs 6   (epoch น้อยมาก เขียนไป results/_smoke_stage1/ เพื่อตรวจว่าสายงานไม่พัง · ตัวเลขไม่มีความหมาย)
"""

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src import paths, runner, selection
from src.check_data import md_table
from src.configs import BASE, variant
from src.settings import SEARCH_SEEDS, TICKERS, VAL_END

# ปัจจัย → {ชื่อ config: สิ่งที่เปลี่ยนจาก BASE} (SPEC ข้อ 5.2 · เปลี่ยนทีละปัจจัย)
FACTORS = {
    "window": {"win64": dict(window=64), "win256": dict(window=256, nperseg=32, hop=4)},
    # complex ใช้สเกลคงที่ 1e-3 (= arcsinh(z·1e3)) ตาม DECISIONS B10
    "type": {"complex": dict(spec_type="complex", scale_mode="fixed")},
    "normalize": {"norm_global": dict(normalize="global"), "norm_image": dict(normalize="image")},
    "channel": {"ch3": dict(channels="ret3"), "ch4": dict(channels="ret4")},
}
COMPARATORS = {"detrend": dict(channels="detrend")}          # ตัวเทียบ (B19) ไม่ร่วมเป็นผู้ชนะ

# คาดการณ์ที่ผู้ใช้ประกาศไว้ก่อนรัน (DECISIONS ส่วน C) — ชื่อ config ที่เกี่ยวข้องกับแต่ละข้อ
P1_CONFIG = "complex"
P2_CONFIGS = ["win64", "win256", "norm_global", "norm_image", "ch3", "ch4"]

STAGE = paths.RESULTS_DIR / "stage1_representation"
COMBINED = paths.RESULTS_DIR / "stage1_representation_combined"
C3_PREDICTIONS = paths.RESULTS_DIR / "stage0_base" / "predictions.csv"


def all_configs(shrink=None):
    cfgs = [BASE]
    for opts in FACTORS.values():
        cfgs += [variant(n, **ch) for n, ch in opts.items()]
    cfgs += [variant(n, **ch) for n, ch in COMPARATORS.items()]
    return [replace(c, **shrink) for c in cfgs] if shrink else cfgs


def fmt(v, kind):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return {"r4": f"{v:.4f}", "r3": f"{v:.3f}", "s3": f"{v:+.3f}", "s4": f"{v:+.4f}"}[kind]


def pivot_by_config(evals, metric, kind, order):
    """แถว = config · คอลัมน์ = 6 evaluation + ค่าเฉลี่ย"""
    rows = []
    for name in order:
        g = evals[evals["config"] == name].set_index(["ticker", "fold"])
        vals = [g.loc[(t, k), metric] for t in TICKERS for k in (1, 2, 3)]
        rows.append({"config": name, **{f"{t[0]}{k}": fmt(float(v), kind) for (t, k), v in
                                         zip([(t, k) for t in TICKERS for k in (1, 2, 3)], vals)},
                     "เฉลี่ย": fmt(float(np.mean(vals)), kind)})
    return pd.DataFrame(rows)


def check_matches_c3(evals_preds, c3_path=C3_PREDICTIONS):
    """base ใน Stage 1 ต้องให้คำทำนายเฉลี่ยตรงกับ base ใน C3 ทุกบิต (รันซ้ำได้ผลเดิม)"""
    if not c3_path.exists():
        return None, "ไม่พบผล C3 ให้เทียบ"
    a = evals_preds[evals_preds["config"] == "base"].sort_values(["ticker", "fold", "date"])
    # float_precision="round_trip" สำคัญ: ตัวอ่านทศนิยมแบบเร็วของ pandas คลาดได้ 1 ULP ซึ่งจะทำให้เทียบ "ตรงกันทุกบิต" พลาด
    b = pd.read_csv(c3_path, float_precision="round_trip")
    b = b[b["config"] == "base"].sort_values(["ticker", "fold", "date"])
    if len(a) != len(b):
        return False, f"จำนวนแถวไม่เท่ากัน ({len(a)} vs {len(b)})"
    diff = float(np.max(np.abs(a["pred"].to_numpy() - b["pred"].to_numpy())))
    return diff == 0.0, f"ต่างกันสูงสุด {diff:.3e}"


def judge_predictions(comparison_tables, scores, noise):
    """
    ตัดสินคาดการณ์ที่ประกาศไว้ล่วงหน้า (DECISIONS ส่วน C) ตาม "วิธีอ่านผล" ที่ประกาศพร้อมกัน — ห้ามเปลี่ยนเกณฑ์ตรงนี้
      P1 : complex เหนือ base เกิน noise และ IC > 0  → "เป็นจริง" · ไม่งั้น "ไม่สนับสนุน"
      P2 : ไม่มี log-power ตัวอื่นเหนือ base เกิน noise → "ไม่ขัด" · มีตัวใดเกิน noise → "ขัด (ตัวนั้น)"
    """
    sc = scores.set_index("config")
    cmp_all = pd.concat(comparison_tables, ignore_index=True).set_index("config")
    p1_delta = float(cmp_all.loc[P1_CONFIG, "delta_vs_base"])
    p1_ic = float(sc.loc[P1_CONFIG, "mean_IC"])
    p1_ok = bool(cmp_all.loc[P1_CONFIG, "beats_base"] and p1_ic > 0)
    breakers = [c for c in P2_CONFIGS if bool(cmp_all.loc[c, "beats_base"])]
    return [
        {"ข้อ": "P1 complex ควรช่วย (IC บวกและเหนือ base เกิน noise)",
         "ผล": f"IC {p1_ic:+.4f} · ต่างจาก base {p1_delta:+.4f} · noise {noise:.4f}",
         "สรุป": "**เป็นจริง**" if p1_ok else "**ไม่สนับสนุน**"},
        {"ข้อ": "P2 log-power ตัวอื่นไม่ควรช่วย (ไม่เหนือ base เกิน noise)",
         "ผล": "ΔIC: " + " · ".join(f"{c} {float(cmp_all.loc[c, 'delta_vs_base']):+.4f}" for c in P2_CONFIGS),
         "สรุป": "**ไม่ขัด**" if not breakers else f"**ขัด** — {', '.join(breakers)} เหนือ base เกิน noise"},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=len(SEARCH_SEEDS))
    ap.add_argument("--smoke", action="store_true", help="epoch น้อยมาก + เขียนไปโฟลเดอร์ชั่วคราว (ตรวจสายงานเท่านั้น)")
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))

    global STAGE, COMBINED
    shrink = None
    if args.smoke:
        STAGE = paths.RESULTS_DIR / "_smoke_stage1" / "stage1_representation"
        COMBINED = paths.RESULTS_DIR / "_smoke_stage1" / "stage1_representation_combined"
        runner.EXPERIMENT_LOG = paths.RESULTS_DIR / "_smoke_stage1" / "experiment_log.csv"   # ไม่แตะ log จริง
        shrink = dict(max_epochs=3, patience=3)
        print("[smoke] ตัวเลขไม่มีความหมาย — ตรวจแค่ว่าสายงานทำงานครบ")

    paths.assert_output_dir(STAGE)
    t_start = time.time()

    # ---------------- 1. รัน 9 config ----------------
    configs = all_configs(shrink)
    evals, scores, preds = runner.run_configs(configs, STAGE, seeds=seeds, jobs=args.jobs)
    selection.assert_scores_have_seed_noise(scores)
    noise = selection.pooled_seed_noise(scores)
    se_ref = selection.sampling_se_reference()
    print(f"\n[stage1] noise ของ seed (pooled) = {noise:.4f} · อ้างอิง noise จากการสุ่มวัน ≈ {se_ref:.4f}")

    # ---------------- 2. ผู้ชนะต่อปัจจัย ----------------
    winners, cmp_tables, changes = {}, [], {}
    for factor, options in FACTORS.items():
        w, tbl = selection.pick_factor_winner(scores, "base", list(options), noise)
        tbl.insert(0, "factor", factor)
        winners[factor] = w
        cmp_tables.append(tbl)
        if w != "base":
            changes.update(options[w])
    detrend_row = selection.compare_to_base(scores, "base", list(COMPARATORS), noise)
    detrend_row.insert(0, "factor", "ตัวเทียบ (ไม่ร่วมเป็นผู้ชนะ)")

    # ---------------- 3. config รวมผู้ชนะ (B14) ----------------
    combined_note, stop_and_ask, d_ic = "", False, float("nan")
    scores_all = scores
    evals_all = evals
    if not changes:
        combined_note = "ไม่มีปัจจัยใดชนะ base เกิน noise → config รวม = base ทุกประการ จึงไม่ต้องรันซ้ำและไม่แพ้ (B14)"
        print(f"[stage1] {combined_note}")
    else:
        comb_cfg = variant("combined", **changes)
        if shrink:
            comb_cfg = replace(comb_cfg, **shrink)
        print(f"\n[stage1] config รวมผู้ชนะ: {changes}")
        paths.assert_output_dir(COMBINED)
        ev_c, sc_c, pr_c = runner.run_configs([comb_cfg], COMBINED, seeds=seeds, jobs=args.jobs)
        scores_all = pd.concat([scores, sc_c], ignore_index=True)
        evals_all = pd.concat([evals, ev_c], ignore_index=True)
        stop_and_ask, d_ic = selection.combined_loses_to_base(scores_all, "combined", "base")
        combined_note = (f"config รวม IC {float(sc_c.iloc[0]['mean_IC']):+.4f} vs base {float(scores.set_index('config').loc['base', 'mean_IC']):+.4f} "
                         f"(ต่าง {d_ic:+.4f}, noise {noise:.4f}) → " + ("**แพ้ base — ต้องหยุดถามผู้ใช้**" if stop_and_ask else "ไม่แพ้ base"))
        print(f"[stage1] {combined_note}")

    # ---------------- 4. รันซ้ำได้ผลเดิมไหม ----------------
    same, same_note = check_matches_c3(preds)
    print(f"[stage1] base ใน Stage 1 เทียบกับ C3: {'ตรงกันทุกบิต' if same else ('ไม่ตรง' if same is False else '—')} ({same_note})")

    # ---------------- 5. คาดการณ์ที่ประกาศไว้ ----------------
    verdicts = judge_predictions(cmp_tables, scores, noise)

    # ---------------- 6. เขียนผล ----------------
    order = ["base"] + [n for o in FACTORS.values() for n in o] + list(COMPARATORS) + (["combined"] if changes else [])
    sc = scores_all.set_index("config")
    n_log = len(pd.read_csv(runner.EXPERIMENT_LOG)) if runner.EXPERIMENT_LOG.exists() else 0
    elapsed = time.time() - t_start

    sel_rows = []
    for name in order:
        r = sc.loc[name]
        factor = next((f for f, o in FACTORS.items() if name in o), "ตัวเทียบ" if name in COMPARATORS else ("รวม" if name == "combined" else "base"))
        beats = ""
        if name in {n for o in FACTORS.values() for n in o}:
            beats = "ชนะ" if bool(pd.concat(cmp_tables).set_index("config").loc[name, "beats_base"]) else "ไม่เกิน noise"
        sel_rows.append({"config": name, "ปัจจัย": factor, "mean IC": fmt(float(r["mean_IC"]), "s4"),
                         "ΔIC vs base": "—" if name == "base" else fmt(float(r["mean_IC"] - sc.loc["base", "mean_IC"]), "s4"),
                         "σ seed": fmt(float(r["ic_seed_std"]), "r4"), "IC ราย seed": r["ic_by_seed"].replace(";", " / "),
                         "Spearman": fmt(float(r["mean_IC_spearman"]), "s4"), "IC>0": f"{int(r['n_evals_ic_positive'])}/6",
                         "เทียบ noise": beats})
    rep_rows = []
    for name in order:
        r = sc.loc[name]
        rep_rows.append({"config": name, "relMAE (return)": fmt(float(r["mean_relMAE"]), "r4"),
                         "relMAE (บาท)": fmt(float(r["mean_relMAE_baht"]), "r4"), "MAE (บาท)": fmt(float(r["mean_MAE_baht"]), "r4"),
                         "ชนะ Naive": f"{int(r['n_evals_beat_naive'])}/6", "DirAcc": fmt(float(r["mean_DirAcc"]), "r3"),
                         "StdRatio เฉลี่ย": fmt(float(r["mean_StdRatio"]), "r3"), "StdRatio ต่ำสุด": fmt(float(r["min_StdRatio"]), "r3"),
                         "ธง StdRatio ≥ 0.05": "ผ่าน" if bool(r["stdratio_ok"]) else "ต่ำกว่า"})
    win_rows = [{"ปัจจัย": f, "ผู้ชนะ (แนะนำ)": w, "เหตุผล": "ชนะ base เกิน noise" if w != "base" else "ไม่มีตัวเลือกใดชนะ base เกิน noise → คงค่า base"}
                for f, w in winners.items()]

    md = ["# C4 (Stage 1) — การแทนข้อมูล · development / walk-forward (โซน dev ≤ " + VAL_END + ")", "",
          ("> **ผลของโหมด --smoke — ตัวเลขไม่มีความหมาย**" if args.smoke else
           "> ผลทุกตัวเป็น development / walk-forward ไม่ใช่ test · ผู้ชนะที่แนะนำรอผู้ใช้ยืนยันก่อนไป Stage 2"), "",
          "## กติกาที่ใช้ (DECISIONS B14, B16, B17)", "",
          "- เลือกด้วย **mean IC (Pearson) ของ 6 evaluation** · ชนะ base ได้ก็ต่อเมื่อ IC สูงกว่า base **เกิน noise ของ seed** · ไม่งั้นคงค่า base",
          f"- noise ของ seed (pooled ข้าม {len(configs)} config) = **{noise:.4f}** · (ตัวเลขอ้างอิง ไม่อยู่ในกติกา: noise จากการสุ่มวันของ mean IC ≈ {se_ref:.4f})",
          "- relMAE (return และบาท) รายงานทุกตาราง · StdRatio เป็นธงรายงาน ไม่ตัดสิทธิ์", "",
          "## 1. ตารางการเลือก (เรียงตามลำดับปัจจัย)", "", md_table(pd.DataFrame(sel_rows)), "",
          "## 2. ผู้ชนะที่แนะนำของแต่ละปัจจัย", "", md_table(pd.DataFrame(win_rows)), "",
          f"ตัวเทียบ detrend: IC {fmt(float(sc.loc['detrend', 'mean_IC']), 's4')} (ต่างจาก base {fmt(float(detrend_row.iloc[0]['delta_vs_base']), 's4')}) — รายงานเท่านั้น ไม่ร่วมเป็นผู้ชนะ (B19)", "",
          "## 3. config รวมผู้ชนะ (กติกา B14)", "",
          f"changes จาก base: `{changes if changes else '(ไม่มี)'}`", "", combined_note, "",
          "## 4. ตัววัดหลักที่รายงาน (relMAE หน่วย return และบาท) และธง StdRatio", "", md_table(pd.DataFrame(rep_rows)), "",
          "## 5. IC (Pearson) แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "IC", "s3", order)), "",
          "## 6. IC แบบ Spearman แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "IC_spearman", "s3", order)), "",
          "## 7. relMAE หน่วย return แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "relMAE", "r4", order)), "",
          "## 8. relMAE หน่วยบาท แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "relMAE_baht", "r4", order)), "",
          "## 9. StdRatio แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "StdRatio", "r3", order)), "",
          "## 10. Direction accuracy แยกตาม 6 evaluation", "", md_table(pivot_by_config(evals_all, "DirAcc", "r3", order)), "",
          "## 11. คาดการณ์ที่ผู้ใช้ประกาศไว้ก่อนรัน (DECISIONS ส่วน C)", "",
          "> ข้อความคาดการณ์ (verbatim): ตัวเทียบมี IC บวกเพราะมี return lag (สัญญาณกลับตัวระยะสั้น) · log-power บอกทิศทางไม่ได้ "
          "ถ้าสมมติฐานถูก complex และ hybrid ควรได้ IC บวก ส่วนตัวแปรอื่นของ log-power ไม่ควรช่วย", "",
          md_table(pd.DataFrame(verdicts)), "",
          "P3 (hybrid) ตัดสินหลัง Stage 3", "",
          "## 12. ตรวจการรันซ้ำและการนับ config", "",
          f"- base ใน Stage 1 เทียบกับ base ใน C3: **{'ตรงกันทุกบิต' if same else ('ไม่ตรง' if same is False else '—')}** ({same_note})",
          f"- จำนวน config ที่ลองแล้วใน experiment_log.csv: **{n_log}**",
          f"- เวลา: {elapsed / 60:.1f} นาที (jobs {args.jobs})", ""]
    (STAGE / "stage1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    decision = {"status": "แนะนำเท่านั้น — รอผู้ใช้ยืนยันที่ C4", "smoke": bool(args.smoke), "noise_pooled": noise,
                "sampling_se_reference": se_ref, "winners": winners, "combined_changes": changes,
                "combined_loses_to_base": bool(stop_and_ask), "combined_delta_ic": None if not np.isfinite(d_ic) else d_ic,
                "base_matches_c3": same, "n_configs_logged": n_log, "seeds": list(seeds)}
    (STAGE / "stage1_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- พิมพ์สรุป ----------------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 110 + "\nตารางการเลือก (mean IC · เทียบ noise)\n" + "=" * 110)
    print(pd.DataFrame(sel_rows).to_string(index=False))
    print("\n" + "=" * 110 + "\nตัววัดหลักที่รายงาน\n" + "=" * 110)
    print(pd.DataFrame(rep_rows).to_string(index=False))
    print("\n" + "=" * 110 + "\nผู้ชนะที่แนะนำ\n" + "=" * 110)
    print(pd.DataFrame(win_rows).to_string(index=False))
    print(f"\nconfig รวม: {combined_note}")
    print("\n" + "=" * 110 + "\nคาดการณ์ที่ประกาศไว้\n" + "=" * 110)
    print(pd.DataFrame(verdicts).to_string(index=False))
    print(f"\nconfig ที่ลองแล้วทั้งหมด (experiment_log): {n_log} · เวลา {elapsed / 60:.1f} นาที")
    print(f"เขียน {STAGE / 'stage1_summary.md'}")
    if stop_and_ask:
        print("\n!!! config รวมแพ้ base → ต้องหยุดถามผู้ใช้ (กติกา B14) !!!")


if __name__ == "__main__":
    main()
