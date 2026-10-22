"""
run_stage345.py — Stage 3 (input ของ ANN) · Stage 4 (สองขั้น) · Stage 5 (loss) (DECISIONS B28–B33)
=================================================================================================
ไฟล์นี้ทำอะไร: รัน 3 Stage ต่อกันได้ด้วยคำสั่งย่อย `3`, `4`, `5` (ผู้ใช้สั่งให้รันต่อกันโดยไม่หยุดรอ — B29)

    Stage 3  base (spectrogram อย่างเดียว) · hybrid (+29 feature + linear skip) · hybrid-noCNN (control B25)
             1) placebo ของคู่ hybrid/hybrid-noCNN: 5 shuffle (seed 0–4) → threshold_pair = ค่าสูงสุดของ ΔIC (B28)  ← รันก่อน
             2) รัน config จริง แล้วตัดสิน P3 (hybrid vs base) และ P3b (hybrid vs hybrid-noCNN)
    Stage 4  two_stage_spec (vs base) · two_stage_hybrid (vs hybrid) — cross-fitting 4 ช่วง (B32) · ประเมินเวลาก่อน: > 30 นาที → ข้าม
    Stage 5  loss: MSE และ MAE เทียบ Huber กับทั้ง track 1 (base) และ track 2 (hybrid) — ไม่มีผู้ชนะ → คง Huber

กติกาผู้ชนะ Stage 4–5 (B22 + B31): ΔIC เทียบ config อ้างอิงของ track ตัวเอง > max(noise ของ stage, +0.0597)
ผลอ้างอิง (base, hybrid) ใช้จาก Stage 3 (config เดียวกัน · deterministic) ไม่รันซ้ำ

รัน:  python -m src.run_stage345 3 --jobs 6
      python -m src.run_stage345 4 --jobs 6 [--estimate-only]
      python -m src.run_stage345 5 --jobs 6        (เติม --smoke เพื่อตรวจสายงานด้วย epoch น้อย · ตัวเลขไม่มีความหมาย)
"""

import argparse
import dataclasses
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src import models, paths, placebo, runner, selection
from src.check_data import md_table
from src.configs import BASE, variant
from src.run_stage1 import fmt, pivot_by_config
from src.settings import SEARCH_SEEDS, TICKERS, VAL_END

SHUFFLES = (0, 1, 2, 3, 4)                                      # B28
HYBRID = variant("hybrid", ann_input="hybrid")
NOCNN = variant("hybrid_nocnn", ann_input="hybrid_nocnn")
STAGE3_CFGS = [BASE, HYBRID, NOCNN]
STAGE4_CFGS = [variant("two_stage_spec", train_mode="two_stage"),
               variant("two_stage_hybrid", train_mode="two_stage", ann_input="hybrid")]
STAGE4_REF = {"two_stage_spec": "base", "two_stage_hybrid": "hybrid"}
STAGE5_CFGS = [variant("base_mse", loss="mse"), variant("base_mae", loss="mae"),
               variant("hybrid_mse", ann_input="hybrid", loss="mse"), variant("hybrid_mae", ann_input="hybrid", loss="mae")]
STAGE5_REF = {"base_mse": "base", "base_mae": "base", "hybrid_mse": "hybrid", "hybrid_mae": "hybrid"}

R = paths.RESULTS_DIR
DIRS = {"3": R / "stage3_input", "3p": R / "stage3_placebo", "4": R / "stage4_twostage", "5": R / "stage5_loss"}
B22_FILE = R / "stage1_placebo" / "stage1_final_decision.json"
EFFICIENCY = 4.6                                                 # ความเร็วขึ้นของ jobs 6 ที่วัดได้จริงใน C3 (6.0 นาที → 1.3 นาที)
STAGE4_LIMIT_MIN = 30.0                                          # B29


def b22_floor():
    return float(json.loads(B22_FILE.read_text(encoding="utf-8"))["b22_threshold_stage2_to_5"])


def shrunk(cfgs, shrink):
    return [replace(c, **shrink) for c in cfgs] if shrink else cfgs


def role_of(name, refs):
    return "อ้างอิง/control" if name not in refs else "ผู้สมัคร"


# ---------------------------------------------------------------
# ตารางรายงาน (ใช้ร่วมกันทุก Stage)
# ---------------------------------------------------------------
def selection_rows(sc, order, ref_of, noise, floor):
    """ตารางการเลือก: IC · ΔIC เทียบ config อ้างอิงของ track ตัวเอง · เกณฑ์ · ผล"""
    rows = []
    for n in order:
        r = sc.loc[n]
        ref = ref_of.get(n)
        d = None if ref is None else float(r["mean_IC"] - sc.loc[ref, "mean_IC"])
        thr = max(noise, floor) if floor is not None else noise
        rows.append({"config": n, "เทียบกับ": ref or "—", "mean IC": fmt(float(r["mean_IC"]), "s4"),
                     "ΔIC": "—" if d is None else fmt(d, "s4"), "เกณฑ์": "—" if ref is None else f"{thr:.4f}",
                     "σ seed": fmt(float(r["ic_seed_std"]), "r4"), "IC ราย seed": r["ic_by_seed"].replace(";", " / "),
                     "Spearman": fmt(float(r["mean_IC_spearman"]), "s4"), "IC>0": f"{int(r['n_evals_ic_positive'])}/6",
                     "ผล": "" if ref is None else ("**ชนะ**" if d > thr else "ไม่ผ่านเกณฑ์")})
    return rows


def reporting_rows(sc, order):
    """ตัววัดหลักที่รายงาน: relMAE (return/บาท) ทั้งก่อนและหลังปรับขนาดด้วย c + การกระจายของ c + ธง StdRatio"""
    rows = []
    for n in order:
        r = sc.loc[n]
        rows.append({"config": n, "relMAE (return)": fmt(float(r["mean_relMAE"]), "r4"), "relMAE (บาท)": fmt(float(r["mean_relMAE_baht"]), "r4"),
                     "หลังปรับขนาด (return)": fmt(float(r["mean_relMAE_shrunk"]), "r4"),
                     "หลังปรับขนาด (บาท)": fmt(float(r["mean_relMAE_baht_shrunk"]), "r4"),
                     "MAE (บาท)": fmt(float(r["mean_MAE_baht"]), "r4"), "ชนะ Naive": f"{int(r['n_evals_beat_naive'])}/6",
                     "ชนะ Naive หลังปรับ": f"{int(r['n_evals_beat_naive_shrunk'])}/6", "c เฉลี่ย": fmt(float(r["mean_c"]), "r3"),
                     "c = 0": f"{int(r['n_c_zero'])}/6", "DirAcc": fmt(float(r["mean_DirAcc"]), "r3"),
                     "StdRatio เฉลี่ย": fmt(float(r["mean_StdRatio"]), "r3"),
                     "ธง StdRatio ≥ 0.05": "ผ่าน" if bool(r["stdratio_ok"]) else "ต่ำกว่า"})
    return rows


def per_eval_sections(evals, order):
    out = []
    for i, (title, metric, kind) in enumerate([("IC (Pearson)", "IC", "s3"), ("IC แบบ Spearman", "IC_spearman", "s3"),
                                               ("relMAE หน่วย return", "relMAE", "r4"), ("relMAE หน่วยบาท", "relMAE_baht", "r4"),
                                               ("relMAE หน่วย return หลังปรับขนาดด้วย c", "relMAE_shrunk", "r4"),
                                               ("c ที่เลือก (ต่อ evaluation)", "c_chosen", "r3"),
                                               ("StdRatio", "StdRatio", "r3"), ("Direction accuracy", "DirAcc", "r3")], start=1):
        out += [f"### แยกตาม 6 evaluation — {title}", "", md_table(pivot_by_config(evals, metric, kind, order)), ""]
    return out


def load_reference(stage_key, names):
    d = DIRS[stage_key]
    sc, ev = pd.read_csv(d / "scores.csv"), pd.read_csv(d / "evals.csv")
    return sc[sc["config"].isin(names)], ev[ev["config"].isin(names)]


def write_stage(stage_dir, name, md, decision):
    (stage_dir / f"{name}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (stage_dir / f"{name}_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def print_tables(*tables):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for title, rows in tables:
        print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)
        print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------
def stage3(jobs, seeds, shrink):
    t0 = time.time()
    DIRS["3"].mkdir(parents=True, exist_ok=True)
    floor = b22_floor()

    # ---- 1) placebo ของคู่ (ก่อนรัน config จริง · B28) ----
    pair = shrunk([HYBRID, NOCNN], shrink)
    ev_p = placebo.run_placebo(pair, DIRS["3p"], shuffle_seeds=SHUFFLES, seeds=seeds, jobs=jobs)
    pa = placebo.analyze_pair(ev_p, "hybrid", "hybrid_nocnn", SHUFFLES)
    thr_pair = pa["threshold_pair"]
    pdf = pa["per_shuffle"]
    prow = [{"shuffle": int(r.shuffle), "IC hybrid": fmt(r.IC_hybrid, "s4"), "IC hybrid-noCNN": fmt(r.IC_hybrid_nocnn, "s4"),
             "ΔIC": fmt(r.delta_ic, "s4")} for r in pdf.itertuples()]
    print(f"\n[stage3] placebo ของคู่ (5 shuffle): threshold_pair = ค่าสูงสุดของ ΔIC = {thr_pair:+.4f} (เฉลี่ย {pa['mean_delta']:+.4f})")
    pdf.to_csv(DIRS["3p"] / "pair_placebo.csv", index=False)
    (DIRS["3p"] / "pair_threshold.json").write_text(json.dumps({"threshold_pair": thr_pair, "mean_delta": pa["mean_delta"],
                                                                "shuffles": list(SHUFFLES), "deltas": pdf["delta_ic"].tolist()}, indent=2), encoding="utf-8")

    # ---- 2) config จริง ----
    cfgs = shrunk(STAGE3_CFGS, shrink)
    evals, scores, preds = runner.run_configs(cfgs, DIRS["3"], seeds=seeds, jobs=jobs)
    selection.assert_scores_have_seed_noise(scores)
    noise = selection.pooled_seed_noise(scores)
    sc = scores.set_index("config")
    ic = lambda n: float(sc.loc[n, "mean_IC"])
    d_hb, d_hn, d_nb = ic("hybrid") - ic("base"), ic("hybrid") - ic("hybrid_nocnn"), ic("hybrid_nocnn") - ic("base")
    p3 = bool(d_hb > noise and ic("hybrid") > 0)                                    # P3 เดิม (ไม่แก้)
    p3b_thr = max(noise, thr_pair)
    p3b = bool(d_hn > p3b_thr)                                                      # P3b (B28/B30)
    b22_info = bool(d_hb > max(noise, floor))
    print(f"[stage3] noise (3 config) = {noise:.4f}")
    print(f"[stage3] P3  (hybrid vs base): Δ {d_hb:+.4f} vs noise {noise:.4f}, IC hybrid {ic('hybrid'):+.4f} → {'เป็นจริง' if p3 else 'ไม่สนับสนุน'}")
    print(f"[stage3] P3b (hybrid vs noCNN): Δ {d_hn:+.4f} vs max(noise, threshold_pair) = {p3b_thr:.4f} → {'เป็นจริง' if p3b else 'ไม่สนับสนุน'}")

    order = ["base", "hybrid", "hybrid_nocnn"]
    sel = selection_rows(sc, order, {"hybrid": "base"}, noise, floor)
    for r in sel:                                                                    # คอลัมน์เสริม: Δ เทียบ control
        r["ΔIC เทียบ hybrid-noCNN"] = {"hybrid": fmt(d_hn, "s4"), "hybrid_nocnn": "—", "base": fmt(-d_nb, "s4")}[r["config"]]
    rep = reporting_rows(sc, order)
    params = {"base": models.layer_report(BASE)[1], "hybrid": models.layer_report(HYBRID, 29)[1],
              "hybrid_nocnn": models.count_params(models.build(NOCNN, 29))}
    verdicts = [
        {"ข้อ": "P3 (เดิม ไม่แก้): mean IC ของ hybrid สูงกว่า base เกิน noise และ > 0", "ผล": f"Δ {d_hb:+.4f} vs noise {noise:.4f} · IC hybrid {ic('hybrid'):+.4f}",
         "สรุป": "**เป็นจริง**" if p3 else "**ไม่สนับสนุน**"},
        {"ข้อ": "P3b: ΔIC(hybrid − hybrid-noCNN) > max(noise, threshold_pair)", "ผล": f"Δ {d_hn:+.4f} vs max({noise:.4f}, {thr_pair:.4f}) = {p3b_thr:.4f}",
         "สรุป": "**เป็นจริง**" if p3b else "**ไม่สนับสนุน**"}]
    elapsed = time.time() - t0
    md = ["# Stage 3 — input ของ ANN (spectrogram อย่างเดียว vs hybrid vs control hybrid-noCNN)", "",
          "> ผลทุกตัวเป็น development / walk-forward · ผู้ชนะ/ธง/กติกาตาม DECISIONS B16, B17, B22, B25, B28, B30, B31", "",
          f"- placebo ของคู่ (B28): 5 shuffle · **threshold_pair = ค่าสูงสุดของ ΔIC = {thr_pair:+.4f}** (ΔIC เฉลี่ย {pa['mean_delta']:+.4f})",
          f"- noise ของ seed (pooled 3 config) = {noise:.4f} · เกณฑ์ B22 (ข้อมูล) = {floor:+.4f}", "",
          "## 1. placebo ของคู่ hybrid / hybrid-noCNN (y สลับใน train)", "", md_table(pd.DataFrame(prow)), "",
          "## 2. คาดการณ์ P3 / P3b (ประกาศไว้ก่อนรัน)", "", md_table(pd.DataFrame(verdicts)), "",
          f"ข้อมูลเสริม: hybrid เทียบ base ตามเกณฑ์ B22 (> max(noise, {floor:.4f})) = {'ผ่าน' if b22_info else 'ไม่ผ่าน'} · "
          f"control เทียบ base ΔIC = {d_nb:+.4f} · **ไม่ทำให้ hybrid มาแทน track 1 (B31)**", "",
          "## 3. ตารางการเลือก (IC เทียบ base)", "", md_table(pd.DataFrame(sel)), "",
          "พารามิเตอร์: " + " · ".join(f"{k} {v:,}" for k, v in params.items()), "",
          "## 4. ตัววัดหลักที่รายงาน (relMAE ก่อน/หลังปรับขนาดด้วย c) และธง", "", md_table(pd.DataFrame(rep)), ""]
    md += per_eval_sections(evals, order)
    md += [f"เวลา: {elapsed / 60:.1f} นาที (jobs {jobs}) · config ที่ต่างกันที่ลองแล้ว: "
           f"{len(pd.read_csv(runner.EXPERIMENT_LOG)) if runner.EXPERIMENT_LOG.exists() else 0}", ""]
    decision = {"status": "ผล Stage 3 — ส่งต่อ Stage 4–5 อัตโนมัติ (B29)", "noise": noise, "threshold_pair": thr_pair, "p3": p3, "p3b": p3b,
                "delta_hybrid_vs_base": d_hb, "delta_hybrid_vs_nocnn": d_hn, "delta_nocnn_vs_base": d_nb, "b22_hybrid_vs_base": b22_info,
                "track1_config": dataclasses.asdict(BASE), "track2_config": dataclasses.asdict(HYBRID), "control_config": dataclasses.asdict(NOCNN),
                "seeds": list(seeds), "smoke": bool(shrink)}
    write_stage(DIRS["3"], "stage3", md, decision)
    print_tables(("ตารางการเลือก", sel), ("ตัววัดหลักที่รายงาน", rep), ("P3 / P3b", verdicts))
    print(f"\nเขียน {DIRS['3'] / 'stage3_summary.md'} · {elapsed / 60:.1f} นาที")


# ---------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------
def estimate_stage4(shrink, jobs):
    """วัดเวลา 1 unit จริง (two_stage_spec · KBANK fold 3 = train ใหญ่สุด · 1 seed) แล้วประมาณทั้ง Stage (B32)"""
    cfg = shrunk([STAGE4_CFGS[0]], shrink)[0]
    t0 = time.time()
    runner.run_unit(cfg, "KBANK", 3, seeds=(0,), cache_dir=None)
    one = time.time() - t0
    serial_min = one * len(SEARCH_SEEDS) * len(STAGE4_CFGS) * 2 * 3 / 60            # 2 config × (2 หุ้น × 3 fold) × 3 seeds
    est = serial_min / min(EFFICIENCY, jobs) if jobs > 1 else serial_min
    info = {"unit_seconds_1seed_largest_train": one, "serial_minutes": serial_min, "efficiency": EFFICIENCY, "jobs": jobs,
            "estimated_minutes": est, "limit_minutes": STAGE4_LIMIT_MIN, "decision": "ข้าม" if est > STAGE4_LIMIT_MIN else "รัน",
            "note": "หน่วยที่วัดคือ train ใหญ่สุด (fold 3) จึงประมาณเกินจริงเล็กน้อย (ปลอดภัย)"}
    DIRS["4"].mkdir(parents=True, exist_ok=True)
    (DIRS["4"] / "time_estimate.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage4] วัดเวลา 1 unit (1 seed) = {one:.0f} วินาที → ประมาณทั้ง Stage ≈ {est:.1f} นาที (เกณฑ์ {STAGE4_LIMIT_MIN:.0f}) → {info['decision']}")
    return info


def stage4(jobs, seeds, shrink, estimate_only):
    info = estimate_stage4(shrink, jobs)
    if estimate_only:
        return
    if info["decision"] == "ข้าม":
        (DIRS["4"] / "stage4_summary.md").write_text(
            f"# Stage 4 — ข้าม\n\nประเมินเวลา ≈ {info['estimated_minutes']:.1f} นาที เกิน {STAGE4_LIMIT_MIN:.0f} นาที → ข้ามตามกติกา B29\n", encoding="utf-8")
        print("[stage4] ข้าม Stage 4 ตามกติกา B29")
        return
    t0 = time.time()
    floor = b22_floor()
    cfgs = shrunk(STAGE4_CFGS, shrink)
    evals, scores, _ = runner.run_configs(cfgs, DIRS["4"], seeds=seeds, jobs=jobs)
    ref_sc, ref_ev = load_reference("3", ["base", "hybrid", "hybrid_nocnn"])
    scores_all = pd.concat([scores, ref_sc], ignore_index=True)
    evals_all = pd.concat([evals, ref_ev], ignore_index=True)
    selection.assert_scores_have_seed_noise(scores_all)
    noise = selection.pooled_seed_noise(scores_all)
    sc = scores_all.set_index("config")
    order = ["base", "two_stage_spec", "hybrid", "two_stage_hybrid", "hybrid_nocnn"]
    sel = selection_rows(sc, order, STAGE4_REF, noise, floor)
    rep = reporting_rows(sc, order)
    d_vs_ctrl = float(sc.loc["two_stage_hybrid", "mean_IC"] - sc.loc["hybrid_nocnn", "mean_IC"])
    winners = {c: bool(sc.loc[c, "mean_IC"] - sc.loc[r, "mean_IC"] > max(noise, floor)) for c, r in STAGE4_REF.items()}
    elapsed = time.time() - t0
    md = ["# Stage 4 — วิธีเทรน: สองขั้น + cross-fitting (B32)", "",
          f"- ประเมินเวลาก่อนรัน: ≈ {info['estimated_minutes']:.1f} นาที (เกณฑ์ {STAGE4_LIMIT_MIN:.0f}) → รัน · ใช้จริง {elapsed / 60:.1f} นาที",
          f"- noise (pooled {len(scores_all)} config) = {noise:.4f} · เกณฑ์ B22 = {floor:+.4f} → เกณฑ์ที่ใช้ {max(noise, floor):.4f} · ผลอ้างอิง (base, hybrid, control) จาก Stage 3",
          "- two_stage_spec: ANN รับ [คำทำนาย OOF ของ CNN] · two_stage_hybrid: ANN รับ [OOF + 29 feature] + linear skip", "",
          "## 1. ตารางการเลือก (เทียบ config อ้างอิงของ track ตัวเอง)", "", md_table(pd.DataFrame(sel)), "",
          f"ข้อมูลเสริม: two_stage_hybrid เทียบ hybrid-noCNN (control) ΔIC = {d_vs_ctrl:+.4f}", "",
          "## 2. ตัววัดหลักที่รายงาน", "", md_table(pd.DataFrame(rep)), ""]
    md += per_eval_sections(evals_all, order)
    write_stage(DIRS["4"], "stage4", md, {"estimate": info, "noise": noise, "floor": floor, "winners": winners,
                                          "two_stage_hybrid_vs_control": d_vs_ctrl, "smoke": bool(shrink)})
    print_tables(("Stage 4 — ตารางการเลือก", sel), ("ตัววัดหลักที่รายงาน", rep))
    print(f"\nผู้ชนะ: {winners} · เขียน {DIRS['4'] / 'stage4_summary.md'} · {elapsed / 60:.1f} นาที")


# ---------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------
def stage5(jobs, seeds, shrink):
    t0 = time.time()
    floor = b22_floor()
    cfgs = shrunk(STAGE5_CFGS, shrink)
    evals, scores, _ = runner.run_configs(cfgs, DIRS["5"], seeds=seeds, jobs=jobs)
    ref_sc, ref_ev = load_reference("3", ["base", "hybrid"])
    scores_all = pd.concat([scores, ref_sc], ignore_index=True)
    evals_all = pd.concat([evals, ref_ev], ignore_index=True)
    selection.assert_scores_have_seed_noise(scores_all)
    noise = selection.pooled_seed_noise(scores_all)
    sc = scores_all.set_index("config")
    order = ["base", "base_mse", "base_mae", "hybrid", "hybrid_mse", "hybrid_mae"]
    sel = selection_rows(sc, order, STAGE5_REF, noise, floor)
    rep = reporting_rows(sc, order)
    winners = {c: bool(sc.loc[c, "mean_IC"] - sc.loc[r, "mean_IC"] > max(noise, floor)) for c, r in STAGE5_REF.items()}
    elapsed = time.time() - t0
    md = ["# Stage 5 — loss (Huber เทียบ MSE และ MAE)", "",
          f"- noise (pooled {len(scores_all)} config) = {noise:.4f} · เกณฑ์ B22 = {floor:+.4f} → เกณฑ์ที่ใช้ {max(noise, floor):.4f} · ผลอ้างอิง base/hybrid (Huber) จาก Stage 3",
          "- ไม่มีผู้ชนะ → คง Huber (สเปกข้อ 5.5) · c และ relMAE หลังปรับขนาดรายงานเพิ่ม (B33)", "",
          "## 1. ตารางการเลือก", "", md_table(pd.DataFrame(sel)), "",
          "## 2. ตัววัดหลักที่รายงาน", "", md_table(pd.DataFrame(rep)), ""]
    md += per_eval_sections(evals_all, order)
    write_stage(DIRS["5"], "stage5", md, {"noise": noise, "floor": floor, "winners": winners, "smoke": bool(shrink)})
    print_tables(("Stage 5 — ตารางการเลือก", sel), ("ตัววัดหลักที่รายงาน", rep))
    print(f"\nผู้ชนะ: {winners} · เขียน {DIRS['5'] / 'stage5_summary.md'} · {elapsed / 60:.1f} นาที")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["3", "4", "5"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=len(SEARCH_SEEDS))
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seeds = tuple(range(a.seeds))
    shrink = dict(max_epochs=3, patience=3) if a.smoke else None
    if a.smoke:
        base = paths.RESULTS_DIR / "_smoke_stage345"
        for k in DIRS:
            DIRS[k] = base / DIRS[k].name
        runner.EXPERIMENT_LOG = base / "experiment_log.csv"
        print("[smoke] ตัวเลขไม่มีความหมาย — ตรวจแค่ว่าสายงานทำงานครบ")
    for d in DIRS.values():
        paths.assert_output_dir(d)
    {"3": lambda: stage3(a.jobs, seeds, shrink), "4": lambda: stage4(a.jobs, seeds, shrink, a.estimate_only),
     "5": lambda: stage5(a.jobs, seeds, shrink)}[a.stage]()


if __name__ == "__main__":
    main()
