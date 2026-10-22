"""
run_placebo.py — placebo test ของ Stage 1 + sign control (ภาคต่อของ C4 · DECISIONS B21–B24)
=========================================================================================
ไฟล์นี้ทำอะไร
    1. placebo : 3 shuffle (seed 0, 1, 2) × 9 config ของ Stage 1 เทรนด้วย y ที่สลับใน train → ใช้กติกา B16/B17 เดิม
         นับ "ผู้ชนะ" ต่อ shuffle · หา ΔIC สูงสุดที่ noise ล้วนให้ได้ (เกณฑ์ B22) · ตัดสินตามกติกา B21
    2. sign control : target = ret_close วันสุดท้ายในหน้าต่าง (มีเครื่องหมาย) กับ 4 config
         (log-power และ complex × window 128 และ 64) รายงาน R² และ corr แยก 6 evaluation + linear probe ว่าข้อมูลหายตรงไหน
    3. เขียนผลลง results/stage1_placebo/ และ results/stage1_sign_control/ แล้วหยุด — ไม่ไป Stage 2

*** สคริปต์นี้รายงานเท่านั้น ไม่เปลี่ยน config/กติกา/pipeline ตามผลเอง (ผู้ใช้สั่ง) ***

รัน:  python -m src.run_placebo --jobs 6
      python -m src.run_placebo --smoke --jobs 6   (epoch น้อยมาก เขียนไป results/_smoke_placebo/ ตรวจสายงานเท่านั้น)
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from src import paths, placebo, sign_control
from src.check_data import md_table
from src.run_stage1 import FACTORS, all_configs, pivot_by_config
from src.settings import SEARCH_SEEDS, TICKERS, VAL_END

STAGE1 = paths.RESULTS_DIR / "stage1_representation"
PLACEBO_DIR = paths.RESULTS_DIR / "stage1_placebo"
SIGN_DIR = paths.RESULTS_DIR / "stage1_sign_control"
NEG_3SEEDS = paths.RESULTS_DIR / "stage0_base" / "negative_control_3seeds.csv"


def fmt(v, kind="s4"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return {"s4": f"{v:+.4f}", "r4": f"{v:.4f}", "s3": f"{v:+.3f}", "r3": f"{v:.3f}"}[kind]


def stage1_reference():
    """ผล Stage 1 จริง (ไว้วางเทียบเท่านั้น): ΔIC ต่อผู้สมัคร, noise, จำนวนผู้ชนะ"""
    sc = pd.read_csv(STAGE1 / "scores.csv").set_index("config")
    dec = json.loads((STAGE1 / "stage1_decision.json").read_text(encoding="utf-8"))
    base_ic = float(sc.loc["base", "mean_IC"])
    delta = {c: float(sc.loc[c, "mean_IC"]) - base_ic for c in sc.index if c != "base"}
    return {"delta": delta, "noise": dec["noise_pooled"], "winners": dec["winners"],
            "n_factor_winners": sum(w != "base" for w in dec["winners"].values()), "base_ic": base_ic,
            "combined_changes": dec["combined_changes"]}


def run_sign_control(jobs, seeds, shrink):
    tasks = []
    for label, cfg in sign_control.SIGN_CONFIGS:
        cfg = replace(cfg, **shrink) if shrink else cfg
        tasks += [(label, cfg, t, k, tuple(seeds), VAL_END) for t in TICKERS for k in (1, 2, 3)]
    print(f"[sign] {len(tasks)} unit (4 config × 2 หุ้น × 3 fold) · jobs {jobs}")
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            res = list(pool.map(sign_control.sign_unit, tasks))
    else:
        res = [sign_control.sign_unit(a) for a in tasks]
    return pd.DataFrame([r for r, _ in res]), pd.DataFrame([p for _, ps in res for p in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=len(SEARCH_SEEDS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    shrink = dict(max_epochs=3, patience=3) if args.smoke else None

    global PLACEBO_DIR, SIGN_DIR
    if args.smoke:
        PLACEBO_DIR = paths.RESULTS_DIR / "_smoke_placebo" / "stage1_placebo"
        SIGN_DIR = paths.RESULTS_DIR / "_smoke_placebo" / "stage1_sign_control"
        print("[smoke] ตัวเลขไม่มีความหมาย — ตรวจแค่ว่าสายงานทำงานครบ")
    paths.assert_output_dir(PLACEBO_DIR)
    paths.assert_output_dir(SIGN_DIR)
    SIGN_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    # ======================= 1. placebo =======================
    configs = all_configs(shrink)
    names = [c.name for c in configs]
    t0 = time.time()
    evals = placebo.run_placebo(configs, PLACEBO_DIR, shuffle_seeds=placebo.SHUFFLE_SEEDS, seeds=seeds, jobs=args.jobs)
    t_placebo = time.time() - t0
    res = placebo.analyze(evals, names, FACTORS, placebo.SHUFFLE_SEEDS)
    S, per, deltas = res["summary"], res["per_shuffle"], res["deltas"]
    consistent, consistent_note = placebo.check_against_negative_control(evals, NEG_3SEEDS)
    ref = stage1_reference()
    print(f"\n[placebo] เทียบกับ negative control ของ C3 (base × shuffle 0): "
          f"{'ตรงกันทุกบิต' if consistent else ('ไม่ตรง' if consistent is False else '—')} ({consistent_note})")

    cand_names = [c for o in FACTORS.values() for c in o]
    delta_pivot = deltas.pivot_table(index="config", columns="shuffle", values="delta_vs_base").loc[cand_names]
    ic_pivot = evals.groupby(["config", "shuffle"])["IC"].mean().unstack("shuffle").loc[names]
    delta_rows = []
    for c in cand_names:
        vals = [float(delta_pivot.loc[c, s]) for s in placebo.SHUFFLE_SEEDS]
        delta_rows.append({"ผู้สมัคร": c, **{f"shuffle {s}": fmt(v) for s, v in zip(placebo.SHUFFLE_SEEDS, vals)},
                           "สูงสุด (placebo)": fmt(max(vals)), "Stage 1 จริง": fmt(ref["delta"][c]),
                           "จริง > เกณฑ์ B22": "ใช่" if ref["delta"][c] > S["b22_threshold"] else "ไม่"})
    per_rows = [{"shuffle": int(r.shuffle), "noise ของ seed": fmt(r.noise, "r4"), "IC ของ base": fmt(r.base_IC),
                 "ผู้ชนะ (ปัจจัย)": int(r.n_factor_winners), "ผู้ชนะที่เลือกได้": r.winners,
                 "config ที่ชนะ base เกิน noise": int(r.n_candidates_beating), "ΔIC สูงสุด": fmt(r.max_delta_IC)}
                for r in per.itertuples()]
    ic_rows = [{"config": c, **{f"shuffle {s}": fmt(float(ic_pivot.loc[c, s])) for s in placebo.SHUFFLE_SEEDS},
                "เฉลี่ย": fmt(float(ic_pivot.loc[c].mean()))} for c in names]

    combined_txt = "ใช้ config รวม (win64 + norm_global + ch3)"
    verdict = {"base": "**ใช้ base ต่อ** (Stage 1 ไม่มีผู้ชนะที่เชื่อได้)", "combined": "**" + combined_txt + "**"}
    md = ["# Placebo test ของ Stage 1 และ sign control (DECISIONS B21–B24)", "",
          "> ผลทุกตัวเป็น development / walk-forward · รายงานเท่านั้น ยังไม่ไป Stage 2" +
          (" · **โหมด --smoke ตัวเลขไม่มีความหมาย**" if args.smoke else ""), "",
          "# ส่วนที่ 1 — placebo (เทรนด้วย y ที่สลับใน train)", "",
          "กติกา (ผู้ใช้ประกาศก่อนรัน · B21): 3 shuffle (seed 0, 1, 2) × 9 config × กติกา B16/B17 เดิม · "
          "จำนวน \"ผู้ชนะ\" เฉลี่ยต่อ shuffle ≥ 1 → ใช้ base ต่อ · < 1 → ใช้ config รวม", "",
          "\"ผู้ชนะ\" (หลัก · B23) = จำนวนปัจจัยที่เลือกผู้ชนะ ≠ base ได้ (สูงสุด 4) · จำนวน config ที่ชนะ base เกิน noise (สูงสุด 7) เป็นตัวเลขรอง", "",
          "## 1.1 ผลต่อ shuffle", "", md_table(pd.DataFrame(per_rows)), "",
          f"- **ผู้ชนะเฉลี่ยต่อ shuffle (หลัก) = {S['mean_n_factor_winners']:.2f}** → ตามกติกา B21: {verdict[S['decision_primary']]}",
          f"- ตัวเลขรอง (config ที่ชนะ base เกิน noise) เฉลี่ย = {S['mean_n_candidates_beating']:.2f} → "
          f"{verdict[S['decision_secondary']]} · สองแบบ{'ให้คำตอบตรงกัน' if S['primary_and_secondary_agree'] else '**ให้คำตอบต่างกัน — ให้ผู้ใช้ตัดสิน (B23)**'}",
          f"- อ้างอิง Stage 1 จริง: ผู้ชนะ (ปัจจัย) = {ref['n_factor_winners']} · noise ของ seed = {ref['noise']:.4f} · "
          f"(placebo: noise เฉลี่ย = {S['mean_noise']:.4f})", "",
          "## 1.2 ΔIC ของผู้สมัครเทียบ base (placebo) และเกณฑ์ B22", "", md_table(pd.DataFrame(delta_rows)), "",
          f"- **เกณฑ์ B22 = ค่าสูงสุดของ ΔIC ข้าม {S['n_delta_values']} ค่า (7 ผู้สมัคร × 3 shuffle) = {S['b22_threshold']:+.4f}** "
          f"(ΔIC สูงสุดต่อ shuffle = {', '.join(f'{x:+.4f}' for x in S['max_delta_per_shuffle'])} · มี 3 ค่า ประมาณเปอร์เซ็นไทล์ 95 ไม่ได้ จึงใช้ค่าสูงสุดตามที่ผู้ใช้ระบุ)",
          "- คอลัมน์ \"จริง > เกณฑ์ B22\" ไว้ดูเทียบเท่านั้น ไม่ใช้ตัดสิน Stage 1 (B23 ข้อ 7)", "",
          "## 1.3 IC เฉลี่ย 6 evaluation ของแต่ละ config ใน placebo", "", md_table(pd.DataFrame(ic_rows)), "",
          "## 1.4 ตรวจความสอดคล้อง", "",
          f"- base × shuffle 0 เทียบกับ negative control 3 seeds ของ C3: **{'ตรงกันทุกบิต' if consistent else ('ไม่ตรง' if consistent is False else '—')}** ({consistent_note})", ""]
    placebo_out = {"summary": {k: v for k, v in S.items()}, "per_shuffle": per.to_dict("records"),
                   "stage1_reference": {"noise": ref["noise"], "n_factor_winners": ref["n_factor_winners"], "delta": ref["delta"]},
                   "consistent_with_c3_negative_control": consistent, "smoke": bool(args.smoke)}
    (PLACEBO_DIR / "placebo_analysis.json").write_text(json.dumps(placebo_out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (PLACEBO_DIR / "placebo_deltas.csv").write_text(deltas.to_csv(index=False), encoding="utf-8")

    # ======================= 2. sign control =======================
    t0 = time.time()
    sign_rows, probes = run_sign_control(args.jobs, seeds, shrink)
    t_sign = time.time() - t0
    sign_rows.to_csv(SIGN_DIR / "sign_control.csv", index=False)
    probes.to_csv(SIGN_DIR / "probe.csv", index=False)
    order = [lab for lab, _ in sign_control.SIGN_CONFIGS]
    expect = sign_control.summarize_expectation(sign_rows.to_dict("records"))
    probe_mean = probes.groupby(["config", "frames", "stage"])["R2"].mean().unstack("stage")[list(sign_control.STAGES)]
    probe_rows = [{"config": c, "frames": f, **{s: fmt(float(probe_mean.loc[(c, f), s]), "s3") for s in sign_control.STAGES}}
                  for c in order for f in sign_control.FRAME_SETS]
    exp_rows = [{"config": e["config"], "R² เฉลี่ย": fmt(e["R² เฉลี่ย"], "s3"), "corr เฉลี่ย": fmt(e["corr เฉลี่ย"], "s3"),
                 "สรุป (ตามวิธีอ่านผลที่ประกาศไว้)": e["สรุป"]} for e in expect]

    md += ["# ส่วนที่ 2 — sign control (DECISIONS B24)", "",
           "target = ret_close ของวันสุดท้ายในหน้าต่าง (มีเครื่องหมาย) = ln(C_{t−1}/C_{t−2}) · เป็นฟังก์ชันของ input ล้วน · 3 seed เฉลี่ย · "
           "R² เทียบกับค่าเฉลี่ยของ target · **คาดการณ์ (ผู้ใช้): log-power R² ≈ 0 · ถ้า complex ต่ำด้วย ให้ตรวจว่าข้อมูลหายตรงไหน แล้วรายงานโดยไม่แก้เอง**", "",
           "## 2.1 สรุปตามวิธีอ่านผลที่ประกาศไว้", "", md_table(pd.DataFrame(exp_rows)), "",
           "## 2.2 R² แยกตาม 6 evaluation", "", md_table(pivot_by_config(sign_rows, "R2", "s3", order)), "",
           "## 2.3 corr แยกตาม 6 evaluation", "", md_table(pivot_by_config(sign_rows, "corr", "s3", order)), "",
           "## 2.4 linear probe — ข้อมูลหายตรงไหน (R² เฉลี่ย 6 evaluation)", "",
           "ridge (α ปรับด้วย LOO บนแถว train) ทำนาย target จากภาพ ณ แต่ละขั้น · `all` = ทุก frame · `last` = frame สุดท้ายอย่างเดียว · "
           "ถ้า probe สูงแต่ CNN ต่ำ = ข้อมูลยังอยู่แต่ CNN/การเทรนดึงออกมาไม่ได้ · ถ้า probe ตกที่ขั้นใด = ข้อมูลหายที่ขั้นนั้น", "",
           md_table(pd.DataFrame(probe_rows)), "",
           f"เวลา: placebo {t_placebo / 60:.1f} นาที · sign control {t_sign / 60:.1f} นาที · รวม {(time.time() - t_all) / 60:.1f} นาที (jobs {args.jobs})", ""]
    (PLACEBO_DIR / "placebo_and_sign_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (SIGN_DIR / "sign_summary.md").write_text("\n".join(md[md.index("# ส่วนที่ 2 — sign control (DECISIONS B24)"):]) + "\n", encoding="utf-8")

    # ======================= พิมพ์สรุป =======================
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 110 + "\nPLACEBO — ผลต่อ shuffle\n" + "=" * 110)
    print(pd.DataFrame(per_rows).to_string(index=False))
    print(f"\nผู้ชนะเฉลี่ยต่อ shuffle (หลัก) = {S['mean_n_factor_winners']:.2f} → {S['decision_primary']}   |   รอง = {S['mean_n_candidates_beating']:.2f} → {S['decision_secondary']}   "
          f"| ตรงกัน: {S['primary_and_secondary_agree']}")
    print(f"เกณฑ์ B22 (สูงสุดของ ΔIC ข้าม {S['n_delta_values']} ค่า) = {S['b22_threshold']:+.4f}")
    print("\n" + "=" * 110 + "\nPLACEBO — ΔIC ของผู้สมัคร\n" + "=" * 110)
    print(pd.DataFrame(delta_rows).to_string(index=False))
    print("\n" + "=" * 110 + "\nSIGN CONTROL — สรุป\n" + "=" * 110)
    print(pd.DataFrame(exp_rows).to_string(index=False))
    print("\nR² แยก evaluation:")
    print(pivot_by_config(sign_rows, "R2", "s3", order).to_string(index=False))
    print("\ncorr แยก evaluation:")
    print(pivot_by_config(sign_rows, "corr", "s3", order).to_string(index=False))
    print("\nlinear probe R² (เฉลี่ย 6 evaluation):")
    print(pd.DataFrame(probe_rows).to_string(index=False))
    print(f"\nเขียน {PLACEBO_DIR / 'placebo_and_sign_summary.md'} · รวม {(time.time() - t_all) / 60:.1f} นาที")


if __name__ == "__main__":
    main()
