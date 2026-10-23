"""
hpo.py — Stage 6 (HPO ของ track 1): ออกแบบ config แบบสุ่มสมดุล + กติกาตัดสินผู้ชนะ (DECISIONS B38)
=================================================================================================
ไฟล์นี้ทำอะไร: ฟังก์ชันล้วน (รับข้อมูล คืนข้อมูล · ไม่เทรน) 3 กลุ่ม
    1. `design_configs()`  : สร้าง 24 config ผู้สมัครแบบ balanced random design ด้วย seed คงที่ 38 + base เป็น config ที่ 0 = 25 config
    2. `placebo_stats()`    : ต่อรอบ placebo s : T_s = max ข้าม 24 ผู้สมัครของ (IC_ผู้สมัคร − IC_base) ภายในรอบนั้น · T_hpo = max_s T_s
    3. `judge_winner()`    : เกณฑ์ชนะ A6 ข้อ (1)–(4) ของ config ที่ mean IC สูงสุด 1 ตัว (ไม่ไล่อันดับ 2, 3)

ทำไมต้องมี
    ค้นหา 24 config แล้วหยิบตัวที่ดีที่สุด = เลือกค่าสูงสุดจาก 24 ค่าที่มี noise · ผลชนะโดย noise ล้วนมีโอกาสสูงมาก
    จึงต้องมีเกณฑ์จาก placebo ของ "การค้นหาทั้งชุด" (ทั้ง 25 config ที่ y ถูกสลับ) แทนที่จะดูแค่ noise ของ seed
    และต้องล็อกวิธีออกแบบ/วิธีตัดสินเป็นโค้ดที่ผู้ใช้ commit ก่อนรัน (ห้ามแก้หลังเห็นผล)

การออกแบบ (A3)
    แต่ละพารามิเตอร์ 3 ระดับ → แต่ละระดับ 8 ครั้งใน 24 config · noise_std 4 ระดับ → ระดับละ 6 ครั้ง
    สร้างแต่ละคอลัมน์ด้วยการสุ่มลำดับแยกกัน (rng.permutation ของ [ระดับ 0 ×8, ระดับ 1 ×8, ระดับ 2 ×8])
    กติกาตรวจ (ตายตัว): ห้ามซ้ำกันเอง · ห้ามซ้ำ base · 1 <= hop < nperseg · hop <= nperseg/2
    ไม่ผ่าน → "สุ่มทั้ง 9 คอลัมน์ใหม่" ด้วย seed ถัดไป (38, 39, 40, …) · เก็บจำนวนครั้งไว้ใน design_log (ครั้งที่ใช้ได้ = ครั้งแรกที่ผ่านครบ)
    (การสุ่มใหม่ทั้งชุดทำให้ความสมดุล 8/8/8 และ 6/6/6/6 ยังเป็นจริงเสมอ — ถ้าแก้ทีละแถวความสมดุลจะพัง)
"""

from dataclasses import fields, replace

import numpy as np
import pandas as pd

from src import models
from src.configs import BASE, Config

DESIGN_SEED = 38                 # seed ของครั้งแรก · ครั้งที่ a ใช้ DESIGN_SEED + a
N_CANDIDATES = 24
MAX_ATTEMPTS = 1000

# ค่าที่ค้นหา (A2) · ลำดับคีย์ = ลำดับที่สร้างคอลัมน์ (ห้ามสลับ เพราะ rng ถูกใช้ต่อเนื่อง)
GRID = {
    "lr": (3e-4, 1e-3, 3e-3),
    "batch": (16, 32, 64),
    "dropout": (0.0, 0.3, 0.5),
    "weight_decay": (1e-4, 1e-3, 1e-2),
    "filters": ((4, 8), (8, 16), (16, 32)),
    "ann_hidden": ((16,), (32,), (64,)),
    "nperseg": (8, 16, 32),
    "hop": (1, 2, 4),
    "noise_std": (0.0, 0.1, 0.3, 0.5),
}
PARAMS = list(GRID)

# เกณฑ์ชนะ (A6) และ placebo (A5)
B22_FLOOR = 0.0597                       # DECISIONS B22 (stage1_final_decision.json: b22_threshold_stage2_to_5)
PLACEBO_SHUFFLES = (100, 101, 102, 103, 104)      # 5 รอบ · ถ้าเวลาเกิน 7 ชม. ลดเป็น (100, 101, 102) — ตัดสินใน B38 ก่อนรันจริง
CONFIRM_SEEDS = (10, 11, 12)             # seed ใหม่สำหรับยืนยันผู้ชนะ (A6 ข้อ 2)
MIN_LAST_DAY_WEIGHT = 0.5                # A8 (c): น้ำหนักวันสุดท้ายของหน้าต่าง / ค่าสูงสุด ต้อง ≥ นี้ ("ไม่ใกล้ 0")


def _key(params):
    """ลายนิ้วมือของชุดค่าพารามิเตอร์ (ไว้ตรวจซ้ำ) — เทียบ tuple ตรง ๆ"""
    return tuple(params[p] for p in PARAMS)


def base_params():
    return {p: getattr(BASE, p) for p in PARAMS}


def _draw(seed):
    """สุ่ม 9 คอลัมน์แบบสมดุลด้วย rng ตัวเดียว (ตามลำดับ GRID) → list ของ dict ยาว N_CANDIDATES"""
    rng = np.random.default_rng(seed)
    cols = {}
    for p, levels in GRID.items():
        if N_CANDIDATES % len(levels):
            raise ValueError(f"{p}: {N_CANDIDATES} หารด้วย {len(levels)} ไม่ลงตัว — ออกแบบสมดุลไม่ได้")
        idx = np.repeat(np.arange(len(levels)), N_CANDIDATES // len(levels))
        cols[p] = [levels[i] for i in rng.permutation(idx)]
    return [{p: cols[p][i] for p in PARAMS} for i in range(N_CANDIDATES)]


def violations(rows):
    """เหตุผลที่การสุ่มครั้งนี้ใช้ไม่ได้ (list ว่าง = ผ่านทุกข้อ)"""
    out, seen = [], {}
    b = _key(base_params())
    for i, r in enumerate(rows, 1):
        if not (1 <= r["hop"] < r["nperseg"] and 2 * r["hop"] <= r["nperseg"]):
            out.append(f"hpo{i:02d}: hop={r['hop']} nperseg={r['nperseg']} ไม่ผ่าน 1<=hop<nperseg และ hop<=nperseg/2")
        k = _key(r)
        if k == b:
            out.append(f"hpo{i:02d}: ซ้ำกับ base")
        if k in seen:
            out.append(f"hpo{i:02d}: ซ้ำกับ hpo{seen[k]:02d}")
        seen.setdefault(k, i)
    return out


def design_configs(seed=DESIGN_SEED):
    """
    คืน (configs, info) · configs[0] = BASE · configs[1..24] = hpo01..hpo24
    info : {"seed_first", "seed_used", "attempts", "redraws", "redraw_reasons"}  (บันทึกลง design_log.json)
    """
    reasons = []
    for a in range(MAX_ATTEMPTS):
        rows = _draw(seed + a)
        bad = violations(rows)
        if not bad:
            cands = [replace(BASE, name=f"hpo{i:02d}", **r) for i, r in enumerate(rows, 1)]
            info = {"seed_first": seed, "seed_used": seed + a, "attempts": a + 1, "redraws": a,
                    "redraw_reasons": reasons}
            return [BASE] + cands, info
        reasons.append({"seed": seed + a, "reasons": bad})
    raise RuntimeError(f"สุ่มไม่ผ่านเงื่อนไขใน {MAX_ATTEMPTS} ครั้ง — หยุดและรายงานผู้ใช้")


def level_counts(configs):
    """จำนวนครั้งที่แต่ละระดับปรากฏใน 24 ผู้สมัคร (ไม่นับ base) — ต้องเป็น 8/8/8 และ 6/6/6/6"""
    out = {}
    for p, levels in GRID.items():
        vals = [getattr(c, p) for c in configs if c.name != "base"]
        out[p] = {str(l): vals.count(l) for l in levels}
    return out


def n_params(cfg):
    return models.count_params(models.build(cfg, 0))


def configs_table(configs):
    """ตาราง 25 แถวของ configs.csv: ลำดับ · ชื่อ · 9 พารามิเตอร์ที่ค้นหา · image_shape · จำนวนพารามิเตอร์ · field ที่คงที่"""
    rows = []
    for i, c in enumerate(configs):
        r = c.to_row()
        searched = {p: r[p] for p in PARAMS}
        fixed = {k: v for k, v in r.items() if k not in PARAMS and k not in ("name", "image_shape")}
        rows.append({"idx": i, "name": c.name, **searched, "image_shape": r["image_shape"],
                     "n_params": n_params(c), **fixed})
    return pd.DataFrame(rows)


def config_key_without_name(cfg):
    """ลายนิ้วมือของ config (ทุก field ยกเว้นชื่อ) — แบบเดียวกับที่ experiment_log ใช้ตัดซ้ำ"""
    return tuple(str(getattr(cfg, f.name)) for f in fields(Config) if f.name != "name")


# ---------------------------------------------------------------
# placebo ของการค้นหาทั้งชุด (A5)
# ---------------------------------------------------------------
def mean_ic_table(evals, config_names, group_col=None):
    """mean IC ของ 6 evaluation ต่อ config (ต่อรอบ placebo ถ้าให้ group_col) · ต้องครบทุก config และ IC finite — ไม่งั้นหยุด (B3)"""
    keys = ([group_col] if group_col else []) + ["config"]
    g = evals.groupby(keys)["IC"].agg(["mean", "size", lambda s: int(np.isfinite(s).sum())]).reset_index()      # size นับ NaN ด้วย (count/mean ข้าม NaN เงียบ ๆ)
    g.columns = keys + ["mean_IC", "n", "n_finite"]
    if (g["n_finite"] != g["n"]).any():
        raise RuntimeError("พบ IC ที่ไม่ finite (NaN) — หยุดและรายงาน ห้ามแก้แล้วรันต่อเงียบ ๆ (B3)")
    if group_col:
        return g.pivot(index="config", columns=group_col, values="mean_IC").loc[config_names]
    return g.set_index("config").loc[config_names, "mean_IC"]


def placebo_stats(evals_placebo, config_names, base="base"):
    """
    evals_placebo : evals.csv ของ placebo (คอลัมน์ config, shuffle, IC, ...)
    คืน dict: ic (DataFrame config × shuffle), t_s (Series ต่อรอบ), t_hpo (float), delta (DataFrame ผู้สมัคร × shuffle)
    T_s = max ข้ามผู้สมัครทุกตัว (ทุก config ยกเว้น base) ของ (IC_ผู้สมัคร − IC_base) ในรอบ s · T_hpo = max_s T_s
    """
    ic = mean_ic_table(evals_placebo, config_names, group_col="shuffle")
    cand = [c for c in config_names if c != base]
    delta = ic.loc[cand].sub(ic.loc[base], axis=1)
    t_s = delta.max(axis=0)
    return {"ic": ic, "delta": delta, "t_s": t_s, "t_hpo": float(t_s.max())}


def level_means(table, ic_by_config, exclude=("base",)):
    """
    descriptive เท่านั้น (ห้ามใช้เลือก config): mean IC เฉลี่ยตามระดับของแต่ละพารามิเตอร์ (เฉพาะ 24 ผู้สมัคร)
    ic_by_config : Series ดัชนี = ชื่อ config  ·  table : configs_table()
    """
    t = table[~table["name"].isin(exclude)].set_index("name")
    rows = []
    for p, levels in GRID.items():
        for lv in levels:
            lv_txt = "-".join(map(str, lv)) if isinstance(lv, tuple) else str(lv)
            names = t.index[t[p].astype(str) == lv_txt]
            rows.append({"param": p, "level": lv_txt, "n_config": len(names),
                         "mean_IC": float(ic_by_config.loc[names].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# เกณฑ์ชนะ (A6) — ตรวจเฉพาะอันดับ 1
# ---------------------------------------------------------------
def judge_winner(scores, evals, confirm_scores, confirm_evals, noise, t_hpo, base="base", floor=B22_FLOOR):
    """
    scores/evals                  : ผลค้นหา (seeds 0–2) — scores มีคอลัมน์ config, mean_IC, mean_relMAE
    confirm_scores/confirm_evals  : ผลยืนยัน (seeds 10–12) ของผู้สมัครอันดับ 1 และ base (None = ไม่ได้รันยืนยัน → ข้อ (2) ไม่ผ่าน)
    คืน dict ที่มี top (ชื่อผู้สมัครอันดับ 1), ผลทีละข้อ และ has_winner

    ตรวจเฉพาะผู้สมัครที่ mean IC สูงสุด 1 ตัว (ห้ามไล่อันดับ 2, 3) · ถ้า base เองอยู่อันดับ 1 ΔIC ของผู้สมัครจะติดลบและตกข้อ (1) เอง

    (1) ΔIC เทียบ base > max(noise, floor, T_hpo)
    (2) ยืนยันด้วย seed ใหม่ ΔIC > เกณฑ์ข้อ (1) ตัวเดิม
    (3) IC>0 อย่างน้อย (n−1) จาก n evaluation (n=6 → 5/6) และตัด evaluation ที่ ΔIC สูงสุดออกแล้ว ΔIC เฉลี่ยของที่เหลือ > max(noise, floor)
    (4) relMAE (return · ก่อนปรับขนาด) ≤ base
    ข้อ (3), (4) ตัดสินจากผลค้นหา (seeds 0–2) · ค่าเดียวกันจากผลยืนยันรายงานเพิ่มเป็นข้อมูล (ไม่ใช้ตัดสิน)
    """
    sc = scores.set_index("config")
    top = str(sc.drop(index=base)["mean_IC"].idxmax())            # ผู้สมัครที่ mean IC สูงสุด (ถ้า base อยู่อันดับ 1 รวม ΔIC < 0 → ข้อ (1) ตกเอง)
    thr1 = max(float(noise), float(floor), float(t_hpo))
    thr3 = max(float(noise), float(floor))
    out = {"top": top, "base_is_overall_top": bool(sc["mean_IC"].idxmax() == base),
           "thr1": thr1, "thr3": thr3, "noise": float(noise), "floor": float(floor), "t_hpo": float(t_hpo)}

    d1 = float(sc.loc[top, "mean_IC"] - sc.loc[base, "mean_IC"])
    out["delta_ic"] = d1
    out["c1"] = bool(d1 > thr1)

    if confirm_scores is not None:
        cs = confirm_scores.set_index("config")
        d2 = float(cs.loc[top, "mean_IC"] - cs.loc[base, "mean_IC"])
        out["delta_ic_confirm"] = d2
        out["c2"] = bool(d2 > thr1)
    else:
        out["delta_ic_confirm"], out["c2"] = float("nan"), False

    out.update(per_eval_check(evals, top, base, thr3))
    out["relMAE_top"] = float(sc.loc[top, "mean_relMAE"])
    out["relMAE_base"] = float(sc.loc[base, "mean_relMAE"])
    out["c4"] = bool(out["relMAE_top"] <= out["relMAE_base"])
    if confirm_evals is not None:
        out["confirm_descriptive"] = per_eval_check(confirm_evals, top, base, thr3)
    out["has_winner"] = bool(out["c1"] and out["c2"] and out["c3"] and out["c4"])
    return out


def per_eval_check(evals, top, base, thr3):
    """A6 ข้อ (3) : IC>0 ≥ (n−1)/n และ ΔIC เฉลี่ยหลังตัด evaluation ที่ ΔIC สูงสุดออก > thr3"""
    key = ["ticker", "fold"]
    a = evals[evals["config"] == top].set_index(key)["IC"].sort_index()
    b = evals[evals["config"] == base].set_index(key)["IC"].sort_index()
    d = (a - b)
    n = len(a)
    n_pos = int((a > 0).sum())
    trimmed = d.drop(d.idxmax()) if n > 1 else d.iloc[0:0]
    mean_trim = float(trimmed.mean()) if len(trimmed) else float("nan")
    return {"n_eval": n, "n_ic_positive": n_pos, "min_positive_needed": n - 1,
            "delta_by_eval": {f"{t}-{k}": float(v) for (t, k), v in d.items()},
            "max_delta_eval": "-".join(map(str, d.idxmax())), "delta_mean_trimmed": mean_trim,
            "c3": bool(n_pos >= n - 1 and np.isfinite(mean_trim) and mean_trim > thr3)}


def collapse_counts(evals):
    """
    B39 : จำนวน run ที่คำทำนายคงที่ (IC := 0) ต่อ config · ต้องรันด้วย const_ic_zero (มีคอลัมน์ collapsed และ collapsed_s*)
    คืน DataFrame ดัชนี = config · คอลัมน์ seed_runs / seed_runs_total (ระดับ seed เดี่ยว) และ evals / evals_total (ระดับ evaluation หลังเฉลี่ย seed)
    """
    seed_cols = [c for c in evals.columns if c.startswith("collapsed_s") and c[11:].isdigit()]
    if "collapsed" not in evals.columns or not seed_cols:
        raise RuntimeError("ไม่มีคอลัมน์ collapsed — ต้องรันด้วย const_ic_zero=True (B39)")
    rows = {}
    for name, d in evals.groupby("config", sort=False):
        flags = d[seed_cols].astype(bool).to_numpy()
        rows[name] = {"seed_runs": int(flags.sum()), "seed_runs_total": int(flags.size),
                      "evals": int(d["collapsed"].astype(bool).sum()), "evals_total": int(len(d))}
    return pd.DataFrame.from_dict(rows, orient="index")


def common_levels(table, total_by_config, top_k=5, exclude=("base",)):
    """
    descriptive (B39 ข้อ 3): config ที่ collapse บ่อยที่สุด top_k ตัว (นับรวม ที่ count > 0) มีระดับของพารามิเตอร์อะไรซ้ำกัน
    คืน (DataFrame param · ระดับที่พบมากที่สุด · จำนวนที่พบ · จาก k ตัว, รายชื่อ config ที่ใช้)
    """
    t = table[~table["name"].isin(exclude)].set_index("name")
    s = total_by_config.drop(labels=[n for n in exclude if n in total_by_config.index]).sort_values(ascending=False, kind="stable")
    top = [n for n in s.index[:top_k] if s[n] > 0]
    rows = []
    for p in GRID:
        vc = t.loc[top, p].astype(str).value_counts() if top else pd.Series(dtype=int)
        rows.append({"param": p, "ระดับที่พบมากสุด": vc.index[0] if len(vc) else "—",
                     "พบใน": f"{int(vc.iloc[0]) if len(vc) else 0}/{len(top)}"})
    return pd.DataFrame(rows), top


def epoch_cap_counts(histories):
    """
    B2 : จำนวน run (หน่วย = หนึ่ง unit × หนึ่ง seed) ที่หยุดเพราะชนเพดาน max_epochs ไม่ใช่ early stopping
    histories : dict จาก histories.json {"config|ticker|foldK": [hist ต่อ seed]} → {config: (จำนวนที่ชนเพดาน, จำนวน run ทั้งหมด)}
    """
    out = {}
    for key, hs in histories.items():
        name = key.split("|")[0]
        hit, tot = out.get(name, (0, 0))
        out[name] = (hit + sum(0 if h["stopped_early"] else 1 for h in hs), tot + len(hs))
    return out
