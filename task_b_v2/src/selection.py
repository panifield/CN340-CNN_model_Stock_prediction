"""
selection.py — กติกาเลือกผู้ชนะด้วย IC และ noise ของ seed (DECISIONS ข้อ B14, B16, B17)
======================================================================================
ไฟล์นี้ทำอะไร: เป็นที่เดียวที่ตัดสินว่า "config ไหนชนะ base" · แยกเป็นฟังก์ชันล้วน (รับตาราง คืนตาราง) เพื่อให้ทดสอบได้ตรง ๆ

กติกา (ผู้ใช้ตัดสิน B16 · นิยาม noise คือ B17)
    1. เกณฑ์เลือก = **mean IC (Pearson) ของ 6 evaluation** (คำทำนายเฉลี่ย 3 seed)
    2. noise ของ stage = sqrt(ค่าเฉลี่ยของ σ² ข้ามทุก config ใน stage รวม base)
       โดย σ ของ config = std (ddof = 1) ข้าม seed ของ "IC เฉลี่ย 6 evaluation" ที่คิดจากคำทำนายของแต่ละ seed เดี่ยว ๆ
    3. config ชนะ base ก็ต่อเมื่อ  IC(config) − IC(base) > noise ของ stage
    4. ผู้ชนะของปัจจัยหนึ่ง = config ที่ IC สูงสุดในปัจจัยนั้นที่ชนะ base · ไม่มี → คงค่า base
    5. (B14) config รวมผู้ชนะ "แพ้ base" = IC ต่ำกว่า base แม้นิดเดียว → ต้องหยุดถามผู้ใช้
    6. (B22) Stage 2–5: ต้องผ่านทั้งข้อ 3 และ ΔIC > floor (เกณฑ์ B22 = ค่าสูงสุดของ ΔIC ใน placebo) → ΔIC > max(noise, floor)

ทำไมต้องมี: การเลือกจากความต่างที่เล็กกว่า noise = เลือกจาก noise · กติกานี้บังคับให้ต้องชนะเกิน noise ก่อนจึงจะเปลี่ยนจาก base
และต้องเป็นฟังก์ชันเดียวที่ทุก Stage ใช้ร่วมกัน ไม่งั้นแต่ละ Stage จะเลือกด้วยเกณฑ์ที่ต่างกันโดยไม่รู้ตัว

หมายเหตุที่ต้องรู้ (ดู B16): noise จากการสุ่ม "วัน" ของช่วงประเมิน (≈ 1/√1596 ≈ 0.025 ต่อ mean IC) เป็นคนละเรื่องกับ noise ของ seed
และ **ไม่อยู่ในกติกานี้** — `sampling_se_reference()` มีไว้พิมพ์เป็นตัวเลขอ้างอิงในรายงานเท่านั้น
"""

import numpy as np
import pandas as pd


def pooled_seed_noise(scores):
    """noise ของ stage = sqrt(ค่าเฉลี่ยของ ic_seed_std² ข้ามทุก config) · NaN ถูกข้าม · ไม่มี config ที่มี σ เลย → NaN"""
    s = pd.to_numeric(scores["ic_seed_std"], errors="coerce").to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    return float(np.sqrt(np.mean(s ** 2))) if len(s) else float("nan")


def sampling_se_reference(n_days=6 * 266, per_eval_days=266):
    """
    ตัวเลขอ้างอิงเท่านั้น (ไม่อยู่ในกติกา): ความคลาดเคลื่อนของ IC จากการมีวันประเมินจำกัด
    IC ของหนึ่ง evaluation มี SE ≈ 1/√266 ≈ 0.061 · ค่าเฉลี่ยของ 6 evaluation ≈ 0.061/√6 ≈ 0.025
    """
    return float(1 / np.sqrt(per_eval_days) / np.sqrt(n_days // per_eval_days))


def compare_to_base(scores, base_name, names, noise, floor=None):
    """
    ตารางเทียบ config ใน `names` กับ base : IC, ผลต่าง, เกณฑ์ที่ใช้, และว่าชนะหรือไม่

    scores : DataFrame หนึ่งแถวต่อ config (คอลัมน์ config, mean_IC, ...) · คืน DataFrame เรียงตาม IC จากมากไปน้อย
    floor  : เกณฑ์ B22 (None = Stage 1 ที่ใช้แค่ noise) · ถ้าให้มา ต้องชนะเกิน max(noise, floor) — ผ่านทั้งสองเกณฑ์
    """
    sc = scores.set_index("config")
    if base_name not in sc.index:
        raise KeyError(f"ไม่พบ base ({base_name!r}) ในตารางคะแนน")
    base_ic = float(sc.loc[base_name, "mean_IC"])
    threshold = noise if floor is None else max(noise, float(floor))
    rows = []
    for n in names:
        ic = float(sc.loc[n, "mean_IC"])
        delta = ic - base_ic
        eligible = bool(np.isfinite(ic) and np.isfinite(noise))
        rows.append({"config": n, "mean_IC": ic, "delta_vs_base": delta, "noise": noise,
                     "floor": np.nan if floor is None else float(floor), "threshold": threshold,
                     "beats_base": bool(eligible and delta > threshold),
                     "note": "IC เป็น NaN — ไม่มีสิทธิ์ชนะ" if not np.isfinite(ic) else
                             ("noise เป็น NaN — ไม่มีสิทธิ์ชนะ" if not np.isfinite(noise) else "")})
    return pd.DataFrame(rows).sort_values("mean_IC", ascending=False, na_position="last").reset_index(drop=True)


def pick_factor_winner(scores, base_name, names, noise, floor=None):
    """
    ผู้ชนะของ "ปัจจัย" หนึ่ง (เช่น window) : คืน (ชื่อผู้ชนะ, ตารางเทียบ)
    ผู้ชนะ = config ที่ IC สูงสุดในกลุ่ม `names` ที่ชนะ base เกิน max(noise, floor) · ไม่มี → base
    """
    table = compare_to_base(scores, base_name, names, noise, floor)
    winners = table[table["beats_base"]]
    return (winners.iloc[0]["config"] if len(winners) else base_name), table


def combined_loses_to_base(scores, combined_name, base_name):
    """
    กติกา B14 : config รวมผู้ชนะ "แพ้" base = IC ต่ำกว่า base **แม้นิดเดียว** (ไม่มีช่วงเผื่อ)
    คืน (แพ้ไหม, ผลต่าง IC) · IC เป็น NaN ถือว่าแพ้ (ต้องหยุดถามผู้ใช้ ไม่ปล่อยผ่านเงียบ ๆ)
    """
    sc = scores.set_index("config")
    d = float(sc.loc[combined_name, "mean_IC"]) - float(sc.loc[base_name, "mean_IC"])
    return (not np.isfinite(d)) or d < 0, d


def assert_scores_have_seed_noise(scores):
    """ก่อนใช้กติกา noise ต้องมี ic_seed_std (จึงต้องรันด้วยโค้ดที่เก็บ IC รายseed) — ไม่งั้นหยุดทันที ไม่เดา"""
    if "ic_seed_std" not in scores.columns or not np.isfinite(pd.to_numeric(scores["ic_seed_std"], errors="coerce")).any():
        raise RuntimeError("ไม่มี ic_seed_std ในตารางคะแนน — ต้องรันด้วย seeds ตั้งแต่ 2 ตัวขึ้นไปและเก็บ IC รายseed")
