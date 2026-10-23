"""
placebo.py — placebo test ของกติกาเลือกผู้ชนะ (DECISIONS B21, B22, B23)
=====================================================================
ไฟล์นี้ทำอะไร
    รัน Stage 1 ซ้ำทั้ง 9 config โดยเทรนด้วย y ที่ถูกสลับใน train (ไม่มีสัญญาณให้เรียนเลย) แล้วใช้กติกาเลือกผู้ชนะ B16/B17
    ตัวเดิมกับผลนั้น — เพื่อวัดว่า "เมื่อไม่มีสัญญาณจริง กติกานี้ยังเลือกผู้ชนะได้กี่ตัว" และได้ ΔIC สูงสุดที่ noise ล้วนก็ให้ได้

ทำไมต้องมี
    Stage 1 จริงเลือกผู้ชนะ 3 ปัจจัย แต่ผู้ชนะมี IC (+0.026…+0.038) ตรงกับค่าสูงสุดที่ noise ล้วนน่าจะให้ได้ (≈ +0.034)
    กติกา B17 คุมแค่ noise ของ seed ไม่คุม noise จากการสุ่มวันประเมิน · placebo คือการวัดตรง ๆ ว่าความต่างนี้ใหญ่แค่ไหน

กติกาที่ผู้ใช้ประกาศก่อนรัน (B21) และข้อตีความที่ Claude ประกาศไว้ก่อนรัน (B23)
    * 3 shuffle (seed 0, 1, 2) × 9 config × 6 evaluation × 3 model seeds
    * จำนวน "ผู้ชนะ" เฉลี่ยต่อ shuffle ≥ 1 → Stage 1 ไม่มีผู้ชนะที่เชื่อได้ ใช้ base ต่อ · < 1 → ใช้ config รวม
    * "ผู้ชนะ" (หลัก) = จำนวนปัจจัยที่ B17 เลือกผู้ชนะ ≠ base ได้ (สูงสุด 4) · จำนวน config ที่ชนะ base เกิน noise (สูงสุด 7) เป็นตัวเลขรอง
    * เกณฑ์ B22 = ค่าสูงสุดของ ΔIC ข้าม (7 ผู้สมัคร × 3 shuffle) = 21 ค่า (3 ค่าต่อ shuffle ไม่พอประมาณเปอร์เซ็นไทล์ 95)
"""

import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import controls, data, evaluate, paths, runner, selection, train
from src.settings import SEARCH_SEEDS, VAL_END

SHUFFLE_SEEDS = (0, 1, 2)
PRIMARY_RULE_THRESHOLD = 1.0        # จำนวนผู้ชนะเฉลี่ยต่อ shuffle ≥ นี้ → ใช้ base ต่อ (DECISIONS B21 ข้อ 2)


def placebo_unit(args):
    """
    หนึ่ง unit ของ placebo : (config, หุ้น, fold, shuffle) → แถวผล (มี IC รวมและ IC รายseed)
    ต้องเป็นฟังก์ชันระดับบนสุดเพื่อให้ ProcessPoolExecutor pickle ได้ · ผลถูก cache ตามลายนิ้วมือของ unit
    """
    cfg, ticker, fold_k, shuffle_seed, seeds, zone_end, cache_dir = args
    key = runner.unit_key(cfg, ticker, fold_k, seeds, zone_end, tag=f"placebo-shuffle{shuffle_seed}")
    cache = None if cache_dir is None else cache_dir / f"{key}.pkl"
    if cache is not None and cache.exists():
        with open(cache, "rb") as f:
            row = pickle.load(f)
        row["from_cache"] = True
        return row

    t0 = time.time()
    ds = data.build_dataset(ticker, zone_end)
    fold = next(f for f in data.make_folds(ds.dates) if f.k == fold_k)
    prepared = data.fold_inputs(ds, cfg, fold)
    extra = train.standardized_extra(ds, prepared) if cfg.needs_features else None
    y_over = controls.shuffled_targets(prepared, shuffle_seed)          # เหมือน negative control ทุกประการ
    pred, hists, seed_preds = train.run_fold(cfg, prepared, seeds=seeds, extra=extra, y_override=y_over,
                                             return_hist=True, return_seed_preds=True)
    row = evaluate.evaluation_row("CNN-ANN-placebo", cfg.name, ticker, fold_k,
                                  prepared["y_eval"], pred, prepared["prev_close_eval"],
                                  shuffle=shuffle_seed, seconds=round(time.time() - t0, 1), n_seeds=len(seeds),
                                  epochs_mean=float(np.mean([h["epochs"] for h in hists])),      # B2 (Stage 6): ธงชนเพดาน epoch
                                  n_hit_cap=int(sum(not h["stopped_early"] for h in hists)))
    for k, sp in zip(seeds, seed_preds):                                  # IC รายseed → noise ของ seed (B17)
        m = evaluate.metrics(prepared["y_eval"], sp, prepared["prev_close_eval"])
        row[f"ic_s{k}"] = m["IC"]
        row[f"relMAE_s{k}"] = m["relMAE"]
    row["from_cache"] = False
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(row, f)
    return row


def run_placebo(configs, stage_dir, *, shuffle_seeds=SHUFFLE_SEEDS, seeds=SEARCH_SEEDS, jobs=1,
                zone_end=VAL_END, verbose=True, units=None):
    """รัน placebo ทุก config × shuffle × หุ้น × fold → DataFrame (เขียน evals.csv ลง stage_dir ด้วย)
    units : (ไม่บังคับ) list ของ (ticker, fold) · None = ครบทุกหุ้น × ทุก fold · ใช้เฉพาะ dry-run ของ Stage 6 (B38)"""
    from src.settings import TICKERS
    stage = paths.assert_output_dir(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    cache_dir = stage / "cache"
    n_folds = len(data.make_folds(data.build_dataset(TICKERS[0], zone_end).dates))
    pairs = [(t, k) for t in TICKERS for k in range(1, n_folds + 1)] if units is None else list(units)
    tasks = [(cfg, t, k, s, tuple(seeds), zone_end, cache_dir)
             for s in shuffle_seeds for cfg in configs for t, k in pairs]
    if verbose:
        print(f"[placebo] {len(configs)} config × {len(shuffle_seeds)} shuffle × {len(pairs)} evaluation "
              f"= {len(tasks)} unit · model seeds {tuple(seeds)} · jobs {jobs}")
    rows, t0 = [], time.time()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, r in enumerate(pool.map(placebo_unit, tasks), 1):
                rows.append(r)
                if verbose and (i % 12 == 0 or i == len(tasks)):
                    print(f"  [{i:3d}/{len(tasks)}] เหลือ ~{(len(tasks) - i) * (time.time() - t0) / i / 60:.0f} นาที")
    else:
        for i, a in enumerate(tasks, 1):
            rows.append(placebo_unit(a))
            if verbose and (i % 12 == 0 or i == len(tasks)):
                print(f"  [{i:3d}/{len(tasks)}]")
    evals = pd.DataFrame(rows)
    evals.to_csv(stage / "evals.csv", index=False)
    return evals


# ---------------------------------------------------------------
# วิเคราะห์ผล (ฟังก์ชันล้วน ทดสอบได้)
# ---------------------------------------------------------------
def score_shuffle(evals_s, config_names):
    """ตารางคะแนนของหนึ่ง shuffle (หนึ่งแถวต่อ config · ต้องครบ 6 evaluation ต่อ config)"""
    return pd.DataFrame([{"config": n, **evaluate.score_config(evals_s[evals_s["config"] == n].to_dict("records"))}
                         for n in config_names])


def analyze(evals, config_names, factors, shuffle_seeds=SHUFFLE_SEEDS, base="base"):
    """
    ใช้กติกา B16/B17 ตัวเดิมกับผลของแต่ละ shuffle แล้วสรุป

    factors : {ปัจจัย: {ชื่อ config ผู้สมัคร: ...}} (ใช้เฉพาะชื่อ) · detrend ไม่อยู่ใน factors จึงไม่นับเป็นผู้สมัคร (B19/B23)
    คืน dict: per_shuffle (DataFrame), deltas (DataFrame ผู้สมัคร × shuffle), summary
    """
    per_rows, delta_rows, scores_by_shuffle = [], [], {}
    for s in shuffle_seeds:
        scores = score_shuffle(evals[evals["shuffle"] == s], config_names)
        scores_by_shuffle[s] = scores
        selection.assert_scores_have_seed_noise(scores)
        noise = selection.pooled_seed_noise(scores)
        base_ic = float(scores.set_index("config").loc[base, "mean_IC"])
        winners, n_cand = {}, 0
        for factor, options in factors.items():
            w, tbl = selection.pick_factor_winner(scores, base, list(options), noise)
            winners[factor] = w
            n_cand += int(tbl["beats_base"].sum())
            for r in tbl.itertuples():
                delta_rows.append({"shuffle": s, "factor": factor, "config": r.config, "mean_IC": r.mean_IC,
                                   "delta_vs_base": r.delta_vs_base, "beats_base": bool(r.beats_base), "noise": noise})
        d_s = [r["delta_vs_base"] for r in delta_rows if r["shuffle"] == s]
        per_rows.append({"shuffle": s, "noise": noise, "base_IC": base_ic,
                         "n_factor_winners": int(sum(w != base for w in winners.values())),
                         "n_candidates_beating": n_cand, "max_delta_IC": float(np.max(d_s)),
                         "winners": ", ".join(f"{f}={w}" for f, w in winners.items() if w != base) or "(ไม่มี)"})
    per = pd.DataFrame(per_rows)
    deltas = pd.DataFrame(delta_rows)
    mean_primary = float(per["n_factor_winners"].mean())
    mean_secondary = float(per["n_candidates_beating"].mean())
    summary = {
        "mean_n_factor_winners": mean_primary,                         # ตัวเลขหลักของกติกา B21
        "mean_n_candidates_beating": mean_secondary,                   # ตัวเลขรอง (B23)
        "decision_primary": decide(mean_primary),
        "decision_secondary": decide(mean_secondary),
        "primary_and_secondary_agree": decide(mean_primary) == decide(mean_secondary),
        "b22_threshold": b22_threshold(deltas),
        "n_delta_values": int(len(deltas)),
        "max_delta_per_shuffle": [float(x) for x in per["max_delta_IC"]],
        "mean_noise": float(per["noise"].mean()),
    }
    return {"per_shuffle": per, "deltas": deltas, "summary": summary, "scores": scores_by_shuffle}


def decide(mean_winners):
    """กติกา B21 ข้อ (2)/(3) : เฉลี่ย ≥ 1 → base · < 1 → config รวม"""
    return "base" if mean_winners >= PRIMARY_RULE_THRESHOLD else "combined"


def b22_threshold(deltas):
    """
    เกณฑ์ B22 : ค่าสูงสุดของ ΔIC ข้ามผู้สมัครทุกตัวทุก shuffle (DECISIONS B23 ข้อ 4)
    ใช้ค่าสูงสุดแทนเปอร์เซ็นไทล์ 95 เพราะ "ΔIC สูงสุดต่อ shuffle" มีแค่ 3 ค่า ประมาณ P95 ไม่ได้ (เงื่อนไขสำรองที่ผู้ใช้ระบุไว้)
    """
    return float(np.max(deltas["delta_vs_base"].to_numpy(dtype=float)))


def check_against_negative_control(evals, neg_path):
    """
    ตรวจความสอดคล้อง (B23 ข้อ 6) : base × shuffle 0 ต้องได้ IC และ relMAE เท่ากับ negative control 3 seeds ของ C3 ทุกบิต
    คืน (ตรงทุกบิตไหม, ข้อความ)
    """
    if not neg_path.exists():
        return None, "ไม่พบไฟล์ negative control ของ C3"
    neg = pd.read_csv(neg_path, float_precision="round_trip").set_index(["ticker", "fold"])
    mine = evals[(evals["config"] == "base") & (evals["shuffle"] == 0)].set_index(["ticker", "fold"])
    if len(mine) != len(neg):
        return False, f"จำนวน evaluation ไม่เท่ากัน ({len(mine)} vs {len(neg)})"
    d_ic = float(np.max(np.abs(mine.loc[neg.index, "IC"].to_numpy() - neg["IC"].to_numpy())))
    d_rel = float(np.max(np.abs(mine.loc[neg.index, "relMAE"].to_numpy() - neg["relMAE"].to_numpy())))
    return (d_ic == 0.0 and d_rel == 0.0), f"ต่างกันสูงสุด IC {d_ic:.3e} · relMAE {d_rel:.3e}"


def analyze_pair(evals, a, b, shuffle_seeds):
    """
    placebo ของ "คู่" config (DECISIONS B28) : ΔIC_s = mean IC ของ a − ของ b ที่เทรนด้วย y ที่สลับชุดเดียวกัน (shuffle s)
    threshold_pair = ค่าสูงสุดของ ΔIC_s ข้าม shuffle (ผู้ใช้กำหนด: ค่าสูงสุดจาก 5 shuffle · ไม่ใช่เปอร์เซ็นไทล์)
    """
    rows = []
    for s in shuffle_seeds:
        ev = evals[evals["shuffle"] == s]
        sa = evaluate.score_config(ev[ev["config"] == a].to_dict("records"))
        sb = evaluate.score_config(ev[ev["config"] == b].to_dict("records"))
        rows.append({"shuffle": s, f"IC_{a}": sa["mean_IC"], f"IC_{b}": sb["mean_IC"],
                     "delta_ic": sa["mean_IC"] - sb["mean_IC"],
                     f"sigma_{a}": sa.get("ic_seed_std", np.nan), f"sigma_{b}": sb.get("ic_seed_std", np.nan)})
    df = pd.DataFrame(rows)
    return {"per_shuffle": df, "threshold_pair": float(df["delta_ic"].max()), "mean_delta": float(df["delta_ic"].mean()),
            "n_shuffles": int(len(df))}
