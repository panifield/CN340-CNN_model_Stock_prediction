"""
run_c6.py — C6 ของ next_day: final + F1/F2 + replay (ตาราง log รายวัน) + S1 ตาม LOCK.md หัวข้อ 9 (DECISIONS B41)
================================================================================================================
ไฟล์นี้ทำอะไร
    `run`      : gate → เทรนทุก track + control + ตัวเทียบ ครั้งเดียวต่อช่วง (10 seeds · c จากส่วน early-stop) → ตัววัด → F1 / F2 →
                 replay 17 วัน + ตาราง log รายวันแยกหุ้น → S1 → เขียนผลที่ results/final/ · results/replay/ · results/s1/ ครั้งเดียว
    `estimate` : วัดเวลา 1 unit **บน dev เท่านั้น** (เทรนถึงก่อน fold 3 แล้วทาย fold 3 · 1 seed) แล้วประเมินเวลารวมของ C6

ทำไมต้องมี — เขียนหลัง commit LOCK.md (0bcbb96) ตามหัวข้อ 9 ของ LOCK · ใช้ฟังก์ชันเทรน/ประเมินชุดเดิมของ Stage 0–6 ทั้งหมด
    (`data.fold_inputs` · `train.run_fold` · `twostage.run_fold_two_stage` · `shrink.choose_c` · `evaluate.metrics` · `evaluate.dm_test` ·
     ตัวเทียบผ่าน `legacy.get_regressors` แบบเดียวกับ `baselines.run_fold`) — ต่างจาก Stage ก่อนแค่ "ช่วงเทรน/ช่วงประเมิน" (Split)

*** gate (ปฏิเสธก่อนโหลดข้อมูลใด ๆ) ***
    git status ทั้ง repo ต้องว่าง · LOCK.md และ mode1600/LOCK_1600.md ต้องถูก commit · LOCK_1600 ถูกแตะโดย commit เดียว ·
    commit ล่าสุดของ LOCK.md = NEXTDAY_LOCK_COMMIT ใน LOCK_1600 · results/final|replay|s1 ต้องยังไม่มีผล
    ถ้ารันแล้ว error / คำทำนายไม่ finite → หยุดและรายงาน **ห้ามแก้แล้วรันใหม่เอง** (LOCK.md หัวข้อ 9 ข้อ (c))

ข้อตกลงที่ระบุไว้ (ผู้ใช้ค้านได้ก่อน commit โค้ดนี้ · B41)
    * คำทำนายคงที่ (std(ŷ)/std(y) < 1e-8 หรือ ptp(ŷ) = 0) → IC Pearson = IC Spearman := 0 + ธง `collapsed` ทุกตาราง (final · replay · S1 · F1 · F2)
      = `const_ic_zero=True` แบบเดียวกับ B39 และโหมด 16:00 (ผู้ใช้ตัดสินก่อนมีผล final ใด ๆ · B41)
    * ผลรวมสองหุ้นของ relMAE / IC = ค่าเฉลี่ยของ 2 หุ้น (แบบเดียวกับตาราง Stage ก่อน) · DM รวม = `evaluate.dm_test` (เฉลี่ย d ของวันเดียวกัน)
    * ตัวเทียบเดิม (Naive · Mean Return · MLP · RF · XGB) ไม่ใช้ c (แบบเดียวกับ C3) · เทรนบนทุกแถวก่อนช่วงประเมินที่ feature 29 ตัวครบ (B11)
    * replay: เทรนครั้งเดียวด้วยแถว ≤ 2026-09-25 แล้วทาย 17 วัน — input ของวัน t ใช้ข้อมูลถึง t−1 จึงเท่ากับทายรายวันด้วยโมเดลเดิม

รัน (จาก task_b_v2/):  python -m src.run_c6 estimate           (dev เท่านั้น · ไม่ต้องผ่าน gate)
                       python -m src.run_c6 run --jobs 6       (หลังผู้ใช้ commit ทุกอย่างและแจ้งเท่านั้น)
"""

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from src import baselines, data, evaluate, paths, shrink, train, twostage
from src.check_data import md_table
from src.configs import BASE, variant
from src.run_stage345 import HYBRID, NOCNN, STAGE4_CFGS
from src.settings import (FINAL_END, FINAL_START, REFIT_SEEDS, REPLAY_END, REPLAY_START, REPLAY_TRAIN_END, TICKERS,
                          VAL_END)

R = paths.RESULTS_DIR
OUT = {"final": R / "final", "replay": R / "replay", "s1": R / "s1"}
LOCK_MD = "task_b_v2/LOCK.md"
LOCK_1600 = "task_b_v2/mode1600/LOCK_1600.md"
NEXTDAY_HASH_RE = re.compile(r"NEXTDAY_LOCK_COMMIT:\s*([0-9a-f]{40})")
S1_TRAIN_START = "2021-01-01"                                 # B37
REPLAY_LABEL = "ทำนายย้อนหลังแบบ walk-forward ไม่ใช่ log สด"   # LOCK.md หัวข้อ 9 ข้อ (a)
BOOT_BLOCK, BOOT_REPS, BOOT_SEED = 20, 10_000, 0              # F1 (LOCK.md หัวข้อ 7)

# track ตาม LOCK.md หัวข้อ 5 (ค่า config มาจากโค้ดชุดเดียวกับที่ตารางใน LOCK สร้าง · ไม่พิมพ์เอง)
TRACKS = {"track1": BASE, "track2": HYBRID, "track3": STAGE4_CFGS[0],
          "track4": variant("cnn1d_raw", arch="cnn1d_raw"), "control": NOCNN}
TRACK_LABEL = {"track1": "Track 1 · CNN→ANN (spectrogram)", "track2": "Track 2 · hybrid", "track3": "Track 3 · สองขั้น",
               "track4": "Track 4 · 1D บน return ดิบ", "control": "Control · hybrid-noCNN"}
LEGACY = list(baselines.ALL_NAMES)                            # Naive · Mean Return · ANN (MLP) · Random Forest · XGBoost


class StopRun(RuntimeError):
    """gate ไม่ผ่าน / error / คำทำนายไม่ finite → หยุดและรายงาน (ห้ามแก้แล้วรันต่อเงียบ ๆ)"""


# ---------------------------------------------------------------
# ช่วงเทรน / ช่วงประเมิน
# ---------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    """ช่วงหนึ่งของ C6 · train = แถวที่ใช้ได้ใน [train_start, วันแรกของช่วงประเมิน) · eval = [eval_start, eval_end]"""
    name: str
    zone_end: str                  # ตัดข้อมูลหลังวันนี้ทิ้งก่อนสร้างอะไรทั้งนั้น (data.dataset_from_prices)
    eval_start: str
    eval_end: str
    train_end: str                 # วันสุดท้ายของ train ที่คาด (assert)
    train_start: str = None        # None = ตั้งแต่ต้นข้อมูล


SPLITS = {"final": Split("final", FINAL_END, FINAL_START, FINAL_END, VAL_END),
          "replay": Split("replay", REPLAY_END, REPLAY_START, REPLAY_END, REPLAY_TRAIN_END),
          "s1": Split("s1", FINAL_END, FINAL_START, FINAL_END, VAL_END, S1_TRAIN_START)}


@dataclass(frozen=True)
class SplitFold(data.Fold):
    """Fold ของ data.py + จุดเริ่ม train (S1) · data.fold_inputs ใช้แค่ k, eval_pos, train_stop และ train_pos()"""
    train_start: int = 0

    def train_pos(self, usable):
        pos = np.arange(self.train_start, self.train_stop)
        return pos[usable[self.train_start:self.train_stop]]


def make_fold(ds, split):
    """Split → SplitFold บนปฏิทินของ ds · ตรวจว่าช่วงประเมินต่อกัน และ train จบที่ train_end พอดี"""
    d = ds.dates
    ev = np.flatnonzero((d >= pd.Timestamp(split.eval_start)) & (d <= pd.Timestamp(split.eval_end)))
    if len(ev) == 0 or not (np.diff(ev) == 1).all():
        raise StopRun(f"{split.name}: ช่วงประเมินว่างหรือไม่ต่อกัน")
    stop = int(ev[0])
    if d[stop - 1] != pd.Timestamp(split.train_end):
        raise StopRun(f"{split.name}: วันสุดท้ายของ train = {d[stop - 1].date()} ไม่ใช่ {split.train_end}")
    start = 0 if split.train_start is None else int(np.searchsorted(d, pd.Timestamp(split.train_start)))
    return SplitFold(k=0, eval_pos=ev, train_stop=stop, train_start=start)


# ---------------------------------------------------------------
# หนึ่ง unit = (split, โมเดล, หุ้น)
# ---------------------------------------------------------------
def nn_unit(split, key, ticker, seeds):
    """track / control หนึ่งตัวบนหุ้นหนึ่ง: เทรนครั้งเดียว (เฉลี่ย seeds) → คำทำนายรายวัน (ก่อน/หลังปรับ c) + ข้อมูลการเทรน"""
    t0 = time.time()
    cfg = TRACKS[key]
    ds = data.build_dataset(ticker, split.zone_end)
    fold = make_fold(ds, split)
    prepared = data.fold_inputs(ds, cfg, fold)
    extra = train.standardized_extra(ds, prepared) if cfg.needs_features else None
    if cfg.train_mode == "two_stage":
        pred, hists, seed_preds, (val_pred, y_val) = twostage.run_fold_two_stage(cfg, prepared, seeds=seeds, extra=extra)
    else:
        pred, hists, seed_preds, (val_pred, y_val) = train.run_fold(
            cfg, prepared, seeds=seeds, extra=extra, return_hist=True, return_seed_preds=True, return_val_pred=True)
    c = shrink.choose_c(y_val, val_pred)
    ev = prepared["pos_eval"]
    frame = evaluate.predictions_frame(ds.dates[ev], ticker, 0, prepared["y_eval"], pred, prepared["prev_close_eval"],
                                       phase=split.name, model=key)
    frame["pred_shrunk"] = shrink.apply(pred, c)
    frame["close"] = ds.close[ev]
    frame["c"] = c
    for s, sp in zip(seeds, seed_preds):
        frame[f"pred_s{s}"] = sp
    info = {"phase": split.name, "model": key, "ticker": ticker, "n_seeds": len(seeds), "c": c,
            "n_fit": len(prepared["pos_fit"]), "n_val": len(prepared["pos_val"]), "n_eval": len(ev),
            "train_first": str(ds.dates[prepared["pos_fit"][0]].date()), "train_last": str(ds.dates[prepared["pos_val"][-1]].date()),
            "image_scale": float(prepared["transform"].scale),
            "epochs_mean": float(np.mean([h["epochs"] for h in hists])),
            "n_hit_epoch_cap": int(sum(not h["stopped_early"] for h in hists)), "seconds": round(time.time() - t0, 1)}
    return frame, info


def legacy_unit(split, ticker, names):
    """ตัวเทียบเดิมบนหุ้นหนึ่ง (แบบเดียวกับ baselines.run_fold แต่ช่วงเทรนตาม Split) · ไม่มี c"""
    t0 = time.time()
    ds = data.build_dataset(ticker, split.zone_end)
    fold = make_fold(ds, split)
    ok = ds.ok29
    tr = np.arange(fold.train_start, fold.train_stop)[ok[fold.train_start:fold.train_stop]]
    ev = fold.eval_pos
    if not (ok[ev].all() and tr[-1] < ev[0]):
        raise StopRun(f"{split.name} {ticker}: feature 29 ตัวไม่ครบในช่วงประเมิน หรือ train ข้ามเวลา")
    X, y = ds.feat29.to_numpy(dtype=float), ds.y
    regs = baselines.get_regressors()
    frames = []
    for name in names:
        if name in baselines.SIMPLE_NAMES:
            p = baselines.predict_simple(name, y[tr], len(ev))
        else:
            m = TransformedTargetRegressor(regressor=regs[name], transformer=StandardScaler())
            m.fit(X[tr], y[tr])
            p = np.asarray(m.predict(X[ev]), dtype=float)
        f = evaluate.predictions_frame(ds.dates[ev], ticker, 0, y[ev], p, ds.prev_close[ev], phase=split.name, model=name)
        f["pred_shrunk"] = p
        f["close"] = ds.close[ev]
        f["c"] = np.nan
        frames.append(f)
    info = {"phase": split.name, "model": "legacy:" + "|".join(names), "ticker": ticker, "n_seeds": 1, "c": np.nan,
            "n_fit": len(tr), "n_val": 0, "n_eval": len(ev), "train_first": str(ds.dates[tr[0]].date()),
            "train_last": str(ds.dates[tr[-1]].date()), "seconds": round(time.time() - t0, 1)}
    return pd.concat(frames, ignore_index=True), info


def _unit(args):
    """ตัวห่อสำหรับ ProcessPoolExecutor (ฟังก์ชันระดับบนสุด)"""
    kind, split_name, key, ticker, seeds = args
    split = SPLITS[split_name] if isinstance(split_name, str) else split_name
    if kind == "nn":
        return nn_unit(split, key, ticker, seeds)
    return legacy_unit(split, ticker, key)


def plan_units(seeds=REFIT_SEEDS):
    """รายการงานทั้งหมดของ C6 (งานหนักขึ้นก่อน เพื่อให้ขนานได้เต็มที่)"""
    tasks = []
    for ph in ("final", "replay"):
        for key in ("track3", "track2", "track1", "track4", "control"):
            tasks += [("nn", ph, key, t, tuple(seeds)) for t in TICKERS]
    tasks += [("nn", "s1", "track1", t, tuple(seeds)) for t in TICKERS]
    for ph in ("final", "replay"):
        tasks += [("legacy", ph, LEGACY, t, None) for t in TICKERS]
    tasks += [("legacy", "s1", ["Random Forest"], t, None) for t in TICKERS]
    return tasks


def run_units(tasks, jobs):
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            out = list(pool.map(_unit, tasks))
    else:
        out = [_unit(a) for a in tasks]
    preds = pd.concat([f for f, _ in out], ignore_index=True)
    info = pd.DataFrame([i for _, i in out])
    bad = preds[~np.isfinite(preds[["pred", "pred_shrunk"]].to_numpy(dtype=float)).all(axis=1)]
    if len(bad):
        raise StopRun(f"คำทำนายไม่ finite {len(bad)} แถว ({sorted(set(bad['model']))}) — หยุดและรายงาน (ข้อ (c))")
    return preds, info


# ---------------------------------------------------------------
# ตัววัด · F1 · F2 · ตาราง log
# ---------------------------------------------------------------
def metrics_table(preds):
    """หนึ่งแถวต่อ (ช่วง, โมเดล, หุ้น): ตัววัดก่อนปรับ c (หลัก) + หลังปรับ + ธงคำทำนายคงที่"""
    rows = []
    for (ph, model, t), g in preds.groupby(["phase", "model", "ticker"], sort=False):
        g = g.sort_values("date")
        m = evaluate.metrics(g["y"], g["pred"], g["prev_close"], const_ic_zero=True)          # มีคอลัมน์ collapsed (B39 · B41)
        ms = evaluate.metrics(g["y"], g["pred_shrunk"], g["prev_close"], const_ic_zero=True)
        rows.append({"phase": ph, "model": model, "ticker": t, **m, "c": float(g["c"].iloc[0]),
                     "relMAE_shrunk": ms["relMAE"], "relMAE_baht_shrunk": ms["relMAE_baht"], "MAE_baht_shrunk": ms["MAE_baht"],
                     "IC_shrunk": ms["IC"], "collapsed_shrunk": ms["collapsed"]})
    return pd.DataFrame(rows)


def ic(y, p):
    """IC (Pearson) · คำทำนายคงที่ → 0 (กติกาเดียวกับ evaluate.metrics(const_ic_zero=True) · B39/B41) · y คงที่ → NaN"""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if np.ptp(y) == 0:
        return float("nan")
    if evaluate.is_constant_prediction(y, p):
        return 0.0
    return float(np.corrcoef(p, y)[0, 1])


def bootstrap_delta_ic(y, a, b, *, block=BOOT_BLOCK, reps=BOOT_REPS, seed=BOOT_SEED):
    """
    F1: ΔIC = IC(a) − IC(b) บนวันเดียวกัน + 95% CI · แบ่งวันตามเวลาเป็นบล็อกติดกันบล็อกละ `block` วัน (บล็อกท้ายสั้นกว่าได้)
    สุ่มบล็อกแบบมีคืน (จำนวนบล็อกเท่าเดิม) · seed 0 · 10,000 รอบ · percentile 2.5/97.5 (นิยามเดียวกับ LOCK.md หัวข้อ 7 และ mode1600)
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y, a, b))
    n = len(y)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        deltas[r] = ic(y[idx], a[idx]) - ic(y[idx], b[idx])
    d = ic(y, a) - ic(y, b)
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {"delta_IC": d, "IC_a": ic(y, a), "IC_b": ic(y, b), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_nan_reps": int(np.isnan(deltas).sum()), "supported": bool(np.isfinite(d) and d > 0 and lo > 0),
            "n_days": n, "block": block, "reps": reps, "seed": seed}


def f1_table(final_preds):
    """F1 แยกหุ้น: hybrid (track2) เทียบ control บนช่วง final"""
    rows = []
    for t in TICKERS:
        a = final_preds[(final_preds["model"] == "track2") & (final_preds["ticker"] == t)].sort_values("date")
        b = final_preds[(final_preds["model"] == "control") & (final_preds["ticker"] == t)].sort_values("date")
        if not (a["date"].to_numpy() == b["date"].to_numpy()).all():
            raise StopRun("F1: วันของ hybrid กับ control ไม่ตรงกัน")
        rows.append({"ticker": t, **bootstrap_delta_ic(a["y"], a["pred"], b["pred"]),
                     "collapsed_a": evaluate.is_constant_prediction(a["y"], a["pred"]),
                     "collapsed_b": evaluate.is_constant_prediction(b["y"], b["pred"])})
    return pd.DataFrame(rows)


def f2_table(final_preds, metrics_final):
    """F2: track 1 เทียบ Naive และเทียบ track 4 — relMAE · อัตราส่วน MAE · Diebold–Mariano (แยกหุ้น + รวม)"""
    p = lambda m, t=None: final_preds[(final_preds["model"] == m) & ((final_preds["ticker"] == t) if t else True)]  # noqa: E731
    mt = metrics_final.set_index(["model", "ticker"])
    rows = []
    for t in list(TICKERS) + [None]:
        tick = t or "รวม"
        rel = [mt.loc[("track1", x), "relMAE"] for x in ([t] if t else TICKERS)]
        relb = [mt.loc[("track1", x), "relMAE_baht"] for x in ([t] if t else TICKERS)]
        e1 = (p("track1", t)["pred"] - p("track1", t)["y"]).abs().mean()
        e4 = (p("track4", t)["pred"] - p("track4", t)["y"]).abs().mean()
        dn = evaluate.dm_test(p("track1", t), p("Naive", t))
        d4 = evaluate.dm_test(p("track1", t), p("track4", t))
        rows.append({"ticker": tick, "relMAE_track1": float(np.mean(rel)), "relMAE_baht_track1": float(np.mean(relb)),
                     "beats_naive": bool(np.mean(rel) < 1), "MAE_ratio_track1_over_track4": float(e1 / e4),
                     "DM_vs_naive": dn["DM"], "p_DM_vs_naive": dn["p_DM"], "mean_d_vs_naive": dn["mean_d"],
                     "DM_vs_track4": d4["DM"], "p_DM_vs_track4": d4["p_DM"], "mean_d_vs_track4": d4["mean_d"],
                     "collapsed_track1": bool(any(mt.loc[("track1", x), "collapsed"] for x in ([t] if t else TICKERS))),
                     "collapsed_track4": bool(any(mt.loc[("track4", x), "collapsed"] for x in ([t] if t else TICKERS)))})
    return pd.DataFrame(rows)


def replay_log(replay_preds, ticker):
    """
    ตาราง log รายวันของหุ้นหนึ่ง (LOCK.md หัวข้อ 9 ข้อ (a)): วันที่ · โมเดล · ราคาปิดวันก่อน · ราคาที่ทาย · ราคาจริง · error บาท · ทิศถูกหรือไม่
    ทิศถูก = sign(ราคาที่ทาย − ปิดวันก่อน) ตรงกับ sign(ราคาจริง − ปิดวันก่อน) · วันที่ราคาไม่เปลี่ยน (y = 0) = "—" (ไม่นับ เหมือน DirAcc)
    """
    g = replay_preds[replay_preds["ticker"] == ticker].sort_values(["date", "model"])
    pred_price = g["prev_close"] * (1 + g["pred"])
    actual = g["prev_close"] * (1 + g["y"])
    direction = np.where(g["y"] == 0, "—", np.where(np.sign(g["pred"]) == np.sign(g["y"]), "ถูก", "ผิด"))
    return pd.DataFrame({"วันที่": g["date"].dt.strftime("%Y-%m-%d").to_numpy(), "โมเดล": g["model"].to_numpy(),
                         "ราคาปิดวันก่อน": g["prev_close"].to_numpy(), "ราคาที่ทาย": pred_price.round(4).to_numpy(),
                         "ราคาจริง": actual.round(4).to_numpy(), "error บาท": (pred_price - actual).round(4).to_numpy(),
                         "ทิศถูกหรือไม่": direction, "ป้าย": REPLAY_LABEL})


def replay_mae_summary(replay_preds):
    """สรุป MAE ต่อโมเดล (บาท และ return · ก่อนปรับ c และหลังปรับ c) แยกหุ้น + จำนวนวันทิศถูก"""
    rows = []
    for (t, model), g in replay_preds.groupby(["ticker", "model"], sort=False):
        err = (g["pred"] - g["y"]) * g["prev_close"]
        err_s = (g["pred_shrunk"] - g["y"]) * g["prev_close"]
        moved = g["y"] != 0
        rows.append({"ticker": t, "model": model, "n_days": len(g), "MAE_baht": float(err.abs().mean()),
                     "MAE_baht_shrunk": float(err_s.abs().mean()), "MAE_return": float((g["pred"] - g["y"]).abs().mean()),
                     "n_direction_correct": int((np.sign(g["pred"][moved]) == np.sign(g["y"][moved])).sum()),
                     "n_direction_counted": int(moved.sum()),
                     "collapsed": evaluate.is_constant_prediction(g["y"], g["pred"]), "ป้าย": REPLAY_LABEL})
    return pd.DataFrame(rows)


def s1_table(metrics_all):
    """S1 (descriptive): track 1 และ RF · ช่วงเทรนสั้น (2021-01-01 →) เทียบเต็ม · relMAE (return/บาท) + IC (Pearson/Spearman) แยกหุ้น"""
    rows = []
    mt = metrics_all.set_index(["phase", "model", "ticker"])
    for model in ("track1", "Random Forest"):
        for t in TICKERS:
            full, short = mt.loc[("final", model, t)], mt.loc[("s1", model, t)]
            row = {"model": model, "ticker": t}
            for col in ("relMAE", "relMAE_baht", "IC", "IC_spearman"):
                row[f"{col}_full"], row[f"{col}_short"] = float(full[col]), float(short[col])
                row[f"{col}_short_minus_full"] = float(short[col] - full[col])
            row["collapsed_full"], row["collapsed_short"] = bool(full["collapsed"]), bool(short["collapsed"])
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# gate
# ---------------------------------------------------------------
def _git(*args):
    r = subprocess.run(["git", *args], cwd=paths.TASK_DIR, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout.strip()


def result_files(d):
    """ไฟล์ผลในโฟลเดอร์ (ไม่นับ cache/ ซึ่ง .gitignore กันไว้)"""
    return [p for p in d.rglob("*") if p.is_file() and "cache" not in p.relative_to(d).parts] if d.exists() else []


def gate(out_dirs=None):
    """ผ่าน → dict (head_commit, run_time, lock_commit, lock1600_commit) · ไม่ผ่าน → StopRun พร้อมปัญหาทั้งหมด"""
    out_dirs = OUT if out_dirs is None else out_dirs
    problems = []
    code, root = _git("rev-parse", "--show-toplevel")
    if code != 0:
        raise StopRun("ไม่อยู่ใน git repo")
    st = _git("-C", root, "status", "--porcelain", "--untracked-files=all")[1]
    if st:
        problems.append("git status ไม่ว่าง:\n      " + "\n      ".join(st.splitlines()[:10]))
    for rel in (LOCK_MD, LOCK_1600):
        if _git("-C", root, "ls-files", "--error-unmatch", "--", rel)[0] != 0:
            problems.append(f"{rel} ยังไม่ถูก commit")
    lock_commit = _git("-C", root, "log", "-1", "--format=%H", "--", LOCK_MD)[1] or None
    commits_1600 = _git("-C", root, "log", "--format=%H", "--", LOCK_1600)[1].split()
    if len(commits_1600) != 1:
        problems.append(f"{LOCK_1600} ถูกแตะโดย {len(commits_1600)} commit (ต้องเป็น 1)")
    try:
        text = (paths.TASK_DIR / "mode1600" / "LOCK_1600.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    m = NEXTDAY_HASH_RE.search(text)
    if not m or m.group(1) != lock_commit:
        problems.append(f"LOCK.md ถูกแก้หลัง LOCK_1600 อ้าง หรือ LOCK_1600 ไม่ได้บันทึก hash ({(m.group(1)[:10] if m else '—')} vs {(lock_commit or '—')[:10]})")
    for name, d in out_dirs.items():
        if result_files(d):
            problems.append(f"{d} มีผลอยู่แล้ว — C6 รันครั้งเดียว ห้ามเขียนทับ")
    if problems:
        raise StopRun("gate ของ C6 ไม่ผ่าน — ห้ามรัน:\n  - " + "\n  - ".join(problems))
    return {"head_commit": _git("-C", root, "rev-parse", "HEAD")[1],
            "run_time": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(timespec="seconds"),
            "lock_commit": lock_commit, "lock1600_commit": commits_1600[0]}


# ---------------------------------------------------------------
# เขียนผล
# ---------------------------------------------------------------
def fmt(v, k="r4"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return {"r4": f"{v:.4f}", "s4": f"{v:+.4f}", "r3": f"{v:.3f}", "p": f"{v:.3f}"}[k]


def save(df, path, g):
    out = df.copy()
    out.insert(0, "run_time", g["run_time"])
    out.insert(0, "run_head_commit", g["head_commit"])
    out.to_csv(path, index=False, encoding="utf-8")


def write_all(g, preds, info, mets, f1, f2, s1, seconds, jobs):
    for d in OUT.values():
        paths.assert_output_dir(d)
        d.mkdir(parents=True, exist_ok=True)
    manifest = {"gate": g, "seconds": round(seconds, 1), "jobs": jobs, "seeds": list(REFIT_SEEDS),
                "tracks": {k: c.to_row() for k, c in TRACKS.items()}, "splits": {k: vars(v) for k, v in SPLITS.items()}}
    lab = lambda m: TRACK_LABEL.get(m, m)  # noqa: E731
    stamp = f"HEAD `{g['head_commit'][:10]}` · รันเมื่อ {g['run_time']} · LOCK.md `{g['lock_commit'][:10]}` · LOCK_1600 `{g['lock1600_commit'][:10]}`"
    for ph, d in OUT.items():
        save(preds[preds["phase"] == ph], d / "predictions.csv", g)
        save(info[info["phase"] == ph], d / "train_info.csv", g)
        save(mets[mets["phase"] == ph], d / "metrics.csv", g)
        (d / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- final ----
    mf = mets[mets["phase"] == "final"]
    rows = [{"โมเดล": lab(r.model), "หุ้น": r.ticker, "relMAE (return)": fmt(r.relMAE), "relMAE (บาท)": fmt(r.relMAE_baht),
             "หลังปรับ c (return)": fmt(r.relMAE_shrunk), "c": fmt(r.c, "r3"), "MAE (บาท)": fmt(r.MAE_baht),
             "IC": fmt(r.IC, "s4"), "Spearman": fmt(r.IC_spearman, "s4"), "DirAcc": fmt(r.DirAcc, "r3"),
             "StdRatio": fmt(r.StdRatio, "r3"), "คงที่": "ใช่" if r.collapsed else ""} for r in mf.itertuples()]
    save(f1, OUT["final"] / "f1.csv", g)
    save(f2, OUT["final"] / "f2.csv", g)
    f1r = [{"หุ้น": r.ticker, "IC hybrid": fmt(r.IC_a, "s4"), "IC hybrid-noCNN": fmt(r.IC_b, "s4"), "ΔIC": fmt(r.delta_IC, "s4"),
            "95% CI": f"{fmt(r.ci_lo, 's4')} … {fmt(r.ci_hi, 's4')}", "F1": "**สนับสนุน**" if r.supported else "ไม่สนับสนุน",
            "คงที่ (hybrid/noCNN)": f"{'ใช่' if r.collapsed_a else '—'}/{'ใช่' if r.collapsed_b else '—'}"}
           for r in f1.itertuples()]
    f2r = [{"หุ้น": r.ticker, "relMAE track 1 (return)": fmt(r.relMAE_track1), "relMAE (บาท)": fmt(r.relMAE_baht_track1),
            "ชนะ Naive": "ใช่" if r.beats_naive else "ไม่", "DM vs Naive (p)": f"{fmt(r.DM_vs_naive, 'p')} ({fmt(r.p_DM_vs_naive, 'p')})",
            "MAE track1 / track4": fmt(r.MAE_ratio_track1_over_track4), "DM vs track 4 (p)": f"{fmt(r.DM_vs_track4, 'p')} ({fmt(r.p_DM_vs_track4, 'p')})",
            "คงที่ (track1/track4)": f"{'ใช่' if r.collapsed_track1 else '—'}/{'ใช่' if r.collapsed_track4 else '—'}"}
           for r in f2.itertuples()]
    md = ["# C6 · final (2025-02-26 → 2026-08-28) — ประเมินครั้งเดียวหลัง LOCK", "", f"> {stamp}",
          "> ช่วง final เคยถูกใช้ประเมินโมเดลรุ่นก่อนของทีมมาแล้ว แต่ไม่เคยใช้ออกแบบหรือเลือก config ของ CNN รุ่นนี้ (LOCK.md หัวข้อ 9)", "",
          "## ตัววัดทุกโมเดล (track 10 seeds · c จากส่วน early-stop · ตัวเทียบเดิมไม่มี c)", "", md_table(pd.DataFrame(rows)), "",
          "## F1 — IC(hybrid) − IC(hybrid-noCNN) แยกหุ้น (block bootstrap 20 วัน · 10,000 รอบ · seed 0)", "", md_table(pd.DataFrame(f1r)), "",
          "## F2 — track 1 เทียบ Naive และ track 4 (DM: ลบ = track 1 error ต่ำกว่า)", "", md_table(pd.DataFrame(f2r)), "",
          "\"รวม\" ของ relMAE = ค่าเฉลี่ยสองหุ้น · DM รวม = เฉลี่ย d ของวันเดียวกันสองหุ้น (evaluate.dm_test) · คำทำนายคงที่ → IC = Spearman = 0 และติดธง (คอลัมน์ คงที่ · B39/B41)", ""]
    (OUT["final"] / "final_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- replay ----
    rp = preds[preds["phase"] == "replay"]
    mae = replay_mae_summary(rp)
    save(mae, OUT["replay"] / "mae_summary.csv", g)
    md = [f"# C6 · replay (2026-09-28 → 2026-10-22 · 17 วันทำการ) — {REPLAY_LABEL}", "", f"> {stamp}",
          "> เทรนครั้งเดียวด้วยข้อมูลถึง 2026-09-25 · input ของวัน t ใช้ข้อมูลถึง t−1 · **replay มี 17 วัน ใช้สรุปทางสถิติไม่ได้**", ""]
    for t in TICKERS:
        log = replay_log(rp, t)
        save(log, OUT["replay"] / f"log_{t}.csv", g)
        m = mae[mae["ticker"] == t]
        md += [f"## {t} — สรุป MAE ต่อโมเดล ({REPLAY_LABEL})", "",
               md_table(pd.DataFrame([{"โมเดล": lab(r.model), "MAE (บาท)": fmt(r.MAE_baht), "หลังปรับ c": fmt(r.MAE_baht_shrunk),
                                       "ทิศถูก": f"{r.n_direction_correct}/{r.n_direction_counted}", "คงที่": "ใช่" if r.collapsed else ""} for r in m.itertuples()])), "",
               f"ตารางรายวันเต็ม: `results/replay/log_{t}.csv`", ""]
    (OUT["replay"] / "replay_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- S1 ----
    save(s1, OUT["s1"] / "s1.csv", g)
    s1r = [{"โมเดล": lab(r.model), "หุ้น": r.ticker, "relMAE เต็ม": fmt(r.relMAE_full), "relMAE สั้น": fmt(r.relMAE_short),
            "Δ": fmt(r.relMAE_short_minus_full, "s4"), "relMAE บาท เต็ม/สั้น": f"{fmt(r.relMAE_baht_full)} / {fmt(r.relMAE_baht_short)}",
            "IC เต็ม/สั้น": f"{fmt(r.IC_full, 's4')} / {fmt(r.IC_short, 's4')}",
            "Spearman เต็ม/สั้น": f"{fmt(r.IC_spearman_full, 's4')} / {fmt(r.IC_spearman_short, 's4')}",
            "คงที่ เต็ม/สั้น": f"{'ใช่' if r.collapsed_full else '—'}/{'ใช่' if r.collapsed_short else '—'}"} for r in s1.itertuples()]
    md = ["# C6 · S1 — ความไวต่อช่วงข้อมูลเทรน (descriptive เท่านั้น · ห้ามใช้เปลี่ยน config · B37)", "", f"> {stamp}",
          "> สั้น = เทรน 2021-01-01 → 2025-02-25 · เต็ม = ต้นข้อมูล → 2025-02-25 (= final) · ประเมินช่วง final ชุดวันเดียวกัน · S2 ไม่ทำ (B40)", "",
          md_table(pd.DataFrame(s1r)), ""]
    (OUT["s1"] / "s1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def cmd_run(args):
    g = gate()                                                     # 1) ก่อนโหลดข้อมูลใด ๆ
    t0 = time.time()
    preds, info = run_units(plan_units(), args.jobs)
    n = preds.groupby(["phase", "model", "ticker"]).size()
    expect = {"final": 362, "s1": 362, "replay": 17}
    wrong = [k for k, v in n.items() if v != expect[k[0]]]
    if wrong:
        raise StopRun(f"จำนวนวันประเมินไม่ตรงที่ล็อก: {wrong[:5]}")
    mets = metrics_table(preds)
    f1 = f1_table(preds[preds["phase"] == "final"])
    f2 = f2_table(preds[preds["phase"] == "final"], mets[mets["phase"] == "final"])
    s1 = s1_table(mets)
    write_all(g, preds, info, mets, f1, f2, s1, time.time() - t0, args.jobs)
    print(f"[c6] เสร็จใน {(time.time() - t0) / 60:.1f} นาที · เขียน {', '.join(str(d) for d in OUT.values())}")


# ---------------------------------------------------------------
# estimate (dev เท่านั้น)
# ---------------------------------------------------------------
def dev_split(ds):
    """split จำลองบน dev: เทรนถึงก่อน fold 3 แล้วทาย fold 3 (ไม่แตะข้อมูล ≥ 2025-02-26)"""
    f3 = data.make_folds(ds.dates)[-1]
    return Split("dev_fold3", VAL_END, str(ds.dates[f3.eval_pos[0]].date()), VAL_END, str(ds.dates[f3.train_stop - 1].date()))


def cmd_estimate(args):
    ds = data.build_dataset("KBANK", VAL_END)
    sp = dev_split(ds)
    dev_rows = make_fold(ds, sp).train_stop
    rows = {}
    for key in ("track1", "track2", "track3", "track4", "control"):
        t0 = time.time()
        nn_unit(sp, key, "KBANK", (0,))
        rows[key] = time.time() - t0
        print(f"  {key:8s} 1 seed (dev · train {dev_rows} แถว → fold 3): {rows[key]:.1f}s")
    t0 = time.time()
    legacy_unit(sp, "KBANK", LEGACY)
    leg = time.time() - t0
    # ประมาณ: เวลาเทรนแปรตามจำนวนแถวก่อนช่วงประเมิน — final = ทุกแถวของ dev · S1 = แถว dev ตั้งแต่ 2021-01-01 (นับจากปฏิทิน dev) ·
    # replay = 2,470 − 17 แถว (จำนวนแถวจาก data/SOURCES.md · ไม่อ่านข้อมูลหลัง 2025-02-25)
    n_rows = {"final": len(ds), "replay": 2470 - 17, "s1": int((ds.dates >= pd.Timestamp(S1_TRAIN_START)).sum())}
    serial = 0.0
    for ph, nr in n_rows.items():
        keys = ("track1",) if ph == "s1" else tuple(rows)
        serial += sum(rows[k] for k in keys) * len(REFIT_SEEDS) * len(TICKERS) * nr / dev_rows
        serial += leg * len(TICKERS) * nr / dev_rows
    biggest = max(rows["track3"] * len(REFIT_SEEDS) * n_rows["replay"] / dev_rows, 1.0)
    wall = max(serial / 4.6, biggest)                              # ประสิทธิภาพขนาน 4.6 ที่ jobs 6 (C3) · ไม่เร็วกว่างานเดี่ยวที่ใหญ่สุด
    info = {"measured_on": "dev (KBANK · train ถึงก่อน fold 3 · ทาย fold 3 · 1 seed)", "dev_train_rows": int(dev_rows),
            "seconds_1seed": rows, "legacy_seconds": leg, "train_rows_assumed": n_rows,
            "c6_serial_minutes": serial / 60, "c6_wall_minutes_jobs6": wall / 60, "largest_single_unit_minutes": biggest / 60,
            "mode1600_wall_minutes_jobs6": "15–25 (HANDOFF_1600 · วัดด้วย y สลับ)"}
    out = R / "c6_prep"
    paths.assert_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "time_estimate.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "estimate"])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    {"run": cmd_run, "estimate": cmd_estimate}[a.cmd](a)


if __name__ == "__main__":
    main()
