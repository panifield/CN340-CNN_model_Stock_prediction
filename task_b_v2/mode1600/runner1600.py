"""
runner1600.py — wrapper ของโหมด 16:00: walk-forward · replay · ตัววัด · F3 · DM (PLAN ข้อ b, c, e, f, g)
=================================================================================================
ไฟล์นี้ทำอะไร
    `wf_blocks(calendar)`        : ขอบบล็อก walk-forward ทุก 20 วันทำการจากปฏิทินไฟล์รายวัน (ร่วมสองหุ้น · M10)
    `run_split(...)`             : เทรนทุก track บน "แถวก่อนวันแรกของบล็อก" แล้วทำนายวันในบล็อก (หนึ่งหุ้น หนึ่งบล็อก)
    `run_walk_forward(...)`      : วนทุกบล็อก (ขนานได้) → คำทำนายรายวันของทุก track
    `run_replay(...)`            : เทรนครั้งเดียวด้วยแถว ≤ 2026-09-25 → ทำนาย replay ทาง (i) และ (ii)
    `summarize(...)`, `f3(...)`  : ตัววัดต่อ track × หุ้น × ช่วง + DM vs Naive · ΔIC(หลัก − control) + block bootstrap

ทำไมต้องมี / หลักการ (M16: **ไม่แก้ src** — เรียกของเดิมแบบอ่านอย่างเดียว)
    * ภาพ + แบ่ง fit/early-stop + สเกลภาพ: `src.data.fold_inputs` บน Dataset ที่เปลี่ยน y เป็น target ของ 16:00
      (`dataclasses.replace(ds, y=y1600, prev_close=C15)` · แถวที่ไม่ labeled ให้ y = NaN → ds.usable() ตัดเอง)
    * โมเดล + เทรน: `src.train.run_fold(cfg, prepared, extra=...)` → `models.build(cfg, n_extra=4)` อ่าน n_extra จาก extra
    * c: `src.shrink.choose_c` บนส่วน early-stop · ตัววัด: `src.evaluate.metrics(y, ŷ, C15, const_ic_zero=True)` (B39)
    * RF: `legacy get_regressors()["Random Forest"]` (phase1c-tuned · M14) ห่อแบบเดียวกับ `src/baselines.py`
    * **ไม่ใช้ `src.runner`** (สร้าง y แบบ next_day และเขียนลง results/ + experiment_log.csv) · ไฟล์นี้ไม่เขียนอะไรเลย
      การเขียนผลอยู่ใน `run1600.py` เท่านั้น (หลังผ่าน gate)

ห้าม: เรียก run_walk_forward / run_replay บนข้อมูลจริงก่อน LOCK_1600 ถูก commit (gate อยู่ใน run1600.py)
"""

import dataclasses
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from mode1600.locked1600 import (BOOT_BLOCK, BOOT_REPS, BOOT_SEED, CONTROL, EARLY_STOP_FRAC, FEATURES, MAIN, MEAN,
                                 NAIVE, NN_TRACKS, REPLAY_END, REPLAY_START, REPLAY_TRAIN_END, RETRAIN_EVERY, RF,
                                 SEEDS, SUBPERIODS, WF_END, WF_START)
from src import data, evaluate, paths, shrink, train
from src.data import Fold

paths.use_legacy()
from models import get_regressors            # noqa: E402  (legacy · พารามิเตอร์ phase1c-tuned)


# ------------------------------------------------------------------ บล็อก walk-forward
def wf_blocks(calendar, start=WF_START, end=WF_END, every=RETRAIN_EVERY):
    """
    calendar : DatetimeIndex ของวันทำการ (ไฟล์รายวัน) · คืน DataFrame block, start, end
    บล็อกละ `every` วันทำการติดกัน เริ่ม `start` · สองหุ้นใช้ขอบเดียวกัน
    M20 (ข): จำนวนบล็อก = floor(วัน / every) และ **เศษรวมเข้าบล็อกสุดท้าย** (ไม่มีบล็อกสั้นกว่า every)
    — ช่วงจริง 461 วัน → 23 บล็อก บล็อกท้าย 21 วัน (ตัดสินจากปฏิทินก่อนมีผลใด ๆ)
    """
    cal = pd.DatetimeIndex(calendar)
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    n = max(1, len(days) // every)
    return pd.DataFrame([{"block": i + 1, "start": days[i * every],
                          "end": days[-1] if i == n - 1 else days[(i + 1) * every - 1]} for i in range(n)])


# ------------------------------------------------------------------ หนึ่งหุ้น หนึ่งช่วงเทรน
def make_dataset(ticker, prices, table, zone_end):
    """
    Dataset รายวันของ src (ตัดที่ zone_end ก่อนทำอะไร) แล้วแทน y ด้วย target ของ 16:00
    คืน (ds16, F) · F = feature 4 ตัว เรียงตาม ds16.dates (NaN ถ้าไม่มี)
    """
    ds = data.dataset_from_prices(ticker, prices, zone_end)
    t = table.reindex(ds.dates)
    lab = t["labeled"].eq(True).to_numpy(dtype=bool)          # แถวที่ไม่มีในตาราง (NaN) = ไม่ labeled
    y = np.where(lab, t["y"].to_numpy(dtype=float), np.nan)
    ds16 = dataclasses.replace(ds, y=y, prev_close=t["C15"].to_numpy(dtype=float),
                               close=t["close"].to_numpy(dtype=float))
    return ds16, t[list(FEATURES)].to_numpy(dtype=float)


def standardize_extra(F, prepared):
    """feature 4 ตัว standardize ด้วยสถิติของแถว fit เท่านั้น (เหมือน train.standardized_extra)"""
    fit = F[prepared["pos_fit"]]
    mean, std = fit.mean(axis=0), np.maximum(fit.std(axis=0), 1e-8)
    return {k: ((F[prepared[f"pos_{k}"]] - mean) / std).astype(np.float32) for k in ("fit", "val", "eval")}


def run_split(ticker, prices, table, cfgs, *, train_before, eval_dates, zone_end, seeds=SEEDS, k=0, tag=""):
    """
    เทรนทุก track ด้วยแถว labeled ที่วันที่ < `train_before` แล้วทำนาย `eval_dates`
    คืน (preds: DataFrame รายวันทุก track, info: dict ขนาด train / c / epoch)
    cfgs : {MAIN: cfg, CONTROL: cfg} จาก `locked1600.track_configs()`
    """
    ds16, F = make_dataset(ticker, prices, table, zone_end)
    eval_dates = pd.DatetimeIndex(eval_dates)
    ev = ds16.dates.get_indexer(eval_dates)
    assert (ev >= 0).all(), "วันประเมินต้องอยู่ในไฟล์รายวัน"
    train_stop = int(ds16.dates.searchsorted(pd.Timestamp(train_before), side="left"))
    assert ev.min() >= train_stop, "วันประเมินต้องไม่อยู่ก่อนขอบ train"
    fold = Fold(k=k, eval_pos=ev, train_stop=train_stop)

    # ภาพ + fit/val ใช้ชุดเดียวกันทั้งหลักและ control (needs_features เหมือนกัน → แถวเดียวกัน)
    prepared = data.fold_inputs(ds16, cfgs[MAIN], fold, frac=EARLY_STOP_FRAC)
    assert prepared["pos_fit"].max() < train_stop and prepared["pos_val"].max() < train_stop
    extra = standardize_extra(F, prepared)
    y_eval, c15 = prepared["y_eval"], ds16.prev_close[ev]
    info = {"ticker": ticker, "block": k, "tag": tag, "n_fit": len(prepared["pos_fit"]), "n_val": len(prepared["pos_val"]),
            "train_first": str(ds16.dates[prepared["pos_fit"][0]].date()),
            "train_last": str(ds16.dates[prepared["pos_val"][-1]].date()),
            "eval_first": str(eval_dates[0].date()), "eval_last": str(eval_dates[-1].date()), "n_eval": len(ev)}

    preds = {}
    for name in NN_TRACKS:
        cfg = cfgs[name]
        pred, hists, seed_preds, (vp, yv) = train.run_fold(cfg, prepared, seeds=seeds, extra=extra, return_hist=True,
                                                           return_seed_preds=True, return_val_pred=True)
        c = shrink.choose_c(yv, vp)
        preds[name] = (pred, shrink.apply(pred, c))
        info[f"c_{name}"] = c
        info[f"epochs_{name}"] = float(np.mean([h["epochs"] for h in hists]))
        info[f"best_epoch_{name}"] = float(np.mean([h["best_epoch"] for h in hists]))

    # ตัวเทียบ: train = ทุกแถวก่อนบล็อก (fit + early-stop · ไม่มี early stopping เหมือน src/baselines.py · B11)
    tr = np.concatenate([prepared["pos_fit"], prepared["pos_val"]])
    y_tr = ds16.y[tr]
    rf = TransformedTargetRegressor(regressor=get_regressors()["Random Forest"], transformer=StandardScaler())
    rf.fit(F[tr], y_tr)
    preds[RF] = (np.asarray(rf.predict(F[ev]), dtype=float), None)
    preds[NAIVE] = (evaluate.naive_prediction(y_eval), None)
    preds[MEAN] = (evaluate.mean_return_prediction(y_tr, len(ev)), None)

    frames = []
    for name, (p, ps) in preds.items():
        f = evaluate.predictions_frame(eval_dates, ticker, k, y_eval, p, c15, model=name, tag=tag)
        f["pred_shrunk"] = ps if ps is not None else p
        frames.append(f)
    return pd.concat(frames, ignore_index=True), info


def _wf_unit(args):
    ticker, prices, table, cfgs, row, seeds = args
    lab = table.index[table["labeled"].to_numpy(dtype=bool)]
    ev = lab[(lab >= row["start"]) & (lab <= row["end"])]
    if len(ev) == 0:
        return None
    return run_split(ticker, prices, table, cfgs, train_before=row["start"], eval_dates=ev, zone_end=row["end"],
                     seeds=seeds, k=int(row["block"]), tag="walk_forward")


def run_walk_forward(inputs, cfgs, blocks, *, seeds=SEEDS, jobs=1):
    """
    inputs : {ticker: (prices รายวัน, ตาราง history_table)} · blocks : จาก wf_blocks
    คืน (preds, info) ของทุกหุ้น × ทุกบล็อก · jobs > 1 = หลาย process (แต่ละ process 1 thread · seed เดิม → ผลเท่ากัน)
    """
    tasks = [(t, p, tab, cfgs, row, tuple(seeds)) for t, (p, tab) in inputs.items() for _, row in blocks.iterrows()]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outs = list(pool.map(_wf_unit, tasks))
    else:
        outs = [_wf_unit(a) for a in tasks]
    outs = [o for o in outs if o is not None]
    return pd.concat([o[0] for o in outs], ignore_index=True), pd.DataFrame([o[1] for o in outs])


def run_replay(ticker, prices, hist_table, replay_tables, cfgs, *, seeds=SEEDS,
               train_end=REPLAY_TRAIN_END, start=REPLAY_START, zone_end=REPLAY_END):
    """
    เทรนด้วยแถว labeled ≤ REPLAY_TRAIN_END (ไฟล์ล็อก) · ทำนายวัน replay ของแต่ละตาราง (ทาง (i), (ii))
    replay_tables : {"i": ตาราง, "ii": ตาราง} · โมเดลของทุกทางเทรนจากข้อมูลชุดเดียวกัน (seed เดิม → โมเดลเดียวกัน)
    train_end / start / zone_end : ค่าเริ่มต้น = ค่าที่ล็อก (เปลี่ยนได้เฉพาะ test ข้อมูลสังเคราะห์)
    คอลัมน์ provenance (bar_source, snapshot_*) ถูกแนบในคำทำนายทุกแถว
    """
    train_part = hist_table[hist_table.index <= pd.Timestamp(train_end)]
    frames, infos = [], []
    for way, rt in replay_tables.items():
        lab = rt[rt["labeled"].to_numpy(dtype=bool)]
        if len(lab) == 0:
            continue
        cols = [c for c in train_part.columns if c in lab.columns]
        table = pd.concat([train_part[cols], lab[cols]])
        p, info = run_split(ticker, prices, table, cfgs, train_before=start, eval_dates=lab.index,
                            zone_end=zone_end, seeds=seeds, k=0, tag=f"replay_{way}")
        prov = [c for c in ("bar_source", "snapshot_file", "snapshot_downloaded_at", "snapshot_sha256",
                            "c15_final", "y_final_base") if c in lab.columns]
        p = p.merge(lab[prov].rename_axis("date").reset_index(), on="date", how="left")
        frames.append(p)
        infos.append(info)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(infos)


# ------------------------------------------------------------------ ตัววัด · DM · F3
def ic(y, p):
    """Pearson IC · คำทำนายคงที่ → 0 (B39)"""
    if evaluate.is_constant_prediction(y, p) or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(p, y)[0, 1])


def bootstrap_delta_ic(y, a, b, *, block=BOOT_BLOCK, reps=BOOT_REPS, seed=BOOT_SEED):
    """
    ΔIC = IC(a) − IC(b) บนวันเดียวกัน + 95% CI แบบ block bootstrap (PLAN ข้อ f)
    แบ่งวันตามเวลาเป็นบล็อกติดกันบล็อกละ `block` วัน · สุ่มบล็อกแบบมีคืน (จำนวนบล็อกเท่าเดิม) · percentile 2.5/97.5
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
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_IC": d, "ci_lo": float(lo), "ci_hi": float(hi), "supported": bool(d > 0 and lo > 0),
            "n_days": n, "block": block, "reps": reps, "seed": seed}


def periods(preds):
    """ช่วงที่รายงาน: ทั้งหมด + ช่วงย่อยของ PLAN ข้อ c (เฉพาะ walk-forward)"""
    out = [("all", None, None)]
    if (preds["tag"] == "walk_forward").all():
        out += [(f"{a}..{b}", a, b) for a, b in SUBPERIODS]
    return out


def summarize(preds):
    """ตัววัดต่อ track × หุ้น × ช่วง (+ ค่าหลังปรับ c ของ NN) + DM vs Naive (ต่อหุ้น)"""
    rows = []
    for pname, a, b in periods(preds):
        P = preds if a is None else preds[(preds["date"] >= pd.Timestamp(a)) & (preds["date"] <= pd.Timestamp(b))]
        for ticker, g in P.groupby("ticker"):
            naive = g[g["model"] == NAIVE].sort_values("date")
            for model, m in g.groupby("model"):
                m = m.sort_values("date")
                row = {"period": pname, "ticker": ticker, "model": model,
                       **evaluate.metrics(m["y"], m["pred"], m["prev_close"], const_ic_zero=True)}
                if model in NN_TRACKS:
                    s = evaluate.metrics(m["y"], m["pred_shrunk"], m["prev_close"])
                    row.update({"relMAE_shrunk": s["relMAE"], "relMAE_baht_shrunk": s["relMAE_baht"]})
                if model != NAIVE:
                    dm = evaluate.dm_test(m, naive)
                    row.update({"DM": dm["DM"], "p_DM": dm["p_DM"], "DM_mean_d": dm["mean_d"]})
                rows.append(row)
    return pd.DataFrame(rows)


def f3(preds, **boot):
    """F3a: ΔIC(หลัก − control) ต่อหุ้น บนช่วงประเมินทั้งหมด + ช่วงย่อย (descriptive) · CI จาก bootstrap"""
    rows = []
    for pname, a, b in periods(preds):
        P = preds if a is None else preds[(preds["date"] >= pd.Timestamp(a)) & (preds["date"] <= pd.Timestamp(b))]
        for ticker, g in P.groupby("ticker"):
            m = g[g["model"] == MAIN].set_index("date").sort_index()
            c = g[g["model"] == CONTROL].set_index("date").sort_index()
            assert m.index.equals(c.index) and np.allclose(m["y"], c["y"]), "หลักกับ control ต้องเป็นวันเดียวกัน"
            rows.append({"period": pname, "ticker": ticker,
                         **bootstrap_delta_ic(m["y"], m["pred"], c["pred"], **boot)})
    return pd.DataFrame(rows)
