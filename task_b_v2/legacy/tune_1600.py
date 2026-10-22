"""
tune_1600.py — Phase 1D (Mode A) model search ตาม PHASE1D_PLAN.md
=================================================================

    python tune_1600.py --smoke    # ข้อมูลสังเคราะห์เท่านั้น เขียนผลลง temp dir (ไม่แตะ repo)
    python tune_1600.py --run      # ข้อมูลจริง -- ใช้ได้เมื่อแผน + โค้ด commit แล้วเท่านั้น

*** --run ต้องรันหลัง human commit PHASE1D_PLAN.md + โค้ดทั้งหมดแล้วเท่านั้น ***
commit นั้นคือหลักฐานว่าแผนถูกเขียน "ก่อน" เห็นผล · สคริปต์ปฏิเสธถ้าไฟล์ที่ล็อกยังไม่ commit

    python tune_1600.py --smoke --round r2   # รอบที่ 2 (PHASE1D_PLAN_R2.md): target = ราคาปิดทางการ · ข้อมูลสังเคราะห์
    python tune_1600.py --run --round r2     # รอบที่ 2 ข้อมูลจริง -> results/phase1d_r2/ (ต้อง commit แผน R2 + โค้ดก่อน)

ไม่มี historical test: ข้อมูลทั้งหมดเป็น development ที่เคย probe แล้ว
ตัวเลขที่ได้ = walk-forward บน development data ห้ามเรียกว่า test
ห้ามวางตัวเลข Phase 1D ในตารางเดียวกับ daily model
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from models import SeedAveragedRegressor, _scaled, _unscaled
from splits import walk_forward_splits
from intraday_1600 import build_dataset, load_ticker_bars, BASE_DIR, BANGKOK
from daily_close import DAILY_DIR, load_locked, locked_path
import metrics_1600 as M

TICKERS = ["KBANK.BK", "ADVANC.BK"]
N_SPLITS = 3
MIN_TRAIN = 360
TIE_TOL = 1e-3              # หน่วย relMAE (= 0.1% ของ Naive MAE)

SEARCH_ANN_SEEDS = [0, 1, 2]          # เลือก config + ตัดสิน guard
LIVE_ANN_SEEDS = [0, 1, 2]            # โมเดลที่ deploy = โมเดลที่ผ่าน selection
SENSITIVITY_ANN_SEEDS = list(range(10))   # refit ผู้ชนะ รายงานแยก ไม่เปลี่ยนอะไร

ANN_FIXED = {"activation": "relu", "solver": "adam", "learning_rate_init": 1e-3,
             "max_iter": 3000, "n_iter_no_change": 20, "early_stopping": False}
RF_FIXED = {"n_estimators": 400, "n_jobs": -1, "random_state": 42}
XGB_FIXED = {"n_estimators": 300, "colsample_bytree": 0.8, "reg_lambda": 1.0,
             "random_state": 42, "n_jobs": -1}

GRIDS = {
    "ANN (MLP)": [{"alpha": a, "hidden_layer_sizes": h}
                  for a in [1, 3, 7, 20] for h in [(8,), (8, 4)]],
    "Random Forest": [{"max_depth": d, "min_samples_leaf": l, "max_features": f}
                      for d in [3, 5] for l in [15, 30] for f in [0.5, 1.0]],
    "XGBoost": [{"max_depth": d, "learning_rate": lr, "subsample": ss}
                for d in [2, 3] for lr in [0.01, 0.05] for ss in [0.8, 1.0]],
}
assert sum(len(g) for g in GRIDS.values()) == 24

# ไฟล์ที่ต้อง commit ก่อน --run (pre-registration)
LOCKED_FILES = ["PHASE1D_PLAN.md", "intraday_1600.py", "tune_1600.py", "metrics_1600.py",
                "models.py", "splits.py", "trading_costs.py", "config.py", "weighting.py"]
# รอบที่ 2: เพิ่มแผน R2 + โมดูลราคาปิดรายวัน + ไฟล์ล็อก (grid / rule / metric เหมือนรอบ 1 ทุกตัวอักษร)
# (path ของไฟล์ล็อกราคาปิดคำนวณจาก daily_close -- ไม่เขียนเป็น string ตรงนี้ เพราะ test_16 ของรอบ 1 ห้ามมีคำว่า raw_data ใน tune_1600.py)
LOCKED_FILES_R2 = LOCKED_FILES + ["PHASE1D_PLAN_R2.md", "daily_close.py", "tools/make_locked_daily.py"] + [
    p.relative_to(BASE_DIR).as_posix()
    for p in [locked_path(t) for t in ("KBANK.BK", "ADVANC.BK")] + [DAILY_DIR / "README.md"]]


# ---------------------------------------------------------------
# models
# ---------------------------------------------------------------
def make_model(family, cfg, ann_seeds=SEARCH_ANN_SEEDS):
    if family == "ANN (MLP)":
        return _scaled(SeedAveragedRegressor(MLPRegressor(**{**ANN_FIXED, **cfg}),
                                             seeds=tuple(ann_seeds)))
    if family == "Random Forest":
        return _unscaled(RandomForestRegressor(**{**RF_FIXED, **cfg}))
    if family == "XGBoost":
        return _unscaled(XGBRegressor(**{**XGB_FIXED, **cfg}))
    raise KeyError(family)


def make_live_model(family, cfg):
    """factory ของโมเดล live / dry-run -- ANN ใช้ LIVE_ANN_SEEDS [0, 1, 2]"""
    return make_model(family, cfg, LIVE_ANN_SEEDS)


def fit_predict(model, X_tr, y_tr, X_ev):
    """fit feature scaler + target scaler + model จาก train fold เท่านั้น · ไม่ refit ใน fold"""
    wrapped = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    wrapped.fit(X_tr, y_tr)
    return wrapped.predict(X_ev), wrapped


# ---------------------------------------------------------------
# data
# ---------------------------------------------------------------
def prepare(datasets):
    """datasets = {ticker: (X, y, info, report)} -> เพิ่ม folds + omitted dates"""
    out = {}
    for t, (X, y, info, rep) in datasets.items():
        folds = list(walk_forward_splits(X, n_splits=N_SPLITS, min_train=MIN_TRAIN))
        assert len(folds) == N_SPLITS
        for tr, ev in folds:
            assert tr[-1] < ev[0]
        last = folds[-1][1][-1]
        omitted = [str(d.date()) for d in X.index[last + 1:]]
        out[t] = {"X": X, "y": y, "C15": info["C15"], "folds": folds,
                  "omitted": omitted, "report": rep}
    return out


def fold_table(data):
    rows = []
    for t, d in data.items():
        X = d["X"]
        for k, (tr, ev) in enumerate(d["folds"], 1):
            rows.append({"ticker": t, "fold": k, "train_rows": len(tr),
                         "train_start": str(X.index[tr[0]].date()),
                         "train_end": str(X.index[tr[-1]].date()),
                         "eval_rows": len(ev), "eval_start": str(X.index[ev[0]].date()),
                         "eval_end": str(X.index[ev[-1]].date())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------
def evaluate(family, cid, cfg, data, ann_seeds=SEARCH_ANN_SEEDS, stage="search"):
    rows = []
    for t, d in data.items():
        X, y, c15 = d["X"], d["y"], d["C15"]
        for k, (tr, ev) in enumerate(d["folds"], 1):
            pred, _ = fit_predict(make_model(family, cfg, ann_seeds),
                                  X.iloc[tr], y.iloc[tr], X.iloc[ev])
            y_ev, c_ev = y.iloc[ev], c15.iloc[ev]
            row = {"stage": stage, "model": family, "config_id": cid, "config": str(cfg),
                   "ticker": t, "fold": k, "n_seeds": len(ann_seeds) if "ANN" in family else np.nan}
            finite = bool(np.isfinite(pred).all())
            row["pred_finite"] = finite
            nm = M.naive_mae(y_ev)
            row["naive_MAE_return"] = nm
            if finite:
                row.update(M.main_metrics(y_ev, pred, c_ev))
                row["relMAE"] = M.rel_mae(row["MAE_return"], nm)   # Naive MAE = 0 -> หยุด
                row["R2_OOS"] = M.r2_oos(y_ev, pred)
                row.update(M.shape(y_ev, pred))
            print(f"[{stage}][{family}][{cid + 1}/{len(GRIDS[family])}][{t}][fold {k}] "
                  + (f"relMAE={row['relMAE']:.4f} StdRatio={row['StdRatio']:.4f}"
                     if finite else "prediction non-finite -> config invalid"), flush=True)
            rows.append(row)
    return rows


def score(evals):
    out = []
    for (fam, cid), g in evals.groupby(["model", "config_id"], sort=False):
        valid = bool(g["pred_finite"].all())
        r = {"model": fam, "config_id": cid, "config": g["config"].iloc[0], "valid": valid,
             "n_evals": len(g)}
        if len(g) != M.N_EVALS:
            raise M.IncompleteEvaluations(f"{fam} config {cid}: {len(g)} evaluations")
        if valid:
            r["primary_score"] = M.primary_score(g["relMAE"])
            r["mean_MAE_return"] = float(g["MAE_return"].mean())
            r["min_StdRatio"] = float(g["StdRatio"].min())
            r["passed_guard"] = M.passes_guard(g["StdRatio"])
        else:
            r.update(primary_score=np.nan, mean_MAE_return=np.nan, min_StdRatio=np.nan,
                     passed_guard=False)
        out.append(r)
    return pd.DataFrame(out)


def tie_key(family, cfg, s):
    if family == "ANN (MLP)":
        return (sum(cfg["hidden_layer_sizes"]), -cfg["alpha"])
    return (cfg["max_depth"], s)


def select_winner(scores, family):
    """คืน config_id ของผู้ชนะ หรือ None ถ้า family นี้ไม่มี valid config"""
    ok = scores[(scores["model"] == family) & scores["valid"] & scores["passed_guard"]]
    if ok.empty:
        return None
    best = ok["primary_score"].min()
    tied = ok[ok["primary_score"] - best < TIE_TOL]
    win = min(tied.itertuples(),
              key=lambda r: tie_key(family, GRIDS[family][r.config_id], r.primary_score))
    return int(win.config_id)


def baseline_rows(data):
    """Naive (ŷ = 0) และ Mean Return (mean y ของ train fold) -- reference ไม่ผ่าน guard"""
    rows = []
    for t, d in data.items():
        X, y, c15 = d["X"], d["y"], d["C15"]
        for k, (tr, ev) in enumerate(d["folds"], 1):
            y_ev, c_ev = y.iloc[ev], c15.iloc[ev]
            for name, pred in (("Baseline: Naive", np.zeros(len(ev))),
                               ("Baseline: Mean Return", np.full(len(ev), y.iloc[tr].mean()))):
                r = {"model": name, "ticker": t, "fold": k, **M.main_metrics(y_ev, pred, c_ev),
                     "relMAE": M.rel_mae(float(np.mean(np.abs(pred - y_ev))), M.naive_mae(y_ev)),
                     "R2_OOS": M.r2_oos(y_ev, pred), **M.shape(y_ev, pred),
                     **M.economic_diagnostics(y_ev, pred, c_ev)}
                rows.append(r)
    return rows


def econ_rows(family, cid, cfg, data, ann_seeds=SEARCH_ANN_SEEDS):
    rows = []
    for t, d in data.items():
        X, y, c15 = d["X"], d["y"], d["C15"]
        for k, (tr, ev) in enumerate(d["folds"], 1):
            pred, _ = fit_predict(make_model(family, cfg, ann_seeds),
                                  X.iloc[tr], y.iloc[tr], X.iloc[ev])
            rows.append({"model": family, "config_id": cid, "ticker": t, "fold": k,
                         **M.economic_diagnostics(y.iloc[ev], pred, c15.iloc[ev])})
    return rows


def run_search(data, out_dir, round_name="r1"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = fold_table(data)
    folds.to_csv(out_dir / "phase1d_folds.csv", index=False)
    for t, d in data.items():
        print(f"[folds] {t}: omitted evaluation dates = {d['omitted'] or 'none'} "
              "(ไม่ใช่ holdout/test · ใช้เทรน live ได้)")

    evals = []
    for fam, grid in GRIDS.items():
        for cid, cfg in enumerate(grid):
            evals += evaluate(fam, cid, cfg, data)
    evals = pd.DataFrame(evals)
    scores = score(evals)
    winners = {fam: select_winner(scores, fam) for fam in GRIDS}
    scores["winner"] = [winners[r.model] == r.config_id for r in scores.itertuples()]

    sens = []
    if winners["ANN (MLP)"] is not None:
        cid = winners["ANN (MLP)"]
        sens = evaluate("ANN (MLP)", cid, GRIDS["ANN (MLP)"][cid], data,
                        ann_seeds=SENSITIVITY_ANN_SEEDS, stage="sensitivity10")
    evals_all = pd.concat([evals, pd.DataFrame(sens)], ignore_index=True)
    evals_all.to_csv(out_dir / "phase1d_evals.csv", index=False)
    scores.to_csv(out_dir / "phase1d_scores.csv", index=False)
    pd.DataFrame(baseline_rows(data)).to_csv(out_dir / "phase1d_baselines.csv", index=False)
    econ = []
    for fam, cid in winners.items():
        if cid is not None:
            econ += econ_rows(fam, cid, GRIDS[fam][cid], data)
    pd.DataFrame(econ).to_csv(out_dir / "phase1d_econ.csv", index=False)

    title = ("# Phase 1D รอบ 2 (target = ราคาปิดทางการ) search summary (development / walk-forward — ไม่ใช่ test)"
             if round_name == "r2" else
             "# Phase 1D search summary (development / walk-forward — ไม่ใช่ test)")
    lines = [title, ""]
    for fam, cid in winners.items():
        if cid is None:
            lines.append(f"- {fam}: **ไม่มี valid config** (ทุก config ไม่ผ่าน guard/invalid)")
        else:
            s = scores[(scores.model == fam) & (scores.config_id == cid)].iloc[0]
            lines.append(f"- {fam}: `{GRIDS[fam][cid]}` primary_score (mean relMAE) = "
                         f"{s.primary_score!r} · mean MAE_return = {s.mean_MAE_return!r} · "
                         f"min StdRatio = {s.min_StdRatio!r}")
    if round_name == "r2":
        zero = pd.concat([d["y"] for d in data.values()])
        lines += ["", f"ราคาปิดทางการ = C15 ใน {100 * float((zero == 0).mean()):.1f}% ของวัน (ทุกแถวหลังตัด warm-up) "
                  "⇒ MAE เอื้อ prediction ที่ใกล้ 0 · StdRatio ต่ำของผู้ชนะเป็นสิ่งที่คาดได้ ไม่ใช่หลักฐานของ skill · "
                  "ห้ามวางตัวเลขรอบ 2 ในตารางเดียวกับรอบ 1 หรือ daily model (คนละ target)", ""]
    else:
        lines += ["", "~43–49% ของวันราคาไม่ขยับเลย ⇒ MAE เอื้อ prediction ที่ใกล้ 0 · "
                  "StdRatio ต่ำของผู้ชนะเป็นสิ่งที่คาดได้ ไม่ใช่หลักฐานของ skill", ""]
    (out_dir / "phase1d_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[tune_1600] เขียนผลที่ {out_dir}")
    return scores, winners


# ---------------------------------------------------------------
# synthetic data (สำหรับ --smoke และ tests เท่านั้น)
# ---------------------------------------------------------------
def synthetic_bars(n_days=420, seed=0, start_price=250.0, start="2023-01-02"):
    """แท่งรายชั่วโมงสังเคราะห์รูปแบบเดียวกับไฟล์ Yahoo (มีแท่ง 13 บ้าง · V10 = 0)"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    rows, price = [], start_price
    for d in days:
        for h in (10, 11, 12, 13, 14, 15, 16):
            if h == 13 and rng.random() < 0.5:
                continue
            o = price
            price = max(1.0, round(price * (1 + rng.normal(0, 0.003)) * 4) / 4)
            hi, lo = max(o, price) + 0.25, min(o, price) - 0.25
            vol = 0 if h == 10 else int(rng.integers(100_000, 2_000_000))
            ts = pd.Timestamp(d.year, d.month, d.day, h, tz=BANGKOK)
            rows.append({"Datetime": ts.isoformat(sep=" "), "Adj Close": price, "Close": price,
                         "High": hi, "Low": lo, "Open": o, "Volume": vol})
    return pd.DataFrame(rows)


def synthetic_daily(bars, seed=0):
    """ราคาปิดทางการสังเคราะห์ = close แท่ง 16 ± 0-2 tick (สำหรับ --smoke --round r2 และ tests เท่านั้น)"""
    rng = np.random.default_rng(1000 + seed)
    c16 = bars[bars["hour"] == 16].set_index("date")["Close"].sort_index()
    return c16 + rng.choice([-0.5, -0.25, 0.0, 0.0, 0.25, 0.5], size=len(c16))


def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} ล้มเหลว: {r.stderr.strip()}")
    return r.stdout.strip()


def check_preregistered(files=LOCKED_FILES):
    """--run: แผน + โค้ดทุกไฟล์ที่ล็อกต้อง tracked และไม่มีการแก้ค้าง"""
    for f in files:
        _git("ls-files", "--error-unmatch", f)
    dirty = _git("status", "--porcelain", "--", *files)
    if dirty:
        raise SystemExit("ปฏิเสธ --run: ไฟล์ที่ล็อกยังไม่ commit\n" + dirty)


def main():
    p = argparse.ArgumentParser(description="Phase 1D (Mode A) search -- ไม่มี test")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="ข้อมูลสังเคราะห์ · เขียนลง temp dir")
    g.add_argument("--run", action="store_true", help="ข้อมูลจริง (ต้อง commit แผนก่อน)")
    p.add_argument("--round", choices=["r1", "r2"], default="r1",
                   help="r1 = target close_bar16 (เดิม) · r2 = target ราคาปิดทางการ (PHASE1D_PLAN_R2.md)")
    p.add_argument("--out", default=None, help="โฟลเดอร์ผล (default: temp ตอน smoke)")
    args = p.parse_args()
    r2 = args.round == "r2"

    if args.smoke:
        datasets = {}
        for i, t in enumerate(TICKERS):
            from intraday_1600 import prepare_bars
            bars = prepare_bars(synthetic_bars(seed=i, start_price=250 + 100 * i))
            datasets[t] = build_dataset(bars, synthetic_daily(bars, seed=i) if r2 else None)
        out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="phase1d_smoke_"))
        run_search(prepare(datasets), out, args.round)
        print("[smoke] จบ -- ข้อมูลสังเคราะห์ ตัวเลขไม่มีความหมาย")
        return
    check_preregistered(LOCKED_FILES_R2 if r2 else LOCKED_FILES)
    if r2:
        datasets = {t: build_dataset(load_ticker_bars(t), load_locked(t)["Close"]) for t in TICKERS}
        default_out = BASE_DIR / "results" / "phase1d_r2"
    else:
        datasets = {t: build_dataset(load_ticker_bars(t)) for t in TICKERS}
        default_out = BASE_DIR / "results" / "phase1d"
    run_search(prepare(datasets), Path(args.out) if args.out else default_out, args.round)


if __name__ == "__main__":
    sys.exit(main())
