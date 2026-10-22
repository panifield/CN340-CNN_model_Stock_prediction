"""
tune.py — จูน hyperparameter บน train+val เท่านั้น (feedback ข้อ 11)
====================================================================

    python tune.py --stage 1     # half-life (5 configs ต่อโมเดล)
    python tune.py --stage 2     # hyperparameter (12 configs ต่อโมเดล) + refit ANN 10 seeds

*** ไฟล์นี้ต้องไม่มีทางแตะ test set ได้เลยในเชิงโครงสร้าง ***
มัน truncate ข้อมูลถึง SPLIT_BY_DATE["val_end"] ก่อนสร้าง feature
แล้ว assert ว่าแถวสุดท้ายไม่เกินวันนั้น -- ไม่มี flag ใดเปิด test ได้

ผลลัพธ์ทั้งหมดถูกเขียนลง results/tuning_*.csv แบบ full precision
รวมทุก config ที่ลอง ไม่ใช่แค่ตัวที่ชนะ

แผนทั้งหมด (grid / validation / เกณฑ์ / guard / กฎเสมอ / คำทำนาย)
ประกาศและ commit ไว้ก่อนรันใน TUNING_PLAN.md -- โค้ดนี้ต้องทำตามไฟล์นั้นเป๊ะ
"""

import argparse
from datetime import date

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import (
    TICKERS, OUTPUT_DIR, SPLIT_BY_DATE,
    ANN_PARAMS, ANN_SEEDS, RF_PARAMS, XGB_PARAMS,
)
from data_loader import load_stock
from features import build_features
from targets import build_targets
from splits import prepare_xy, walk_forward_splits
from models import SeedAveragedRegressor, _scaled, _unscaled
from weighting import recency_weights
from evaluate import regression_metrics, prediction_shape

# ---------------------------------------------------------------
# ค่าที่ประกาศไว้ใน TUNING_PLAN.md -- ห้ามแก้หลังเห็นผล
# ---------------------------------------------------------------
N_SPLITS = 3
MIN_TRAIN = 1250
GUARD_MIN_STDRATIO = 0.05
TIE_TOL = 1e-5
TUNING_ANN_SEEDS = [0, 1, 2]     # ตอนจูน -- ลดเวลารัน · refit ผู้ชนะด้วย ANN_SEEDS (10)

HALF_LIFE_GRID = [None, 1000, 500, 250, 125]   # หน่วย = วันทำการ

ANN_GRID = [                      # 4 alpha x 3 hidden = 12
    {"alpha": a, "hidden_layer_sizes": h}
    for a in [0.3, 1.0, 3.0, 7.0]
    for h in [(8, 4), (16, 8), (32, 16)]
]
RF_GRID = [                       # 3 depth x 2 leaf x 2 max_features = 12
    {"max_depth": d, "min_samples_leaf": l, "max_features": f}
    for d in [4, 8, 12]
    for l in [20, 50]
    for f in [1.0, 0.5]
]
XGB_GRID = [                      # 3 depth x 2 lr x 2 subsample = 12
    {"max_depth": d, "learning_rate": lr, "subsample": ss}
    for d in [2, 3, 4]
    for lr in [0.01, 0.05]
    for ss in [0.8, 1.0]
]
MODEL_NAMES = ["ANN (MLP)", "Random Forest", "XGBoost"]
GRIDS = {"ANN (MLP)": ANN_GRID, "Random Forest": RF_GRID, "XGBoost": XGB_GRID}
PARAM_COLS = ["alpha", "hidden_layer_sizes", "max_depth", "min_samples_leaf",
              "max_features", "learning_rate", "subsample"]

STAGE1_CSV = OUTPUT_DIR / "tuning_stage1_halflife.csv"
STAGE2_CSV = OUTPUT_DIR / "tuning_stage2_params.csv"
SCORES_CSV = OUTPUT_DIR / "tuning_scores.csv"
SUMMARY_MD = OUTPUT_DIR / "tuning_summary.md"


# ---------------------------------------------------------------
# ข้อมูล -- เตรียมเหมือน main.py ทุกอย่าง แยกต่อหุ้น
# ---------------------------------------------------------------
def load_dev_data():
    val_end = pd.Timestamp(SPLIT_BY_DATE["val_end"])
    data = {}
    for ticker in TICKERS:
        df = load_stock(ticker, verbose=False)
        # 1) ตัดข้อมูลก่อนสร้าง feature -- เหมือนที่ main.py ทำในโหมด dev
        df = df.loc[:val_end]
        X = build_features(df, verbose=False)
        targets = build_targets(df, verbose=False)
        # 2) ใช้ฟังก์ชันร่วมตัวเดียวกับ main.py -- ห้ามเขียน logic เตรียมข้อมูลใหม่
        X, y, extra = prepare_xy(X, targets["y_return"],
                                 targets[["prev_close", "close"]], verbose=False)
        # 3) assert เชิงโครงสร้าง
        assert X.index[-1] <= val_end, "tune.py เห็นแถวที่เลย val_end -- หยุดทันที"
        folds = list(walk_forward_splits(X, n_splits=N_SPLITS, min_train=MIN_TRAIN))
        assert len(folds) == N_SPLITS
        for tr, ev in folds:
            assert tr[-1] < ev[0], "fold ข้ามเวลา"
        print(f"[data] {ticker}: {len(X)} แถว ({X.index[0].date()} -> {X.index[-1].date()}) "
              f"· folds: " + " | ".join(
                  f"train[0:{len(tr)}] eval[{ev[0]}:{ev[-1] + 1}]" for tr, ev in folds))
        data[ticker] = (X, y, folds)
    return data


def make_model(name, override=None, ann_seeds=TUNING_ANN_SEEDS):
    """โครงเดียวกับ models.get_regressors() ต่างแค่พารามิเตอร์ที่ override"""
    override = override or {}
    if name == "ANN (MLP)":
        return _scaled(SeedAveragedRegressor(MLPRegressor(**{**ANN_PARAMS, **override}),
                                             seeds=tuple(ann_seeds)))
    if name == "Random Forest":
        return _unscaled(RandomForestRegressor(**{**RF_PARAMS, **override}))
    if name == "XGBoost":
        return _unscaled(XGBRegressor(**{**XGB_PARAMS, **override}))
    raise KeyError(name)


# ---------------------------------------------------------------
# ประเมิน 1 config = 6 evaluation (2 หุ้น x 3 folds)
# ---------------------------------------------------------------
def evaluate_config(stage, name, config_id, n_configs, half_life, override, data,
                    ann_seeds=TUNING_ANN_SEEDS):
    rows = []
    for ticker, (X, y, folds) in data.items():
        for k, (tr, ev) in enumerate(folds, start=1):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            X_ev, y_ev = X.iloc[ev], y.iloc[ev]
            # น้ำหนัก recency คำนวณใหม่ทุก fold ทุกหุ้น (§4.5.2)
            w = recency_weights(len(X_tr), half_life)
            if w is not None:
                assert len(w) == len(X_tr)
                assert np.isclose(w.mean(), 1.0)
                assert w[-1] == w.max()
            fit_params = {} if w is None else {"model__sample_weight": w}
            wrapped = TransformedTargetRegressor(
                regressor=make_model(name, override, ann_seeds),
                transformer=StandardScaler(),
            )
            wrapped.fit(X_tr, y_tr, **fit_params)
            pred = wrapped.predict(X_ev)
            mae = regression_metrics(y_ev, pred)["MAE_return"]
            shape = prediction_shape(y_ev, pred)
            print(f"[stage{stage}][{name}][{config_id + 1}/{n_configs}][{ticker}][fold {k}] "
                  f"MAE_return={mae:.6f} StdRatio={shape['StdRatio']:.4f}", flush=True)
            row = {"stage": stage, "model": name, "config_id": config_id,
                   "half_life": half_life, "ticker": ticker, "fold": k,
                   "train_rows": len(tr), "eval_rows": len(ev),
                   "MAE_return": mae, "StdRatio": shape["StdRatio"], "Bias": shape["Bias"],
                   "n_seeds": len(ann_seeds) if name == "ANN (MLP)" else np.nan}
            for c in PARAM_COLS:
                if c in (override or {}):
                    v = override[c]
                    row[c] = str(v) if isinstance(v, tuple) else v
            rows.append(row)
    return rows


def score_configs(evals):
    """หนึ่งแถวต่อ config: primary_score = mean MAE_return ของ 6 evaluation"""
    g = evals.groupby(["stage", "model", "config_id"], sort=False)
    scores = g.agg(primary_score=("MAE_return", "mean"),
                   min_StdRatio=("StdRatio", "min"),
                   n_evals=("MAE_return", "size")).reset_index()
    assert (scores["n_evals"] == 2 * N_SPLITS).all()
    # guard: StdRatio >= 0.05 ทุกหุ้นทุก fold (NaN = ไม่ผ่าน)
    passed = g["StdRatio"].apply(lambda s: bool((s >= GUARD_MIN_STDRATIO).all()))
    scores["passed_guard"] = passed.values
    scores["rank"] = np.nan
    for m, idx in scores.groupby("model").groups.items():
        sub = scores.loc[idx]
        ok = sub[sub["passed_guard"]]
        scores.loc[ok.index, "rank"] = ok["primary_score"].rank(method="min")
    return scores.drop(columns="n_evals")


def tie_key(stage, name, cfg, primary):
    """ลำดับความเรียบง่ายตาม TUNING_PLAN ข้อ 6 (ค่าน้อย = ถูกเลือกก่อน)"""
    if stage == 1:
        return (HALF_LIFE_GRID.index(cfg["half_life"]),)
    if name == "ANN (MLP)":
        return (sum(cfg["hidden_layer_sizes"]), -cfg["alpha"])
    return (cfg["max_depth"], primary)


def select_winner(scores, stage, name, configs):
    sub = scores[(scores["stage"] == stage) & (scores["model"] == name)]
    ok = sub[sub["passed_guard"]]
    if ok.empty:
        raise SystemExit(
            f"\n!! HARD STOP: ทุก config ของ {name} ใน stage {stage} ไม่ผ่าน guard "
            f"StdRatio >= {GUARD_MIN_STDRATIO}\n"
            "   ห้ามลดเกณฑ์หลังเห็นผล -- รายงานว่าโมเดลนี้ภายใต้ grid นี้ยุบเป็นค่าคงที่ทุกค่า"
        )
    best = ok["primary_score"].min()
    tied = ok[ok["primary_score"] - best < TIE_TOL]
    win = min(tied.itertuples(),
              key=lambda r: tie_key(stage, name, configs[r.config_id], r.primary_score))
    return int(win.config_id), tied


# ---------------------------------------------------------------
# Stage 1 / Stage 2
# ---------------------------------------------------------------
def stage1_configs():
    return [{"half_life": h} for h in HALF_LIFE_GRID]


def run_stage1(data):
    rows = []
    for name in MODEL_NAMES:
        cfgs = stage1_configs()
        for cid, cfg in enumerate(cfgs):
            rows += evaluate_config(1, name, cid, len(cfgs), cfg["half_life"], None, data)
    evals = pd.DataFrame(rows)
    evals["half_life"] = evals["half_life"].astype("Int64")
    evals.drop(columns=[c for c in PARAM_COLS if c in evals.columns]).to_csv(
        STAGE1_CSV, index=False)
    scores = score_configs(evals)
    scores["half_life"] = [HALF_LIFE_GRID[c] for c in scores["config_id"]]
    scores["half_life"] = scores["half_life"].astype("Int64")
    scores["winner"] = False
    for name in MODEL_NAMES:
        wid, _ = select_winner(scores, 1, name, stage1_configs())
        scores.loc[(scores["model"] == name) & (scores["config_id"] == wid), "winner"] = True
    scores.to_csv(SCORES_CSV, index=False)
    print(f"\n[tune] บันทึก {STAGE1_CSV.name}, {SCORES_CSV.name}")
    print(scores.to_string())


def stage1_winners():
    if not SCORES_CSV.exists():
        raise SystemExit("ยังไม่มีผล stage 1 -- รัน python tune.py --stage 1 ก่อน")
    s = pd.read_csv(SCORES_CSV)
    w = s[(s["stage"].astype(str) == "1") & (s["winner"].astype(str).isin(["True", "1.0"]))]
    assert sorted(w["model"]) == sorted(MODEL_NAMES)
    return {r.model: (None if pd.isna(r.half_life) else int(r.half_life))
            for r in w.itertuples()}


def run_stage2(data):
    half_lives = stage1_winners()
    print(f"[tune] half-life ที่ fix จาก stage 1: {half_lives}")
    rows = []
    for name in MODEL_NAMES:
        grid = GRIDS[name]
        for cid, cfg in enumerate(grid):
            rows += evaluate_config(2, name, cid, len(grid), half_lives[name], cfg, data)
    evals = pd.DataFrame(rows)
    scores = score_configs(evals)
    scores["half_life"] = [half_lives[m] for m in scores["model"]]
    scores["winner"] = False
    winners = {}
    for name in MODEL_NAMES:
        wid, _ = select_winner(scores, 2, name, GRIDS[name])
        winners[name] = wid
        scores.loc[(scores["model"] == name) & (scores["config_id"] == wid), "winner"] = True

    # ANN: winner ถูก freeze แล้ว -> refit ด้วย 10 seeds บน 6 evaluation เดิม (§4.7)
    # ตัวเลขนี้ใช้รายงานเท่านั้น ห้ามใช้เลือก config ใหม่
    name = "ANN (MLP)"
    wid = winners[name]
    print(f"\n[tune] refit ผู้ชนะ ANN (config {wid}) ด้วย {len(ANN_SEEDS)} seeds "
          "-- รายงานเท่านั้น ไม่ใช้เลือกใหม่")
    refit = pd.DataFrame(evaluate_config("2_refit10", name, wid, len(GRIDS[name]),
                                         half_lives[name], GRIDS[name][wid], data,
                                         ann_seeds=ANN_SEEDS))
    evals_all = pd.concat([evals, refit], ignore_index=True)
    evals_all["half_life"] = evals_all["half_life"].astype("Int64")
    evals_all.to_csv(STAGE2_CSV, index=False)

    refit_score = score_configs(refit)
    refit_score["half_life"] = half_lives[name]
    refit_score["winner"] = False            # แถวรายงาน ไม่ใช่ผู้ชนะ (คง dtype bool)
    refit_score["rank"] = np.nan

    old = pd.read_csv(SCORES_CSV)
    old = old[old["stage"].astype(str) == "1"]
    all_scores = pd.concat([old, scores, refit_score], ignore_index=True)
    all_scores["half_life"] = all_scores["half_life"].astype("Int64")
    all_scores.to_csv(SCORES_CSV, index=False)
    print(f"\n[tune] บันทึก {STAGE2_CSV.name}, {SCORES_CSV.name}")
    print(scores.to_string())
    write_summary(all_scores, half_lives, winners)


# ---------------------------------------------------------------
# สรุป
# ---------------------------------------------------------------
def _cfg_str(stage, name, cid):
    if stage == 1:
        return f"half_life={HALF_LIFE_GRID[cid]}"
    return ", ".join(f"{k}={v}" for k, v in GRIDS[name][cid].items())


def write_summary(scores, half_lives, winners):
    # numpy 2 แสดง scalar เป็น np.float64(...) -- ให้ repr เป็นตัวเลขล้วน (full precision เท่าเดิม)
    np.set_printoptions(legacy="1.25")
    s = scores.copy()
    s["stage"] = s["stage"].astype(str)
    # ค่าจาก CSV + NaN ของแถว refit ทำให้เป็น object -> แปลงให้ ~ / == ทำงานถูก
    as_bool = lambda v: str(v) in ("True", "1", "1.0")   # noqa: E731
    s["passed_guard"] = s["passed_guard"].map(as_bool)
    s["winner"] = s["winner"].map(as_bool)
    L = [f"# tuning_summary — รันวันที่ {date.today().isoformat()}", "",
         "> สร้างโดย `tune.py` ตามแผนที่ commit ไว้ก่อนใน `TUNING_PLAN.md`",
         "> primary_score = mean MAE_return ของ 6 evaluation (2 หุ้น x 3 folds) · "
         f"guard = StdRatio >= {GUARD_MIN_STDRATIO} ทุก evaluation · ANN จัดอันดับด้วย 3 seeds",
         "> StdRatio ไม่ถูกใช้จัดอันดับ config การจัดอันดับใช้ mean MAE_return เพียงตัวเดียว "
         "แต่ใช้เป็น validity guard ที่ประกาศไว้ล่วงหน้าเพื่อกันคำตอบเสื่อม (degenerate solution)",
         ""]
    facts = {}
    for name in MODEL_NAMES:
        L.append(f"## {name}")
        base = s[(s["stage"] == "1") & (s["model"] == name) & (s["config_id"] == 0)]
        base_score = float(base["primary_score"].iloc[0])
        for stage in ("1", "2"):
            sub = s[(s["stage"] == stage) & (s["model"] == name)].sort_values("primary_score")
            win = sub[sub["winner"] == True].iloc[0]  # noqa: E712
            losers = sub[(sub["passed_guard"]) & (sub["config_id"] != win["config_id"])]
            cut = sub[~sub["passed_guard"]]
            L.append(f"\n### Stage {stage}")
            L.append("\n| config | primary_score | min_StdRatio | passed_guard | rank |")
            L.append("|---|---|---|---|---|")
            for r in sub.itertuples():
                mark = " **(winner)**" if r.winner == True else ""  # noqa: E712
                L.append(f"| {_cfg_str(int(stage), name, r.config_id)}{mark} | "
                         f"{r.primary_score!r} | {r.min_StdRatio!r} | {r.passed_guard} | "
                         f"{'' if pd.isna(r.rank) else int(r.rank)} |")
            L.append(f"\n- ผู้ชนะ: `{_cfg_str(int(stage), name, win['config_id'])}` "
                     f"primary_score = {win['primary_score']!r}")
            if len(losers):
                c = losers.iloc[0]
                L.append(f"- ค่าที่แพ้ที่ใกล้ที่สุด: `{_cfg_str(int(stage), name, c['config_id'])}` "
                         f"= {c['primary_score']!r} (ต่างกัน {c['primary_score'] - win['primary_score']:+.3e})"
                         + (f" -- คะแนนต่ำกว่าผู้ชนะแต่ต่างไม่ถึง {TIE_TOL:g} จึงตัดสินด้วยกฎเสมอ "
                            "(TUNING_PLAN ข้อ 6)"
                            if c["primary_score"] < win["primary_score"] else ""))
            L.append("- ถูก guard ตัด: " + (", ".join(
                f"`{_cfg_str(int(stage), name, r.config_id)}` (min StdRatio {r.min_StdRatio:.4f})"
                for r in cut.itertuples()) if len(cut) else "ไม่มี"))
            facts[(name, stage)] = (win, cut)
        final = facts[(name, "2")][0]["primary_score"]
        change = (final - base_score) / base_score
        facts[(name, "change")] = change
        L.append(f"\n- เทียบ config Phase 1 (half_life=None, ค่าเดิม) = {base_score!r} → "
                 f"สุดท้าย {final!r} ({change:+.3%} บน validation แบบ walk-forward)")
        if name == "ANN (MLP)":
            r10 = s[s["stage"] == "2_refit10"].iloc[0]["primary_score"]
            L.append(f"- ANN ผู้ชนะ: 3 seeds = {final!r} · 10 seeds = {r10!r} "
                     f"(ต่าง {r10 - final:+.3e}) -- ตัวเลข 10 seeds ใช้รายงานเท่านั้น")
            r10row = s[s["stage"] == "2_refit10"].iloc[0]
            L.append(f"- ANN ผู้ชนะ 10 seeds: min StdRatio = {r10row['min_StdRatio']:.4f} "
                     f"(3 seeds {facts[(name, '2')][0]['min_StdRatio']:.4f}) · "
                     f"ผ่าน guard = {r10row['passed_guard']}")
            facts["ann10"] = r10
        L.append("")

    # --- คำทำนายใน TUNING_PLAN ข้อ 9 (ข้อ 1-6 ตรวจอัตโนมัติ · ข้อ 7 ต้องดู main.py --dev) ---
    hl = half_lives
    wins2 = {n: GRIDS[n][winners[n]] for n in MODEL_NAMES}
    final_scores = {n: facts[(n, "2")][0]["primary_score"] for n in MODEL_NAMES}
    n_cut = sum(len(facts[(n, st)][1]) for n in MODEL_NAMES for st in ("1", "2"))
    checks = [
        ("1. half-life: >= 2 ใน 3 โมเดลได้ None/1000 และไม่มีโมเดลใดได้ 125",
         sum(h in (None, 1000) for h in hl.values()) >= 2 and 125 not in hl.values(),
         f"ได้ {hl}"),
        ("2. primary_score ดีขึ้นจาก config Phase 1 ไม่เกิน 2% ทุกโมเดล",
         all(facts[(n, 'change')] >= -0.02 for n in MODEL_NAMES),
         ", ".join(f"{n} {facts[(n, 'change')]:+.3%}" for n in MODEL_NAMES)),
        ("3. ANN ผู้ชนะมี alpha >= 3.0 และ ANN แย่สุดใน 3 โมเดล",
         wins2["ANN (MLP)"]["alpha"] >= 3.0
         and max(final_scores, key=final_scores.get) == "ANN (MLP)",
         f"ANN {wins2['ANN (MLP)']} · scores {final_scores}"),
        ("4. XGB ผู้ชนะมี learning_rate = 0.01",
         wins2["XGBoost"]["learning_rate"] == 0.01, f"{wins2['XGBoost']}"),
        ("5. RF ผู้ชนะมี min_samples_leaf = 50",
         wins2["Random Forest"]["min_samples_leaf"] == 50, f"{wins2['Random Forest']}"),
        ("6. >= 1 config ไม่ผ่าน guard แต่ไม่มีโมเดลใดไม่ผ่านทุกค่า",
         n_cut >= 1, f"ถูกตัด {n_cut} configs จาก 51"),
    ]
    L.append("## คำทำนายก่อนรัน (TUNING_PLAN ข้อ 9) — ตรง / ไม่ตรง")
    L.append("\n| คำทำนาย | ผล | สิ่งที่เกิดจริง |")
    L.append("|---|---|---|")
    for text, ok, got in checks:
        L.append(f"| {text} | {'ตรง' if ok else '**ไม่ตรง**'} | {got} |")
    L.append("| 7. ตารางหลัก main.py --dev: ส่วนต่างโมเดลที่เลือกกับ Naive < 1% ทั้งสองหุ้น "
             "| (ตรวจหลังนำค่าเข้า config) | ดูหัวข้อด้านล่าง |")
    L += ["", "## ข้อจำกัด",
          "- ค่าที่จูนได้เหมาะกับ training set 1,250–1,782 แถว (1,688 ของ main.py อยู่ในช่วงนี้)",
          "- fold 2–3 ทับกับ validation split ของ main.py → ตาราง val ของ main.py หลังจูน"
          " ไม่ใช่การประเมินอิสระ (มองในแง่ดีเกินจริง) · generalization gap เคยเกิดแล้ว (KBANK RF)",
          "- ANN จัดอันดับด้วย 3 seeds → ranking มี noise มากกว่ารายงานสุดท้าย 10 seeds",
          "- ผลทั้งหมดเป็น validation แบบ walk-forward ไม่ใช่ test", ""]
    SUMMARY_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[tune] บันทึก {SUMMARY_MD.name}")


def main():
    p = argparse.ArgumentParser(description="จูนบน train+val เท่านั้น -- ไม่มีทางเปิด test")
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    args = p.parse_args()          # ห้ามมี argparse flag ใดที่เกี่ยวกับ test เลย

    print("=" * 78)
    print(f"  tune.py stage {args.stage} -- train+val เท่านั้น (ถึง {SPLIT_BY_DATE['val_end']})")
    print(f"  !! ANN ใช้ {len(TUNING_ANN_SEEDS)} seeds ตอนจูน (เต็มคือ {len(ANN_SEEDS)}) -- "
          "เพิ่ม noise ในการจัดอันดับ จึงต้องมี guard + กฎตัดสินเสมอ")
    print("=" * 78)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dev_data()
    if args.stage == 1:
        run_stage1(data)
    else:
        run_stage2(data)


if __name__ == "__main__":
    main()
