"""
ทดลองหลายวิธีเพื่อหาวิธีที่ดีที่สุดในการทำนายราคาปิดคู่/คี่

ขั้นที่ 1 -- ค้นหา (ข้อมูลตามโจทย์: 7 ก.ย. 2568 -> ปัจจุบัน)
  - Walk-forward 4 fold x 20 วัน -> ใช้ "เลือก" วิธี
    แต่ละ fold เทรนด้วยข้อมูลก่อน fold เท่านั้น (15% ท้ายเป็น val สำหรับ early stopping)
  - Holdout 32 วันสุดท้าย -> ใช้ "รายงาน" ผลเท่านั้น
ขั้นที่ 2 -- ยืนยัน (--replicate)
  - รันวิธีที่คัดไว้ด้วยขั้นตอนเดียวกันทุกอย่างบนปีก่อนหน้า (ก.ย. 2559 - ส.ค. 2568, ปีละชุด)
    ข้อมูลเหล่านี้ไม่เคยถูกใช้เลือกวิธี -> ถ้าวิธีไหนดีจริง ต้องชนะการเดาบนข้อมูลชุดนี้ด้วย

Usage:
    python A/experiments/search.py                       # ขั้นที่ 1 ทุก config ที่ยังไม่ได้รัน
    python A/experiments/search.py --only base,cwt_hybrid --force
    python A/experiments/search.py --replicate --only hybrid_extras,base,B_majority
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import CHANNELS, START_DATE, STOCKS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MAIN_TAG = "2025-09_now"
REPLICATE_YEARS = range(2016, 2025)
SEEDS = (0, 1, 2)
N_HOLDOUT, N_FOLDS, FOLD_SIZE = 32, 4, 20

CH6 = CHANNELS + ["parity_sig", "int_chg"]
BASE = dict(method="cnn", transform="stft", window=30, nperseg=10, noverlap=8, channels=CHANNELS,
            extras=False, rounds=False, arch="current", pooled=False, noise=0.0, batch=1)

# ---------------------------------------------------------------- configs
CONFIGS = {
    # ===== รอบที่ 1 =====
    "base":               dict(desc="W30, STFT 10, 4 ch, CNN 2 ชั้น"),
    "window20":           dict(window=20, desc="ย้อนหลัง 20 วัน"),
    "window60":           dict(window=60, desc="ย้อนหลัง 60 วัน"),
    "stft6":              dict(nperseg=6, noverlap=4, desc="STFT window 6 (เน้นเวลา)"),
    "stft16":             dict(nperseg=16, noverlap=14, desc="STFT window 16 (เน้นความถี่)"),
    "ch_return_only":     dict(channels=["ret_close"], desc="ใช้แค่ return ราคาปิด"),
    "ch6_parity":         dict(channels=CH6, desc="4 ch + สัญญาณคู่/คี่ + ส่วนต่างเต็มบาท"),
    "ch_parity_only":     dict(channels=["parity_sig", "int_chg"], desc="ใช้แค่สัญญาณคู่/คี่"),
    "hybrid_extras":      dict(extras=True, desc="base + คู่/คี่เมื่อวานเข้า ANN ตรง"),
    "ch6_hybrid":         dict(channels=CH6, extras=True, desc="6 ch + extras"),
    "arch_small":         dict(arch="small", desc="CNN 1 ชั้น (441 params)"),
    "arch_deep":          dict(arch="deep", desc="CNN 4 ชั้น (520k params)"),
    "pooled":             dict(pooled=True, desc="เทรนรวม 2 หุ้น"),
    "pooled_ch6_hybrid":  dict(pooled=True, channels=CH6, extras=True, desc="รวม 2 หุ้น + 6 ch + extras"),
    "B_majority":         dict(method="majority", desc="ทายคลาสที่พบบ่อยใน train"),
    "B_persistence":      dict(method="persistence", desc="ทายเหมือนเมื่อวาน"),
    "B_logreg":           dict(method="logreg", channels=CH6, extras=True, desc="Logistic regression (6 ch + extras)"),
    "B_random_forest":    dict(method="rf", channels=CH6, extras=True, desc="Random forest (6 ch + extras)"),
    # ===== รอบที่ 2: วิธีแปลงเป็นภาพแบบอื่น =====
    "cwt":                dict(batch=2, transform="cwt", desc="Wavelet scalogram (Morlet CWT)"),
    "cwt_hybrid":         dict(batch=2, transform="cwt", extras=True, desc="CWT + extras"),
    "gaf":                dict(batch=2, transform="gaf", desc="Gramian Angular Field"),
    "gaf_hybrid":         dict(batch=2, transform="gaf", extras=True, desc="GAF + extras"),
    "complex_stft":       dict(batch=2, transform="stft_complex", desc="STFT เก็บ phase (real+imag)"),
    "complex_stft_hybrid": dict(batch=2, transform="stft_complex", extras=True, desc="STFT เก็บ phase + extras"),
    "raw1d_hybrid":       dict(batch=2, transform="raw", extras=True, desc="1D-CNN บนข้อมูลดิบ (ไม่แปลงเป็นภาพ) + extras"),
    # ===== รอบที่ 2: เทคนิคการเทรน =====
    "hybrid_stft6":       dict(batch=2, extras=True, nperseg=6, noverlap=4, desc="extras + STFT 6 (รวม 2 อันดับแรก)"),
    "hybrid_noise":       dict(batch=2, extras=True, noise=0.3, desc="hybrid + noise augmentation"),
    "pretrain_hybrid":    dict(batch=2, method="cnn_pretrain", extras=True,
                               desc="pretrain CNN ทายความผันผวน แล้ว freeze + ANN ทายคู่/คี่"),
    # ===== รอบที่ 2: ablation / ไม่ใช้ CNN =====
    "ann_extras_only":    dict(batch=2, method="mlp_extras", desc="ANN อย่างเดียว ใช้แค่ extras (ไม่มี CNN)"),
    "B_logreg_extras":    dict(batch=2, method="logreg_extras", desc="Logistic regression ใช้แค่ extras"),
    "B_gboost":           dict(batch=2, method="hgb", channels=CH6, extras=True, desc="Gradient boosting (5 วันล่าสุด + extras)"),
    # ===== รอบที่ 2: ensemble (เฉลี่ยความน่าจะเป็นของหลายโมเดล) =====
    "ens_top3":           dict(batch=2, method="ensemble", members=["hybrid_extras", "stft6", "base"],
                               desc="เฉลี่ย hybrid_extras + stft6 + base"),
    "ens_hybrids":        dict(batch=2, method="ensemble",
                               members=["hybrid_extras", "hybrid_stft6", "cwt_hybrid", "gaf_hybrid",
                                        "complex_stft_hybrid"],
                               desc="เฉลี่ย 5 โมเดล hybrid"),
    # ===== รอบที่ 3: price clustering (ดู experiments/clustering/preregistration.md) =====
    "hybrid_round":       dict(batch=3, extras=True, rounds=True,
                               desc="hybrid + เลขท้ายราคาเมื่อวาน (price clustering)"),
}
CONFIGS = {k: {**BASE, **v} for k, v in CONFIGS.items()}


TRAIN10Y_TAG = "train10y"  # เทรนด้วยข้อมูลก่อน START_DATE ทั้งหมด แล้วทดสอบช่วงของโจทย์


def period_of(tag):
    if tag == MAIN_TAG:
        return START_DATE, None
    if tag == TRAIN10Y_TAG:
        return "2016-01-01", None
    y = int(tag[:4])
    return f"{y}-09-07", f"{y + 1}-09-07"


# ---------------------------------------------------------------- data (ต่อ process)
_cache = {}


def get_data(stock, cfg, tag):
    from methods import aux_labels, build_images, tail_features
    from src.data import build_extras, build_round_features, load_stock

    key = (stock, tag, cfg["transform"], cfg["window"], tuple(cfg["channels"]), cfg["nperseg"], cfg["noverlap"],
           cfg["rounds"])
    if key not in _cache:
        start, end = period_of(tag)
        df = load_stock(stock, start, end)
        X, y, d = build_images(df, cfg["window"], cfg["channels"], cfg["transform"],
                               cfg["nperseg"], cfg["noverlap"])
        E = build_extras(df, cfg["window"])
        if cfg["rounds"]:
            E = np.hstack([E, build_round_features(df, cfg["window"])])
        _cache[key] = dict(X=X, y=y, d=d, E=E,
                           T=tail_features(df, cfg["window"], CH6), A=aux_labels(df, cfg["window"]))
    return _cache[key]


def periods_for(tag):
    """[(kind, start, end)] -- main: 4 fold selection + holdout; ปีอื่น: แบบเดียวกันแต่ทุกช่วงเป็น 'replicate'"""
    d = get_data(STOCKS[0], CONFIGS["base"], tag)["d"]
    if tag == TRAIN10Y_TAG:
        return [("test10y", pd.Timestamp(START_DATE), d[-1] + np.timedelta64(1, "D"))]
    starts = [d[-N_HOLDOUT - FOLD_SIZE * (N_FOLDS - k)] for k in range(N_FOLDS)] + [d[-N_HOLDOUT]]
    ends = starts[1:] + [d[-1] + np.timedelta64(1, "D")]
    kinds = ["selection"] * N_FOLDS + ["holdout"]
    if tag != MAIN_TAG:
        kinds = ["replicate"] * len(kinds)
    return list(zip(kinds, starts, ends))


# ---------------------------------------------------------------- fit / predict
def run_task(name, stocks, tag, kind, start, end):
    import torch
    torch.set_num_threads(2)
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from src.train import predict_proba, train_one

    cfg = CONFIGS[name]
    datas = {s: get_data(s, cfg, tag) for s in stocks}
    tr, va = [], []
    for D in datas.values():
        idx = np.where(D["d"] < start)[0]
        n_va = max(int(len(idx) * 0.15), 5)
        tr.append((D, idx[:-n_va]))
        va.append((D, idx[-n_va:]))

    def stack(parts, key):
        return np.concatenate([D[key][i] for D, i in parts])

    Xtr, ytr, Etr, Ttr, Atr = (stack(tr, k) for k in "XyETA")
    Xva, yva, Eva, Tva, Ava = (stack(va, k) for k in "XyETA")
    mu = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    sd = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6

    def norm(X):
        return ((X - mu) / sd).astype(np.float32)

    use_e = cfg["extras"]
    m = cfg["method"]
    yall = np.concatenate([ytr, yva])

    models, clf = [], None
    if m == "cnn":
        for s in SEEDS:
            models.append(train_one(norm(Xtr), ytr, norm(Xva), yva, seed=s,
                                    Etr=Etr if use_e else None, Eva=Eva if use_e else None,
                                    arch=cfg["arch"], noise_std=cfg["noise"])[0])
    elif m == "cnn_pretrain":
        thr = np.median(Atr)
        for s in SEEDS:
            aux_model = train_one(norm(Xtr), (Atr > thr).astype(np.float32), norm(Xva),
                                  (Ava > thr).astype(np.float32), seed=s, arch=cfg["arch"])[0]
            models.append(train_one(norm(Xtr), ytr, norm(Xva), yva, seed=s, Etr=Etr, Eva=Eva,
                                    arch=cfg["arch"], init_cnn=aux_model.cnn.state_dict(),
                                    freeze_cnn=True)[0])

    def feats(X, E, T):
        if m in ("logreg", "rf"):
            return np.hstack([norm(X).reshape(len(X), -1), E if use_e else E[:, :0]])
        if m in ("mlp_extras", "logreg_extras"):
            return E
        if m == "hgb":
            return np.hstack([T, E])

    if m in ("logreg", "rf", "mlp_extras", "logreg_extras", "hgb"):
        Fall = np.vstack([feats(Xtr, Etr, Ttr), feats(Xva, Eva, Tva)])
        if m == "mlp_extras":
            clf = [MLPClassifier((16,), alpha=1e-2, max_iter=2000, random_state=s).fit(Fall, yall)
                   for s in SEEDS]
        else:
            clf = [{"logreg": LogisticRegression(C=0.05, max_iter=5000),
                    "logreg_extras": LogisticRegression(C=1.0, max_iter=5000),
                    "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
                    "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                          random_state=0)}[m].fit(Fall, yall)]

    rows = []
    for s, D in datas.items():
        te = np.where((D["d"] >= start) & (D["d"] < end))[0]
        X, E, T = D["X"][te], D["E"][te], D["T"][te]
        if models:
            p = np.mean([predict_proba(md, norm(X), E if use_e else None) for md in models], axis=0)
        elif clf:
            p = np.mean([c.predict_proba(feats(X, E, T))[:, 1] for c in clf], axis=0)
        elif m == "majority":
            p = np.full(len(te), float(yall.mean() >= 0.5))
        elif m == "persistence":
            p = E[:, 0]  # prev_parity
        rows += [dict(config=name, tag=tag, stock=s, period=kind, date=di, y=int(yi), p_odd=float(pi))
                 for yi, pi, di in zip(D["y"][te], p, D["d"][te])]
    return rows


def pred_path(tag, name):
    return RESULTS / tag / "preds" / f"{name}.csv"


def run_configs(names, tags, workers, force):
    todo = [(n, t) for t in tags for n in names
            if CONFIGS[n]["method"] != "ensemble" and (force or not pred_path(t, n).exists())]
    tasks = []
    for name, tag in todo:
        groups = [STOCKS] if CONFIGS[name]["pooled"] else [[s] for s in STOCKS]
        tasks += [(name, g, tag, k, a, b) for g in groups for k, a, b in periods_for(tag)]
    if not tasks:
        return
    print(f"รัน {len(todo)} (config, ช่วงเวลา) = {len(tasks)} งาน ด้วย {workers} process", flush=True)
    results = {key: [] for key in todo}
    remaining = {key: sum(1 for t in tasks if (t[0], t[2]) == key) for key in todo}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_task, *t): (t[0], t[2]) for t in tasks}
        for f in as_completed(futs):
            key = futs[f]
            results[key] += f.result()
            remaining[key] -= 1
            if remaining[key] == 0:
                df = pd.DataFrame(results[key])
                pred_path(*key[::-1]).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(pred_path(key[1], key[0]), index=False)
                acc = ((df.p_odd > 0.5).astype(int) == df.y).mean()
                print(f"  [{time.time() - t0:5.0f}s] {key[1]:13s} {key[0]:20s} acc={acc:.3f} (n={len(df)})",
                      flush=True)


def load_preds(tag, names):
    frames = {}
    base_names = [m for n in names for m in CONFIGS[n].get("members", [])] + names
    for n in dict.fromkeys(base_names):
        if CONFIGS[n]["method"] == "ensemble":
            continue
        if pred_path(tag, n).exists():
            frames[n] = pd.read_csv(pred_path(tag, n), parse_dates=["date"])
    for n in names:  # ensemble = เฉลี่ยความน่าจะเป็นของสมาชิก
        cfg = CONFIGS[n]
        if cfg["method"] == "ensemble" and all(m in frames for m in cfg["members"]):
            keys = ["tag", "stock", "period", "date", "y"]
            merged = None
            for mbr in cfg["members"]:
                f = frames[mbr][keys + ["p_odd"]].rename(columns={"p_odd": mbr})
                merged = f if merged is None else merged.merge(f, on=keys)
            merged["p_odd"] = merged[cfg["members"]].mean(axis=1)
            frames[n] = merged[keys + ["p_odd"]].assign(config=n)
    frames = {n: f for n, f in frames.items() if n in names}  # ส่งคืนเฉพาะที่ขอ
    return pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- summary (ขั้นที่ 1)
def acc_stats(g):
    k, n = int(g.correct.sum()), len(g)
    return k / n, n, binomtest(k, n, 0.5, alternative="greater").pvalue


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.assign(correct=((pred.p_odd > 0.5).astype(int) == pred.y).astype(int))
    rows = []
    for c, g in pred.groupby("config", sort=False):
        r = dict(config=c, desc=CONFIGS[c]["desc"], batch=CONFIGS[c]["batch"])
        for per, suf in [("selection", "sel"), ("holdout", "hold")]:
            gp = g[g.period == per]
            for s in STOCKS + ["ALL"]:
                gs = gp if s == "ALL" else gp[gp.stock == s]
                r[f"acc_{s}_{suf}"], r[f"n_{s}_{suf}"], r[f"p_{s}_{suf}"] = acc_stats(gs)
        rows.append(r)
    return pd.DataFrame(rows).set_index("config").sort_values("acc_ALL_sel", ascending=False)


COLORS = {"KBANK": "#2a78d6", "ADVANC": "#eb6834"}
MARKERS = {"KBANK": "o", "ADVANC": "s"}


def style(ax):
    ax.grid(axis="x", color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_search(summary: pd.DataFrame, path: Path):
    names = list(summary.index)[::-1]
    ypos = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.36 * len(names) + 1.8), sharey=True)
    for ax, suf, n_each, title in [
        (axes[0], "sel", N_FOLDS * FOLD_SIZE, f"Walk-forward ({N_FOLDS * FOLD_SIZE} days/stock)\nused to SELECT"),
        (axes[1], "hold", N_HOLDOUT, f"Holdout (last {N_HOLDOUT} days/stock)\nREPORT only"),
    ]:
        band = 1.96 * np.sqrt(0.25 / n_each)
        ax.axvspan(0.5 - band, 0.5 + band, color="#d9d9d6", alpha=0.6, lw=0, label="95% range of pure guessing")
        ax.axvline(0.5, color="#8a8a85", lw=1)
        for s in STOCKS:
            ax.scatter(summary.loc[names, f"acc_{s}_{suf}"], ypos, s=42, color=COLORS[s], marker=MARKERS[s],
                       edgecolor="white", linewidth=1.5, zorder=3, label=s)
        ax.set_xlim(0.15, 0.85)
        ax.set_title(title, fontsize=10, color="#333")
        ax.set_xlabel("accuracy")
        style(ax)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([f"{n}  (round {CONFIGS[n]['batch']})" for n in names], fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Task A: even/odd prediction - all methods (sorted by walk-forward accuracy)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_search_report(summary: pd.DataFrame, path: Path):
    n_cfg = len(summary)
    lines = [
        "# Task A ขั้นที่ 1: ค้นหาวิธี (ข้อมูล 7 ก.ย. 2568 - ปัจจุบัน)", "",
        f"- เลือกวิธีด้วย walk-forward {N_FOLDS} fold x {FOLD_SIZE} วัน = {N_FOLDS * FOLD_SIZE} วัน/หุ้น; "
        f"holdout {N_HOLDOUT} วันสุดท้าย/หุ้น ไว้รายงานเท่านั้น",
        f"- p-value: binomial test เทียบการเดา 50% (one-sided); ลองทั้งหมด {n_cfg} วิธี -> "
        f"เกณฑ์ Bonferroni = {0.05 / n_cfg:.4f}", "",
        "| อันดับ | config | รอบ | คำอธิบาย | WF KBANK | WF ADVANC | WF รวม | p (WF) | Holdout รวม | p (holdout) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, (c, r) in enumerate(summary.iterrows(), 1):
        lines.append(f"| {i} | `{c}` | {r.batch} | {r.desc} | {r.acc_KBANK_sel:.3f} | {r.acc_ADVANC_sel:.3f} | "
                     f"**{r.acc_ALL_sel:.3f}** | {r.p_ALL_sel:.3f} | {r.acc_ALL_hold:.3f} | {r.p_ALL_hold:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- replication (ขั้นที่ 2)
def replication_report(names, tags):
    pred = pd.concat([load_preds(t, names) for t in tags], ignore_index=True)
    pred = pred.assign(correct=((pred.p_odd > 0.5).astype(int) == pred.y).astype(int))
    pred.to_csv(RESULTS / "replication" / "predictions.csv", index=False)
    rows = []
    for c, g in pred.groupby("config", sort=False):
        acc, n, p = acc_stats(g)
        lo, hi = binomtest(int(g.correct.sum()), n).proportion_ci(0.95)
        r = dict(config=c, desc=CONFIGS[c]["desc"], acc_all=acc, n=n, p=p, ci_lo=lo, ci_hi=hi)
        for s in STOCKS:
            r[f"acc_{s}"] = acc_stats(g[g.stock == s])[0]
        yearly = g.groupby("tag").correct.mean()
        r["years_above_50"] = f"{int((yearly > 0.5).sum())}/{len(yearly)}"
        for t, v in yearly.items():
            r[t] = v
        rows.append(r)
    t = pd.DataFrame(rows).set_index("config").sort_values("acc_all", ascending=False)
    t.to_csv(RESULTS / "replication" / "summary.csv", float_format="%.4f")

    lines = [
        "# Task A ขั้นที่ 2: ยืนยันผลบนข้อมูลปีอื่น (ไม่เคยใช้เลือกวิธี)", "",
        f"- ใช้ขั้นตอนเดียวกันทุกอย่าง (ข้อมูล 1 ปี, walk-forward + holdout) ซ้ำบน {len(tags)} ปี: "
        f"{tags[0]} ถึง {tags[-1]}",
        "- ถ้าวิธีไหนทำนายได้จริง accuracy รวมต้องสูงกว่า 50% อย่างมีนัยสำคัญ (p < 0.05)", "",
        "| config | คำอธิบาย | acc รวม | 95% CI | p-value | n | KBANK | ADVANC | ปีที่ > 50% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c, r in t.iterrows():
        lines.append(f"| `{c}` | {r.desc} | **{r.acc_all:.3f}** | {r.ci_lo:.3f}-{r.ci_hi:.3f} | {r.p:.3f} | "
                     f"{r.n} | {r.acc_KBANK:.3f} | {r.acc_ADVANC:.3f} | {r.years_above_50} |")
    (RESULTS / "replication" / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # plot: accuracy ต่อปีของแต่ละวิธี + ค่ารวมพร้อม 95% CI
    names_sorted = list(t.index)[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(names_sorted) + 1.6))
    ypos = np.arange(len(names_sorted))
    for i, c in enumerate(names_sorted):
        ax.scatter([t.loc[c, tg] for tg in tags], np.full(len(tags), i), s=14, color="#b4b3ad", zorder=2,
                   label="single year" if i == 0 else None)
        ax.plot([t.loc[c, "ci_lo"], t.loc[c, "ci_hi"]], [i, i], color="#2a78d6", lw=2, zorder=3)
        ax.scatter(t.loc[c, "acc_all"], i, s=60, color="#2a78d6", edgecolor="white", linewidth=1.5, zorder=4,
                   label="all years (95% CI)" if i == 0 else None)
    ax.axvline(0.5, color="#8a8a85", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel("accuracy")
    ax.set_title(f"Replication on {len(tags)} earlier years (data never used for selection)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "replication" / "replication.png", dpi=130)
    plt.close(fig)
    print((RESULTS / "replication" / "report.md").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="รายชื่อ config คั่นด้วย comma")
    ap.add_argument("--replicate", action="store_true", help="ขั้นที่ 2: รันบนปีก่อนหน้า")
    ap.add_argument("--train10y", action="store_true", help="เทรนด้วยข้อมูล 10 ปี ทดสอบช่วงของโจทย์")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true", help="รันใหม่แม้มีผลอยู่แล้ว")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(CONFIGS)
    # ensemble ต้องมีผลของสมาชิกก่อน
    needed = list(dict.fromkeys(names + [m for n in names for m in CONFIGS[n].get("members", [])]))

    if args.train10y:
        out = RESULTS / TRAIN10Y_TAG
        out.mkdir(parents=True, exist_ok=True)
        run_configs(needed, [TRAIN10Y_TAG], args.workers, args.force)
        pred = load_preds(TRAIN10Y_TAG, names)
        pred["correct"] = ((pred.p_odd > 0.5).astype(int) == pred.y).astype(int)
        lines = ["# เทรนด้วยข้อมูล 10 ปี (ก่อน 7 ก.ย. 2568) -> ทดสอบช่วงของโจทย์ทั้งหมด", "",
                 "| config | คำอธิบาย | KBANK | ADVANC | รวม | n | p-value |", "|---|---|---|---|---|---|---|"]
        rows = []
        for c, g in pred.groupby("config"):
            acc, n, pv = acc_stats(g)
            rows.append((acc, f"| `{c}` | {CONFIGS[c]['desc']} | {g[g.stock == 'KBANK'].correct.mean():.3f} | "
                              f"{g[g.stock == 'ADVANC'].correct.mean():.3f} | **{acc:.3f}** | {n} | {pv:.3f} |"))
        lines += [r for _, r in sorted(rows, reverse=True)]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return

    if args.replicate:
        tags = [f"{y}-09_{y + 1}-08" for y in REPLICATE_YEARS]
        (RESULTS / "replication").mkdir(parents=True, exist_ok=True)
        run_configs(needed, tags, args.workers, args.force)
        replication_report(names, tags)
        return

    run_configs(needed, [MAIN_TAG], args.workers, args.force)
    pred = load_preds(MAIN_TAG, list(CONFIGS))
    summary = summarize(pred)
    out = RESULTS / MAIN_TAG
    summary.to_csv(out / "summary.csv", float_format="%.4f")
    plot_search(summary, out / "comparison.png")
    write_search_report(summary, out / "report.md")
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
