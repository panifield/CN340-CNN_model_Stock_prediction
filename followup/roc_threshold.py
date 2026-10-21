"""
ข้อ 12 + ข้อ 9: ROC/AUC และ threshold sweep จากค่า P(odd) ที่โมเดลทำนายไว้แล้ว (ไม่เทรนใหม่)

แหล่งข้อมูล (ทั้งหมดเป็นค่าที่ทายแบบ out-of-sample: walk-forward / holdout / replication)
  - A   : A/experiments/results/2025-09_now/preds, replication/predictions.csv (9 ปี), train10y/preds
  - A2  : A2/outputs/extended/predictions.csv (16:00), extended_1630/predictions.csv (16:30)

ROC     : แยกตามหุ้น และตามช่วงราคา (tick 0.5 บาท = ราคา < 200, tick 1 บาท = ราคา >= 200)
          p-value ของ AUC > 0.5 = Mann-Whitney U (one-sided), 95% CI = bootstrap 2000 รอบ
Threshold: ไล่ 0.30-0.70 ดู accuracy (ข้อมูลย้อนหลัง ไม่ใช่การใช้งานจริง)
          แล้วทดสอบแบบไม่โกง: เลือก threshold บนครึ่งแรกตามเวลา -> วัดผลบนครึ่งหลัง

Usage:
    python followup/roc_threshold.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "A"))
sys.path.insert(0, str(ROOT / "A2"))
from src.data import load_stock  # noqa: E402
from intraday import tick_size  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
RES_A = ROOT / "A" / "experiments" / "results"
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.01), 2)
COL = {"KBANK": "#2a78d6", "ADVANC": "#eb6834"}
MODEL_COL = ["#2a78d6", "#eb6834", "#1b7f4b", "#8a5cc2", "#8a8a85"]


# ---------------------------------------------------------------- load
def prev_close_map():
    """ราคาปิดวัน t-1 ของแต่ละวัน t -> ใช้แบ่งช่วงราคาของ Task A (รู้ได้ก่อนทำนาย)"""
    out = {}
    for s in ("KBANK", "ADVANC"):
        df = load_stock(s, "2015-01-01")
        out[s] = pd.Series(df["Close"].shift(1).to_numpy(), index=df["Date"])
    return out


def load_a():
    pc = prev_close_map()
    frames = []
    for c in ("base", "hybrid_extras", "stft6", "complex_stft", "B_persistence"):
        f = pd.read_csv(RES_A / "2025-09_now" / "preds" / f"{c}.csv", parse_dates=["date"])
        frames.append(f.assign(source="A 2025-26 (WF+holdout)"))
    rep = pd.read_csv(RES_A / "replication" / "predictions.csv", parse_dates=["date"])
    frames.append(rep[rep.config.isin(["base", "hybrid_extras", "stft6", "complex_stft", "B_persistence"])]
                  .assign(source="A replication 9 yr"))
    for c in ("base", "hybrid_extras", "hybrid_round", "arch_deep"):
        f = pd.read_csv(RES_A / "train10y" / "preds" / f"{c}.csv", parse_dates=["date"])
        frames.append(f.assign(source="A train 10 yr"))
    a = pd.concat(frames, ignore_index=True).rename(columns={"config": "model"})
    a["price"] = [pc[s].get(d, np.nan) for s, d in zip(a.stock, a.date)]
    return a[["source", "model", "stock", "date", "y", "p_odd", "price"]]


def load_a2():
    frames = []
    for sub, src in (("extended", "A2 @16:00"), ("extended_1630", "A2 @16:30")):
        f = pd.read_csv(ROOT / "A2" / "outputs" / sub / "predictions.csv", parse_dates=["date"])
        f = f[f.method.isin(["cnn_ann", "cnn_ann_skip_mlp", "cnn_only", "B_logreg", "B_now_parity"])]
        frames.append(f.rename(columns={"method": "model", "price_16": "price"}).assign(source=src))
    return pd.concat(frames, ignore_index=True)[["source", "model", "stock", "date", "y", "p_odd", "price"]]


# ---------------------------------------------------------------- stats
def auc_stats(y, p, n_boot=2000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return dict(auc=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(y))
    auc = roc_auc_score(y, p)
    pv = mannwhitneyu(p[y == 1], p[y == 0], alternative="greater").pvalue
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(auc=auc, lo=lo, hi=hi, p=pv, n=len(y))


def regime(price):
    return np.where(np.array([tick_size(x) for x in price]) == 0.5, "tick 0.5", "tick 1")


def auc_table(df):
    rows = []
    for (src, m), g in df.groupby(["source", "model"], sort=False):
        groups = [("ALL", g)] + [(s, g[g.stock == s]) for s in ("KBANK", "ADVANC")]
        g = g.assign(reg=regime(g.price))
        groups += [(r, g[g.reg == r]) for r in ("tick 0.5", "tick 1")]
        for name, gg in groups:
            if len(gg):
                rows.append(dict(source=src, model=m, subset=name, **auc_stats(gg.y, gg.p_odd),
                                 acc=((gg.p_odd > 0.5).astype(int) == gg.y).mean()))
    return pd.DataFrame(rows)


def threshold_curve(g):
    return np.array([((g.p_odd > t).astype(int) == g.y).mean() for t in THRESHOLDS])


def threshold_holdout(g):
    """เลือก threshold บนครึ่งแรก (ตามวันที่) แล้ววัดบนครึ่งหลัง"""
    g = g.sort_values("date")
    cut = g.date.iloc[len(g) // 2]
    a, b = g[g.date < cut], g[g.date >= cut]
    best = THRESHOLDS[np.argmax(threshold_curve(a))]
    acc_b = ((b.p_odd > best).astype(int) == b.y).mean()
    acc_b05 = ((b.p_odd > 0.5).astype(int) == b.y).mean()
    k = int(round(acc_b * len(b)))
    return dict(best_thr_first_half=best, acc_first_half=threshold_curve(a).max(),
                acc_second_half_best=acc_b, acc_second_half_05=acc_b05, n_second=len(b),
                p_second=binomtest(k, len(b), 0.5, alternative="greater").pvalue)


# ---------------------------------------------------------------- plots
def style(ax):
    ax.grid(color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_roc(df, panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.4))
    for ax, (src, models) in zip(axes, panels):
        g0 = df[df.source == src]
        for c, m in zip(MODEL_COL, models):
            g = g0[g0.model == m]
            fpr, tpr, _ = roc_curve(g.y, g.p_odd)
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{m}  AUC {roc_auc_score(g.y, g.p_odd):.3f}")
        ax.plot([0, 1], [0, 1], color="#8a8a85", lw=1, ls="--", label="random guess (0.5)")
        ax.set_title(f"{src}\n(n = {len(g0[g0.model == models[0]])})", fontsize=10)
        ax.set_xlabel("False positive rate")
        ax.set_aspect("equal")
        ax.legend(fontsize=7.5, loc="lower right", frameon=False)
        style(ax)
    axes[0].set_ylabel("True positive rate")
    fig.suptitle("ROC curves of P(odd) - out-of-sample predictions", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_auc_subsets(tab, items, path):
    subsets = ["KBANK", "ADVANC", "tick 0.5", "tick 1"]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(items) * len(subsets) / 2 + 1.8))
    ylabels, y = [], 0
    for src, m in items:
        for sub in subsets:
            r = tab[(tab.source == src) & (tab.model == m) & (tab.subset == sub)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = COL.get(sub, "#1b7f4b" if sub == "tick 0.5" else "#8a8a85")
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=2.2, alpha=0.8)
            ax.scatter(r.auc, y, color=c, s=46, zorder=3, edgecolor="white", linewidth=1.2)
            ax.text(r.hi + 0.01, y, f"{r.auc:.3f}  (n={r.n}, p={r.p:.3g})", va="center", fontsize=7.5)
            ylabels.append((y, f"{src} | {m} | {sub}"))
            y += 1
        y += 0.6
    ax.axvline(0.5, color="#8a8a85", lw=1, ls="--")
    ax.set_yticks([a for a, _ in ylabels])
    ax.set_yticklabels([b for _, b in ylabels], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 0.92)
    ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.set_title("AUC by stock and tick regime (0.5 = no better than guessing)", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_threshold(df, items, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for c, (src, m) in zip(MODEL_COL, items):
        g = df[(df.source == src) & (df.model == m)]
        acc = threshold_curve(g)
        ax.plot(THRESHOLDS, acc, color=c, lw=2, label=f"{src} | {m}")
        i = np.argmax(acc)
        ax.scatter(THRESHOLDS[i], acc[i], color=c, s=40, zorder=3, edgecolor="white")
    ax.axvline(0.5, color="#8a8a85", lw=1, ls="--")
    ax.axhline(0.5, color="#8a8a85", lw=1)
    ax.set_xlabel("threshold  (predict odd if P(odd) > threshold)")
    ax.set_ylabel("accuracy")
    ax.set_title("Threshold sweep on past predictions (dot = best threshold, chosen in hindsight)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_a(), load_a2()], ignore_index=True)
    tab = auc_table(df)
    tab.to_csv(OUT / "auc_table.csv", index=False, float_format="%.4f")

    plot_roc(df, [("A 2025-26 (WF+holdout)", ["base", "hybrid_extras", "stft6", "complex_stft"]),
                  ("A replication 9 yr", ["base", "hybrid_extras", "stft6", "complex_stft"]),
                  ("A2 @16:00", ["cnn_ann", "cnn_ann_skip_mlp", "cnn_only", "B_logreg"]),
                  ("A2 @16:30", ["cnn_ann_skip_mlp", "cnn_only", "B_logreg"])], OUT / "roc_curves.png")
    key_items = [("A replication 9 yr", "hybrid_extras"), ("A train 10 yr", "hybrid_extras"),
                 ("A2 @16:00", "cnn_ann_skip_mlp"), ("A2 @16:30", "cnn_ann_skip_mlp")]
    plot_auc_subsets(tab, key_items, OUT / "auc_by_subset.png")

    th_items = [("A replication 9 yr", "base"), ("A replication 9 yr", "hybrid_extras"),
                ("A2 @16:00", "cnn_ann"), ("A2 @16:00", "cnn_ann_skip_mlp"), ("A2 @16:30", "cnn_ann_skip_mlp")]
    plot_threshold(df, th_items, OUT / "threshold_sweep.png")
    th_rows = []
    for src, m in th_items:
        g = df[(df.source == src) & (df.model == m)]
        acc = threshold_curve(g)
        th_rows.append(dict(source=src, model=m, acc_at_05=acc[THRESHOLDS == 0.5][0],
                            best_thr_hindsight=THRESHOLDS[np.argmax(acc)], acc_best_hindsight=acc.max(),
                            **threshold_holdout(g)))
    th = pd.DataFrame(th_rows)
    th.to_csv(OUT / "threshold_table.csv", index=False, float_format="%.4f")

    # ---- report
    L = ["# ROC/AUC และ threshold sweep", "",
         "ทุกค่ามาจากการทายแบบ out-of-sample ที่บันทึกไว้แล้ว (ไม่ได้เทรนใหม่)", "",
         "## AUC (ทั้งหมด)", "",
         "| แหล่ง | โมเดล | n | AUC | 95% CI | p (AUC>0.5) | acc @0.5 |", "|---|---|---|---|---|---|---|"]
    for _, r in tab[tab.subset == "ALL"].iterrows():
        L.append(f"| {r.source} | `{r.model}` | {r.n} | **{r.auc:.3f}** | {r.lo:.3f}-{r.hi:.3f} | {r.p:.3g} | "
                 f"{r.acc:.3f} |")
    L += ["", "## AUC แยกหุ้น / ช่วงราคา (โมเดลหลัก)", "",
          "| แหล่ง | โมเดล | กลุ่ม | n | AUC | 95% CI | p |", "|---|---|---|---|---|---|---|"]
    for src, m in key_items:
        for _, r in tab[(tab.source == src) & (tab.model == m) & (tab.subset != "ALL")].iterrows():
            L.append(f"| {src} | `{m}` | {r.subset} | {r.n} | **{r.auc:.3f}** | {r.lo:.3f}-{r.hi:.3f} | "
                     f"{r.p:.3g} |")
    L += ["", "## Threshold sweep (0.30-0.70)", "",
          "| แหล่ง | โมเดล | acc @0.5 | threshold ดีสุด (มองย้อนหลัง) | acc ดีสุด | "
          "threshold จากครึ่งแรก | acc ครึ่งหลัง (threshold นั้น) | acc ครึ่งหลัง @0.5 | p ครึ่งหลัง |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in th.iterrows():
        L.append(f"| {r.source} | `{r.model}` | {r.acc_at_05:.3f} | {r.best_thr_hindsight:.2f} | "
                 f"{r.acc_best_hindsight:.3f} | {r.best_thr_first_half:.2f} | **{r.acc_second_half_best:.3f}** | "
                 f"{r.acc_second_half_05:.3f} | {r.p_second:.3g} |")
    (OUT / "roc_threshold.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
