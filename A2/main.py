"""
Task A2: ทำนายราคาปิดคู่/คี่ของวัน t ตอน 16:00 (ก่อนตลาดปิด) ด้วยข้อมูลรายชั่วโมง
         Spectrogram (แท่งชั่วโมง) -> CNN + feature ราคา ณ 16:00 -> ANN

ประเมินแบบเดียวกับ Task A: walk-forward 4 fold x 20 วัน + holdout 32 วันสุดท้าย, ensemble 3 seed
โมเดลหลักกำหนดไว้ก่อนดูผล: cnn_ann

Usage:
    python A2/main.py                    # ข้อมูลตั้งแต่ 7 ก.ย. 2568 (4 fold + holdout)
    python A2/main.py --extended         # ใช้ข้อมูลรายชั่วโมงทั้งหมด (ต.ค. 2566 -) เทรนแบบ walk-forward
                                         # retrain ทุก 20 วัน ทดสอบ ต.ค. 2567 - ปัจจุบัน
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

A2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(A2_DIR))
sys.path.insert(0, str(A2_DIR.parent / "A"))  # ใช้โมเดล/การเทรนชุดเดียวกับ Task A
from intraday import CHANNELS, EXTRA_NAMES, N_BARS, START_DATE, STOCKS, build_dataset, tick_size  # noqa: E402
from src.train import DEVICE, predict_proba, train_one  # noqa: E402

OUT = A2_DIR / "outputs"
SEEDS = (0, 1, 2)
N_HOLDOUT, N_FOLDS, FOLD_SIZE = 32, 4, 20
MAIN_MODEL = "cnn_ann"
METHODS = {
    "cnn_ann": "Spectrogram -> CNN + feature ราคา ณ 16:00 -> ANN (โมเดลหลัก)",
    "cnn_ann_skip": "cnn_ann + ย่อ CNN เหลือ 8 ค่า + ทางลัด linear จาก feature ราคา ณ 16:00 (รอบแก้)",
    "cnn_ann_skip_mlp": "cnn_ann + ย่อ CNN เหลือ 8 ค่า + ทางลัด MLP เล็ก (รอบแก้)",
    "cnn_only": "Spectrogram -> CNN -> ANN (ไม่มี feature ราคา ณ 16:00)",
    "B_now_parity": "ทายตามคู่/คี่ของราคา ณ 16:00",
    "B_logreg": "Logistic regression บน feature ราคา ณ 16:00",
    "B_majority": "ทายคลาสที่พบบ่อยใน train",
}


EXT_DATA_FROM = "2023-10-01"   # ข้อมูลรายชั่วโมงเริ่ม ต.ค. 2566
EXT_EVAL_FROM = "2024-10-01"   # ช่วงแรกใช้เทรนอย่างเดียว ~1 ปี
EXT_STEP = 20                  # retrain ทุก 20 วันทำการ


def periods_extended(dates):
    first = int(np.searchsorted(dates.values, np.datetime64(EXT_EVAL_FROM)))
    out = []
    for i in range(first, len(dates), EXT_STEP):
        j = min(i + EXT_STEP, len(dates))
        out.append(("extended", dates[i], dates[j - 1] + pd.Timedelta(days=1)))
    return out


def periods(dates):
    starts = [dates[-N_HOLDOUT - FOLD_SIZE * (N_FOLDS - k)] for k in range(N_FOLDS)] + [dates[-N_HOLDOUT]]
    ends = starts[1:] + [dates[-1] + pd.Timedelta(days=1)]
    return list(zip(["selection"] * N_FOLDS + ["holdout"], starts, ends))


def fit_predict(method, D, start, end):
    X, E, y, d = D["X"], D["E"], D["y"], D["d"]
    idx = np.where(d < start)[0]
    n_va = max(int(len(idx) * 0.15), 5)
    tr, va = idx[:-n_va], idx[-n_va:]
    te = np.where((d >= start) & (d < end))[0]
    mu = X[tr].mean(axis=(0, 2, 3), keepdims=True)
    sd = X[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    if method.startswith("cnn"):
        use_e = method != "cnn_only"
        mk = {"cnn_ann_skip": dict(bottleneck=8, skip="linear"),
              "cnn_ann_skip_mlp": dict(bottleneck=8, skip="mlp")}.get(method)
        ps = []
        for s in SEEDS:
            m, _ = train_one(Xn[tr], y[tr], Xn[va], y[va], seed=s,
                             Etr=E[tr] if use_e else None, Eva=E[va] if use_e else None, model_kwargs=mk)
            ps.append(predict_proba(m, Xn[te], E[te] if use_e else None))
        p = np.mean(ps, axis=0)
    elif method == "B_now_parity":
        p = E[te, EXTRA_NAMES.index("parity_now")]
    elif method == "B_logreg":
        trva = np.r_[tr, va]
        p = LogisticRegression(C=1.0, max_iter=5000).fit(E[trva], y[trva]).predict_proba(E[te])[:, 1]
    elif method == "B_majority":
        p = np.full(len(te), float(y[np.r_[tr, va]].mean() >= 0.5))
    return te, p


def _job(method, stock, D, kind, a, b):
    te, p = fit_predict(method, D, a, b)
    return [dict(method=method, stock=stock, period=kind, date=D["d"][i], y=int(D["y"][i]),
                 p_odd=float(pi), price_16=float(D["p16"][i])) for i, pi in zip(te, p)]


def evaluate(extended=False, workers=4, only=None):
    data, jobs = {}, []
    for stock in STOCKS:
        X, E, y, d, p16 = build_dataset(stock, start=EXT_DATA_FROM if extended else START_DATE)
        data[stock] = D = dict(X=X, E=E, y=y, d=pd.to_datetime(d), p16=p16)
        print(f"{stock}: {len(y)} วัน ({D['d'][0].date()} - {D['d'][-1].date()}), "
              f"spectrogram {X.shape[1:]}, extras {E.shape[1]}")
        for kind, a, b in (periods_extended if extended else periods)(D["d"]):
            jobs += [(m, stock, D, kind, a, b) for m in (only or METHODS)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_job, *zip(*jobs)):
            rows += r
    return pd.DataFrame(rows), data


def stats(g):
    k, n = int(g.correct.sum()), len(g)
    return k / n, n, binomtest(k, n, 0.5, alternative="greater").pvalue


def plot_spectrograms(data, path):
    fig, axes = plt.subplots(len(STOCKS), len(CHANNELS), figsize=(3.6 * len(CHANNELS), 2.8 * len(STOCKS)))
    labels = {"ret": "hourly return", "range": "hourly high-low", "vol": "volume change"}
    for r, s in enumerate(STOCKS):
        D = data[s]
        for c, ch in enumerate(CHANNELS):
            im = axes[r, c].imshow(D["X"][-1, c], aspect="auto", origin="lower", cmap="magma")
            axes[r, c].set_title(f"{s} {D['d'][-1].date()} | {labels[ch]}", fontsize=8)
            axes[r, c].set_xlabel("time frame", fontsize=7)
            axes[r, c].set_ylabel("freq bin", fontsize=7)
            plt.colorbar(im, ax=axes[r, c], fraction=0.046)
    fig.suptitle(f"A2 input: spectrogram of last {N_BARS} hourly bars up to 16:00", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_comparison(summary, path):
    colors = {"KBANK": "#2a78d6", "ADVANC": "#eb6834"}
    markers = {"KBANK": "o", "ADVANC": "s"}
    names = list(summary.index)[::-1]
    ypos = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(names) + 1.8), sharey=True)
    for ax, per, n_each, title in [(axes[0], "sel", N_FOLDS * FOLD_SIZE, "Walk-forward (80 days/stock)"),
                                   (axes[1], "hold", N_HOLDOUT, f"Holdout (last {N_HOLDOUT} days/stock)")]:
        band = 1.96 * np.sqrt(0.25 / n_each)
        ax.axvspan(0.5 - band, 0.5 + band, color="#d9d9d6", alpha=0.6, lw=0, label="95% range of pure guessing")
        ax.axvline(0.5, color="#8a8a85", lw=1)
        for s in STOCKS:
            ax.scatter(summary.loc[names, f"acc_{s}_{per}"], ypos, s=46, color=colors[s], marker=markers[s],
                       edgecolor="white", linewidth=1.5, zorder=3, label=s)
        ax.set_xlim(0.2, 0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("accuracy")
        ax.grid(axis="x", color="#ececea", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("A2: even/odd close prediction at 16:00 (intraday)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report_extended(pred, out):
    """ผลของโหมด extended: รวมทั้งช่วง + แยกช่วงย่อย"""
    pred = pred.assign(tick_half=pred.price_16.map(tick_size) < 1,
                       task_period=pred.date >= pd.Timestamp(START_DATE))
    subsets = [("ทั้งหมด (ต.ค. 67 - ต.ค. 69)", pred),
               ("เฉพาะช่วงของโจทย์ (ตั้งแต่ 7 ก.ย. 68)", pred[pred.task_period]),
               ("ราคา < 200 (ช่วงราคา 0.5 บาท)", pred[pred.tick_half]),
               ("ราคา >= 200 (ช่วงราคา 1 บาท)", pred[~pred.tick_half])]
    lines = ["# A2 (extended): ใช้ข้อมูลรายชั่วโมงทั้งหมด ทำนาย ณ 16:00", "",
             f"- ข้อมูล: {EXT_DATA_FROM} - ล่าสุด; ทดสอบตั้งแต่ {EXT_EVAL_FROM}; retrain ทุก {EXT_STEP} วัน "
             "(เทรนด้วยข้อมูลก่อนช่วงทดสอบทั้งหมด, 15% ท้ายเป็น val)",
             f"- โมเดลหลัก (กำหนดก่อนดูผล): `{MAIN_MODEL}`", ""]
    for title, g0 in subsets:
        lines += [f"## {title}", "", "| method | KBANK | ADVANC | รวม | n | p (vs 50%) |", "|---|---|---|---|---|---|"]
        for m in [m for m in METHODS if m in set(g0.method)]:
            g = g0[g0.method == m]
            acc, n, pv = stats(g)
            per = {s: g[g.stock == s].correct.mean() for s in STOCKS}
            lines.append(f"| `{m}` | {per['KBANK']:.3f} | {per['ADVANC']:.3f} | **{acc:.3f}** | {n} | {pv:.3g} |")
        piv = g0.pivot_table(index=["stock", "date"], columns="method", values="correct")
        diff = piv[MAIN_MODEL] - piv["B_now_parity"]
        w, l_ = int((diff > 0).sum()), int((diff < 0).sum())
        pv = binomtest(w, w + l_, 0.5, alternative="greater").pvalue if w + l_ else 1.0
        lines += ["", f"`{MAIN_MODEL}` vs `B_now_parity`: ถูกกว่า {w} วัน / ผิดกว่า {l_} วัน (sign test p = {pv:.3g})", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true", help="ใช้ข้อมูลรายชั่วโมงทั้งหมด (ต.ค. 2566 -)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", help="รันเฉพาะบาง method (คั่นด้วย comma) แล้วรวมกับผลเดิม")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    print(f"device: {DEVICE}")
    if args.extended:
        out = OUT / "extended"
        out.mkdir(parents=True, exist_ok=True)
        pred, _ = evaluate(extended=True, workers=args.workers, only=only)
        pred["correct"] = ((pred.p_odd > 0.5).astype(int) == pred.y).astype(int)
        if only and (out / "predictions.csv").exists():
            old = pd.read_csv(out / "predictions.csv", parse_dates=["date"])
            pred = pd.concat([old[~old.method.isin(only)], pred], ignore_index=True)
        pred.to_csv(out / "predictions.csv", index=False)
        report_extended(pred, out)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    pred, data = evaluate(workers=args.workers)
    pred["correct"] = ((pred.p_odd > 0.5).astype(int) == pred.y).astype(int)
    pred.to_csv(OUT / "predictions.csv", index=False)

    rows = []
    for m in METHODS:
        r = {"method": m, "desc": METHODS[m]}
        for per, suf in [("selection", "sel"), ("holdout", "hold")]:
            g = pred[(pred.method == m) & (pred.period == per)]
            for s in STOCKS + ["ALL"]:
                gs = g if s == "ALL" else g[g.stock == s]
                r[f"acc_{s}_{suf}"], r[f"n_{s}_{suf}"], r[f"p_{s}_{suf}"] = stats(gs)
        rows.append(r)
    summary = pd.DataFrame(rows).set_index("method")
    summary.to_csv(OUT / "summary.csv", float_format="%.4f")
    plot_comparison(summary, OUT / "comparison.png")
    plot_spectrograms(data, OUT / "spectrogram_examples.png")

    # โมเดลหลักต้องชนะ baseline "ทายตามคู่/คี่ ณ 16:00" ด้วย (ไม่ใช่แค่ชนะ 50%)
    g = pred[pred.period == "holdout"].pivot_table(index=["stock", "date"], columns="method", values="correct")
    diff = g[MAIN_MODEL] - g["B_now_parity"]
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    p_vs_rule = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0

    lines = ["# A2: ทำนายราคาปิดคู่/คี่ ณ 16:00 ด้วยข้อมูลรายชั่วโมง", "",
             "- ข้อมูล: แท่ง 1 ชั่วโมงจาก Yahoo Finance (แท่ง 10:00-15:00 ของวัน t และวันก่อนหน้า)",
             "- label: ราคาปิดทางการจาก raw_data (ต่อด้วย Yahoo หลัง 28 ส.ค. 2569)",
             f"- ประเมิน: walk-forward {N_FOLDS}x{FOLD_SIZE} วัน + holdout {N_HOLDOUT} วัน/หุ้น; โมเดลหลัก `{MAIN_MODEL}`",
             "", "| method | คำอธิบาย | WF KBANK | WF ADVANC | WF รวม | p | Holdout KBANK | Holdout ADVANC | Holdout รวม | p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in summary.iterrows():
        lines.append(f"| `{m}` | {r.desc} | {r.acc_KBANK_sel:.3f} | {r.acc_ADVANC_sel:.3f} | **{r.acc_ALL_sel:.3f}** | "
                     f"{r.p_ALL_sel:.3f} | {r.acc_KBANK_hold:.3f} | {r.acc_ADVANC_hold:.3f} | **{r.acc_ALL_hold:.3f}** | "
                     f"{r.p_ALL_hold:.3f} |")
    lines += ["", f"- `{MAIN_MODEL}` เทียบ `B_now_parity` บน holdout: ถูกกว่า {wins} วัน, ผิดกว่า {losses} วัน "
              f"(sign test p = {p_vs_rule:.3f})"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.reset_index().to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
