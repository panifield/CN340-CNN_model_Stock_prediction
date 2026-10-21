"""
ข้อ 1: เปลี่ยน label เป็น round(close * 2) % 2 (คู่/คี่ของจำนวน tick 0.5 บาท) เทียบกับ floor(close) % 2

ผลโมเดลมาจาก  python A/experiments/search.py --label x2 [--replicate] --only base,hybrid_extras,B_majority,B_persistence
(บันทึกใน A/experiments/results_x2/) สคริปต์นี้แค่สรุป + วาดกราฟ

Usage:
    python followup/x2_label.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "A"))
sys.path.insert(0, str(ROOT / "A2"))
from src.data import load_stock  # noqa: E402
from intraday import tick_size  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
RES = {"floor": ROOT / "A" / "experiments" / "results", "x2": ROOT / "A" / "experiments" / "results_x2"}
MODELS = ["base", "hybrid_extras", "B_persistence", "B_majority"]
YEARS = [f"{y}-09_{y + 1}-08" for y in range(2016, 2025)]


def load(label, tag, model):
    p = RES[label] / tag / "preds" / f"{model}.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def all_preds(label):
    frames = []
    for m in MODELS:
        for tag, src in [("2025-09_now", "2025-26")] + [(t, "replication 9 yr") for t in YEARS]:
            f = load(label, tag, m)
            if f is not None:
                frames.append(f.assign(model=m, source=src, label=label))
    return pd.concat(frames, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- สัดส่วนคี่ต่อปี
    dist = []
    pc = {}
    for s in ("KBANK", "ADVANC"):
        df = load_stock(s, "2016-01-01")
        df["x2"] = np.round(df.Close * 2).astype(int) % 2
        pc[s] = pd.Series(df.Close.shift(1).to_numpy(), index=df.Date)
        for y, g in df.groupby(df.Date.dt.year):
            dist.append(dict(stock=s, year=y, odd_floor=g.parity.mean(), odd_x2=g.x2.mean(),
                             share_tick05=np.mean([tick_size(p) == 0.5 for p in g.Close])))
    dist = pd.DataFrame(dist)
    dist.to_csv(OUT / "x2_label_balance.csv", index=False, float_format="%.3f")

    pred = pd.concat([all_preds("floor"), all_preds("x2")], ignore_index=True)
    pred["correct"] = ((pred.p_odd > 0.5).astype(int) == pred.y).astype(int)
    pred["tick05"] = [tick_size(pc[s].get(d, np.nan)) == 0.5 for s, d in zip(pred.stock, pred.date)]

    rows = []
    for (lab, src, m), g in pred.groupby(["label", "source", "model"]):
        for sub, gg in [("ALL", g), ("KBANK", g[g.stock == "KBANK"]), ("ADVANC", g[g.stock == "ADVANC"]),
                        ("tick 0.5 days", g[g.tick05])]:
            if len(gg) == 0:
                continue
            k, n = int(gg.correct.sum()), len(gg)
            rows.append(dict(label=lab, source=src, model=m, subset=sub, n=n, acc=k / n,
                             odd_share=gg.y.mean(), p=binomtest(k, n, 0.5, alternative="greater").pvalue))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "x2_label_results.csv", index=False, float_format="%.4f")

    # model - majority (ถ้า label ไม่สมดุล accuracy สูงได้แค่เพราะทายคลาสใหญ่)
    piv = tab.pivot_table(index=["label", "source", "subset"], columns="model", values="acc")
    for m in ("base", "hybrid_extras", "B_persistence"):
        if m in piv:
            piv[f"{m}_minus_majority"] = piv[m] - piv["B_majority"]
    piv.to_csv(OUT / "x2_vs_majority.csv", float_format="%.4f")

    # ---- plot
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    ax = axes[0]
    for s, c in (("KBANK", "#2a78d6"), ("ADVANC", "#eb6834")):
        d = dist[dist.stock == s]
        ax.plot(d.year, d.odd_floor, color=c, lw=1.6, ls="--", marker="o", ms=3, label=f"{s} floor label")
        ax.plot(d.year, d.odd_x2, color=c, lw=2.2, marker="s", ms=4, label=f"{s} x2 label")
    ax.axhline(0.5, color="#8a8a85", lw=1)
    ax.set_ylim(-0.03, 0.75)
    ax.set_ylabel("share of 'odd' days")
    ax.set_title("Label balance per year: x2 label collapses to 0 when tick = 1 baht", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[1]
    subs = [("replication 9 yr", "ALL"), ("replication 9 yr", "tick 0.5 days"), ("2025-26", "ALL"),
            ("2025-26", "KBANK")]
    x = np.arange(len(subs))
    for i, (m, c) in enumerate(zip(["hybrid_extras", "base", "B_persistence", "B_majority"],
                                   ["#2a78d6", "#1b7f4b", "#eb6834", "#8a8a85"])):
        v = [piv.loc[("x2", s, sub), m] if ("x2", s, sub) in piv.index else np.nan for s, sub in subs]
        ax.bar(x + (i - 1.5) * 0.2, v, width=0.2, color=c, label=m)
    ax.axhline(0.5, color="#8a8a85", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{sub}" for s, sub in subs], fontsize=8)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("accuracy (x2 label)")
    ax.set_title("x2 label: models vs. 'always predict majority'", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    for a in axes:
        a.grid(axis="y", color="#ececea")
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "x2_label.png", dpi=140)
    plt.close(fig)

    L = ["# Label แบบ x2: round(close*2) % 2 เทียบ floor(close) % 2", "",
         "## สัดส่วนวันคี่ต่อปี", "", "| หุ้น | ปี | คี่ (floor) | คี่ (x2) | วันที่ tick 0.5 |", "|---|---|---|---|---|"]
    L += [f"| {r.stock} | {r.year} | {r.odd_floor:.3f} | {r.odd_x2:.3f} | {r.share_tick05:.2f} |"
          for _, r in dist.iterrows()]
    L += ["", "## Accuracy (เทียบกับทายคลาสใหญ่เสมอ)", "",
          "| label | ช่วง | กลุ่ม | n | สัดส่วนคี่ | base | hybrid_extras | persistence | majority | "
          "hybrid - majority |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (lab, src, sub), r in piv.iterrows():
        n = tab[(tab.label == lab) & (tab.source == src) & (tab.subset == sub)].n.max()
        odd = tab[(tab.label == lab) & (tab.source == src) & (tab.subset == sub)].odd_share.max()
        L.append(f"| {lab} | {src} | {sub} | {n} | {odd:.3f} | {r.get('base', np.nan):.3f} | "
                 f"**{r.get('hybrid_extras', np.nan):.3f}** | {r.get('B_persistence', np.nan):.3f} | "
                 f"{r.get('B_majority', np.nan):.3f} | {r.get('hybrid_extras_minus_majority', np.nan):+.3f} |")
    (OUT / "x2_label.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
