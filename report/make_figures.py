"""
สร้างกราฟวิเคราะห์ข้อมูลสำหรับสไลด์ (เก็บไว้ที่ report/figures/)

Usage:
    python report/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram, stft
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT / "A2"))
from intraday import load_daily, load_hourly, tick_size  # noqa: E402

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b4b3ad", "#28282a", "#6b6a66"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlecolor": INK})


def style(ax, grid="y"):
    ax.grid(axis=grid, color="#ececea", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- 1) ทำไมคู่/คี่รายวันสุ่ม
def fig_daily_change():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, s in zip(axes, ["KBANK", "ADVANC"]):
        d = pd.read_csv(ROOT / "raw_data" / f"{s}_10Y_Cleaned.csv")
        c = np.floor(d["Price"]).astype(int)
        raw_ch = c.diff().dropna().astype(int)
        keep = (raw_ch % 2 == 0).mean() * 100          # คำนวณจากค่าจริง (ก่อน clip)
        ch = raw_ch.clip(-5, 5)
        cnt = ch.value_counts().reindex(range(-5, 6), fill_value=0) / len(ch) * 100
        colors = [GREY if abs(k) == 5 else BLUE if k % 2 == 0 else ORANGE for k in cnt.index]
        ax.bar(cnt.index, cnt.values, color=colors, width=0.75, edgecolor="white", linewidth=2)
        ax.set_title(f"{s}: ส่วนต่างราคา (บาทเต็ม) จากเมื่อวาน, 10 ปี", loc="left")
        ax.set_xticks(range(-5, 6))
        ax.set_xticklabels([("≤-5" if k == -5 else "≥5" if k == 5 else str(k)) for k in range(-5, 6)])
        ax.set_xlabel("ส่วนต่าง (บาท)")
        ax.text(0.98, 0.95, f"คู่/คี่เหมือนเดิม {keep:.1f}%\nคู่/คี่สลับ {100 - keep:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("% ของวัน")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=BLUE, label="ขยับเลขคู่ (คู่/คี่เหมือนเดิม)"),
                            Patch(color=ORANGE, label="ขยับเลขคี่ (คู่/คี่สลับ)"),
                            Patch(color=GREY, label="≥5 บาท (รวมคู่และคี่)")],
                   loc="upper left", frameon=False, fontsize=9)
    save(fig, "f1_daily_change_parity.png")


# ---------------------------------------------------------------- 2) STFT bug ก่อน/หลังแก้
def fig_stft_fix():
    w_old, w_new = [], []
    for k in range(30):
        e = np.zeros(30)
        e[k] = 1
        _, _, s = spectrogram(e, fs=1, nperseg=10, noverlap=8, window="hann", detrend=False)
        w_old.append(s.sum())
        _, _, z = stft(e, fs=1, nperseg=10, noverlap=8, window="hann", boundary="even", padded=False,
                       detrend=False)
        w_new.append((np.abs(z) ** 2).sum())
    w_old = np.array(w_old) / max(w_old)
    w_new = np.array(w_new) / max(w_new)
    x = np.arange(-30, 0)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(x, w_old * 100, color=ORANGE, lw=2, marker="o", ms=4, label="ก่อนแก้ (ไม่ pad ขอบ)")
    ax.plot(x, w_new * 100, color=BLUE, lw=2, marker="o", ms=4, label="หลังแก้ (pad ขอบแบบสะท้อน)")
    ax.annotate(f"เมื่อวาน (t-1) เหลือ {w_old[-1] * 100:.1f}%", xy=(-1, w_old[-1] * 100), xytext=(-9, 20),
                arrowprops=dict(arrowstyle="->", color=MUTED), color=INK)
    ax.set_xticks([-30, -25, -20, -15, -10, -5, -1])
    ax.set_xticklabels(["t-30", "t-25", "t-20", "t-15", "t-10", "t-5", "t-1"])
    ax.set_ylabel("น้ำหนักใน spectrogram (% ของสูงสุด)")
    ax.set_title("แต่ละวันในหน้าต่าง 30 วันมีผลต่อ spectrogram แค่ไหน", loc="left")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="lower center", ncol=2)
    style(ax)
    save(fig, "f2_stft_fix.png")


# ---------------------------------------------------------------- 3) ตัวควบคุมก่อน/หลังแก้
def fig_controls():
    labels = ["KBANK\nlabel ความผันผวน", "ADVANC\nlabel ความผันผวน", "KBANK\nlabel คู่/คี่", "ADVANC\nlabel คู่/คี่"]
    before = [0.448, 0.688, 0.475, 0.631]
    after = [0.844, 0.729, 0.494, 0.581]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(x - 0.19, before, 0.36, color=ORANGE, label="ก่อนแก้ STFT", edgecolor="white", linewidth=2)
    ax.bar(x + 0.19, after, 0.36, color=BLUE, label="หลังแก้ STFT", edgecolor="white", linewidth=2)
    for i in range(len(x)):
        ax.text(x[i] - 0.19, before[i] + 0.015, f"{before[i]:.0%}", ha="center", fontsize=9, color=INK)
        ax.text(x[i] + 0.19, after[i] + 0.015, f"{after[i]:.0%}", ha="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(3.55, 0.51, "เดาสุ่ม", color=MUTED, fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("โมเดลเรียนรู้ได้จริงเมื่อ label มีรูปแบบ (ความผันผวน) แต่คู่/คี่ยังเท่าการเดา", loc="left")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    save(fig, "f3_controls.png")


# ---------------------------------------------------------------- 4) ราคาปิด - ราคา ณ เวลาทำนาย (A2)
def intraday_table():
    rows = []
    for s in ["KBANK", "ADVANC"]:
        h, d = load_hourly(s), load_daily(s).set_index("Date")
        p = h.pivot_table(index="day", columns="hm", values="Close", aggfunc="last")
        t = p[["15:00", "16:00"]].join(d[["Close", "parity"]], how="inner").dropna()
        t["stock"] = s
        rows.append(t)
    return pd.concat(rows)


def fig_close_vs_cutoff(t):
    t = t.copy()
    t["tick"] = t["15:00"].map(tick_size)
    t["dt"] = ((t["Close"] - t["15:00"]) / t["tick"]).round().clip(-3, 3).astype(int)
    groups = [("KBANK ราคา < 200 (ช่วงราคา 0.5 บาท)", (t.stock == "KBANK") & (t.tick < 1)),
              ("ราคา ≥ 200 (ช่วงราคา 1 บาท, ทั้ง 2 หุ้น)", t.tick >= 1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, m) in zip(axes, groups):
        g = t[m]
        cnt = g["dt"].value_counts().reindex(range(-3, 4), fill_value=0) / len(g) * 100
        ax.bar(cnt.index, cnt.values, color=BLUE, width=0.7, edgecolor="white", linewidth=2)
        same = ((np.floor(g["15:00"]) % 2) == g["parity"]).mean()
        ax.set_title(title, loc="left")
        ax.set_xlabel("ราคาปิด − ราคา ณ 16:00 (จำนวนช่วงราคา)")
        ax.set_xticks(range(-3, 4))
        ax.set_xticklabels(["≤-3", "-2", "-1", "0", "+1", "+2", "≥+3"])
        ax.text(0.98, 0.95, f"n = {len(g)} วัน\nคู่/คี่ตอนปิด = ตอน 16:00: {same:.1%}",
                transform=ax.transAxes, ha="right", va="top", color=INK)
        style(ax)
    axes[0].set_ylabel("% ของวัน")
    save(fig, "f4_close_vs_1600.png")


# ---------------------------------------------------------------- 5) กฎ: ช่วงหาไอเดีย vs ข้อมูลยืนยัน
def fig_rule_confirm(t):
    t = t.copy()
    rows = []
    periods = [("ข้อมูลยืนยัน\nต.ค. 66 – ก.ย. 67", t.index < "2024-10-01"),
               ("ช่วงหาไอเดีย\nต.ค. 67 – ต.ค. 69", t.index >= "2024-10-01")]
    cases = [("KBANK ช่วงราคา 0.5 ณ 16:00", "KBANK", True, "15:00"),
             ("KBANK ช่วงราคา 0.5 ณ 16:30", "KBANK", True, "16:00"),
             ("ช่วงราคา 1 บาท ณ 16:00", None, False, "15:00"),
             ("ช่วงราคา 1 บาท ณ 16:30", None, False, "16:00")]
    for cname, stock, half, bar in cases:
        for pname, pm in periods:
            g = t[pm & ((t.stock == stock) if stock else True)]
            g = g[(g[bar].map(tick_size) < 1) == half]
            ok = (np.floor(g[bar]) % 2) == g["parity"]
            if len(g) == 0:
                continue
            ci = binomtest(int(ok.sum()), len(g)).proportion_ci(0.95)
            rows.append((cname, pname, ok.mean(), ci.low, ci.high, len(g)))
    df = pd.DataFrame(rows, columns=["case", "period", "acc", "lo", "hi", "n"])
    fig, ax = plt.subplots(figsize=(11, 3.9))
    names = [c[0] for c in cases][::-1]
    for i, c in enumerate(names):
        for j, (pname, col, mk) in enumerate([(periods[0][0], BLUE, "o"), (periods[1][0], ORANGE, "s")]):
            r = df[(df.case == c) & (df.period == pname)]
            if r.empty:
                continue
            y = i + (0.15 if j == 0 else -0.15)
            r = r.iloc[0]
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=2)
            ax.scatter(r.acc, y, color=col, marker=mk, s=55, zorder=3, edgecolor="white", linewidth=1.5,
                       label=pname.replace("\n", " ") if i == len(names) - 1 else None)
            ax.text(r.hi + 0.008, y, f"{r.acc:.1%} (n={r.n})", va="center", fontsize=8.5, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(0.3, 0.85)
    ax.set_xlabel("accuracy ของกฎ \"ทายคู่/คี่เหมือนราคาตอนนั้น\" (เส้น = 95% CI)")
    ax.set_title("กฎใช้ได้จริงเฉพาะช่วงราคา 0.5 บาท และยืนยันได้บนข้อมูลที่ไม่เคยใช้", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style(ax, grid="x")
    save(fig, "f5_rule_confirmation.png")


# ---------------------------------------------------------------- 6) A2: CNN+ANN vs กฎ แยกตามช่วงราคา
def fig_a2_models():
    methods = [("cnn_only", "CNN อย่างเดียว"), ("cnn_ann", "CNN+ANN เดิม"),
               ("cnn_ann_skip_mlp", "CNN+ANN (skip MLP)"), ("B_now_parity", "กฎ: คู่/คี่ตอนนั้น")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (folder, when) in zip(axes, [("extended", "16:00"), ("extended_1630", "16:30")]):
        p = pd.read_csv(ROOT / "A2" / "outputs" / folder / "predictions.csv", parse_dates=["date"])
        p["ok"] = ((p.p_odd > 0.5).astype(int) == p.y)
        p["half"] = p.price_16.map(tick_size) < 1
        x = np.arange(len(methods))
        a_half = [p[(p.method == m) & p.half].ok.mean() if (p.method == m).any() else np.nan for m, _ in methods]
        a_one = [p[(p.method == m) & ~p.half].ok.mean() if (p.method == m).any() else np.nan for m, _ in methods]
        ax.bar(x - 0.19, a_half, 0.36, color=BLUE, label="ช่วงราคา 0.5 บาท", edgecolor="white", linewidth=2)
        ax.bar(x + 0.19, a_one, 0.36, color=ORANGE, label="ช่วงราคา 1 บาท", edgecolor="white", linewidth=2)
        for i in range(len(x)):
            for dx, v in [(-0.19, a_half[i]), (0.19, a_one[i])]:
                ax.text(x[i] + dx, (0.02 if np.isnan(v) else v + 0.012), "ไม่ได้รัน" if np.isnan(v) else f"{v:.0%}",
                        ha="center", fontsize=8.5, color=MUTED if np.isnan(v) else INK)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in methods], fontsize=9)
        ax.set_title(f"ทำนาย ณ {when} (walk-forward ต.ค. 67 – ต.ค. 69)", loc="left")
        ax.set_ylim(0, 0.8)
        style(ax)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, "f6_a2_models_by_tick.png")


COPIES = {  # กราฟเดิมจากการทดลอง -> รวมไว้ในโฟลเดอร์เดียวสำหรับทำสไลด์
    "A/outputs/KBANK/figures/1_full_period_spectrogram.png": "a1_kbank_full_spectrogram.png",
    "A/outputs/KBANK/figures/2_input_examples.png": "a2_kbank_input_examples.png",
    "A/outputs/KBANK/figures/3_mean_even_vs_odd.png": "a3_kbank_mean_even_vs_odd.png",
    "A/experiments/results/2025-09_now/comparison.png": "a4_methods_comparison.png",
    "A/experiments/results/replication/replication.png": "a5_replication_9y.png",
    "A2/outputs/spectrogram_examples.png": "b1_a2_spectrogram_examples.png",
    # follow-up (followup/roc_threshold.py, x2_label.py, optuna_search.py)
    "followup/results/roc_curves.png": "g1_roc_curves.png",
    "followup/results/auc_by_subset.png": "g2_auc_by_subset.png",
    "followup/results/threshold_sweep.png": "g3_threshold_sweep.png",
    "followup/results/x2_label.png": "g4_x2_label.png",
    "followup/results/optuna/optuna.png": "g5_optuna.png",
}


def main():
    import shutil
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in COPIES.items():
        shutil.copy(ROOT / src, OUT / dst)
    plt.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "DejaVu Sans"]
    fig_daily_change()
    fig_stft_fix()
    fig_controls()
    t = intraday_table()
    fig_close_vs_cutoff(t)
    fig_rule_confirm(t)
    fig_a2_models()


if __name__ == "__main__":
    main()
