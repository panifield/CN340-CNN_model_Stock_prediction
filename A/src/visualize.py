"""ภาพ spectrogram และกราฟผลการเทรน"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .config import CHANNEL_LABELS, CHANNELS, WINDOW
from .data import stft_logpower

PARITY_NAME = {0: "even", 1: "odd"}


def plot_full_period(df, stock: str, path: Path):
    """ราคาปิด + spectrogram ของทั้งช่วงเวลา (STFT เลื่อนตลอดทั้ง series)"""
    n_ch = len(CHANNELS)
    fig, axes = plt.subplots(n_ch + 1, 1, figsize=(12, 2.2 * (n_ch + 1)), sharex=True)
    days = np.arange(len(df))

    ax = axes[0]
    ax.plot(days, df["Close"], color="black", lw=1)
    odd = df["parity"].to_numpy() == 1
    ax.scatter(days[odd], df["Close"][odd], s=8, color="tab:red", label="odd")
    ax.scatter(days[~odd], df["Close"][~odd], s=8, color="tab:blue", label="even")
    ax.set_ylabel("Close")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"{stock}: close price and spectrogram of each signal "
                 f"({df.Date.min().date()} to {df.Date.max().date()})")

    for i, ch in enumerate(CHANNELS):
        f, t, logp = stft_logpower(df[ch].to_numpy())
        ax = axes[i + 1]
        im = ax.pcolormesh(t, f, logp, shading="nearest", cmap="magma")
        ax.set_ylabel(f"{CHANNEL_LABELS[ch]}\nfreq (1/day)", fontsize=8)
        fig.colorbar(im, ax=ax, pad=0.01, label="log power")
    axes[-1].set_xlabel("date")

    ticks = np.linspace(0, len(df) - 1, 8).astype(int)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([df.Date.iloc[k].strftime("%Y-%m-%d") for k in ticks], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _draw_sample(axes, spec, title_prefix=""):
    for c, ch in enumerate(CHANNELS):
        ax = axes[c]
        im = ax.imshow(spec[c], aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"{title_prefix}{CHANNEL_LABELS[ch]}", fontsize=8)
        ax.set_xlabel("time frame", fontsize=7)
        ax.set_ylabel("freq bin", fontsize=7)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)


def plot_examples(X, y, dates, stock: str, path: Path, n_each: int = 2):
    """ตัวอย่าง input spectrogram: วันคู่ n_each ภาพ และวันคี่ n_each ภาพ (ล่าสุด)"""
    idxs = [(int(i), 0) for i in np.where(y == 0)[0][-n_each:]] + \
           [(int(i), 1) for i in np.where(y == 1)[0][-n_each:]]
    fig, axes = plt.subplots(len(idxs), len(CHANNELS),
                             figsize=(3.4 * len(CHANNELS), 2.6 * len(idxs)))
    for r, (i, lab) in enumerate(idxs):
        d = str(dates[i])[:10]
        _draw_sample(axes[r], X[i], f"{d} [{PARITY_NAME[lab]}]\n")
    fig.suptitle(f"{stock}: input spectrograms (each uses previous {WINDOW} days)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_mean(X, y, stock: str, path: Path):
    """spectrogram เฉลี่ยของวันคู่ vs วันคี่ และผลต่าง"""
    m_even, m_odd = X[y == 0].mean(0), X[y == 1].mean(0)
    diff = m_odd - m_even
    fig, axes = plt.subplots(3, len(CHANNELS), figsize=(3.4 * len(CHANNELS), 7.5))
    _draw_sample(axes[0], m_even, f"mean EVEN (n={int((y == 0).sum())})\n")
    _draw_sample(axes[1], m_odd, f"mean ODD (n={int((y == 1).sum())})\n")
    lim = np.abs(diff).max()
    for c, ch in enumerate(CHANNELS):
        ax = axes[2, c]
        im = ax.imshow(diff[c], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"ODD - EVEN\n{CHANNEL_LABELS[ch]}", fontsize=8)
        ax.set_xlabel("time frame", fontsize=7)
        ax.set_ylabel("freq bin", fontsize=7)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{stock}: average spectrogram by target class")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_per_day(X, y, dates, out_dir: Path):
    """บันทึก spectrogram ของทุก sample แยกรายวัน: YYYY-MM-DD_even.png"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    for i in range(len(X)):
        d = str(dates[i])[:10]
        lab = PARITY_NAME[int(y[i])]
        fig, axes = plt.subplots(1, len(CHANNELS), figsize=(3.2 * len(CHANNELS), 2.6))
        _draw_sample(axes, X[i])
        fig.suptitle(f"target day {d}  ->  {lab}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{d}_{lab}.png", dpi=80)
        plt.close(fig)


def plot_training_curves(hist, stock: str, path: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(hist["train_loss"], label="train")
    a1.plot(hist["val_loss"], label="val")
    a1.set_title("BCE loss")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(hist["val_acc"], color="tab:green")
    a2.axhline(0.5, ls="--", color="gray")
    a2.set_title("val accuracy")
    a2.set_xlabel("epoch")
    fig.suptitle(f"{stock}: training curves (selected seed)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_spectrogram_figures(df, X, y, dates, stock: str, fig_dir: Path, per_day: bool = True):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_full_period(df, stock, fig_dir / "1_full_period_spectrogram.png")
    plot_examples(X, y, dates, stock, fig_dir / "2_input_examples.png")
    plot_class_mean(X, y, stock, fig_dir / "3_mean_even_vs_odd.png")
    if per_day:
        save_per_day(X, y, dates, fig_dir / "per_day")
