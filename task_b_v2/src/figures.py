"""
figures.py — รูปสำหรับตรวจงานและสำหรับสไลด์ (SPEC ข้อ 10)
========================================================
ไฟล์นี้ทำอะไร: ฟังก์ชันวาดรูปแต่ละชนิด เขียนไฟล์ลง results/figures/ · ไม่คำนวณผลโมเดลเอง
ทำไมต้องมี: แยก "การวาด" ออกจาก "การคำนวณ" เพื่อให้แก้รูปได้โดยไม่ต้องรันเทรนใหม่

รูปที่ทำแล้ว
    (2) spectrogram ตัวอย่าง 3 วัน + เทียบช่องราคา (detrend) กับช่อง return   → C1
ที่เหลือ ((1) แผนภาพ pipeline, (3) โครง CNN, (4) ตารางผล, (5) ราคาทำนาย replay, (6) cumulative error) ทำเมื่อถึง C7
"""

import matplotlib

matplotlib.use("Agg")                      # ไม่เปิดหน้าต่าง (รันบน terminal)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ฟอนต์ที่มีอักษรไทย -- ไม่ตั้งแล้วตัวหนังสือไทยจะกลายเป็นสี่เหลี่ยมเปล่า (DejaVu Sans ไม่มีสระไทย)
_THAI_FONTS = [f for f in ("Leelawadee UI", "Tahoma", "Leelawadee")
               if f in {x.name for x in font_manager.fontManager.ttflist}]
plt.rcParams["font.family"] = _THAI_FONTS + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src import paths
from src.configs import variant
from src.spectrogram import channel_matrix, raw_channels

FIG_DIR = paths.RESULTS_DIR / "figures"


def _fig_path(name):
    paths.assert_output_dir(FIG_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / name


def pick_sample_days(ds, cfg, n=3):
    """
    เลือกวันตัวอย่างแบบคงที่ (ไม่สุ่ม) จากความผันผวน 20 วันล่าสุดในหน้าต่างของแต่ละวัน:
    วันที่ผันผวนสูงสุด · ต่ำสุด · และวันสุดท้ายของโซน — เพื่อให้เห็นว่าภาพต่างกันตามสภาพตลาด
    """
    ret = raw_channels(ds.prices)["ret_close"].to_numpy()
    usable = np.flatnonzero(ds.usable(cfg))
    vol = np.array([np.std(ret[p - 20:p]) for p in usable])
    order = np.argsort(vol)
    chosen = [usable[order[-1]], usable[order[0]], usable[-1]]
    labels = ["ผันผวนสูงสุด", "ผันผวนต่ำสุด", "วันสุดท้ายของ dev"]
    return list(zip(chosen[:n], labels[:n]))


def display_transform(ds, cfg, fold_index=-1):
    """
    ImageTransform สำหรับ "เอาไว้วาดรูป" — fit จากส่วน fit ของ train ใน fold ที่เลือก (ค่าตั้งต้น = fold สุดท้าย)
    ใช้สถิติชุดเดียวกับตอนเทรนจริง จึงเป็นภาพที่ CNN ได้เห็นจริง ๆ ไม่ใช่ภาพที่ normalize ด้วยข้อมูลทั้งโซน
    """
    from src import data as _data
    fold = _data.make_folds(ds.dates)[fold_index]
    return _data.fold_inputs(ds, cfg, fold)["transform"], fold


def plot_sample_spectrograms(ds, cfg, name=None):
    """
    1 แถวต่อวันตัวอย่าง : (ซ้าย) หน้าต่าง return 128 วันที่เป็น input · (ขวา) ภาพที่ CNN เห็นจริง
    (ผ่าน log1p(|Z|²/m) แล้ว standardize ทีละแถบความถี่ ด้วยสถิติจาก train ของ fold สุดท้าย)
    แกนตั้งของภาพ = ความถี่ (รอบ/วัน) · แกนนอน = เวลาในหน้าต่าง (วันทำการก่อนวัน t)
    """
    samples = pick_sample_days(ds, cfg)
    arr, _ = channel_matrix(ds.prices, cfg)
    raw, has = ds.raw_images(cfg, [p for p, _ in samples])
    assert has.all()
    tf, fold = display_transform(ds, cfg)
    X = tf.transform(raw)

    fig, axes = plt.subplots(len(samples), 2, figsize=(11, 2.6 * len(samples)),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    for row, ((p, label), img) in enumerate(zip(samples, X)):
        day = ds.dates[p].date()
        win = arr[p - cfg.window:p, 0]
        ax = axes[row, 0]
        ax.plot(np.arange(-cfg.window, 0), win, lw=0.8, color="#2b6cb0")
        ax.axhline(0, lw=0.5, color="#999")
        ax.set_title(f"{ds.ticker} {day} — {label}\ninput: {cfg.channel_names[0]} ย้อนหลัง {cfg.window} วัน", fontsize=9)
        ax.set_xlabel("วันทำการก่อนวันที่ทำนาย (−1 = เมื่อวาน)", fontsize=8)
        ax.tick_params(labelsize=7)

        ax = axes[row, 1]
        freqs = np.linspace(0, 0.5, img.shape[1])
        lim = float(np.abs(img).max())
        im = ax.imshow(img[0], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[-cfg.window, 0, freqs[0], freqs[-1]])
        ax.set_title(f"ภาพที่ CNN เห็น ({cfg.spec_type}) · {img.shape[1]}×{img.shape[2]}", fontsize=9)
        ax.set_ylabel("ความถี่ (รอบ/วัน)", fontsize=8)
        ax.set_xlabel("เวลาในหน้าต่าง", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"ภาพที่ CNN ได้เห็น — {ds.ticker} · config {cfg.name} · "
                 f"สถิติจาก train ของ fold {fold.k} (scale = {tf.scale:.2e}, normalize ทีละแถบความถี่)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _fig_path(name or f"c1_spectrogram_samples_{ds.ticker}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_price_vs_return_channel(ds, cfg, name=None):
    """
    เทียบภาพของ "ช่องราคาที่หัก trend (C/MA20 − 1)" กับ "ช่อง return" ของวันเดียวกัน (รูปที่ 2 ของข้อ 10)
    ให้เห็นว่าการแทนข้อมูลสองแบบให้ภาพต่างกันอย่างไร — ช่องราคามีพลังงานกระจุกที่ความถี่ต่ำ (trend ที่เหลือ)
    """
    p = ds.pos_of(ds.dates[-1])
    out_cfgs = [("ช่อง return : ln(C_t/C_{t−1})", cfg),
                ("ช่องราคาหัก trend : C_t/MA20 − 1", variant(cfg.name + "_detrend", channels="detrend"))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for row, (title, c) in enumerate(out_cfgs):
        arr, _ = channel_matrix(ds.prices, c)
        raw, has = ds.raw_images(c, [p])
        assert has[0]
        tf, _fold = display_transform(ds, c)
        img = tf.transform(raw)
        ax = axes[row, 0]
        ax.plot(np.arange(-c.window, 0), arr[p - c.window:p, 0], lw=0.8, color="#2b6cb0")
        ax.axhline(0, lw=0.5, color="#999")
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("วันทำการก่อนวันที่ทำนาย", fontsize=8)
        ax = axes[row, 1]
        freqs = np.linspace(0, 0.5, img.shape[2])
        lim = float(np.abs(img).max())
        im = ax.imshow(img[0, 0], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[-c.window, 0, freqs[0], freqs[-1]])
        ax.set_title(f"ภาพที่ CNN เห็น ({c.spec_type})", fontsize=9)
        ax.set_ylabel("ความถี่ (รอบ/วัน)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"ราคา vs return เป็น input — {ds.ticker} วันที่ {ds.dates[p].date()}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _fig_path(name or f"c1_price_vs_return_{ds.ticker}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_loss_curves(histories, config_name, name=None, title=None):
    """
    loss curve ของ train กับ early-stop (val) ทุก evaluation (หุ้น × fold) · seed แรกเป็นเส้นหนา seed อื่นเป็นเส้นบาง
    เส้นประแนวตั้ง = epoch ที่ val ดีที่สุด (state ที่ใช้ทำนายจริง) · ลูกศร/ข้อความ = epoch ที่หยุด
    loss คิดบน y ที่ standardize แล้ว (Huber δ = 1) จึงเทียบข้ามหุ้น/fold ได้

    histories : dict จาก runner (key = "config|ticker|foldK" → list ของ history ต่อ seed)
    """
    keys = sorted(k for k in histories if k.split("|")[0] == config_name)
    tickers = sorted({k.split("|")[1] for k in keys}, reverse=True)           # KBANK ก่อน ADVANC
    folds = sorted({k.split("|")[2] for k in keys})
    fig, axes = plt.subplots(len(tickers), len(folds), figsize=(4.4 * len(folds), 3.4 * len(tickers)), squeeze=False)
    for r, tk in enumerate(sorted(tickers, key=lambda t: ["KBANK", "ADVANC"].index(t) if t in ("KBANK", "ADVANC") else 9)):
        for c, fd in enumerate(folds):
            ax = axes[r, c]
            hs = histories[f"{config_name}|{tk}|{fd}"]
            for i, h in enumerate(hs):
                lw, alpha = (1.6, 1.0) if i == 0 else (0.8, 0.55)
                x = np.arange(1, len(h["train_loss"]) + 1)
                ax.plot(x, h["train_loss"], color="#2b6cb0", lw=lw, alpha=alpha, label="train" if i == 0 else None)
                ax.plot(x, h["val_loss"], color="#dd6b20", lw=lw, alpha=alpha, label="early-stop (val)" if i == 0 else None)
                ax.axvline(h["best_epoch"], color="#dd6b20", ls=":", lw=0.9, alpha=alpha)
            h0 = hs[0]
            head = f"{tk} {fd.replace('fold', 'fold ')} · seed 0 หยุดที่ epoch {h0['epochs']} (ดีสุด {h0['best_epoch']})"
            if len(hs) > 1:
                head += chr(10) + "ทุก seed หยุดที่ " + str([h["epochs"] for h in hs])
            ax.set_title(head, fontsize=8.5)
            ax.set_xlabel("epoch", fontsize=8)
            if c == 0:
                ax.set_ylabel("Huber loss (y standardize)", fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(title or f"loss curve — config {config_name} (เส้นหนา = seed 0 · เส้นประ = epoch ที่ val ดีที่สุด)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = _fig_path(name or f"c3_loss_curves_{config_name}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
