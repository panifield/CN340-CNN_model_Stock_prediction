"""รวมขั้นตอนของหุ้น 1 ตัว: data -> spectrogram -> train -> evaluate -> save"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import CHANNELS, OUT_DIR, SPLIT, WINDOW
from .data import build_dataset, build_latest_input, load_stock, split_and_normalize
from .train import predict_proba, train_one
from .visualize import make_spectrogram_figures, plot_training_curves


def run_stock(stock: str, seeds: list[int], per_day_figs: bool = True) -> dict:
    out = OUT_DIR / stock
    model_dir, res_dir, fig_dir = out / "model", out / "results", out / "figures"
    for d in (model_dir, res_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_stock(stock)
    X, y, dates = build_dataset(df)
    parts, sl, mu, sd = split_and_normalize(X, y, SPLIT)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]

    print(f"\n=== {stock} === {df.Date.min().date()} ถึง {df.Date.max().date()} "
          f"| samples={len(X)} (train {len(ytr)}, val {len(yva)}, test {len(yte)}) "
          f"| spectrogram shape={X.shape[1:]}")

    make_spectrogram_figures(df, X, y, dates, stock, fig_dir, per_day=per_day_figs)

    # baseline: ทายคลาสที่พบมากสุดใน train ตลอด
    majority = int(ytr.mean() >= 0.5)
    base_acc = accuracy_score(yte, np.full_like(yte, majority))

    # หลาย seed เพราะข้อมูลน้อย ผลจาก seed เดียวเชื่อไม่ได้
    runs = []
    for s in seeds:
        model, hist = train_one(Xtr, ytr, Xva, yva, seed=s)
        p_te = predict_proba(model, Xte)
        acc = accuracy_score(yte, p_te > 0.5)
        val_acc = accuracy_score(yva, predict_proba(model, Xva) > 0.5)
        runs.append({"seed": s, "val_acc": val_acc, "test_acc": acc, "model": model,
                     "hist": hist, "p_te": p_te})
        print(f"  seed {s}: val acc={val_acc:.3f}  test acc={acc:.3f}  epochs={len(hist['val_loss'])}")

    test_accs = np.array([r["test_acc"] for r in runs])
    best = max(runs, key=lambda r: r["val_acc"])  # เลือกโมเดลด้วย val ไม่ใช่ test
    model, p_te = best["model"], best["p_te"]
    pred_te = (p_te > 0.5).astype(int)
    cm = confusion_matrix(yte, pred_te, labels=[0, 1])

    # ทำนายวันซื้อขายถัดไปหลังข้อมูลล่าสุด
    x_next = ((build_latest_input(df) - mu) / sd).astype(np.float32)
    p_next = float(predict_proba(model, x_next)[0])

    summary = {
        "stock": stock,
        "date_range": [str(df.Date.min().date()), str(df.Date.max().date())],
        "n_samples": len(X), "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "test_acc_mean": float(test_accs.mean()), "test_acc_std": float(test_accs.std()),
        "selected_seed": best["seed"], "selected_test_acc": float(best["test_acc"]),
        "selected_test_f1": float(f1_score(yte, pred_te, zero_division=0)),
        "baseline_majority_test_acc": float(base_acc),
        "confusion_matrix[[even,odd]true x pred]": cm.tolist(),
        "next_day_after": str(df.Date.max().date()),
        "next_day_p_odd": p_next,
        "next_day_prediction": "คี่ (odd)" if p_next > 0.5 else "คู่ (even)",
    }
    print(f"  test acc เฉลี่ย {test_accs.mean():.3f} ± {test_accs.std():.3f} "
          f"| baseline (majority) {base_acc:.3f}")
    print(f"  โมเดลที่เลือก (seed {best['seed']}, val ดีสุด): test acc={best['test_acc']:.3f}")
    print(f"  confusion matrix (แถว=จริง คู่/คี่, คอลัมน์=ทาย คู่/คี่): {cm.tolist()}")
    print(f"  ทำนายวันซื้อขายถัดจาก {df.Date.max().date()}: {summary['next_day_prediction']} "
          f"(P(คี่)={p_next:.3f})")

    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "window": WINDOW, "channels": CHANNELS}, model_dir / "model.pt")
    with open(res_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame({
        "Date": dates[sl["test"]], "true_parity": yte.astype(int), "pred_parity": pred_te,
        "p_odd": p_te.round(4),
        "close": df.set_index("Date").loc[dates[sl["test"]], "Close"].to_numpy(),
    }).to_csv(res_dir / "test_predictions.csv", index=False)
    plot_training_curves(best["hist"], stock, fig_dir / "4_training_curves.png")
    return summary
