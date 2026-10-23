"""Pipeline ครบวงจรของหุ้นหนึ่งตัว: data -> spectrogram -> train -> evaluate -> save."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .config import CHANNELS, EPOCHS, OUT_DIR, SPLIT, WINDOW
from .data import build_dataset, load_stock, split_and_normalize
from .model import SpectroCNNANN
from .train import DEVICE, evaluate, predict_proba, train_one
from .visualize import plot_confusion_matrix, plot_training_curves, save_sample_spectrograms


def _date_ranges(parts: dict) -> dict:
    return {name: [str(dates[0].date()), str(dates[-1].date())]
            for name, (_, _, dates) in parts.items()}


def _report(summary: dict) -> str:
    """สร้างรายงานสั้น ๆ จาก summary โดยไม่ตีความผลเกินกว่า test split."""
    validation, test = summary["validation_metrics"], summary["test_metrics"]
    return f"""# {summary['stock']} — Next-Day Up/Down Prediction

## Task

{summary['task']}

- ข้อมูล: {summary['data_date_range'][0]} ถึง {summary['data_date_range'][1]}
- จำนวนตัวอย่าง: {summary['n_samples']} (train {summary['n_train']}, valid {summary['n_valid']}, test {summary['n_test']})
- Temporal split: 70% / 15% / 15% ของตัวอย่างที่ใช้ได้ (เว้น 1 sample ที่รอยต่อแต่ละด้าน)
- Input: {summary['window']} วันย้อนหลัง, {len(summary['channels'])} channels, spectrogram shape `{summary['spectrogram_shape']}`
- จำนวน epoch ที่รันจริง: {summary['epochs_ran']}

## Results

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.3f} | {validation['precision']:.3f} | {validation['recall']:.3f} | {validation['f1']:.3f} | {_format_auc(validation['roc_auc'])} |
| Test | {test['accuracy']:.3f} | {test['precision']:.3f} | {test['recall']:.3f} | {test['f1']:.3f} | {_format_auc(test['roc_auc'])} |

Majority-class baseline บน test: **{summary['test_majority_class_baseline_accuracy']:.3f}**.

Confusion matrix ของ test (แถว = จริง `[DOWN, UP]`, คอลัมน์ = ทำนาย `[DOWN, UP]`):

```text
{test['confusion_matrix[[down,up] true x predicted]']}
```

## Leakage controls

ตัวอย่างที่มี label วัน `t` ใช้เฉพาะ feature ตั้งแต่ `t-64` ถึง `t-1`; label ใช้ `Close(t+1)` เทียบกับ `Close(t)`. ชุดข้อมูลเรียงตามวันและแบ่งต่อเนื่องตามเวลา โดยเว้น 1 sample ที่รอยต่อ train/valid และ valid/test เพื่อไม่ให้ label ของชุดก่อนแตะราคาของชุดถัดไป. ค่า normalization คำนวณจาก train split เท่านั้น. Validation ใช้สำหรับ early stopping; test ถูกประเมินครั้งสุดท้ายเท่านั้น.

ผลลัพธ์นี้เป็นการประเมินจากช่วงทดสอบเดียว ไม่ใช่คำแนะนำการลงทุน.
"""


def _format_auc(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _validation_report(summary: dict) -> str:
    """รายงานของรอบจูนที่ตั้งใจไม่แสดงผล test."""
    metrics = summary["validation_metrics"]
    return f"""# {summary['stock']} — Train / Validation Result

## Scope

รอบนี้ใช้ **Train + Validation เท่านั้น**; ยังไม่ได้ประเมิน Test set.

- Train: {summary['sample_date_ranges']['train'][0]} ถึง {summary['sample_date_ranges']['train'][1]} ({summary['n_train']} samples)
- Validation: {summary['sample_date_ranges']['valid'][0]} ถึง {summary['sample_date_ranges']['valid'][1]} ({summary['n_valid']} samples)
- Test (สงวนไว้): {summary['sample_date_ranges']['test'][0]} ถึง {summary['sample_date_ranges']['test'][1]}
- Epoch ที่รันจริง: {summary['epochs_ran']}

## Validation metrics

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {_format_auc(metrics['roc_auc'])} |

Confusion matrix (แถว = จริง `[DOWN, UP]`, คอลัมน์ = ทำนาย `[DOWN, UP]`):

```text
{metrics['confusion_matrix[[down,up] true x predicted]']}
```

เลือก/จูนโมเดลจากผล Validation นี้เท่านั้น แล้วจึงใช้ `--mode test` เพื่อประเมิน Test ครั้งสุดท้าย.
"""


def _prepare(stock: str, years: int):
    """เตรียม split เดิมแบบ deterministic เพื่อใช้ได้ทั้ง tune และ test."""
    output = OUT_DIR / stock
    model_dir, result_dir, figure_dir = output / "model", output / "results", output / "figures"
    for directory in (model_dir, result_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    df = load_stock(stock, years)
    X, y, dates = build_dataset(df)
    parts, slices, mean, std = split_and_normalize(X, y, dates, SPLIT)
    return df, X, y, dates, parts, mean, std, model_dir, result_dir, figure_dir


def run_train_valid(stock: str, years: int = 3, epochs: int = EPOCHS) -> dict:
    """เทรนด้วย train และเลือก checkpoint ด้วย validation เท่านั้น; ไม่ประเมิน test."""
    df, X, y, dates, parts, mean, std, model_dir, result_dir, figure_dir = _prepare(stock, years)
    Xtr, ytr, _ = parts["train"]
    Xva, yva, _ = parts["valid"]

    print(f"\n=== {stock} === {df.Date.min().date()} ถึง {df.Date.max().date()} | "
          f"samples={len(X)} (train={len(ytr)}, valid={len(yva)}) | "
          f"spectrogram={X.shape[1:]} | device={DEVICE}")
    model, history = train_one(Xtr, ytr, Xva, yva, epochs=epochs)
    valid_probabilities = predict_proba(model, Xva)
    valid_metrics, valid_predictions = evaluate(yva, valid_probabilities)

    summary = {
        "stock": stock,
        "mode": "train-valid",
        "task": "Predict whether Close(t+1) > Close(t): 1=UP, 0=DOWN",
        "data_date_range": [str(df.Date.min().date()), str(df.Date.max().date())],
        "sample_date_ranges": _date_ranges(parts),
        "n_samples": int(len(X)), "n_train": int(len(ytr)), "n_valid": int(len(yva)),
        "n_test": int(len(parts["test"][1])),
        "split": {"train": SPLIT[0], "valid": SPLIT[1], "test": SPLIT[2]},
        "purged_boundary_samples": 2,
        "window": WINDOW, "channels": CHANNELS, "spectrogram_shape": list(X.shape[1:]),
        "epochs_ran": len(history["train_loss"]),
        "validation_metrics": valid_metrics,
    }
    print(f"  validation accuracy={valid_metrics['accuracy']:.3f} | test ยังไม่ถูกประเมิน")

    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "window": WINDOW, "channels": CHANNELS, "years": years}, model_dir / "model.pt")
    with (result_dir / "validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (result_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    (result_dir / "validation_report.md").write_text(_validation_report(summary), encoding="utf-8")
    validation_dates = parts["valid"][2]
    pd.DataFrame({
        "Date": validation_dates, "true_direction": yva.astype(int),
        "pred_direction": valid_predictions, "p_up": valid_probabilities.round(6),
        "close_t": df.set_index("Date").loc[validation_dates, "Close"].to_numpy(),
    }).to_csv(result_dir / "validation_predictions.csv", index=False)
    plot_training_curves(history, stock, figure_dir / "training_curves.png")
    plot_confusion_matrix(valid_metrics["confusion_matrix[[down,up] true x predicted]"], stock,
                          figure_dir / "validation_confusion_matrix.png")
    # ใช้ input ได้ถึง validation เท่านั้น จึงไม่เปิดดู test sample ระหว่างการจูน
    validation_end = np.where(dates == validation_dates[-1])[0][0] + 1
    save_sample_spectrograms(stock, X[:validation_end], y[:validation_end], dates[:validation_end],
                             figure_dir / "train_validation_spectrogram_examples.png")
    return summary


def run_test(stock: str, years: int = 3) -> dict:
    """โหลด checkpoint ที่จูนแล้วและประเมิน test เพียงครั้งเดียว."""
    df, X, y, dates, parts, _, _, model_dir, result_dir, figure_dir = _prepare(stock, years)
    checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("ไม่พบ checkpoint; รัน --mode train-valid และจูนโมเดลก่อน")
    # checkpoint นี้สร้างโดยโปรเจกต์ใน workspace เดียวกันและมี numpy arrays (mean/std)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if checkpoint.get("years") != years or checkpoint.get("window") != WINDOW or checkpoint.get("channels") != CHANNELS:
        raise ValueError("checkpoint ไม่ตรงกับค่าตั้งต้นปัจจุบัน; กรุณารัน --mode train-valid ใหม่")
    model = SpectroCNNANN(in_channels=X.shape[1]).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    mean, std = checkpoint["mean"], checkpoint["std"]
    _, yte, test_dates = parts["test"]
    # ใช้ slice ของ test โดยตรง (ไม่พึ่ง offset เพราะมี purged boundary)
    test_start = np.where(dates == test_dates[0])[0][0]
    Xte = ((X[test_start:] - mean) / std).astype(np.float32)
    probabilities = predict_proba(model, Xte)
    metrics, predictions = evaluate(yte, probabilities)
    majority = int(parts["train"][1].mean() >= 0.5)
    summary = {
        "stock": stock, "mode": "test", "test_date_range": [str(test_dates[0].date()), str(test_dates[-1].date())],
        "n_test": int(len(yte)), "test_metrics": metrics,
        "test_majority_class_baseline_accuracy": float((yte == majority).mean()),
    }
    with (result_dir / "test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    pd.DataFrame({"Date": test_dates, "true_direction": yte.astype(int), "pred_direction": predictions,
                  "p_up": probabilities.round(6),
                  "close_t": df.set_index("Date").loc[test_dates, "Close"].to_numpy()}).to_csv(
                      result_dir / "test_predictions.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix[[down,up] true x predicted]"], stock,
                          figure_dir / "test_confusion_matrix.png")
    print(f"  test accuracy={metrics['accuracy']:.3f} | majority baseline={summary['test_majority_class_baseline_accuracy']:.3f}")
    return summary


def run_stock(stock: str, years: int = 3, epochs: int = EPOCHS) -> dict:
    """Backward-compatible full run; prefer train-valid then test as separate commands."""
    return {"train_valid": run_train_valid(stock, years, epochs), "test": run_test(stock, years)}
