"""ภาพประกอบการเทรนและผลลัพธ์ของงาน C."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

from .config import CHANNEL_LABELS, CHANNELS


def plot_training_curves(history: dict, stock: str, path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["valid_loss"], label="valid")
    axes[0].set(title=f"{stock}: loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[1].plot(epochs, history["valid_acc"], color="tab:green")
    axes[1].axhline(0.5, color="gray", ls="--", lw=1)
    axes[1].set(title=f"{stock}: validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(matrix: list, stock: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=["DOWN", "UP"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{stock}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_validation_accuracy(history: dict, stock: str, path: Path) -> None:
    """กราฟ accuracy ของ validation ในแต่ละ epoch สำหรับใช้เปรียบเทียบการจูน."""
    epochs = np.arange(1, len(history["valid_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["valid_acc"], marker="o", markersize=4, color="#2455F5", linewidth=1.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline (50%)")
    ax.set(title=f"{stock}: CNN Validation Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, probabilities: np.ndarray, stock: str, path: Path,
                   split_name: str = "Validation") -> None:
    """ROC curve ของผลทำนายจริง; บันทึกภาพแจ้งเหตุผลหาก split มีคลาสเดียว."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(np.unique(y_true)) < 2:
        ax.text(0.5, 0.5, "ROC curve unavailable\n(only one class in this split)",
                ha="center", va="center", transform=ax.transAxes)
    else:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        score = auc(false_positive_rate, true_positive_rate)
        ax.plot(false_positive_rate, true_positive_rate, color="#2455F5", linewidth=2.2,
                label=f"CNN (AUC = {score:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2, label="Random baseline")
        ax.legend(loc="lower right")
    ax.set(title=f"{stock}: {split_name} ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate",
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sample_spectrograms(stock: str, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                             path: Path, n_samples: int = 4) -> None:
    """บันทึกตัวอย่าง spectrogram ในภาพเดียวเพื่อดูว่า input ถูกสร้างตามคาด."""
    indices = np.unique(np.linspace(0, len(X) - 1, min(n_samples, len(X)), dtype=int))
    fig, axes = plt.subplots(len(indices), X.shape[1], figsize=(12, 2.2 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        for channel in range(X.shape[1]):
            ax = axes[row, channel]
            image = ax.imshow(X[index, channel], aspect="auto", origin="lower", cmap="magma")
            ax.set_title(CHANNEL_LABELS.get(CHANNELS[channel], CHANNELS[channel]), fontsize=9)
            if channel == 0:
                ax.set_ylabel(f"{dates[index]:%Y-%m-%d}\n{'UP' if y[index] else 'DOWN'}", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(f"{stock}: historical-window spectrogram inputs", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
