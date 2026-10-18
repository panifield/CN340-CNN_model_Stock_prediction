"""
ทดสอบสมมติฐาน price clustering ตาม preregistration.md (รันครั้งเดียวบนข้อมูลทดสอบ)

Usage:
    python A/experiments/search.py --only hybrid_round        # H2: เทรน/ประเมินโมเดลก่อน
    python A/experiments/clustering/test_hypotheses.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
TEST_START = "2025-09-07"


def h1_rule(stock):
    d = pd.read_csv(ROOT / "raw_data" / f"{stock}_10Y_Cleaned.csv")
    d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
    c = np.floor(d["Price"]).astype(int)
    prev = c.shift(1)
    m = (d["Date"] >= TEST_START) & (prev % 10 == 0)
    hits = int((c[m] % 2 == 0).sum())
    n = int(m.sum())
    p = binomtest(hits, n, 0.5, alternative="greater").pvalue if n else float("nan")
    return n, hits, p, d.loc[m, "Date"].dt.strftime("%Y-%m-%d").tolist(), c[m].tolist()


def h2_model():
    pred = pd.read_csv(HERE.parent / "results" / "2025-09_now" / "preds" / "hybrid_round.csv")
    pred["correct"] = ((pred.p_odd > 0.5).astype(int) == pred.y).astype(int)
    out = {}
    for per in ["selection", "holdout"]:
        g = pred[pred.period == per]
        k, n = int(g.correct.sum()), len(g)
        out[per] = (k / n, n, binomtest(k, n, 0.5, alternative="greater").pvalue,
                    {s: gs.correct.mean() for s, gs in g.groupby("stock")})
    return out


def main():
    lines = ["# ผลทดสอบ price clustering (ตาม preregistration.md)", "",
             f"ข้อมูลทดสอบ: {TEST_START} ถึงวันสุดท้ายในไฟล์", ""]

    lines += ["## H1 (หลัก): ADVANC, ราคาปิดเมื่อวานลงท้าย 0 -> ทายวันนี้คู่", ""]
    for stock, role in [("ADVANC", "หลัก"), ("KBANK", "สำรวจ")]:
        n, hits, p, dates, closes = h1_rule(stock)
        verdict = ("ผ่าน" if hits / n > 0.5 and p < 0.05 else "ไม่ผ่าน") if role == "หลัก" else "-"
        lines.append(f"- **{stock}** ({role}): เข้าเงื่อนไข {n} วัน, ทายถูก {hits} วัน = "
                     f"**{hits / n:.3f}**, p = {p:.3f} -> **{verdict}**")
        lines.append(f"  - วันที่/ราคาปิด (floor): " + ", ".join(f"{a} ({b})" for a, b in zip(dates, closes)))

    h2 = h2_model()
    acc_h, n_h, p_h, per_h = h2["holdout"]
    acc_s, n_s, p_s, per_s = h2["selection"]
    lines += ["", "## H2 (รอง): CNN+ANN `hybrid_round`", "",
              f"- Walk-forward: {acc_s:.3f} (n={n_s}, p={p_s:.3f}) | "
              + ", ".join(f"{k} {v:.3f}" for k, v in per_s.items()),
              f"- Holdout: **{acc_h:.3f}** (n={n_h}, p={p_h:.3f}) | "
              + ", ".join(f"{k} {v:.3f}" for k, v in per_h.items()),
              f"- เกณฑ์: holdout > 50% และ p < 0.05 -> **{'ผ่าน' if acc_h > 0.5 and p_h < 0.05 else 'ไม่ผ่าน'}**"]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
