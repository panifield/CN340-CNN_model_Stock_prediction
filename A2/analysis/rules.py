"""
ทดสอบกฎคู่/คี่ช่วงท้ายตลาดตาม preregistration.md (ไม่มีการเทรน)

Usage:
    python A2/analysis/rules.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from intraday import STOCKS, load_daily, load_hourly, tick_size  # noqa: E402

CONFIRM = ("2023-10-01", "2024-10-01")
DISCOVER = ("2024-10-01", "2026-12-31")
ALPHA = 0.05 / 3
# เวลาทำนาย -> แท่งชั่วโมงสุดท้ายที่ใช้ได้ (label ของแท่ง = เวลาเริ่ม)
CUTOFFS = {"16:00": "15:00", "16:30": "16:00"}


def table(stock):
    h, d = load_hourly(stock), load_daily(stock).set_index("Date")
    p = h.pivot_table(index="day", columns="hm", values="Close", aggfunc="last")
    t = p.join(d["parity"], how="inner")
    for when, bar in CUTOFFS.items():
        t[f"half_{when}"] = t[bar].map(tick_size) < 1
        t[f"same_{when}"] = (np.floor(t[bar]) % 2) == t["parity"]
    return t.dropna(subset=list(CUTOFFS.values()))


def acc(ok):
    k, n = int(ok.sum()), int(len(ok))
    return (k / n if n else float("nan")), n, (binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan"))


def rule_results(t, period):
    t = t[(t.index >= period[0]) & (t.index < period[1])]
    out = {}
    out["H1"] = acc(t.loc[t["half_16:00"], "same_16:00"])
    k = t[t.stock == "KBANK"]
    out["H2"] = acc(k.loc[k["half_16:30"], "same_16:30"])
    a = t[t.stock == "ADVANC"]
    out["H3"] = acc(~a.loc[~a["half_16:30"], "same_16:30"])
    cov = t["half_16:00"].mean()
    return out, cov, t


def main():
    t = pd.concat([table(s).assign(stock=s) for s in STOCKS])
    names = {"H1": "ทายเฉพาะวันช่วงราคา 0.5 บาท ณ 16:00 (ทั้ง 2 หุ้น) -> คู่/คี่เหมือนตอนนี้",
             "H2": "KBANK ช่วงราคา 0.5 บาท ณ 16:30 -> คู่/คี่เหมือนตอนนี้",
             "H3": "ADVANC ช่วงราคา 1 บาท ณ 16:30 -> คู่/คี่ตรงข้าม"}
    res_c, cov_c, tc = rule_results(t, CONFIRM)
    res_d, cov_d, td = rule_results(t, DISCOVER)

    lines = ["# ผลยืนยันกฎคู่/คี่ช่วงท้ายตลาด (ตาม preregistration.md)", "",
             f"- ข้อมูลยืนยัน: {tc.index.min().date()} ถึง {tc.index.max().date()} | "
             f"ช่วงหาไอเดีย: {td.index.min().date()} ถึง {td.index.max().date()}",
             f"- เกณฑ์ผ่าน: accuracy > 50% และ p < {ALPHA:.4f}", "",
             "| สมมติฐาน | กฎ | ช่วงหาไอเดีย | **ข้อมูลยืนยัน** | p (ยืนยัน) | ผล |", "|---|---|---|---|---|---|"]
    for h in ["H1", "H2", "H3"]:
        (ad, nd, _), (ac, nc, pc) = res_d[h], res_c[h]
        ok = nc > 0 and ac > 0.5 and pc < ALPHA
        lines.append(f"| {h} | {names[h]} | {ad:.3f} (n={nd}) | **{ac:.3f}** (n={nc}) | {pc:.2g} | "
                     f"{'ผ่าน' if ok else 'ไม่ผ่าน'} |")

    lines += ["", "## Coverage และผลแยกหุ้น (ทายเฉพาะวันที่ช่วงราคา 0.5 บาท ณ 16:00)", "",
              "| ช่วง | หุ้น | วันที่ทาย / วันทั้งหมด | coverage | accuracy |", "|---|---|---|---|---|"]
    for lab, tt in [("ยืนยัน", tc), ("หาไอเดีย", td)]:
        for s in STOCKS:
            g = tt[tt.stock == s]
            sel = g[g["half_16:00"]]
            a = sel["same_16:00"].mean() if len(sel) else float("nan")
            lines.append(f"| {lab} | {s} | {len(sel)} / {len(g)} | {len(sel) / len(g):.0%} | {a:.3f} |")
    lines += ["", "## ผลรวมทุกวัน (ไม่เลือกวัน) ของกฎ \"คู่/คี่เหมือนตอนนี้\"", "",
              "| ช่วง | หุ้น | ณ 16:00 | ณ 16:30 |", "|---|---|---|---|"]
    for lab, tt in [("ยืนยัน", tc), ("หาไอเดีย", td)]:
        for s in STOCKS:
            g = tt[tt.stock == s]
            lines.append(f"| {lab} | {s} | {g['same_16:00'].mean():.3f} | {g['same_16:30'].mean():.3f} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
