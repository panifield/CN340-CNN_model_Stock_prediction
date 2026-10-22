"""
daring_curve.py — daring curve จากทุก config ที่รันแล้ว (รวม placebo) แทน HPO (DECISIONS B29, B33 · SPEC ข้อ 5.6 ที่ปรับแล้ว)
============================================================================================================================
ไฟล์นี้ทำอะไร
    รวมผลของทุก config ที่เคยรันในโซน dev (C3, Stage 1–5, config รวม, ตัวเทียบเดิม และ placebo ของ Stage 1 กับ Stage 3)
    แล้วพล็อตหนึ่งจุดต่อหนึ่ง config (ค่าเฉลี่ย 6 evaluation):
        รูปที่ 1  แกน x = StdRatio  แกน y = relMAE   + เส้นทฤษฎีของตารางข้อ 12.1
        รูปที่ 2  แกน x = StdRatio  แกน y = IC        + เส้นทฤษฎี IC = StdRatio (โมเดลที่ดีที่สุดได้ StdRatio = ρ และ IC = ρ)

ทำไมต้องมี
    "daring curve" คือภาพที่ตอบว่า "ยิ่งกล้าทำนาย (StdRatio สูง) ยิ่งแย่ไหม" และ "โมเดลอยู่ใกล้ขอบเขตที่ดีที่สุดเท่าที่เป็นไปได้หรือไม่"
    จุดของ placebo (y สลับ) แสดงว่า noise ล้วนอยู่ตรงไหนของภาพ — จุดจริงที่ไม่แยกจากกลุ่ม placebo = ไม่มีสัญญาณที่วัดได้
    (ข้ามการรัน HPO 30 config ตามคำสั่งผู้ใช้ ใช้ผลที่มีอยู่แล้วแทน · ห้ามบังคับให้โมเดล "กล้า" ขึ้นแบบประดิษฐ์ ตามข้อ 5.6)
"""

import numpy as np
import pandas as pd

from src import paths

# ตารางทฤษฎีข้อ 12.1 ของสเปก: ρ → relMAE ดีที่สุด (StdRatio = ρ · จำลอง 2 ล้านจุด noise หางยาว t4)
THEORY = pd.DataFrame({"rho": [0.0, 0.02, 0.05, 0.10, 0.20], "relMAE": [1.0, 0.9997, 0.9981, 0.9925, 0.9703]})

# (ไฟล์ evals, ชื่อกลุ่ม, คอลัมน์ที่ใช้เป็นชื่อ config, คอลัมน์แยกรอบ placebo)
REAL_SOURCES = [("stage1_representation/evals.csv", "config"), ("stage1_representation_combined/evals.csv", "config"),
                ("stage2_structure/evals.csv", "config"), ("stage3_input/evals.csv", "config"),
                ("stage4_twostage/evals.csv", "config"), ("stage5_loss/evals.csv", "config")]
PLACEBO_SOURCES = [("stage1_placebo/evals.csv", "ส่วนแทน (placebo) Stage 1"), ("stage3_placebo/evals.csv", "ส่วนแทน (placebo) Stage 3")]
COLS = ["StdRatio", "relMAE", "IC"]


def _agg(df, keys):
    agg = {c: "mean" for c in COLS}
    for extra in ("relMAE_shrunk", "StdRatio_shrunk"):
        if extra in df.columns:
            agg[extra] = "mean"
    return df.groupby(keys, sort=False).agg(agg).reset_index()


def collect_points(results_dir=None):
    """หนึ่งแถวต่อหนึ่งจุด: category, label, StdRatio, relMAE, IC (+ relMAE_shrunk, StdRatio_shrunk ถ้ามี) · ไฟล์ที่ไม่มีถูกข้าม"""
    root = paths.RESULTS_DIR if results_dir is None else results_dir
    rows, seen = [], set()

    c3 = root / "stage0_base" / "c3_all_models_evals.csv"
    if c3.exists():
        d = _agg(pd.read_csv(c3), ["model"])
        for r in d.to_dict("records"):
            cat = "ตัวเทียบเดิม" if r["model"] not in ("CNN→ANN (base)",) else "CNN (ข้อมูลจริง)"
            label = "base" if cat == "CNN (ข้อมูลจริง)" else r["model"]
            if cat == "CNN (ข้อมูลจริง)":
                seen.add("base")
            rows.append({"category": cat, "label": label, "source": "stage0_base", **{c: r[c] for c in COLS}})

    for rel, namecol in REAL_SOURCES:
        f = root / rel
        if not f.exists():
            continue
        d = _agg(pd.read_csv(f), [namecol])
        for r in d.to_dict("records"):
            lab = r[namecol]
            if lab in seen:                                    # config เดียวกันที่ถูกรันซ้ำข้าม Stage นับครั้งเดียว
                continue
            seen.add(lab)
            rows.append({"category": "CNN (ข้อมูลจริง)", "label": lab, "source": rel.split("/")[0],
                         **{c: r[c] for c in COLS}, **{k: r[k] for k in ("relMAE_shrunk", "StdRatio_shrunk") if k in r}})

    for rel, cat in PLACEBO_SOURCES:
        f = root / rel
        if not f.exists():
            continue
        d = _agg(pd.read_csv(f), ["config", "shuffle"])
        for r in d.to_dict("records"):
            rows.append({"category": cat, "label": f"{r['config']}@s{int(r['shuffle'])}", "source": rel.split("/")[0],
                         **{c: r[c] for c in COLS}})
    return pd.DataFrame(rows)


STYLE = {"CNN (ข้อมูลจริง)": ("#2b6cb0", "o", 38), "ตัวเทียบเดิม": ("#dd6b20", "s", 46),
         "ส่วนแทน (placebo) Stage 1": ("#9ca3af", "x", 26), "ส่วนแทน (placebo) Stage 3": ("#6b7280", "+", 30)}
LABELLED = {"base", "hybrid", "hybrid_nocnn", "two_stage_spec", "two_stage_hybrid", "combined", "cnn1d_raw", "win256", "gap",
            "ANN (MLP)", "Random Forest", "XGBoost", "Mean Return"}


def plot_curves(points, out_dir=None):
    """วาดสองรูป · คืน list ของ path"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src import figures                                   # ตั้งฟอนต์ไทย + โฟลเดอร์รูป

    outs = []
    for kind in ("relMAE", "IC"):
        fig, ax = plt.subplots(figsize=(9.6, 6.0))
        for cat, (color, marker, size) in STYLE.items():
            d = points[points["category"] == cat]
            if len(d):
                ax.scatter(d["StdRatio"], d[kind], c=color, marker=marker, s=size, alpha=0.85, label=f"{cat} ({len(d)})",
                           linewidths=1.2 if marker in "x+" else 0.5, edgecolors="white" if marker not in "x+" else None)
        if "relMAE_shrunk" in points.columns and kind == "relMAE":
            d = points[(points["category"] == "CNN (ข้อมูลจริง)") & points["relMAE_shrunk"].notna()]
            ax.scatter(d["StdRatio_shrunk"], d["relMAE_shrunk"], facecolors="none", edgecolors="#2b6cb0", marker="o", s=34,
                       label=f"CNN หลังปรับขนาดด้วย c ({len(d)})")
        for r in points[points["label"].isin(LABELLED)].itertuples():
            ax.annotate(r.label, (r.StdRatio, getattr(r, kind)), fontsize=7, xytext=(4, 3), textcoords="offset points")
        if kind == "relMAE":
            ax.plot(THEORY["rho"], THEORY["relMAE"], "-", color="#c53030", lw=1.6, label="ขอบเขตทฤษฎีที่ดีที่สุด (ข้อ 12.1)")
            ax.axhline(1.0, color="#444", lw=0.8, ls="--")
            ax.annotate("Naive = 1.000", (0.0, 1.0), fontsize=7, xytext=(4, -10), textcoords="offset points")
            ax.set_ylabel("relMAE (return) — ต่ำกว่า 1 = ชนะ Naive", fontsize=9)
            ax.set_title("Daring curve: ยิ่งกล้าทำนาย (StdRatio สูง) relMAE เป็นอย่างไร — จุดอยู่เหนือเส้น 1.0 = แพ้ Naive", fontsize=10)
        else:
            xs = np.linspace(0, max(0.3, float(points["StdRatio"].max()) * 1.05), 50)
            ax.plot(xs, xs, "-", color="#c53030", lw=1.4, label="ขอบเขตทฤษฎี: IC = StdRatio = ρ")
            ax.axhline(0.0, color="#444", lw=0.8, ls="--")
            ax.set_ylabel("IC เฉลี่ย 6 evaluation (Pearson)", fontsize=9)
            ax.set_title("IC เทียบความกล้าทำนาย — จุดจริงที่ไม่แยกจากกลุ่ม placebo = ไม่มีสัญญาณที่วัดได้", fontsize=10)
        ax.set_xlabel("StdRatio เฉลี่ย = std(ŷ)/std(y)", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = figures._fig_path(f"daring_curve_{kind}.png") if out_dir is None else out_dir / f"daring_curve_{kind}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outs.append(out)
    return outs


def main():
    out_dir = paths.RESULTS_DIR / "daring_curve"
    paths.assert_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = collect_points()
    pts.to_csv(out_dir / "daring_points.csv", index=False)
    figs = plot_curves(pts)
    print(pts.groupby("category").size().to_string())
    for f in figs:
        print("เขียน", f)
    print("เขียน", out_dir / "daring_points.csv", "· จุดทั้งหมด", len(pts))


if __name__ == "__main__":
    main()
