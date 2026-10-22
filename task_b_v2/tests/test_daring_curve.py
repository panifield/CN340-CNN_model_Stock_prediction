"""
test_daring_curve.py — การรวมจุดของ daring curve ต้องไม่นับ config ซ้ำและแยกกลุ่มถูก (DECISIONS B29, B33)
==========================================================================================================
ไฟล์นี้ทำอะไร: สร้างไฟล์ผลปลอมในโฟลเดอร์ชั่วคราว แล้วตรวจว่า collect_points รวมจุดถูก (config เดียวกันข้าม Stage นับครั้งเดียว ·
placebo แยกตาม shuffle · ไฟล์ที่ไม่มีถูกข้ามโดยไม่พัง) และตารางทฤษฎีตรงกับข้อ 12.1 ของสเปก
"""

import numpy as np
import pandas as pd
import pytest

from src import daring_curve as dc


def _evals(config, std, rel, ic, extra=None):
    rows = [{"config": config, "ticker": t, "fold": k, "StdRatio": std, "relMAE": rel, "IC": ic, **(extra or {})}
            for t in ("KBANK", "ADVANC") for k in (1, 2, 3)]
    return pd.DataFrame(rows)


def _write(root, rel, df):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def test_theory_table_matches_spec_section_12_1():
    assert dc.THEORY["rho"].tolist() == [0.0, 0.02, 0.05, 0.10, 0.20]
    assert dc.THEORY["relMAE"].tolist() == [1.0, 0.9997, 0.9981, 0.9925, 0.9703]
    assert (np.diff(dc.THEORY["relMAE"]) < 0).all(), "ยิ่ง ρ สูง relMAE ที่ดีที่สุดยิ่งต่ำ"


def test_collect_dedups_configs_and_separates_categories(tmp_path):
    c3 = pd.concat([_evals("x", 0, 1, 0).assign(model=m) for m in ("Naive", "Random Forest", "CNN→ANN (base)")])
    c3.loc[c3["model"] == "Random Forest", ["StdRatio", "relMAE", "IC"]] = [0.09, 1.002, 0.11]
    _write(tmp_path, "stage0_base/c3_all_models_evals.csv", c3)
    _write(tmp_path, "stage1_representation/evals.csv", pd.concat([_evals("base", 0.045, 1.012, -0.01), _evals("win64", 0.04, 1.007, 0.038)]))
    _write(tmp_path, "stage3_input/evals.csv", pd.concat([_evals("base", 0.045, 1.012, -0.01), _evals("hybrid", 0.2, 1.003, 0.1)]))
    _write(tmp_path, "stage1_placebo/evals.csv", pd.concat([_evals("base", 0.02, 1.01, -0.01).assign(shuffle=s) for s in (0, 1)]))
    pts = dc.collect_points(tmp_path)

    real = pts[pts["category"] == "CNN (ข้อมูลจริง)"]
    assert sorted(real["label"]) == ["base", "hybrid", "win64"], "base ปรากฏในหลายไฟล์ต้องนับครั้งเดียว"
    assert pts[pts["category"] == "ตัวเทียบเดิม"]["label"].tolist() == ["Naive", "Random Forest"]
    assert pts[pts["category"] == "ส่วนแทน (placebo) Stage 1"].shape[0] == 2, "placebo แยกจุดตาม shuffle"
    rf = pts[pts["label"] == "Random Forest"].iloc[0]
    assert (rf["StdRatio"], rf["relMAE"], rf["IC"]) == pytest.approx((0.09, 1.002, 0.11))


def test_missing_files_are_skipped_without_error(tmp_path):
    assert len(dc.collect_points(tmp_path)) == 0
    _write(tmp_path, "stage2_structure/evals.csv", _evals("gap", 0.024, 1.0065, 0.03))
    assert dc.collect_points(tmp_path)["label"].tolist() == ["gap"]


def test_shrunk_columns_are_carried_when_present(tmp_path):
    _write(tmp_path, "stage3_input/evals.csv", _evals("hybrid", 0.2, 1.003, 0.1, {"relMAE_shrunk": 1.0, "StdRatio_shrunk": 0.0}))
    p = dc.collect_points(tmp_path).iloc[0]
    assert p["relMAE_shrunk"] == 1.0 and p["StdRatio_shrunk"] == 0.0


def test_plot_writes_two_figures(tmp_path):
    _write(tmp_path, "stage3_input/evals.csv", pd.concat([_evals("base", 0.045, 1.012, -0.01), _evals("hybrid", 0.2, 1.003, 0.1)]))
    pts = dc.collect_points(tmp_path)
    figs = dc.plot_curves(pts, out_dir=tmp_path)
    assert [f.name for f in figs] == ["daring_curve_relMAE.png", "daring_curve_IC.png"] and all(f.stat().st_size > 5000 for f in figs)
