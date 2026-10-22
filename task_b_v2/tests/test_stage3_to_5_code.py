"""
test_stage3_to_5_code.py — hybrid-noCNN, สองขั้น (cross-fitting), shrinkage ต้องทำงานตามที่ประกาศไว้ (DECISIONS B25, B28, B32, B33)
===================================================================================================================================
ไฟล์นี้ทำอะไร
    * hybrid-noCNN = ANN เดียวกับ hybrid แต่ไม่มี CNN · ไม่ใส่ noise augmentation (ไม่มีภาพ) · ใช้แถวชุดเดียวกับ hybrid
    * cross-fitting: feature OOF ของช่วง j ต้องไม่ขึ้นกับ y ของช่วง j (ถ้าขึ้น = leak ที่ทำให้ ANN เชื่อ feature เกินจริง)
    * shrinkage: choose_c เลือก c ถูก เสมอกันเลือก c เล็กกว่า
ทำไมต้องมี: Stage 3–5 รันต่อกันโดยไม่มีคนคอยตรวจ ถ้าโค้ดเหล่านี้ผิด ผลทั้งหมดผิดตาม
"""

import numpy as np
import pandas as pd
import pytest
import torch
from dataclasses import replace

from src import data, models, placebo, runner, shrink, train, twostage
from src.configs import BASE, variant
from src.settings import VAL_END

NOCNN = variant("nocnn", ann_input="hybrid_nocnn")
HYB = variant("hyb", ann_input="hybrid")
TINY = dict(max_epochs=2, patience=2)


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n,) + BASE.image_shape).astype(np.float32)
    y = X.mean(axis=(1, 2, 3)) * 0.02 + rng.normal(0, 0.002, n)
    return X, y


# ---------------------------------------------------------------
# hybrid-noCNN
# ---------------------------------------------------------------
def test_nocnn_is_the_same_ann_head_without_a_cnn():
    a, b = models.build(NOCNN, 29), models.build(HYB, 29)
    assert isinstance(a, models.FeatureANN) and a.skip is not None and a.skip.in_features == 29
    assert not any("cnn" in n for n, _ in a.named_parameters())
    assert models.count_params(a) == 29 * 32 + 32 + 32 + 1 + 29 + 1, "หัว ANN 29→32→1 + linear skip จาก 29 feature"
    assert models.count_params(a) < models.count_params(b) / 5


def test_nocnn_takes_features_only_and_two_stage_is_rejected():
    assert train._inputs(NOCNN, "IMG", "FEAT") == ["FEAT"]
    assert train._inputs(HYB, "IMG", "FEAT") == ["IMG", "FEAT"] and train._inputs(BASE, "IMG", None) == ["IMG"]
    assert NOCNN.needs_features and HYB.needs_features and not BASE.needs_features
    with pytest.raises(ValueError):
        variant("bad", ann_input="hybrid_nocnn", train_mode="two_stage")


def test_nocnn_gets_no_noise_augmentation_but_hybrid_does(monkeypatch):
    """B25: noise ใช้กับภาพเท่านั้น — ตัว control ไม่มีภาพจึงต้องไม่เรียก torch.randn เพื่อใส่ noise เลย"""
    calls = []
    real = train.torch.randn
    monkeypatch.setattr(train.torch, "randn", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    X, y = toy()
    E = np.random.default_rng(1).normal(size=(len(y), 29)).astype(np.float32)
    ysc = (y - y.mean()) / y.std()
    train.fit(replace(NOCNN, **TINY), [E[:90]], ysc[:90], [E[90:]], ysc[90:], n_extra=29, seed=0)
    assert calls == [], "hybrid-noCNN ต้องไม่ใส่ noise (B25)"
    train.fit(replace(HYB, **TINY), [X[:90], E[:90]], ysc[:90], [X[90:], E[90:]], ysc[90:], n_extra=29, seed=0)
    assert len(calls) > 0, "hybrid ต้องใส่ noise กับภาพตามปกติ"


def test_nocnn_and_hybrid_use_exactly_the_same_rows():
    ds = data.build_dataset("KBANK", VAL_END)
    assert (ds.usable(NOCNN) == ds.usable(HYB)).all()
    # 29 feature ต้องการแค่ 21 แถวอุ่นเครื่อง แต่หน้าต่าง 128 วันต้องการมากกว่า → หน้าต่างเป็นตัวกำหนด
    # ผลดี: base / hybrid / hybrid-noCNN เทรนและประเมินบนแถวชุดเดียวกันทุกตัว (เทียบกันตรง ๆ ได้)
    assert (ds.usable(HYB) == ds.usable(BASE)).all()


# ---------------------------------------------------------------
# shrinkage
# ---------------------------------------------------------------
def test_choose_c_recovers_known_scalings():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 0.01, 500)
    assert shrink.choose_c(p, p) == 1.0                                   # ทำนายสมบูรณ์ → ไม่หด
    assert shrink.choose_c(0.5 * p, p) == 0.5                              # y = 0.5·pred → c = 0.5
    assert shrink.choose_c(rng.normal(0, 0.01, 500), p) == 0.0             # ไม่มีความสัมพันธ์ → หดเป็น 0 (= Naive)
    assert shrink.choose_c(-p, p) == 0.0, "สัญญาณกลับด้านไม่ควรถูกขยายหรือพลิก (c ไม่ติดลบ)"


def test_choose_c_ties_go_to_the_smaller_c_and_grid_is_correct():
    assert shrink.C_GRID == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert shrink.choose_c(np.zeros(10), np.zeros(10)) == 0.0              # ทุก c ให้ MAE เท่ากัน → เลือกเล็กสุด
    with pytest.raises(ValueError):
        shrink.choose_c([], [])


def test_apply_scales_prediction():
    np.testing.assert_allclose(shrink.apply([1.0, -2.0], 0.5), [0.5, -1.0])


# ---------------------------------------------------------------
# สองขั้น + cross-fitting
# ---------------------------------------------------------------
def test_blocks_are_contiguous_disjoint_and_cover_everything():
    b = twostage.block_slices(103)
    assert len(b) == 4 and sum(len(x) for x in b) == 103
    flat = np.concatenate(b)
    np.testing.assert_array_equal(flat, np.arange(103))
    assert all(x[-1] + 1 == y[0] for x, y in zip(b, b[1:])), "ต่อเนื่องตามเวลา"


def test_oof_feature_of_a_block_never_depends_on_that_blocks_y():
    """หัวใจของ cross-fitting: เปลี่ยน y ของช่วง j แล้ว OOF ของช่วง j ต้องเหมือนเดิมทุกบิต (ส่วนช่วงอื่นต้องเปลี่ยน)"""
    X, y = toy(n=160)
    Xev = toy(n=20, seed=5)[0]
    cfg = replace(BASE, **TINY, noise_std=0.0)
    oof_a, ev_a = twostage.oof_features(cfg, X, y, Xev, seed=0)
    blk = twostage.block_slices(len(y))[1]
    y2 = y.copy()
    y2[blk] = np.random.default_rng(9).normal(0, 1.0, len(blk))            # ทำลาย y ของช่วงที่ 1
    oof_b, ev_b = twostage.oof_features(cfg, X, y2, Xev, seed=0)
    np.testing.assert_array_equal(oof_a[blk], oof_b[blk])
    others = np.setdiff1d(np.arange(len(y)), blk)
    assert not np.allclose(oof_a[others], oof_b[others]), "ช่วงอื่นต้องเปลี่ยน (CNN ของมันเห็น y ที่ถูกแก้)"
    assert np.isfinite(oof_a).all() and ev_a.shape == (20,)


def _prepared(n=200, ne=30):
    X, y = toy(n=n + ne)
    return {"X_fit": X[:int(n * .85)], "X_val": X[int(n * .85):n], "X_eval": X[n:], "y_fit": y[:int(n * .85)],
            "y_val": y[int(n * .85):n], "y_eval": y[n:]}


def test_two_stage_returns_the_same_interface_as_run_fold_for_spec_and_hybrid():
    prep = _prepared()
    E = {k: np.random.default_rng(i).normal(size=(len(prep[f"X_{k}"]), 29)).astype(np.float32) for i, k in enumerate(("fit", "val", "eval"))}
    for cfg, extra in ((replace(variant("ts", train_mode="two_stage"), **TINY), None),
                       (replace(variant("tsh", train_mode="two_stage", ann_input="hybrid"), **TINY), E)):
        pred, hists, seed_preds, (vp, yv) = twostage.run_fold_two_stage(cfg, prep, seeds=(0, 1), extra=extra)
        assert pred.shape == (30,) and len(hists) == 2 and len(seed_preds) == 2 and np.isfinite(pred).all()
        assert len(vp) == len(yv) == int(round(200 * 0.15)), "ส่วน early-stop ของขั้น ANN = 15% ท้ายของ train"
        np.testing.assert_allclose(np.mean(seed_preds, axis=0), pred, rtol=1e-12, atol=1e-15)
        assert {"epochs", "best_epoch"} <= set(hists[0])


def test_two_stage_is_deterministic():
    prep, cfg = _prepared(), replace(variant("ts", train_mode="two_stage"), **TINY)
    a = twostage.run_fold_two_stage(cfg, prep, seeds=(0,))[0]
    b = twostage.run_fold_two_stage(cfg, prep, seeds=(0,))[0]
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------
# runner / evaluate / placebo
# ---------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("cfg", [NOCNN, HYB, variant("ts", train_mode="two_stage"),
                                 variant("tsh", train_mode="two_stage", ann_input="hybrid")], ids=lambda c: c.name)
def test_runner_unit_works_for_every_new_mode_and_reports_shrunk_columns(cfg):
    cfg = replace(cfg, **TINY)
    row, preds, hists = runner.run_unit(cfg, "KBANK", 3, seeds=(0, 1), cache_dir=None)
    assert row["c_chosen"] in shrink.C_GRID and np.isfinite(row["relMAE_shrunk"]) and "ic_s0" in row and "ic_s1" in row
    assert "pred_shrunk" in preds.columns and len(preds) == 266 and len(hists) == 2
    np.testing.assert_allclose(preds["pred_shrunk"], row["c_chosen"] * preds["pred"], rtol=1e-12, atol=1e-18)


def test_score_config_adds_shrunk_columns_when_present():
    from src import evaluate
    rows = [{"pred_finite": True, "relMAE": 1.01, "relMAE_baht": 1.01, "MAE_baht": 1.0, "RMSE_baht": 1.0, "MAE_return": 0.01,
             "R2_OOS": 0.0, "DirAcc": 0.5, "IC": 0.01, "IC_spearman": 0.0, "StdRatio": 0.1, "relMAE_shrunk": 1.0 + 0.001 * k,
             "relMAE_baht_shrunk": 1.0, "MAE_baht_shrunk": 1.0, "c_chosen": 0.0 if k < 2 else 0.5} for k in range(6)]
    s = evaluate.score_config(rows)
    assert s["n_c_zero"] == 2 and s["mean_c"] == pytest.approx(0.5 * 4 / 6) and s["n_evals_beat_naive_shrunk"] == 0
    assert s["mean_relMAE_shrunk"] == pytest.approx(1.0025)


def test_analyze_pair_threshold_is_the_max_delta_over_the_shuffles():
    def frame(ic_a, ic_b, s):
        rows = []
        for name, ic in (("hybrid", ic_a), ("hybrid_nocnn", ic_b)):
            for t in ("KBANK", "ADVANC"):
                for k in (1, 2, 3):
                    rows.append({"pred_finite": True, "config": name, "shuffle": s, "relMAE": 1.0, "relMAE_baht": 1.0, "MAE_baht": 1.0,
                                 "RMSE_baht": 1.0, "MAE_return": 0.01, "R2_OOS": 0.0, "DirAcc": 0.5, "IC": ic, "IC_spearman": 0.0,
                                 "StdRatio": 0.1, "ic_s0": ic - 0.01, "ic_s1": ic, "ic_s2": ic + 0.01})
        return pd.DataFrame(rows)
    ev = pd.concat([frame(0.02, 0.01, 0), frame(0.05, 0.00, 1), frame(-0.01, 0.00, 2), frame(0.00, 0.03, 3), frame(0.03, 0.01, 4)])
    r = placebo.analyze_pair(ev, "hybrid", "hybrid_nocnn", (0, 1, 2, 3, 4))
    assert r["per_shuffle"]["delta_ic"].tolist() == pytest.approx([0.01, 0.05, -0.01, -0.03, 0.02])
    assert r["threshold_pair"] == pytest.approx(0.05) and r["n_shuffles"] == 5
