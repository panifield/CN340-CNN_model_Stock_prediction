"""
test_train.py — loop เทรนต้องรันซ้ำได้ผลเดิม, early stopping ต้องใช้ val จริง, y ต้องถูก scale ด้วยสถิติ train
=========================================================================================================
ไฟล์นี้ทำอะไร: ทดสอบ `src/train.py` ด้วยข้อมูลสังเคราะห์ขนาดเล็ก (เร็ว · ไม่ต้องใช้ข้อมูลจริง)
ทำไมต้องมี
    * ถ้าผลรันซ้ำไม่เหมือนเดิม จะเทียบ config ไม่ได้เลย (ความต่างระหว่าง config เล็กกว่า noise ของ seed)
    * ถ้า TargetScaler ใช้สถิติผิดชุด δ ของ Huber จะไม่มีความหมาย (return รายวัน ~0.01 แต่ δ = 1)
"""

import numpy as np
import pytest
from dataclasses import replace

from src import train
from src.configs import BASE, variant


def toy(n=120, cfg=BASE, seed=0):
    """ข้อมูลปลอม: y เป็นฟังก์ชันของภาพ (เรียนได้) + noise"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n,) + cfg.image_shape).astype(np.float32)
    y = X.mean(axis=tuple(range(1, X.ndim))) * 0.02 + rng.normal(0, 0.002, n)
    return X, y


FAST = replace(BASE, max_epochs=6, patience=6, noise_std=0.0)


def test_same_seed_gives_identical_predictions():
    X, y = toy()
    out = []
    for _ in range(2):
        m, _ = train.fit(FAST, [X[:90]], y[:90], [X[90:]], y[90:], seed=0)
        out.append(train.predict(m, [X[90:]]))
    np.testing.assert_array_equal(out[0], out[1], err_msg="รันซ้ำด้วย seed เดิมต้องได้ผลเหมือนกันทุกบิต")


def test_different_seeds_give_different_predictions():
    X, y = toy()
    a, _ = train.fit(FAST, [X[:90]], y[:90], [X[90:]], y[90:], seed=0)
    b, _ = train.fit(FAST, [X[:90]], y[:90], [X[90:]], y[90:], seed=1)
    assert not np.allclose(train.predict(a, [X[90:]]), train.predict(b, [X[90:]]))


def test_early_stopping_keeps_the_best_val_state_not_the_last():
    X, y = toy(n=200)
    cfg = replace(BASE, max_epochs=40, patience=3, noise_std=0.0)
    model, hist = train.fit(cfg, [X[:150]], y[:150], [X[150:]], y[150:], seed=0)
    assert hist["epochs"] <= cfg.max_epochs
    assert hist["best_val"] == pytest.approx(min(hist["val_loss"])), "ต้องเก็บ state ของ epoch ที่ val ดีที่สุด"
    if hist["epochs"] < cfg.max_epochs:
        assert hist["val_loss"][-1] >= hist["best_val"], "หยุดเพราะ val ไม่ดีขึ้นแล้ว"


def test_patience_stops_training_early():
    X, y = toy(n=200)
    short = train.fit(replace(BASE, max_epochs=40, patience=1, noise_std=0.0),
                      [X[:150]], y[:150], [X[150:]], y[150:], seed=0)[1]["epochs"]
    assert short < 40


@pytest.mark.parametrize("loss", ["huber", "mae", "mse"])
def test_every_loss_trains(loss):
    X, y = toy()
    m, hist = train.fit(replace(FAST, loss=loss), [X[:90]], y[:90], [X[90:]], y[90:], seed=0)
    assert np.isfinite(hist["best_val"]) and np.isfinite(train.predict(m, [X[90:]])).all()


def test_unknown_loss_is_rejected():
    from types import SimpleNamespace
    with pytest.raises(KeyError):
        variant("bad", loss="mape")                       # Config ปฏิเสธตั้งแต่สร้าง
    with pytest.raises(KeyError):
        train.loss_fn(SimpleNamespace(loss="mape"))       # และ loss_fn ก็ปฏิเสธด้วย


# ---------------------------------------------------------------
# TargetScaler
# ---------------------------------------------------------------
def test_target_scaler_round_trip():
    y = np.random.default_rng(0).normal(0.0003, 0.012, 500)
    s = train.TargetScaler().fit(y)
    z = s.transform(y)
    assert abs(float(z.mean())) < 1e-5 and float(z.std()) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(s.inverse(z), y, rtol=1e-5, atol=1e-9)


def test_target_scaler_uses_train_statistics_only():
    """สถิติต้องมาจาก y ที่ fit เท่านั้น — ข้อมูลประเมินที่มี scale ต่างออกไปต้องไม่ถูกจัดกึ่งกลางเป็น 0"""
    y_train = np.random.default_rng(0).normal(0.0, 0.01, 300)
    y_eval = np.random.default_rng(1).normal(0.05, 0.03, 100)
    s = train.TargetScaler().fit(y_train)
    assert abs(float(s.transform(y_eval).mean())) > 1, "ถ้าใกล้ 0 แปลว่าเผลอ fit ด้วย y ของช่วงประเมิน"


def test_target_scaler_rejects_constant_y():
    with pytest.raises(ValueError):
        train.TargetScaler().fit(np.ones(10))


# ---------------------------------------------------------------
# hybrid : ต้องรับ extra feature เข้า ANN จริง
# ---------------------------------------------------------------
def test_hybrid_uses_the_extra_features():
    cfg = replace(variant("hyb", ann_input="hybrid"), max_epochs=6, patience=6, noise_std=0.0)
    X, y = toy()
    rng = np.random.default_rng(2)
    E = rng.normal(0, 1, (len(y), 29)).astype(np.float32)
    m, _ = train.fit(cfg, [X[:90], E[:90]], y[:90], [X[90:], E[90:]], y[90:], n_extra=29, seed=0)
    base = train.predict(m, [X[90:], E[90:]])
    other = train.predict(m, [X[90:], rng.normal(0, 1, E[90:].shape).astype(np.float32)])
    assert not np.allclose(base, other), "เปลี่ยน extra feature แล้วคำทำนายต้องเปลี่ยน"


def test_run_fold_can_return_per_seed_predictions_whose_mean_is_the_ensemble():
    """IC รายseed (DECISIONS B17) ต้องมาจากคำทำนายของแต่ละ seed เดี่ยว ๆ ซึ่งเฉลี่ยแล้วต้องเท่ากับคำทำนาย ensemble"""
    X, y = toy(n=160)
    prepared = {"X_fit": X[:100], "X_val": X[100:130], "X_eval": X[130:], "y_fit": y[:100], "y_val": y[100:130]}
    cfg = replace(BASE, max_epochs=3, patience=3, noise_std=0.0)
    pred, hists, seed_preds = train.run_fold(cfg, prepared, seeds=(0, 1, 2), return_hist=True, return_seed_preds=True)
    assert len(hists) == 3 and len(seed_preds) == 3 and all(len(sp) == 30 for sp in seed_preds)
    np.testing.assert_allclose(np.mean(seed_preds, axis=0), pred, rtol=1e-12, atol=1e-15)
    assert not np.allclose(seed_preds[0], seed_preds[1]), "คนละ seed ต้องให้คำทำนายต่างกัน"
    # ขอแค่ pred → คืนค่าเดียว (ไม่เปลี่ยนพฤติกรรมเดิมของ controls)
    only = train.run_fold(cfg, prepared, seeds=(0,))
    assert isinstance(only, np.ndarray)
