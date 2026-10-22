"""
test_transform.py — ImageTransform ต้อง fit จากแถว fit เท่านั้น และต้องบีบอัดค่าจริง
===================================================================================
ไฟล์นี้ทำอะไร
    1. ตรวจว่าสเกลและสถิติ normalize มาจากแถว fit (85% แรกของ train) เท่านั้น — ไม่ขยับเมื่อราคาอนาคตเปลี่ยน
    2. ตรวจว่า `log1p(|Z|²/m)` บีบอัดจริง (ต่างจาก `log1p(|Z|²)` ที่วัดได้ corr = 1.000000 ซึ่งเป็นที่มาของ DECISIONS ข้อ B9)
    3. ตรวจว่าภาพหลัง normalize มี mean ≈ 0 std ≈ 1 ตามโหมดที่เลือก

ทำไมต้องมี: การแปลงภาพเป็นส่วนหนึ่งของโมเดล ถ้าสถิติของมันเห็นช่วงประเมิน = leak ที่มองไม่เห็นจากผลลัพธ์
"""

import numpy as np
import pytest

from src import data
from src.configs import BASE, variant
from src.settings import EARLY_STOP_FRAC, TICKERS, VAL_END
from src.spectrogram import ImageTransform

TICKER = TICKERS[0]


@pytest.fixture(scope="module")
def ds():
    return data.build_dataset(TICKER, VAL_END)


@pytest.fixture(scope="module")
def fold1(ds):
    return data.make_folds(ds.dates)[0]


def test_scale_is_median_of_fit_rows(ds, fold1):
    tr = fold1.train_pos(ds.usable(BASE))
    fit_m, _ = data.early_stop_split(len(tr), EARLY_STOP_FRAC)
    raw, has = ds.raw_images(BASE, tr)
    assert has.all()
    tf = ImageTransform(BASE).fit(raw[fit_m])
    assert tf.scale == pytest.approx(float(np.median(np.abs(raw[fit_m]))))
    assert tf.scale != pytest.approx(float(np.median(np.abs(raw))), rel=1e-9), \
        "สเกลต้องมาจากแถว fit เท่านั้น ไม่ใช่ train ทั้งก้อน"


def test_fitted_stats_ignore_future_prices(ds, fold1):
    """เปลี่ยนราคาตั้งแต่วันแรกของช่วงประเมินเป็นต้นไป → สเกลและสถิติ normalize ต้องไม่ขยับเลย"""
    from tests.test_leak import _perturbed_prices
    alt = data.dataset_from_prices(TICKER, _perturbed_prices(ds.prices, fold1.eval_pos[0], seed=7), VAL_END)
    a = data.fold_inputs(ds, BASE, fold1)["transform"]
    b = data.fold_inputs(alt, BASE, fold1)["transform"]
    assert a.scale == b.scale
    np.testing.assert_array_equal(np.asarray(a.mean), np.asarray(b.mean))
    np.testing.assert_array_equal(np.asarray(a.std), np.asarray(b.std))


def test_log1p_with_fitted_scale_actually_compresses(ds, fold1):
    """
    หัวใจของ DECISIONS ข้อ B9 : ถ้าไม่หารด้วยสเกล log1p จะเป็นการแปลงเชิงเส้น (corr ≈ 1)
    หารด้วย median แล้วต้องเห็นความไม่เป็นเชิงเส้นชัดเจน
    """
    raw, _ = ds.raw_images(BASE, fold1.train_pos(ds.usable(BASE)))
    flat = raw.ravel()
    assert np.corrcoef(np.log1p(flat), flat)[0, 1] == pytest.approx(1.0, abs=1e-6), \
        "ถ้าค่านี้ไม่ใช่ 1 แปลว่าข้อมูลเปลี่ยนไป -- ทบทวน DECISIONS ข้อ B9 ใหม่"
    scaled = np.log1p(flat / np.median(flat))
    assert np.corrcoef(scaled, flat)[0, 1] < 0.9, "หารด้วย median แล้วต้องบีบอัดจริง"
    assert np.percentile(scaled, 99) / np.median(scaled) < 10, "หางของการแจกแจงต้องสั้นลงมาก"


@pytest.mark.parametrize("mode,expect_per_band", [("band", True), ("global", False), ("image", False)])
def test_normalized_images_are_centered(ds, fold1, mode, expect_per_band):
    cfg = variant(f"norm_{mode}", normalize=mode)
    out = data.fold_inputs(ds, cfg, fold1)
    X = out["X_fit"]
    assert X.shape[1:] == cfg.image_shape and np.isfinite(X).all()
    if mode == "image":
        np.testing.assert_allclose(X.mean(axis=(1, 2, 3)), 0, atol=1e-4)
        np.testing.assert_allclose(X.std(axis=(1, 2, 3)), 1, atol=1e-3)
    elif expect_per_band:
        np.testing.assert_allclose(X.mean(axis=(0, 3)), 0, atol=1e-4)      # mean ต่อ (channel, แถบความถี่)
        np.testing.assert_allclose(X.std(axis=(0, 3)), 1, atol=1e-3)
    else:
        assert abs(float(X.mean())) < 1e-4 and float(X.std()) == pytest.approx(1, abs=1e-3)


def test_complex_uses_the_ported_fixed_scale(ds, fold1):
    """complex ที่ port มาใช้ arcsinh(z·1e3) = หารด้วย 1e-3 (DECISIONS ข้อ B10)"""
    cfg = variant("complex", spec_type="complex", scale_mode="fixed")
    tf = data.fold_inputs(ds, cfg, fold1)["transform"]
    assert tf.scale == pytest.approx(1e-3)


def test_fit_rejects_rows_with_nan():
    cfg = BASE
    with pytest.raises(ValueError):
        ImageTransform(cfg).fit(np.full((3,) + cfg.image_shape, np.nan, dtype=np.float32))


# ---------------------------------------------------------------
# fold_inputs
# ---------------------------------------------------------------
def test_fold_inputs_splits_are_chronological_and_disjoint(ds):
    for fold in data.make_folds(ds.dates):
        out = data.fold_inputs(ds, BASE, fold)
        pf, pv, pe = out["pos_fit"], out["pos_val"], out["pos_eval"]
        assert pf.max() < pv.min() < pe.min(), "ลำดับต้องเป็น fit → val → eval ตามเวลา"
        assert not (set(pf) | set(pv)) & set(pe)
        n_train = len(pf) + len(pv)
        assert len(pv) == max(1, round(n_train * EARLY_STOP_FRAC)), "val ต้องเป็น 15% ท้ายของ train"
        for key, pos in (("X_fit", pf), ("X_val", pv), ("X_eval", pe)):
            assert out[key].shape == (len(pos),) + BASE.image_shape
            assert np.isfinite(out[key]).all()
        np.testing.assert_allclose(out["prev_close_eval"] * (1 + out["y_eval"]), out["close_eval"], rtol=1e-12)
