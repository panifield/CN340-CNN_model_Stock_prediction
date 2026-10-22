"""
test_dataset.py — ขนาดภาพ, การตัดโซน, 29 feature, และ walk-forward fold ต้องถูกต้อง (SPEC ข้อ 4, 5.1)
=================================================================================================
ไฟล์นี้ทำอะไร: ตรวจว่า dataset ที่สร้างออกมาตรงกับที่สเปกกำหนด ทั้งขนาดภาพ (9×65 สำหรับ base),
การตัดข้อมูลที่ 2025-02-25, และโครง fold (3 fold × 266 วัน จบที่วันสุดท้ายของ dev พอดี)
ทำไมต้องมี: ตัวเลขเหล่านี้คือ protocol ที่ประกาศไว้ล่วงหน้า ถ้าเพี้ยนทีหลังผลทุกตัวจะเทียบกันไม่ได้
"""

import numpy as np
import pandas as pd
import pytest

from src import data
from src.configs import BASE, Config, variant
from src.settings import EARLY_STOP_FRAC, FOLD_DAYS, N_FOLDS, TICKERS, VAL_END


@pytest.fixture(scope="module")
def datasets():
    return data.build_all(VAL_END)


# ---------------------------------------------------------------
# ขนาดภาพ (สูตร L5: (W − K + 2P)/S + 1)
# ---------------------------------------------------------------
@pytest.mark.parametrize("cfg,expected", [
    (BASE, (1, 9, 65)),                                             # window 128 / nperseg 16 / hop 2 ตามสเปกข้อ 5.1
    (variant("w64", window=64), (1, 9, 33)),
    (variant("w256", window=256, nperseg=32, hop=4), (1, 17, 65)),
    (variant("cplx", spec_type="complex"), (2, 9, 65)),
    (variant("ch3", channels="ret3"), (3, 9, 65)),
    (variant("ch4c", channels="ret4", spec_type="complex"), (8, 9, 65)),
    (variant("raw", arch="cnn1d_raw"), (1, 128)),
])
def test_image_shape_formula_matches_actual_stft(cfg, expected, datasets):
    assert cfg.image_shape == expected
    X, has = datasets[TICKERS[0]].raw_images(cfg, [len(datasets[TICKERS[0]]) - 1])
    assert has[0] and X.shape[1:] == expected and np.isfinite(X[0]).all()


def test_config_rejects_bad_values():
    with pytest.raises(KeyError):
        Config(channels="ไม่มีช่องนี้")
    with pytest.raises(KeyError):
        Config(spec_type="db")
    with pytest.raises(ValueError):
        Config(window=8, nperseg=16)


# ---------------------------------------------------------------
# โซนข้อมูล
# ---------------------------------------------------------------
@pytest.mark.parametrize("ticker", TICKERS)
def test_zone_is_cut_at_val_end(datasets, ticker):
    ds = datasets[ticker]
    assert ds.dates[-1] == pd.Timestamp(VAL_END), "แถวสุดท้ายของ dev ต้องเป็น 2025-02-25 พอดี"
    assert len(ds) == 2071
    assert ds.feat29.shape == (2071, 29)


@pytest.mark.parametrize("ticker", TICKERS)
def test_target_definition(datasets, ticker):
    """y = Close(t)/Close(t−1) − 1 และ prev_close = Close(t−1)"""
    ds = datasets[ticker]
    c = ds.prices["Close"].to_numpy()
    np.testing.assert_allclose(ds.prev_close[1:], c[:-1])
    np.testing.assert_allclose(ds.y[1:], c[1:] / c[:-1] - 1)
    assert np.isnan(ds.y[0])
    # ราคาที่ทำนายได้จาก y ต้องกลับไปเป็นราคาจริง (ข้อ 3.1)
    np.testing.assert_allclose(ds.prev_close[1:] * (1 + ds.y[1:]), ds.close[1:], rtol=1e-12)


def test_both_tickers_have_identical_calendar(datasets):
    a, b = (datasets[t].dates for t in TICKERS)
    assert a.equals(b)


# ---------------------------------------------------------------
# fold (DECISIONS ข้อ B6)
# ---------------------------------------------------------------
def test_folds_cover_last_798_days_and_end_at_val_end(datasets):
    ds = datasets[TICKERS[0]]
    folds = data.make_folds(ds.dates)
    assert len(folds) == N_FOLDS and all(len(f.eval_pos) == FOLD_DAYS for f in folds)
    assert folds[0].eval_pos[0] == len(ds) - N_FOLDS * FOLD_DAYS
    assert ds.dates[folds[-1].eval_pos[-1]] == pd.Timestamp(VAL_END)
    # ต่อกันไม่ซ้อนทับ และ train ของทุก fold จบก่อนช่วงประเมิน
    for f in folds:
        assert f.train_stop == f.eval_pos[0]
    assert folds[0].eval_pos[0] == 1273


def test_eval_days_are_identical_for_every_config(datasets):
    """หัวใจของ SPEC ข้อ 4.2 : window ยาวขึ้นต้องตัดแค่ฝั่ง train ห้ามเปลี่ยนวันประเมิน"""
    ds = datasets[TICKERS[0]]
    folds = data.make_folds(ds.dates)
    cfgs = [BASE, variant("w64", window=64), variant("w256", window=256, nperseg=32, hop=4),
            variant("hyb", ann_input="hybrid")]
    eval_days = {f.k: set(ds.dates[f.eval_pos]) for f in folds}
    train_sizes = {}
    for cfg in cfgs:
        usable = ds.usable(cfg)
        for f in folds:
            assert usable[f.eval_pos].all(), f"{cfg.name} fold {f.k}: วันประเมินต้องใช้ได้ทุกวัน"
            assert set(ds.dates[f.eval_pos]) == eval_days[f.k]
            train_sizes.setdefault(f.k, {})[cfg.name] = len(f.train_pos(usable))
    # window ยาวกว่า → train น้อยกว่า (ยืนยันว่าการตัดไปลงที่ฝั่ง train จริง)
    assert train_sizes[1]["w256"] < train_sizes[1]["base"] < train_sizes[1]["w64"]


@pytest.mark.parametrize("ticker", TICKERS)
def test_fold_summary_is_chronological(datasets, ticker):
    ds = datasets[ticker]
    df = data.fold_summary(ds, data.make_folds(ds.dates), BASE)
    assert len(df) == N_FOLDS
    for r in df.itertuples():
        assert r.train_last < r.eval_first
    assert df["eval_last"].iloc[-1] == VAL_END


# ---------------------------------------------------------------
# early stopping split
# ---------------------------------------------------------------
def test_early_stop_split_takes_the_tail():
    fit, val = data.early_stop_split(1000, EARLY_STOP_FRAC)
    assert fit.sum() == 850 and val.sum() == 150
    assert fit[:850].all() and val[850:].all(), "val ต้องเป็นช่วงท้ายตามเวลา (ห้าม shuffle)"
    with pytest.raises(ValueError):
        data.early_stop_split(1, EARLY_STOP_FRAC)
