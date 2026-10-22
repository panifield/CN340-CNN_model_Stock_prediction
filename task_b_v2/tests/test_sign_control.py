"""
test_sign_control.py — target ของ sign control ต้องเป็นฟังก์ชันของ input ล้วน และข้อตั้งต้นเชิงทฤษฎีต้องเป็นจริง
===================================================================================================================
ไฟล์นี้ทำอะไร
    1. target = ln(C_{t−1}/C_{t−2}) ตรงตามนิยาม และ **ไม่ขึ้นกับราคาวัน t เป็นต้นไป** (ไม่ leak)
    2. ข้อตั้งต้นของทฤษฎี: ค่าดิบ log-power (|Z|²) ของหน้าต่าง x กับ −x เหมือนกันทุกพิกเซล แต่ complex (Re/Im) กลับเครื่องหมาย
       — ถ้าสองข้อนี้ไม่จริง การคาดว่า "log-power R² ≈ 0" ก็ไม่มีเหตุผลรองรับ
    3. ตรรกะตัดสินคาดการณ์ตาม "วิธีอ่านผล" ที่ประกาศไว้ก่อนรัน (DECISIONS B24)
ทำไมต้องมี: เทสต์ข้อ 2 พิสูจน์เฉพาะ "ข้อตั้งต้น" ไม่ได้พิสูจน์ผลของ CNN (ผลต้องมาจากการรันจริง)
"""

import numpy as np
import pytest

from src import data, sign_control
from src.configs import BASE, variant
from src.spectrogram import channel_matrix, raw_coeffs


@pytest.fixture(scope="module")
def ds():
    return data.build_dataset("KBANK", "2025-02-25")


def test_target_is_the_last_return_in_the_window(ds):
    pos = np.array([300, 1000, 2000])
    c = ds.prices["Close"].to_numpy()
    np.testing.assert_allclose(sign_control.last_return_target(ds, pos), np.log(c[pos - 1] / c[pos - 2]))


def test_target_is_the_final_element_of_the_image_input_window(ds):
    """หน้าต่างของวัน p คือ ret_close[p−window : p] → องค์ประกอบสุดท้ายคือ ret_close[p−1] = target"""
    arr, _ = channel_matrix(ds.prices, BASE)
    pos = np.array([500, 1500])
    last_in_window = np.array([arr[p - BASE.window:p, 0][-1] for p in pos])
    np.testing.assert_array_equal(sign_control.last_return_target(ds, pos), last_in_window)


def test_target_does_not_depend_on_prices_from_day_t_onwards(ds):
    from tests.test_leak import _perturbed_prices
    pos = len(ds) - 40
    alt = data.dataset_from_prices("KBANK", _perturbed_prices(ds.prices, pos, seed=3), "2025-02-25")
    p = np.array([pos])
    np.testing.assert_array_equal(sign_control.last_return_target(ds, p), sign_control.last_return_target(alt, p))
    # แต่ target ของวัน t+1 ต้องเปลี่ยน (มันคือ return ของวัน t ซึ่งถูกแก้)
    assert sign_control.last_return_target(ds, p + 1)[0] != sign_control.last_return_target(alt, p + 1)[0]


# ---------------------------------------------------------------
# ข้อตั้งต้นของทฤษฎี
# ---------------------------------------------------------------
def test_logpower_coefficients_cannot_see_the_sign_of_the_window(ds):
    arr, _ = channel_matrix(ds.prices, BASE)
    w = arr[800:800 + BASE.window]
    np.testing.assert_allclose(raw_coeffs(w, BASE), raw_coeffs(-w, BASE), rtol=1e-5, atol=1e-12)


def test_complex_coefficients_flip_sign_with_the_window(ds):
    cfg = variant("c", spec_type="complex", scale_mode="fixed")
    arr, _ = channel_matrix(ds.prices, cfg)
    w = arr[800:800 + cfg.window]
    np.testing.assert_allclose(raw_coeffs(w, cfg), -raw_coeffs(-w, cfg), rtol=1e-5, atol=1e-12)
    assert np.abs(raw_coeffs(w, cfg)).max() > 0


def test_sign_flip_changes_the_target_but_not_the_logpower_image(ds):
    """รวมสองข้อ: ภาพ log-power ของ x กับ −x เหมือนกัน ในขณะที่ target (เครื่องหมายของ return ล่าสุด) กลับด้าน"""
    arr, _ = channel_matrix(ds.prices, BASE)
    w = arr[800:800 + BASE.window]
    assert np.sign(w[-1, 0]) == -np.sign((-w)[-1, 0]) != 0
    np.testing.assert_allclose(raw_coeffs(w, BASE), raw_coeffs(-w, BASE), rtol=1e-5, atol=1e-12)


# ---------------------------------------------------------------
# config และตรรกะตัดสินผล
# ---------------------------------------------------------------
def test_four_sign_configs_with_expected_shapes():
    shapes = {lab: cfg.image_shape for lab, cfg in sign_control.SIGN_CONFIGS}
    assert shapes == {"logpower_w128": (1, 9, 65), "complex_w128": (2, 9, 65),
                      "logpower_w64": (1, 9, 33), "complex_w64": (2, 9, 33)}
    assert [c.scale_mode for _, c in sign_control.SIGN_CONFIGS] == ["fit_median", "fixed", "fit_median", "fixed"]


def _rows(label, r2, corr):
    return [{"config": label, "R2": r2, "corr": corr} for _ in range(6)]


def test_expectation_logic_matches_the_announced_reading():
    ok = sign_control.summarize_expectation(_rows("logpower_w128", 0.02, 0.05))[0]["สรุป"]
    assert "ยืนยัน" in ok
    bad_r2 = sign_control.summarize_expectation(_rows("logpower_w128", 0.06, 0.05))[0]["สรุป"]
    bad_corr = sign_control.summarize_expectation(_rows("logpower_w128", 0.0, 0.11))[0]["สรุป"]      # เกินเส้น 0.10 ชัดเจน (กันปัญหาทศนิยมที่เส้นพอดี)
    assert "ขัด" in bad_r2 and "ขัด" in bad_corr
    assert "ต่ำ" in sign_control.summarize_expectation(_rows("complex_w128", 0.29, 0.5))[0]["สรุป"]
    assert "เรียนเครื่องหมายได้" in sign_control.summarize_expectation(_rows("complex_w128", 0.30, 0.6))[0]["สรุป"]


@pytest.mark.slow
def test_probe_returns_finite_results_for_every_stage_and_frame_set(ds):
    fold = data.make_folds(ds.dates)[2]
    cfg = variant("c", spec_type="complex", scale_mode="fixed")
    res = sign_control.probe_r2(ds, cfg, fold)
    assert len(res) == len(sign_control.STAGES) * len(sign_control.FRAME_SETS)
    assert all(np.isfinite(r["R2"]) and np.isfinite(r["corr"]) for r in res)
    n_all = next(r["n_features"] for r in res if r["stage"] == "stft" and r["frames"] == "all")
    n_last = next(r["n_features"] for r in res if r["stage"] == "stft" and r["frames"] == "last")
    assert n_all == 2 * 9 * 65 and n_last == 2 * 9
