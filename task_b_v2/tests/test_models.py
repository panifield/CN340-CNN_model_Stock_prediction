"""
test_models.py — ขนาดทุกชั้นและจำนวนพารามิเตอร์ต้องตรงกับสูตรมือ (SPEC ข้อ 5.0)
=============================================================================
ไฟล์นี้ทำอะไร: ตรวจว่าโครง CNN ที่เขียนไว้ให้ shape และจำนวนพารามิเตอร์ตามที่คำนวณด้วยสูตร
    out = (W − K + 2P)/S + 1   ·   params ของ conv = Cin·K·K·Cout + Cout
ทำไมต้องมี: สเปกบังคับให้ตรวจมือ · และถ้าโครงเปลี่ยนโดยไม่ตั้งใจ (เช่นลืม pad) เทสต์นี้จับได้ทันที
"""

import numpy as np
import pytest
import torch

from src import models
from src.configs import BASE, variant


def test_base_architecture_matches_spec():
    rows, total = models.layer_report(BASE)
    shapes = [r["out_shape"] for r in rows]
    assert shapes[0] == (1, 9, 65)                      # input
    assert (8, 9, 65) in shapes and (8, 4, 32) in shapes      # conv1 แล้ว pool
    assert (16, 4, 32) in shapes and (16, 2, 16) in shapes    # conv2 แล้ว pool
    assert (512,) in shapes                                    # flatten = 16·2·16
    assert sum(1 for r in rows if r["layer"] == "= CNN feature") == 1
    assert next(r for r in rows if r["layer"] == "= CNN feature")["out_shape"] == (16,)
    assert total == 80 + 16 + 1168 + 32 + 8208 + 544 + 33 == 10081


@pytest.mark.parametrize("w,k,p,s,expected", [(9, 3, 1, 1, 9), (65, 3, 1, 1, 65), (9, 2, 0, 2, 4), (65, 2, 0, 2, 32)])
def test_conv_output_formula(w, k, p, s, expected):
    assert models.conv_output_size(w, k, p, s) == expected


@pytest.mark.parametrize("cin,k,cout,expected", [(1, 3, 8, 80), (8, 3, 16, 1168), (3, 3, 8, 224)])
def test_conv_param_formula(cin, k, cout, expected):
    assert models.conv_param_count(cin, k, cout) == expected


@pytest.mark.parametrize("cfg,n_extra", [
    (BASE, 0),
    (variant("hybrid", ann_input="hybrid"), 29),
    (variant("gap", head="gap"), 0),
    (variant("c1d", arch="cnn1d_time"), 0),
    (variant("raw", arch="cnn1d_raw"), 0),
    (variant("cplx", spec_type="complex", scale_mode="fixed"), 0),
    (variant("ch4", channels="ret4"), 0),
])
def test_forward_gives_one_number_per_row(cfg, n_extra):
    torch.manual_seed(0)
    m = models.build(cfg, n_extra)
    m.eval()
    x = torch.randn((7,) + tuple(cfg.image_shape))
    extra = torch.randn(7, n_extra) if n_extra else None
    with torch.no_grad():
        out = m(x, extra)
    assert out.shape == (7,) and torch.isfinite(out).all()


def test_gap_head_has_far_fewer_parameters_than_flatten():
    _, flat = models.layer_report(BASE)
    _, gap = models.layer_report(variant("gap", head="gap"))
    assert gap < flat / 4, "GAP ต้องประหยัดพารามิเตอร์กว่า flatten มาก (ทิ้งตำแหน่งเวลา)"


def test_hybrid_has_linear_skip_from_the_29_features():
    cfg = variant("hybrid", ann_input="hybrid")
    m = models.build(cfg, 29)
    assert m.skip is not None and m.skip.in_features == 29
    # ถ้า ANN ถูกกดเป็น 0 ทั้งหมด โมเดลต้องยังเหลือส่วนเชิงเส้นของ 29 feature (ข้อดีของ skip)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.ann[-1].parameters():
            p.zero_()
        x, e = torch.randn(5, *cfg.image_shape), torch.randn(5, 29)
        assert torch.allclose(m(x, e), m.skip(e).squeeze(-1))


def test_hybrid_requires_extra_features():
    with pytest.raises(ValueError):
        models.build(variant("hybrid", ann_input="hybrid"), 0)


def test_same_seed_gives_identical_initial_weights():
    def weights():
        torch.manual_seed(3)
        return torch.cat([p.detach().flatten() for p in models.build(BASE).parameters()])
    assert torch.equal(weights(), weights())
