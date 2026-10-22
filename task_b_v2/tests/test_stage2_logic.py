"""
test_stage2_logic.py — config ของ Stage 2 ต้องต่างจาก base ทีละปัจจัยตาม B27 และอ่านเกณฑ์ B22 จากผล placebo
================================================================================================================
ไฟล์นี้ทำอะไร: ยืนยันรายการ config (4 ตัว), บทบาทของแต่ละตัว (ผู้สมัคร/ตัวเทียบ), filter (8,16) ตาม B13, และการอ่านค่าเกณฑ์ B22
ทำไมต้องมี: Stage 2 รันครั้งเดียว ถ้า config ผิดหรือเกณฑ์ผิดจะรู้ตอนเห็นผลแล้ว
"""

import dataclasses
import json

import pytest

from src import run_stage2
from src.configs import BASE


def changed(cfg):
    a, b = dataclasses.asdict(BASE), dataclasses.asdict(cfg)
    return {k for k in a if a[k] != b[k] and k != "name"}


def test_four_configs_base_first():
    names = [c.name for c in run_stage2.all_configs()]
    assert names == ["base", "cnn1d_time", "gap", "cnn1d_raw"]


@pytest.mark.parametrize("name,expected", [("cnn1d_time", {"arch"}), ("gap", {"head"}), ("cnn1d_raw", {"arch"})])
def test_each_option_changes_exactly_one_field(name, expected):
    assert changed(next(c for c in run_stage2.all_configs() if c.name == name)) == expected


def test_candidates_are_arch_and_head_only_and_raw_is_a_comparator():
    assert {f: list(o) for f, o in run_stage2.FACTORS.items()} == {"arch": ["cnn1d_time"], "head": ["gap"]}
    assert list(run_stage2.COMPARATORS) == ["cnn1d_raw"]
    assert all("cnn1d_raw" not in opts for opts in run_stage2.FACTORS.values()), "ค. เป็นตัวเทียบ ไม่ใช่ผู้สมัคร (B27)"


def test_1d_variants_use_the_same_filters_as_the_2d_base():
    """B13: filter (8, 16) ทั้งสามตัว — เทียบ 1D กับ 2D ที่ความจุใกล้เคียงกัน"""
    for c in run_stage2.all_configs():
        assert c.filters == BASE.filters == (8, 16)


def test_reads_b22_threshold_from_the_placebo_decision_file(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"b22_threshold_stage2_to_5": 0.0597}), encoding="utf-8")
    assert run_stage2.read_b22_threshold(f) == pytest.approx(0.0597)


@pytest.mark.skipif(not run_stage2.B22_FILE.exists(), reason="ยังไม่มีไฟล์ผล placebo")
def test_real_threshold_is_the_value_the_user_confirmed():
    assert run_stage2.read_b22_threshold() == pytest.approx(0.0597)
