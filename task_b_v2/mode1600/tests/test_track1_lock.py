"""
test_track1_lock.py — config track 1 ของโหมด 16:00 ต้องตรงกับ Track 1 ใน LOCK.md ของ next_day หัวข้อ 5 ทุก field (M22)
================================================================================================================
อ่าน task_b_v2/LOCK.md (อ่านอย่างเดียว) · ข้ามถ้ายังไม่มีไฟล์ · มี test ด้านลบว่าตัวตรวจจับ field ที่ต่างได้จริง
"""

from dataclasses import fields, replace

import pytest

from mode1600 import draft_lock1600 as DL
from mode1600 import locked1600 as L
from src.configs import BASE, Config

needs_lock = pytest.mark.skipif(not DL.NEXTDAY_LOCK_PATH.exists(), reason="ยังไม่มี task_b_v2/LOCK.md")


@needs_lock
def test_track1_matches_lock_md_every_field():
    want = DL.nextday_track1(DL.NEXTDAY_LOCK_PATH.read_text(encoding="utf-8"))
    assert set(want) == {f.name for f in fields(Config)} - {"name"}, "ตาราง LOCK.md ต้องมีครบทุก field ของ Config"
    assert DL.check_track1_matches_nextday_lock() == []


@needs_lock
def test_checker_detects_a_changed_field():
    assert DL.check_track1_matches_nextday_lock(replace(BASE, lr=3e-3)) == ["lr"]
    assert DL.check_track1_matches_nextday_lock(replace(BASE, filters=(16, 32))) == ["filters"]


def test_track1_is_not_typed_by_hand():
    """locked1600.py ต้องอ้าง BASE (import) ไม่พิมพ์ค่า hyperparameter เอง"""
    src = (L.__file__ and open(L.__file__, encoding="utf-8").read())
    assert "from src.configs import BASE" in src and "TRACK1_CONFIG = BASE" in src
    assert L.TRACK1_CONFIG is BASE
