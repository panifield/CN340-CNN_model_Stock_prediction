"""
test_legacy_snapshot.py — legacy/ ต้องไม่ถูกแก้ · src/ ต้องไม่ตั้งชื่อชนกับ legacy
===============================================================================
ไฟล์นี้ทำอะไร: ตรวจ SHA-256 ของทุกไฟล์ใน legacy/ เทียบตารางใน legacy/SNAPSHOT.md
ทำไมต้องมี: legacy คือ "โมเดลเดิมที่ใช้เป็นตัวเทียบ" ถ้ามีใครแก้ไฟล์ ตัวเทียบจะเปลี่ยนโดยที่ไม่รู้ตัว
(ported from task_b/tests/test_legacy_snapshot.py, branch task_b — ปรับให้ตรวจเฉพาะไฟล์ที่เรา copy)
"""
import hashlib
import re

import pytest

from src import paths

LEGACY = paths.LEGACY_DIR
ROW = re.compile(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \|$", re.M)


def _table():
    return {n: (int(s), h) for n, s, h in ROW.findall((LEGACY / "SNAPSHOT.md").read_text(encoding="utf-8"))}


def _files_on_disk():
    return sorted(p for p in LEGACY.rglob("*") if p.is_file() and p.name != "SNAPSHOT.md" and "__pycache__" not in p.parts)


def test_every_copied_file_is_in_snapshot_table():
    t = _table()
    missing = [p.relative_to(LEGACY).as_posix() for p in _files_on_disk() if p.relative_to(LEGACY).as_posix() not in t]
    assert not missing, f"ไฟล์ใน legacy/ ที่ไม่มีใน SNAPSHOT.md: {missing}"


@pytest.mark.parametrize("path", _files_on_disk(), ids=lambda p: p.name)
def test_sha256_unchanged(path):
    size, sha = _table()[path.relative_to(LEGACY).as_posix()]
    raw = path.read_bytes()
    assert len(raw) == size, f"{path.name}: ขนาดเปลี่ยน"
    assert hashlib.sha256(raw).hexdigest() == sha, f"{path.name}: SHA-256 เปลี่ยน -- ห้ามแก้ legacy"


def test_no_bytecode_in_legacy():
    assert not list(LEGACY.rglob("__pycache__")), "มี __pycache__ ใน legacy (ตั้ง PYTHONDONTWRITEBYTECODE=1)"


def test_gitattributes_protects_legacy():
    ga = (paths.TASK_DIR / ".gitattributes").read_text(encoding="utf-8")
    assert "legacy/** -text" in ga


def test_src_modules_do_not_clash_with_legacy_names():
    """
    โมดูลใน src/ ถูก import เป็น `src.xxx` จึงไม่ชนกับ legacy ในทางเทคนิค
    แต่เทสต์นี้บังคับให้ "ชื่อซ้ำ" ต้องเป็นการตัดสินใจที่ตั้งใจ ไม่ใช่เผลอตั้งชน
    ชื่อที่อนุญาต: models / evaluate (SPEC ข้อ 10 กำหนดไว้) · baselines (ตัวเทียบตาม SPEC ข้อ 7 — ชื่อนี้ตรงความหมายที่สุด)
    """
    allowed = {"models", "evaluate", "baselines"}
    mine = {p.stem for p in (paths.TASK_DIR / "src").glob("*.py") if p.stem != "__init__"}
    clash = (mine & set(paths.legacy_module_names())) - allowed
    assert not clash, f"ชื่อโมดูลซ้ำกับ legacy: {clash}"


def test_legacy_feature_builder_imports_and_gives_29_columns():
    paths.use_legacy()
    import pandas as pd
    from features import build_features
    idx = pd.bdate_range("2020-01-01", periods=80)
    import numpy as np
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": rng.uniform(1e6, 2e6, len(idx))},
                      index=idx)
    f = build_features(df, verbose=False)
    assert f.shape[1] == 29, f"ต้องมี 29 feature (24 market + 5 dow) ได้ {f.shape[1]}"
