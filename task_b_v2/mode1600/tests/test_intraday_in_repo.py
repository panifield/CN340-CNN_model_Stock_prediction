"""
test_intraday_in_repo.py — ข้อมูลรายชั่วโมงของโหมด 16:00 ต้องโหลดได้จาก `task_b_v2/data/intraday/` (ใน repo) และ SHA ตรง
=====================================================================================================================
สิ่งที่ต้องจริงเสมอ (DECISIONS B41 · ไม่ดูค่า y ใด ๆ — ตรวจแค่ไฟล์ · SHA · ชื่อ snapshot)
    * ค่าเริ่มต้นของ data1600 ชี้มาที่ data/intraday/ (env MODE1600_TEAM_TASK_B เป็นทางเลือกเท่านั้น)
    * ไฟล์ล็อก 730d ทั้งสองหุ้น SHA ตรง README ทีม · snapshot ทุกไฟล์ SHA ตรง .meta.json
    * มีครบเฉพาะชุดที่ run1600 ใช้: ทาง (i) 20261023 03:55 + ทาง (ii) 10 วัน (ตาม LOCK_1600 หัวข้อ 1)
    * ทุกไฟล์ใน data/intraday/ มี SHA-256 บันทึกใน data/SOURCES.md · git ไม่แปลง line ending (`data/intraday/** -text`)
"""

import os
import subprocess

import pytest

from mode1600 import data1600 as D
from mode1600 import features1600 as FE

INTRADAY = D.TASK_DIR / "data" / "intraday"
SOURCES = (D.TASK_DIR / "data" / "SOURCES.md").read_text(encoding="utf-8")
SNAP_I = "20261023/{t}_BK_1h_035532.csv"
SNAP_II = ["20260930/{t}_BK_1h_162242.csv", "20261001/{t}_BK_1h_160222.csv", "20261002/{t}_BK_1h_160234.csv",
           "20261005/{t}_BK_1h_160223.csv", "20261012/{t}_BK_1h_160232.csv", "20261014/{t}_BK_1h_160257.csv",
           "20261015/{t}_BK_1h_160310.csv", "20261019/{t}_BK_1h_160311.csv", "20261020/{t}_BK_1h_160345.csv",
           "20261022/{t}_BK_1h_160230.csv"]

in_repo_default = pytest.mark.skipif("MODE1600_TEAM_TASK_B" in os.environ,
                                     reason="ตั้ง env MODE1600_TEAM_TASK_B ไว้ — test นี้ตรวจค่าเริ่มต้นเท่านั้น")


@in_repo_default
def test_default_paths_point_into_repo():
    assert D.TEAM_TASK_B.resolve() == INTRADAY.resolve()
    assert D.HIST_DIR.resolve() == (INTRADAY / "raw_data_intraday").resolve()
    assert D.LIVE_DIR.resolve() == (INTRADAY / "raw_data_intraday_live").resolve()


@in_repo_default
@pytest.mark.parametrize("ticker", D.TICKERS)
def test_history_loads_from_repo_with_team_sha(ticker):
    assert D.history_path(ticker).is_file()
    assert D.sha256_file(D.history_path(ticker)) == D.HIST_SHA[ticker]
    h = D.load_history(ticker)                       # ตรวจ SHA อีกรอบในตัวโหลด
    assert h["date"].min().strftime("%Y-%m-%d") == "2023-10-02" and h["date"].max().strftime("%Y-%m-%d") == "2026-09-25"


@in_repo_default
@pytest.mark.parametrize("ticker", D.TICKERS)
def test_snapshots_in_repo_are_exactly_the_ones_run1600_uses(ticker):
    idx = D.snapshot_index(ticker)
    names = sorted(p.parent.name + "/" + p.name for p in idx["path"])
    assert names == sorted([SNAP_I.format(t=ticker)] + [s.format(t=ticker) for s in SNAP_II])
    assert idx["sha_ok"].all() and idx["downloaded_at"].notna().all()
    for p in idx["path"]:
        D.load_snapshot(p)                           # ปฏิเสธถ้า SHA ไม่ตรง meta


@in_repo_default
def test_replay_way_ii_picks_ten_days_from_repo_copy():
    """กฎ (ii) บนสำเนาใน repo ต้องได้ 10/17 วัน และวัน 09-30 = ไฟล์ 16:22 (ตรวจแค่ชื่อไฟล์ ไม่ดูค่าใด ๆ)"""
    snaps = D.snapshot_index("KBANK")
    sb = {p: D.load_snapshot(p) for p in snaps["path"]}
    days = FE.replay_dates(D.load_daily("KBANK"))
    picks = [FE.select_snapshot_ii(snaps, d, sb) for d in days]
    assert len(days) == 17 and sum(p is not None for p in picks) == 10
    assert FE.select_snapshot_ii(snaps, "2026-09-30", sb)["path"].name == "KBANK_BK_1h_162242.csv"


def test_every_intraday_file_is_recorded_in_sources_md():
    files = [p for p in INTRADAY.rglob("*") if p.is_file()]
    assert len(files) == 46                          # 2 ไฟล์ล็อก + 2 หุ้น × 11 snapshot × (csv + meta)
    for p in files:
        assert D.sha256_file(p) in SOURCES, f"SHA ของ {p.relative_to(INTRADAY)} ไม่อยู่ใน data/SOURCES.md"


def test_git_keeps_intraday_bytes_as_is():
    out = subprocess.run(["git", "check-attr", "text", "--", "data/intraday/raw_data_intraday/KBANK_BK_1h_730d.csv"],
                         cwd=D.TASK_DIR, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("ไม่มี git")
    assert out.stdout.strip().endswith("text: unset")
