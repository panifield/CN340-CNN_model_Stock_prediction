"""
test_gate1600.py — gate ของการรันจริง (lock_gate.check + run1600.main) ตาม M21 (แทน M19)
=======================================================================================
ใช้ git repo ชั่วคราวใน tmp_path (นอกโปรเจกต์ · ไม่แตะ repo จริง) จำลองโครง task_b_v2/{src, mode1600, LOCK.md}:
    (ก) src/ หรือ mode1600/ ไม่สะอาด (แก้ค้าง / ไฟล์ใหม่ที่ยังไม่ track) → ปฏิเสธ
    (ข) LOCK_1600.md ยังไม่ commit · แก้ใน working tree · แก้ใน commit ถัดไป · ยังเป็นร่าง → ปฏิเสธ
    (+) LOCK.md ของ next_day ยังไม่ commit / ถูกแก้หลังจาก LOCK_1600 อ้าง hash → ปฏิเสธ
    ไม่ตรวจ SHA ของโค้ดแล้ว: แก้โค้ดแล้ว commit (สะอาด) → ผ่าน
และตรวจว่า run1600.main เรียก gate ก่อนทำอย่างอื่น (ไม่โหลดข้อมูล ไม่สร้างโฟลเดอร์ผล)
"""

import subprocess

import pytest

from mode1600 import lock_gate as G


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                        *args], check=True, capture_output=True, text=True)
    return r.stdout.strip()


def _commit_all(r, msg="c"):
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", msg)


class Repo:
    """repo จำลอง: tb/src/x.py · tb/mode1600/{a.py, LOCK_1600.md} · tb/LOCK.md"""

    def __init__(self, root):
        self.root = root
        self.tb = root / "tb"
        self.mode = self.tb / "mode1600"
        (self.tb / "src").mkdir(parents=True)
        self.mode.mkdir()
        _git(root, "init", "-q")
        (self.tb / "src" / "x.py").write_text("x = 1\n", encoding="utf-8")
        (self.mode / "a.py").write_text("a = 1\n", encoding="utf-8")
        (self.tb / "LOCK.md").write_text("# LOCK next_day\n", encoding="utf-8")

    def nextday_hash(self):
        return _git(self.root, "log", "-1", "--format=%H", "--", "tb/LOCK.md")

    def write_lock1600(self, body="config ครบ", nd_hash=None):
        nd = self.nextday_hash() if nd_hash is None else nd_hash
        (self.mode / "LOCK_1600.md").write_text(f"# LOCK_1600\n{body}\nNEXTDAY_LOCK_COMMIT: {nd}\n", encoding="utf-8")

    def ready(self):
        """ลำดับจริง: commit 2 (LOCK.md) → สร้าง LOCK_1600 ที่อ้าง hash → commit mode1600"""
        _commit_all(self.root, "commit 2")
        self.write_lock1600()
        _commit_all(self.root, "lock 1600")
        return self

    def check(self):
        return G.check(mode_dir=self.mode)


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path / "repo")


def test_passes_when_everything_committed_and_clean(repo):
    repo.ready()
    out = repo.check()
    head = _git(repo.root, "rev-parse", "HEAD")
    assert out["head_commit"] == head and out["lock_commit"] == head
    assert out["nextday_lock_commit"] == repo.nextday_hash()
    assert out["run_time"]


def test_refuses_when_lock1600_not_committed(repo):
    _commit_all(repo.root)
    repo.write_lock1600()
    with pytest.raises(G.LockGateError, match="ยังไม่ถูก commit"):
        repo.check()


def test_refuses_when_src_dirty(repo):
    repo.ready()
    (repo.tb / "src" / "x.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(G.LockGateError, match="src/ ไม่สะอาด"):
        repo.check()


def test_refuses_when_mode1600_has_untracked_file(repo):
    repo.ready()
    (repo.mode / "new.py").write_text("", encoding="utf-8")
    with pytest.raises(G.LockGateError, match="mode1600/ ไม่สะอาด"):
        repo.check()


def test_refuses_when_lock1600_edited_in_worktree(repo):
    repo.ready()
    repo.write_lock1600("แก้หลัง commit")
    with pytest.raises(G.LockGateError, match="ถูกแก้หลัง commit"):
        repo.check()


def test_refuses_when_lock1600_edited_in_later_commit(repo):
    repo.ready()
    repo.write_lock1600("แก้ใน commit ใหม่")
    _commit_all(repo.root, "edit lock")
    with pytest.raises(G.LockGateError, match="commit หลัง commit แรก"):
        repo.check()


@pytest.mark.parametrize("marker", [G.PENDING_MARKER, G.NEXTDAY_PENDING_MARKER])
def test_refuses_draft_lock1600(repo, marker):
    _commit_all(repo.root)
    repo.write_lock1600(f"<!-- {marker} -->")
    _commit_all(repo.root, "draft")
    with pytest.raises(G.LockGateError, match="ร่าง"):
        repo.check()


def test_refuses_when_nextday_lock_not_committed(tmp_path):
    r = Repo(tmp_path / "r2")
    (r.tb / "LOCK.md").unlink()
    _commit_all(r.root)
    (r.tb / "LOCK.md").write_text("# LOCK next_day\n", encoding="utf-8")
    r.write_lock1600(nd_hash="0" * 40)
    with pytest.raises(G.LockGateError, match="LOCK ของ next_day\\) ยังไม่ถูก commit"):
        r.check()


def test_refuses_when_nextday_lock_changed_after_reference(repo):
    repo.ready()
    (repo.tb / "LOCK.md").write_text("# LOCK next_day v2\n", encoding="utf-8")
    _commit_all(repo.root, "edit next_day lock")
    with pytest.raises(G.LockGateError, match="ถูกแก้หลังจาก LOCK_1600 อ้าง"):
        repo.check()


def test_code_edit_committed_passes_no_code_sha_check(repo):
    """M19 ถูกยกเลิก: แก้โค้ดแล้ว commit (สะอาด) → gate ผ่าน · CRLF/LF ไม่มีผล"""
    repo.ready()
    (repo.mode / "a.py").write_bytes(b"a = 2\r\n")
    (repo.tb / "src" / "x.py").write_text("x = 3\n", encoding="utf-8")
    _commit_all(repo.root, "edit code")
    assert repo.check()["head_commit"] == _git(repo.root, "rev-parse", "HEAD")


def test_real_repo_gate_refuses_now():
    """repo จริงตอนนี้: LOCK_1600.md ยังไม่ถูก commit → ต้องปฏิเสธ (ข้ามถ้าผู้ใช้ commit แล้ว)"""
    if G.last_commit_of(G.MODE_DIR / G.LOCK_NAME) is not None:
        pytest.skip("LOCK_1600.md ถูก commit แล้ว")
    with pytest.raises(G.LockGateError):
        G.check()


def test_run1600_calls_gate_first(monkeypatch):
    """run1600.main ต้องหยุดที่ gate ก่อนโหลดข้อมูล/สร้างโฟลเดอร์ผล"""
    from mode1600 import run1600

    def boom(*a, **k):
        raise G.LockGateError("gate ไม่ผ่าน (จำลอง)")

    def must_not_run(*a, **k):
        raise AssertionError("ห้ามโหลดข้อมูลก่อนผ่าน gate")

    existed = run1600.RESULTS_DIR.exists()
    monkeypatch.setattr(run1600.lock_gate, "check", boom)
    monkeypatch.setattr(run1600, "load_inputs", must_not_run)
    with pytest.raises(G.LockGateError):
        run1600.main([])
    assert run1600.RESULTS_DIR.exists() == existed


def test_run1600_refuses_empty_track1_even_if_gate_passes(monkeypatch):
    from mode1600 import run1600
    monkeypatch.setattr(run1600.lock_gate, "check", lambda *a, **k: {"head_commit": "x", "run_time": "t"})
    monkeypatch.setattr(run1600.L, "TRACK1_CONFIG", None)
    monkeypatch.setattr(run1600, "load_inputs", lambda: (_ for _ in ()).throw(AssertionError("ห้ามโหลดข้อมูล")))
    with pytest.raises(RuntimeError, match="TRACK1_CONFIG"):
        run1600.main([])


def test_real_draft_lock_state():
    """LOCK_1600.md ที่สร้างไว้: track 1 เติมแล้ว (ไม่มี TRACK1_PENDING) · ถ้า LOCK.md ของ next_day ยังไม่ commit ต้องมี NEXTDAY_LOCK_PENDING"""
    lock = G.MODE_DIR / G.LOCK_NAME
    if not lock.exists():
        pytest.skip("ยังไม่ได้สร้าง LOCK_1600.md")
    text = lock.read_text(encoding="utf-8")
    assert G.PENDING_MARKER not in text
    nd = G.last_commit_of(G.TASK_DIR / "LOCK.md")
    if nd is None:
        assert G.NEXTDAY_PENDING_MARKER in text
    else:
        assert G.NEXTDAY_HASH_RE.search(text).group(1) == nd, "สร้าง LOCK_1600 ใหม่หลัง commit 2"
