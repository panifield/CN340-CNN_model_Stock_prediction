"""
lock_gate.py — gate ของการรันจริงโหมด 16:00 (M21 · แทน M19 ที่ผู้ใช้ไม่เอา)
=====================================================================
ไฟล์นี้ทำอะไร: `check(...)` ปฏิเสธ (raise LockGateError) ถ้า
    (ก) `git status` ของ `task_b_v2/src/` หรือ `task_b_v2/mode1600/` ไม่สะอาด ณ ตอนรัน (รวมไฟล์ที่ยังไม่ track)
    (ข) `mode1600/LOCK_1600.md` ยังไม่ถูก commit · ถูกแก้หลัง commit (working tree / staged / commit ถัดไปที่แตะไฟล์นี้) ·
        หรือยังเป็นร่าง (มี marker `TRACK1_PENDING` หรือ `NEXTDAY_LOCK_PENDING`)
    (+) `task_b_v2/LOCK.md` ของ next_day ต้องถูก commit · สะอาด · และ commit ล่าสุดของไฟล์ = hash ที่บันทึกใน LOCK_1600.md (M23)
    คืน HEAD commit + เวลา เพื่อบันทึกลงไฟล์ผลทุกไฟล์ (ค)
    SHA-256 ของ **ไฟล์ข้อมูล** ยังตรวจทุกครั้งที่โหลด (data1600.py) · **ไม่ตรวจ SHA ของไฟล์โค้ด** (ผู้ใช้ไม่เอา M19:
    เปราะเรื่อง CRLF/LF บน Windows และขวางการรวมกับ task อื่น)
ทำไมต้องมี: ประกาศก่อนเห็นผล — ถ้ารันได้โดยไม่ commit LOCK จะพิสูจน์ไม่ได้ว่าแผน/โค้ดไม่ถูกแก้หลังเห็นผล

ใช้ git แบบอ่านอย่างเดียว (rev-parse / ls-files / status / log) · ไม่ add / commit อะไรเลย
"""

import re
import subprocess
from pathlib import Path

import pandas as pd

MODE_DIR = Path(__file__).resolve().parent
TASK_DIR = MODE_DIR.parent
LOCK_NAME = "LOCK_1600.md"
PENDING_MARKER = "TRACK1_PENDING"
NEXTDAY_PENDING_MARKER = "NEXTDAY_LOCK_PENDING"
# marker ของร่าง = comment บรรทัดตรงตัวเท่านั้น (รูปแบบที่ draft_lock1600 เขียนจริง) — ไม่ใช่คำเปล่าที่ไหนก็ได้ในไฟล์ (DECISIONS B42:
# แถว M23 ในหัวข้อ 5 มีคำ NEXTDAY_LOCK_PENDING อยู่ในคำอธิบาย ทำให้ LOCK ที่ไม่ใช่ร่างถูกปฏิเสธผิด ๆ)
def has_marker(text, marker):
    return f"<!-- {marker} -->" in text


NEXTDAY_HASH_RE = re.compile(r"NEXTDAY_LOCK_COMMIT:\s*([0-9a-f]{40})")


class LockGateError(RuntimeError):
    pass


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout.strip()


def repo_root(start=MODE_DIR):
    code, out = _git(start, "rev-parse", "--show-toplevel")
    if code != 0:
        raise LockGateError(f"{start} ไม่อยู่ใน git repo")
    return Path(out)


def last_commit_of(path, root=None):
    """commit ล่าสุดที่แตะไฟล์ `path` (None ถ้ายังไม่ถูก commit) — ใช้ทั้งใน gate และ draft_lock1600"""
    path = Path(path).resolve()
    root = repo_root(path.parent) if root is None else root
    rel = path.relative_to(Path(root).resolve()).as_posix()
    if _git(root, "ls-files", "--error-unmatch", "--", rel)[0] != 0:
        return None
    out = _git(root, "log", "-1", "--format=%H", "--", rel)[1]
    return out or None


def check(mode_dir=MODE_DIR, lock_name=LOCK_NAME, clean_dirs=None, nextday_lock=None):
    """
    ผ่าน → dict (head_commit, run_time, lock_commit, nextday_lock_commit) · ไม่ผ่าน → LockGateError พร้อมรายการปัญหาทั้งหมด
    clean_dirs   : โฟลเดอร์ที่ต้องสะอาด (ค่าเริ่มต้น task_b_v2/src และ mode1600)
    nextday_lock : LOCK.md ของ next_day (ค่าเริ่มต้น task_b_v2/LOCK.md)
    """
    mode_dir = Path(mode_dir).resolve()
    root = repo_root(mode_dir).resolve()
    clean_dirs = [mode_dir.parent / "src", mode_dir] if clean_dirs is None else [Path(d).resolve() for d in clean_dirs]
    nextday_lock = (mode_dir.parent / "LOCK.md") if nextday_lock is None else Path(nextday_lock)
    rel = lambda p: Path(p).resolve().relative_to(root).as_posix()   # noqa: E731
    problems = []

    # (ก) โค้ดต้องสะอาด ณ ตอนรัน
    for d in clean_dirs:
        st = _git(root, "status", "--porcelain", "--untracked-files=all", "--", rel(d))[1]
        if st:
            problems.append(f"{rel(d)}/ ไม่สะอาด:\n      " + "\n      ".join(st.splitlines()[:10]))

    # (ข) LOCK_1600.md
    lock = mode_dir / lock_name
    lock_rel = rel(lock)
    text = ""
    lock_commits = []
    if not lock.exists():
        problems.append(f"ไม่มีไฟล์ {lock_rel}")
    elif _git(root, "ls-files", "--error-unmatch", "--", lock_rel)[0] != 0:
        problems.append(f"{lock_rel} ยังไม่ถูก commit")
    else:
        text = lock.read_text(encoding="utf-8")
        if _git(root, "status", "--porcelain", "--", lock_rel)[1]:
            problems.append(f"{lock_rel} มีการแก้ที่ยังไม่ commit (ถูกแก้หลัง commit)")
        lock_commits = _git(root, "log", "--format=%H", "--", lock_rel)[1].split()
        if len(lock_commits) > 1:
            problems.append(f"{lock_rel} ถูกแก้ใน commit หลัง commit แรก ({len(lock_commits)} commit แตะไฟล์นี้) — ต้องเป็นรอบใหม่ที่ประกาศชัด")
        if has_marker(text, PENDING_MARKER):
            problems.append(f"{lock_rel} ยังเป็นร่าง (ช่อง track 1 ว่าง)")
        if has_marker(text, NEXTDAY_PENDING_MARKER):
            problems.append(f"{lock_rel} ยังเป็นร่าง (ยังไม่มี commit hash ของ LOCK.md ของ next_day — สร้างใหม่หลัง commit 2)")

    # (+) LOCK.md ของ next_day ต้อง commit แล้วและตรงกับที่ LOCK_1600 อ้าง
    nd_commit = last_commit_of(nextday_lock, root) if nextday_lock.exists() else None
    if nd_commit is None:
        problems.append(f"{rel(nextday_lock) if nextday_lock.exists() else nextday_lock} (LOCK ของ next_day) ยังไม่ถูก commit")
    else:
        if _git(root, "status", "--porcelain", "--", rel(nextday_lock))[1]:
            problems.append(f"{rel(nextday_lock)} มีการแก้ที่ยังไม่ commit")
        m = NEXTDAY_HASH_RE.search(text)
        if text and not m:
            problems.append(f"{lock_rel} ไม่ได้บันทึก commit hash ของ LOCK.md ของ next_day")
        elif m and m.group(1) != nd_commit:
            problems.append(f"LOCK.md ของ next_day ถูกแก้หลังจาก LOCK_1600 อ้าง ({m.group(1)[:10]} → {nd_commit[:10]})")

    if problems:
        raise LockGateError("gate ไม่ผ่าน — ห้ามรันจริง:\n  - " + "\n  - ".join(problems))
    return {"head_commit": _git(root, "rev-parse", "HEAD")[1],
            "run_time": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(timespec="seconds"),
            "lock_rel": lock_rel, "lock_commit": lock_commits[-1], "nextday_lock_commit": nd_commit}
