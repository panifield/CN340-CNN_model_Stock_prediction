"""
paths.py — ที่อยู่ของโฟลเดอร์ต่าง ๆ + เปิดทางให้ import โค้ด legacy (อ่านอย่างเดียว)
=============================================================================
ไฟล์นี้ทำอะไร
    1) บอกว่า data / legacy / results อยู่ที่ไหน (ที่เดียว ไม่ hardcode ที่อื่น)
    2) use_legacy() เพิ่ม task_b_v2/legacy เข้า sys.path เพื่อให้ `import features` / `import models` ของ legacy ใช้ได้
       (ported idea from task_b/phase2/paths.py, branch task_b)
    3) assert_output_dir() กันเขียนไฟล์นอก task_b_v2/results/ (SPEC ข้อ 2.1: เขียนได้เฉพาะใน task_b_v2/)

ทำไมต้องมี
    legacy เป็น snapshot ที่ห้ามแก้ (SHA-256 ใน legacy/SNAPSHOT.md) และหา path จาก Path(__file__) ของตัวเอง
    เราจึงใช้เป็นไลบรารีผ่าน sys.path เท่านั้น · ตั้ง sys.dont_write_bytecode กัน __pycache__ โผล่ใน legacy
"""

import sys
from pathlib import Path

# ต้องตั้งก่อน import โมดูล legacy ตัวแรก (ไม่งั้นจะมี __pycache__ ใน legacy ซึ่งผิดกติกา snapshot)
sys.dont_write_bytecode = True

# legacy print ภาษาไทย -- stdout แบบ pipe บน Windows เป็น cp1252 จะ UnicodeEncodeError
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

TASK_DIR = Path(__file__).resolve().parent.parent      # task_b_v2/
DATA_DIR = TASK_DIR / "data"
LEGACY_DIR = TASK_DIR / "legacy"
RESULTS_DIR = TASK_DIR / "results"


def use_legacy():
    """เพิ่ม legacy/ เข้า sys.path (ครั้งเดียว) แล้วคืน LEGACY_DIR · เรียกก่อน `import features` ฯลฯ"""
    p = str(LEGACY_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return LEGACY_DIR


def legacy_module_names():
    """ชื่อโมดูลระดับบนสุดของ legacy -- ไว้ test ว่า src/*.py ไม่ตั้งชื่อชน"""
    return sorted(p.stem for p in LEGACY_DIR.glob("*.py"))


def assert_output_dir(path):
    """เขียนผลได้เฉพาะใต้ task_b_v2/results/ (ห้ามแตะ legacy/ data/ หรือ A/ A2/ ที่ root)"""
    p = Path(path).resolve()
    if p != RESULTS_DIR.resolve() and RESULTS_DIR.resolve() not in p.parents:
        raise PermissionError(f"ห้ามเขียนนอก task_b_v2/results/ : {p}")
    return p
