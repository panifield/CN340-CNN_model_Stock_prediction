# HANDOFF_1600 — ส่งต่องานโหมด 16:00 ให้ session หลัก

> เขียน 2026-10-24 โดย session ที่ทำโหมด 16:00 (ระยะที่ 1–2) · ผู้อ่าน: session หลัก (Stage 6 / next_day) และผู้ใช้
> ไฟล์นี้อยู่ใน `mode1600/` → **ต้อง commit ไปพร้อมโฟลเดอร์** (gate ต้องการให้ `mode1600/` สะอาด)
> **ปรับ 2026-10-24 (session หลัก · ผู้ใช้สั่ง · DECISIONS B41):** ลำดับ = 16:00 รันทันทีหลัง C6 ไม่ต้องรอ C7 · Claude รัน C6 หลังผู้ใช้แจ้ง · ข้อมูลรายชั่วโมงย้ายเข้า `task_b_v2/data/intraday/` (ค่าเริ่มต้นของ data1600 · env `MODE1600_TEAM_TASK_B` เป็นทางเลือก) · เพิ่มข้อจำกัดเรื่องสถิติรวมใน AUDIT (ข้อ 4)

---

## 1. สถานะ

**เสร็จแล้ว**
- ตรวจข้อมูล (`AUDIT.md`) · แผน (`PLAN_DRAFT.md` · ผู้ใช้อนุมัติ · M1–M23)
- โค้ด: `data1600.py` (โหลด + ตรวจ SHA) · `features1600.py` (4 feature · target · replay (i)/(ii) + provenance) ·
  `locked1600.py` (protocol · `TRACK1_CONFIG = BASE`) · `runner1600.py` (wrapper ไม่แก้ src) · `lock_gate.py` · `run1600.py` ·
  `draft_lock1600.py` · `time1600.py` · `audit1600.py`
- test 60/60 ผ่าน (รันล่าสุด 2026-10-24 ~01:50 · ก่อนเขียนไฟล์นี้)
- วัดเวลาแล้วด้วย y ที่สลับ (KBANK รอบใหญ่สุด 700 แถว): track หลัก 10 seeds 86.7 วินาที · control 13.9 วินาที · RF 0.4 วินาที
  → ประเมินการรันจริงทั้งหมด ~56 นาที CPU ≈ **15–25 นาที ที่ `--jobs 6`**
- `LOCK_1600.md` สร้างแล้วแต่ **ยังเป็นร่าง** (marker `NEXTDAY_LOCK_PENDING` เพราะ `task_b_v2/LOCK.md` ยังไม่ถูก commit) → gate ปฏิเสธ

**ยังค้าง**
1. ผู้ใช้ commit 2 ของ next_day (`task_b_v2/LOCK.md` ฯลฯ ตาม LOCK.md หัวข้อ 10 · **ไม่รวม `mode1600/`**)
2. รัน `draft_lock1600` ใหม่ → บันทึก commit hash ของ LOCK.md และ marker ร่างหายไป
3. รัน test ของ mode1600 อีกครั้ง
4. ผู้ใช้ commit `task_b_v2/mode1600/` ทั้งโฟลเดอร์ (**ก่อนเริ่ม C6 ของ next_day** — LOCK.md หัวข้อ 9 ข้อ (d))
5. หลัง C6 (final + replay + S1) ของ next_day เสร็จ — รันได้ทันที ไม่ต้องรอ C7 (LOCK.md B40) และผู้ใช้สั่ง: Claude รันจริง `run1600`

## 2. ขั้นตอนต่อไป (ทีละขั้น · คำสั่งจริง)

ทุกคำสั่งใช้ `PYTHONDONTWRITEBYTECODE=1` (กัน `__pycache__` โผล่ใน `src/` / `mode1600/` ซึ่งทำให้ gate ไม่ผ่าน)

| ขั้น | ใครทำ | คำสั่ง / การกระทำ | ผลที่ต้องเห็น |
|---|---|---|---|
| 1 | ผู้ใช้ | commit 2 ของ next_day (ตาม LOCK.md หัวข้อ 10 · ไม่รวม `mode1600/`) | `git log -1 -- task_b_v2/LOCK.md` มี hash |
| 2 | Claude | `cd task_b_v2 && PYTHONDONTWRITEBYTECODE=1 python -m mode1600.draft_lock1600` | `LOCK_1600.md` ไม่มี `NEXTDAY_LOCK_PENDING` และมีบรรทัด `NEXTDAY_LOCK_COMMIT: <hash>` (หัวข้อ 7) · ถ้า Track 1 ไม่ตรง LOCK.md สคริปต์จะ raise — **หยุดถามผู้ใช้** |
| 3 | Claude | จาก root repo: `PYTHONDONTWRITEBYTECODE=1 python -m pytest task_b_v2/mode1600/tests` | ผ่านทั้งหมด (`test_real_repo_gate_refuses_now` ยังปฏิเสธเพราะ LOCK_1600 ยังไม่ commit — ถูกต้อง) |
| 4 | ผู้ใช้ | `git add task_b_v2/mode1600` · `git commit -m "task_b_v2: mode 16:00 plan, code, tests and LOCK_1600 (track 1 = base, before any real run)"` | `git status task_b_v2/mode1600` ว่าง |
| 5 | Claude (หลังผู้ใช้แจ้ง) | ตรวจ gate อย่างเดียว: `cd task_b_v2 && PYTHONDONTWRITEBYTECODE=1 python -c "from mode1600 import lock_gate; print(lock_gate.check())"` | คืน dict (head_commit, lock_commit, nextday_lock_commit) ไม่ raise |
| 6 | Claude (หลังผู้ใช้ commit โค้ด C6 และแจ้ง) | รัน C6 ของ next_day (final + replay + S1) ให้จบก่อน — C7 ไม่ต้องรอ | — |
| 7 | Claude (หลังผู้ใช้สั่งชัด ๆ) | `cd task_b_v2 && PYTHONDONTWRITEBYTECODE=1 python -m mode1600.run1600 --jobs 6` | เขียน `task_b_v2/results/mode1600/` ครั้งเดียว (~15–25 นาที) |

## 3. gate ของ 16:00 (`lock_gate.check`) — ตรวจอะไร และถ้าปฏิเสธแปลว่าอะไร

`run1600.main` เรียก gate **เป็นคำสั่งแรก** แล้วจึงตรวจ `TRACK1_CONFIG` (ห้ามว่าง) และว่า `results/mode1600/` ยังไม่มีผล

| ข้อความที่เห็น | แปลว่า | ทำอย่างไร |
|---|---|---|
| `task_b_v2/src/ ไม่สะอาด` | มีไฟล์ใน src แก้ค้าง/ยังไม่ track (รวม `__pycache__` ถ้าไม่ถูก ignore) | ให้ผู้ใช้ commit หรือคืนค่า · อย่าแก้ src เพื่อให้ผ่าน |
| `task_b_v2/mode1600/ ไม่สะอาด` | มีไฟล์ใน mode1600 ที่ยังไม่ commit (เช่น HANDOFF นี้ · ผลชั่วคราว · pycache) | commit หรือลบไฟล์ชั่วคราว (ถามผู้ใช้ก่อนลบ) |
| `ไม่มีไฟล์ …LOCK_1600.md` / `ยังไม่ถูก commit` | ยังไม่ได้ทำขั้น 2–4 | ทำตามหัวข้อ 2 |
| `มีการแก้ที่ยังไม่ commit (ถูกแก้หลัง commit)` | LOCK_1600 ถูกแก้ใน working tree หลัง commit | คืนค่าจาก git · ห้ามแก้ LOCK หลัง commit |
| `ถูกแก้ใน commit หลัง commit แรก` | มีมากกว่า 1 commit ที่แตะ LOCK_1600 | **ถือเป็นรอบใหม่** ต้องให้ผู้ใช้ตัดสินและประกาศ — ไม่มีทางลัด |
| `ยังเป็นร่าง (ช่อง track 1 ว่าง)` | มี `TRACK1_PENDING` | `TRACK1_CONFIG` เป็น None ตอนสร้าง LOCK — ไม่ควรเกิดแล้ว (ตอนนี้ = BASE) |
| `ยังเป็นร่าง (ยังไม่มี commit hash ของ LOCK.md …)` | มี `NEXTDAY_LOCK_PENDING` | สร้าง LOCK_1600 ก่อน commit 2 → ต้องสร้างใหม่หลัง commit 2 (ระวังข้อ 6.1) |
| `(LOCK ของ next_day) ยังไม่ถูก commit` | `task_b_v2/LOCK.md` ไม่อยู่ใน git | ผู้ใช้ต้อง commit 2 |
| `LOCK.md … มีการแก้ที่ยังไม่ commit` | LOCK.md ของ next_day ถูกแก้ค้าง | คืนค่า / ถามผู้ใช้ |
| `LOCK.md ของ next_day ถูกแก้หลังจาก LOCK_1600 อ้าง` | commit ล่าสุดของ LOCK.md ≠ hash ใน LOCK_1600 | config ที่อ้างอาจเปลี่ยน → หยุดถามผู้ใช้ (ถ้าจะสร้าง LOCK_1600 ใหม่ = รอบใหม่) |
| `ไม่ได้บันทึก commit hash ของ LOCK.md` | LOCK_1600 ไม่มีบรรทัด `NEXTDAY_LOCK_COMMIT:` | LOCK ไม่ได้มาจาก `draft_lock1600` ฉบับปัจจุบัน |
| `RuntimeError: TRACK1_CONFIG ยังว่าง` | `locked1600.TRACK1_CONFIG` เป็น None | ไม่ควรเกิด |
| `FileExistsError … มีผลอยู่แล้ว` | รันจริงไปแล้ว | **ห้ามลบเพื่อรันซ้ำ** — รันครั้งเดียว (ถามผู้ใช้) |
| `ValueError: SHA-256 ของ … ไม่ตรง` (จาก data1600) | ไฟล์ข้อมูลเปลี่ยน (repo ทีม / data/) | หยุดถามผู้ใช้ — ห้ามแก้ค่า SHA ในโค้ด |

**ไม่ตรวจ SHA ของไฟล์โค้ด** (M19 ถูกยกเลิก → M21) · ผลทุกไฟล์มีคอลัมน์ `run_head_commit` + `run_time` และ `MANIFEST.json`

## 4. ข้อห้ามที่ยังมีผล

- ข้อจำกัดที่ต้องเปิดเผย: สถิติรวมของ Close_official − C15 (สัดส่วนที่เท่ากันพอดี · ค่าเฉลี่ยของส่วนต่าง) บนวัน ≤ 2025-02-25 ถูกคำนวณใน AUDIT.md (ข้อ 3) ซึ่งรวม 99 วันแรกของช่วงประเมิน walk-forward (2024-10-01 → 2025-02-25) — เป็นตัวเลขรวมเท่านั้น ไม่มีคำทำนาย · IC หรือตัววัดของโมเดลใด ๆ แต่ถือว่าผู้วางแผนเห็นลักษณะของ target ในช่วงนั้นแล้ว (B41)
- **ห้ามคำนวณ/ดู/รายงานคำทำนาย · IC · relMAE หรือสถิติใด ๆ ของ target บนข้อมูลจริงตั้งแต่ 2024-10-01** ก่อน LOCK_1600 ถูก commit
  (ช่วงประเมิน walk-forward เริ่มวันนั้น) · การวัดเวลาให้ใช้ `time1600.py` (y สลับ · ไม่มีตัววัด) เท่านั้น
- **ห้ามแก้ไฟล์ใน `mode1600/` หลัง commit** (ยกเว้นประกาศรอบใหม่โดยผู้ใช้) · ห้ามแก้ `LOCK_1600.md` ด้วยมือเลย — สร้างจาก `draft_lock1600.py` เท่านั้น
- **ห้ามรัน `run1600` จนกว่า** LOCK_1600 commit แล้ว + C6 (final + replay + S1) ของ next_day จบ + ผู้ใช้สั่ง (ไม่ต้องรอ C7 · LOCK.md B40)
- ไม่แก้ `src/` เพื่อโหมดนี้ (M16) · `runner1600` ไม่ใช้ `src.runner` และไม่เขียน `experiment_log.csv`
- ข้อมูล replay: **(i) = snapshot `raw_data_intraday_live/20261023/*_035532.csv` = ผลหลัก 17 วัน** ·
  **(ii) = snapshot ล่าสุดที่ดาวน์โหลด 16:00–16:30 ของวันนั้น = ผลเสริม 10 วัน** (09-30 → ไฟล์ 16:22 ตามกฎ ไม่เลือกรายวัน) ·
  target ของ (ii) ใช้ C15 ของ snapshot นั้น (มี `c15_final` / `y_final_base` เป็นตารางเสริม) · ทุกแถวติด `bar_source`
- **M20: 23 บล็อก** (บล็อกละ 20 วัน เศษรวมบล็อกท้าย → บล็อก 23 = 2026-07-30 → 2026-08-28 = 21 วัน)
- **M19 ถูกยกเลิก** → ใช้ M21 (git status ของ src/ + mode1600/ ต้องสะอาด · LOCK_1600 commit + ไม่ใช่ร่าง · บันทึก HEAD + เวลาในผล)
- ห้าม `git add/commit/push` เอง (ผู้ใช้ commit) · ห้ามแก้ repo ทีม `StockPrediction2/` (อ่านอย่างเดียว)

## 5. ไฟล์ที่ต้องอ่าน + การตัดสินใจ M1–M23

อ่านตามลำดับ: `PLAN_DRAFT.md` (แผนเต็ม + ตาราง M) → `LOCK_1600.md` (สิ่งที่ล็อก · สร้างจากโค้ด) → `AUDIT.md` (ผลตรวจข้อมูล) →
`locked1600.py` (ค่าทุกตัว) → `runner1600.py` / `features1600.py` (docstring อธิบาย causal) · ทีม: `StockPrediction2/task_b/PHASE1D_PLAN_R2.md`

| # | สรุป |
|---|---|
| M1 | label ของแท่ง Yahoo = เวลาเริ่มแท่ง (ยืนยันจาก snapshot) · แท่ง 15 = 15:00–16:00 · แท่ง 16 ห้ามใช้ |
| M2 | eligible day ตามทีม R2: แท่ง 10,11,12,14,15 ครบ + ราคาปิดรายวัน t และ t−1 · วันไม่ครบตัดทั้งวัน ห้าม ffill |
| M3 | ฐานของ r15/gap/range15 = Close_official(วันทำการก่อนหน้า) จากไฟล์รายวัน (ไม่อ่านแถว t) |
| M4 | range15 ไม่รวมแท่ง 13 |
| M5 | ตัวหาร vol15 = เฉลี่ย 20 eligible day ก่อนหน้า (V11+V12+V14+V15 · ไม่ใช้ V10, V13) |
| M6 | (→ M13) ข้อมูลแท่งของ replay |
| M7 | (→ M14) RF |
| M8 | (→ M15) track 6 รายชั่วโมง |
| M9 | (→ M16) การนำ track 1 ไปใช้ |
| M10 | ขอบบล็อก walk-forward จากปฏิทินไฟล์รายวัน ร่วมสองหุ้น |
| M11 | 09-30 ใน (ii) ใช้ไฟล์ 16:22 ตามกฎ "ล่าสุดก่อน 16:30" |
| M12 | ผู้ใช้อนุมัติแผน + M4, M5, M10 |
| M13 | replay: (i) snapshot 2026-10-23 03:55 ผลหลัก 17 วัน · (ii) แท่งที่รู้ ณ 16:00–16:30 ผลเสริม 10 วัน · ติดป้ายที่มาต่อวัน |
| M14 | RF = phase1c-tuned ของ next_day (400 ต้น · depth 4 · leaf 50 · max_features 0.5) |
| M15 | track 6 (spectrogram รายชั่วโมงแบบ A2) ไม่ทำ · เป็นงานต่อยอด |
| M16 | wrapper ไม่แก้ src · คง linear skip ทั้ง track หลัก (hybrid) และ control (hybrid_nocnn) |
| M17 | train รอบแรก 222 แถว: ยอมรับ + เขียนข้อจำกัด + รายงานแยก 2 ช่วง (→ 2025-02-25 / 2025-02-26 →) |
| M18 | ข้อจำกัดบังคับ: ราคาปิดเทรน = investing แต่ replay = Yahoo 1d · ราคาปิด = C15 ใน 40–44% ของวัน (Naive แข็งมาก) |
| M19 | gate แบบตรวจ SHA ของโค้ด — **ผู้ใช้ไม่เอา** (CRLF/LF บน Windows · ขวางการรวมกับ task อื่น) |
| M20 | บล็อกท้าย 1 วัน → รวมเข้าบล็อก 23 (23 บล็อก · บล็อกท้าย 21 วัน) · ตัดสินจากปฏิทินก่อนมีผล |
| M21 | gate ใหม่: src/ + mode1600/ สะอาด · LOCK_1600 commit + ไม่ถูกแก้ + ไม่ใช่ร่าง · บันทึก HEAD + เวลาลงทุกไฟล์ผล · คงตรวจ SHA ข้อมูล |
| M22 | track 1 = base (Stage 6 ไม่มีผู้ชนะ) · `TRACK1_CONFIG = BASE` import จาก src · test ตรวจทุก field ตรง LOCK.md หัวข้อ 5 |
| M23 | LOCK_1600 บันทึก commit hash ของ `task_b_v2/LOCK.md` จาก git (`NEXTDAY_LOCK_COMMIT`) · gate ตรวจซ้ำตอนรัน |

คาดการณ์ที่ประกาศ (F3 · ห้ามแก้หลังเห็นผล): **F3a** ΔIC(หลัก − control) บน 2024-10-01 → 2026-08-28 > 0 แยกหุ้น + 95% CI block bootstrap (20 วัน · seed 0 · 10,000 รอบ) ขอบล่าง > 0 ·
**F3b** relMAE ทุก track เทียบ Naive + DM

## 6. จุดเสี่ยงที่ session ใหม่อาจพลาด

1. **อย่าให้ผู้ใช้ commit LOCK_1600 ฉบับร่าง** (มี `NEXTDAY_LOCK_PENDING`) แล้วค่อยสร้างใหม่ commit ทับ — gate จะเห็น LOCK_1600 ถูกแตะ 2 commit
   และปฏิเสธถาวร ("ถูกแก้ใน commit หลัง commit แรก") · ลำดับต้องเป็น commit 2 → `draft_lock1600` → test → commit `mode1600/` ครั้งเดียว
2. **ไฟล์ใหม่ใน `mode1600/` ทำให้ gate ไม่ผ่าน** — รวม HANDOFF นี้ ไฟล์ scratch และ `__pycache__` (ถ้า .gitignore ไม่ครอบ) · รันทุกอย่างด้วย `PYTHONDONTWRITEBYTECODE=1`
   · pytest ของ mode1600 ใช้ `mode1600/pytest.ini` (มี `-p no:cacheprovider`) — **รันด้วย path `task_b_v2/mode1600/tests`** ไม่ใช่จาก pytest.ini ของ task_b_v2
3. **ห้ามแก้ `task_b_v2/LOCK.md` หลัง LOCK_1600 อ้าง hash** — แม้แก้คำเดียวแล้ว commit, gate จะปฏิเสธการรัน 16:00
4. **`src/` ต้องสะอาดตอนรัน 16:00** — โค้ด C6 ของ next_day ต้อง commit ก่อน · ถ้าเริ่มงาน C7 (เช่นแก้ `src/figures.py`) ให้รัน 16:00 ก่อนหรือ commit ก่อน
5. **ห้ามใช้ `src.runner` / `run_configs` กับโหมดนี้** (มันสร้าง y แบบ next_day และต่อท้าย `experiment_log.csv`)
6. **ห้ามรัน `time1600.py` ดัดแปลงด้วย y จริง** และห้ามเพิ่มการพิมพ์คำทำนาย/ตัววัดลงไป
7. `test_track1_lock.py` อ่านตาราง LOCK.md หัวข้อ 5 (คอลัมน์ที่ขึ้นต้น "Track 1" · ค่า tuple ใช้เครื่องหมาย `−` U+2212) — ถ้า `src/draft_lock.py` เปลี่ยนรูปแบบตาราง test จะล้ม (ไม่ใช่ว่า config ผิด)
8. ข้อมูลรายชั่วโมงอยู่ใน repo นี้ที่ `task_b_v2/data/intraday/` (สำเนา byte-for-byte จาก repo ทีม `CN340/StockPrediction2/task_b/` · env `MODE1600_TEAM_TASK_B` ชี้กลับไปที่ทีมได้) · ถ้าผู้ใช้ pull แล้วไฟล์ล็อกหรือ snapshot ที่ใช้ถูกแก้ SHA จะไม่ตรง → หยุดถาม
9. ช่วงประเมินของ 16:00 ทับกับ final ของ next_day → **LOCK_1600 ต้อง commit ก่อนเริ่ม C6** (LOCK.md หัวข้อ 9 ข้อ (d)) · ถ้าผู้ใช้เผลอรัน C6 ก่อน ให้หยุดและบอกผู้ใช้ทันที
10. replay ใน `run1600` เทรนใหม่ต่อทาง (i)/(ii) (โมเดลเหมือนกันเพราะ seed เดิม) และรันแบบ serial — เวลา replay ~7 นาทีไม่ขนาน
11. ผลการรันจริงเขียนครั้งเดียว — ถ้าล้มกลางทางก่อนเขียน (โฟลเดอร์ยังไม่ถูกสร้าง) รันใหม่ได้ · ถ้าเขียนบางส่วนแล้ว **ห้ามลบเอง** ถามผู้ใช้ และรายงานตามกติกา "พบบั๊ก → หยุดและรายงาน" (LOCK.md ข้อ (c))
12. ตัวเลขของโหมด 16:00 เทียบกับ next_day ตรง ๆ ไม่ได้ (คนละ target) · ต้องติดป้าย development / walk-forward เสมอ
