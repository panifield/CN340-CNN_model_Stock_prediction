"""
draft_lock1600.py — สร้าง mode1600/LOCK_1600.md (ร่าง) จากของจริงในโค้ด/ข้อมูล แบบเดียวกับ src/draft_lock.py ของ next_day
====================================================================================================================
ไฟล์นี้ทำอะไร: ประกอบ LOCK_1600.md จาก
    ข้อมูล + SHA (data1600) · วัน replay และ snapshot ที่กฎเลือก (features1600) · จำนวนแถว (นับเท่านั้น ไม่ดูค่า y) ·
    protocol + track + config (locked1600 · ช่อง track 1 ว่างจนกว่า Stage 6 จบ) · RF (legacy config) · F3 ·
    การตัดสินใจหัวข้อ M และข้อจำกัด (คัดจาก PLAN_DRAFT.md) · SHA-256 ของโค้ด · ขั้นตอนรัน · ช่อง commit
ทำไมต้องมี: LOCK ต้องตรงกับโค้ดและข้อมูลจริงทุกตัวอักษร — สร้างจากโค้ดจึงไม่มีการพิมพ์ตกหล่น · **ห้ามแก้ LOCK_1600.md ด้วยมือ**
    ถ้าช่อง track 1 ยังว่าง ไฟล์จะมี marker `TRACK1_PENDING` ซึ่ง gate (lock_gate.py) ปฏิเสธเสมอ

รัน (จาก task_b_v2/):  PYTHONDONTWRITEBYTECODE=1 python -m mode1600.draft_lock1600
"""

import sys

import pandas as pd

from mode1600 import data1600 as D
from mode1600 import features1600 as FE
from mode1600 import lock_gate as G
from mode1600 import locked1600 as L
from mode1600 import runner1600 as R
from src import paths

paths.use_legacy()
import config as legacy_config   # noqa: E402  (legacy · RF_PARAMS, CONFIG_TAG)

OUT = D.MODE_DIR / "LOCK_1600.md"
PLAN = D.MODE_DIR / "PLAN_DRAFT.md"
CODE_FILES = ("__init__.py", "data1600.py", "features1600.py", "locked1600.py", "runner1600.py", "run1600.py",
              "lock_gate.py", "draft_lock1600.py", "audit1600.py", "time1600.py", "pytest.ini")
TEST_FILES = ("tests/conftest.py", "tests/synth1600.py", "tests/test_data1600.py", "tests/test_features1600.py",
              "tests/test_runner1600.py", "tests/test_gate1600.py", "tests/test_track1_lock.py", "tests/test_intraday_in_repo.py")
DOC_FILES = ("PLAN_DRAFT.md", "AUDIT.md", "HANDOFF_1600.md")
NEXTDAY_LOCK_PATH = D.TASK_DIR / L.NEXTDAY_LOCK


def config_cells(cfg):
    """ค่าของ Config ในรูปแบบเดียวกับตาราง LOCK.md หัวข้อ 5 (src/draft_lock.py: tuple → "8−16")"""
    from dataclasses import asdict
    return {k: ("−".join(map(str, v)) if isinstance(v, tuple) else str(v)) for k, v in asdict(cfg).items() if k != "name"}


def nextday_track1(lock_text):
    """อ่านคอลัมน์ Track 1 ของตาราง LOCK.md หัวข้อ 5 → {field: ค่าแบบข้อความ}"""
    lines = lock_text.splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith("## 5."))
    hdr = next(k for k in range(i, len(lines)) if lines[k].startswith("| field |"))
    cols = [c.strip() for c in lines[hdr].strip("|").split("|")]
    j = next(n for n, c in enumerate(cols) if c.startswith("Track 1"))
    out = {}
    for s in lines[hdr + 2:]:
        if not s.startswith("|"):
            break
        cells = [c.strip().strip("`") for c in s.strip().strip("|").split("|")]
        out[cells[0]] = cells[j]
    return out


def check_track1_matches_nextday_lock(cfg=None, lock_text=None):
    """TRACK1_CONFIG ต้องตรงกับ Track 1 ของ LOCK.md หัวข้อ 5 ทุก field — คืนรายการที่ไม่ตรง (ว่าง = ตรง)"""
    cfg = L.TRACK1_CONFIG if cfg is None else cfg
    lock_text = NEXTDAY_LOCK_PATH.read_text(encoding="utf-8") if lock_text is None else lock_text
    want, got = nextday_track1(lock_text), config_cells(cfg)
    return sorted(set(want) ^ set(got)) + [k for k in want if k in got and want[k] != got[k]]


def _plan_section(start, stop_prefix="## "):
    lines = PLAN.read_text(encoding="utf-8").splitlines()
    i = next(k for k, s in enumerate(lines) if s.startswith(start))
    out = []
    for s in lines[i + 1:]:
        if s.startswith(stop_prefix):
            break
        out.append(s)
    return [s for s in out if s.strip()]


def _counts(tables, blocks):
    """จำนวนแถว usable ที่มีวันในไฟล์รายวัน (= มี target ได้) ต่อช่วง · นับเท่านั้น"""
    periods = [("ประเมิน WF ทั้งหมด", L.WF_START, L.WF_END), *[(f"ช่วงย่อย {a} → {b}", a, b) for a, b in L.SUBPERIODS],
               ("train ก่อนบล็อกแรก", "1900-01-01", (pd.Timestamp(L.WF_START) - pd.Timedelta(days=1)).date().isoformat()),
               ("train ของ replay (≤ 2026-09-25)", "1900-01-01", L.REPLAY_TRAIN_END)]
    rows = []
    for name, a, b in periods:
        r = {"ช่วง": name}
        for t, (tab, daily) in tables.items():
            m = tab["usable"] & tab.index.isin(daily.index) & (tab.index >= pd.Timestamp(a)) & (tab.index <= pd.Timestamp(b))
            r[t] = int(m.sum())
        rows.append(r)
    brow = []
    for b in blocks.itertuples():
        r = {"บล็อก": b.block, "วันแรก": str(b.start.date()), "วันสุดท้าย": str(b.end.date())}
        for t, (tab, daily) in tables.items():
            m = tab["usable"] & tab.index.isin(daily.index) & (tab.index >= b.start) & (tab.index <= b.end)
            r[f"{t} วันประเมิน"] = int(m.sum())
            r[f"{t} train (ก่อนบล็อก)"] = int((tab["usable"] & tab.index.isin(daily.index) & (tab.index < b.start)).sum())
        brow.append(r)
    return pd.DataFrame(rows), pd.DataFrame(brow)


def _md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return out


def render():
    tables, replay_rows, snap_rows = {}, [], []
    for t in L.TICKERS:
        hist, daily, snaps = D.load_history(t), D.load_daily(t), D.snapshot_index(t)
        assert snaps["sha_ok"].all()
        tables[t] = (FE.feature_table(hist, daily), daily)
        sb = {p: D.load_snapshot(p) for p in snaps["path"]}
        for day in FE.replay_dates(daily):
            r = FE.select_snapshot_ii(snaps, day, sb)
            replay_rows.append({"หุ้น": t, "วัน": str(day.date()),
                                "(ii) snapshot": f"`{r['path'].parent.name}/{r['path'].name}`" if r is not None else "ไม่มี",
                                "ดาวน์โหลด": r["downloaded_at"].strftime("%H:%M") if r is not None else "—",
                                "SHA-256 (16 ตัวแรก)": f"`{r['meta_sha'][:16]}`" if r is not None else "—"})
        fin = snaps[snaps["path"].map(lambda p: f"{p.parent.name}/{p.name}") == L.REPLAY_FINAL_SNAPSHOT.format(ticker=t)].iloc[0]
        snap_rows.append(f"| `task_b_v2/data/intraday/raw_data_intraday_live/{L.REPLAY_FINAL_SNAPSHOT.format(ticker=t)}` (replay ทาง i) | "
                         f"{fin['downloaded_at']} | `{fin['meta_sha']}` |")
    blocks = R.wf_blocks(tables[L.TICKERS[0]][1].index)
    counts, bcounts = _counts(tables, blocks)
    rep = pd.DataFrame(replay_rows)
    rep_k = rep[rep["หุ้น"] == "KBANK"].drop(columns="หุ้น").reset_index(drop=True)
    same_ii = (rep[rep["หุ้น"] == "KBANK"]["ดาวน์โหลด"].to_numpy() == rep[rep["หุ้น"] == "ADVANC"]["ดาวน์โหลด"].to_numpy()).all()
    n_ii = int((rep_k["(ii) snapshot"] != "ไม่มี").sum())

    pending = L.TRACK1_CONFIG is None
    if not pending:
        bad = check_track1_matches_nextday_lock()
        if bad:
            raise ValueError(f"TRACK1_CONFIG ไม่ตรงกับ Track 1 ใน LOCK.md หัวข้อ 5: {bad} — หยุดถามผู้ใช้")
    nd_commit = G.last_commit_of(NEXTDAY_LOCK_PATH) if NEXTDAY_LOCK_PATH.exists() else None
    md = ["# LOCK_1600 — โหมด 16:00 ของ task_b_v2 (ล็อกก่อนรันจริง)", ""]
    if pending:
        md += [f"<!-- {G.PENDING_MARKER} -->",
               "> **สถานะ: ร่าง — ช่อง config track 1 ยังว่าง** · gate ปฏิเสธไฟล์นี้จนกว่าจะสร้างใหม่หลังเติม `locked1600.TRACK1_CONFIG`"]
    if nd_commit is None:
        md += [f"<!-- {G.NEXTDAY_PENDING_MARKER} -->",
               "> **สถานะ: ร่าง — `task_b_v2/LOCK.md` ของ next_day ยังไม่ถูก commit** → หลังผู้ใช้ commit 2 ให้รัน"
               " `python -m mode1600.draft_lock1600` ใหม่ (จะบันทึก commit hash ของ LOCK.md ลงหัวข้อ 7 และ marker นี้จะหายไป) · gate ปฏิเสธไฟล์นี้จนกว่าจะเป็นเช่นนั้น"]
    if not pending and nd_commit is not None:
        md += ["> **สถานะ: พร้อม commit** — track 1 = config ที่ล็อกใน LOCK.md ของ next_day (หัวข้อ 3, 7)"]
    md += [f"> สร้างโดย `python -m mode1600.draft_lock1600` เมื่อ {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · **ห้ามแก้ไฟล์นี้ด้วยมือ** (แก้ที่โค้ดแล้วสร้างใหม่)",
           "> ทุกตัวเลขของโหมดนี้เป็น **development / walk-forward** — ไม่ใช่ test · ยังไม่มีการ fit โมเดลใดบนข้อมูลจริงของโหมดนี้",
           "> หลัง commit ไฟล์นี้ ห้ามแก้ config/feature/protocol เพราะเห็นผล — การแก้ = รอบใหม่ที่ประกาศชัด (gate ปฏิเสธ LOCK ที่ถูกแก้หลัง commit)", "",
           "## 1. ข้อมูล (ตรวจ SHA-256 ทุกครั้งที่โหลด · `data1600.py`)", "",
           "| ไฟล์ | หมายเหตุ | SHA-256 |", "|---|---|---|"]
    for t in L.TICKERS:
        md.append(f"| `task_b_v2/data/intraday/raw_data_intraday/{t}_BK_1h_730d.csv` (สำเนา byte-for-byte จาก repo ทีม · SHA ตรง README ทีม) | แท่ง 1h Yahoo 2023-10-02 → 2026-09-25 | `{D.HIST_SHA[t]}` |")
    for t in L.TICKERS:
        md.append(f"| `task_b_v2/data/{t}_10Y_Cleaned.csv` | ราคาปิดรายวัน (ไฟล์เดียวกับ next_day · LOCK.md) | `{D.DAILY_SHA[t]}` |")
    md += snap_rows
    md += ["", f"- replay ทาง (ii): snapshot ตามกฎ \"ล่าสุดที่ดาวน์โหลดในช่วง {L.SNAPSHOT_WINDOW[0]}–{L.SNAPSHOT_WINDOW[1]} ของวันนั้น"
           f" และมีแท่งบังคับครบ\" (M13) → **{n_ii}/{len(rep_k)} วัน** · ADVANC {'ได้ไฟล์เวลาเดียวกันทุกวัน' if same_ii else 'ต่างจาก KBANK บางวัน — ดูตาราง'} (ตาราง = KBANK):", ""]
    md += _md(rep_k)
    md += ["", "- แหล่งราคาปิดรายวัน: ≤ 2026-09-25 = investing.com · replay 2026-09-28 → 2026-10-22 = Yahoo 1d (AUDIT ข้อ 3)",
           "- ไฟล์รายชั่วโมงทั้งหมดอยู่ใน repo นี้ที่ `task_b_v2/data/intraday/` (เฉพาะไฟล์ที่ run1600 ใช้ · ที่มา + SHA ทุกไฟล์ใน `data/SOURCES.md` · B41)"
           " · env `MODE1600_TEAM_TASK_B` ชี้ไป clone ของทีมได้เป็นทางเลือก", "",
           "## 2. Protocol (PLAN_DRAFT ข้อ b, c · `locked1600.py`)", "",
           "- target: `y = Close_official(t) / C15(t) − 1` · ราคาที่ทำนาย `C15 · (1 + ŷ)` · Naive: ŷ = 0",
           f"- feature: {', '.join(f'`{f}`' for f in L.FEATURES)} (นิยามใน `features1600.py` · แท่ง range = {L.RANGE_HOURS} · volume = {L.VOL_HOURS} · ตัวหาร vol = {L.VOL_LOOKBACK} eligible day ก่อนหน้า)",
           f"- walk-forward: ประเมิน **{L.WF_START} → {L.WF_END}** · เทรนใหม่ทุก **{L.RETRAIN_EVERY}** วันทำการ (ขอบบล็อกจากปฏิทินรายวัน ร่วมสองหุ้น) → **{len(blocks)} บล็อก**",
           f"  · train = ทุกแถวที่ใช้ได้ก่อนวันแรกของบล็อก · early stop + เลือก c = {int(L.EARLY_STOP_FRAC * 100)}% ท้ายของ train · scaler/สเกลภาพ/normalize จาก 85% แรก",
           f"  · seeds {L.SEEDS[0]}–{L.SEEDS[-1]} ({len(L.SEEDS)} ตัว) เฉลี่ยคำทำนาย · รายงานแยก {' และ '.join(f'{a} → {b}' for a, b in L.SUBPERIODS)}",
           f"- replay: เทรนครั้งเดียวด้วยแถว ≤ **{L.REPLAY_TRAIN_END}** · ทำนาย {L.REPLAY_START} → {L.REPLAY_END} ทาง (i) ผลหลัก 17 วัน + ทาง (ii) ผลเสริม · ตาราง log ติดป้าย `bar_source` ต่อวัน",
           "- ตัววัด: relMAE (return, บาท) · MAE/RMSE บาท · IC Pearson + Spearman (คงที่ → 0 · B39) · DirAcc · StdRatio (ธง) · DM vs Naive · c ∈ {0, 0.1, …, 1} (รายงานก่อน/หลังปรับ)",
           "", "จำนวนแถวที่ใช้ได้ (นับเท่านั้น · ไม่ดูค่า y):", ""]
    md += _md(counts)
    md += ["", "<details><summary>ตารางบล็อก walk-forward</summary>", ""] + _md(bcounts) + ["", "</details>", "",
           "## 3. Track และ config (M14–M16)", ""]
    rf = dict(legacy_config.RF_PARAMS)
    t1 = ("**[ช่องว่าง — config track 1 ที่ล็อกจาก next_day (รอ Stage 6)]**" if pending else
          "config track 1 ที่ล็อกจาก next_day = **`src.configs.BASE`** (Stage 6 ไม่มีผู้ชนะ · ตารางล่าง)")
    md += ["| track | input | โมเดล |", "|---|---|---|",
           f"| 1 หลัก `{L.MAIN}` | spectrogram รายวัน (ถึง t−1) + 4 feature | {t1} + `ann_input = \"hybrid\"` (CNN feature + 4 feature + linear skip) |",
           f"| 2 control `{L.CONTROL}` | 4 feature | ANN เดียวกับ track 1 + `ann_input = \"hybrid_nocnn\"` (ไม่มี CNN · มี linear skip) |",
           f"| 3 `{L.RF}` | 4 feature | `{legacy_config.CONFIG_TAG}` RF_PARAMS `{rf}` · y standardize · train ทุกแถวก่อนบล็อก |",
           f"| 4 `{L.NAIVE}` | — | ŷ = 0 (ราคาปิด = C15) |",
           f"| 5 `{L.MEAN}` | — | ค่าเฉลี่ย y ของ train |"]
    if not pending:
        rows = {k: c.to_row() for k, c in L.track_configs().items()}
        md += ["", f"| field | `{L.MAIN}` | `{L.CONTROL}` |", "|---|---|---|"]
        md += [f"| `{k}` | `{rows[L.MAIN][k]}` | `{rows[L.CONTROL][k]}` |" for k in rows[L.MAIN]]
    md += [
           "", "- track 6 (spectrogram รายชั่วโมง) **ไม่ทำ** (M15 · งานต่อยอด)", "- โค้ดไม่แก้ `src/` (M16) — wrapper `runner1600.py` เรียก `src.data.fold_inputs`, `src.train.run_fold`, `src.shrink`, `src.evaluate` แบบอ่านอย่างเดียว", "",
           "## 4. คาดการณ์ที่ประกาศก่อนรัน (F3 · ห้ามแก้หลังเห็นผล)", "",
           "| ข้อ | เนื้อหา | เกณฑ์อ่านผล |", "|---|---|---|",
           f"| **F3a** | IC(`{L.MAIN}`) − IC(`{L.CONTROL}`) บน {L.WF_START} → {L.WF_END} **> 0 แยกหุ้น** | \"สนับสนุน\" ในหุ้นนั้น = ΔIC > 0 **และ** ขอบล่าง 95% CI > 0 |",
           "| **F3b** | relMAE (return และบาท) ของทุก track เทียบ Naive บนช่วงเดียวกัน | รายงานค่าตรง ๆ + Diebold–Mariano vs Naive · < 1 = ชนะ Naive |",
           "", f"block bootstrap: บล็อกติดกัน {L.BOOT_BLOCK} วัน · สุ่มแบบมีคืน · seed {L.BOOT_SEED} · {L.BOOT_REPS:,} รอบ · percentile 2.5/97.5 (`runner1600.bootstrap_delta_ic`) ·"
           " ช่วงย่อยและ replay = descriptive เท่านั้น", "",
           "## 5. การตัดสินใจ (หัวข้อ M ทั้งหมด · คัดจาก PLAN_DRAFT.md)", ""]
    plan_lines = PLAN.read_text(encoding="utf-8").splitlines()
    hdr = next(i for i, s in enumerate(plan_lines) if s.startswith("| # | วันที่ | เรื่อง"))
    md += plan_lines[hdr:hdr + 2] + [s for s in plan_lines if s.startswith("| M")]
    md += ["", "## 6. ข้อจำกัดที่ต้องเขียนในรายงาน (PLAN_DRAFT · M17, M18)", ""]
    md += _plan_section("## ข้อจำกัดที่ต้องเขียนในรายงาน")
    md += ["", "## 7. อ้างอิง LOCK ของ next_day และ gate (M21, M23)", "",
           (f"- `task_b_v2/LOCK.md` (next_day) ที่ใช้อ้าง config track 1: commit `NEXTDAY_LOCK_COMMIT: {nd_commit}`"
            if nd_commit else "- `task_b_v2/LOCK.md` (next_day): **ยังไม่ถูก commit** — สร้างไฟล์นี้ใหม่หลัง commit 2"),
           "- gate ตรวจ commit ล่าสุดของ `task_b_v2/LOCK.md` ตอนรันจาก git ต้องเท่ากับ hash ข้างบน และไฟล์ต้องสะอาด",
           "- gate (`lock_gate.py`) ปฏิเสธการรันจริงถ้า: (ก) `git status` ของ `task_b_v2/src/` หรือ `task_b_v2/mode1600/` ไม่สะอาด ·"
           " (ข) ไฟล์นี้ยังไม่ถูก commit · ถูกแก้หลัง commit · หรือยังเป็นร่าง · (+) LOCK.md ของ next_day ไม่ตรงข้างบน",
           "- (ค) ทุกไฟล์ผลมีคอลัมน์ `run_head_commit` + `run_time` · SHA-256 ของไฟล์ข้อมูลตรวจทุกครั้งที่โหลด (หัวข้อ 1)",
           "- **ไม่ตรวจ SHA ของไฟล์โค้ด** — ผู้ใช้ไม่เอา M19 (เปราะเรื่อง CRLF/LF บน Windows และขวางการรวมกับ task อื่น)", ""]
    md += ["## 8. ขั้นตอนรัน (หลังผู้ใช้ commit และแจ้งเท่านั้น)", "",
           "1. ผู้ใช้ commit 2 ของ next_day (`task_b_v2/LOCK.md` ฯลฯ · ไม่รวม `mode1600/`) — **ทำแล้ว** (hash ในหัวข้อ 7)",
           "2. `python -m mode1600.draft_lock1600` สร้างไฟล์นี้ใหม่ (บันทึก hash ของ LOCK.md · marker ร่างหายไป) · รัน `pytest task_b_v2/mode1600/tests`",
           "3. ผู้ใช้ตรวจแล้ว commit `task_b_v2/mode1600/` ทั้งโฟลเดอร์ + `task_b_v2/data/intraday/` (หัวข้อ 9 · Claude ไม่ commit) — **ก่อนเริ่ม C6 ของ next_day** (LOCK.md หัวข้อ 9 ข้อ (d))",
           "   · C6 ของ next_day: Claude เป็นผู้รัน หลังผู้ใช้ commit LOCK ทั้งสองฉบับ + โค้ด C6 และแจ้ง (LOCK.md หัวข้อ 9)",
           "4. หลัง C6 (final + replay + S1) ของ next_day เสร็จ — รันได้ทันที ไม่ต้องรอ C7 (LOCK.md B40): Claude รัน `python -m mode1600.run1600 --jobs 6` หลังผู้ใช้สั่ง — ผ่าน gate (หัวข้อ 7) แล้วจึงโหลดข้อมูล",
           "   → เขียนผลที่ `task_b_v2/results/mode1600/` ครั้งเดียว (มีผลอยู่แล้ว = ปฏิเสธ)",
           "5. รายงาน F3a/F3b + ช่วงย่อย + replay (i)/(ii) พร้อมป้ายที่มา · ห้ามแก้ config หลังเห็นผล", "",
           "## 9. ช่องสำหรับผู้ใช้", "",
           "ไฟล์ที่ต้อง commit พร้อมกัน (ทั้งโฟลเดอร์ `task_b_v2/mode1600/` — gate ต้องการให้โฟลเดอร์นี้สะอาด):", ""]
    md += [f"- `task_b_v2/mode1600/{f}`" for f in ("LOCK_1600.md", *CODE_FILES, *DOC_FILES, *TEST_FILES)]
    md += ["- `task_b_v2/mode1600/audit/*.csv` (รายละเอียดของ AUDIT.md)",
           "- `task_b_v2/data/intraday/**` (ไฟล์ข้อมูลรายชั่วโมง 46 ไฟล์) + `task_b_v2/data/SOURCES.md` (ส่วน intraday) — commit ชุดเดียวกัน", "",
           "- commit hash: `__________` · วันที่ผู้ใช้ยืนยัน: `__________`"]
    return "\n".join(md) + "\n"


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    OUT.write_text(render(), encoding="utf-8")
    print(f"เขียน {OUT}")


if __name__ == "__main__":
    main()
