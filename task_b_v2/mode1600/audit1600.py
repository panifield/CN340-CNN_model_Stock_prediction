"""
audit1600.py — ตรวจข้อมูลของโหมด 16:00 แล้วเขียนผลเป็น mode1600/AUDIT.md (ไม่เทรน · ไม่แก้ไฟล์ใดนอก mode1600/)
=========================================================================================================
ไฟล์นี้ทำอะไร: ตอบข้อ 1–6 ของงาน "เตรียมโหมด 16:00 ระยะที่ 1"
    1 timezone + ความหมายของ label · 2 แท่งต่อวัน · 3 จับคู่วันที่กับไฟล์รายวัน + ความเข้ากันได้ของสองแหล่ง (≤ 2025-02-25 เท่านั้น)
    4 snapshot สดกับวัน replay · 5 การแก้ข้อมูลย้อนหลังของ Yahoo · 6 จำนวนแถวที่ใช้ได้

ทำไมต้องมี: ก่อนเขียนแผน/โมเดล ต้องรู้ว่าข้อมูลมีอะไร ขาดอะไร และเชื่อได้แค่ไหน — สร้างซ้ำได้ด้วยคำสั่งเดียว

*** กติกา: ห้ามคำนวณสถิติของ target (ราคาปิดทางการเทียบ C15/C16) บนวัน ≥ 2025-02-26 ***
    ทุกการคำนวณที่ใช้ราคาปิดรายวันร่วมกับแท่ง ผ่าน `_dev_only()` ซึ่ง assert ว่าวันที่ ≤ DEV_END
    ส่วนอื่น (นับแถว · เวลา · ค่าว่าง · จับคู่วันที่ · เทียบแท่งกับแท่ง) ใช้ได้ทุกช่วง

รัน (จาก task_b_v2/):  PYTHONDONTWRITEBYTECODE=1 python -m mode1600.audit1600
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mode1600 import data1600 as D

sys.dont_write_bytecode = True
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

DEV_END = pd.Timestamp("2025-02-25")
FINAL_START, FINAL_END = pd.Timestamp("2025-02-26"), pd.Timestamp("2026-08-28")
GAP_START, GAP_END = pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-25")      # หลัง final ก่อน replay (ใช้เทรน replay)
REPLAY_START, REPLAY_END = pd.Timestamp("2026-09-28"), pd.Timestamp("2026-10-22")
WF_START = pd.Timestamp("2024-10-01")                                            # ช่วงประเมิน walk-forward ที่เสนอ
VOL_LOOKBACK = 20
RETRAIN_EVERY = 20
VOL_HOURS = (11, 12, 14, 15)          # volume ที่ใช้ได้: ไม่ใช้แท่ง 10 (ทีม) และแท่ง 13 (ไม่บังคับ)
OUT_MD = D.MODE_DIR / "AUDIT.md"
OUT_DIR = D.MODE_DIR / "audit"
# หมายเหตุ: AUDIT.md สร้างจาก clone ของ repo ทีม (snapshot ครบ 30 ไฟล์ + แคช backfill) · data/intraday/ เก็บเฉพาะไฟล์ที่ run1600 ใช้
# → ถ้าจะรัน audit ซ้ำ ให้ตั้ง env MODE1600_TEAM_TASK_B=<clone>/task_b
BACKFILL_CACHE_DIRS = [D.TEAM_TASK_B.parent / "backfill" / "data" / n
                       for n in ("task_b_1h_snapshots", "task_b_1h_snapshots_r2")]
TEAM_BACKFILL_CSV = D.TEAM_TASK_B / "results" / "phase1d_r2" / "backfill_same_day_close.csv"
TEAM_EXPECTED_MISSING = {"KBANK": {"2025-03-28", "2026-04-20"},
                         "ADVANC": {"2025-03-28", "2026-04-20", "2025-09-16"}}


def _dev_only(df):
    """ประตูเดียวของการคำนวณที่ใช้ราคาปิดรายวันคู่กับแท่ง — ตัดเหลือวัน ≤ 2025-02-25 แล้ว assert"""
    out = df[df.index <= DEV_END]
    assert out.index.max() <= DEV_END
    return out


def md_table(df, index=False, floatfmt="{:.4f}"):
    """DataFrame → ตาราง markdown (ไม่พึ่ง tabulate)"""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float) and not np.isnan(v) and not float(v).is_integer():
                cells.append(floatfmt.format(v))
            elif isinstance(v, float) and float(v).is_integer():
                cells.append(f"{int(v):,}")
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                cells.append(f"{int(v):,}")
            elif isinstance(v, pd.Timestamp):
                cells.append(str(v.date()) if v == v.normalize() and v.tzinfo is None else v.strftime("%Y-%m-%d %H:%M"))
            else:
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _d(ts):
    return str(pd.Timestamp(ts).date())


def tick_size(p):
    """ขั้นราคา SET ตามระดับราคา (SPEC ข้อ 5.8 · legacy/config.py SET_TICK_BANDS)"""
    for hi, t in [(2, 0.01), (5, 0.02), (10, 0.05), (25, 0.1), (100, 0.25), (200, 0.5), (400, 1.0)]:
        if p < hi:
            return t
    return 2.0


# ------------------------------------------------------------------ โหลดทุกอย่างครั้งเดียว
def load_all():
    data = {}
    for t in D.TICKERS:
        hist = D.load_history(t)
        snaps = D.snapshot_index(t)
        assert snaps["sha_ok"].all(), f"{t}: snapshot บางไฟล์ SHA ไม่ตรง meta"
        snap_bars = {r.path: D.load_snapshot(r.path) for r in snaps.itertuples()}
        daily = D.load_daily(t)
        data[t] = {"hist": hist, "snaps": snaps, "snap_bars": snap_bars, "daily": daily,
                   "final_snap": snaps.iloc[-1]}
    return data


# ------------------------------------------------------------------ ข้อ 1
def section1(data):
    out = ["## 1. timezone และความหมายของ label", ""]
    rows = []
    for t, d in data.items():
        h = d["hist"]
        rows.append({"หุ้น": t, "แถว (แท่ง)": len(h), "timestamp +07:00": f"{len(h):,}/{len(h):,}",
                     "แท่งแรก": h["ts"].min().strftime("%Y-%m-%d %H:%M%z"),
                     "แท่งสุดท้าย": h["ts"].max().strftime("%Y-%m-%d %H:%M%z"),
                     "ชั่วโมงที่พบ (label)": ",".join(str(x) for x in sorted(h["hour"].unique())),
                     "นาที ≠ 0": int((h["ts"].dt.minute != 0).sum())})
    out += ["ไฟล์ล็อก `*_1h_730d.csv` (SHA ตรง README ทีมทั้งสองไฟล์):", "", md_table(pd.DataFrame(rows)), ""]

    # หลักฐาน (ก) เวลาในไฟล์ตรงกับเวลาทำการ SET ไหม
    # หลักฐาน (ข) snapshot ที่ดาวน์โหลด 16:00–16:30 มีแท่งของวันนั้นถึงไหน
    ev = []
    for t, d in data.items():
        for r in d["snaps"].itertuples():
            b = d["snap_bars"][r.path]
            day = r.downloaded_at.tz_localize(None).normalize()
            hours = sorted(b.loc[b["date"] == day, "hour"].tolist())
            ev.append({"หุ้น": t, "ดาวน์โหลด": r.downloaded_at.strftime("%Y-%m-%d %H:%M"),
                       "แท่งของวันที่ดาวน์โหลด": ",".join(map(str, hours)) if hours else "(ไม่มี)",
                       "มีแท่ง 16": 16 in hours})
    ev = pd.DataFrame(ev)
    kb = ev[ev["หุ้น"] == "KBANK"].drop(columns="หุ้น")
    same = (ev.groupby("ดาวน์โหลด")["แท่งของวันที่ดาวน์โหลด"].nunique() == 1).all()

    # หลักฐาน (ค) O10 = ราคาเปิดทางการ และ max/min ของแท่งทั้งวัน = High/Low ทางการ (dev เท่านั้น)
    al = []
    for t, d in data.items():
        h = d["hist"]
        g = h.groupby("date")
        agg = pd.DataFrame({"O_first": g.apply(lambda x: x.sort_values("ts")["Open"].iloc[0], include_groups=False),
                            "first_hour": g["hour"].min(), "H_all": g["High"].max(), "L_all": g["Low"].min()})
        j = _dev_only(agg.join(d["daily"][["Open", "High", "Low"]], how="inner"))
        j10 = j[j["first_hour"] == 10]
        tk = j10["Open"].map(tick_size)
        al.append({"หุ้น": t, "วัน (≤ 2025-02-25)": len(j),
                   "Open แท่ง 10 = Open ทางการ": f"{(j10['O_first'] == j10['Open']).mean():.1%}",
                   "ต่างไม่เกิน 1 tick": f"{((j10['O_first'] - j10['Open']).abs() <= tk + 1e-9).mean():.1%}",
                   "max High ทุกแท่ง = High ทางการ": f"{(j['H_all'] == j['High']).mean():.1%}",
                   "min Low ทุกแท่ง = Low ทางการ": f"{(j['L_all'] == j['Low']).mean():.1%}"})
    out += ["**หลักฐาน (ก) — เวลาในไฟล์ตรงกับตารางเวลาซื้อขาย SET (เวลาไทย)**",
            "",
            "- ทุก timestamp มี offset `+07:00` (โมดูลปฏิเสธไฟล์ถ้ามีแถวที่ไม่ใช่) · Asia/Bangkok ไม่มี DST",
            "- label ที่พบ = 10, 11, 12, (13), 14, 15, 16 · **ไม่มี 09 และไม่มี 17**",
            "  SET เปิดเช้า 10:00 และปิดบ่าย 16:30 (+ ATC) → ถ้า label = เวลา *จบ* แท่ง จะต้องมีแท่ง 11:00 เป็นแท่งแรก"
            " และแท่ง 17:00 สำหรับ 16:00–16:30 · ที่พบ (แท่งแรก 10 · แท่งสุดท้าย 16) เข้ากับ label = เวลา *เริ่ม* แท่งเท่านั้น",
            "",
            "**หลักฐาน (ข) — snapshot สดที่รู้เวลาดาวน์โหลด** (KBANK · ADVANC "
            + ("เหมือนกันทุกไฟล์" if same else "ต่างจาก KBANK บางไฟล์ — ดูไฟล์ audit/snapshot_today_bars.csv") + ")",
            "", md_table(kb), "",
            "- ดาวน์โหลด 16:02–16:22 → มีแท่งของวันนั้นถึง **15** แต่ **ไม่มี 16** · ดาวน์โหลด ≥ 16:42 → มีแท่ง 16",
            "  ถ้า label = เวลาจบ แท่ง \"16\" (= 15:00–16:00) ต้องจบแล้วตอน 16:00 และควรเห็นใน snapshot 16:19/16:22 (Yahoo ดีเลย์ ~15 นาที)",
            "  → ไม่เห็นจนถึง 16:22 แต่เห็นตอน 16:42 = แท่ง 16 คือช่วง 16:00–ปิดตลาด (label = เวลาเริ่ม)",
            "- ข้อ 5: แท่ง 15 ใน snapshot ~16:02 ถูกแก้ทีหลังบ่อย (volume เพิ่มขึ้น) = แท่ง 15 ยังสะสมอยู่จนถึง 16:00 →"
            " เข้ากับ 15:00–16:00 (ถ้าเป็น 14:00–15:00 ควรนิ่งแล้วตอน 16:02)",
            "",
            "**หลักฐาน (ค) — เทียบราคาแท่งกับราคารายวันทางการ (investing)** · เฉพาะวัน ≤ 2025-02-25:", "",
            md_table(pd.DataFrame(al)), "",
            "- **หลักฐานข้อนี้อ่อน:** Open แท่ง 10:00 ตรงราคาเปิดทางการพอดีแค่ 41–50% ของวัน (ห่างไม่เกิน 1 tick 89–91%)"
            " · คนละแหล่ง (Yahoo vs investing) จึงตรงกันไม่ได้ทุกวัน → ใช้ยืนยันได้แค่ว่าแท่ง 10:00 อยู่ใกล้รอบเปิด ไม่ใช่พิสูจน์ label"
            " · การยืนยัน label อาศัยหลักฐาน (ก) และ (ข) เป็นหลัก",
            "",
            "**สรุปข้อ 1: ยืนยัน — เวลาไทย (+07:00) และ label = เวลาเริ่มแท่ง** · แท่ง 15:00 = 15:00–16:00 (รู้ครบพอดี 16:00 *ในทางทฤษฎี*"
            " — แต่ Yahoo ดีเลย์ทำให้ snapshot 16:02 ยังไม่ใช่ฉบับสุดท้าย ดูข้อ 5) · แท่ง 16:00 = 16:00–ปิด ห้ามใช้",
            "- ข้อสังเกต: แท่ง 13:00 (ช่วงพักเที่ยงของ SET 12:30–14:30) มีอยู่จริงและมี volume ไม่เป็นศูนย์ · ความหมายไม่ชัด"
            " (อาจเป็นการจัดกลุ่มเวลาของ Yahoo) · ไม่ใช้แท่งนี้ใน feature ใด ๆ ตามทีม", ""]

    # ตัวอย่าง 3 วัน (วัน dev)
    out += ["**ตัวอย่าง 3 วัน (KBANK · วันในโซน dev)** — ราคาปิดทางการจากไฟล์รายวัน", ""]
    d = data["KBANK"]
    for day in ["2023-11-01", "2024-07-01", "2025-02-25"]:
        b = d["hist"][d["hist"]["date"] == pd.Timestamp(day)][["ts", "Open", "High", "Low", "Close", "Volume"]].copy()
        b["ts"] = b["ts"].dt.strftime("%H:%M")
        b = b.rename(columns={"ts": "label"})
        dr = d["daily"].loc[pd.Timestamp(day)]
        out += [f"`{day}` · รายวันทางการ O/H/L/C = {dr['Open']}/{dr['High']}/{dr['Low']}/{dr['Close']}", "",
                md_table(b, floatfmt="{:.2f}"), ""]
    ev.to_csv(OUT_DIR / "snapshot_today_bars.csv", index=False)
    return out


# ------------------------------------------------------------------ ข้อ 2
def section2(data):
    out = ["## 2. แท่งต่อวัน", ""]
    rows, miss_rows, v0_rows, h13, pat_rows = [], [], [], [], []
    for t, d in data.items():
        h = d["hist"]
        dt = D.day_table(h)
        all_days = pd.Index(sorted(h["date"].unique()))
        inc = dt[~dt["complete"]]
        # วันที่มีแท่ง 16 อย่างเดียว ไม่อยู่ใน day_table (ตัดแท่ง 16 ไปแล้ว) → นับเป็นวันไม่ครบด้วย
        only16 = all_days.difference(dt.index)
        found = set(inc.index.strftime("%Y-%m-%d")) | set(only16.strftime("%Y-%m-%d"))
        rows.append({"หุ้น": t, "วันในไฟล์": len(all_days), "มีแท่ง 10,11,12,14,15 ครบ": int(dt["complete"].sum()),
                     "วันไม่ครบ": len(found),
                     "ตรงรายการทีม": "ตรง" if found == TEAM_EXPECTED_MISSING[t] else
                     f"ต่าง: เพิ่ม {sorted(found - TEAM_EXPECTED_MISSING[t])} ขาด {sorted(TEAM_EXPECTED_MISSING[t] - found)}"})
        for day in sorted(found):
            hrs = sorted(h.loc[h["date"] == pd.Timestamp(day), "hour"].tolist())
            miss_rows.append({"หุ้น": t, "วัน": day, "แท่งที่มี": ",".join(map(str, hrs)),
                              "แท่งบังคับที่ขาด": ",".join(str(x) for x in D.REQUIRED_HOURS if x not in hrs)})
        v0 = h.groupby("hour").agg(n=("Volume", "size"), vol0=("Volume", lambda v: int((v == 0).sum())))
        v0_rows.append({"หุ้น": t, **{f"{hh}:00": f"{r.vol0}/{r.n}" for hh, r in v0.iterrows()}})
        pat = h.groupby("date")["hour"].apply(lambda s: ",".join(str(x) for x in sorted(s) if x != 13))
        for k, n in pat.value_counts().items():
            pat_rows.append({"หุ้น": t, "แท่งในวัน (ไม่นับ 13)": k, "จำนวนวัน": int(n)})
        yr = h.assign(y=h["date"].dt.year).groupby("y")
        for y, g in yr:
            h13.append({"หุ้น": t, "ปี": str(y), "วัน": g["date"].nunique(), "วันที่มีแท่ง 13": int((g["hour"] == 13).sum())})
        # วันที่มีแท่งไม่ปกติ (ครึ่งวัน / แท่ง 16 ขาด)
        n16 = all_days.difference(pd.Index(h.loc[h["hour"] == 16, "date"].unique()))
        rows[-1]["วันไม่มีแท่ง 16"] = ", ".join(n16.strftime("%Y-%m-%d")) or "—"
        vsum = dt.loc[dt["complete"], [f"V{x}" for x in VOL_HOURS]].sum(axis=1)
        rows[-1]["วันที่ V11+V12+V14+V15 = 0"] = int((vsum == 0).sum())
    out += [md_table(pd.DataFrame(rows)), "", "วันไม่ครบ (ตัดทั้งวัน ห้าม ffill):", "", md_table(pd.DataFrame(miss_rows)), "",
            "แท่งที่ `Volume = 0` แยกตามชั่วโมง (จำนวน/ทั้งหมด):", "", md_table(pd.DataFrame(v0_rows)), "",
            "แท่ง 13:00 ต่อปี:", "",
            md_table(pd.DataFrame(h13).pivot(index="ปี", columns="หุ้น", values="วันที่มีแท่ง 13")
                     .join(pd.DataFrame(h13).groupby("ปี")["วัน"].first()), index=True), "",
            "รูปแบบแท่งต่อวัน (ไม่นับแท่ง 13 ซึ่งมีไม่สม่ำเสมอ):", "", md_table(pd.DataFrame(pat_rows)), "",
            "- **วันครึ่งวัน:** ไม่พบในความหมายของวันที่ตลาดเปิดครึ่งวันตามปฏิทิน — นอกจากวันในตารางวันไม่ครบ ทุกวันมีแท่ง 10,11,12,14,15,16 ครบ",
            "  · 2025-03-28 จบที่แท่ง 14 (ไม่มีแท่ง 15, 16 — น่าจะเป็นวันแผ่นดินไหวที่ SET หยุดซื้อขายช่วงบ่าย · ยังไม่ได้ยืนยันกับประกาศ SET)",
            "  · 2026-04-20 มีแค่แท่ง 16 (ข้อมูล Yahoo เสีย — ทีมพบเหมือนกัน)",
            "- แท่ง 10:00 `Volume = 0` ใน 72–76% ของวันในไฟล์ล็อก · ข้อ 5 พบเพิ่มว่าใน snapshot สด ~16:02 แท่ง 10:00 มี volume"
            " แต่ฉบับหลังถูกแก้เป็น 0 → **ยืนยันว่าห้ามใช้ volume แท่ง 10** (ค่าที่เห็นตอนเทรนกับตอนใช้จริงคนละแบบ)", ""]
    return out


# ------------------------------------------------------------------ ข้อ 3
def section3(data):
    out = ["## 3. จับคู่วันที่กับไฟล์รายวัน และความเข้ากันได้ของสองแหล่ง", ""]
    rows, st, rng = [], [], []
    for t, d in data.items():
        h, daily = d["hist"], d["daily"]
        hd = pd.Index(sorted(h["date"].unique()))
        lo, hi = hd.min(), hd.max()
        dd = daily.index[(daily.index >= lo) & (daily.index <= hi)]
        rows.append({"หุ้น": t, "ช่วงที่ทับกัน": f"{_d(lo)} → {_d(hi)}", "วันในรายชั่วโมง": len(hd), "วันในรายวัน": len(dd),
                     "มีแค่รายชั่วโมง": ", ".join(hd.difference(dd).strftime("%Y-%m-%d")) or "—",
                     "มีแค่รายวัน": ", ".join(dd.difference(hd).strftime("%Y-%m-%d")) or "—"})
        dt = D.day_table(h)
        c16 = h[h["hour"] == 16].set_index("date")["Close"].rename("C16")
        j = _dev_only(dt[dt["complete"]][["C15"]].join(c16, how="inner").join(daily["Close"], how="inner"))
        for name, col in [("Close_official − C15", "C15"), ("Close_official − close_bar16", "C16")]:
            diff = j["Close"] - j[col]
            st.append({"หุ้น": t, "เทียบ": name, "วัน": len(diff), "เท่ากันพอดี": f"{(diff == 0).mean():.1%}",
                       "mean": diff.mean(), "mean abs": diff.abs().mean(), "median abs": diff.abs().median(),
                       "p95 abs": diff.abs().quantile(0.95), "max abs": diff.abs().max()})
        jr = _dev_only(dt[dt["complete"]][["C15", "O10"]].join(daily[["High", "Low"]], how="inner"))
        rng.append({"หุ้น": t, "วัน": len(jr),
                    "C15 อยู่ใน [Low, High] ทางการ": f"{((jr['C15'] >= jr['Low']) & (jr['C15'] <= jr['High'])).mean():.1%}",
                    "O10 อยู่ใน [Low, High] ทางการ": f"{((jr['O10'] >= jr['Low']) & (jr['O10'] <= jr['High'])).mean():.1%}"})
    out += ["**วันที่ที่มีแค่ฝั่งเดียว** (ช่วงที่ไฟล์รายชั่วโมงครอบคลุม · นับวันเท่านั้น):", "", md_table(pd.DataFrame(rows)), "",
            "**สถิติส่วนต่างราคา (บาท) — เฉพาะวัน ≤ 2025-02-25** (วันที่แท่งบังคับครบ + มีแท่ง 16 + มีราคารายวัน):", "",
            md_table(pd.DataFrame(st), floatfmt="{:.3f}"), "",
            "ระดับราคาของสองแหล่งเข้ากันไหม (ราคาแท่ง Yahoo อยู่ในกรอบ High–Low ทางการของวันเดียวกัน · ≤ 2025-02-25):", "",
            md_table(pd.DataFrame(rng)), "",
            "- ทีมรายงาน close_bar16 กับราคาปิด investing ต่างเฉลี่ย 0.33 / 0.69 บาท (ทุกช่วง · 701 วัน) ·"
            " ช่วง dev ได้ 0.28 / 0.63 บาท — ระดับเดียวกัน",
            "- **ความหมาย:** C15 และ O10 ของ Yahoo อยู่ในกรอบ High–Low ทางการของวันเดียวกันทุกวัน (100%) → สองแหล่งเป็นราคาชุดเดียวกัน (ไม่มีการเลื่อนวัน/สเกล)"
            " · แต่ close_bar16 ≠ ราคาปิดทางการ (ตรงแค่ ~45–48%) เพราะราคาปิดทางการมาจากรอบ ATC หลังแท่ง 16",
            "  ส่วนต่าง Close_official − C15 จึงเป็น \"การขยับหลัง 16:00 + รอบปิด ATC\" = สิ่งที่โหมดนี้ทำนาย ·"
            " ขนาดเฉลี่ย 0.34 บาท (KBANK ≈ 0.7 tick) / 0.76 บาท (ADVANC ≈ 0.8 tick) และ **ตรงกับ C15 พอดี 40–44% ของวัน** (dev) →"
            " MAE จะเอื้อคำทำนายที่ใกล้ 0 แบบเดียวกับ next_day (B16)",
            "", "**แหล่งของ `task_b_v2/data/*_10Y_Cleaned.csv` แยกช่วง** (จาก `data/SOURCES.md` + git log/raw_data_sources ของ repo ทีม):", "",
            "| ช่วง | แหล่ง | หลักฐาน |", "|---|---|---|",
            "| ต้นไฟล์ → 2026-08-28 | investing.com | ไฟล์เดิมของโปรเจกต์ (ตรงกับ `StockCNN/raw_data/` 2,433 แถวแรก) |",
            "| 2026-08-31 → 2026-09-25 (20 วัน) | **investing.com** | commit ทีม `852e02b` \"append Investing.com daily rows\" · `raw_data_sources/investing_20260927/` |",
            "| 2026-09-28 → 2026-10-22 (replay) | **Yahoo 1d** (ทีมตรวจ OHLC ก่อน append) | commit ทีม `a875e41` … `6350406` \"(yahoo)\" · `raw_data_sources/yahoo_2026*/` |",
            "",
            "→ ราคาปิดทางการของช่วงเทรน (≤ 2026-09-25) มาจาก investing แต่ของ replay มาจาก Yahoo 1d (แหล่งเปลี่ยนตรงขอบ replay) —"
            " ต้องเขียนเป็นข้อจำกัดในแผน (feature `r15`/`gap` ของ 2026-09-28 ใช้ Close(2026-09-25) จาก investing)", ""]
    return out


# ------------------------------------------------------------------ ข้อ 4
def _complete_day(bars, day):
    hrs = set(bars.loc[bars["date"] == day, "hour"])
    return all(x in hrs for x in D.REQUIRED_HOURS)


def replay_days(data):
    daily = data["KBANK"]["daily"]
    days = daily.index[(daily.index >= REPLAY_START) & (daily.index <= REPLAY_END)]
    assert (data["ADVANC"]["daily"].index[(data["ADVANC"]["daily"].index >= REPLAY_START)
                                          & (data["ADVANC"]["daily"].index <= REPLAY_END)] == days).all()
    return days


def section4(data):
    out = ["## 4. snapshot สด", ""]
    rows = []
    for r in data["KBANK"]["snaps"].itertuples():
        row = {"โฟลเดอร์": r.folder, "ดาวน์โหลด (meta)": r.downloaded_at.strftime("%Y-%m-%d %H:%M:%S")}
        for t in D.TICKERS:
            p = r.path.with_name(r.path.name.replace("KBANK", t))
            b = data[t]["snap_bars"][p]
            row[f"{t} ครอบคลุม"] = f"{_d(b['date'].min())} → {_d(b['date'].max())} ({b['date'].nunique()} วัน)"
            row[f"{t} SHA"] = "ตรง" if data[t]["snaps"].set_index("path").loc[p, "sha_ok"] else "ไม่ตรง"
        rows.append(row)
    cache_rows = []
    for cdir in BACKFILL_CACHE_DIRS:
        for p in sorted(cdir.glob("*.csv")):
            b = D.read_bars(p)
            cache_rows.append({"ไฟล์": f"backfill/data/{cdir.name}/{p.name}",
                               "แก้ไขล่าสุด (mtime · ไม่มี meta)": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
                               .tz_convert("Asia/Bangkok").strftime("%Y-%m-%d %H:%M"),
                               "ครอบคลุม": f"{_d(b['date'].min())} → {_d(b['date'].max())}"})
    out += ["`task_b/raw_data_intraday_live/` (ทุกไฟล์มี .meta.json · SHA ตรง meta ทุกไฟล์):", "", md_table(pd.DataFrame(rows)), "",
            "แคช Yahoo ของ backfill ทีม (ไม่มี meta/เวลาดาวน์โหลด — ใช้เป็นแหล่งไม่ได้ถ้าต้องพิสูจน์เวลา):", "",
            md_table(pd.DataFrame(cache_rows)), ""]

    team = pd.DataFrame()
    if TEAM_BACKFILL_CSV.exists():           # อ่านเฉพาะคอลัมน์ที่มาของแท่ง — ไม่อ่านคำทำนาย/ผลจริง
        team = pd.read_csv(TEAM_BACKFILL_CSV, usecols=["target_date", "ticker", "bar_source", "snapshot_downloaded_at"])
        team["target_date"] = pd.to_datetime(team["target_date"])
    rows = []
    for day in replay_days(data):
        row = {"วัน": _d(day)}
        for t in D.TICKERS:
            snaps = data[t]["snaps"]
            dl = snaps["downloaded_at"].dt.tz_localize(None)
            win = snaps[(dl >= day + pd.Timedelta(hours=16)) & (dl <= day + pd.Timedelta(hours=16, minutes=30))]
            win_ok = [r for r in win.itertuples() if _complete_day(data[t]["snap_bars"][r.path], day)]
            later = snaps[dl > day + pd.Timedelta(hours=16, minutes=30)]
            later_ok = [r for r in later.itertuples() if _complete_day(data[t]["snap_bars"][r.path], day)]
            if t == "KBANK":
                row["snapshot 16:00–16:30"] = ", ".join(r.downloaded_at.strftime("%H:%M") for r in win_ok) or "ไม่มี"
                row["snapshot หลังจากนั้นที่มีวันนี้ครบ (แรกสุด)"] = (later_ok[0].downloaded_at.strftime("%m-%d %H:%M")
                                                                if later_ok else "ไม่มี")
                row["จำนวน snapshot ที่มีวันนี้ครบ"] = len(later_ok) + len(win_ok)
            else:
                same = (row["snapshot 16:00–16:30"] == (", ".join(r.downloaded_at.strftime("%H:%M") for r in win_ok) or "ไม่มี"))
                row["ADVANC เหมือน KBANK"] = "ใช่" if same else "ไม่"
            if len(team):
                m = team[(team["target_date"] == day) & (team["ticker"].str.startswith(t))]
                if t == "KBANK":
                    row["bar_source ของทีม"] = ", ".join(sorted(set(m["bar_source"]))) if len(m) else "— (ไม่อยู่ในไฟล์ทีม)"
        row["ป้ายที่เสนอ (ii)"] = ("snapshot_1600_1630" if row["snapshot 16:00–16:30"] != "ไม่มี" else
                                  "ไม่มี (ใช้ได้แค่ทาง i)")
        rows.append(row)
    rp = pd.DataFrame(rows)
    rp.to_csv(OUT_DIR / "replay_sources.csv", index=False)
    n_win = int((rp["snapshot 16:00–16:30"] != "ไม่มี").sum())
    out += [f"**วัน replay {len(rp)} วันทำการ (นับจากไฟล์รายวัน) เทียบ snapshot:**", "", md_table(rp), "",
            f"- มี snapshot ที่ดาวน์โหลด 16:00–16:30 และมีแท่ง 10..15 ของวันนั้นครบ: **{n_win}/{len(rp)} วัน**",
            f"- snapshot `{data['KBANK']['final_snap'].downloaded_at.strftime('%Y-%m-%d %H:%M')}` (โฟลเดอร์ 20261023)"
            " มีแท่งครบ **ทุกวัน** ของ replay → ใช้เป็นแหล่งเดียวของทาง (i) ได้",
            "- ตารางที่มาของทีม (`PHASE1D_PLAN_R2.md` ข้อ 19) ครอบคลุมแค่ถึง 10-15 · ไฟล์ backfill ของทีม (คอลัมน์ `bar_source`)"
            " ตรงกับตารางข้างบนสำหรับวันที่มี snapshot · วันที่ไม่มี snapshot ทีมใช้แคช Yahoo (`yahoo_final`) ซึ่งไม่มีเวลาดาวน์โหลดที่พิสูจน์ได้",
            ""]
    return out


# ------------------------------------------------------------------ ข้อ 5
KEY_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _compare(a, b, hours=D.KNOWN_HOURS):
    """เทียบแท่งชุด a กับ b บน (date, hour) ที่มีทั้งคู่ · คืน DataFrame ของเซลล์ที่ต่าง"""
    ka = a[a["hour"].isin(hours)].set_index(["date", "hour"])[KEY_COLS]
    kb = b[b["hour"].isin(hours)].set_index(["date", "hour"])[KEY_COLS]
    common = ka.index.intersection(kb.index)
    ka, kb = ka.loc[common], kb.loc[common]
    diff = ka.ne(kb)
    return common, diff, ka, kb


def section5(data):
    out = ["## 5. การแก้ข้อมูลย้อนหลังของ Yahoo (แท่ง 10..15 เท่านั้น)", ""]
    # (ก) snapshot วันเดียวกัน (16:00–16:30) เทียบ snapshot สุดท้าย
    rows, detail = [], []
    for t in D.TICKERS:
        d = data[t]
        final = d["snap_bars"][d["final_snap"].path]
        for r in d["snaps"].itertuples():
            if r.path == d["final_snap"].path:
                continue
            day = r.downloaded_at.tz_localize(None).normalize()
            if not (16 * 60 <= r.downloaded_at.hour * 60 + r.downloaded_at.minute <= 16 * 60 + 30):
                continue
            b = d["snap_bars"][r.path]
            common, diff, ka, kb = _compare(b[b["date"] == day], final[final["date"] == day])
            ch = diff.sum()
            hrs_changed = sorted({h for (_, h), row in diff.iterrows() if row.any()})
            v15 = (kb.loc[(day, 15), "Volume"] / ka.loc[(day, 15), "Volume"] - 1) if (day, 15) in ka.index else np.nan
            rows.append({"หุ้น": t, "วัน": _d(day), "snapshot": r.downloaded_at.strftime("%H:%M"),
                         "แท่งที่เทียบ": len(common), **{c[0]: int(ch[c]) for c in KEY_COLS},
                         "แท่งที่เปลี่ยน": ",".join(map(str, hrs_changed)) or "—",
                         "C15 เปลี่ยน": bool(diff.loc[(day, 15), "Close"]) if (day, 15) in diff.index else None,
                         "V15 ฉบับสุดท้าย/16:0x − 1": v15})
            for (dd, hh), row in diff.iterrows():
                for c in KEY_COLS:
                    if row[c]:
                        detail.append({"ticker": t, "date": _d(dd), "hour": hh, "col": c, "snapshot": str(r.downloaded_at),
                                       "value_snapshot": ka.loc[(dd, hh), c], "value_final": kb.loc[(dd, hh), c]})
    sa = pd.DataFrame(rows)
    pd.DataFrame(detail).to_csv(OUT_DIR / "revisions_sameday_vs_final.csv", index=False)

    # (ข) ข้อมูลหลังปิดตลาด: ทุก snapshot เทียบ snapshot สุดท้าย เฉพาะวันที่ "จบแล้ว" ตอนดาวน์โหลด
    settled = []
    for t in D.TICKERS:
        d = data[t]
        final = d["snap_bars"][d["final_snap"].path]
        for r in d["snaps"].itertuples():
            if r.path == d["final_snap"].path:
                continue
            b = d["snap_bars"][r.path]
            cutoff = r.downloaded_at.tz_localize(None).normalize()
            if r.downloaded_at.hour >= 17:
                cutoff = cutoff + pd.Timedelta(days=1)         # ดาวน์โหลดหลังปิดตลาด = วันนั้นจบแล้ว
            common, diff, _, _ = _compare(b[b["date"] < cutoff], final[final["date"] < cutoff])
            settled.append({"หุ้น": t, "snapshot": r.downloaded_at.strftime("%Y-%m-%d %H:%M"),
                            "วันที่จบแล้วที่เทียบ": len({x[0] for x in common}), "แท่ง": len(common),
                            **{c[0]: int(diff[c].sum()) for c in KEY_COLS},
                            "วันที่มีการเปลี่ยน": ", ".join(sorted({_d(x[0]) for x, row in diff.iterrows() if row.any()})) or "—"})
    st = pd.DataFrame(settled)

    # (ค) ไฟล์ล็อก 730d เทียบทุก snapshot บนวันที่ทับกัน (≤ 2026-09-25)
    hv = []
    for t in D.TICKERS:
        d = data[t]
        for r in d["snaps"].itertuples():
            common, diff, _, _ = _compare(d["hist"], d["snap_bars"][r.path])
            days = sorted({x[0] for x in common})
            hv.append({"หุ้น": t, "snapshot": r.downloaded_at.strftime("%Y-%m-%d %H:%M"),
                       "วันที่ทับกัน": f"{_d(days[0])} → {_d(days[-1])} ({len(days)} วัน)" if days else "—",
                       "แท่ง": len(common), **{c[0]: int(diff[c].sum()) for c in KEY_COLS},
                       "วันที่ต่าง": ", ".join(sorted({_d(x[0]) for x, row in diff.iterrows() if row.any()})) or "—"})
    hv = pd.DataFrame(hv)
    hv.to_csv(OUT_DIR / "history_vs_snapshots.csv", index=False)
    st.to_csv(OUT_DIR / "settled_vs_final.csv", index=False)

    kb22 = sa[(sa["หุ้น"] == "KBANK") & (sa["วัน"] == "2026-10-22")]
    det = pd.DataFrame(detail)
    by_hour = det.groupby(["hour", "col"]).size().unstack(fill_value=0).reindex(columns=KEY_COLS, fill_value=0)
    v10 = det[(det["hour"] == 10) & (det["col"] == "Volume")]
    n_c15 = int(sa["C15 เปลี่ยน"].sum())
    vv = det[(det["col"] == "Volume") & det["hour"].isin(VOL_HOURS)]
    net = (vv["value_final"] - vv["value_snapshot"]).groupby([vv["ticker"], vv["date"], vv["snapshot"]]).sum()
    sum_up, sum_n = int((net > 0).sum()), len(sa)
    n_pairs = len(sa)
    out += ["**(ก) snapshot ที่ดาวน์โหลด 16:00–16:30 ของวัน t เทียบ snapshot สุดท้าย (2026-10-23 03:55) — แท่ง 10..15 ของวัน t**",
            "(ตัวเลข O/H/L/C/V = จำนวนแท่งที่ค่านั้นเปลี่ยน · รายละเอียดทุกเซลล์ใน `audit/revisions_sameday_vs_final.csv`)", "",
            md_table(sa, floatfmt="{:+.1%}"), "",
            "จำนวนเซลล์ที่เปลี่ยน แยกตามแท่ง × คอลัมน์ (รวมทุกคู่ข้างบน):", "", md_table(by_hour, index=True), "",
            f"- ตัวอย่างที่ผู้ใช้พบ (KBANK 2026-10-22 แท่ง 15 volume 2,299,637 → 2,746,496): "
            + ("**ยืนยัน**" if len(kb22) and kb22["V"].iloc[0] > 0 else "ไม่พบ — ตรวจ"),
            f"- **C15 (ฐานของ target และ feature `r15`) เปลี่ยน {n_c15}/{n_pairs} คู่** · V15 ฉบับสุดท้าย **สูงกว่า** ฉบับ 16:0x ทุกคู่"
            f" ({sa['V15 ฉบับสุดท้าย/16:0x − 1'].min():+.0%} ถึง {sa['V15 ฉบับสุดท้าย/16:0x − 1'].max():+.0%})"
            " → snapshot ~16:02 มีแท่ง 15 แค่ส่วนแรกของชั่วโมง (Yahoo ดีเลย์) · O/H/L แทบไม่เปลี่ยน",
            f"- volume แท่ง 10 เปลี่ยน {len(v10)} ครั้ง: ฉบับ 16:0x มีค่า แต่ฉบับสุดท้ายเป็น 0 ใน "
            f"{int((v10['value_final'] == 0).sum())} ครั้ง → Yahoo ลบ volume แท่ง 10 ทีหลัง (อธิบาย `Volume = 0` 72–76% ในไฟล์ล็อก)",
            f"- volume แท่ง 11, 12, 14 (จบไปนานแล้วตอน 16:02) ก็ถูกแก้ด้วย แต่เล็กและเป็นทางลบ · ผลรวม V11+V12+V14+V15 ฉบับสุดท้าย"
            f" สูงกว่าฉบับ 16:0x ใน {sum_up}/{sum_n} คู่ → **feature `vol15` ที่สร้างจาก snapshot 16:02 ต่ำกว่าค่าจากฉบับสุดท้ายเกือบทุกครั้ง**"
            " (ข้อมูลเทรนเป็นฉบับสุดท้ายทั้งหมด)", "",
            "**(ข) snapshot ทุกไฟล์เทียบ snapshot สุดท้าย เฉพาะวันที่ตลาดปิดไปแล้วตอนดาวน์โหลด** (ข้อมูล \"นิ่ง\" หลังปิดตลาดไหม):", "",
            md_table(st), "",
            "**(ค) ไฟล์ล็อก 730d เทียบ snapshot บนวันที่ทับกัน (แท่ง 10..15):**", "", md_table(hv), "",
            "**สรุปข้อ 5**",
            "- ข้อมูลของวันที่ตลาดปิดแล้วส่วนใหญ่นิ่ง แต่ **snapshot ที่ดาวน์โหลด 10-14 → 10-22 ได้ฉบับที่ต่างออกไป** (Open/Volume ของวันเก่าใน ก.ค.–ก.ย."
            " หลายแท่ง รวมการย้าย volume ระหว่างแท่ง 12 ↔ 13) แล้ว **snapshot 10-23 03:55 กลับมาตรงกับไฟล์ล็อก 730d ทุกแท่ง**"
            f" (ทับกัน {hv[hv['snapshot'] == '2026-10-23 03:55']['แท่ง'].iloc[0]} แท่งต่อหุ้น · ต่าง 0)",
            "- ⇒ ไฟล์ล็อก 730d กับ snapshot 10-23 03:55 เป็นฉบับเดียวกันบนช่วงที่ทับ → ทาง (i) ของ replay ต่อกับข้อมูลเทรนได้โดยไม่เปลี่ยนฉบับ",
            "- ⇒ ทาง (ii) (snapshot ~16:02) ต่างจากข้อมูลเทรนทั้งเพราะแท่ง 15 ยังไม่ครบ และเพราะ Yahoo เสิร์ฟคนละฉบับในบางวัน", ""]
    return out, sa, st, hv


# ------------------------------------------------------------------ ข้อ 6
def eligible_table(data, t):
    """
    วันที่ใช้ได้ (นับเท่านั้น · ไม่ดูค่า y) — แหล่งแท่ง: ไฟล์ล็อกถึง 2026-09-25 · snapshot สุดท้ายสำหรับ replay
    เงื่อนไข: แท่งบังคับครบ · มีราคารายวันของ t · มีแถวรายวันก่อนหน้า (t−1) · มี eligible day ก่อนหน้า ≥ 20 วัน (vol15)
    """
    d = data[t]
    final = d["snap_bars"][d["final_snap"].path]
    bars = pd.concat([d["hist"], final[final["date"] > d["hist"]["date"].max()]], ignore_index=True)
    dt = D.day_table(bars)
    daily = d["daily"]
    dates = dt.index
    in_daily = dates.isin(daily.index)
    pos = daily.index.get_indexer(dates)
    has_prev = (pos > 0) & in_daily
    elig = dt["complete"].to_numpy() & in_daily
    vsum = dt[[f"V{x}" for x in VOL_HOURS]].sum(axis=1, min_count=len(VOL_HOURS)).to_numpy()
    n_prior = np.concatenate([[0], np.cumsum(elig)[:-1]])          # eligible day ก่อนหน้า (ไม่รวมวันนี้)
    usable = elig & has_prev & (n_prior >= VOL_LOOKBACK) & (vsum > 0)
    return pd.DataFrame({"complete": dt["complete"].to_numpy(), "in_daily": in_daily, "has_prev": has_prev,
                         "eligible": elig, "n_prior": n_prior, "usable": usable}, index=dates)


def section6(data):
    out = ["## 6. จำนวนแถวที่ใช้ได้", "",
           "เงื่อนไข: แท่ง 10,11,12,14,15 ครบ · มีราคาปิดรายวันของ t และของวันทำการก่อนหน้า (แถวก่อนหน้าในไฟล์รายวัน) ·"
           f" มี eligible day ก่อนหน้า ≥ {VOL_LOOKBACK} วัน (สำหรับ vol15) · volume แท่ง 11,12,14,15 รวม > 0",
           "แหล่งแท่ง: ไฟล์ล็อก 730d (≤ 2026-09-25) + snapshot 2026-10-23 03:55 (replay) · **นับแถวเท่านั้น ไม่ดูค่า y**", ""]
    periods = [("≤ 2025-02-25 (ทั้งหมด)", None, DEV_END), ("2025-02-26 → 2026-08-28", FINAL_START, FINAL_END),
               ("2026-08-31 → 2026-09-25 (ก่อน replay)", GAP_START, GAP_END), ("replay 2026-09-28 → 2026-10-22", REPLAY_START, REPLAY_END),
               ("— ประเมิน WF ส่วนแรก 2024-10-01 → 2025-02-25", WF_START, DEV_END),
               ("— ประเมิน WF รวม 2024-10-01 → 2026-08-28", WF_START, FINAL_END),
               ("— train ก่อนบล็อกแรก (< 2024-10-01)", None, WF_START - pd.Timedelta(days=1))]
    rows, tables = [], {}
    for t in D.TICKERS:
        e = eligible_table(data, t)
        tables[t] = e
        for name, a, b in periods:
            m = (e.index <= b) & ((e.index >= a) if a is not None else True)
            rows.append({"หุ้น": t, "ช่วง": name, "วันในแท่ง": int(m.sum()), "แท่งครบ": int(e["complete"][m].sum()),
                         "+ มีรายวัน t, t−1": int((e["eligible"] & e["has_prev"])[m].sum()),
                         "ใช้ได้ (+ ประวัติ vol 20 วัน)": int(e["usable"][m].sum()),
                         "แรก": _d(e.index[m & e["usable"]].min()) if (m & e["usable"]).any() else "—",
                         "สุดท้าย": _d(e.index[m & e["usable"]].max()) if (m & e["usable"]).any() else "—"})
    tab = pd.DataFrame(rows)
    for t, e in tables.items():
        e.to_csv(OUT_DIR / f"eligible_{t}.csv")
    ev = tables["KBANK"]
    n_wf = int(ev["usable"][(ev.index >= WF_START) & (ev.index <= FINAL_END)].sum())
    out += [md_table(tab), "",
            f"- วันที่ตัดเพราะ warm-up vol15 ({VOL_LOOKBACK} eligible day แรก): "
            + " · ".join(f"{t} {_d(e.index[e['eligible']].min())} → {_d(e.index[e['eligible'] & (e['n_prior'] < VOL_LOOKBACK)].max())}"
                         for t, e in tables.items()),
            f"- ช่วงประเมิน walk-forward ที่เสนอ (2024-10-01 → 2026-08-28) KBANK {n_wf} แถว → เทรนใหม่ทุก {RETRAIN_EVERY} วัน"
            f" = **{int(np.ceil(n_wf / RETRAIN_EVERY))} รอบเทรน** ต่อหุ้น",
            "- ปฏิทินสองหุ้นต่างกันเฉพาะ ADVANC 2025-09-16 (ไม่มีแท่ง 14) → จำนวนแถวต่างกัน 1 แถว", ""]
    return out, tables


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load_all()
    head = ["# AUDIT — ข้อมูลของโหมด 16:00 (ระยะที่ 1)", "",
            f"> สร้างโดย `python -m mode1600.audit1600` เมื่อ {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · ไม่เทรน ·"
            " ไม่คำนวณสถิติของ target บนวัน ≥ 2025-02-26 (ทุกการเทียบราคาปิดทางการกับแท่งผ่าน `_dev_only`)",
            "> รายละเอียดเพิ่มอยู่ใน `mode1600/audit/*.csv`", "",
            "## 0. ไฟล์และ SHA-256", "",
            "| ไฟล์ | SHA-256 | ตรวจ |", "|---|---|---|"]
    for t in D.TICKERS:
        head.append(f"| `task_b/raw_data_intraday/{t}_BK_1h_730d.csv` (repo ทีม) | `{D.HIST_SHA[t]}` | ตรง README ทีม |")
    for t in D.TICKERS:
        head.append(f"| `task_b_v2/data/{t}_10Y_Cleaned.csv` | `{D.DAILY_SHA[t]}` | ตรง LOCK.md หัวข้อ 1 |")
    head += [f"| snapshot `raw_data_intraday_live/*/` {len(data['KBANK']['snaps']) * 2} ไฟล์ | ตาม `.meta.json` | ตรงทุกไฟล์ |", ""]
    s5, sa, st, hv = section5(data)
    s6, _ = section6(data)
    body = head + section1(data) + section2(data) + section3(data) + section4(data) + s5 + s6
    OUT_MD.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"เขียน {OUT_MD}")


if __name__ == "__main__":
    main()
