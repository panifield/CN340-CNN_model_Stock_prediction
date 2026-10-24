"""
run1600.py — สคริปต์รันจริงของโหมด 16:00 (walk-forward + replay (i)/(ii) + ตัววัด + F3)
====================================================================================
ไฟล์นี้ทำอะไร: ตรวจ gate → โหลดข้อมูล (ตรวจ SHA) → walk-forward 2024-10-01 → 2026-08-28 → replay → เขียนผลที่
    `task_b_v2/results/mode1600/` (predictions · blocks · summary · f3 · replay · MANIFEST)
ทำไมต้องมี: ทางเดียวที่จะได้ผลจริง และผ่านการตรวจ gate เสมอ (`lock_gate.check`) — รันไม่ได้ถ้า LOCK_1600.md ยังไม่ถูก commit
    ทุกไฟล์ผลมีคอลัมน์ run_head_commit + run_time (MANIFEST.json มีด้วย)

รัน (จาก task_b_v2/ · หลังผู้ใช้ commit LOCK_1600.md และแจ้งแล้วเท่านั้น):
    PYTHONDONTWRITEBYTECODE=1 python -m mode1600.run1600 --jobs 6
"""

import argparse
import json
import sys
import time

import pandas as pd

from mode1600 import data1600 as D
from mode1600 import features1600 as FE
from mode1600 import lock_gate, locked1600 as L
from mode1600 import runner1600 as R

RESULTS_DIR = D.TASK_DIR / "results" / "mode1600"


def load_inputs():
    """ข้อมูลทุกชิ้นของการรันจริง (ทุกไฟล์ตรวจ SHA ใน data1600)"""
    out = {}
    for t in L.TICKERS:
        hist, daily, snaps = D.load_history(t), D.load_daily(t), D.snapshot_index(t)
        if not snaps["sha_ok"].all():
            raise ValueError(f"{t}: snapshot บางไฟล์ SHA ไม่ตรง meta")
        snap_bars = {p: D.load_snapshot(p) for p in snaps["path"]}
        hist_t = FE.history_table(t)
        rep_i = FE.replay_table_i(t, hist, daily, snaps, snap_bars)
        rep_ii = FE.replay_table_ii(t, hist, daily, snaps, snap_bars, final_table=rep_i)
        out[t] = {"prices": daily, "hist": hist_t, "replay": {"i": rep_i, "ii": rep_ii}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)

    gate = lock_gate.check()                      # 1) src/ + mode1600/ สะอาด · LOCK_1600 + LOCK.md commit แล้ว (M21)
    cfgs = L.track_configs()                      # 2) ช่อง track 1 ต้องไม่ว่าง
    if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()):
        raise FileExistsError(f"{RESULTS_DIR} มีผลอยู่แล้ว — รันครั้งเดียว ห้ามเขียนทับ")
    t0 = time.time()
    inputs = load_inputs()
    blocks = R.wf_blocks(inputs[L.TICKERS[0]]["prices"].index)      # ปฏิทินรายวัน (สองหุ้นชุดเดียวกัน · SOURCES.md)

    wf_preds, wf_info = R.run_walk_forward({t: (v["prices"], v["hist"]) for t, v in inputs.items()},
                                           cfgs, blocks, jobs=args.jobs)
    rp, rinfo = [], []
    for t, v in inputs.items():
        p, i = R.run_replay(t, v["prices"], v["hist"], v["replay"], cfgs)
        rp.append(p)
        rinfo.append(i)
    rp, rinfo = pd.concat(rp, ignore_index=True), pd.concat(rinfo, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def save(df, name, **kw):
        """ทุกไฟล์ผลมี HEAD commit + เวลารัน (M21 ค)"""
        out = df.copy()
        out.insert(0, "run_time", gate["run_time"])
        out.insert(0, "run_head_commit", gate["head_commit"])
        out.to_csv(RESULTS_DIR / name, index=False, **kw)

    save(blocks, "blocks.csv")
    save(wf_preds, "predictions_walk_forward.csv")
    save(wf_info, "train_info_walk_forward.csv")
    save(R.summarize(wf_preds), "summary_walk_forward.csv")
    save(R.f3(wf_preds), "f3_walk_forward.csv")
    save(rp, "predictions_replay.csv")                                  # มี bar_source ต่อวัน (log ที่มา)
    save(rinfo, "train_info_replay.csv")
    for way, g in rp.groupby("tag"):
        save(R.summarize(g), f"summary_{way}.csv")
    for t, v in inputs.items():
        for way, tab in v["replay"].items():
            save(tab.drop(columns=["y", "close", "y_final_base"], errors="ignore").rename_axis("date").reset_index(),
                 f"replay_inputs_{way}_{t}.csv")
    manifest = {"gate": gate, "seconds": round(time.time() - t0, 1), "jobs": args.jobs,
                "tracks": {k: c.to_row() for k, c in cfgs.items()},
                "data_sha256": {**{f"hist_{t}": D.HIST_SHA[t] for t in L.TICKERS},
                                **{f"daily_{t}": D.DAILY_SHA[t] for t in L.TICKERS}}}
    (RESULTS_DIR / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
    print(f"เสร็จใน {(time.time() - t0) / 60:.1f} นาที · เขียน {RESULTS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
