"""
time1600.py — วัดเวลาเทรน 1 รอบที่ใหญ่ที่สุดของ track หลัก ด้วย **y ที่สลับแล้ว** (ไม่ใช่ผล · ไม่มีตัววัดใด ๆ)
=====================================================================================================
ไฟล์นี้ทำอะไร: สร้างชุดเทรนขนาดเท่ารอบ replay (แถว labeled ≤ 2026-09-25 = รอบที่ใหญ่ที่สุด) ของหุ้นหนึ่งตัว
    แล้ว **สลับ y ของแถวเทรนแบบสุ่ม** (seed 0) · y ของวันที่ใช้เป็น eval ถูกแทนด้วย 0 (ไม่อ่านราคาปิดจริงของ replay)
    → เทรน track หลัก 10 seeds (+ control และ RF เพื่อประเมินเวลารวม) · พิมพ์เฉพาะเวลา · shape · จำนวน epoch
ทำไมต้องมี: ประเมินเวลารวมของการรันจริงก่อน commit LOCK_1600 โดยไม่เห็นผลใด ๆ ของ target จริง
    **ห้ามพิมพ์คำทำนาย / IC / relMAE** (ไฟล์นี้ไม่คำนวณสิ่งเหล่านั้นเลย) · ไม่เขียนไฟล์ใด

รัน (จาก task_b_v2/):  PYTHONDONTWRITEBYTECODE=1 python -m mode1600.time1600 [--ticker KBANK]
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from mode1600 import data1600 as D
from mode1600 import features1600 as FE
from mode1600 import locked1600 as L
from mode1600 import runner1600 as R
from src import data, train
from src.data import Fold


def shuffled_table(ticker, seed=0):
    """แถวเทรน ≤ REPLAY_TRAIN_END ที่ y ถูกสลับ + วัน replay ที่ y = 0 (ใช้แค่เป็นแถว eval ให้โค้ดเดินได้)"""
    hist = FE.history_table(ticker)
    tr = hist[hist.index <= pd.Timestamp(L.REPLAY_TRAIN_END)].copy()
    lab = tr["labeled"].to_numpy(dtype=bool)
    tr.loc[lab, "y"] = np.random.default_rng(seed).permutation(tr.loc[lab, "y"].to_numpy())
    # วัน replay: สร้าง feature อย่างเดียว (ไม่เรียก attach_target → ไม่อ่านราคาปิดของวัน replay) แล้วตั้ง y = 0
    snaps = D.snapshot_index(ticker)
    final = snaps[snaps["path"].map(lambda p: f"{p.parent.name}/{p.name}") == L.REPLAY_FINAL_SNAPSHOT.format(ticker=ticker)]
    daily = D.load_daily(ticker)
    ft = FE.feature_table(FE.splice(D.load_history(ticker), D.load_snapshot(final["path"].iloc[0])), daily)
    ev = ft.reindex(FE.replay_dates(daily))
    ev = ev[ev["usable"].eq(True)].copy()
    ev["y"], ev["close"], ev["labeled"] = 0.0, ev["C15"], True
    cols = [c for c in tr.columns if c in ev.columns]
    return pd.concat([tr[cols], ev[cols]]), ev.index


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="KBANK")
    args = ap.parse_args(argv)
    cfgs = L.track_configs()
    table, ev_dates = shuffled_table(args.ticker)
    prices = D.load_daily(args.ticker)

    t0 = time.time()
    ds16, F = R.make_dataset(args.ticker, prices, table, L.REPLAY_END)
    ev = ds16.dates.get_indexer(ev_dates)
    fold = Fold(k=0, eval_pos=ev, train_stop=int(ds16.dates.searchsorted(pd.Timestamp(L.REPLAY_START))))
    prepared = data.fold_inputs(ds16, cfgs[L.MAIN], fold, frac=L.EARLY_STOP_FRAC)
    extra = R.standardize_extra(F, prepared)
    t_prep = time.time() - t0
    print(f"[time1600] {args.ticker} · y สลับแล้ว (seed 0) · ไม่มีตัววัดใด ๆ")
    print(f"  shape: X_fit {prepared['X_fit'].shape} · X_val {prepared['X_val'].shape} · X_eval {prepared['X_eval'].shape}"
          f" · extra fit {extra['fit'].shape} · train {prepared['pos_fit'][0]}..{prepared['pos_val'][-1]}"
          f" ({len(prepared['pos_fit'])} fit + {len(prepared['pos_val'])} early-stop)")
    print(f"  เตรียมข้อมูล + ภาพ: {t_prep:.1f} วินาที")

    for name in L.NN_TRACKS:
        t1 = time.time()
        _, hists = train.run_fold(cfgs[name], prepared, seeds=L.SEEDS, extra=extra, return_hist=True)
        dt = time.time() - t1
        ep = [h["epochs"] for h in hists]
        print(f"  {name}: {len(L.SEEDS)} seeds {dt:.1f} วินาที ({dt / len(L.SEEDS):.2f} ต่อ seed) · epoch ที่หยุด {min(ep)}–{max(ep)}"
              f" (เฉลี่ย {np.mean(ep):.0f})")

    t1 = time.time()
    tr = np.concatenate([prepared["pos_fit"], prepared["pos_val"]])
    rf = TransformedTargetRegressor(regressor=R.get_regressors()["Random Forest"], transformer=StandardScaler())
    rf.fit(F[tr], ds16.y[tr])
    print(f"  {L.RF}: {time.time() - t1:.1f} วินาที")
    print(f"  รวม: {time.time() - t0:.1f} วินาที (1 process · 1 thread)")


if __name__ == "__main__":
    main()
