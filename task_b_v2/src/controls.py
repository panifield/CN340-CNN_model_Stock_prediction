"""
controls.py — ข้อสอบว่า pipeline "เรียนได้จริง" และ "ไม่รั่ว" (SPEC ข้อ 5.0 · Checkpoint C2)
=========================================================================================
ไฟล์นี้ทำอะไร — รันการทดลอง 2 อย่างที่รู้คำตอบล่วงหน้า

  positive control : เปลี่ยน target เป็น **std ของ 5 return ล่าสุดในหน้าต่างของภาพเอง**
      ค่านี้เป็นฟังก์ชันของ input โดยตรง (คำนวณจากตัวเลขที่อยู่ในภาพ) → โมเดลที่ถูกต้องต้องเรียนได้
      เกณฑ์: R² ≥ 0.3 · **ถ้าไม่ผ่าน แปลว่า pipeline มีบั๊ก ไม่ใช่ว่าตลาดทำนายยาก**

  negative control : สลับ y ใน train แบบสุ่ม (ทำลายความสัมพันธ์ระหว่าง input กับ target)
      โมเดลจึงไม่ควรทำนายอะไรได้ → relMAE ต้อง ≥ 0.99 (ไม่ดีกว่า Naive)
      เกณฑ์: relMAE ≥ 0.99 · **ถ้าต่ำกว่านี้ แปลว่ามีข้อมูลรั่วจากที่อื่น**

ทำไมต้องมี: ผลจริงของงานนี้คาดว่าจะใกล้ Naive มาก (ดู SPEC ข้อ 12) ซึ่งหน้าตาเหมือน "โมเดลพัง" ทุกประการ
สอง control นี้แยกสองกรณีออกจากกันก่อนจะไปเชื่อผลใด ๆ

*** ถ้า control ไม่ผ่าน : หยุด รายงานผู้ใช้ ห้ามปรับเกณฑ์หรือ hyperparameter เพื่อให้ผ่าน ***
"""

import numpy as np

from src import data, evaluate, train
from src.settings import POS_CONTROL_MIN_R2, NEG_CONTROL_MIN_RELMAE
from src.spectrogram import raw_channels

VOL_LAST_N = 5                 # positive control : std ของ 5 return ล่าสุด (SPEC ข้อ 5.0)


def volatility_target(ds, cfg, positions):
    """
    target ของ positive control : std ของ `VOL_LAST_N` return ล่าสุด **ที่อยู่ในหน้าต่างของภาพนั้นเอง**
    (แถว t−5 … t−1 ของ ret_close) — เป็นฟังก์ชันของ input ล้วน ๆ ไม่ได้ใช้ข้อมูลวัน t เลย
    """
    ret = raw_channels(ds.prices)["ret_close"].to_numpy(dtype=float)
    out = np.array([np.std(ret[p - VOL_LAST_N:p]) for p in np.asarray(positions, dtype=int)])
    if not np.isfinite(out).all():
        raise ValueError("target ของ positive control มีค่าไม่ finite")
    return out


def r2(y, pred):
    """R² เทียบกับค่าเฉลี่ยของ y เอง (positive control วัดว่า 'เรียนได้ไหม' ไม่ได้เทียบ Naive)"""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((pred - y) ** 2) / ss) if ss > 0 else np.nan


def run_positive(ds, cfg, fold, seeds):
    """เทรนให้ทำนายความผันผวนของหน้าต่างตัวเอง → คืน dict ผลของ evaluation นี้"""
    prepared = data.fold_inputs(ds, cfg, fold)
    y_over = {k: volatility_target(ds, cfg, prepared[f"pos_{k}"]) for k in ("fit", "val", "eval")}
    pred = train.run_fold(cfg, prepared, seeds=seeds, y_override=y_over)
    y_eval = y_over["eval"]
    return {"control": "positive", "ticker": ds.ticker, "fold": fold.k, "n": len(y_eval),
            "R2": r2(y_eval, pred), "corr": float(np.corrcoef(pred, y_eval)[0, 1]),
            "pred_std_ratio": float(np.std(pred) / np.std(y_eval)),
            "passed": bool(r2(y_eval, pred) >= POS_CONTROL_MIN_R2)}


def shuffled_targets(prepared, shuffle_seed):
    """
    สลับ y ของ train แบบสุ่ม (ส่วน fit และส่วน val แยกกัน) ด้วย `default_rng(shuffle_seed)` · ไม่แตะ y ของช่วงประเมิน
    ใช้ร่วมกันทั้ง negative control (ข้อ 5.0) และ placebo test (DECISIONS B21/B23) เพื่อการันตีว่าขั้นตอนสลับเหมือนกันทุกประการ
    """
    rng = np.random.default_rng(shuffle_seed)
    return {k: rng.permutation(prepared[f"y_{k}"]) for k in ("fit", "val")}


def run_negative(ds, cfg, fold, seeds, seed=0):
    """
    สลับ y ของ train (fit และ val) แบบสุ่ม แล้วประเมินบน y จริงของช่วงประเมิน
    การสลับทำ **ภายใน train เท่านั้น** ไม่แตะช่วงประเมิน (ไม่งั้นจะวัดอะไรไม่ได้เลย)
    """
    prepared = data.fold_inputs(ds, cfg, fold)
    y_over = shuffled_targets(prepared, seed)
    pred = train.run_fold(cfg, prepared, seeds=seeds, y_override=y_over)
    m = evaluate.metrics(prepared["y_eval"], pred, prepared["prev_close_eval"])
    return {"control": "negative", "ticker": ds.ticker, "fold": fold.k, "n": m["n"],
            "relMAE": m["relMAE"], "IC": m["IC"], "StdRatio": m["StdRatio"],
            "passed": bool(m["relMAE"] >= NEG_CONTROL_MIN_RELMAE)}


def verdict(rows, kind):
    """สรุปผลรวมของ control หนึ่งชนิด : ต้องผ่านทุก evaluation (เกณฑ์ประกาศไว้ใน settings.py)"""
    if kind == "positive":
        vals = [r["R2"] for r in rows]
        return {"control": kind, "n_eval": len(rows), "mean": float(np.mean(vals)),
                "min": float(np.min(vals)), "threshold": POS_CONTROL_MIN_R2,
                "n_passed": int(sum(r["passed"] for r in rows)),
                "passed": bool(all(r["passed"] for r in rows))}
    vals = [r["relMAE"] for r in rows]
    return {"control": kind, "n_eval": len(rows), "mean": float(np.mean(vals)),
            "min": float(np.min(vals)), "threshold": NEG_CONTROL_MIN_RELMAE,
            "n_passed": int(sum(r["passed"] for r in rows)),
            "passed": bool(all(r["passed"] for r in rows))}


def run_negative_unit(args):
    """
    หนึ่ง unit ของ negative control สำหรับรันขนาน (ต้องเป็นฟังก์ชันระดับบนสุดเพื่อให้ ProcessPoolExecutor pickle ได้)
    args = (cfg, ticker, fold_k, seeds, zone_end) · สร้างข้อมูลเองในแต่ละ process
    """
    cfg, ticker, fold_k, seeds, zone_end = args
    ds = data.build_dataset(ticker, zone_end)
    fold = next(f for f in data.make_folds(ds.dates) if f.k == fold_k)
    return run_negative(ds, cfg, fold, seeds)
