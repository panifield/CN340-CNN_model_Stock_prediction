"""
runner.py — ตัวขับการทดลอง: รันหลาย config แล้วเก็บผลลง results/ (SPEC ข้อ 4.2, 5.2–5.6, 0.4)
==========================================================================================
ไฟล์นี้ทำอะไร
    `run_configs(configs, stage_dir)` : สำหรับทุก config → ทุกหุ้น → ทุก fold
        เตรียมข้อมูล → เทรน (เฉลี่ย SEARCH_SEEDS) → วัดผล → เก็บคำทำนายรายวัน
    แล้วเขียน  evals.csv (1 แถวต่อ หุ้น×fold) · scores.csv (1 แถวต่อ config) · predictions.csv
    และ **ต่อท้าย** results/experiment_log.csv ทุก config ที่ลอง (รวมที่แพ้ — SPEC ข้อ 0.4)

ทำไมต้องมี
    * ทุก config ต้องผ่านเส้นทางเดียวกันเป๊ะ ๆ ไม่งั้นเทียบกันไม่ได้
    * cache ต่อ (config, หุ้น, fold) ทำให้หยุดกลางทางแล้วรันต่อได้ โดยไม่ต้องเทรนซ้ำ
    * `--jobs N` รันหลาย unit พร้อมกันแบบคนละ process (แต่ละ process ยัง 1 thread และ seed เดิม)
      → ผลเหมือนรันทีละตัวทุกบิต · ตรวจได้ด้วย `tests/test_runner_parallel_matches_serial.py`

หน่วยของงาน (unit) = (config, ticker, fold) · เป็นอิสระจากกันทั้งหมด จึงขนานได้ปลอดภัย
"""

import dataclasses
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import data, evaluate, paths, shrink, train, twostage
from src.configs import Config
from src.settings import SEARCH_SEEDS, TICKERS, VAL_END

EXPERIMENT_LOG = paths.RESULTS_DIR / "experiment_log.csv"


def unit_key(cfg, ticker, fold_k, seeds, zone_end, tag=""):
    """ลายนิ้วมือของหนึ่ง unit — ใช้เป็นชื่อไฟล์ cache · เปลี่ยน config แม้ field เดียวก็ได้คีย์ใหม่"""
    payload = json.dumps({"cfg": cfg.to_row(), "ticker": ticker, "fold": fold_k,
                          "seeds": list(seeds), "zone_end": zone_end, "tag": tag}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_unit(cfg, ticker, fold_k, *, seeds=SEARCH_SEEDS, zone_end=VAL_END, cache_dir=None, tag="", const_ic_zero=False):
    """
    เทรนและประเมินหนึ่ง (config, หุ้น, fold) · คืน (แถว metric, DataFrame คำทำนายรายวัน, history ของทุก seed)
    history = list ของ dict {train_loss, val_loss, epochs (epoch ที่หยุด), best_epoch, best_val} ต่อ seed — ไว้วาด loss curve
    ถ้ามี cache ของ unit นี้อยู่แล้วจะอ่านจาก cache (ผลเหมือนกันทุกบิตเพราะ seed ตายตัว)
    const_ic_zero : เฉพาะ Stage 6 (B39) — คำทำนายคงที่ → IC := 0 ทั้งระดับ evaluation และระดับ seed · ไม่แตะ IC_shrunk · คีย์ cache แยกจากแบบปิด
    """
    # ที่นี่ไม่เรียก assert_output_dir เพราะ `run_configs` ตรวจโฟลเดอร์ปลายทางให้แล้วตั้งแต่ต้น
    # (และ worker เป็นคนละ process จึงเห็นค่า paths.RESULTS_DIR ของจริงเสมอ ทำให้ test ที่ใช้ tmp_path พัง)
    key = unit_key(cfg, ticker, fold_k, seeds, zone_end, tag + ("|const_ic0" if const_ic_zero else ""))
    cache = None if cache_dir is None else cache_dir / f"{key}.pkl"
    if cache is not None and cache.exists():
        with open(cache, "rb") as f:
            row, preds, hists = pickle.load(f)
        row["from_cache"] = True
        return row, preds, hists

    t0 = time.time()
    ds = data.build_dataset(ticker, zone_end)
    fold = next(f for f in data.make_folds(ds.dates) if f.k == fold_k)
    prepared = data.fold_inputs(ds, cfg, fold)
    extra = train.standardized_extra(ds, prepared) if cfg.needs_features else None
    if cfg.train_mode == "two_stage":                                   # สองขั้น + cross-fitting (B32)
        pred, hists, seed_preds, (val_pred, y_val) = twostage.run_fold_two_stage(cfg, prepared, seeds=seeds, extra=extra)
    else:
        pred, hists, seed_preds, (val_pred, y_val) = train.run_fold(
            cfg, prepared, seeds=seeds, extra=extra, return_hist=True, return_seed_preds=True, return_val_pred=True)

    row = evaluate.evaluation_row("CNN-ANN", cfg.name, ticker, fold_k,
                                  prepared["y_eval"], pred, prepared["prev_close_eval"], const_ic_zero=const_ic_zero,
                                  seconds=round(time.time() - t0, 1), n_seeds=len(seeds),
                                  image_scale=prepared["transform"].scale,
                                  n_train_fit=len(prepared["pos_fit"]), n_train_val=len(prepared["pos_val"]),
                                  epochs_mean=float(np.mean([h["epochs"] for h in hists])),
                                  best_epoch_mean=float(np.mean([h["best_epoch"] for h in hists])))
    # IC / relMAE ของแต่ละ seed เดี่ยว ๆ — ไว้ประมาณ noise ของ seed (DECISIONS B17)
    for k, sp in zip(seeds, seed_preds):
        m = evaluate.metrics(prepared["y_eval"], sp, prepared["prev_close_eval"], const_ic_zero=const_ic_zero)
        row[f"ic_s{k}"] = m["IC"]
        row[f"relMAE_s{k}"] = m["relMAE"]
        if const_ic_zero:
            row[f"collapsed_s{k}"] = m["collapsed"]
    # shrinkage (B18): เลือก c จากส่วน early-stop เพื่อลด MAE · รายงานเพิ่ม ไม่ใช้เลือก config (IC ไม่เปลี่ยนเมื่อคูณ c > 0)
    c = shrink.choose_c(y_val, val_pred)
    pred_shrunk = shrink.apply(pred, c)
    ms = evaluate.metrics(prepared["y_eval"], pred_shrunk, prepared["prev_close_eval"])
    row.update({"c_chosen": c, "relMAE_shrunk": ms["relMAE"], "relMAE_baht_shrunk": ms["relMAE_baht"],
                "MAE_baht_shrunk": ms["MAE_baht"], "StdRatio_shrunk": ms["StdRatio"], "IC_shrunk": ms["IC"]})
    row["from_cache"] = False
    preds = evaluate.predictions_frame(prepared["dates_eval"], ticker, fold_k,
                                       prepared["y_eval"], pred, prepared["prev_close_eval"],
                                       model="CNN-ANN", config=cfg.name)
    preds["pred_shrunk"] = pred_shrunk
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump((row, preds, hists), f)
    return row, preds, hists


def _worker(args):
    """ตัวห่อสำหรับ ProcessPoolExecutor (ต้องเป็นฟังก์ชันระดับบนสุดเพื่อให้ pickle ได้)"""
    cfg, ticker, fold_k, seeds, zone_end, cache_dir, tag, *rest = args
    return run_unit(cfg, ticker, fold_k, seeds=seeds, zone_end=zone_end, cache_dir=cache_dir, tag=tag,
                    const_ic_zero=bool(rest[0]) if rest else False)


def run_configs(configs, stage_dir, *, seeds=SEARCH_SEEDS, zone_end=VAL_END, jobs=1, tag="", verbose=True, units=None,
                const_ic_zero=False):
    """
    รันทุก config × ทุกหุ้น × ทุก fold · เขียนผลลง `stage_dir` แล้วคืน (evals, scores, predictions)

    jobs > 1 : แบ่ง unit ไปรันคนละ process (แต่ละ process ตั้ง thread = 1 เหมือนเดิม) — ผลเท่ากับรันทีละตัว
    units    : (ไม่บังคับ) list ของ (ticker, fold) ที่จะรัน · None = ครบทุกหุ้น × ทุก fold (ใช้จริงทุก Stage)
               ระบุเฉพาะตอน dry-run ตรวจสายงานของ Stage 6 (B38) — คะแนนรวมจะคิดจากจำนวน evaluation ที่รันจริง
    const_ic_zero : เปิดเฉพาะ Stage 6 (B39) · ค่าเริ่มต้นปิด = ทุก Stage ก่อนหน้าเหมือนเดิม
    """
    stage = paths.assert_output_dir(stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    cache_dir = stage / "cache"

    folds_k = list(range(1, len(data.make_folds(data.build_dataset(TICKERS[0], zone_end).dates)) + 1))
    pairs = [(t, k) for t in TICKERS for k in folds_k] if units is None else list(units)
    tasks = [(cfg, t, k, tuple(seeds), zone_end, cache_dir, tag, const_ic_zero) for cfg in configs for t, k in pairs]
    if verbose:
        print(f"[runner] {len(configs)} config × {len(pairs)} evaluation "
              f"= {len(tasks)} unit · seeds {tuple(seeds)} · jobs {jobs}")

    results = []
    t0 = time.time()
    if jobs > 1:
        env = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")      # กันไม่ให้แต่ละ process แย่ง CPU กันเอง
        os.environ.update(env)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, out in enumerate(pool.map(_worker, tasks), 1):
                results.append(out)
                if verbose:
                    _progress(i, len(tasks), out[0], t0)
    else:
        for i, args in enumerate(tasks, 1):
            out = _worker(args)
            results.append(out)
            if verbose:
                _progress(i, len(tasks), out[0], t0)

    evals = pd.DataFrame([r for r, _, _ in results])
    preds = pd.concat([p for _, p, _ in results], ignore_index=True)
    histories = {f"{r['config']}|{r['ticker']}|fold{r['fold']}": h for r, _, h in results}
    scores = pd.DataFrame([{"config": cfg.name, **cfg.to_row(),
                            **evaluate.score_config(evals[evals["config"] == cfg.name].to_dict("records"),
                                                    n_expected=len(pairs))}
                           for cfg in configs]).sort_values("mean_IC", ascending=False)      # เกณฑ์เลือก = IC (B16)

    evals.to_csv(stage / "evals.csv", index=False)
    scores.to_csv(stage / "scores.csv", index=False)
    preds.to_csv(stage / "predictions.csv", index=False)
    with open(stage / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f)                                   # loss curve ของทุก unit ทุก seed
    append_experiment_log(scores, stage.name)
    if verbose:
        print(f"[runner] เสร็จใน {(time.time() - t0) / 60:.1f} นาที · เขียน {stage}")
    return evals, scores, preds


def _progress(i, n, row, t0):
    done = time.time() - t0
    rate = done / i
    print(f"  [{i:3d}/{n}] {row['config']:14s} {row['ticker']:6s} fold {row['fold']} "
          f"relMAE {row['relMAE']:.4f} StdRatio {row['StdRatio']:.3f} IC {row['IC']:+.3f}"
          f"{'  (cache)' if row.get('from_cache') else ''} · เหลือ ~{(n - i) * rate / 60:.0f} นาที")


def append_experiment_log(scores, stage_name):
    """ต่อท้าย results/experiment_log.csv — ทุก config ที่เคยลอง รวมที่แพ้ (SPEC ข้อ 0.4)"""
    paths.assert_output_dir(EXPERIMENT_LOG.parent)
    df = scores.copy()
    df.insert(0, "stage", stage_name)
    df.insert(0, "timestamp", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    if EXPERIMENT_LOG.exists():                                  # ต่อท้ายแบบรวมคอลัมน์ (รอบหลังอาจมีคอลัมน์ต่าง)
        df = pd.concat([pd.read_csv(EXPERIMENT_LOG), df], ignore_index=True)
    # config เดียวกัน (ทุก field ยกเว้นชื่อ) ที่ถูกรันซ้ำ — จาก cache หรือรันใหม่ใน stage อื่นเพื่อวัดเพิ่ม — ไม่ใช่ config ใหม่
    # → เก็บแถวแรกไว้แถวเดียว · จำนวนแถวใน log = จำนวน config ที่ต่างกันจริง ซึ่งต้องรายงานได้ (SPEC ข้อ 0.4 และ LOCK.md ข้อ 4.4)
    key = [c for c in (f.name for f in dataclasses.fields(Config)) if c != "name" and c in df.columns]
    # เทียบเป็นข้อความ: CSV อ่านกลับ "32" เป็นเลข 32 แต่ตอนสร้างเป็นข้อความ → เทียบตรง ๆ จะไม่เท่ากัน
    df = df[~df[key].astype(str).duplicated(keep="first")]
    df.to_csv(EXPERIMENT_LOG, index=False, encoding="utf-8")
    return EXPERIMENT_LOG
