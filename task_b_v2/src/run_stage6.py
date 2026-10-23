"""
run_stage6.py — Stage 6 (HPO ของ track 1) · ทำก่อน LOCK (DECISIONS B38)
=========================================================================
ไฟล์นี้ทำอะไร: คำสั่งย่อยตามลำดับงาน
    design    เขียน results/stage6_hpo/configs.csv (25 config + image_shape + จำนวนพารามิเตอร์) และ design_log.json
              · ถ้าไฟล์มีอยู่แล้วจะ "ตรวจว่าเหมือนเดิมทุกตัวอักษร" ไม่เขียนทับ (รายการห้ามเปลี่ยนหลังผู้ใช้ commit)
    checks    A8 (a)(b)(c): รูปภาพ/cache key/สถิติ normalize ต่อ config · น้ำหนักวันสุดท้ายของ Hann ทุกคู่ (nperseg, hop)
    estimate  A8 (e): วัดเวลาจริงของ unit ที่ช้าที่สุด + ประเมินเวลารวม (ค้นหา + placebo + ยืนยัน) → ตัดสิน 5 หรือ 3 รอบ placebo
    dryrun    A8 (f): 1 ผู้สมัคร + base บน 1 evaluation ครบสายงาน (ค้นหา → placebo 1 รอบ → ยืนยัน → รายงาน) ลง results/_smoke_stage6/
    run       ส่วน B: ค้นหา 25 config → placebo ทุกรอบ → ยืนยันด้วย seed ใหม่ → รายงาน (ปฏิเสธถ้าไฟล์ที่ประกาศยังไม่ commit)
    report    สร้าง stage6_summary.md / stage6_decision.json / รูป ใหม่จากผลที่มีอยู่ (ไม่เทรน)

B39: ทุกส่วนของ `run`/`dryrun` เปิด const_ic_zero (คำทำนายคงที่ → IC := 0) · รายงานจำนวน collapse ในข้อ 7 ของ stage6_summary.md

*** สคริปต์นี้รายงานเท่านั้น ไม่แก้ draft_lock.py / LOCK.md ตามผลเอง (ผู้ใช้สั่ง) · ห้ามแตะช่วง final/replay (zone_end = VAL_END เสมอ) ***

รัน:  python -m src.run_stage6 design
      python -m src.run_stage6 checks
      python -m src.run_stage6 estimate
      python -m src.run_stage6 dryrun --jobs 6
      python -m src.run_stage6 run --jobs 6          (หลังผู้ใช้ commit ประกาศแล้วเท่านั้น)
"""

import argparse
import json
import subprocess
import time
from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                      # ไม่เปิดหน้าต่าง (รันบน terminal)
import matplotlib.pyplot as plt            # noqa: E402

from src import data, hpo, models, paths, placebo, runner, selection, train      # noqa: E402
from src import figures                                                          # noqa: E402,F401  (ตั้งฟอนต์ไทยของ matplotlib)
from src.check_data import md_table                                              # noqa: E402
from src.configs import BASE                                                     # noqa: E402
from src.run_stage1 import fmt                                                   # noqa: E402
from src.settings import SEARCH_SEEDS, TICKERS, VAL_END                          # noqa: E402
from src.spectrogram import window_weights                                       # noqa: E402

R = paths.RESULTS_DIR
OUT = R / "stage6_hpo"
SMOKE = R / "_smoke_stage6"
STAGE3_EVALS = R / "stage3_input" / "evals.csv"
B22_FILE = R / "stage1_placebo" / "stage1_final_decision.json"
EFFICIENCY = 4.6                        # ความเร็วขึ้นของ jobs 6 ที่วัดได้จริงใน C3 (6.0 → 1.3 นาที) · ใช้ค่าเดียวกับ Stage 4
TIME_LIMIT_H = 7.0                      # A5: ประเมินเวลารวมเกินนี้ → ลด placebo เหลือ 3 รอบ
FOLD3_ROWS = {1: 1144, 2: 1410, 3: 1676}  # จำนวนแถว train (fit+val) ต่อ fold ของ dev — ไว้ถ่วงน้ำหนักเวลา (ตรวจกับข้อมูลจริงใน estimate)
BASE_CHECK_COLS = ["IC", "IC_spearman", "relMAE", "relMAE_baht", "StdRatio", "DirAcc", "MAE_baht",
                   "ic_s0", "ic_s1", "ic_s2", "relMAE_s0", "relMAE_s1", "relMAE_s2",
                   "c_chosen", "relMAE_shrunk", "IC_shrunk"]
ANNOUNCED_FILES = ["src/hpo.py", "src/run_stage6.py", "src/runner.py", "src/placebo.py", "src/evaluate.py", "tests/test_stage6.py",
                   "DECISIONS.md", "results/stage6_hpo/configs.csv", "results/stage6_hpo/design_log.json"]


class StopRun(RuntimeError):
    """B3 : error/NaN/base ไม่ตรงของเดิม → หยุดและรายงาน ห้ามแก้แล้วรันต่อเงียบ ๆ"""


def b22_floor():
    v = float(json.loads(B22_FILE.read_text(encoding="utf-8"))["b22_threshold_stage2_to_5"])
    if v != hpo.B22_FLOOR:
        raise StopRun(f"เกณฑ์ B22 ในไฟล์ ({v}) ไม่ตรงกับค่าที่ประกาศใน hpo.py ({hpo.B22_FLOOR})")
    return v


def stamp():
    return time.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------
# design
# ---------------------------------------------------------------
def design_log_dict(configs, info):
    return {"design_seed": hpo.DESIGN_SEED, **info, "n_candidates": hpo.N_CANDIDATES, "n_configs_total": len(configs),
            "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in hpo.GRID.items()},
            "level_counts": hpo.level_counts(configs),
            "placebo_shuffles": list(hpo.PLACEBO_SHUFFLES), "confirm_seeds": list(hpo.CONFIRM_SEEDS),
            "search_seeds": list(SEARCH_SEEDS), "b22_floor": hpo.B22_FLOOR,
            "min_last_day_weight": hpo.MIN_LAST_DAY_WEIGHT,
            "fixed": {"channels": BASE.channels, "window": BASE.window, "spec_type": BASE.spec_type,
                      "normalize": BASE.normalize, "loss": BASE.loss, "huber_delta": BASE.huber_delta,
                      "train_mode": BASE.train_mode, "ann_input": BASE.ann_input, "max_epochs": BASE.max_epochs,
                      "patience": BASE.patience, "cnn_feat": BASE.cnn_feat}}


def design_files(out=OUT):
    configs, info = hpo.design_configs()
    csv = hpo.configs_table(configs).to_csv(index=False, lineterminator="\n")
    log = json.dumps(design_log_dict(configs, info), ensure_ascii=False, indent=2)
    return configs, csv, log


def write_design(out=OUT):
    """เขียน configs.csv / design_log.json ถ้ายังไม่มี · ถ้ามีแล้วต้องเหมือนเดิมทุกตัวอักษร ไม่งั้นหยุด (ห้ามเปลี่ยนหลังประกาศ)"""
    paths.assert_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    configs, csv, log = design_files(out)
    for name, text in (("configs.csv", csv), ("design_log.json", log)):
        f = out / name
        if f.exists():
            if f.read_text(encoding="utf-8") != text:
                raise StopRun(f"{f} ไม่ตรงกับที่โค้ดสร้างได้ — รายการ config ห้ามเปลี่ยนหลังประกาศ (หยุดและรายงาน)")
        else:
            f.write_text(text, encoding="utf-8", newline="\n")
    return configs


def load_design(out=OUT):
    """config ทั้ง 25 ตัวจากโค้ด หลังตรวจว่าตรงกับ configs.csv ที่ commit ไว้ทุกตัวอักษร"""
    configs, csv, log = design_files(out)
    f = out / "configs.csv"
    if not f.exists() or f.read_text(encoding="utf-8") != csv:
        raise StopRun(f"{f} ไม่มีหรือไม่ตรงกับโค้ด — รัน `python -m src.run_stage6 design` ก่อน (และห้ามแก้รายการหลังประกาศ)")
    return configs


def cmd_design(args):
    configs = write_design()
    t = pd.read_csv(OUT / "configs.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t.iloc[:, :13].to_string(index=False))
    log = json.loads((OUT / "design_log.json").read_text(encoding="utf-8"))
    print(f"\nseed {log['design_seed']} · สุ่มทั้งหมด {log['attempts']} ครั้ง (สุ่มใหม่ {log['redraws']} ครั้ง) · ใช้ seed {log['seed_used']}")
    print("ระดับของแต่ละพารามิเตอร์ใน 24 ผู้สมัคร:", json.dumps(log["level_counts"], ensure_ascii=False))
    print(f"เขียน {OUT / 'configs.csv'} · {OUT / 'design_log.json'}")


# ---------------------------------------------------------------
# checks (A8 a, b, c)
# ---------------------------------------------------------------
def perturb_future(prices, pos, seed=7):
    """คูณ OHLCV ของแถว pos เป็นต้นไปด้วยตัวเลขสุ่ม (เหมือน tests/test_leak.py) — ไว้ตรวจว่าสถิติที่ fit ไม่ขยับ"""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    block = out.iloc[pos:]
    out.iloc[pos:] = block.to_numpy() * rng.uniform(1.05, 1.60, block.shape)
    return out


def check_window_weights(configs):
    """A8 (c): น้ำหนักของวันสุดท้ายของหน้าต่าง (เทียบค่าสูงสุด) ของทุกคู่ (nperseg, hop) ที่ใช้"""
    rows, seen = [], set()
    for c in configs:
        k = (c.nperseg, c.hop)
        if k in seen:
            continue
        seen.add(k)
        w = window_weights(c)
        rows.append({"nperseg": c.nperseg, "hop": c.hop, "image_shape": "x".join(map(str, c.image_shape)),
                     "last_day_weight": float(w[-1]), "prev_day_weight": float(w[-2]), "min_weight": float(w.min()),
                     "ok": bool(w[-1] >= hpo.MIN_LAST_DAY_WEIGHT)})
    return pd.DataFrame(rows).sort_values(["nperseg", "hop"]).reset_index(drop=True)


def check_images_and_keys(configs, ticker="KBANK"):
    """
    A8 (a): (i) ภาพที่ build ได้มี shape ตรง cfg.image_shape และ (nperseg, hop) ต่างกันได้ shape ต่างกันจริง
            (ii) unit_key ของทั้ง 25 config ไม่ซ้ำกัน และเปลี่ยนเมื่อเปลี่ยน nperseg/hop อย่างเดียว (แม้ชื่อเหมือนกัน)
    """
    ds = data.build_dataset(ticker, VAL_END)
    pos = np.arange(len(ds) - 300, len(ds) - 297)
    shapes = {}
    for c in configs:
        X, has = ds.raw_images(c, pos)
        assert has.all() and X.shape == (len(pos),) + c.image_shape and np.isfinite(X).all(), f"{c.name}: ภาพผิด shape/มี NaN"
        shapes.setdefault((c.nperseg, c.hop), set()).add(tuple(X.shape[1:]))
    one_shape_per_pair = all(len(v) == 1 for v in shapes.values())
    distinct = len({next(iter(v)) for v in shapes.values()}) == len(shapes)
    keys = [runner.unit_key(c, ticker, 1, SEARCH_SEEDS, VAL_END) for c in configs]
    keys_unique = len(set(keys)) == len(keys)
    renamed = [runner.unit_key(replace(BASE, nperseg=n, hop=h), ticker, 1, SEARCH_SEEDS, VAL_END)
               for n in (8, 16, 32) for h in (1, 2, 4)]                  # ชื่อ "base" เหมือนกันทุกตัว
    same_name_keys_distinct = len(set(renamed)) == len(renamed)
    return {"n_pairs": len(shapes), "one_shape_per_pair": one_shape_per_pair, "pairs_have_distinct_shapes": distinct,
            "unit_keys_unique_25": keys_unique, "unit_keys_distinct_when_only_nperseg_hop_change": same_name_keys_distinct,
            "ok": bool(one_shape_per_pair and distinct and keys_unique and same_name_keys_distinct)}


def check_cache_isolation(tmp_dir):
    """
    A8 (a) แบบรันจริง: สอง config ชื่อเดียวกัน ต่างกันแค่ nperseg/hop ใช้ cache_dir เดียวกัน (epoch น้อยมาก)
    → ตัวที่สองต้องไม่ดึง cache ของตัวแรก และ image shape/สเกลภาพต้องต่างกัน · รันซ้ำตัวแรกต้องได้ cache
    """
    a = replace(BASE, name="cachetest", nperseg=8, hop=4, max_epochs=2, patience=2)
    b = replace(BASE, name="cachetest", nperseg=32, hop=1, max_epochs=2, patience=2)
    ra, pa, _ = runner.run_unit(a, "KBANK", 1, seeds=(0,), cache_dir=tmp_dir)
    rb, pb, _ = runner.run_unit(b, "KBANK", 1, seeds=(0,), cache_dir=tmp_dir)
    ra2, _, _ = runner.run_unit(a, "KBANK", 1, seeds=(0,), cache_dir=tmp_dir)
    ok = (not ra["from_cache"]) and (not rb["from_cache"]) and ra2["from_cache"] and ra["IC"] == ra2["IC"] \
        and ra["image_scale"] != rb["image_scale"] and not np.array_equal(pa["pred"].to_numpy(), pb["pred"].to_numpy())
    return {"first_from_cache": ra["from_cache"], "second_from_cache": rb["from_cache"], "rerun_first_from_cache": ra2["from_cache"],
            "image_scale_a": ra["image_scale"], "image_scale_b": rb["image_scale"], "ok": bool(ok)}


def check_fit_statistics(configs, cases=(("KBANK", 3), ("ADVANC", 1))):
    """
    A8 (b): สเกลภาพและสถิติ normalize ของทุก config ต้อง (i) เท่ากับที่คำนวณซ้ำอิสระจากแถว fit (85% แรกของ train) เท่านั้น
    (ii) ไม่ใช่สถิติของ train ทั้งก้อน (iii) ไม่ขยับเมื่อราคาตั้งแต่วันแรกของช่วงประเมินถูกเปลี่ยน (fold เดียวกัน)
    """
    rows = []
    for ticker, k in cases:
        ds = data.build_dataset(ticker, VAL_END)
        fold = next(f for f in data.make_folds(ds.dates) if f.k == k)
        alt = data.dataset_from_prices(ticker, perturb_future(ds.prices, int(fold.eval_pos[0])), VAL_END)
        for c in configs:
            tf = data.fold_inputs(ds, c, fold)["transform"]
            tr = fold.train_pos(ds.usable(c))
            fit_m, _ = data.early_stop_split(len(tr), 0.15)
            raw, has = ds.raw_images(c, tr)
            assert has.all()
            scale = float(np.median(np.abs(raw[fit_m])))
            comp = np.log1p(raw[fit_m] / scale)
            mean_ok = np.allclose(np.asarray(tf.mean), comp.mean(axis=(0, 3), keepdims=True), rtol=1e-5, atol=1e-6)
            std_ok = np.allclose(np.asarray(tf.std), np.maximum(comp.std(axis=(0, 3), keepdims=True), 1e-8), rtol=1e-5, atol=1e-6)
            not_all_train = tf.scale != float(np.median(np.abs(raw)))
            tf2 = data.fold_inputs(alt, c, fold)["transform"]
            invariant = tf.scale == tf2.scale and np.array_equal(np.asarray(tf.mean), np.asarray(tf2.mean)) \
                and np.array_equal(np.asarray(tf.std), np.asarray(tf2.std))
            rows.append({"config": c.name, "ticker": ticker, "fold": k, "scale": tf.scale, "scale_matches_fit_median": bool(tf.scale == scale),
                         "mean_ok": bool(mean_ok), "std_ok": bool(std_ok), "not_whole_train": bool(not_all_train),
                         "invariant_to_future_prices": bool(invariant)})
    df = pd.DataFrame(rows)
    ok_cols = ["scale_matches_fit_median", "mean_ok", "std_ok", "not_whole_train", "invariant_to_future_prices"]
    return df, bool(df[ok_cols].all().all())


def check_shuffle_identical_across_configs(configs, ticker="KBANK", fold_k=3, shuffle_seed=100):
    """A5: config ทุกตัวในรอบเดียวกันต้องได้ y ที่สลับเหมือนกัน (window/ช่องเหมือนกัน → แถว train เท่ากัน → permutation เดียวกัน)"""
    from src import controls
    ds = data.build_dataset(ticker, VAL_END)
    fold = next(f for f in data.make_folds(ds.dates) if f.k == fold_k)
    ref = None
    for c in configs:
        prep = data.fold_inputs(ds, c, fold)
        y = controls.shuffled_targets(prep, shuffle_seed)
        if ref is None:
            ref = y
            assert not np.array_equal(y["fit"], prep["y_fit"]), "การสลับต้องเปลี่ยนลำดับ y จริง"
        elif not (np.array_equal(ref["fit"], y["fit"]) and np.array_equal(ref["val"], y["val"])):
            return False
        # y ของ test เป็นของจริง: shuffled_targets ไม่แตะ y_eval
        assert np.array_equal(prep["y_eval"], ds.y[fold.eval_pos])
    return True


def cmd_checks(args):
    import tempfile
    configs = load_design()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ww = check_window_weights(configs)
    ww.to_csv(OUT / "a8_window_weights.csv", index=False)
    print("A8 (c) น้ำหนักวันสุดท้ายของหน้าต่าง (เทียบค่าสูงสุด · ต้อง ≥ %.2f):" % hpo.MIN_LAST_DAY_WEIGHT)
    print(ww.to_string(index=False))
    img = check_images_and_keys(configs)
    print("\nA8 (a) ภาพ/คีย์:", img)
    with tempfile.TemporaryDirectory() as tmp:
        cache = check_cache_isolation(__import__("pathlib").Path(tmp))
    print("A8 (a) cache แบบรันจริง:", cache)
    fit_df, fit_ok = check_fit_statistics(configs)
    fit_df.to_csv(OUT / "a8_fit_statistics.csv", index=False)
    print(f"\nA8 (b) สถิติ fit ต่อ config/หุ้น/fold: {len(fit_df)} กรณี · ผ่านทั้งหมด = {fit_ok}")
    shuf = check_shuffle_identical_across_configs(configs)
    print(f"A5 y ที่สลับเหมือนกันทุก config ในรอบเดียวกัน = {shuf}")
    res = {"checked_at": stamp(), "a8c_window_weights_ok": bool(ww["ok"].all()), "a8c_min_last_day_weight": float(ww["last_day_weight"].min()),
           "a8c_max_last_day_weight": float(ww["last_day_weight"].max()), "a8a_images_and_keys": img, "a8a_cache_isolation": cache,
           "a8b_fit_statistics_ok": fit_ok, "a8b_n_cases": int(len(fit_df)), "a5_shuffle_identical_across_configs": shuf,
           "seconds": round(time.time() - t0, 1)}
    res["all_ok"] = bool(res["a8c_window_weights_ok"] and img["ok"] and cache["ok"] and fit_ok and shuf)
    (OUT / "a8_checks.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nทั้งหมดผ่าน = {res['all_ok']} · เขียน {OUT / 'a8_checks.json'}")
    if not res["all_ok"]:
        raise StopRun("A8 ไม่ผ่านบางข้อ — หยุดและรายงาน")


# ---------------------------------------------------------------
# estimate (A8 e)
# ---------------------------------------------------------------
def time_epoch(cfg, ds, fold, epochs=3):
    """เวลาเตรียมภาพ (วินาที) และเวลาต่อ epoch (วินาที · 1 seed · รวมการวัด loss ของ fit/val ทุก epoch) ของ config นี้"""
    t0 = time.time()
    prepared = data.fold_inputs(ds, cfg, fold)
    t_prep = time.time() - t0
    t0 = time.time()
    train.run_fold(replace(cfg, max_epochs=epochs, patience=epochs), prepared, seeds=(0,))
    return t_prep, (time.time() - t0) / epochs


def cmd_estimate(args):
    configs = load_design()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = data.build_dataset("KBANK", VAL_END)
    fold3 = data.make_folds(ds.dates)[2]
    base_e = float(pd.read_csv(STAGE3_EVALS).query("config == 'base'")["epochs_mean"].mean())     # epoch ที่หยุดจริงของ base (เฉลี่ย)
    rows = []
    for c in configs:
        t_prep, t_ep = time_epoch(c, ds, fold3)
        rows.append({"name": c.name, "image_shape": "x".join(map(str, c.image_shape)), "batch": c.batch, "n_params": hpo.n_params(c),
                     "t_prep_s": t_prep, "t_epoch_s": t_ep})
        print(f"  {c.name:6s} ภาพ {rows[-1]['image_shape']:9s} batch {c.batch:2d} · เตรียมภาพ {t_prep:5.1f}s · {t_ep:6.2f}s/epoch")
    df = pd.DataFrame(rows)
    worst_grid = replace(BASE, name="grid_worst", nperseg=32, hop=1, filters=(16, 32), batch=16)
    tp, te = time_epoch(worst_grid, ds, fold3)
    print(f"  {'grid_worst':10s} (nperseg 32, hop 1, filters 16-32, batch 16) · เตรียมภาพ {tp:.1f}s · {te:.2f}s/epoch")

    # ---- unit จริงที่ช้าที่สุด (A8 e): fold 3 · KBANK · 1 seed · เทรนจนหยุดจริง ----
    df["cost_unit_s"] = df["t_prep_s"] + 3 * base_e * df["t_epoch_s"]
    slow = configs[int(df["cost_unit_s"].idxmax())]
    real = {}
    for label, cfg in (("base", BASE), ("slowest", slow)):
        t0 = time.time()
        row, _, hists = runner.run_unit(cfg, "KBANK", 3, seeds=(0,), cache_dir=None)
        real[label] = {"config": cfg.name, "seconds": time.time() - t0, "epochs": hists[0]["epochs"], "best_epoch": hists[0]["best_epoch"],
                       "hit_cap": not hists[0]["stopped_early"]}
        print(f"[unit จริง · KBANK fold 3 · 1 seed] {label} ({cfg.name}): {real[label]['seconds']:.1f}s · หยุดที่ epoch {hists[0]['epochs']} (ดีสุด {hists[0]['best_epoch']})")

    # ---- ประเมินเวลารวม ----
    weight = 2 * sum(FOLD3_ROWS[k] / FOLD3_ROWS[3] for k in (1, 2, 3))            # 6 evaluation เทียบเป็น "fold 3 · 1 หุ้น"
    df["serial_min_search"] = weight * (df["t_prep_s"] + 3 * base_e * df["t_epoch_s"]) / 60
    # calibrate: epoch จริงของ 2 unit ที่วัด เทียบกับ base_e ที่ใช้ประมาณ
    def serial_total(e_mult):
        return float((weight * (df["t_prep_s"] + 3 * base_e * e_mult * df["t_epoch_s"]) / 60).sum())
    search_c, search_p = serial_total(1.0), serial_total(1.5)
    worst_cfg_serial = float(weight * (df["t_prep_s"].max() + 3 * base_e * 1.5 * df["t_epoch_s"].max()) / 60)
    confirm_c = float(2 * weight * (df["t_prep_s"].max() + 3 * base_e * df["t_epoch_s"].max()) / 60)        # ผู้สมัคร (ช้าสุดไว้ก่อน) + base ≈ 2 × config ช้าสุด (ขอบบน)
    confirm_p = float(2 * worst_cfg_serial)

    def total_h(rounds, search, confirm):
        return (search * (1 + rounds) + confirm) / EFFICIENCY / 60
    scen = {f"{name}_{r}rounds": total_h(r, s, cf) for name, s, cf in (("central", search_c, confirm_c), ("pessimistic_e1.5", search_p, confirm_p))
            for r in (5, 3)}
    # ขอบบนสุด: ทุก config ช้าเท่า unit ที่ช้าที่สุดที่วัดจริง (ไม่ได้ถ่วงน้ำหนัก fold) — 25 config × 6 eval × 3 seeds
    slow_unit_s = real["slowest"]["seconds"]
    upper_serial_search_min = 25 * 6 * 3 * slow_unit_s / 60
    scen["upper_bound_all_as_slow_as_slowest_5rounds"] = (upper_serial_search_min * 6 + 2 * 6 * 3 * slow_unit_s / 60) / EFFICIENCY / 60
    rounds = 5 if scen["central_5rounds"] <= TIME_LIMIT_H else 3          # A5: ตัดสินจากค่ากลาง (ค่าอื่นรายงานเป็นข้อมูล)
    info = {"measured_at": stamp(), "assumed_epochs_per_run": base_e, "efficiency_jobs6": EFFICIENCY,
            "unit_weight_vs_fold3_unit": weight, "real_units": real,
            "grid_worst_case": {"t_prep_s": tp, "t_epoch_s": te},
            "serial_minutes": {"search_central": search_c, "search_pessimistic_e1.5": search_p, "confirm_central": confirm_c, "confirm_pessimistic": confirm_p},
            "wall_hours": scen, "limit_hours": TIME_LIMIT_H,
            "rule": "central ≤ 7 ชม. → 5 รอบ · เกิน → 3 รอบ (A5) · pessimistic = epoch ที่หยุดจริง ×1.5 เทียบ base",
            "placebo_rounds_decided": rounds,
            "per_config": df.round(3).to_dict("records")}
    (OUT / "time_estimate.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nเวลาประเมิน (ชม. · jobs 6 · ประสิทธิภาพ 4.6):")
    for k, v in scen.items():
        print(f"  {k:48s} {v:6.2f} ชม.")
    print(f"→ จำนวนรอบ placebo ที่ใช้: {rounds} (เกณฑ์ {TIME_LIMIT_H:.0f} ชม. ของค่ากลาง) · เขียน {OUT / 'time_estimate.json'}")


# ---------------------------------------------------------------
# pipeline ส่วนที่ใช้ร่วมกันของ run และ dryrun
# ---------------------------------------------------------------
def check_base_matches_stage3(evals, ref_path=None):
    """
    A4: base ที่รันใหม่ต้องตรงกับ base ของ Stage 3 ทุกบิต (เทียบเฉพาะ evaluation ที่รัน) · ไม่ตรง → หยุด
    B39: NaN เทียบกับ NaN ถือว่าตรง (เช่น IC_shrunk เมื่อ c = 0 เป็น NaN ทั้งสองฝั่ง) · NaN เทียบกับค่า finite ถือว่าไม่ตรง (ต่าง = inf)
    """
    ref = pd.read_csv(ref_path or STAGE3_EVALS, float_precision="round_trip")
    ref = ref[ref["config"] == "base"].set_index(["ticker", "fold"])
    mine = evals[evals["config"] == "base"].set_index(["ticker", "fold"])
    ref = ref.loc[mine.index]
    diffs = {}
    for c in BASE_CHECK_COLS:
        a, b = mine[c].to_numpy(dtype=float), ref[c].to_numpy(dtype=float)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        diffs[c] = float(np.max(np.where(np.isnan(d), np.inf, d)))
    return bool(all(v == 0.0 for v in diffs.values())), diffs


def check_finite(evals, what):
    bad = evals[~np.isfinite(evals[["IC", "relMAE"]].to_numpy(dtype=float)).all(axis=1)]
    if len(bad) or not evals["pred_finite"].all():
        raise StopRun(f"{what}: พบ NaN/ค่าไม่ finite {len(bad)} แถว — หยุดและรายงาน (B3)")


def ensure_announced_committed():
    """ส่วน B เริ่มได้หลังผู้ใช้ commit ประกาศแล้วเท่านั้น: ไฟล์ที่ประกาศต้องไม่มีการเปลี่ยนที่ยังไม่ commit"""
    root = paths.TASK_DIR
    out = subprocess.run(["git", "status", "--porcelain", "--"] + ANNOUNCED_FILES, cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise StopRun(f"git status ล้มเหลว: {out.stderr.strip()}")
    if out.stdout.strip():
        raise StopRun("ไฟล์ที่ประกาศยังไม่ได้ commit (หรือถูกแก้หลัง commit) — ห้ามเริ่มส่วน B:\n" + out.stdout)


def run_pipeline(configs, out, *, jobs, units=None, shuffles=hpo.PLACEBO_SHUFFLES, smoke=False):
    """ค้นหา → placebo → ยืนยัน → ตัดสิน → รายงาน · คืน decision dict"""
    floor = b22_floor()
    names = [c.name for c in configs]
    cands = [c for c in configs if c.name != "base"]
    timing = {}

    t0 = time.time()
    evals, scores, _ = runner.run_configs(configs, out, seeds=SEARCH_SEEDS, jobs=jobs, units=units, const_ic_zero=True)
    timing["search_wall_min"] = (time.time() - t0) / 60
    check_finite(evals, "ค้นหา")
    ok, diffs = check_base_matches_stage3(evals)
    (out / "base_check.json").write_text(json.dumps({"matches_stage3_bitwise": ok, "max_abs_diff": diffs, "units": units}, indent=2), encoding="utf-8")
    if not ok:
        raise StopRun(f"base ที่รันใหม่ไม่ตรงกับ Stage 3 (ต่างสูงสุด {max(diffs.values()):.3e}) — หยุดและรายงาน (ดู {out / 'base_check.json'})")
    print(f"[stage6] base ตรงกับ Stage 3 ทุกบิต · ค้นหาเสร็จ {timing['search_wall_min']:.1f} นาที")

    t0 = time.time()
    ev_p = placebo.run_placebo(configs, out / "placebo", shuffle_seeds=shuffles, seeds=SEARCH_SEEDS, jobs=jobs, units=units,
                                const_ic_zero=True)
    timing["placebo_wall_min"] = (time.time() - t0) / 60
    check_finite(ev_p, "placebo")
    pstat = hpo.placebo_stats(ev_p, names)
    print(f"[stage6] placebo {len(shuffles)} รอบ · T_s = {', '.join(f'{x:+.4f}' for x in pstat['t_s'])} · T_hpo = {pstat['t_hpo']:+.4f}")

    selection.assert_scores_have_seed_noise(scores)
    noise = selection.pooled_seed_noise(scores)
    top = str(scores[scores["config"] != "base"].sort_values("mean_IC", ascending=False).iloc[0]["config"])
    t0 = time.time()
    c_evals, c_scores, _ = runner.run_configs([BASE, next(c for c in cands if c.name == top)], out / "confirm",
                                              seeds=hpo.CONFIRM_SEEDS, jobs=jobs, units=units, const_ic_zero=True)
    timing["confirm_wall_min"] = (time.time() - t0) / 60
    check_finite(c_evals, "ยืนยัน")
    judge = hpo.judge_winner(scores, evals, c_scores, c_evals, noise, pstat["t_hpo"], floor=floor)
    (out / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    histories = json.loads((out / "histories.json").read_text(encoding="utf-8"))
    ctx = {"configs": configs, "table": hpo.configs_table(configs), "scores": scores, "evals": evals, "histories": histories,
           "ev_p": ev_p, "pstat": pstat, "c_scores": c_scores, "c_evals": c_evals, "c_hist": json.loads((out / "confirm" / "histories.json").read_text(encoding="utf-8")),
           "judge": judge, "noise": noise, "timing": timing, "shuffles": shuffles, "smoke": smoke, "units": units, "out": out}
    write_report(ctx)
    return judge


# ---------------------------------------------------------------
# รายงาน
# ---------------------------------------------------------------
def plot_ic_vs_placebo(path, ic_real, ic_placebo, base_name, thr):
    """
    จุด = IC ของจริง (เรียงตาม IC) · แถบเทา = ช่วง min–max ของ IC ของ config เดียวกันตอน y ถูกสลับ (placebo ทุกรอบ)
    เส้นประ = IC ของ base · เส้นจุด = base + เกณฑ์สุดท้ายของ A6 (1) (ผู้สมัครต้องอยู่เหนือเส้นนี้)
    """
    order = list(ic_real.sort_values(ascending=False).index)
    x = np.arange(len(order))
    ink, muted, grid = "#0b0b0b", "#52514e", "#d9d8d3"
    blue, orange, band = "#2a78d6", "#eb6834", "#a8a7a0"
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=grid, lw=0.6)
    lo, hi = ic_placebo.loc[order].min(axis=1), ic_placebo.loc[order].max(axis=1)
    ax.vlines(x, lo, hi, color=band, lw=5, alpha=0.55, zorder=1, label=f"ช่วง min–max ของ placebo ({ic_placebo.shape[1]} รอบ · y ถูกสลับ)")
    ax.scatter(x, ic_placebo.loc[order].mean(axis=1), marker="_", s=70, color=muted, zorder=2, lw=1.4, label="ค่าเฉลี่ย placebo")
    cand = [i for i, n in enumerate(order) if n != base_name]
    ax.scatter(x[cand], ic_real.loc[[order[i] for i in cand]], s=36, color=blue, edgecolor="white", linewidth=0.8, zorder=3, label="IC ของจริง (ผู้สมัคร)")
    b = order.index(base_name)
    ax.scatter([x[b]], [ic_real[base_name]], s=70, marker="D", color=orange, edgecolor="white", linewidth=0.8, zorder=4, label="IC ของจริง (base)")
    ax.axhline(ic_real[base_name], color=orange, lw=1, ls="--", zorder=0)
    ax.axhline(ic_real[base_name] + thr, color=ink, lw=1, ls=":", zorder=0)
    ax.text(-0.45, ic_real[base_name] + thr, f"base + เกณฑ์ = {ic_real[base_name] + thr:+.3f}", va="bottom", ha="left", fontsize=8, color=ink)
    ys = np.concatenate([ic_real.to_numpy(), lo.to_numpy(), hi.to_numpy(), [ic_real[base_name] + thr]])
    pad = 0.08 * (ys.max() - ys.min())
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8, color=ink)
    ax.set_ylabel("mean IC (6 evaluation)", color=ink)
    ax.set_title("Stage 6: IC ของ 25 config เทียบช่วงของ placebo", loc="left", fontsize=11, color=ink)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main_table(ctx):
    sc, tb = ctx["scores"].set_index("config"), ctx["table"].set_index("name")
    base_ic = float(sc.loc["base", "mean_IC"])
    caps = hpo.epoch_cap_counts(ctx["histories"])
    p_caps = ctx["ev_p"].groupby("config")["n_hit_cap"].sum() if "n_hit_cap" in ctx["ev_p"].columns else None
    rows = []
    for rank, n in enumerate(sc.sort_values("mean_IC", ascending=False).index, 1):
        r, t = sc.loc[n], tb.loc[n]
        hit, tot = caps[n]
        rows.append({"#": rank, "config": n, "lr": t["lr"], "batch": t["batch"], "dropout": t["dropout"], "wd": t["weight_decay"],
                     "filters": t["filters"], "hidden": t["ann_hidden"], "nperseg": t["nperseg"], "hop": t["hop"], "noise": t["noise_std"],
                     "ภาพ": t["image_shape"], "พารามิเตอร์": f"{int(t['n_params']):,}",
                     "mean IC": fmt(float(r["mean_IC"]), "s4"), "ΔIC": "—" if n == "base" else fmt(float(r["mean_IC"]) - base_ic, "s4"),
                     "σ seed": fmt(float(r["ic_seed_std"]), "r4"), "Spearman": fmt(float(r["mean_IC_spearman"]), "s4"),
                     "IC>0": f"{int(r['n_evals_ic_positive'])}/{int(r['n_evals'])}",
                     "relMAE (return)": fmt(float(r["mean_relMAE"]), "r4"), "relMAE (บาท)": fmt(float(r["mean_relMAE_baht"]), "r4"),
                     "หลังปรับ c (return)": fmt(float(r["mean_relMAE_shrunk"]), "r4"), "หลังปรับ c (บาท)": fmt(float(r["mean_relMAE_baht_shrunk"]), "r4"),
                     "StdRatio": fmt(float(r["mean_StdRatio"]), "r3"), "DirAcc": fmt(float(r["mean_DirAcc"]), "r3"),
                     "ชนเพดาน epoch (จริง)": f"{hit}/{tot}",
                     "ชนเพดาน (placebo)": "—" if p_caps is None else f"{int(p_caps.loc[n])}/{len(ctx['shuffles']) * int(r['n_evals']) * len(SEARCH_SEEDS)}"})
    return pd.DataFrame(rows)


def collapse_section(ctx):
    """B39 ข้อ 3: จำนวน run ที่คำทำนายคงที่ (IC := 0) ต่อ config · ระดับ seed และระดับ evaluation · ค้นหาและ placebo + descriptive ของพารามิเตอร์"""
    out, tb = ctx["out"], ctx["table"].set_index("name")
    cs, cp, cc = hpo.collapse_counts(ctx["evals"]), hpo.collapse_counts(ctx["ev_p"]), hpo.collapse_counts(ctx["c_evals"])
    names = list(ctx["table"]["name"])
    tot = (cs["seed_runs"] + cp["seed_runs"]).loc[names]
    order = list(tot.sort_values(ascending=False, kind="stable").index)
    s_tot, p_tot = int(cs["seed_runs_total"].iloc[0]), int(cp["seed_runs_total"].iloc[0])
    se_tot, pe_tot = int(cs["evals_total"].iloc[0]), int(cp["evals_total"].iloc[0])
    cols = ["lr", "batch", "dropout", "weight_decay", "filters", "ann_hidden", "nperseg", "hop", "noise_std"]
    rows, csv_rows = [], []
    for n in order:
        rows.append({"config": n, f"ค้นหา: seed-run (จาก {s_tot})": int(cs.loc[n, "seed_runs"]), f"ค้นหา: evaluation (จาก {se_tot})": int(cs.loc[n, "evals"]),
                     f"placebo: seed-run (จาก {p_tot})": int(cp.loc[n, "seed_runs"]), f"placebo: evaluation (จาก {pe_tot})": int(cp.loc[n, "evals"]),
                     **{c: tb.loc[n, c] for c in cols}})
        csv_rows.append({"config": n, "search_seed_runs": int(cs.loc[n, "seed_runs"]), "search_seed_runs_total": s_tot,
                         "search_evals": int(cs.loc[n, "evals"]), "search_evals_total": se_tot,
                         "placebo_seed_runs": int(cp.loc[n, "seed_runs"]), "placebo_seed_runs_total": p_tot,
                         "placebo_evals": int(cp.loc[n, "evals"]), "placebo_evals_total": pe_tot})
    pd.DataFrame(csv_rows).to_csv(out / "collapse_by_config.csv", index=False)
    md = ["## 7. Collapse — คำทำนายคงที่ (IC := 0 ตาม B39)", "",
          "กติกา: std(ŷ)/std(y) < 1e-8 หรือ ptp(ŷ) = 0 → IC Pearson = IC Spearman := 0 · ใช้เหมือนกันทุกส่วน (ค้นหา · placebo · ยืนยัน) · "
          "seed-run = หนึ่ง unit × หนึ่ง seed เดี่ยว · evaluation = คำทำนายเฉลี่ย 3 seed ของหนึ่ง unit ", "",
          f"- ค้นหา: **{int(cs['seed_runs'].sum())}/{int(cs['seed_runs_total'].sum())}** seed-run · **{int(cs['evals'].sum())}/{int(cs['evals_total'].sum())}** evaluation",
          f"- placebo ({len(ctx['shuffles'])} รอบ): **{int(cp['seed_runs'].sum())}/{int(cp['seed_runs_total'].sum())}** seed-run · **{int(cp['evals'].sum())}/{int(cp['evals_total'].sum())}** evaluation",
          f"- ยืนยัน (seeds 10–12): {int(cc['seed_runs'].sum())}/{int(cc['seed_runs_total'].sum())} seed-run · {int(cc['evals'].sum())}/{int(cc['evals_total'].sum())} evaluation", "",
          "### 7.1 ต่อ config (เรียงตาม collapse รวมค้นหา + placebo)", "", md_table(pd.DataFrame(rows)), ""]
    if len(names) == hpo.N_CANDIDATES + 1:
        lv_s = hpo.level_means(ctx["table"], cs["seed_runs"].astype(float))
        lv_p = hpo.level_means(ctx["table"], cp["seed_runs"].astype(float))
        lv = lv_s[["param", "level", "n_config"]].copy()
        lv[f"ค้นหา: seed-run เฉลี่ยต่อ config (จาก {s_tot})"] = [f"{v:.2f}" for v in lv_s["mean_IC"]]
        lv[f"placebo: seed-run เฉลี่ยต่อ config (จาก {p_tot})"] = [f"{v:.2f}" for v in lv_p["mean_IC"]]
        common, top = hpo.common_levels(ctx["table"], tot)
        md += ["### 7.2 (descriptive) collapse เฉลี่ยตามระดับของแต่ละพารามิเตอร์ (เฉพาะ 24 ผู้สมัคร)", "", md_table(lv), "",
               f"### 7.3 (descriptive) ระดับที่ซ้ำกันใน config ที่ collapse บ่อยที่สุด {len(top)} ตัว ({', '.join(top) if top else 'ไม่มี'})", "",
               md_table(common), "",
               "เป็นภาพรวมเท่านั้น: 24 config สุ่มค่าหลายพารามิเตอร์พร้อมกัน จึงแยกผลของแต่ละพารามิเตอร์ออกจากกันไม่ได้ · ห้ามใช้เปลี่ยน config", ""]
    return md, {"search": cs, "placebo": cp, "confirm": cc}


def write_report(ctx):
    out, j, ps = ctx["out"], ctx["judge"], ctx["pstat"]
    sc = ctx["scores"].set_index("config")
    names = list(ctx["table"]["name"])
    base_ic = float(sc.loc["base", "mean_IC"])
    n_rounds = len(ctx["shuffles"])
    n_eval = int(sc["n_evals"].iloc[0])
    smoke_note = " · **dry-run บน 1 evaluation — ตัวเลขไม่ใช่ผลจริงและห้ามนำไปใช้ตัดสิน**" if ctx["smoke"] else ""
    fp_rate = {5: "≈ 1/6", 3: "≈ 1/4"}.get(n_rounds, f"≈ 1/{n_rounds + 1}")

    # ---- placebo ----
    pic = ps["ic"]
    prow = []
    for s in ctx["shuffles"]:
        d = ps["delta"][s]
        prow.append({"รอบ (seed สลับ)": int(s), "IC ของ base": fmt(float(pic.loc["base", s]), "s4"),
                     "ผู้สมัครที่ IC สูงสุด": d.idxmax(), "IC สูงสุด": fmt(float(pic.loc[d.idxmax(), s]), "s4"),
                     "T_s = max(IC − IC_base)": fmt(float(ps["t_s"][s]), "s4")})
    # ---- ตรวจผู้ชนะ ----
    def mark(v):
        return "**ผ่าน**" if v else "ไม่ผ่าน"
    vrow = [
        {"ข้อ": "(1) ΔIC เทียบ base > max(noise, 0.0597, T_hpo)", "ค่า": f"ΔIC {j['delta_ic']:+.4f} vs {j['thr1']:.4f}", "ผล": mark(j["c1"])},
        {"ข้อ": "(2) **[บังคับ]** ยืนยัน seeds 10–12: ΔIC > เกณฑ์ข้อ (1)", "ค่า": f"ΔIC {j['delta_ic_confirm']:+.4f} vs {j['thr1']:.4f}", "ผล": mark(j["c2"])},
        {"ข้อ": f"(3) IC>0 ≥ {j['min_positive_needed']}/{j['n_eval']} evaluation · ตัด evaluation ที่ ΔIC สูงสุด ({j['max_delta_eval']}) แล้ว ΔIC เฉลี่ย > max(noise, 0.0597)",
         "ค่า": f"IC>0 {j['n_ic_positive']}/{j['n_eval']} · ΔIC เฉลี่ยที่เหลือ {j['delta_mean_trimmed']:+.4f} vs {j['thr3']:.4f}", "ผล": mark(j["c3"])},
        {"ข้อ": "(4) relMAE (return · ก่อนปรับขนาด) ไม่สูงกว่า base", "ค่า": f"{j['relMAE_top']:.4f} vs base {j['relMAE_base']:.4f}", "ผล": mark(j["c4"])}]
    verdict = "**มีผู้ชนะ**" if j["has_winner"] else "**ไม่มีผู้ชนะ**"
    # ---- descriptive: mean IC ตามระดับ ----
    lvl_md = []
    if len(names) == hpo.N_CANDIDATES + 1:
        real_lv = hpo.level_means(ctx["table"], sc["mean_IC"])
        pl_cols = [hpo.level_means(ctx["table"], pic[s]) for s in ctx["shuffles"]]
        pl_all = pd.concat([d["mean_IC"] for d in pl_cols], axis=1)
        lv = real_lv[["param", "level", "n_config"]].copy()
        lv["mean IC (จริง)"] = [fmt(v, "s4") for v in real_lv["mean_IC"]]
        lv["mean IC (placebo เฉลี่ย)"] = [fmt(float(v), "s4") for v in pl_all.mean(axis=1)]
        lv["placebo min–max ข้ามรอบ"] = [f"{a:+.4f} … {b:+.4f}" for a, b in zip(pl_all.min(axis=1), pl_all.max(axis=1))]
        lvl_md = [md_table(lv), ""]
    # ---- เวลา / จำนวน config ----
    log = pd.read_csv(runner.EXPERIMENT_LOG) if runner.EXPERIMENT_LOG.exists() else pd.DataFrame()
    n_log = len(log)
    n_new = int((log["stage"] == out.name).sum()) if len(log) else 0
    sec = {"ค้นหา": float(ctx["evals"]["seconds"].sum()), "placebo": float(ctx["ev_p"]["seconds"].sum()), "ยืนยัน": float(ctx["c_evals"]["seconds"].sum())}
    t = ctx["timing"]

    md = [f"# Stage 6 — HPO ของ track 1 (DECISIONS B38){smoke_note}", "",
          f"> สร้างเมื่อ {stamp()} · ผลทุกตัวเป็น development / walk-forward (dev ≤ {VAL_END}) · track 1 เท่านั้น (CNN→ANN spectrogram อย่างเดียว) · "
          f"{len(names)} config × {n_eval} evaluation × seeds {tuple(SEARCH_SEEDS)} · placebo {n_rounds} รอบ (seed {ctx['shuffles'][0]}–{ctx['shuffles'][-1]})", "",
          f"## สรุป: {verdict}", "",
          f"- ผู้สมัครอันดับ 1 (mean IC สูงสุดใน 24 ผู้สมัคร): **{j['top']}** · mean IC {fmt(float(sc.loc[j['top'], 'mean_IC']), 's4')} · base {fmt(base_ic, 's4')}"
          + (" · **base อยู่อันดับ 1 เมื่อรวมทุก config**" if j["base_is_overall_top"] else ""),
          f"- เกณฑ์สุดท้ายของข้อ (1) = max(noise ของ seed {j['noise']:.4f}, B22 {j['floor']:.4f}, T_hpo {j['t_hpo']:+.4f}) = **{j['thr1']:.4f}**",
          f"- อัตราผลบวกปลอมของเกณฑ์ T_hpo (max ของ {n_rounds} รอบ) {fp_rate}",
          "- กติกา B39: คำทำนายคงที่ (std(ŷ)/std(y) < 1e-8 หรือ ptp(ŷ) = 0) → IC := 0 ทั้งระดับ seed และ evaluation ในทุกส่วน (รายละเอียดข้อ 7) · เกณฑ์ A6 (2) เป็นเกณฑ์บังคับ", ""]
    md += ["## 1. ตาราง config ทั้งหมด (เรียงตาม mean IC)", "", md_table(main_table(ctx)), "",
           "σ seed = std ข้าม seed ของ IC เฉลี่ย 6 evaluation (B17) · ชนเพดาน epoch = จำนวน run (หนึ่ง unit × หนึ่ง seed) ที่หยุดเพราะครบ 200 epoch ไม่ใช่ early stopping · "
           "หลังปรับ c = หลังปรับขนาดคำทำนายด้วย c ที่เลือกจากส่วน early-stop (B18 · รายงานเพิ่ม ไม่ใช้เลือก)", "",
           "## 2. placebo ของการค้นหาทั้งชุด (y ใน train ถูกสลับ · y ของ test เป็นของจริง)", "", md_table(pd.DataFrame(prow)), "",
           f"- **T_hpo = max ข้ามรอบของ T_s = {ps['t_hpo']:+.4f}** · noise ของ seed (pooled {len(names)} config) = {j['noise']:.4f} · B22 = {j['floor']:.4f}",
           f"- **เกณฑ์สุดท้ายของ A6 ข้อ (1) = max({j['noise']:.4f}, {j['floor']:.4f}, {ps['t_hpo']:+.4f}) = {j['thr1']:.4f}**", "",
           "## 3. ตรวจผู้ชนะทีละข้อ (A6 · ตรวจเฉพาะอันดับ 1 · ไม่ไล่อันดับ 2, 3)", "", md_table(pd.DataFrame(vrow)), "",
           f"→ {verdict}", ""]
    cd = j.get("confirm_descriptive")
    if cd:
        md += [f"ข้อมูลเสริม (ไม่ใช้ตัดสิน): ข้อ (3) คิดซ้ำจากผลยืนยัน seeds 10–12 → IC>0 {cd['n_ic_positive']}/{cd['n_eval']} · ΔIC เฉลี่ยหลังตัด {cd['delta_mean_trimmed']:+.4f}", ""]
    md += ["## 4. (descriptive เท่านั้น) mean IC เฉลี่ยตามระดับของแต่ละพารามิเตอร์ · ห้ามใช้เลือก config", ""]
    md += lvl_md or ["(ข้ามใน dry-run — ต้องมีครบ 25 config)", ""]
    md += ["เฉลี่ยจาก 24 ผู้สมัคร (ไม่นับ base) · การออกแบบสมดุลทำให้แต่ละระดับมี 8 config (noise_std 6 config) แต่ปัจจัยอื่นยังปนกันอยู่ จึงใช้เป็นภาพรวมเท่านั้น", "",
           "## 5. จำนวน config และเวลา", "",
           f"- config ที่ต่างกันทั้งหมดที่ลองใน dev ตาม experiment_log: **{n_log}** (แถวจาก Stage 6 = {n_new}; เดิม 21 + ใหม่ 24 = 45 ถ้าครบ)",
           f"- เวลา (wall · jobs ตามที่รัน · ของครั้งนี้): ค้นหา {t['search_wall_min']:.1f} นาที · placebo {t['placebo_wall_min']:.1f} นาที · ยืนยัน {t['confirm_wall_min']:.1f} นาที",
           f"- ผลรวมเวลา train ของทุก unit (วินาที · จากคอลัมน์ seconds): ค้นหา {sec['ค้นหา'] / 60:.0f} นาที · placebo {sec['placebo'] / 60:.0f} นาที · ยืนยัน {sec['ยืนยัน'] / 60:.0f} นาที", "",
           "## 6. รูป", "", f"![IC ของ 25 config เทียบ placebo]({'stage6_ic_vs_placebo.png'})", ""]
    col_md, col_counts = collapse_section(ctx)
    md += col_md
    (out / "stage6_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    decision = {"generated_at": stamp(), "smoke": bool(ctx["smoke"]), "has_winner": j["has_winner"], "judge": j,
                "placebo": {"shuffles": list(ctx["shuffles"]), "t_s": {str(k): float(v) for k, v in ps["t_s"].items()}, "t_hpo": ps["t_hpo"],
                            "false_positive_rate_note": fp_rate},
                "noise": ctx["noise"], "n_configs_logged": n_log, "timing": t, "const_ic_zero": True,
                "collapse": {k: {"seed_runs": int(v["seed_runs"].sum()), "seed_runs_total": int(v["seed_runs_total"].sum()),
                                 "evals": int(v["evals"].sum()), "evals_total": int(v["evals_total"].sum())} for k, v in col_counts.items()}}
    (out / "stage6_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    plot_ic_vs_placebo(out / "stage6_ic_vs_placebo.png", sc["mean_IC"], pic, "base", j["thr1"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 110 + "\nสรุป Stage 6\n" + "=" * 110)
    print(pd.DataFrame(prow).to_string(index=False))
    print(pd.DataFrame(vrow).to_string(index=False))
    print(f"→ {verdict.replace('*', '')} · เขียน {out / 'stage6_summary.md'}")


# ---------------------------------------------------------------
# dryrun / run / report
# ---------------------------------------------------------------
def cmd_dryrun(args):
    """A8 (f): base + ผู้สมัคร 1 ตัว (hpo01) บน 1 evaluation (KBANK fold 1) ครบสายงาน · เขียนลง results/_smoke_stage6/ เท่านั้น"""
    configs = load_design()
    pick = [configs[0], configs[1]]
    paths.assert_output_dir(SMOKE)
    SMOKE.mkdir(parents=True, exist_ok=True)
    runner.EXPERIMENT_LOG = SMOKE / "experiment_log.csv"           # ไม่แตะ experiment_log จริง
    t0 = time.time()
    j = run_pipeline(pick, SMOKE, jobs=args.jobs, units=[("KBANK", 1)], shuffles=(hpo.PLACEBO_SHUFFLES[0],), smoke=True)
    files = sorted(str(p.relative_to(SMOKE)) for p in SMOKE.rglob("*") if p.is_file() and "cache" not in p.parts)
    cache_files = sum(1 for p in SMOKE.rglob("*.pkl"))
    summary = {"dryrun_at": stamp(), "configs": [c.name for c in pick], "units": [["KBANK", 1]], "placebo_shuffles": [hpo.PLACEBO_SHUFFLES[0]],
               "base_matches_stage3": json.loads((SMOKE / "base_check.json").read_text(encoding="utf-8"))["matches_stage3_bitwise"],
               "files_written": files, "n_cache_files": cache_files, "seconds": round(time.time() - t0, 1),
               "verdict_has_winner_dryrun_only": j["has_winner"], "note": "ตัวเลขของ dry-run ไม่ใช่ผลจริง · ไม่ใช้ตัดสิน"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dryrun_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[dryrun] เสร็จใน {summary['seconds'] / 60:.1f} นาที · ไฟล์ที่ออก: {files}")


def cmd_run(args):
    ensure_announced_committed()
    configs = load_design()
    paths.assert_output_dir(OUT)
    if tuple(hpo.PLACEBO_SHUFFLES) != tuple(json.loads((OUT / "design_log.json").read_text(encoding="utf-8"))["placebo_shuffles"]):
        raise StopRun("รอบ placebo ใน hpo.py ไม่ตรงกับ design_log.json")
    print(f"[stage6] เริ่ม {stamp()} · {len(configs)} config · placebo {len(hpo.PLACEBO_SHUFFLES)} รอบ · jobs {args.jobs}")
    run_pipeline(configs, OUT, jobs=args.jobs)


def cmd_report(args):
    """สร้างรายงานใหม่จากผลที่มีอยู่ใน results/stage6_hpo/ (ใช้ cache · unit ที่ cache ไม่ครบจะถูกเทรนเพิ่ม)"""
    ensure_announced_committed()
    configs = load_design()
    run_pipeline(configs, OUT, jobs=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "checks", "estimate", "dryrun", "run", "report"])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    {"design": cmd_design, "checks": cmd_checks, "estimate": cmd_estimate, "dryrun": cmd_dryrun,
     "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
