"""
test_runner.py — ตัวขับการทดลองต้องให้ผลเหมือนเดิมทุกครั้ง และขนานแล้วต้องได้ผลเท่ากับรันทีละตัว
==========================================================================================
ไฟล์นี้ทำอะไร
    1. ตรวจว่า cache คืนค่าเดิมเป๊ะ และคีย์ของ cache เปลี่ยนเมื่อ config เปลี่ยน
    2. ตรวจว่า `jobs > 1` (รันคนละ process) ให้ผลเหมือน `jobs = 1` ทุกบิต
    3. ตรวจว่า experiment_log.csv ถูกต่อท้าย ไม่ถูกเขียนทับ

ทำไมต้องมี: ความต่างระหว่าง config ในงานนี้เล็กมาก (relMAE ต่างกันหลัก 0.001)
ถ้าการขนานทำให้ผลขยับแม้นิดเดียว การเลือก config จะกลายเป็นการสุ่ม
เทสต์นี้ใช้ config ที่ย่อขนาด (epoch น้อย) จึงเร็วพอรันทุกครั้ง
"""

import numpy as np
import pytest
from dataclasses import replace

from src import paths, runner
from src.configs import BASE, variant

TINY = replace(BASE, name="tiny", max_epochs=3, patience=3)


def test_unit_key_changes_with_config_and_inputs():
    a = runner.unit_key(TINY, "KBANK", 1, (0,), "2025-02-25")
    assert a == runner.unit_key(TINY, "KBANK", 1, (0,), "2025-02-25")
    assert a != runner.unit_key(replace(TINY, window=64), "KBANK", 1, (0,), "2025-02-25")
    assert a != runner.unit_key(TINY, "ADVANC", 1, (0,), "2025-02-25")
    assert a != runner.unit_key(TINY, "KBANK", 2, (0,), "2025-02-25")
    assert a != runner.unit_key(TINY, "KBANK", 1, (0, 1), "2025-02-25")


@pytest.mark.slow
def test_cache_returns_identical_results(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    cache = tmp_path / "cache"
    a, pa, ha = runner.run_unit(TINY, "KBANK", 3, seeds=(0,), cache_dir=cache)
    b, pb, hb = runner.run_unit(TINY, "KBANK", 3, seeds=(0,), cache_dir=cache)
    assert a["from_cache"] is False and b["from_cache"] is True
    assert a["relMAE"] == b["relMAE"] and a["IC"] == b["IC"]
    np.testing.assert_array_equal(pa["pred"].to_numpy(), pb["pred"].to_numpy())
    assert ha == hb and ha[0]["epochs"] == len(ha[0]["val_loss"]) and 1 <= ha[0]["best_epoch"] <= ha[0]["epochs"]


@pytest.mark.slow
def test_parallel_gives_the_same_numbers_as_serial(tmp_path, monkeypatch):
    """หัวใจ: ขนานได้ก็ต่อเมื่อผลไม่ขยับแม้แต่บิตเดียว"""
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "EXPERIMENT_LOG", tmp_path / "experiment_log.csv")
    cfgs = [TINY, variant("tiny_gap", head="gap", max_epochs=3, patience=3)]
    _, s1, p1 = runner.run_configs(cfgs, tmp_path / "serial", seeds=(0,), jobs=1, verbose=False)
    _, s2, p2 = runner.run_configs(cfgs, tmp_path / "parallel", seeds=(0,), jobs=4, verbose=False)
    np.testing.assert_array_equal(p1.sort_values(["config", "ticker", "date"])["pred"].to_numpy(),
                                  p2.sort_values(["config", "ticker", "date"])["pred"].to_numpy())
    np.testing.assert_array_equal(s1.sort_values("config")["mean_IC"].to_numpy(),
                                  s2.sort_values("config")["mean_IC"].to_numpy())
    np.testing.assert_array_equal(s1.sort_values("config")["ic_seed_std"].to_numpy(),
                                  s2.sort_values("config")["ic_seed_std"].to_numpy())


@pytest.mark.slow
def test_experiment_log_is_appended_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "EXPERIMENT_LOG", tmp_path / "experiment_log.csv")
    import pandas as pd
    # config ต่างกัน (window ต่างกัน) → ต้องต่อท้ายทั้งคู่ และคอลัมน์ stage ต้องบอกว่ามาจาก stage ไหน
    a = pd.DataFrame([{"config": "a", **variant("a", window=64).to_row(), "mean_IC": 0.1}])
    b = pd.DataFrame([{"config": "b", **variant("b", window=256, nperseg=32, hop=4).to_row(), "mean_IC": 0.2}])
    runner.append_experiment_log(a, "stage1")
    runner.append_experiment_log(b, "stage2")
    df = pd.read_csv(tmp_path / "experiment_log.csv")
    assert len(df) == 2 and list(df["stage"]) == ["stage1", "stage2"]


def test_experiment_log_does_not_double_count_a_rerun(tmp_path, monkeypatch):
    """
    config เดียวกัน (ทุก field ยกเว้นชื่อ) ที่ถูกรันซ้ำ ไม่ว่าจะใน stage เดิมหรือ stage อื่น ต้องไม่เพิ่มแถว
    (เช่น Stage 1 รัน base ซ้ำเพื่อเก็บ IC รายseed) · แต่ config ที่ต่างกันแม้ field เดียวต้องเพิ่ม
    """
    import pandas as pd
    monkeypatch.setattr(runner, "EXPERIMENT_LOG", tmp_path / "experiment_log.csv")
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    row = lambda cfg: pd.DataFrame([{"config": cfg.name, **cfg.to_row(), "primary_score": 1.0}])
    runner.append_experiment_log(row(BASE), "stage1")
    runner.append_experiment_log(row(BASE), "stage1")                              # รันซ้ำ → ไม่นับ
    assert len(pd.read_csv(tmp_path / "experiment_log.csv")) == 1
    runner.append_experiment_log(row(BASE), "stage2")                              # config เดิมใน stage อื่น → ไม่นับ
    runner.append_experiment_log(row(variant("base_renamed")), "stage2")           # ชื่อต่างแต่ field เหมือนกัน → ไม่นับ
    assert len(pd.read_csv(tmp_path / "experiment_log.csv")) == 1
    runner.append_experiment_log(row(variant("w64", window=64)), "stage1")         # field ต่าง → นับ
    assert len(pd.read_csv(tmp_path / "experiment_log.csv")) == 2
