"""
check_env.py — ตรวจ environment และลายนิ้วมือของข้อมูล (Checkpoint C0)
====================================================================
ไฟล์นี้ทำอะไร
    1) พิมพ์เวอร์ชัน python / numpy / pandas / scipy / scikit-learn / xgboost / torch และเทียบกับ pin ใน requirements.txt
    2) ตรวจว่า torch รัน deterministic แบบ CPU 1 thread ได้ (ข้อ 2.0)
    3) พิมพ์ SHA-256, จำนวนแถว, วันแรก/วันสุดท้าย, เรียงเวลา/ไม่ซ้ำ/ไม่มีค่าว่าง ของ data/*.csv และจำนวนแถวในโซน dev
    4) เขียนสรุปลง results/stage0_setup/environment.md

ทำไมต้องมี: ให้ผู้ใช้เห็นหลักฐานว่า "รันบนข้อมูลและ library ชุดไหน" ก่อนเริ่มงานจริง (ผล C0)
รัน:  python -m src.check_env
"""

import hashlib
import platform
import sys

import numpy as np
import pandas as pd

from src import paths
from src.settings import TICKERS, VAL_END

# เวอร์ชันที่ pin ไว้ (SPEC ข้อ 2.0) -- ไม่ตรงไม่ถือว่าพัง แต่ legacy อาจให้ผลต่างเดิม จึงพิมพ์เตือน
PINS = {"numpy": "2.1.1", "pandas": "2.3.3", "scikit-learn": "1.9.1", "xgboost": "3.4.1", "torch": "2.8.0"}


def versions():
    import scipy
    import sklearn
    import torch
    import xgboost
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "torch": torch.__version__.split("+")[0]}


def torch_determinism_ok():
    """เทรน 2 ครั้งด้วย seed เดียวกันบน CPU 1 thread -- น้ำหนักต้องเหมือนกันทุกบิต"""
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    def one_run():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                                torch.nn.Flatten(), torch.nn.LazyLinear(1))
        x = torch.randn(16, 1, 9, 65)
        y = torch.randn(16)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(5):
            opt.zero_grad()
            torch.nn.functional.mse_loss(m(x).squeeze(-1), y).backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    return bool(torch.equal(one_run(), one_run()))


def fingerprint(path):
    raw = path.read_bytes()
    df = pd.read_csv(path)
    d = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    dev = d <= pd.Timestamp(VAL_END)
    return {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(df),
            "first": str(d.iloc[0].date()), "last": str(d.iloc[-1].date()),
            "sorted": bool(d.is_monotonic_increasing), "unique": bool(d.is_unique),
            "nulls": int(df.isna().sum().sum()), "rows_dev": int(dev.sum()),
            "rows_after_dev": int((~dev).sum()), "columns": list(df.columns)}


def main():
    v = versions()
    print("== environment ==")
    for k, val in v.items():
        pin = PINS.get(k)
        flag = "" if pin is None or val == pin else f"   <-- ต่างจาก pin {pin}"
        print(f"  {k:13s} {val}{flag}")
    print(f"  platform      {platform.platform()}")
    import os
    print(f"  cpu_count     {os.cpu_count()}")
    det = torch_determinism_ok()
    print(f"  torch deterministic (CPU, 1 thread, 2 runs bit-identical): {det}")

    print("\n== data (data/*.csv) ==")
    fps = [fingerprint(paths.DATA_DIR / f"{t}_10Y_Cleaned.csv") for t in TICKERS]
    for f in fps:
        print(f"  {f['file']}: rows={f['rows']} {f['first']} -> {f['last']} sha256={f['sha256'][:16]}.. "
              f"sorted={f['sorted']} unique={f['unique']} nulls={f['nulls']} | dev(<= {VAL_END})={f['rows_dev']} "
              f"after-dev={f['rows_after_dev']}")

    out = paths.assert_output_dir(paths.RESULTS_DIR / "stage0_setup")
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# C0 — environment และข้อมูล", "", "| รายการ | ค่า | pin |", "|---|---|---|"]
    lines += [f"| {k} | {val} | {PINS.get(k, '')} |" for k, val in v.items()]
    lines += [f"| torch deterministic (CPU 1 thread) | {det} | |", "",
              "| ไฟล์ | แถว | วันแรก | วันสุดท้าย | แถว dev | SHA-256 |", "|---|---|---|---|---|---|"]
    lines += [f"| {f['file']} | {f['rows']} | {f['first']} | {f['last']} | {f['rows_dev']} | `{f['sha256']}` |"
              for f in fps]
    (out / "environment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nเขียน {out / 'environment.md'}")
    ok = det and all(f["sorted"] and f["unique"] and f["nulls"] == 0 for f in fps)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
