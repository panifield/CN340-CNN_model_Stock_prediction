"""
ตรวจความถูกต้องของ pipeline Task A (data leakage + sanity checks)

Usage:
    python A/tests/check_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from src.config import CHANNELS, DATA_DIR, SPLIT, START_DATE, STOCKS, WINDOW  # noqa: E402
from src.data import build_dataset, build_extras, build_round_features, load_stock, split_and_normalize  # noqa: E402
from src.train import predict_proba, train_one  # noqa: E402
from methods import build_images, tail_features  # noqa: E402

EXP_TRANSFORMS = ["stft_complex", "cwt", "gaf", "raw"]  # วิธีแปลงภาพใน experiments

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def data_quality(stock):
    raw = pd.read_csv(DATA_DIR / f"{stock}_10Y_Cleaned.csv")
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    df = load_stock(stock)
    check("ไม่มีวันที่ซ้ำ", not raw["Date"].duplicated().any())
    check(f"ข้อมูลเริ่มหลัง {START_DATE}", df["Date"].min() >= pd.Timestamp(START_DATE),
          str(df["Date"].min().date()))
    check("Volume > 0 และ Low <= Close <= High",
          bool((df.Volume > 0).all() and ((df.Close >= df.Low) & (df.Close <= df.High)).all()))
    check("feature ไม่มี NaN/inf", bool(np.isfinite(df[CHANNELS].to_numpy()).all()))


def future_perturbation_test(stock, n_probe=15):
    """เปลี่ยนข้อมูลของวัน t และหลังจากนั้นแบบสุ่ม -> input ของ sample วัน t ต้องไม่เปลี่ยน
    ถ้าเปลี่ยน แปลว่า input ใช้ข้อมูลของวันที่จะทาย/อนาคต (leak)"""
    raw = pd.read_csv(DATA_DIR / f"{stock}_10Y_Cleaned.csv")
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    all_ch = CHANNELS + ["parity_sig", "int_chg"]  # รวม channel ที่ใช้ใน experiments

    def make(df):
        X, y, d = build_dataset(df, channels=all_ch)
        inputs = [X, build_extras(df), build_round_features(df), tail_features(df, WINDOW, all_ch)]
        inputs += [build_images(df, WINDOW, all_ch, tf)[0] for tf in EXP_TRANSFORMS]
        return inputs, y, d

    I0, y0, d0 = make(load_stock(stock))
    X0 = I0[0]
    rng = np.random.default_rng(0)
    leaks, label_changes = 0, 0
    for k in rng.choice(len(X0), n_probe, replace=False):
        target_date = d0[k]
        mod = raw.copy()
        fut = mod["Date"] >= target_date
        for col in ["Price", "Open", "High", "Low", "Vol. ('000)"]:
            mod.loc[fut, col] = mod.loc[fut, col] * rng.uniform(0.8, 1.25, fut.sum())
        mod.loc[fut, "High"] = mod.loc[fut, ["Price", "Open", "High", "Low"]].max(axis=1)
        mod.loc[fut, "Low"] = mod.loc[fut, ["Price", "Open", "High", "Low"]].min(axis=1)

        # ใช้ load_stock เดิม แต่ชี้ไปที่ข้อมูลที่ถูกแก้
        import src.data as data_mod
        orig_read = data_mod.pd.read_csv
        data_mod.pd.read_csv = lambda *_a, **_k: mod.assign(
            Date=mod["Date"].dt.strftime("%m/%d/%Y"))
        try:
            I1, y1, _ = make(load_stock(stock))
        finally:
            data_mod.pd.read_csv = orig_read
        leaks += not all(np.array_equal(a[k], b[k]) for a, b in zip(I0, I1))
        label_changes += y0[k] != y1[k]
    check("input ทุกแบบของวัน t (STFT/CWT/GAF/complex/raw/extras/round/tail) ไม่เปลี่ยนเมื่อแก้ข้อมูลวัน t เป็นต้นไป", leaks == 0,
          f"{leaks}/{n_probe} sample เปลี่ยน")
    # ตรวจว่า test นี้มีผลจริง: label ต้องเปลี่ยนในบาง sample
    check("(ตัวควบคุม) การแก้ข้อมูลทำให้ label เปลี่ยนจริง", label_changes > 0,
          f"{label_changes}/{n_probe} label เปลี่ยน")


def split_and_norm_test(stock):
    df = load_stock(stock)
    X, y, dates = build_dataset(df)
    parts, sl, mu, sd = split_and_normalize(X, y, SPLIT)
    tr_end, va_start = dates[sl["train"]][-1], dates[sl["val"]][0]
    va_end, te_start = dates[sl["val"]][-1], dates[sl["test"]][0]
    check("train < val < test ตามเวลา ไม่ทับกัน", tr_end < va_start and va_end < te_start,
          f"train ถึง {str(tr_end)[:10]}, val {str(va_start)[:10]}..{str(va_end)[:10]}, "
          f"test จาก {str(te_start)[:10]}")
    check("จำนวน sample ครบ ไม่หาย/ไม่ซ้ำ",
          sum(len(p[1]) for p in parts.values()) == len(X))
    mu_tr = X[sl["train"]].mean(axis=(0, 2, 3), keepdims=True)
    mu_all = X.mean(axis=(0, 2, 3), keepdims=True)
    check("normalize ใช้สถิติจาก train เท่านั้น",
          np.allclose(mu, mu_tr) and not np.allclose(mu, mu_all))


def label_control_test(stock, seeds=(0, 1, 2)):
    """
    ตัวควบคุมบวก : label ที่หาได้จาก input จริง (ความผันผวน 5 วันล่าสุด > median ของ train)
                   -> โมเดลต้องทำได้ดีกว่า 50% ชัดเจน (พิสูจน์ว่า pipeline เรียนรู้ได้จริง)
    ตัวควบคุมลบ : label สุ่ม -> ต้องได้ราว 50% (ถ้าสูงผิดปกติแปลว่ามี leak/bug)
    """
    df = load_stock(stock)
    X, y_par, dates = build_dataset(df)
    rets = df["ret_close"].to_numpy()
    vol5 = np.array([rets[t - 5:t].std() for t in range(WINDOW, len(df))])

    n_tr = int(len(X) * SPLIT[0])
    y_pos = (vol5 > np.median(vol5[:n_tr])).astype(np.float32)
    y_neg = np.random.default_rng(42).integers(0, 2, len(X)).astype(np.float32)

    for name, yy, cond in [("ตัวควบคุมบวก (volatility label) ทำได้ > 65%", y_pos, lambda a: a > 0.65),
                           ("ตัวควบคุมลบ (random label) อยู่ระหว่าง 30-70%", y_neg,
                            lambda a: 0.30 <= a <= 0.70)]:
        parts, _, _, _ = split_and_normalize(X, yy, SPLIT)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
        accs = [accuracy_score(yte, predict_proba(train_one(Xtr, ytr, Xva, yva, s)[0], Xte) > 0.5)
                for s in seeds]
        check(name, cond(np.mean(accs)), f"test acc เฉลี่ย {np.mean(accs):.3f}")


def main():
    for stock in STOCKS:
        print(f"\n===== {stock} =====")
        print("- คุณภาพข้อมูล")
        data_quality(stock)
        print("- Data leakage: แก้ข้อมูลอนาคต")
        future_perturbation_test(stock)
        print("- Data leakage: การแบ่งข้อมูล / normalization")
        split_and_norm_test(stock)
        print("- Sanity check ของโมเดล")
        label_control_test(stock)
    print("\nสรุป:", "ผ่านทั้งหมด" if not failures else f"ไม่ผ่าน {len(failures)} ข้อ: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
