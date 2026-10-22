"""
signal_diagnostic.py — diagnostic เรื่อง "สัญญาณในข้อมูล" โดยไม่เทรนโมเดลใด ๆ (DECISIONS B26)
=================================================================================================
*** diagnostic — ห้ามใช้เลือก feature หรือ config *** (ผู้ใช้สั่ง · ป้ายนี้อยู่ในทุกไฟล์ผลลัพธ์)

ไฟล์นี้ทำอะไร (ไม่มีการเทรน · ไม่ import torch)
    1. autocorrelation ของ ret_close ที่ lag 1–5 ตลอดโซน dev แยกหุ้น (+ lag-1 ในแต่ละช่วงประเมินทั้ง 6)
    2. IC (Pearson และ Spearman) ของ feature เดิมทั้ง 29 ตัวกับ y แยกตาม 6 evaluation เรียงจากมากไปน้อย
       พร้อมธงว่า feature ไหนมีเครื่องหมาย IC คงที่ (เหมือนกัน ≥ 5 จาก 6 evaluation)
    3. เทียบ IC ของ feature ที่ดีที่สุดกับ IC ของ RF / MLP / XGBoost (จากคำทำนายของ C3 บนวันเดียวกัน)
    4. ข้อมูลอ้างอิงเพื่อให้ธงข้อ 2 ตีความได้: permutation null (สลับ y ภายในแต่ละช่วงประเมิน)

ทำไมต้องมี
    ผู้ใช้ตั้งสมมติฐานว่าตัวเทียบมี IC บวกเพราะ "return lag" (สัญญาณกลับตัวระยะสั้น) — diagnostic นี้ดูตรง ๆ ว่ามีสัญญาณแบบนั้นในข้อมูลหรือไม่
    โดยไม่ต้องรอโมเดล · และเพราะมี 29 feature × 6 evaluation การที่บาง feature "ดูคงที่" เกิดโดยบังเอิญได้มาก จึงต้องมี null อ้างอิง

รัน:  python -m src.signal_diagnostic
"""

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src import data, paths
from src.check_data import md_table
from src.settings import TICKERS, VAL_END
from src.spectrogram import raw_channels

LABEL = "diagnostic — ห้ามใช้เลือก feature หรือ config"
MAX_LAG = 5
N_PERM = 1000
PERM_SEED = 0
CONSISTENT_MIN = 5                      # ธง "คงที่" = เครื่องหมายเหมือนกัน >= 5 จาก 6 evaluation
EVALS = [(t, k) for t in TICKERS for k in (1, 2, 3)]
EVAL_LABELS = [f"{t[0]}{k}" for t, k in EVALS]
OUT_DIR = paths.RESULTS_DIR / "diagnostic_signal"


# ---------------------------------------------------------------
# 1. autocorrelation
# ---------------------------------------------------------------
def acf(x, max_lag=MAX_LAG):
    """
    sample autocorrelation มาตรฐาน : Σ (x_t − m)(x_{t+k} − m) / Σ (x_t − m)²  ที่ lag 1…max_lag
    (ตัวหารเป็นผลรวมกำลังสองทั้งชุด — ตัวประมาณแบบเดียวกับ statsmodels.acf)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    d = x - x.mean()
    denom = float(np.dot(d, d))
    return np.array([float(np.dot(d[:-k], d[k:])) / denom for k in range(1, max_lag + 1)])


def acf_table(datasets):
    """ACF ของ ret_close ตลอดโซน dev ต่อหุ้น · z = ACF·√n เทียบ ±1.96 (ภายใต้ white noise)"""
    rows = []
    for t, ds in datasets.items():
        r = raw_channels(ds.prices)["ret_close"].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        for lag, a in enumerate(acf(r), start=1):
            z = a * np.sqrt(n)
            rows.append({"ticker": t, "lag": lag, "ACF": float(a), "n": n, "z": float(z), "เกิน ±1.96": bool(abs(z) > 1.96)})
    return pd.DataFrame(rows)


def lag1_by_eval(datasets, folds):
    """
    lag-1 correlation ในแต่ละช่วงประเมิน : corr(ret_close[t], ret_close[t−1]) เหนือวันประเมิน t ของ evaluation นั้น
    (คู่ชุดเดียวกับที่ IC ของ feature ret_1d_prev ใช้ ต่างแค่ log กับ simple return)
    """
    rows = []
    for t, ds in datasets.items():
        r = raw_channels(ds.prices)["ret_close"].to_numpy(dtype=float)
        for f in folds:
            p = f.eval_pos
            rows.append({"ticker": t, "fold": f.k, "lag1_corr": float(np.corrcoef(r[p], r[p - 1])[0, 1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 2. IC ของ 29 feature
# ---------------------------------------------------------------
def pearson_matrix(X, y):
    """corr(X[:, j], y) ทุกคอลัมน์พร้อมกัน · คอลัมน์คงที่ → NaN"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def spearman_matrix(X, y):
    """Spearman = Pearson ของอันดับ (ค่าซ้ำใช้อันดับเฉลี่ย)"""
    Xr = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    return pearson_matrix(Xr, rankdata(y))


def feature_ic_by_eval(datasets, folds):
    """ตารางยาว: ticker, fold, feature, pearson, spearman — ในแต่ละช่วงประเมิน (266 วัน)"""
    rows = []
    for t, ds in datasets.items():
        cols = list(ds.feat29.columns)
        for f in folds:
            ev = f.eval_pos
            assert ds.ok29[ev].all(), "วันประเมินต้องมี 29 feature ครบทุกวัน"
            X = ds.feat29.iloc[ev].to_numpy(dtype=float)
            y = ds.y[ev]
            pe, sp = pearson_matrix(X, y), spearman_matrix(X, y)
            rows += [{"ticker": t, "fold": f.k, "feature": c, "pearson": float(pe[j]), "spearman": float(sp[j])}
                     for j, c in enumerate(cols)]
    return pd.DataFrame(rows)


def summarize_features(long):
    """
    ต่อ feature: IC ราย evaluation (Pearson) + ค่าเฉลี่ย Pearson/Spearman + จำนวนบวก/ลบ + ธง "คงที่"
    ธง "คงที่" = เครื่องหมาย IC (Pearson) เหมือนกัน (บวกทั้งหมดหรือลบทั้งหมด) อย่างน้อย CONSISTENT_MIN จาก 6 evaluation
    เรียงจากมากไปน้อยตาม IC Pearson เฉลี่ย
    """
    pe = long.pivot_table(index="feature", columns=["ticker", "fold"], values="pearson")
    pe = pe[EVALS]
    sp = long.pivot_table(index="feature", columns=["ticker", "fold"], values="spearman")[EVALS]
    out = pd.DataFrame(index=pe.index)
    for lab, col in zip(EVAL_LABELS, EVALS):
        out[lab] = pe[col]
    out["mean_pearson"] = pe.mean(axis=1)
    out["mean_spearman"] = sp.mean(axis=1)
    out["n_pos"] = (pe > 0).sum(axis=1)
    out["n_neg"] = (pe < 0).sum(axis=1)
    out["consistent"] = np.maximum(out["n_pos"], out["n_neg"]) >= CONSISTENT_MIN
    out["sign"] = np.where(out["n_pos"] >= CONSISTENT_MIN, "+", np.where(out["n_neg"] >= CONSISTENT_MIN, "−", ""))
    return out.sort_values("mean_pearson", ascending=False)


# ---------------------------------------------------------------
# 4. permutation null (ข้อมูลอ้างอิง)
# ---------------------------------------------------------------
def chance_consistency_probability(n_eval=6, min_same=CONSISTENT_MIN):
    """ถ้าเครื่องหมายแต่ละ evaluation เป็นเหรียญยุติธรรมอิสระกัน: P(เหมือนกัน ≥ min_same จาก n_eval) = 2·Σ_{k≥min} C(n,k)/2^n"""
    from math import comb
    return 2 * sum(comb(n_eval, k) for k in range(min_same, n_eval + 1)) / 2 ** n_eval


def permutation_null(datasets, folds, n_perm=N_PERM, seed=PERM_SEED):
    """
    สลับ y ภายในแต่ละช่วงประเมิน (ทำลายความสัมพันธ์กับ feature) แล้วคิดสถิติเดียวกับของจริง n_perm ครั้ง
    คืน dict ของอาร์เรย์ความยาว n_perm : n_consistent · max_abs_mean_ic · max_signed_mean_ic
    สถิติ 6 evaluation ในหนึ่งรอบใช้การสลับคนละชุดกัน (สุ่มต่อเนื่องจาก rng ตัวเดียว)
    """
    rng = np.random.default_rng(seed)
    per_eval = []                                              # แต่ละ evaluation: (n_perm, 29)
    n_feat = None
    for t, ds in datasets.items():
        X = ds.feat29.to_numpy(dtype=float)
        for f in folds:
            ev = f.eval_pos
            Z = X[ev] - X[ev].mean(axis=0)
            Z = Z / np.sqrt((Z ** 2).sum(axis=0))              # คอลัมน์ละ norm 1
            y = ds.y[ev]
            yz = (y - y.mean()) / np.sqrt(((y - y.mean()) ** 2).sum())
            idx = np.argsort(rng.random((n_perm, len(ev))), axis=1)          # แถวละหนึ่ง permutation
            per_eval.append(yz[idx] @ Z)                      # (n_perm, 29) = corr ของ feature กับ y ที่ถูกสลับ
            n_feat = Z.shape[1]
    stack = np.stack(per_eval, axis=0)                        # (6, n_perm, 29)
    n_pos, n_neg = (stack > 0).sum(axis=0), (stack < 0).sum(axis=0)         # (n_perm, 29)
    consistent = np.maximum(n_pos, n_neg) >= CONSISTENT_MIN
    mean_ic = stack.mean(axis=0)
    return {"n_consistent": consistent.sum(axis=1), "max_abs_mean_ic": np.abs(mean_ic).max(axis=1),
            "max_signed_mean_ic": mean_ic.max(axis=1), "n_features": n_feat}


def null_summary(null):
    """ค่าเฉลี่ยและเปอร์เซ็นไทล์ที่ 95 ของแต่ละสถิติ"""
    return {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))}
            for k, v in null.items() if k != "n_features"}


# ---------------------------------------------------------------
# 3. เทียบกับโมเดล
# ---------------------------------------------------------------
def model_ic_table(pred_path, cnn_evals_path):
    """IC (Pearson/Spearman) ต่อ evaluation ของ ANN/RF/XGB จากคำทำนายของ C3 + CNN base (อ้างอิง)"""
    p = pd.read_csv(pred_path, float_precision="round_trip")
    rows = []
    for name in ("ANN (MLP)", "Random Forest", "XGBoost"):
        for (t, k), g in p[p["model"] == name].groupby(["ticker", "fold"]):
            rows.append({"model": name, "ticker": t, "fold": int(k), "pearson": float(np.corrcoef(g["pred"], g["y"])[0, 1]),
                         "spearman": float(spearmanr(g["pred"], g["y"]).statistic)})
    cnn = pd.read_csv(cnn_evals_path)
    cnn = cnn[cnn["config"] == "base"]
    rows += [{"model": "CNN→ANN base (อ้างอิง)", "ticker": r.ticker, "fold": int(r.fold), "pearson": float(r.IC),
              "spearman": float(r.IC_spearman)} for r in cnn.itertuples()]
    return pd.DataFrame(rows)


def compare_models(summary, model_ic):
    """แถวเทียบ: feature ที่ดีที่สุด (ตามค่าที่มีเครื่องหมาย และตามค่าสัมบูรณ์) กับ IC ของโมเดล — ทุกแถวคือ IC เฉลี่ย 6 evaluation"""
    rows = []
    top = summary.index[0]
    top_abs = summary["mean_pearson"].abs().idxmax()
    sgn = np.sign(summary.loc[top_abs, "mean_pearson"])
    for label, feat, flip in ((f"feature ดีที่สุด (มีเครื่องหมาย): {top}", top, 1.0),
                              (f"feature ดีที่สุด (|IC|): {top_abs} · ใช้ตามเครื่องหมายของมัน", top_abs, sgn)):
        rows.append({"รายการ": label, "Pearson เฉลี่ย": float(flip * summary.loc[feat, "mean_pearson"]),
                     "Spearman เฉลี่ย": float(flip * summary.loc[feat, "mean_spearman"]),
                     "IC>0": int((flip * summary.loc[feat, EVAL_LABELS] > 0).sum())})
    for m, g in model_ic.groupby("model", sort=False):
        rows.append({"รายการ": m, "Pearson เฉลี่ย": float(g["pearson"].mean()), "Spearman เฉลี่ย": float(g["spearman"].mean()),
                     "IC>0": int((g["pearson"] > 0).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def fmt(v, k="s3"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return {"s3": f"{v:+.3f}", "s4": f"{v:+.4f}", "r2": f"{v:.2f}"}[k]


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths.assert_output_dir(OUT_DIR)
    datasets = data.build_all(VAL_END)
    folds = data.make_folds(datasets[TICKERS[0]].dates)

    acf_df = acf_table(datasets)
    lag1 = lag1_by_eval(datasets, folds)
    long = feature_ic_by_eval(datasets, folds)
    summ = summarize_features(long)
    null = permutation_null(datasets, folds)
    ns = null_summary(null)
    p_chance = chance_consistency_probability()
    n_feat = int(null["n_features"])
    models = model_ic_table(paths.RESULTS_DIR / "stage0_base" / "comparators_predictions.csv",
                            paths.RESULTS_DIR / "stage1_representation" / "evals.csv")
    cmp_df = compare_models(summ, models)

    obs_consistent = int(summ["consistent"].sum())
    obs_max_abs = float(summ["mean_pearson"].abs().max())
    obs_max_signed = float(summ["mean_pearson"].max())

    acf_df.to_csv(OUT_DIR / "acf.csv", index=False)
    lag1.to_csv(OUT_DIR / "lag1_by_eval.csv", index=False)
    long.to_csv(OUT_DIR / "feature_ic_by_eval.csv", index=False)
    summ.to_csv(OUT_DIR / "feature_ic_summary.csv")
    models.to_csv(OUT_DIR / "model_ic.csv", index=False)
    (OUT_DIR / "permutation_null.json").write_text(json.dumps({
        "label": LABEL, "n_perm": N_PERM, "seed": PERM_SEED, "chance_probability_per_feature": p_chance,
        "expected_consistent_by_chance_independent": p_chance * n_feat,
        "null_summary": ns, "observed": {"n_consistent": obs_consistent, "max_abs_mean_ic": obs_max_abs,
                                         "max_signed_mean_ic": obs_max_signed}}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- รายงาน ----------------
    acf_pivot = acf_df.pivot(index="lag", columns="ticker", values="ACF")
    acf_rows = []
    for lag in range(1, MAX_LAG + 1):
        row = {"lag": lag}
        for t in TICKERS:
            r = acf_df[(acf_df.ticker == t) & (acf_df.lag == lag)].iloc[0]
            row[t] = f"{r['ACF']:+.4f} (z {r['z']:+.2f}{' *' if r['เกิน ±1.96'] else ''})"
        acf_rows.append(row)
    n_days = int(acf_df["n"].iloc[0])
    lag1_rows = [{"หุ้น": t, **{f"fold {k}": fmt(float(lag1[(lag1.ticker == t) & (lag1.fold == k)]["lag1_corr"].iloc[0]), "s3") for k in (1, 2, 3)}}
                 for t in TICKERS]
    feat_rows = [{"feature": f, **{lab: fmt(float(r[lab])) for lab in EVAL_LABELS}, "Pearson เฉลี่ย": fmt(float(r["mean_pearson"]), "s4"),
                  "Spearman เฉลี่ย": fmt(float(r["mean_spearman"]), "s4"), "+/−": f"{int(r['n_pos'])}/{int(r['n_neg'])}",
                  "คงที่ ≥ 5/6": r["sign"] if r["consistent"] else ""} for f, r in summ.iterrows()]
    cmp_rows = [{c: (fmt(v, "s4") if isinstance(v, float) else v) for c, v in r.items()} for r in cmp_df.to_dict("records")]
    null_rows = [
        {"สถิติ": "จำนวน feature ที่เครื่องหมายคงที่ (≥ 5/6)", "ของจริง": obs_consistent,
         "null เฉลี่ย": f"{ns['n_consistent']['mean']:.1f}", "null P95": f"{ns['n_consistent']['p95']:.0f}",
         "หมายเหตุ": f"ถ้าอิสระกัน คาด {p_chance:.1%} × {n_feat} = {p_chance * n_feat:.1f}"},
        {"สถิติ": "|IC เฉลี่ย| สูงสุดของ feature", "ของจริง": fmt(obs_max_abs, "s4"),
         "null เฉลี่ย": fmt(ns["max_abs_mean_ic"]["mean"], "s4"), "null P95": fmt(ns["max_abs_mean_ic"]["p95"], "s4"), "หมายเหตุ": ""},
        {"สถิติ": "IC เฉลี่ยสูงสุด (มีเครื่องหมาย)", "ของจริง": fmt(obs_max_signed, "s4"),
         "null เฉลี่ย": fmt(ns["max_signed_mean_ic"]["mean"], "s4"), "null P95": fmt(ns["max_signed_mean_ic"]["p95"], "s4"), "หมายเหตุ": ""}]

    md = [f"# Diagnostic สัญญาณในข้อมูล (ไม่เทรนโมเดล) — **{LABEL}**", "",
          f"> **{LABEL}** · ไม่มีการเทรนโมเดล · โซน dev ≤ {VAL_END} · นิยามตาม DECISIONS B26", "",
          "ชื่อ evaluation: K = KBANK · A = ADVANC · เลข = fold · IC = corr(feature, y) โดย y = Close(t)/Close(t−1) − 1 บนวันประเมิน 266 วันของ evaluation นั้น", "",
          f"## 1. autocorrelation ของ ret_close (โซน dev ทั้งหมด · n = {n_days} วันต่อหุ้น)", "",
          "ค่าในวงเล็บ = z = ACF·√n เทียบ ±1.96 ภายใต้ white noise · `*` = เกิน ±1.96", "", md_table(pd.DataFrame(acf_rows)), "",
          "lag-1 correlation ในแต่ละช่วงประเมิน (คู่ (t−1, t) ของวันประเมิน)", "", md_table(pd.DataFrame(lag1_rows)), "",
          "## 2. IC (Pearson) ของ 29 feature กับ y แยกตาม 6 evaluation — เรียงจากมากไปน้อย", "",
          f"`คงที่ ≥ 5/6` = เครื่องหมาย IC เหมือนกัน ≥ {CONSISTENT_MIN} จาก 6 evaluation (`+` บวกคงที่ · `−` ลบคงที่) · **ดูข้อ 4 ก่อนตีความธงนี้**", "",
          md_table(pd.DataFrame(feat_rows)), "",
          "## 3. เทียบ IC ของ feature ที่ดีที่สุดกับโมเดล (IC เฉลี่ย 6 evaluation)", "", md_table(pd.DataFrame(cmp_rows)), "",
          "## 4. ข้อมูลอ้างอิง — permutation null (สลับ y ภายในแต่ละช่วงประเมิน " + f"{N_PERM:,} ครั้ง seed {PERM_SEED})", "",
          md_table(pd.DataFrame(null_rows)), "",
          f"> **{LABEL}**", ""]
    (OUT_DIR / "diagnostic_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"[{LABEL}]\n")
    print(f"== 1. ACF ของ ret_close (n = {n_days}) ==")
    print(pd.DataFrame(acf_rows).to_string(index=False))
    print("\nlag-1 corr ในแต่ละช่วงประเมิน:")
    print(pd.DataFrame(lag1_rows).to_string(index=False))
    print("\n== 2. IC (Pearson) ของ 29 feature ==")
    print(pd.DataFrame(feat_rows).to_string(index=False))
    print("\n== 3. เทียบกับโมเดล ==")
    print(pd.DataFrame(cmp_rows).to_string(index=False))
    print("\n== 4. permutation null ==")
    print(pd.DataFrame(null_rows).to_string(index=False))
    print(f"\nเขียน {OUT_DIR / 'diagnostic_summary.md'} · {time.time() - t0:.0f} วินาที\n[{LABEL}]")


if __name__ == "__main__":
    main()
