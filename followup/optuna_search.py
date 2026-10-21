"""
ข้อ 3: HyperOps -- ใช้ Optuna (TPE) ค้นหา feature + hyperparameter ของ Task A

กันการ "หาดวง":
  1. Optuna เห็นแค่ walk-forward 4 fold (selection, 80 วัน/หุ้น) ของช่วงโจทย์ (2025-09 -> ปัจจุบัน)
  2. best trial ถูกวัดผลบน holdout 32 วันสุดท้าย/หุ้น ซึ่ง Optuna ไม่เคยเห็น
  3. p-value ของ best trial บน selection ต้องผ่าน Bonferroni (0.05 / จำนวน trial) จึงจะถือว่าดีจริง
  4. รัน best config ซ้ำบน 9 ปีก่อนหน้า (2016-2025) ด้วยขั้นตอนเดียวกัน -- ข้อมูลที่ไม่เคยใช้เลือกเลย

Usage:
    python followup/optuna_search.py --trials 50 --workers 8
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "A"))
sys.path.insert(0, str(HERE.parent / "A" / "experiments"))
import search as S  # noqa: E402

OUT = HERE / "results" / "optuna"
STOCKS = ["KBANK", "ADVANC"]
TRIAL_SEEDS = (0, 1)          # ระหว่างค้นหาใช้ 2 seed (ประหยัดเวลา)
FINAL_SEEDS = (0, 1, 2)       # วัดผลจริงใช้ 3 seed เท่ากับ config อื่น


def suggest(trial):
    transform = trial.suggest_categorical("transform", ["stft", "stft_complex", "cwt"])
    nperseg = trial.suggest_categorical("nperseg", [6, 8, 10, 16])
    chans = [c for c in S.CH6 if trial.suggest_categorical(f"ch_{c}", [True, False])] or ["ret_close"]
    extras = trial.suggest_categorical("extras", [True, False])
    cfg = dict(S.BASE, transform=transform, window=trial.suggest_categorical("window", [20, 30, 45, 60]),
               nperseg=nperseg, noverlap=nperseg - 2, channels=chans, extras=extras,
               rounds=extras and trial.suggest_categorical("rounds", [True, False]),
               arch=trial.suggest_categorical("arch", ["small", "current", "deep"]),
               noise=trial.suggest_categorical("noise", [0.0, 0.1, 0.3]),
               lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
               weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
               batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
               bottleneck=trial.suggest_categorical("bottleneck", [0, 4, 8, 16]),
               skip=trial.suggest_categorical("skip", ["none", "linear", "mlp"]) if extras else "none",
               seeds=TRIAL_SEEDS, desc="optuna")
    cfg["skip"] = None if cfg["skip"] == "none" else cfg["skip"]
    return cfg


def run_periods(ex, cfg, tag, kinds):
    periods = [p for p in S.periods_for(tag) if p[0] in kinds]
    futs = [ex.submit(S.run_task, "optuna", [s], tag, k, a, b, cfg) for s in STOCKS for k, a, b in periods]
    rows = [r for f in futs for r in f.result()]
    df = pd.DataFrame(rows)
    df["correct"] = ((df.p_odd > 0.5).astype(int) == df.y).astype(int)
    return df


def acc_p(df):
    k, n = int(df.correct.sum()), len(df)
    return k / n, n, binomtest(k, n, 0.5, alternative="greater").pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name="taskA",
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=12),
                                storage=f"sqlite:///{(OUT / 'study.db').as_posix()}", load_if_exists=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        def objective(trial):
            cfg = suggest(trial)
            df = run_periods(ex, cfg, S.MAIN_TAG, ["selection"])
            acc, n, p = acc_p(df)
            trial.set_user_attr("p_sel", p)
            trial.set_user_attr("cfg", {k: v for k, v in cfg.items() if k != "desc"})
            print(f"[{time.time() - t0:5.0f}s] trial {trial.number:3d} acc_sel={acc:.3f} (n={n}, p={p:.3f})",
                  flush=True)
            return acc

        todo = args.trials - len(study.trials)
        if todo > 0:
            study.optimize(objective, n_trials=todo)

        trials = study.trials_dataframe()
        trials.to_csv(OUT / "trials.csv", index=False)
        n_trials = len(study.trials)
        best = study.best_trial
        cfg = dict(best.user_attrs["cfg"], seeds=FINAL_SEEDS, desc="optuna best")
        cfg["channels"] = list(cfg["channels"])
        print("best:", best.number, best.value, json.dumps(best.params, ensure_ascii=False), flush=True)

        # วัดผล best config ด้วย 3 seed: selection (ค่าที่ใช้เลือก), holdout, และ 9 ปีก่อนหน้า
        main = run_periods(ex, cfg, S.MAIN_TAG, ["selection", "holdout"])
        rep_tags = [f"{y}-09_{y + 1}-08" for y in S.REPLICATE_YEARS]
        rep = pd.concat([run_periods(ex, cfg, t, ["replicate"]) for t in rep_tags], ignore_index=True)
    main.to_csv(OUT / "best_main_preds.csv", index=False)
    rep.to_csv(OUT / "best_replication_preds.csv", index=False)

    # baseline (config เดิม) จากผลที่บันทึกไว้
    base_main = pd.concat([S.load_preds(S.MAIN_TAG, [c]) for c in ("base", "hybrid_extras")])
    base_main["correct"] = ((base_main.p_odd > 0.5).astype(int) == base_main.y).astype(int)
    base_rep = pd.read_csv(S.RESULTS / "replication" / "predictions.csv")
    base_rep = base_rep[base_rep.config.isin(["base", "hybrid_extras"])]

    bonf = 0.05 / n_trials
    rows = []
    for name, sel, hold, rp in [
        ("optuna best", main[main.period == "selection"], main[main.period == "holdout"], rep),
        *[(c, base_main[(base_main.config == c) & (base_main.period == "selection")],
           base_main[(base_main.config == c) & (base_main.period == "holdout")], base_rep[base_rep.config == c])
          for c in ("base", "hybrid_extras")]]:
        a1, n1, p1 = acc_p(sel)
        a2, n2, p2 = acc_p(hold)
        a3, n3, p3 = acc_p(rp)
        rows.append(dict(model=name, acc_sel=a1, p_sel=p1, acc_hold=a2, n_hold=n2, p_hold=p2,
                         acc_rep=a3, n_rep=n3, p_rep=p3))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")

    # ---- plot: ประวัติการค้นหา + ผลเทียบ
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    vals = trials["value"].to_numpy()
    ax = axes[0]
    band = 1.96 * np.sqrt(0.25 / 160)
    ax.axhspan(0.5 - band, 0.5 + band, color="#d9d9d6", alpha=0.6, lw=0, label="95% range of guessing")
    ax.scatter(np.arange(len(vals)), vals, s=18, color="#2a78d6", label="trial (selection acc)")
    ax.plot(np.maximum.accumulate(vals), color="#eb6834", lw=2, label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("walk-forward accuracy (selection)")
    ax.set_title(f"Optuna search: {n_trials} trials", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    x = np.arange(3)
    for i, (c, col) in enumerate(zip(["acc_sel", "acc_hold", "acc_rep"], ["#2a78d6", "#eb6834", "#1b7f4b"])):
        ax.bar(x + (i - 1) * 0.26, res[c], width=0.26, color=col,
               label=["selection (used to pick)", "holdout 2026", "replication 9 yr"][i])
        for xi, v in zip(x, res[c]):
            ax.text(xi + (i - 1) * 0.26, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, color="#8a8a85", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(res.model)
    ax.set_ylim(0.35, 0.75)
    ax.set_title("Best trial vs. original configs", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for a in axes:
        a.grid(axis="y", color="#ececea")
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "optuna.png", dpi=140)
    plt.close(fig)

    b = res.iloc[0]
    L = ["# Optuna hyperparameter search (Task A)", "",
         f"- ค้นหา {n_trials} trial (TPE) บน walk-forward selection เท่านั้น (80 วัน/หุ้น x 2 หุ้น = 160 วัน)",
         f"- best trial #{best.number}: selection acc {best.value:.3f} (2 seed ระหว่างค้นหา)",
         f"- เกณฑ์ Bonferroni = 0.05 / {n_trials} = {bonf:.5f} -> p ของ best (3 seed) = {b.p_sel:.4f} -> "
         f"**{'ผ่าน' if b.p_sel < bonf else 'ไม่ผ่าน'}**", "",
         "## ค่าที่ดีที่สุด", "", "```json", json.dumps(best.params, indent=2, ensure_ascii=False), "```", "",
         "## ผลเทียบ", "",
         "| โมเดล | selection (ใช้เลือก) | p | holdout | p | replication 9 ปี | n | p |", "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r.model} | {r.acc_sel:.3f} | {r.p_sel:.3g} | **{r.acc_hold:.3f}** | {r.p_hold:.3g} | "
                 f"**{r.acc_rep:.3f}** | {r.n_rep} | {r.p_rep:.3g} |")
    try:
        imp = optuna.importance.get_param_importances(study)
        L += ["", "## ความสำคัญของ hyperparameter (fANOVA)", "", "| param | importance |", "|---|---|"]
        L += [f"| {k} | {v:.3f} |" for k, v in list(imp.items())[:10]]
    except Exception as e:  # noqa: BLE001
        L += ["", f"(คำนวณ importance ไม่ได้: {e})"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
