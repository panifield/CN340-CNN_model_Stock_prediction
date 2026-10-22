"""
models.py — งาน B (ราคาปิด / return)
====================================
นิยามโมเดลของงาน B: ANN + Random Forest + XGBoost (regression)

*** เรื่อง ANN ***
ที่นี่ใช้ MLPRegressor ของ sklearn ซึ่งเป็น Multi-Layer Perceptron = ANN
ตามนิยามจริง ข้อดี: ไม่ต้องลง TensorFlow ให้ยุ่งยาก
ANN ถูกเทรนซ้ำ 10 seeds แล้วเฉลี่ยการทำนาย (C3) -- ดู SeedAveragedRegressor

*** เรื่อง Scaling ***
ANN ต้อง standardize ไม่งั้นไม่ converge
tree-based ไม่ต้อง scale

ใช้ Pipeline ของ sklearn ครอบไว้ ทำให้ scaler ถูก fit
เฉพาะบน train set โดยอัตโนมัติ -> ไม่มีทาง leak

*** เรื่อง reproducibility ***
แก้ 2026-09-29: RF_PARAMS/XGB_PARAMS เปลี่ยนเป็น n_jobs=1 แล้ว (เดิม -1)
เพราะเทรนขนานทำให้ลำดับการรวมผลไม่คงที่จริง -- ทดสอบรัน main.py --dev ซ้ำ 5
ครั้งด้วยข้อมูลเดียวกัน เจอ diff 2 ใน 4 ครั้ง (บางครั้งถึง ~1e-7 ไม่ใช่แค่
~1e-18 ตามที่เคยคาดไว้ในคอมเมนต์เดิม) ทำให้ regression-check guard ของ
daily_next_day.py (git diff results/*_val.csv ต้องว่าง) false-positive
แบบสุ่มได้แม้ข้อมูลไม่เปลี่ยนเลย -- n_jobs=1 การันตี byte-identical จริง
ไม่ใช่แค่ "numerically reproducible within floating-point precision" อีก
ต่อไป (ดู KNOWN_ISSUES.md)

SimpleImputer เก็บไว้เป็นตาข่ายนิรภัยเท่านั้น -- splits.prepare_xy ตัดแถว
ที่มี NaN ทิ้งหมดแล้ว (A2) ในทางปฏิบัติ imputer จึงไม่ได้เติมค่าอะไรเลย
"""

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from config import (
    ANN_PARAMS, ANN_SEEDS, RF_PARAMS, XGB_PARAMS, RECENCY_HALF_LIFE,
)
from weighting import recency_weights


class SeedAveragedRegressor(RegressorMixin, BaseEstimator):
    """เทรนโมเดลเดิมซ้ำด้วย random_state หลายค่า แล้วเฉลี่ยการทำนาย"""

    def __init__(self, estimator=None, seeds=(0,)):
        self.estimator = estimator
        self.seeds = seeds

    def fit(self, X, y, sample_weight=None):
        """
        sample_weight ถูกส่งต่อให้ทุก seed ตรง ๆ (MLPRegressor รองรับตั้งแต่ sklearn 1.7)
        sample_weight = None -> เรียก fit(X, y) แบบเดิมทุกประการ
        (ต้องให้ผลเหมือนก่อนเพิ่มฟีเจอร์นี้เป๊ะ)
        """
        if sample_weight is not None and len(sample_weight) != len(X):
            raise ValueError(
                f"sample_weight ยาว {len(sample_weight)} แต่ X มี {len(X)} แถว"
            )
        self.estimators_ = []
        for s in self.seeds:
            est = clone(self.estimator).set_params(random_state=s)
            if sample_weight is None:
                est.fit(X, y)
            else:
                est.fit(X, y, sample_weight=sample_weight)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)


def _scaled(estimator):
    """ครอบโมเดลด้วย imputer + scaler"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _unscaled(estimator):
    """สำหรับ tree-based ที่ไม่ต้อง scale (imputer เป็นตาข่ายนิรภัย)"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def get_regressors():
    """
    ANN + Random Forest + XGBoost สำหรับทำนาย return

    หมายเหตุ: ค่า return มีขนาดเล็กมาก (~0.01)
    ใน main.py จะห่อด้วย TransformedTargetRegressor เพื่อ scale target ให้
    """
    ann = _scaled(SeedAveragedRegressor(MLPRegressor(**ANN_PARAMS),
                                        seeds=tuple(ANN_SEEDS)))
    rf = _unscaled(RandomForestRegressor(**RF_PARAMS))
    xgb = _unscaled(XGBRegressor(**XGB_PARAMS))
    return {"ANN (MLP)": ann, "Random Forest": rf, "XGBoost": xgb}


def fit_live_models(X, y, verbose=True):
    """
    เทรนโมเดลทั้ง 3 ตัวด้วยข้อมูลทั้งหมดที่มี y จริงแล้ว (B2)
    ใช้สำหรับ prospective prediction เท่านั้น

    *** ต่างจาก main.py โดยตั้งใจ ***
    main.py เทรนถึง train_end (2023-09-01) เพื่อประเมินบน val
    ส่วนที่นี่เทรนถึงวันล่าสุด เพราะ "การทดสอบ" คืออนาคตที่ยังไม่เกิด
    ไม่ใช่ข้อมูลที่กันไว้ -- ช่วงที่เคยเป็น historical test ก็เป็นแค่
    labeled data ที่รู้ผลแล้ว ณ เวลาที่ทำนายวันข้างหน้า

    *** ห้ามเอาโมเดลจากฟังก์ชันนี้ไปรายงานเป็นผล test set เด็ดขาด ***
    มันเห็นช่วง test ไปแล้ว -- คนละวัตถุประสงค์ คนละไฟล์ คนละ log

    recency weighting (§3.5): half-life อ่านจาก config.RECENCY_HALF_LIFE
    ตัวเดียวกับ main.py และคำนวณน้ำหนักด้วย recency_weights() ตัวเดียวกัน
    -> แถวใหม่สุดของข้อมูล live ได้น้ำหนักมากสุด · ห้าม hardcode / ห้ามมี default แยก

    คืน (fitted, half_lives) -- half_lives[name] คือค่าที่ใช้จริง ไว้บันทึกลง log
    """
    fitted, half_lives = {}, {}
    for name, model in get_regressors().items():
        half_life = RECENCY_HALF_LIFE[name]      # KeyError = ชื่อไม่ตรง -> ต้องพัง
        w = recency_weights(len(X), half_life)
        fit_params = {} if w is None else {"model__sample_weight": w}
        if verbose:
            print(f"[live] {name}: recency half-life = {half_life} "
                  f"({'ถ่วงน้ำหนัก' if half_life else 'ไม่ถ่วงน้ำหนัก'})")
            print(f"    เทรน {name} บน {len(y)} แถว ...", end=" ", flush=True)
        wrapped = TransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
        wrapped.fit(X, y, **fit_params)
        fitted[name] = wrapped
        half_lives[name] = half_life
        if verbose:
            print("เสร็จ")
    return fitted, half_lives
