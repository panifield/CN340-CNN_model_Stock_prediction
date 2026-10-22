"""
configs.py — "ใบสั่ง" ของการทดลองหนึ่งครั้ง (Config) + ค่า base config ตาม SPEC ข้อ 5.1
======================================================================================
ไฟล์นี้ทำอะไร
    เก็บทุกตัวเลือกที่การทดลองหนึ่งครั้งต้องรู้ไว้ใน dataclass เดียว (`Config`) — ตั้งแต่วิธีทำภาพ
    ไปจนถึงโครง CNN, input ของ ANN, loss และ optimizer · `BASE` = ค่าตั้งต้นตามสเปกข้อ 5.1

ทำไมต้องมี
    1. การทดลองแต่ละ Stage คือ "เปลี่ยนจาก BASE ทีละอย่าง" → เขียนเป็น `replace(BASE, window=64)` ได้เลย
    2. ทุก field ถูกเขียนลง results/experiment_log.csv ได้ตรง ๆ → ย้อนดูได้ว่า config ไหนลองไปแล้ว (ข้อ 0.4)
    3. `frozen=True` กันการแก้ค่ากลางทาง (config ที่รันไปแล้วต้องไม่เปลี่ยนค่า)

หมายเหตุชื่อไฟล์: ไม่ใช้ชื่อ `config.py` เพราะ legacy มีโมดูลชื่อนั้นอยู่แล้ว (กันสับสนตอนอ่านโค้ด)
"""

from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------
# ชุดช่องสัญญาณดิบก่อนทำ STFT (SPEC ข้อ 5.2) · สูตรอยู่ใน src/spectrogram.py
# ---------------------------------------------------------------
CHANNEL_SETS = {
    "ret": ["ret_close"],                                           # base: 1 ช่อง
    "ret3": ["ret_close", "range_hl", "chg_vol"],                    # 3 ช่อง
    "ret4": ["ret_close", "range_hl", "chg_vol", "gap_open"],        # 4 ช่อง
    "detrend": ["detrend_ma20"],                                     # ตัวเทียบ: ราคาที่หัก trend
}

SPEC_TYPES = ("logpower", "complex")
SCALE_MODES = ("fit_median", "fixed")
NORMALIZE = ("band", "global", "image")
ARCHS = ("cnn2d", "cnn1d_time", "cnn1d_raw")
HEADS = ("flatten", "gap")
ANN_INPUTS = ("spec", "hybrid", "hybrid_nocnn")      # hybrid_nocnn = control ของ B25 (ANN เดียวกับ hybrid แต่ไม่มี CNN)
TRAIN_MODES = ("e2e", "two_stage")
LOSSES = ("huber", "mae", "mse")


@dataclass(frozen=True)
class Config:
    """ค่าตั้งต้นทุกตัว = base config ของ SPEC ข้อ 5.1 · เปลี่ยนด้วย `dataclasses.replace`"""

    name: str = "base"

    # ---- การแทนข้อมูล (Stage 1 · ข้อ 5.2) ----
    channels: str = "ret"           # คีย์ของ CHANNEL_SETS
    window: int = 128               # ความยาวหน้าต่าง (วันทำการ) ที่ใช้ทำภาพ 1 ภาพ
    nperseg: int = 16               # ความยาวหน้าต่าง Hann ของ STFT
    hop: int = 2                    # ก้าวของหน้าต่าง (noverlap = nperseg − hop)
    spec_type: str = "logpower"     # "logpower" = log1p(|Z|²/s) · "complex" = arcsinh(Re Z/s), arcsinh(Im Z/s)
    scale_mode: str = "fit_median"  # s มาจากไหน: "fit_median" = median(|ค่าดิบ|) ของส่วน fit ใน train ของ fold นั้น
    fixed_scale: float = 1e-3       #              "fixed" = ใช้ค่านี้ (complex ที่ port มาใช้ 1e-3 → arcsinh(z·1e3))
    normalize: str = "band"         # "band" = ทีละแถบความถี่ · "global" = ทั้งภาพรวมกัน · "image" = ต่อภาพ

    # ---- โครงสร้าง (Stage 2 · ข้อ 5.3) ----
    arch: str = "cnn2d"             # "cnn2d" (ก) · "cnn1d_time" (ข) · "cnn1d_raw" (ค · ไม่ทำ STFT)
    head: str = "flatten"           # หัวต่อจาก conv: "flatten" หรือ "gap" (global average pooling)
    filters: tuple = (8, 16)        # จำนวน filter ของ conv แต่ละชั้น
    cnn_feat: int = 16              # จำนวน CNN feature ที่ส่งให้ ANN
    ann_hidden: tuple = (32,)       # ขนาดชั้นซ่อนของ ANN
    dropout: float = 0.3

    # ---- input ของ ANN และวิธีเทรน (Stage 3–4 · ข้อ 5.4) ----
    ann_input: str = "spec"         # "spec" = spectrogram อย่างเดียว · "hybrid" = + 29 feature เดิม + linear skip · "hybrid_nocnn" = 29 feature อย่างเดียว (control)
    train_mode: str = "e2e"         # "e2e" = end-to-end · "two_stage" = เทรน CNN แล้ว freeze (ต้องใช้ cross-fitting)

    # ---- loss และ optimizer (Stage 5–6 · ข้อ 5.5–5.6) ----
    loss: str = "huber"
    huber_delta: float = 1.0        # δ บน y ที่ standardize ด้วยสถิติ train
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch: int = 32
    max_epochs: int = 200
    patience: int = 25
    noise_std: float = 0.3          # Gaussian noise augmentation บน input ภาพ

    def __post_init__(self):
        if self.channels not in CHANNEL_SETS:
            raise KeyError(f"channels ต้องเป็นหนึ่งใน {sorted(CHANNEL_SETS)} (ได้ {self.channels!r})")
        for value, allowed, label in ((self.spec_type, SPEC_TYPES, "spec_type"), (self.scale_mode, SCALE_MODES, "scale_mode"),
                                      (self.normalize, NORMALIZE, "normalize"),
                                      (self.arch, ARCHS, "arch"), (self.head, HEADS, "head"),
                                      (self.ann_input, ANN_INPUTS, "ann_input"),
                                      (self.train_mode, TRAIN_MODES, "train_mode"), (self.loss, LOSSES, "loss")):
            if value not in allowed:
                raise KeyError(f"{label} ต้องเป็นหนึ่งใน {allowed} (ได้ {value!r})")
        if self.train_mode == "two_stage" and self.ann_input == "hybrid_nocnn":
            raise ValueError("hybrid_nocnn ไม่มี CNN จึงใช้กับ train_mode two_stage ไม่ได้")
        if self.nperseg <= self.hop or self.hop < 1:
            raise ValueError("ต้องมี 1 <= hop < nperseg")
        if self.window < self.nperseg:
            raise ValueError("window ต้อง >= nperseg")

    # ---- ขนาดของ input ที่โมเดลจะได้รับ ----
    @property
    def channel_names(self):
        return list(CHANNEL_SETS[self.channels])

    @property
    def n_raw_channels(self):
        return len(self.channel_names)

    @property
    def noverlap(self):
        return self.nperseg - self.hop

    @property
    def needs_features(self):
        """ต้องใช้ 29 feature เดิมเป็น input ด้วยไหม (hybrid และ control hybrid-noCNN)"""
        return self.ann_input in ("hybrid", "hybrid_nocnn")

    @property
    def image_shape(self):
        """
        (C, F, T) ของภาพหนึ่งภาพ
            F = nperseg/2 + 1                                  (จำนวนแถบความถี่ของ rfft)
            T = (window + 2·(nperseg//2) − nperseg)/hop + 1     (boundary="even" เติมขอบข้างละ nperseg//2)
            C = ช่องดิบ (logpower) หรือ ช่องดิบ × 2 (complex: real + imag)
        arch "cnn1d_raw" ไม่ทำ STFT → input เป็น (C, window)
        """
        if self.arch == "cnn1d_raw":
            return (self.n_raw_channels, self.window)
        f = self.nperseg // 2 + 1
        t = (self.window + 2 * (self.nperseg // 2) - self.nperseg) // self.hop + 1
        c = self.n_raw_channels * (2 if self.spec_type == "complex" else 1)
        return (c, f, t)

    def to_row(self):
        """dict แบน ๆ สำหรับเขียนลง experiment_log.csv"""
        from dataclasses import asdict
        row = asdict(self)
        row["filters"] = "-".join(map(str, self.filters))
        row["ann_hidden"] = "-".join(map(str, self.ann_hidden))
        row["image_shape"] = "x".join(map(str, self.image_shape))
        return row


BASE = Config(name="base")

# ภาพของ base config ต้องเป็น 9 × 65 ตามที่สเปกข้อ 5.1 ระบุ -- ถ้าสูตรเพี้ยนต้องพังตอน import
assert BASE.image_shape == (1, 9, 65), f"base image shape ต้องเป็น (1, 9, 65) ได้ {BASE.image_shape}"


def variant(name, **changes):
    """สร้าง config ใหม่จาก BASE โดยเปลี่ยนทีละอย่าง (วิธีทดลองของ Stage 1–5)"""
    return replace(BASE, name=name, **changes)
