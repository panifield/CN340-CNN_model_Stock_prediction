"""
models.py — โครง CNN → ANN ทั้งหมด (SPEC ข้อ 5.1, 5.3, 5.4)
===========================================================
ไฟล์นี้ทำอะไร: นิยามโมเดล PyTorch ทุกแบบที่การทดลองต้องใช้ + ฟังก์ชันพิมพ์ shape ทุกชั้นและนับพารามิเตอร์

โครงของ base config (ข้อ 5.1) · input 1×9×65
    Conv2d(1→8, 3×3, pad 1) – BatchNorm – ReLU – MaxPool 2×2      → 8×4×32
    Conv2d(8→16, 3×3, pad 1) – BatchNorm – ReLU – MaxPool 2×2     → 16×2×16
    Flatten → Linear(512→16) – ReLU                               → **CNN feature 16 ตัว**
    Linear(16→32) – ReLU – Dropout 0.3 → Linear(32→1)             → ŷ (หน่วย y ที่ standardize แล้ว)

สูตรที่ใช้คำนวณขนาด (CS231n lecture 5) : out = (W − K + 2P)/S + 1
    conv 3×3 pad 1 stride 1 → ขนาดเท่าเดิม · maxpool 2×2 stride 2 → ครึ่งหนึ่ง (ปัดลง)
    จำนวนพารามิเตอร์ของ conv = Cin·K·K·Cout + Cout

ทำไมต้องมี: `layer_report()` พิมพ์ shape จริงของทุกชั้นเทียบกับที่คำนวณมือ (ข้อ 5.0 บังคับ)
และโครงทุกแบบอยู่ไฟล์เดียวทำให้เทียบกันได้ว่าต่างกันตรงไหน
"""

import numpy as np
import torch
import torch.nn as nn


def _conv_block_2d(cin, cout):
    """Conv–BN–ReLU–MaxPool หนึ่งชุด (ตาม base config)"""
    return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]


def _conv_block_1d(cin, cout):
    return [nn.Conv1d(cin, cout, 3, padding=1), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2)]


class CNNFeature(nn.Module):
    """
    ส่วนสกัด feature : ภาพ → เวกเตอร์ `cfg.cnn_feat` ตัว
      arch "cnn2d"      : Conv2d ซ้อนกันบนภาพ (C, F, T)        ← ตัวหลัก (ก)
      arch "cnn1d_time" : มองแถบความถี่เป็น channel แล้ว Conv1d ตามแกนเวลา   (ข)
      arch "cnn1d_raw"  : Conv1d บนหน้าต่าง return ดิบ (C, L) ไม่มี STFT     (ค · ตัวเทียบ)
      head "flatten"    : แบนทุกจุดแล้วเข้า Linear (เก็บตำแหน่งเวลาไว้)
      head "gap"        : เฉลี่ยทั้งภาพต่อ filter (ทิ้งตำแหน่ง แต่พารามิเตอร์น้อยกว่ามาก)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        shape = cfg.image_shape
        if cfg.arch == "cnn2d":
            layers, c = [], shape[0]
            for f in cfg.filters:
                layers += _conv_block_2d(c, f)
                c = f
            self.body = nn.Sequential(*layers)
        else:
            # cnn1d_time : (C, F, T) → ยุบ (C, F) เป็น channel เดียวกัน เหลือสัญญาณตามแกนเวลา
            # cnn1d_raw  : (C, L) อยู่ในรูป 1D อยู่แล้ว
            cin = shape[0] * shape[1] if cfg.arch == "cnn1d_time" else shape[0]
            layers, c = [], cin
            for f in cfg.filters:
                layers += _conv_block_1d(c, f)
                c = f
            self.body = nn.Sequential(*layers)
        self.out_shape = self._probe(shape)
        n_flat = int(np.prod(self.out_shape)) if cfg.head == "flatten" else self.out_shape[0]
        self.pool = nn.Identity() if cfg.head == "flatten" else _GlobalAvgPool()
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, cfg.cnn_feat), nn.ReLU())
        self.n_features = cfg.cnn_feat

    def _reshape(self, x):
        if self.cfg.arch == "cnn1d_time":
            return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        return x

    def _probe(self, shape):
        """รันข้อมูลปลอม 1 แถวเพื่อดูขนาดหลัง conv (ไม่ต้องคำนวณมือซ้ำในโค้ด)"""
        with torch.no_grad():
            d = self.body(self._reshape(torch.zeros((1,) + tuple(shape))))
        return tuple(d.shape[1:])

    def forward(self, x):
        return self.proj(self.pool(self.body(self._reshape(x))))


class _GlobalAvgPool(nn.Module):
    """เฉลี่ยทุกแกนที่เหลือต่อ filter → (N, C)"""

    def forward(self, x):
        return x.mean(dim=tuple(range(2, x.ndim)))


class SpectroANN(nn.Module):
    """
    โมเดลหลัก (track 1 และ 2)
      ann_input "spec"   : ANN เห็นแค่ CNN feature
      ann_input "hybrid" : ANN เห็น [CNN feature, 29 feature เดิมที่ standardize แล้ว] + linear skip จาก 29 feature
        linear skip ทำให้โมเดล "อย่างน้อยทำได้เท่าโมเดลเชิงเส้นของ feature เดิม"
        (ported idea from task_b/phase2/cnn.py : Hybrid, branch task_b)
    forward(img, extra) → ŷ (หน่วย y ที่ standardize ด้วยสถิติ train)
    """

    def __init__(self, cfg, n_extra=0):
        super().__init__()
        self.cfg = cfg
        self.use_extra = cfg.ann_input == "hybrid"
        if self.use_extra and n_extra <= 0:
            raise ValueError("hybrid ต้องมี n_extra > 0")
        self.cnn = CNNFeature(cfg)
        d = self.cnn.n_features + (n_extra if self.use_extra else 0)
        ann = []
        for h in cfg.ann_hidden:
            ann += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = h
        ann.append(nn.Linear(d, 1))
        self.ann = nn.Sequential(*ann)
        self.skip = nn.Linear(n_extra, 1) if self.use_extra else None

    def forward(self, img, extra=None):
        h = self.cnn(img)
        if self.use_extra:
            h = torch.cat([h, extra], dim=1)
        out = self.ann(h)
        if self.skip is not None:
            out = out + self.skip(extra)
        return out.squeeze(-1)


class FeatureANN(nn.Module):
    """
    ANN ที่รับเวกเตอร์ feature ตรง ๆ (ไม่มี CNN)
    ใช้กับ (1) ขั้นที่สองของการเทรนแบบสองขั้น — รับ CNN feature ที่ freeze แล้ว
          (2) ตัวควบคุมของ hybrid — ถอด CNN ออกเพื่อดูว่า CNN เพิ่มอะไรจริงไหม
    """

    def __init__(self, n_in, hidden, dropout, skip=False):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.skip = nn.Linear(n_in, 1) if skip else None

    def forward(self, x):
        out = self.net(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out.squeeze(-1)


def build(cfg, n_extra=0):
    """ทางเข้าเดียวสำหรับสร้างโมเดลจาก config (เรียกหลังตั้ง seed เสมอ เพื่อให้ weight เริ่มต้นเหมือนเดิม)

    ann_input "hybrid_nocnn" (control ของ B25) = ANN เดียวกับ hybrid (หัว hidden–ReLU–Dropout, linear skip จาก 29 feature)
    แต่ **ไม่มี CNN** → forward(extra) รับ 29 feature อย่างเดียว
    """
    if cfg.ann_input == "hybrid_nocnn":
        return FeatureANN(n_extra, cfg.ann_hidden, cfg.dropout, skip=True)
    return SpectroANN(cfg, n_extra)


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def layer_report(cfg, n_extra=0):
    """
    ตารางรายชั้น : ชื่อชั้น · shape ออก · จำนวนพารามิเตอร์ (SPEC ข้อ 5.0 บังคับให้พิมพ์)
    คืน (list ของ dict, จำนวนพารามิเตอร์รวม) · ขนาดได้จากการรันข้อมูลปลอมจริง ไม่ใช่การคำนวณซ้ำ
    """
    torch.manual_seed(0)
    model = build(cfg, n_extra)
    model.eval()
    rows = []
    x = torch.zeros((1,) + tuple(cfg.image_shape))
    rows.append({"layer": "input (ภาพ)", "out_shape": tuple(x.shape[1:]), "params": 0})

    h = model.cnn._reshape(x)
    if tuple(h.shape[1:]) != tuple(x.shape[1:]):
        rows.append({"layer": f"reshape ({cfg.arch})", "out_shape": tuple(h.shape[1:]), "params": 0})
    with torch.no_grad():
        for layer in model.cnn.body:
            h = layer(h)
            rows.append({"layer": type(layer).__name__, "out_shape": tuple(h.shape[1:]), "params": count_params(layer)})
        h = model.cnn.pool(h)
        if cfg.head == "gap":
            rows.append({"layer": "GlobalAvgPool", "out_shape": tuple(h.shape[1:]), "params": 0})
        for layer in model.cnn.proj:
            h = layer(h)
            rows.append({"layer": type(layer).__name__, "out_shape": tuple(h.shape[1:]), "params": count_params(layer)})
        rows.append({"layer": "= CNN feature", "out_shape": tuple(h.shape[1:]), "params": 0})
        if model.use_extra:
            h = torch.cat([h, torch.zeros(1, n_extra)], dim=1)
            rows.append({"layer": f"concat 29 feature เดิม ({n_extra})", "out_shape": tuple(h.shape[1:]), "params": 0})
        for layer in model.ann:
            h = layer(h)
            rows.append({"layer": type(layer).__name__, "out_shape": tuple(h.shape[1:]), "params": count_params(layer)})
        if model.skip is not None:
            rows.append({"layer": "linear skip (+)", "out_shape": (1,), "params": count_params(model.skip)})
    return rows, count_params(model)


def conv_output_size(w, k, pad, stride):
    """สูตร L5 ไว้เทียบกับขนาดจริง : (W − K + 2P)/S + 1"""
    return (w - k + 2 * pad) // stride + 1


def conv_param_count(cin, k, cout):
    """พารามิเตอร์ของ conv ชั้นเดียว : Cin·K·K·Cout + Cout"""
    return cin * k * k * cout + cout
