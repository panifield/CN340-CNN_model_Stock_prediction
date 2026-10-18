"""Spectrogram -> CNN (สกัด feature) -> ANN (classifier)"""
import torch
import torch.nn as nn

# ขนาด CNN ที่ลองในการทดลอง: (จำนวน filter ของแต่ละ conv layer, ขนาด pooling สุดท้าย, hidden ของ ANN)
ARCHS = {
    "small": ([8], (1, 2), [8]),
    "current": ([16, 32], (2, 4), [64, 16]),
    "deep": ([32, 64, 128, 128], (2, 4), [256, 64]),
}


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int, filters=(16, 32), pool=(2, 4)):
        super().__init__()
        layers, c = [], in_ch
        for f in filters:
            layers += [nn.Conv2d(c, f, 3, padding=1), nn.BatchNorm2d(f), nn.ReLU()]
            c = f
        layers += [nn.AdaptiveAvgPool2d(pool), nn.Flatten()]
        self.net = nn.Sequential(*layers)
        self.out_dim = c * pool[0] * pool[1]

    def forward(self, x):
        return self.net(x)


class ANNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 16)):
        super().__init__()
        layers, d = [], in_dim
        for i, h in enumerate(hidden):
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.4 if i == 0 else 0.2)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpectroCNNANN(nn.Module):
    """n_extra > 0 : ต่อ feature เสริม (เช่น คู่/คี่ของเมื่อวาน) เข้ากับ feature จาก CNN ก่อนเข้า ANN
    bottleneck    : ย่อ feature จาก CNN เหลือกี่ค่า (กัน feature CNN จำนวนมากกลบ feature เสริม)
    skip          : ทางลัดจาก feature เสริมตรงไป output -- "linear" หรือ "mlp" (เรียน interaction ได้)"""

    def __init__(self, in_ch: int, n_extra: int = 0, arch: str = "current",
                 bottleneck: int | None = None, skip: str | None = None):
        super().__init__()
        filters, pool, hidden = ARCHS[arch]
        self.cnn = CNNFeatureExtractor(in_ch, filters, pool)
        cnn_dim = self.cnn.out_dim
        self.proj = None
        if bottleneck:
            self.proj = nn.Sequential(nn.Linear(cnn_dim, bottleneck), nn.ReLU(), nn.Dropout(0.3))
            cnn_dim = bottleneck
        self.ann = ANNClassifier(cnn_dim + n_extra, hidden)
        self.skip = None
        if skip == "linear":
            self.skip = nn.Linear(n_extra, 1)
        elif skip == "mlp":
            self.skip = nn.Sequential(nn.Linear(n_extra, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x, extra=None):
        h = self.cnn(x)
        if self.proj is not None:
            h = self.proj(h)
        if extra is not None:
            h = torch.cat([h, extra], dim=1)
        out = self.ann(h)
        if self.skip is not None:
            out = out + self.skip(extra)
        return out.squeeze(-1)  # logit ของ "คี่"
