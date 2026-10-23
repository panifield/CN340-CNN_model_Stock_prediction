"""Spectrogram -> CNN feature extractor -> ANN classifier."""
import torch.nn as nn


class SpectroCNNANN(nn.Module):
    """CNN สอง block และ ANN สำหรับ binary next-day direction classifier."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 2), padding=(1, 0)),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 2), padding=(1, 0)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)
