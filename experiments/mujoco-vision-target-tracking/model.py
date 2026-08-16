from __future__ import annotations

"""64×64 RGB画像から目標x、y座標を推定する小型CNN。"""

import torch
from torch import nn


class TargetCoordinateCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        # 空間位置を残した8×8特徴を平坦化し、連続した2座標へ回帰する。
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(images))
