"""Tiny CNN on 16x16 eye crops: predicts p(eye closed) from both eyes."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EyeBlinkCNN(nn.Module):
    """Shared per-eye encoder (GroupNorm, no BatchNorm) + small fusion head.

    Input: (B, 2, H, W) float in [0, 1]; channel 0 = left eye, 1 = mirrored right eye.
    Output: (B,) logit of "eyes closed".
    """

    def __init__(self, size: int = 16, width: int = 16, embed: int = 32) -> None:
        super().__init__()
        if size % 4:
            raise ValueError("size must be divisible by 4")
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GroupNorm(4, width * 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 3, 3, padding=1),
            nn.GroupNorm(4, width * 3),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(width * 3 * (size // 4) ** 2, embed),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(embed * 2, embed), nn.SiLU(), nn.Linear(embed, 1)
        )

    @staticmethod
    def normalise(crops: Tensor) -> Tensor:
        """Per-crop standardisation so the model is robust to exposure."""
        mean = crops.mean(dim=(-2, -1), keepdim=True)
        std = crops.std(dim=(-2, -1), keepdim=True)
        return (crops - mean) / (std + 1e-2)

    def forward(self, crops: Tensor) -> Tensor:
        batch = crops.shape[0]
        x = self.normalise(crops).reshape(batch * 2, 1, self.size, self.size)
        features = self.encoder(x).reshape(batch, -1)
        return self.head(features).squeeze(-1)
