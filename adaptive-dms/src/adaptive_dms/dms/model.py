"""Tiny crop/geometry FiLM model with a causal temporal core."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)


def geometry_from_landmarks(
    landmarks: np.ndarray, quality: np.ndarray
) -> np.ndarray:
    """Return compact eye/mouth/head geometry for FiLM conditioning."""
    points = np.asarray(landmarks, dtype=np.float32)[..., :2]
    quality = np.asarray(quality, dtype=np.float32)

    def distance(first: int, second: int) -> np.ndarray:
        return np.linalg.norm(points[:, first] - points[:, second], axis=1)

    left_ear = (distance(385, 380) + distance(387, 373)) / (
        2.0 * distance(362, 263).clip(1e-5)
    )
    right_ear = (distance(160, 144) + distance(158, 153)) / (
        2.0 * distance(33, 133).clip(1e-5)
    )
    mar = distance(13, 14) / distance(61, 291).clip(1e-5)
    left_center = points[:, list(LEFT_EYE)].mean(axis=1)
    right_center = points[:, list(RIGHT_EYE)].mean(axis=1)
    eye_center = (left_center + right_center) / 2.0
    interocular = np.linalg.norm(left_center - right_center, axis=1).clip(1e-5)
    nose = (points[:, 1] - eye_center) / interocular[:, None]
    span = points.max(axis=1) - points.min(axis=1)
    normalized_span = span / interocular[:, None]
    return np.column_stack(
        (
            left_ear,
            right_ear,
            mar,
            nose,
            normalized_span,
            quality,
        )
    ).astype(np.float32)


class TinyCropEncoder(nn.Module):
    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16),
            nn.Conv2d(16, output_dim, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        return self.network(crop).flatten(1)


class CausalDmsBaseline(nn.Module):
    """Shared-eye/mouth appearance, bounded FiLM, and one causal GRU."""

    def __init__(self, geometry_dim: int = 10, hidden_dim: int = 96) -> None:
        super().__init__()
        self.eye_encoder = TinyCropEncoder(32)
        self.mouth_encoder = TinyCropEncoder(32)
        self.film = nn.Sequential(
            nn.Linear(geometry_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 192),
        )
        self.temporal = nn.GRU(96, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.eye_head = nn.Linear(hidden_dim, 4)
        self.yawn_head = nn.Linear(hidden_dim, 3)
        self.blink_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        mouth: torch.Tensor,
        geometry: torch.Tensor,
        face: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps = left_eye.shape[:2]

        def encode(encoder: nn.Module, crops: torch.Tensor) -> torch.Tensor:
            flattened = crops.reshape(batch * steps, 1, *crops.shape[-2:])
            return encoder(flattened).reshape(batch, steps, -1)

        left = encode(self.eye_encoder, left_eye)
        right = encode(self.eye_encoder, right_eye)
        mouth_features = encode(self.mouth_encoder, mouth)
        appearance = torch.cat(((left + right) / 2.0, (left - right).abs(), mouth_features), dim=-1)
        gamma, beta = self.film(geometry).chunk(2, dim=-1)
        conditioned = (1.0 + 0.1 * torch.tanh(gamma)) * appearance
        conditioned = conditioned + 0.1 * torch.tanh(beta)
        temporal, _ = self.temporal(conditioned)
        temporal = self.norm(temporal)
        return {
            "eye": self.eye_head(temporal),
            "yawn": self.yawn_head(temporal),
            "blink": self.blink_head(temporal),
        }


class CausalDmsBaselineV2(CausalDmsBaseline):
    """Binary-yawn baseline with a direct geometry residual."""

    def __init__(self, geometry_dim: int = 10, hidden_dim: int = 96) -> None:
        super().__init__(geometry_dim=geometry_dim, hidden_dim=hidden_dim)
        self.yawn_head = nn.Linear(hidden_dim, 2)
        self.geometry_yawn = nn.Sequential(
            nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
        )

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        mouth: torch.Tensor,
        geometry: torch.Tensor,
        face: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(left_eye, right_eye, mouth, geometry)
        output["yawn"] = output["yawn"] + self.geometry_yawn(geometry)
        return output


class CausalDmsBaselineV3(CausalDmsBaselineV2):
    """V2 plus a tiny aligned-face residual for occluded-yawn context."""

    def __init__(self, geometry_dim: int = 10, hidden_dim: int = 96) -> None:
        super().__init__(geometry_dim=geometry_dim, hidden_dim=hidden_dim)
        self.face_encoder = TinyCropEncoder(32)
        self.face_film = nn.Sequential(
            nn.Linear(geometry_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 64),
        )
        self.face_yawn = nn.Linear(32, 2)

    def forward(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        mouth: torch.Tensor,
        geometry: torch.Tensor,
        face: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if face is None:
            raise ValueError("V3 requires an aligned face crop")
        output = super().forward(left_eye, right_eye, mouth, geometry)
        batch, steps = face.shape[:2]
        flattened = face.reshape(batch * steps, 1, *face.shape[-2:])
        features = self.face_encoder(flattened).reshape(batch, steps, -1)
        gamma, beta = self.face_film(geometry).chunk(2, dim=-1)
        features = (1.0 + 0.1 * torch.tanh(gamma)) * features
        features = features + 0.1 * torch.tanh(beta)
        output["yawn"] = output["yawn"] + self.face_yawn(features)
        return output
