"""Per-driver gaze calibration: affine PoG alignment + few-shot head fine-tuning.

Follows WebEyeTrack's adaptation: fit a least-squares affine map from the
frozen head's predictions to the canonical zone positions, then fine-tune the
small gaze MLP through that map. Zones are assigned by nearest prototype, where
each prototype is the mean calibrated prediction of that zone's support frames.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from adaptive_dms.data import GAZE_ZONE_CANONICAL_POG, GAZE_ZONES
from adaptive_dms.dms.blazegaze import BlazeGazeMlp

PROTOTYPES = np.array([GAZE_ZONE_CANONICAL_POG[z] for z in GAZE_ZONES], dtype=np.float32)


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares (2, 3) affine matrix mapping source (N, 2) to target (N, 2)."""
    augmented = np.hstack([source, np.ones((len(source), 1), dtype=np.float32)])
    matrix, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return matrix.T.astype(np.float32)


class GazeCalibrator:
    def __init__(
        self,
        base_mlp: BlazeGazeMlp,
        prototypes: np.ndarray = PROTOTYPES,
        steps: int = 25,
        learning_rate: float = 1e-3,
        temperature: float = 0.05,
    ) -> None:
        self.base = copy.deepcopy(base_mlp).cpu().eval()
        self.canonical = np.asarray(prototypes, dtype=np.float32)
        self.steps = steps
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.mlp: BlazeGazeMlp | None = None
        self.affine_t: torch.Tensor | None = None
        self.prototypes = self.canonical.copy()

    @property
    def fitted(self) -> bool:
        return self.mlp is not None

    def fit(
        self,
        embedding: np.ndarray,
        head_vector: np.ndarray,
        face_origin_3d: np.ndarray,
        zone_id: np.ndarray,
    ) -> None:
        zone_id = np.asarray(zone_id, dtype=np.int64)
        if len(zone_id) < 4 or len(np.unique(zone_id)) < 3:
            raise ValueError("calibration needs >= 4 samples covering >= 3 zones")
        emb = torch.from_numpy(np.asarray(embedding, np.float32))
        head = torch.from_numpy(np.asarray(head_vector, np.float32))
        origin = torch.from_numpy(np.asarray(face_origin_3d, np.float32))
        target_np = self.canonical[zone_id]
        target = torch.from_numpy(target_np)

        mlp = copy.deepcopy(self.base).train()
        with torch.no_grad():
            initial = mlp(emb, head, origin).numpy()
        self.affine_t = torch.from_numpy(fit_affine(initial, target_np).T)

        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.learning_rate, betas=(0.85, 0.9))
        for _ in range(self.steps):
            optimizer.zero_grad()
            F.smooth_l1_loss(self._align(mlp(emb, head, origin)), target).backward()
            optimizer.step()
        self.mlp = mlp.eval()

        with torch.no_grad():
            adapted = self._align(self.mlp(emb, head, origin)).numpy()
        self.prototypes = self.canonical.copy()
        for zone in np.unique(zone_id):
            self.prototypes[zone] = adapted[zone_id == zone].mean(axis=0)

    def _align(self, raw: torch.Tensor) -> torch.Tensor:
        assert self.affine_t is not None
        return torch.cat([raw, torch.ones_like(raw[:, :1])], dim=1) @ self.affine_t

    @torch.no_grad()
    def predict(
        self, embedding: np.ndarray, head_vector: np.ndarray, face_origin_3d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated PoG (N, 2), nearest zone (N,), and its softmax confidence (N,)."""
        if self.mlp is None:
            raise RuntimeError("calibrator is not fitted")
        pog = self._align(
            self.mlp(
                torch.from_numpy(np.asarray(embedding, np.float32)),
                torch.from_numpy(np.asarray(head_vector, np.float32)),
                torch.from_numpy(np.asarray(face_origin_3d, np.float32)),
            )
        ).numpy()
        distance = np.linalg.norm(pog[:, None, :] - self.prototypes[None], axis=-1)
        logits = -distance / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        zone = distance.argmin(axis=1)
        return pog, zone, probability[np.arange(len(zone)), zone]


def save_profile(
    path: Path | str,
    embedding: np.ndarray,
    head_vector: np.ndarray,
    face_origin_3d: np.ndarray,
    zone_id: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embedding=np.asarray(embedding, np.float32),
        head_vector=np.asarray(head_vector, np.float32),
        face_origin_3d=np.asarray(face_origin_3d, np.float32),
        zone_id=np.asarray(zone_id, np.int64),
    )


def load_profile(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in ("embedding", "head_vector", "face_origin_3d", "zone_id")}
