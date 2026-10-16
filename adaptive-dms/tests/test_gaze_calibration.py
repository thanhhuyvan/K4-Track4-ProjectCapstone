import numpy as np
import pytest
import torch

from adaptive_dms.dms import BlazeGazeMlp
from adaptive_dms.gaze_calibration import GazeCalibrator, fit_affine, load_profile, save_profile


def _data(n: int = 90, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    zone = np.repeat(np.arange(9), n // 9)
    centers = rng.normal(size=(9, 512)).astype(np.float32)
    emb = centers[zone] + 0.05 * rng.normal(size=(len(zone), 512)).astype(np.float32)
    head = rng.normal(size=(len(zone), 3)).astype(np.float32)
    origin = rng.normal(size=(len(zone), 3)).astype(np.float32)
    return emb, head, origin, zone


def test_affine_recovers_known_map() -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(size=(40, 2)).astype(np.float32)
    matrix = np.array([[2.0, 0.3, 0.1], [-0.2, 1.5, -0.4]], np.float32)
    target = source @ matrix[:, :2].T + matrix[:, 2]
    assert np.allclose(fit_affine(source, target), matrix, atol=1e-4)


def test_calibrator_shapes_and_requires_fit() -> None:
    torch.manual_seed(0)
    calibrator = GazeCalibrator(BlazeGazeMlp(), steps=5)
    emb, head, origin, zone = _data()
    with pytest.raises(RuntimeError):
        calibrator.predict(emb, head, origin)
    calibrator.fit(emb, head, origin, zone)
    pog, predicted, confidence = calibrator.predict(emb, head, origin)
    assert pog.shape == (90, 2) and predicted.shape == (90,)
    assert np.all((confidence > 0) & (confidence <= 1))
    again = calibrator.predict(emb, head, origin)[1]
    assert np.array_equal(predicted, again)


def test_calibrator_rejects_too_little_data() -> None:
    emb, head, origin, zone = _data()
    with pytest.raises(ValueError):
        GazeCalibrator(BlazeGazeMlp()).fit(emb[:3], head[:3], origin[:3], zone[:3])
    keep = zone < 2
    with pytest.raises(ValueError):
        GazeCalibrator(BlazeGazeMlp()).fit(emb[keep], head[keep], origin[keep], zone[keep])


def test_profile_roundtrip(tmp_path) -> None:
    emb, head, origin, zone = _data()
    save_profile(tmp_path / "p.npz", emb, head, origin, zone)
    loaded = load_profile(tmp_path / "p.npz")
    assert np.array_equal(loaded["zone_id"], zone) and np.allclose(loaded["embedding"], emb)
