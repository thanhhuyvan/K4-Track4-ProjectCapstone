import numpy as np
import torch

from adaptive_dms.dms import (
    CausalDmsBaseline,
    CausalDmsBaselineV2,
    CausalDmsBaselineV3,
    geometry_from_landmarks,
)


def test_geometry_shape_and_finite_values():
    rng = np.random.default_rng(7)
    landmarks = rng.random((3, 478, 3), dtype=np.float32)
    quality = rng.random((3, 3), dtype=np.float32)

    geometry = geometry_from_landmarks(landmarks, quality)

    assert geometry.shape == (3, 10)
    assert np.isfinite(geometry).all()


def test_causal_baseline_shapes_and_no_future_leakage():
    torch.manual_seed(4)
    model = CausalDmsBaseline().eval()
    left = torch.rand(2, 6, 32, 64)
    right = torch.rand(2, 6, 32, 64)
    mouth = torch.rand(2, 6, 48, 64)
    geometry = torch.rand(2, 6, 10)

    with torch.no_grad():
        original = model(left, right, mouth, geometry)
        left[:, 4:] = torch.rand_like(left[:, 4:])
        changed = model(left, right, mouth, geometry)

    assert original["eye"].shape == (2, 6, 4)
    assert original["yawn"].shape == (2, 6, 3)
    assert original["blink"].shape == (2, 6, 2)
    for head in original:
        torch.testing.assert_close(original[head][:, :4], changed[head][:, :4])


def test_v2_uses_binary_yawn_head():
    model = CausalDmsBaselineV2().eval()
    with torch.no_grad():
        output = model(
            torch.rand(1, 3, 32, 64),
            torch.rand(1, 3, 32, 64),
            torch.rand(1, 3, 48, 64),
            torch.rand(1, 3, 10),
        )

    assert output["yawn"].shape == (1, 3, 2)


def test_v3_requires_and_uses_face_context():
    model = CausalDmsBaselineV3().eval()
    inputs = (
        torch.rand(1, 3, 32, 64),
        torch.rand(1, 3, 32, 64),
        torch.rand(1, 3, 48, 64),
        torch.rand(1, 3, 10),
    )
    with torch.no_grad():
        output = model(*inputs, torch.rand(1, 3, 96, 96))

    assert output["yawn"].shape == (1, 3, 2)
