import cv2
import numpy as np
import pytest

from scripts.cache_dmd_mediapipe import affine_face_crop, sample_indices


def test_sample_indices_approximately_five_fps():
    indices = sample_indices(total_frames=298, native_fps=29.8, target_fps=5.0)

    assert indices[0] == 0
    assert indices[-1] < 298
    assert len(indices) == 50
    assert set(np.diff(indices)) <= {5, 6}


@pytest.mark.parametrize("value", [0, -1])
def test_sample_indices_rejects_invalid_frame_count(value):
    with pytest.raises(ValueError):
        sample_indices(value, native_fps=30.0, target_fps=5.0)


def test_affine_face_crop_has_expected_shape() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    landmarks = [
        (0.25 + 0.5 * (index % 2), 0.2 + 0.6 * ((index // 2) % 2))
        for index in range(478)
    ]
    landmarks[33] = (0.35, 0.4)
    landmarks[133] = (0.45, 0.4)
    landmarks[362] = (0.55, 0.4)
    landmarks[263] = (0.65, 0.4)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))

    crop = affine_face_crop(frame, landmarks, (96, 96), clahe)

    assert crop.shape == (96, 96)
