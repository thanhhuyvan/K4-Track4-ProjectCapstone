import numpy as np

from adaptive_dms.realtime import (
    DriverMonitor,
    FrameSignals,
    SlidingWindow,
    affine_eye_crop,
    eye_crops,
)


def test_sliding_window_is_causal_and_evicts() -> None:
    window = SlidingWindow(1.0)
    for i in range(30):
        window.push(i * 0.1, 1.0 if i >= 20 else 0.0)
    assert window.span <= 1.0 + 1e-9
    assert window.mean() > 0.8
    before = len(window)
    window.drop_last()
    assert len(window) == before - 1


def _landmarks() -> np.ndarray:
    pts = np.zeros((478, 2), dtype=np.float32)
    pts[33], pts[133] = (100, 100), (140, 100)
    pts[362], pts[263] = (180, 100), (220, 100)
    return pts


def test_eye_crop_shape_and_mirror() -> None:
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    crops = eye_crops(frame, _landmarks())
    assert crops.shape == (2, 16, 16) and crops.dtype == np.uint8
    plain = affine_eye_crop(frame, _landmarks(), (33, 133))
    assert np.array_equal(crops[1], plain[:, ::-1])


def test_monitor_microsleep_and_blink() -> None:
    monitor = DriverMonitor()
    t = 0.0
    for _ in range(60):  # 2 s open at 30 FPS
        out = monitor.update(t, FrameSignals(face=True, ear=0.30, blink_score=0.05))
        t += 1 / 30
    assert out.state == "alert"
    for _ in range(6):  # 0.2 s blink
        monitor.update(t, FrameSignals(face=True, ear=0.10, blink_score=0.9))
        t += 1 / 30
    for _ in range(30):
        out = monitor.update(t, FrameSignals(face=True, ear=0.30, blink_score=0.05))
        t += 1 / 30
    assert out.state == "alert"
    for _ in range(40):  # 1.3 s closed
        out = monitor.update(t, FrameSignals(face=True, ear=0.10, blink_score=0.9))
        t += 1 / 30
    assert out.state == "microsleep"


def test_turned_head_does_not_corrupt_ear_baseline() -> None:
    monitor = DriverMonitor()
    t = 0.0
    for _ in range(90):  # head turned: EAR is distorted high, must be ignored
        monitor.update(t, FrameSignals(face=True, ear=0.55, blink_score=0.05, head_yaw=40.0))
        t += 1 / 30
    for _ in range(90):  # frontal, normal open eyes
        out = monitor.update(t, FrameSignals(face=True, ear=0.28, blink_score=0.05, head_yaw=3.0))
        t += 1 / 30
    assert not out.eye_closed and out.ear_threshold < 0.28


def test_monitor_distracted_from_gaze() -> None:
    monitor = DriverMonitor()
    out = None
    for i in range(90):
        out = monitor.update(i / 30, FrameSignals(face=True, ear=0.3, off_road=True))
    assert out is not None and out.state == "distracted"
