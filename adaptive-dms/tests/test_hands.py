import cv2
import numpy as np

from adaptive_dms.hands import (
    HANDS_OFF,
    Hand,
    HandsConfig,
    HandsMonitor,
    Wheel,
    contact_state,
    fit_wheel,
    on_wheel_sides,
)

WHEEL = Wheel(100.0, 100.0, 50.0)
ON_RIM = [Hand((55.0, 100.0), 0.9), Hand((145.0, 100.0), 0.9)]


def test_wheel_radii_and_manual_points() -> None:
    assert WHEEL.radii((150.0, 100.0)) == 1.0
    wheel = Wheel.from_points((10, 10), (10, 40))
    assert (wheel.cx, wheel.cy, wheel.r) == (10.0, 10.0, 30.0)


def test_sides_follow_camera_orientation() -> None:
    facing = HandsConfig(driver_left_on_image_left=False)
    behind = HandsConfig(driver_left_on_image_left=True)
    image_left_only = [ON_RIM[0]]
    assert on_wheel_sides(image_left_only, WHEEL, facing) == (False, True)
    assert on_wheel_sides(image_left_only, WHEEL, behind) == (True, False)
    assert contact_state(*on_wheel_sides(ON_RIM, WHEEL, facing)) == "both_hands"


def test_far_or_unsure_hands_are_off() -> None:
    config = HandsConfig(rim_radii=1.2, min_score=0.5)
    lap = [Hand((100.0, 190.0), 0.9)]  # 1.8 radii below the centre
    unsure = [Hand((55.0, 100.0), 0.2)]
    assert on_wheel_sides(lap, WHEEL, config) == (False, False)
    assert on_wheel_sides(unsure, WHEEL, config) == (False, False)


def test_monitor_debounces_and_emits_causal_event() -> None:
    monitor = HandsMonitor(HandsConfig(switch_s=0.2))
    states = []
    for i in range(30):
        hands = ON_RIM if i < 10 or i > 20 else []
        states.append(monitor.update(i / 30, hands, WHEEL).state)
    assert states[11] == "both_hands"  # a short gap is not reported yet
    assert states[18] == HANDS_OFF
    assert states[-1] == "both_hands"
    (event,) = monitor.finish()
    assert event.label == HANDS_OFF
    assert abs(event.start_s - 10 / 30) < 1e-9
    assert event.confirmed_s >= event.start_s + 0.2 - 1e-9


def test_monitor_without_wheel() -> None:
    assert HandsMonitor().update(0.0, ON_RIM, None).state == "no_wheel"


def test_fit_wheel_finds_a_drawn_rim() -> None:
    frame = np.full((360, 640, 3), 200, np.uint8)
    cv2.circle(frame, (300, 170), 120, (30, 30, 30), 14)
    wheel = fit_wheel([frame] * 3)
    assert wheel is not None
    assert abs(wheel.cx - 300) < 8 and abs(wheel.cy - 170) < 8 and abs(wheel.r - 120) < 12
