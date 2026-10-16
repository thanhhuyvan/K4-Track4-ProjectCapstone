"""Hands-on-wheel branch: wheel geometry + MediaPipe hand landmarks, decided causally.

The wheel does not move, so the per-pixel median of a few seconds of frames erases the
moving hands and a Hough circle fit finds the rim. A hand is on the wheel when the
centroid of its 21 landmarks lies within ``rim_radii`` wheel radii of the wheel centre
(1.2 on DMD: on-wheel hands fall inside 93-98% of the time, hands in the lap 4%).
A new state must persist for ``switch_s`` before it is reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from adaptive_dms.events import TemporalEvent

HANDS_OFF = "hands_off"
CONTACT_STATES = ("both_hands", "only_right", "only_left", HANDS_OFF)
# Radius bounds as a fraction of image height. DMD's hands camera sees the whole wheel.
DMD_RADIUS_RANGE = (0.28, 0.40)
FIT_HEIGHT = 360  # Hough parameters were tuned at this height


@dataclass(frozen=True)
class Wheel:
    cx: float
    cy: float
    r: float

    def radii(self, xy: Sequence[float]) -> float:
        """Distance of a pixel position from the wheel centre, in wheel radii."""
        return float(np.hypot(xy[0] - self.cx, xy[1] - self.cy) / self.r)

    @classmethod
    def from_points(cls, centre: Sequence[float], rim: Sequence[float]) -> Wheel:
        """Manual fallback: one click on the centre, one on the rim."""
        r = float(np.hypot(rim[0] - centre[0], rim[1] - centre[1]))
        return cls(float(centre[0]), float(centre[1]), max(r, 1.0))


def fit_wheel(
    frames: Sequence[np.ndarray], radius_range: tuple[float, float] = DMD_RADIUS_RANGE
) -> Wheel | None:
    """Median frame -> Hough circle. ``frames`` are BGR images of one static camera.

    Needs a tight radius range: with a wide one the top vote can be a circle around wheel
    and lap, or the gear-selector ring. The live demo therefore defaults to two clicks.
    """
    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    scale = FIT_HEIGHT / median.shape[0]
    small = cv2.resize(median, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=FIT_HEIGHT / 2, param1=80, param2=40,
        minRadius=int(radius_range[0] * FIT_HEIGHT),
        maxRadius=int(radius_range[1] * FIT_HEIGHT),
    )
    if circles is None:
        return None
    x, y, r = (float(v) / scale for v in circles[0][0])
    return Wheel(x, y, r)


class WheelCalibrator:
    """Collects ``seconds`` of frames (one every ``every_s``), then fits the wheel once."""

    def __init__(
        self, seconds: float = 2.0, every_s: float = 0.1,
        radius_range: tuple[float, float] = DMD_RADIUS_RANGE,
    ) -> None:
        self.seconds, self.every_s, self.radius_range = seconds, every_s, radius_range
        self.active = False
        self.failed = False
        self.frames: list[np.ndarray] = []
        self.t0 = self.last_t = 0.0

    def start(self, t: float) -> None:
        self.active, self.failed = True, False
        self.frames = []
        self.t0, self.last_t = t, -np.inf

    def progress(self, t: float) -> float:
        return min(1.0, (t - self.t0) / self.seconds) if self.active else 0.0

    def push(self, t: float, frame: np.ndarray) -> Wheel | None:
        """Returns the wheel once calibration finishes, otherwise None."""
        if not self.active:
            return None
        if t - self.last_t >= self.every_s:
            self.frames.append(frame.copy())
            self.last_t = t
        if t - self.t0 < self.seconds:
            return None
        self.active = False
        wheel = fit_wheel(self.frames, self.radius_range)
        self.failed = wheel is None
        self.frames = []
        return wheel


@dataclass(frozen=True)
class Hand:
    centre: tuple[float, float]  # pixels
    score: float
    points: np.ndarray | None = None  # (21, 2) pixels, for drawing


def hands_from_result(result: object, width: int, height: int) -> list[Hand]:
    """MediaPipe HandLandmarker result -> hands in pixel coordinates."""
    hands = []
    for landmarks, handedness in zip(
        getattr(result, "hand_landmarks", None) or [],
        getattr(result, "handedness", None) or [], strict=False,
    ):
        pts = np.array([(p.x * width, p.y * height) for p in landmarks], np.float32)
        centre = pts.mean(axis=0)
        hands.append(Hand((float(centre[0]), float(centre[1])), float(handedness[0].score), pts))
    return hands


@dataclass
class HandsConfig:
    # Defaults = what leave-one-driver-out tuning picked on DMD (results/hands_lodo.json).
    rim_radii: float = 1.2
    min_score: float = 0.3
    switch_s: float = 0.17  # ~5 frames at 30 FPS; shorter alerts flicker
    # Which side of the wheel the driver's left hand appears on. False for a webcam
    # facing the driver (mirror view); True for DMD's camera behind the driver.
    driver_left_on_image_left: bool = False


def on_wheel_sides(
    hands: Sequence[Hand], wheel: Wheel, config: HandsConfig
) -> tuple[bool, bool]:
    """(driver's left hand on wheel, driver's right hand on wheel) for one frame."""
    left = right = False
    for hand in hands:
        if hand.score < config.min_score or wheel.radii(hand.centre) > config.rim_radii:
            continue
        if (hand.centre[0] < wheel.cx) == config.driver_left_on_image_left:
            left = True
        else:
            right = True
    return left, right


def contact_state(left: bool, right: bool) -> str:
    if left and right:
        return "both_hands"
    if right:
        return "only_right"
    if left:
        return "only_left"
    return HANDS_OFF


@dataclass
class HandsOutput:
    state: str  # one of CONTACT_STATES, or "no_wheel"
    raw_state: str
    hands_off_seconds: float
    left_on: bool = False
    right_on: bool = False
    reasons: list[str] = field(default_factory=list)


class HandsMonitor:
    """Causal debounce of the per-frame contact state, plus hands-off events."""

    def __init__(self, config: HandsConfig | None = None) -> None:
        self.config = config or HandsConfig()
        self.events: list[TemporalEvent] = []
        self.reset()

    def reset(self) -> None:
        self.state: str | None = None
        self.since = 0.0  # when the current state's evidence started
        self.confirmed = 0.0  # when the current state was reported
        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.last_t = 0.0

    def update(self, t: float, hands: Sequence[Hand], wheel: Wheel | None) -> HandsOutput:
        if wheel is None:
            return HandsOutput("no_wheel", "no_wheel", 0.0, reasons=["wheel not calibrated"])
        left, right = on_wheel_sides(hands, wheel, self.config)
        raw = contact_state(left, right)
        if self.state is None:
            self.state, self.since, self.confirmed = raw, t, t
        elif raw == self.state:
            self.candidate = None
        elif raw != self.candidate:
            self.candidate, self.candidate_since = raw, t
        elif t - self.candidate_since >= self.config.switch_s:
            self._switch(raw, t)
        self.last_t = t

        off = t - self.since if self.state == HANDS_OFF else 0.0
        reasons = [f"hands off {off:.1f}s"] if self.state == HANDS_OFF else []
        return HandsOutput(self.state, raw, off, left, right, reasons)

    def _switch(self, new_state: str, t: float) -> None:
        if self.state == HANDS_OFF:
            self._close_event(self.candidate_since)
        self.state, self.since, self.confirmed = new_state, self.candidate_since, t
        self.candidate = None

    def _close_event(self, end_s: float) -> None:
        self.events.append(TemporalEvent(
            label=HANDS_OFF, start_s=self.since, end_s=max(end_s, self.confirmed),
            confirmed_s=self.confirmed,
        ))

    def finish(self) -> list[TemporalEvent]:
        """Close an open hands-off event at the last timestamp and return all events."""
        if self.state == HANDS_OFF:
            self._close_event(self.last_t)
            self.state = None
        return list(self.events)


def draw_hands(
    frame: np.ndarray, wheel: Wheel | None, hands: Sequence[Hand], out: HandsOutput,
    config: HandsConfig, calibrating: float | None = None,
) -> None:
    """Wheel rim, on-rim band, hand centroids and a hands-off banner."""
    h, w = frame.shape[:2]
    if calibrating is not None:
        cv2.putText(frame, f"WHEEL CALIBRATION {calibrating:.0%} - keep the wheel in view",
                    (12, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if wheel is not None:
        on = out.state not in (HANDS_OFF, "no_wheel")
        colour = (80, 200, 80) if on else (0, 0, 255)
        centre = (int(wheel.cx), int(wheel.cy))
        cv2.circle(frame, centre, int(wheel.r), colour, 3)
        cv2.circle(frame, centre, int(wheel.r * config.rim_radii), colour, 1, cv2.LINE_AA)
    for hand in hands:
        inside = wheel is not None and wheel.radii(hand.centre) <= config.rim_radii
        colour = (80, 200, 80) if inside and hand.score >= config.min_score else (0, 165, 255)
        if hand.points is not None:
            for x, y in hand.points:
                cv2.circle(frame, (int(x), int(y)), 2, colour, -1)
        cv2.circle(frame, tuple(int(v) for v in hand.centre), 7, colour, -1)
    if out.state == "no_wheel":
        text, colour = "HANDS: press w (auto) or click centre+rim", (160, 160, 160)
    elif out.state == HANDS_OFF:
        text, colour = f"HANDS OFF {out.hands_off_seconds:.1f}s", (0, 0, 255)
    else:
        text, colour = f"HANDS: {out.state.replace('_', ' ')}", (80, 200, 80)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x0 = w - tw - 20
    cv2.rectangle(frame, (x0 - 8, h - th - 30), (w - 10, h - 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
