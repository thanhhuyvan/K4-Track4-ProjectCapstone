"""Causal real-time building blocks: 16x16 eye crops, sliding windows, state monitor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# MediaPipe eye corners (outer, inner) per eye, in image coordinates.
LEFT_EYE_CORNERS = (263, 362)
RIGHT_EYE_CORNERS = (33, 133)


@dataclass(frozen=True)
class EyeCropConfig:
    """Same alignment as the LIPE gaze crop, applied per eye at 16x16."""

    size: int = 16
    eye_width_fraction: float = 0.80
    clahe_clip_limit: float = 1.2


DEFAULT_EYE_CROP = EyeCropConfig()


def affine_eye_crop(
    frame_bgr: np.ndarray,
    landmarks_px: np.ndarray,
    corners: tuple[int, int],
    config: EyeCropConfig = DEFAULT_EYE_CROP,
    mirror: bool = False,
) -> np.ndarray:
    """Rotate by the corner line, scale by eye width, return a size x size gray crop."""
    first = np.asarray(landmarks_px[corners[0], :2], dtype=np.float32)
    second = np.asarray(landmarks_px[corners[1], :2], dtype=np.float32)
    left, right = (first, second) if first[0] <= second[0] else (second, first)
    eye_width = float(np.linalg.norm(right - left))
    if eye_width < 1.0:
        raise ValueError("eye too small for affine crop")

    angle = float(np.degrees(np.arctan2(*(right - left)[::-1])))
    scale = config.size * config.eye_width_fraction / eye_width
    center = (left + right) / 2.0
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
    matrix[0, 2] += config.size / 2.0 - center[0]
    matrix[1, 2] += config.size / 2.0 - center[1]
    # Supersample then area-downscale so 16x16 is not aliased.
    big = config.size * 4
    matrix_big = matrix * np.array([[4.0], [4.0]], dtype=np.float64)
    crop = cv2.warpAffine(
        frame_bgr, matrix_big, (big, big), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.createCLAHE(config.clahe_clip_limit, (4, 4)).apply(gray)
    small = cv2.resize(gray, (config.size, config.size), interpolation=cv2.INTER_AREA)
    return cv2.flip(small, 1) if mirror else small


def eye_crops(
    frame_bgr: np.ndarray, landmarks_px: np.ndarray, config: EyeCropConfig = DEFAULT_EYE_CROP
) -> np.ndarray:
    """Return (2, size, size) uint8: [left eye, mirrored right eye]."""
    return np.stack(
        (
            affine_eye_crop(frame_bgr, landmarks_px, LEFT_EYE_CORNERS, config),
            affine_eye_crop(frame_bgr, landmarks_px, RIGHT_EYE_CORNERS, config, mirror=True),
        )
    )


class SlidingWindow:
    """Causal time window of scalar samples with O(1) amortised push and mean."""

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self._items: deque[tuple[float, float]] = deque()
        self._sum = 0.0

    def push(self, t: float, value: float) -> None:
        self._items.append((t, float(value)))
        self._sum += float(value)
        self.evict(t)

    def evict(self, now: float) -> None:
        while self._items and self._items[0][0] < now - self.seconds:
            self._sum -= self._items.popleft()[1]

    def drop_last(self) -> None:
        if self._items:
            self._sum -= self._items.pop()[1]

    def clear(self) -> None:
        self._items.clear()
        self._sum = 0.0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def span(self) -> float:
        return self._items[-1][0] - self._items[0][0] if len(self._items) > 1 else 0.0

    def mean(self) -> float:
        return self._sum / len(self._items) if self._items else 0.0

    def total(self) -> float:
        return self._sum

    def values(self) -> np.ndarray:
        return np.fromiter((v for _, v in self._items), dtype=np.float32, count=len(self))


class RunTracker:
    """Track contiguous True runs; completed runs are kept in a time window."""

    def __init__(self, history_seconds: float) -> None:
        self.start: float | None = None
        self.completed = SlidingWindow(history_seconds)

    def update(self, t: float, active: bool) -> float | None:
        """Return the finished run duration when a run ends, else None."""
        finished = None
        if active and self.start is None:
            self.start = t
        elif not active and self.start is not None:
            finished = t - self.start
            self.completed.push(t, finished)
            self.start = None
        self.completed.evict(t)
        return finished

    def current(self, t: float) -> float:
        return t - self.start if self.start is not None else 0.0

    def reset(self) -> None:
        self.start = None


@dataclass
class FrameSignals:
    """Per-frame inputs. Any field may be None when that branch did not run."""

    face: bool
    ear: float | None = None
    blink_score: float | None = None  # mean MediaPipe eyeBlink blendshape
    jaw_open: float | None = None
    mar: float | None = None
    head_yaw: float | None = None
    off_road: bool | None = None  # from gaze zone attention group
    cnn_prob: float | None = None  # eye-crop CNN p(closed), display/logging only
    closed_override: bool | None = None  # when set, replaces the EAR/blendshape rule


@dataclass
class MonitorConfig:
    blink_score_closed: float = 0.50
    ear_closed_ratio: float = 0.65  # closed if EAR < ratio * open-eye baseline
    ear_fallback_threshold: float = 0.20
    # EAR is distorted when the head is turned: ignore it (and keep it out of the
    # open-eye baseline) beyond this yaw. None disables the gate.
    ear_max_yaw_deg: float | None = 25.0
    jaw_open_yawn: float = 0.55
    microsleep_seconds: float = 1.0
    blink_max_seconds: float = 0.5
    yawn_min_seconds: float = 1.0
    perclos_window: float = 30.0
    perclos_drowsy: float = 0.15
    perclos_min_span: float = 10.0
    event_window: float = 60.0
    yawns_drowsy: int = 2
    distract_window: float = 2.0
    distract_ratio: float = 0.70
    head_yaw_distract: float = 35.0
    no_face_reset: float = 1.0
    hold_seconds: float = 1.0


@dataclass
class MonitorOutput:
    state: str
    eye_closed: bool
    ear_threshold: float
    perclos: float
    closure_seconds: float
    longest_closure: float
    blinks_per_min: float
    yawning_seconds: float
    yawns_recent: int
    off_road_ratio: float
    reasons: list[str] = field(default_factory=list)


class DriverMonitor:
    """Causal sliding-window driver-state decision from per-frame signals."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        c = self.config
        self.ear_baseline = SlidingWindow(30.0)
        self.closed = SlidingWindow(c.perclos_window)
        self.closure = RunTracker(c.event_window)
        self.yawn = RunTracker(c.event_window)
        self.off_road = SlidingWindow(c.distract_window)
        self.last_face_t: float | None = None
        self._hold_state = "alert"
        self._hold_until = -np.inf

    def reset(self) -> None:
        for w in (self.closed, self.off_road):
            w.clear()
        self.closure.reset()
        self.yawn.reset()

    def ear_threshold(self) -> float:
        values = self.ear_baseline.values()
        if len(values) < 20:
            return self.config.ear_fallback_threshold
        return float(np.percentile(values, 90)) * self.config.ear_closed_ratio

    def update(self, t: float, s: FrameSignals) -> MonitorOutput:
        c = self.config
        if not s.face:
            if self.last_face_t is not None and t - self.last_face_t > c.no_face_reset:
                self.reset()
            return self._output(t, "no_face", False, [])
        self.last_face_t = t

        threshold = self.ear_threshold()
        by_blend = s.blink_score is not None and s.blink_score >= c.blink_score_closed
        ear_usable = s.ear is not None and (
            c.ear_max_yaw_deg is None
            or s.head_yaw is None
            or abs(s.head_yaw) <= c.ear_max_yaw_deg
        )
        by_ear = ear_usable and s.ear < threshold
        eye_closed = bool(by_blend or by_ear)
        if s.closed_override is not None:
            eye_closed = bool(s.closed_override)
        if ear_usable and not eye_closed:
            self.ear_baseline.push(t, s.ear)

        self.closed.push(t, float(eye_closed))
        self.closure.update(t, eye_closed)

        yawn_active = (s.jaw_open is not None and s.jaw_open >= c.jaw_open_yawn) or (
            s.jaw_open is None and s.mar is not None and s.mar >= 0.6
        )
        finished_yawn = self.yawn.update(t, bool(yawn_active))
        if finished_yawn is not None and finished_yawn < c.yawn_min_seconds:
            # Too short to be a yawn (talking): drop it from history.
            self.yawn.completed.drop_last()

        if s.off_road is not None:
            self.off_road.push(t, float(s.off_road))
        elif s.head_yaw is not None:
            self.off_road.push(t, float(abs(s.head_yaw) > c.head_yaw_distract))

        reasons: list[str] = []
        state = "alert"
        closure_now = self.closure.current(t)
        perclos = self.closed.mean()
        yawns = len(self.yawn.completed)
        if closure_now >= c.microsleep_seconds:
            state = "microsleep"
            reasons.append(f"eyes closed {closure_now:.1f}s")
        elif (self.closed.span >= c.perclos_min_span and perclos >= c.perclos_drowsy) or (
            yawns >= c.yawns_drowsy
        ):
            state = "drowsy"
            reasons.append(f"PERCLOS {perclos:.0%}" if perclos >= c.perclos_drowsy
                           else f"{yawns} yawns/{c.event_window:.0f}s")
        elif self.yawn.current(t) >= c.yawn_min_seconds:
            state = "yawning"
            reasons.append(f"mouth open {self.yawn.current(t):.1f}s")
        elif self.off_road.span >= 0.5 * c.distract_window and (
            self.off_road.mean() >= c.distract_ratio
        ):
            state = "distracted"
            reasons.append(f"off-road {self.off_road.mean():.0%}")

        # Short hold so alarms do not flicker off between frames.
        if state != "alert":
            self._hold_state, self._hold_until = state, t + c.hold_seconds
        elif t < self._hold_until:
            state = self._hold_state
        return self._output(t, state, eye_closed, reasons, threshold)

    def _output(
        self, t: float, state: str, eye_closed: bool, reasons: list[str],
        threshold: float | None = None,
    ) -> MonitorOutput:
        c = self.config
        runs = self.closure.completed.values()
        blinks = int(np.sum(runs < c.blink_max_seconds)) if len(runs) else 0
        minutes = max(min(self.closure.completed.span, c.event_window), 1.0) / 60.0
        return MonitorOutput(
            state=state,
            eye_closed=eye_closed,
            ear_threshold=threshold if threshold is not None else self.ear_threshold(),
            perclos=self.closed.mean(),
            closure_seconds=self.closure.current(t),
            longest_closure=max(float(runs.max()) if len(runs) else 0.0,
                                self.closure.current(t)),
            blinks_per_min=blinks / minutes if len(runs) > 1 else 0.0,
            yawning_seconds=self.yawn.current(t),
            yawns_recent=len(self.yawn.completed),
            off_road_ratio=self.off_road.mean(),
            reasons=reasons,
        )
