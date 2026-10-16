"""Conservative mapping from auxiliary DMS signals to competition states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from adaptive_dms.events import EventDecoderConfig, TemporalEvent, decode_events


class CompetitionState(StrEnum):
    ALERT = "alert"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    YAWNING = "yawning"
    MICROSLEEP = "microsleep"


@dataclass(frozen=True)
class DriverStatePolicy:
    """Temporal policy; microsleep requires its own validated probability."""

    minimum_quality: float = 0.25
    drowsy: EventDecoderConfig = field(
        default_factory=lambda: EventDecoderConfig(0.65, 0.45, 1.0, 0.4)
    )
    distracted: EventDecoderConfig = field(
        default_factory=lambda: EventDecoderConfig(0.65, 0.45, 1.0, 0.4)
    )
    yawning: EventDecoderConfig = field(
        default_factory=lambda: EventDecoderConfig(0.60, 0.40, 0.8, 0.4)
    )
    microsleep: EventDecoderConfig = field(
        default_factory=lambda: EventDecoderConfig(0.80, 0.60, 2.0, 0.2)
    )


@dataclass(frozen=True)
class StateTimeline:
    timestamps_s: np.ndarray
    states: tuple[CompetitionState, ...]
    events: Mapping[CompetitionState, tuple[TemporalEvent, ...]]


def competition_state_timeline(
    timestamps_s: Sequence[float] | np.ndarray,
    *,
    drowsy_probability: Sequence[float] | np.ndarray,
    distracted_probability: Sequence[float] | np.ndarray,
    yawning_probability: Sequence[float] | np.ndarray,
    microsleep_probability: Sequence[float] | np.ndarray | None = None,
    quality: Sequence[float] | np.ndarray | None = None,
    policy: DriverStatePolicy | None = None,
) -> StateTimeline:
    """Produce causal five-state outputs at the supplied timestamps.

    Eye closure is intentionally not accepted as an implicit microsleep signal.
    Pass a separately trained and validated ``microsleep_probability`` to enable it.
    """
    active_policy = policy or DriverStatePolicy()
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    signals = {
        CompetitionState.DROWSY: np.asarray(drowsy_probability, dtype=np.float64),
        CompetitionState.DISTRACTED: np.asarray(
            distracted_probability, dtype=np.float64
        ),
        CompetitionState.YAWNING: np.asarray(yawning_probability, dtype=np.float64),
    }
    if microsleep_probability is not None:
        signals[CompetitionState.MICROSLEEP] = np.asarray(
            microsleep_probability, dtype=np.float64
        )
    if quality is not None:
        reliability = np.asarray(quality, dtype=np.float64)
        if reliability.shape != timestamps.shape:
            raise ValueError("quality and timestamps must have equal shape")
        for state, signal in signals.items():
            signals[state] = np.where(
                reliability >= active_policy.minimum_quality, signal, 0.0
            )

    configs = {
        CompetitionState.DROWSY: active_policy.drowsy,
        CompetitionState.DISTRACTED: active_policy.distracted,
        CompetitionState.YAWNING: active_policy.yawning,
        CompetitionState.MICROSLEEP: active_policy.microsleep,
    }
    events = {
        state: tuple(decode_events(signal, timestamps, state.value, configs[state]))
        for state, signal in signals.items()
    }
    states = [CompetitionState.ALERT] * len(timestamps)
    precedence = (
        CompetitionState.DROWSY,
        CompetitionState.DISTRACTED,
        CompetitionState.YAWNING,
        CompetitionState.MICROSLEEP,
    )
    for state in precedence:
        for event in events.get(state, ()):
            causal_start = float(event.confirmed_s)
            mask = (timestamps >= causal_start) & (timestamps <= event.end_s)
            for index in np.flatnonzero(mask):
                states[int(index)] = state
    return StateTimeline(timestamps, tuple(states), events)
