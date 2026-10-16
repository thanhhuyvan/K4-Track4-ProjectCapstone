import numpy as np

from adaptive_dms.dms import (
    CompetitionState,
    DriverStatePolicy,
    competition_state_timeline,
)
from adaptive_dms.events import EventDecoderConfig


def test_policy_is_causal_and_uses_explicit_precedence() -> None:
    timestamps = np.arange(0.0, 4.5, 0.5)
    drowsy = np.full(len(timestamps), 0.9)
    distracted = np.zeros(len(timestamps))
    distracted[2:7] = 0.9
    yawning = np.zeros(len(timestamps))
    yawning[3:7] = 0.9
    microsleep = np.zeros(len(timestamps))
    microsleep[2:8] = 0.95
    immediate = EventDecoderConfig(0.6, 0.4, 0.5, 0.0)
    policy = DriverStatePolicy(
        drowsy=immediate,
        distracted=immediate,
        yawning=immediate,
        microsleep=EventDecoderConfig(0.8, 0.6, 1.0, 0.0),
    )
    timeline = competition_state_timeline(
        timestamps,
        drowsy_probability=drowsy,
        distracted_probability=distracted,
        yawning_probability=yawning,
        microsleep_probability=microsleep,
        policy=policy,
    )
    assert timeline.states[0] is CompetitionState.ALERT
    assert timeline.states[1] is CompetitionState.DROWSY
    assert timeline.states[3] is CompetitionState.DISTRACTED
    assert timeline.states[4] is CompetitionState.MICROSLEEP


def test_policy_does_not_invent_microsleep() -> None:
    timestamps = np.arange(0.0, 4.0, 0.5)
    zeros = np.zeros(len(timestamps))
    timeline = competition_state_timeline(
        timestamps,
        drowsy_probability=np.ones(len(timestamps)),
        distracted_probability=zeros,
        yawning_probability=zeros,
    )
    assert CompetitionState.MICROSLEEP not in timeline.events
    assert CompetitionState.MICROSLEEP not in timeline.states
