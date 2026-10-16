import numpy as np
import pytest

from adaptive_dms.events import (
    EventDecoderConfig,
    TemporalEvent,
    decode_events,
    evaluate_events,
)


def test_decoder_rejects_spike_and_bridges_short_gap() -> None:
    timestamps = np.arange(0.0, 3.0, 0.2)
    probability = np.zeros_like(timestamps)
    probability[1] = 0.9
    probability[5:12] = 0.9
    probability[8] = 0.2
    events = decode_events(
        probability,
        timestamps,
        "yawning",
        EventDecoderConfig(0.6, 0.4, min_duration_s=0.6, max_gap_s=0.25),
    )
    assert len(events) == 1
    assert events[0].start_s == pytest.approx(1.0)
    assert events[0].confirmed_s == pytest.approx(1.6)
    assert events[0].end_s == pytest.approx(2.2)


def test_event_metrics_report_delay_and_false_alarms() -> None:
    reference = [TemporalEvent("yawning", 2.0, 5.0)]
    prediction = [
        TemporalEvent("yawning", 2.1, 5.1, confirmed_s=2.8),
        TemporalEvent("yawning", 8.0, 9.0, confirmed_s=8.5),
    ]
    result = evaluate_events(reference, prediction, recording_duration_s=360.0)
    yawn = result["per_label"]["yawning"]
    assert yawn["true_positive"] == 1
    assert yawn["false_positive"] == 1
    assert yawn["false_negative"] == 0
    assert yawn["f1"] == pytest.approx(2.0 / 3.0)
    assert yawn["mean_onset_delay_s"] == pytest.approx(0.8)
    assert yawn["false_alarms_per_hour"] == pytest.approx(10.0)


def test_event_validation_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="event end"):
        TemporalEvent("drowsy", 2.0, 1.0)
