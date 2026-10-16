"""Causal temporal decoding and event-level evaluation utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalEvent:
    """A detected interval and the causal time at which it was confirmed."""

    label: str
    start_s: float
    end_s: float
    confidence: float = 1.0
    confirmed_s: float | None = None

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError("event end must not precede its start")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be in [0, 1]")
        confirmed = self.start_s if self.confirmed_s is None else self.confirmed_s
        if confirmed < self.start_s:
            raise ValueError("confirmation must not precede event onset")
        object.__setattr__(self, "confirmed_s", confirmed)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class EventDecoderConfig:
    """Hysteresis parameters for one causal event stream."""

    onset_threshold: float = 0.6
    offset_threshold: float = 0.4
    min_duration_s: float = 0.5
    max_gap_s: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.offset_threshold <= self.onset_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= offset <= onset <= 1")
        if self.min_duration_s < 0.0 or self.max_gap_s < 0.0:
            raise ValueError("durations must be non-negative")


def _inputs(
    probabilities: Sequence[float] | np.ndarray,
    timestamps_s: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if probability.ndim != 1 or timestamps.ndim != 1:
        raise ValueError("probabilities and timestamps must be one-dimensional")
    if len(probability) != len(timestamps):
        raise ValueError("probabilities and timestamps must have equal length")
    if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(timestamps)):
        raise ValueError("probabilities and timestamps must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return probability, timestamps


def decode_events(
    probabilities: Sequence[float] | np.ndarray,
    timestamps_s: Sequence[float] | np.ndarray,
    label: str,
    config: EventDecoderConfig,
) -> list[TemporalEvent]:
    """Decode a probability stream without consulting future samples."""
    probability, timestamps = _inputs(probabilities, timestamps_s)
    events: list[TemporalEvent] = []
    start: float | None = None
    confirmed: float | None = None
    last_evidence: float | None = None
    samples: list[float] = []

    def finish(end_s: float) -> None:
        nonlocal start, confirmed, last_evidence, samples
        if start is not None and confirmed is not None:
            events.append(
                TemporalEvent(
                    label=label,
                    start_s=start,
                    end_s=end_s,
                    confidence=float(np.mean(samples)),
                    confirmed_s=confirmed,
                )
            )
        start = confirmed = last_evidence = None
        samples = []

    for timestamp, score in zip(timestamps, probability, strict=True):
        current = float(timestamp)
        value = float(score)
        if start is None:
            if value >= config.onset_threshold:
                start = last_evidence = current
                samples = [value]
                if config.min_duration_s == 0.0:
                    confirmed = current
            continue

        samples.append(value)
        if value >= config.offset_threshold:
            last_evidence = current
        if confirmed is None:
            gap = current - last_evidence if last_evidence is not None else 0.0
            if value < config.offset_threshold and gap > config.max_gap_s:
                finish(current)
            elif current - start >= config.min_duration_s:
                confirmed = current
            continue
        if last_evidence is not None and current - last_evidence > config.max_gap_s:
            finish(last_evidence)

    if start is not None and confirmed is not None and len(timestamps):
        finish(float(timestamps[-1]))
    return events


def interval_iou(first: TemporalEvent, second: TemporalEvent) -> float:
    intersection = max(
        0.0,
        min(first.end_s, second.end_s) - max(first.start_s, second.start_s),
    )
    union = max(first.end_s, second.end_s) - min(first.start_s, second.start_s)
    return intersection / union if union > 0.0 else float(intersection == union)


def _event_metrics(
    references: list[TemporalEvent],
    predictions: list[TemporalEvent],
    duration_s: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    candidates = []
    for reference_index, reference in enumerate(references):
        for prediction_index, prediction in enumerate(predictions):
            overlap = interval_iou(reference, prediction)
            if overlap >= iou_threshold:
                candidates.append((overlap, reference_index, prediction_index))
    candidates.sort(reverse=True)
    matched_reference: set[int] = set()
    matched_prediction: set[int] = set()
    delays: list[float] = []
    duration_errors: list[float] = []
    for _, reference_index, prediction_index in candidates:
        if reference_index in matched_reference or prediction_index in matched_prediction:
            continue
        matched_reference.add(reference_index)
        matched_prediction.add(prediction_index)
        reference = references[reference_index]
        prediction = predictions[prediction_index]
        delays.append(float(prediction.confirmed_s) - reference.start_s)
        duration_errors.append(prediction.duration_s - reference.duration_s)

    true_positive = len(matched_reference)
    false_positive = len(predictions) - true_positive
    false_negative = len(references) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_hour": false_positive / (duration_s / 3600.0),
        "mean_onset_delay_s": float(np.mean(delays)) if delays else 0.0,
        "mean_abs_onset_delay_s": float(np.mean(np.abs(delays))) if delays else 0.0,
        "mean_duration_error_s": (
            float(np.mean(duration_errors)) if duration_errors else 0.0
        ),
    }


def evaluate_events(
    references: Sequence[TemporalEvent],
    predictions: Sequence[TemporalEvent],
    recording_duration_s: float,
    iou_threshold: float = 0.3,
) -> dict[str, object]:
    """Match same-label intervals and report event metrics by class and overall."""
    if recording_duration_s <= 0.0:
        raise ValueError("recording duration must be positive")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in [0, 1]")
    labels = sorted({event.label for event in (*references, *predictions)})
    per_label = {
        label: _event_metrics(
            [event for event in references if event.label == label],
            [event for event in predictions if event.label == label],
            recording_duration_s,
            iou_threshold,
        )
        for label in labels
    }
    totals: dict[str, float | int] = {
        key: sum(int(metrics[key]) for metrics in per_label.values())
        for key in ("true_positive", "false_positive", "false_negative")
    }
    totals["macro_f1"] = (
        float(np.mean([float(metrics["f1"]) for metrics in per_label.values()]))
        if per_label
        else 0.0
    )
    totals["false_alarms_per_hour"] = totals["false_positive"] / (
        recording_duration_s / 3600.0
    )
    return {"per_label": per_label, "overall": totals}
