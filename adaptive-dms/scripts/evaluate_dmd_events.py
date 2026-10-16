"""Evaluate frozen V2 yawn probabilities as subject-disjoint temporal events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_dms.dms import (
    CausalDmsBaselineV2,
    DriverStatePolicy,
    geometry_from_landmarks,
)
from adaptive_dms.events import (
    EventDecoderConfig,
    TemporalEvent,
    decode_events,
    evaluate_events,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-subject", default="7")
    parser.add_argument("--test-subject", default="9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def subject_cache(cache: Path, subject: str) -> Path:
    matches = sorted(cache.rglob(f"{subject}_s5.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one cache for subject {subject}, found {matches}")
    return matches[0]


@torch.no_grad()
def infer(
    model: CausalDmsBaselineV2,
    cache_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as data:
        timestamps = data["timestamps_s"].astype(np.float64)
        valid = data["valid"].astype(bool)
        target = (data["yawning"] > 0).astype(np.float64)
        geometry = geometry_from_landmarks(data["landmarks"], data["quality"])

        def tensor(key: str) -> torch.Tensor:
            value = torch.from_numpy(data[key]).float().div_(255.0)
            return value.unsqueeze(0).to(device)

        output = model(
            tensor("left_eye"),
            tensor("right_eye"),
            tensor("mouth"),
            torch.from_numpy(geometry).unsqueeze(0).to(device),
        )
    probability = output["yawn"].softmax(-1)[0, :, 1].cpu().numpy()
    probability[~valid] = 0.0
    return timestamps, target, probability


def ground_truth(target: np.ndarray, timestamps: np.ndarray) -> list[TemporalEvent]:
    return decode_events(
        target,
        timestamps,
        "yawning",
        EventDecoderConfig(0.5, 0.5, min_duration_s=0.0, max_gap_s=0.0),
    )


def duration(timestamps: np.ndarray) -> float:
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.2
    return float(timestamps[-1] - timestamps[0] + step)


def score(
    target_events: list[TemporalEvent],
    probability: np.ndarray,
    timestamps: np.ndarray,
    config: EventDecoderConfig,
) -> dict[str, object]:
    prediction = decode_events(probability, timestamps, "yawning", config)
    result = evaluate_events(target_events, prediction, duration(timestamps))
    result["reference_events"] = len(target_events)
    result["predicted_events"] = len(prediction)
    return result


def tune(
    target_events: list[TemporalEvent],
    probability: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[EventDecoderConfig, dict[str, object]]:
    best: tuple[tuple[float, float], EventDecoderConfig, dict[str, object]] | None = None
    for threshold in np.arange(0.4, 0.81, 0.05):
        for minimum in (0.4, 0.8, 1.2):
            config = EventDecoderConfig(
                onset_threshold=float(threshold),
                offset_threshold=float(max(0.05, threshold - 0.2)),
                min_duration_s=minimum,
                max_gap_s=0.4,
            )
            metrics = score(target_events, probability, timestamps, config)
            yawn = metrics["per_label"]["yawning"]
            rank = (float(yawn["f1"]), -float(yawn["false_alarms_per_hour"]))
            if best is None or rank > best[0]:
                best = (rank, config, metrics)
    if best is None:
        raise RuntimeError("event-policy search produced no candidate")
    return best[1], best[2]


def config_dict(config: EventDecoderConfig) -> dict[str, float]:
    return {
        "onset_threshold": config.onset_threshold,
        "offset_threshold": config.offset_threshold,
        "min_duration_s": config.min_duration_s,
        "max_gap_s": config.max_gap_s,
    }


def main() -> None:
    args = arguments()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CausalDmsBaselineV2().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    validation = infer(
        model, subject_cache(args.cache, args.validation_subject), device
    )
    test = infer(model, subject_cache(args.cache, args.test_subject), device)
    validation_truth = ground_truth(validation[1], validation[0])
    test_truth = ground_truth(test[1], test[0])
    default = DriverStatePolicy().yawning
    tuned, tuned_validation = tune(validation_truth, validation[2], validation[0])
    result = {
        "device": str(device),
        "validation_subject": args.validation_subject,
        "test_subject": args.test_subject,
        "default_policy": config_dict(default),
        "default_validation": score(
            validation_truth, validation[2], validation[0], default
        ),
        "default_test": score(test_truth, test[2], test[0], default),
        "validation_tuned_policy": config_dict(tuned),
        "tuned_validation": tuned_validation,
        "tuned_test": score(test_truth, test[2], test[0], tuned),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
