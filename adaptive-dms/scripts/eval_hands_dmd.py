"""Evaluate the live hands-on-wheel branch (src/adaptive_dms/hands.py) on DMD.

Runs the same HandsMonitor the demo uses on each driver's hands-camera video and reports
leave-one-driver-out results: rim_radii / min_score / switch_s are tuned on three drivers
(first reach >= 90% hands-off frame recall, then maximise 4-class macro-F1) and tested on
the fourth.

    python scripts/eval_hands_dmd.py --dmd-root data/dmd --out results/hands_lodo.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from adaptive_dms.data import load_gaze_session
from adaptive_dms.hands import (
    CONTACT_STATES,
    DMD_RADIUS_RANGE,
    HANDS_OFF,
    Hand,
    HandsConfig,
    HandsMonitor,
    Wheel,
    fit_wheel,
    hands_from_result,
)

DMD_TO_STATE = {"both_hands": 0, "only_right": 1, "only_left": 2, "none": 3}
OFF = CONTACT_STATES.index(HANDS_OFF)
MIN_RECALL = 0.9
GRID = {
    "rim_radii": (1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
    "min_score": (0.3, 0.5, 0.7),
    "switch_s": (0.17, 0.27, 0.4, 0.6),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmd-root", type=Path, default=Path("data/dmd"))
    parser.add_argument("--hand-task", type=Path, default=Path("weights/hand_landmarker.task"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/hands_dmd"))
    parser.add_argument("--out", type=Path, default=Path("results/hands_lodo.json"))
    return parser.parse_args()


def extract(video: Path, task: Path, cache: Path) -> tuple[Wheel, np.ndarray, float]:
    """Wheel + per-frame hands (N, 2, 3: cx, cy, score; NaN when absent), cached."""
    if cache.with_suffix(".npz").exists():
        d = np.load(cache.with_suffix(".npz"))
        return Wheel(*d["wheel"]), d["hands"], float(d["fps"])
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, 150).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    wheel = fit_wheel(frames, DMD_RADIUS_RANGE)
    if wheel is None:
        raise SystemExit(f"no wheel found in {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(task)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3,
    )
    rows = []
    with vision.HandLandmarker.create_from_options(options) as detector:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                int(index * 1000 / fps),
            )
            row = np.full((2, 3), np.nan, np.float32)
            for slot, hand in enumerate(hands_from_result(result, w, h)[:2]):
                row[slot] = (*hand.centre, hand.score)
            rows.append(row)
            index += 1
    capture.release()
    hands = np.stack(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache.with_suffix(".npz"), wheel=(wheel.cx, wheel.cy, wheel.r),
                        hands=hands, fps=fps)
    return wheel, hands, fps


def load_driver(session_dir: Path, args: argparse.Namespace) -> dict:
    gaze_json = next(session_dir.glob("*_rgb_ann_gaze.json"))
    hands_json = next(session_dir.glob("*_rgb_ann_hands.json"))
    session = load_gaze_session(gaze_json, hands_json)
    streams = json.loads(hands_json.read_text(encoding="utf-8"))["openlabel"]["streams"]
    shift = int(streams["hands_camera"]["stream_properties"]["sync"]["frame_shift"])
    name = f"{session.group}_{session.subject}"
    wheel, hands, fps = extract(next(session_dir.glob("*_rgb_hands.mp4")), args.hand_task,
                                args.cache / name)
    labels = np.array([DMD_TO_STATE.get(s, -1) for s in session.hands_on_wheel])
    master = np.arange(shift, min(len(labels), shift + len(hands)))
    keep = labels[master] >= 0
    return {"name": name, "wheel": wheel, "hands": hands[master - shift][keep],
            "y": labels[master][keep], "t": master[keep] / fps}


def run(driver: dict, config: HandsConfig) -> tuple[np.ndarray, list]:
    monitor = HandsMonitor(config)
    states = np.empty(len(driver["y"]), int)
    for i, (t, row) in enumerate(zip(driver["t"], driver["hands"], strict=True)):
        hands = [Hand((float(x), float(y)), float(s)) for x, y, s in row if not np.isnan(s)]
        states[i] = CONTACT_STATES.index(monitor.update(float(t), hands, driver["wheel"]).state)
    return states, monitor.finish()


def frame_metrics(y: np.ndarray, pred: np.ndarray, fps: float) -> dict:
    f1s = []
    for c in range(len(CONTACT_STATES)):
        tp = np.sum((pred == c) & (y == c))
        denom = np.sum(pred == c) + np.sum(y == c)
        f1s.append(2 * tp / denom if denom else 0.0)
    off = y == OFF
    return {
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": dict(zip(CONTACT_STATES, map(float, f1s), strict=True)),
        "hands_off_recall": float(np.mean(pred[off] == OFF)) if off.any() else None,
        "false_hands_off_seconds": float(np.sum((pred == OFF) & ~off) / fps),
    }


def gt_segments(y: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    off = np.concatenate([[False], y == OFF, [False]]).astype(int)
    starts, ends = np.flatnonzero(np.diff(off) == 1), np.flatnonzero(np.diff(off) == -1) - 1
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends, strict=True)]


def event_metrics(y: np.ndarray, t: np.ndarray, events: list) -> dict:
    gt = gt_segments(y, t)
    hit, delays = 0, []
    for gs, ge in gt:
        overlapping = [e for e in events if e.start_s <= ge and e.end_s >= gs]
        if overlapping:
            hit += 1
            delays.append(float(overlapping[0].confirmed_s) - gs)
    false = sum(1 for e in events if not any(e.start_s <= ge and e.end_s >= gs for gs, ge in gt))
    return {"gt_events": len(gt), "hit": hit, "false_alarms": false, "onset_delays_s": delays}


def tune(drivers: list[dict]) -> HandsConfig:
    best = None
    for rim, score, switch in itertools.product(*GRID.values()):
        config = HandsConfig(rim_radii=rim, min_score=score, switch_s=switch,
                             driver_left_on_image_left=True)
        ms = [frame_metrics(d["y"], run(d, config)[0], 30.0) for d in drivers]
        recall = float(np.mean([m["hands_off_recall"] for m in ms]))
        key = (max(0.0, MIN_RECALL - recall), -float(np.mean([m["macro_f1"] for m in ms])))
        if best is None or key < best[0]:
            best = (key, config)
    return best[1]


def main() -> None:
    args = arguments()
    sessions = sorted(p.parent for p in args.dmd_root.glob("*/*/*/*_rgb_ann_hands.json"))
    drivers = [load_driver(s, args) for s in sessions]
    print(f"{len(drivers)} drivers: {[d['name'] for d in drivers]}", flush=True)
    folds = []
    for test in drivers:
        config = tune([d for d in drivers if d is not test])
        pred, events = run(test, config)
        fps = 1.0 / float(np.median(np.diff(test["t"])))
        fold = {
            "driver": test["name"],
            "config": {k: getattr(config, k) for k in GRID},
            **frame_metrics(test["y"], pred, fps),
            "events": event_metrics(test["y"], test["t"], events),
        }
        folds.append(fold)
        ev = fold["events"]
        print(f"test={test['name']} {fold['config']} macroF1={fold['macro_f1']:.2f} "
              f"hands-off recall={fold['hands_off_recall']:.1%} "
              f"events {ev['hit']}/{ev['gt_events']} false={ev['false_alarms']}", flush=True)

    delays = [d for f in folds for d in f["events"]["onset_delays_s"]]
    summary = {
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
        "macro_f1_min": float(np.min([f["macro_f1"] for f in folds])),
        "hands_off_recall_mean": float(np.mean([f["hands_off_recall"] for f in folds])),
        "hands_off_recall_min": float(np.min([f["hands_off_recall"] for f in folds])),
        "events_hit": f"{sum(f['events']['hit'] for f in folds)}/"
                      f"{sum(f['events']['gt_events'] for f in folds)}",
        "false_alarms": sum(f["events"]["false_alarms"] for f in folds),
        "false_hands_off_seconds": sum(f["false_hands_off_seconds"] for f in folds),
        "minutes": sum(len(d["t"]) for d in drivers) / 29.76 / 60,
        "onset_delay_s_median": float(np.median(delays)) if delays else None,
    }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"protocol": __doc__.strip().splitlines()[0],
                                    "summary": summary, "per_driver": folds}, indent=2))


if __name__ == "__main__":
    main()
