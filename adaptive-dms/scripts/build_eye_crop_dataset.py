"""Build a 16x16 eye-crop dataset from DMD face videos (MediaPipe once per frame).

Per frame it stores the aligned L/R eye crops, the blink label, and the EAR /
blendshape baselines so the CNN can be compared with them on identical frames.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from adaptive_dms.data import load_gaze_session
from adaptive_dms.dms import compute_ear
from adaptive_dms.dms.blazegaze import rotation_matrix_to_euler_angles
from adaptive_dms.realtime import EyeCropConfig, eye_crops


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/dmd"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eye_crops_16"))
    parser.add_argument(
        "--task", type=Path, default=Path("weights/face_landmarker_v2_with_blendshapes.task")
    )
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    parser.add_argument("--max-width", type=int, default=960, help="downscale before MediaPipe")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


BLEND_NAMES: list[str] = []  # MediaPipe blendshape order, filled from the first valid frame


def blendshape_vector(result: object) -> np.ndarray:
    categories = result.face_blendshapes[0]
    if not BLEND_NAMES:
        BLEND_NAMES.extend(c.category_name for c in categories)
    return np.array([c.score for c in categories], np.float32)


def build_session(session, detector, config: EyeCropConfig, stride: int, max_width: int,
                  out_file: Path) -> dict[str, float | int]:
    t0 = time.perf_counter()
    capture = cv2.VideoCapture(str(session.face_video_path))
    native_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 29.76
    total = min(session.total_frames, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9)
    wanted = np.arange(0, total, stride)
    count = len(wanted)

    crops = np.zeros((count, 2, config.size, config.size), np.uint8)
    ear = np.zeros(count, np.float32)
    blink_score = np.zeros(count, np.float32)
    blend = np.zeros((count, 52), np.float16)
    euler = np.zeros((count, 3), np.float32)  # head rotation x/y/z in degrees (y = yaw)
    valid = np.zeros(count, bool)

    pos = frame_idx = 0
    last_ts = -1
    while pos < count:
        if not capture.grab():
            break
        if frame_idx != int(wanted[pos]):
            frame_idx += 1
            continue
        ok, frame = capture.retrieve()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, None, fx=max_width / w, fy=max_width / w,
                                   interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            ts = max(int(frame_idx / native_fps * 1000), last_ts + 1)
            last_ts = ts
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            if result.face_landmarks and len(result.face_landmarks[0]) == 478:
                pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], np.float32)
                try:
                    crops[pos] = eye_crops(frame, pts, config)
                    ear[pos] = 0.5 * (compute_ear(pts, "left") + compute_ear(pts, "right"))
                    vector = blendshape_vector(result)
                    blend[pos] = vector
                    blink_score[pos] = 0.5 * (vector[BLEND_NAMES.index("eyeBlinkLeft")]
                                              + vector[BLEND_NAMES.index("eyeBlinkRight")])
                    rt = np.array(result.facial_transformation_matrixes[0], np.float32)
                    euler[pos] = rotation_matrix_to_euler_angles(rt.reshape(4, 4)[:3, :3])
                    valid[pos] = True
                except (ValueError, cv2.error):
                    pass
        pos += 1
        frame_idx += 1
        if pos % 500 == 0 or pos == count:
            print(f"  {session.group}/{session.subject} {pos}/{count} valid={int(valid.sum())} "
                  f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)
    capture.release()

    n = pos
    idx = wanted[:n]
    blink = np.array([session.blinking[i] for i in idx], bool)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, crops=crops[:n], ear=ear[:n], blink_score=blink_score[:n], valid=valid[:n],
        blink=blink, frame_indices=idx.astype(np.int32),
        blend=blend[:n], blend_names=np.asarray(BLEND_NAMES), euler=euler[:n],
        gaze_zone_id=np.array([session.gaze_zone_id[i] for i in idx], np.int16),
        timestamps_s=(idx / native_fps).astype(np.float32),
        subject=np.asarray(f"{session.group}/{session.subject}"),
    )
    tmp.replace(out_file)
    return {"frames": n, "valid": int(valid[:n].sum()), "blink": int(blink[valid[:n]].sum()),
            "seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    args = arguments()
    annotations = sorted(args.root.rglob("*_ann_gaze.json"))
    if not annotations:
        raise SystemExit(f"No *_ann_gaze.json under {args.root}")
    config = EyeCropConfig(size=args.size)
    summary = {}
    for ann in annotations:
        session = load_gaze_session(ann)
        out_file = args.output / f"{session.group}_{session.subject}_{session.session_id}.npz"
        if out_file.exists() and not args.overwrite:
            print(f"skip {out_file}", flush=True)
            continue
        # A fresh detector per video keeps MediaPipe timestamps monotonic.
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.task)),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        with vision.FaceLandmarker.create_from_options(options) as detector:
            stats = build_session(session, detector, config, args.stride, args.max_width, out_file)
        summary[f"{session.group}/{session.subject}"] = stats
        print(f"done {session.group}/{session.subject} {json.dumps(stats)}", flush=True)
    print("summary", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
