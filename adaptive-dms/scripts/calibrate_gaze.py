"""Guided per-driver gaze calibration (webcam, or a video to test the mechanics).

The driver looks at each of the 9 DMD gaze zones in turn, for `--passes` rounds.
Frames from the capture phase give BlazeGaze features that are saved as a
profile; demo_realtime.py --gaze-profile fits the calibrator from it.
With >= 2 passes the script also self-checks: fit on pass 1, test on the rest.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import demo_realtime as demo
from adaptive_dms.data import GAZE_ZONES
from adaptive_dms.gaze_calibration import GazeCalibrator, save_profile

PROMPTS = {
    "left_mirror": "LEFT MIRROR",
    "left": "LEFT SIDE WINDOW",
    "front": "ROAD AHEAD",
    "center_mirror": "REAR-VIEW MIRROR",
    "front_right": "FRONT RIGHT",
    "right_mirror": "RIGHT MIRROR",
    "right": "RIGHT SIDE WINDOW",
    "infotainment": "CENTER SCREEN",
    "steering_wheel": "STEERING WHEEL / DASH",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument("--task", type=Path,
                        default=Path("weights/face_landmarker_v2_with_blendshapes.task"))
    parser.add_argument("--keras-weights", type=Path,
                        default=Path("weights/blazegaze_mpiifacegaze.keras"))
    parser.add_argument("--adapter", type=Path,
                        default=Path("weights/master_4driver_blazegaze_dmd_adapter.pt"))
    parser.add_argument("--profile-out", type=Path, default=Path("weights/gaze_profile.npz"))
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--ready-seconds", type=float, default=2.0)
    parser.add_argument("--capture-seconds", type=float, default=2.5)
    parser.add_argument("--settle-seconds", type=float, default=0.7)
    parser.add_argument("--gaze-fps", type=float, default=10.0)
    parser.add_argument("--gaze-window", type=int, default=8)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch = demo.GazeBranch(args, device)
    is_webcam = args.source.isdigit()
    capture = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0

    sequence = [(p, z) for p in range(args.passes) for z in range(len(GAZE_ZONES))]
    segment = args.ready_seconds + args.capture_seconds
    total = len(sequence) * segment
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.VIDEO,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    store: dict[str, list] = {"embedding": [], "head": [], "origin": [], "zone": [], "pass": []}
    start = time.perf_counter()
    last_sample = -np.inf
    frame_index = 0
    aborted = False
    with vision.FaceLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] > args.max_width:
                s = args.max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            t = (time.perf_counter() - start) if is_webcam else frame_index / native_fps
            frame_index += 1
            if t >= total:
                break
            index = int(t // segment)
            within = t - index * segment
            pass_id, zone = sequence[index]
            capturing = within >= args.ready_seconds
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            face = bool(result.face_landmarks and len(result.face_landmarks[0]) == 478)
            if (face and capturing and within - args.ready_seconds >= args.settle_seconds
                    and t - last_sample >= branch.period):
                pts_norm = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], np.float32)
                pts_px = pts_norm[:, :2] * np.array([w, h], np.float32)
                rt = np.array(result.facial_transformation_matrixes[0], np.float32).reshape(4, 4)
                try:
                    f = branch.extract(rgb, pts_norm, pts_px, rt)
                except (ValueError, cv2.error):
                    f = None
                if f is not None:
                    last_sample = t
                    store["embedding"].append(f["embedding"])
                    store["head"].append(f["head"])
                    store["origin"].append(f["origin"])
                    store["zone"].append(zone)
                    store["pass"].append(pass_id)
            if not args.no_display:
                color = (0, 220, 0) if capturing else (0, 200, 255)
                cv2.putText(frame, f"LOOK AT: {PROMPTS[GAZE_ZONES[zone]]}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)
                cv2.putText(frame, ("HOLD STILL, KEEP LOOKING" if capturing else "get ready...")
                            + f"   pass {pass_id + 1}/{args.passes}  zone {zone + 1}/9  "
                              f"samples {len(store['zone'])}"
                            + ("" if face else "   NO FACE"), (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.rectangle(frame, (0, h - 12), (int(w * t / total), h), color, -1)
                cv2.imshow("Gaze calibration (q = abort)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break
    capture.release()
    cv2.destroyAllWindows()
    if aborted:
        raise SystemExit("aborted, nothing saved")

    zone = np.asarray(store["zone"], np.int64)
    pass_id = np.asarray(store["pass"], np.int64)
    counts = {GAZE_ZONES[z]: int(np.sum(zone == z)) for z in range(len(GAZE_ZONES))}
    print(f"collected {len(zone)} samples; per zone {counts}")
    missing = [z for z, c in counts.items() if c < 3]
    if missing:
        print(f"WARNING: fewer than 3 samples for {missing}; look at those targets longer")
    emb, head, origin = (np.stack(store[k]) for k in ("embedding", "head", "origin"))
    if len(zone) < 4 or len(np.unique(zone)) < 3:
        raise SystemExit("too few samples to calibrate; check that your face is visible")
    save_profile(args.profile_out, emb, head, origin, zone)
    print(f"saved {args.profile_out}")

    if args.passes >= 2:
        first, rest = pass_id == 0, pass_id > 0
        if first.sum() >= 4 and len(np.unique(zone[first])) >= 3 and rest.any():
            calibrator = GazeCalibrator(branch.blazegaze.gaze_mlp)
            calibrator.fit(emb[first], head[first], origin[first], zone[first])
            _, predicted, _ = calibrator.predict(emb[rest], head[rest], origin[rest])
            print(f"self-check (fit on pass 1, test on later passes): "
                  f"zone accuracy {np.mean(predicted == zone[rest]):.2f}, "
                  f"{int(rest.sum())} frames")


if __name__ == "__main__":
    main()
