"""Real-time DMS demo: webcam or DMD video -> MediaPipe (once per frame)
-> BlazeGaze + 4-driver DMD adapter (gaze zone) and 16x16 eye crops + EAR/blendshapes
-> causal sliding windows -> driver state overlay.
With --hands: MediaPipe hand landmarks on the same frame -> wheel geometry -> hands-off alert.

Keys: q = quit, space = pause, r = reset windows, w = re-fit the wheel (--hands);
with --hands, two clicks (wheel centre, then rim) place the wheel by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from adaptive_dms.data import GAZE_ZONES
from adaptive_dms.dms import (
    BlazeGazeDmdAdapter,
    BlazeGazePyTorch,
    EyeBlinkCNN,
    compute_ear,
    compute_face_origin_3d,
    geometry_from_landmarks,
    get_head_vector,
    obtain_eyepatch,
)
from adaptive_dms.dms.blazegaze import rotation_matrix_to_euler_angles
from adaptive_dms.gaze_calibration import GazeCalibrator, load_profile
from adaptive_dms.hands import (
    HandsConfig,
    HandsMonitor,
    Wheel,
    WheelCalibrator,
    draw_hands,
    hands_from_result,
)
from adaptive_dms.realtime import DriverMonitor, FrameSignals, MonitorConfig, eye_crops

WINDOW = "Adaptive DMS demo"

# 0 on_road, 1 mirror_check, 2 distracted (same grouping as training).
ZONE_TO_ATTENTION = (1, 2, 0, 1, 0, 1, 2, 2, 2)
STATE_COLORS = {
    "alert": (80, 200, 80),
    "distracted": (0, 200, 255),
    "yawning": (0, 170, 255),
    "drowsy": (0, 100, 255),
    "microsleep": (0, 0, 255),
    "no_face": (160, 160, 160),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument(
        "--task", type=Path, default=Path("weights/face_landmarker_v2_with_blendshapes.task")
    )
    parser.add_argument(
        "--keras-weights", type=Path, default=Path("weights/blazegaze_mpiifacegaze.keras")
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("weights/master_4driver_blazegaze_dmd_adapter.pt"),
        help="BlazeGazeDmdAdapter checkpoint; omit gaze with --no-gaze",
    )
    parser.add_argument("--no-gaze", action="store_true")
    parser.add_argument(
        "--gaze-profile", type=Path, default=None,
        help="per-driver calibration profile from calibrate_gaze.py; replaces the 4-driver adapter",
    )
    parser.add_argument(
        "--eye-model", type=Path, default=Path("weights/eye_cnn_16.pt"),
        help="16x16 eye-crop CNN checkpoint (from train_eye_crop.py)",
    )
    parser.add_argument(
        "--eye-signal", choices=("rules", "cnn"), default="rules",
        help="rules = EAR + blendshape; cnn = eye-crop CNN decides closed/open "
        "(the CNN probability is always displayed when the checkpoint exists)",
    )
    parser.add_argument("--gaze-fps", type=float, default=10.0, help="matches training cache")
    parser.add_argument("--gaze-window", type=int, default=8, help="matches training windows")
    parser.add_argument("--max-width", type=int, default=1280, help="downscale input above this")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save", type=Path, default=None, help="write annotated mp4")
    parser.add_argument("--log-csv", type=Path, default=None, help="per-frame signals/state")
    parser.add_argument("--save-crops", type=Path, default=None, help="npz of 16x16 eye crops")
    parser.add_argument("--hands", action="store_true", help="hands-on-wheel branch")
    parser.add_argument(
        "--hand-task", type=Path, default=Path("weights/hand_landmarker.task")
    )
    parser.add_argument(
        "--driver-left-on-image-left", action="store_true",
        help="camera behind the driver (DMD hands camera); default is a webcam facing them",
    )
    parser.add_argument(
        "--hands-events", type=Path, default=None, help="json of hands-off TemporalEvents"
    )
    parser.add_argument(
        "--wheel-radius-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="auto-fit the wheel at start-up with this radius range (fraction of frame "
        "height; DMD hands camera: 0.28 0.40). Default: click centre, then rim",
    )
    return parser.parse_args()


def quality_features(points: np.ndarray) -> np.ndarray:
    """Same as cache_dmd_blazegaze.quality_features (normalized landmarks)."""
    xy = points[:, :2]
    left = xy[[362, 263]].mean(axis=0)
    right = xy[[33, 133]].mean(axis=0)
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 0.0)
    inside = float(np.mean(np.all((xy >= 0.0) & (xy <= 1.0), axis=1)))
    return np.asarray(
        (float(np.linalg.norm(left - right)), float(span[0] * span[1]), inside), np.float32
    )


class GazeBranch:
    """BlazeGaze features at gaze_fps, adapter run causally over the last N samples."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.device = device
        self.blazegaze = BlazeGazePyTorch().to(device).eval()
        self.blazegaze.load_keras_archive(args.keras_weights)
        self.adapter: BlazeGazeDmdAdapter | None = None
        if args.adapter.exists():  # optional: not distributed with the repo
            checkpoint = torch.load(args.adapter, map_location=device, weights_only=False)
            self.adapter = BlazeGazeDmdAdapter(num_zones=len(GAZE_ZONES)).to(device).eval()
            self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        self.period = 1.0 / args.gaze_fps
        self.history: deque[tuple[np.ndarray, ...]] = deque(maxlen=args.gaze_window)
        self.last_t = -np.inf
        self.face_width_cm: float | None = None
        self.zone: int | None = None
        self.confidence = 0.0
        self.calibrator: GazeCalibrator | None = None
        self.votes: deque[int] = deque(maxlen=5)

    def load_profile(self, path: Path) -> None:
        """Fit the per-driver calibrator from a profile saved by calibrate_gaze.py."""
        profile = load_profile(path)
        calibrator = GazeCalibrator(self.blazegaze.gaze_mlp)
        calibrator.fit(profile["embedding"], profile["head_vector"],
                       profile["face_origin_3d"], profile["zone_id"])
        self.calibrator = calibrator

    def reset(self) -> None:
        self.history.clear()
        self.votes.clear()
        self.zone = None

    @torch.no_grad()
    def extract(
        self, frame_rgb: np.ndarray, pts_norm: np.ndarray, pts_px: np.ndarray, rt: np.ndarray
    ) -> dict[str, np.ndarray]:
        """BlazeGaze features for one frame (shared by demo and calibration)."""
        height, width = frame_rgb.shape[:2]
        patch = obtain_eyepatch(frame_rgb, pts_px)
        origin, self.face_width_cm = compute_face_origin_3d(
            pts_norm, rt, width=width, height=height, face_width_cm=self.face_width_cm
        )
        head = get_head_vector(rt)
        out = self.blazegaze(
            torch.from_numpy(patch[None]).float().div_(255.0).to(self.device),
            torch.from_numpy(head[None]).float().to(self.device),
            torch.from_numpy(origin[None]).float().to(self.device),
        )
        return {
            "embedding": out["embedding"][0].cpu().numpy(),
            "hidden": out["hidden"][0].cpu().numpy(),
            "pog": out["norm_pog"][0].cpu().numpy(),
            "head": head.astype(np.float32),
            "origin": origin.astype(np.float32),
            "geometry": geometry_from_landmarks(
                pts_norm[None], quality_features(pts_norm)[None])[0],
        }

    @torch.no_grad()
    def update(
        self, t: float, frame_rgb: np.ndarray, pts_norm: np.ndarray, pts_px: np.ndarray,
        rt: np.ndarray,
    ) -> int | None:
        if t - self.last_t < self.period:
            return self.zone
        self.last_t = t
        features = self.extract(frame_rgb, pts_norm, pts_px, rt)
        if self.calibrator is not None:
            _, zone, confidence = self.calibrator.predict(
                features["embedding"][None], features["head"][None], features["origin"][None])
            self.votes.append(int(zone[0]))  # causal majority of the last 5 samples
            self.zone = max(set(self.votes), key=list(self.votes).count)
            self.confidence = float(confidence[0])
            return self.zone
        self.history.append(tuple(
            features[k] for k in ("embedding", "hidden", "pog", "head", "origin", "geometry")))
        seq = [
            torch.from_numpy(np.stack([h[i] for h in self.history]))[None].float().to(self.device)
            for i in range(6)
        ]
        probs = torch.softmax(self.adapter(*seq)["zone_logits"][0, -1], dim=-1)
        self.confidence, zone = (float(v) for v in probs.max(dim=-1))
        self.zone = int(zone)
        return self.zone


def blendshape_scores(result: object) -> dict[str, float]:
    if not getattr(result, "face_blendshapes", None):
        return {}
    return {c.category_name: float(c.score) for c in result.face_blendshapes[0]}


def draw_overlay(
    frame: np.ndarray, out: object, signals: FrameSignals, crops: np.ndarray | None,
    zone: int | None, zone_conf: float, fps: float, pts_px: np.ndarray | None,
) -> None:
    h, w = frame.shape[:2]
    color = STATE_COLORS.get(out.state, (255, 255, 255))
    if pts_px is not None:
        for idx in (33, 133, 159, 145, 362, 263, 386, 374, 13, 14, 61, 291):
            cv2.circle(frame, tuple(int(v) for v in pts_px[idx, :2]), 2, (0, 255, 255), -1)

    panel = frame[0:230, 0:360]
    cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0, panel)
    cv2.putText(frame, out.state.upper(), (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    lines = [
        f"EAR {signals.ear or 0:.3f} thr {out.ear_threshold:.3f} "
        f"blink {signals.blink_score or 0:.2f} cnn "
        f"{'-' if signals.cnn_prob is None else format(signals.cnn_prob, '.2f')} "
        f"{'CLOSED' if out.eye_closed else 'open'}",
        f"PERCLOS(30s) {out.perclos:5.1%}  closure {out.closure_seconds:.1f}s",
        f"longest(60s) {out.longest_closure:.1f}s  blinks/min {out.blinks_per_min:.0f}",
        f"jaw {signals.jaw_open or 0:.2f}  yawn {out.yawning_seconds:.1f}s  "
        f"yawns(60s) {out.yawns_recent}",
        f"gaze {GAZE_ZONES[zone] if zone is not None else '-'} ({zone_conf:.2f})  "
        f"off-road {out.off_road_ratio:.0%}",
        f"yaw {signals.head_yaw or 0:+.0f}deg   {fps:4.1f} FPS",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 75 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if out.reasons:
        cv2.putText(frame, " | ".join(out.reasons), (12, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if crops is not None:
        scale = 6
        for i, crop in enumerate(crops):
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            x0 = w - (2 - i) * (16 * scale + 8)
            frame[8:8 + 16 * scale, x0:x0 + 16 * scale] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "16x16 L | R(mirrored)", (w - 2 * (16 * scale + 8), 16 * scale + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main() -> None:
    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_webcam = args.source.isdigit()
    capture = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not capture.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0

    gaze = None if args.no_gaze else GazeBranch(args, device)
    if gaze is not None and args.gaze_profile is not None:
        gaze.load_profile(args.gaze_profile)
        print(f"gaze calibrated from {args.gaze_profile}", flush=True)
    if gaze is not None and gaze.calibrator is None and gaze.adapter is None:
        raise SystemExit(
            "gaze needs a calibration profile (run scripts/calibrate_gaze.py, then pass "
            "--gaze-profile) or an adapter checkpoint; use --no-gaze to skip gaze")
    eye_net, eye_threshold = None, 0.0
    if args.eye_model.exists():
        checkpoint = torch.load(args.eye_model, map_location="cpu", weights_only=False)
        eye_net = EyeBlinkCNN(size=checkpoint["size"]).eval()
        eye_net.load_state_dict(checkpoint["state_dict"])
        eye_threshold = float(checkpoint["threshold"])  # logit threshold from validation
        print(f"eye CNN loaded, logit threshold {eye_threshold:.3f}", flush=True)
    elif args.eye_signal == "cnn":
        raise SystemExit(f"--eye-signal cnn needs {args.eye_model}")
    monitor = DriverMonitor(MonitorConfig())
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    hand_detector = None
    hands_config = HandsConfig(driver_left_on_image_left=args.driver_left_on_image_left)
    hands_monitor = HandsMonitor(hands_config)
    calibrator = WheelCalibrator(radius_range=tuple(args.wheel_radius_range or (0.28, 0.40)))
    wheel: Wheel | None = None
    clicks: list[tuple[int, int]] = []
    if args.hands:
        if not args.hand_task.exists():
            raise SystemExit(f"--hands needs {args.hand_task} (MediaPipe hand_landmarker.task)")
        hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(args.hand_task)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        ))
        if not args.no_display:
            cv2.namedWindow(WINDOW)
            cv2.setMouseCallback(
                WINDOW, lambda event, x, y, *_: clicks.append((x, y))
                if event == cv2.EVENT_LBUTTONDOWN else None,
            )

    writer = None
    csv_file = open(args.log_csv, "w", newline="", encoding="utf-8") if args.log_csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["t", "face", "ear", "blink", "cnn", "jaw", "yaw", "zone", "closed",
                             "perclos", "closure_s", "yawn_s", "off_road", "state"]
                            + (["hands_state", "hands_off_s"] if args.hands else []))
    saved_crops: list[np.ndarray] = []
    saved_t: list[float] = []

    frame_index = 0
    t_start = time.perf_counter()
    last_log = -1.0
    fps_times: deque[float] = deque(maxlen=30)
    paused = False
    state_counts: dict[str, int] = {}

    with vision.FaceLandmarker.create_from_options(options) as detector:
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q"):
                    break
                continue
            ok, frame = capture.read()
            if not ok or (args.max_frames and frame_index >= args.max_frames):
                break
            if frame.shape[1] > args.max_width:
                s = args.max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            t = (time.perf_counter() - t_start) if is_webcam else frame_index / native_fps
            frame_index += 1
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect_for_video(image, int(t * 1000))

            hands, hands_out = [], None
            if hand_detector is not None:
                if frame_index == 1 and args.wheel_radius_range:
                    calibrator.start(t)  # auto-fit once at start-up
                fitted = calibrator.push(t, frame)
                if len(clicks) >= 2:
                    fitted, clicks[:] = Wheel.from_points(clicks[0], clicks[1]), []
                if fitted is not None:
                    wheel = fitted
                    hands_monitor.reset()
                hands = hands_from_result(hand_detector.detect_for_video(image, int(t * 1000)),
                                          w, h)
                hands_out = hands_monitor.update(t, hands, wheel)

            pts_px = None
            crops = None
            zone = None
            if result.face_landmarks and len(result.face_landmarks[0]) == 478:
                pts_norm = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]],
                                    dtype=np.float32)
                pts_px = pts_norm[:, :2] * np.array([w, h], dtype=np.float32)
                rt = np.array(result.facial_transformation_matrixes[0], np.float32).reshape(4, 4)
                bs = blendshape_scores(result)
                _, yaw, _ = rotation_matrix_to_euler_angles(rt[:3, :3])
                try:
                    crops = eye_crops(frame, pts_px)
                except (ValueError, cv2.error):
                    crops = None
                if gaze is not None:
                    try:
                        zone = gaze.update(t, frame_rgb, pts_norm, pts_px, rt)
                    except (ValueError, cv2.error):
                        zone = gaze.zone
                ear = 0.5 * (compute_ear(pts_px, "left") + compute_ear(pts_px, "right"))
                mar = float(np.linalg.norm(pts_px[13] - pts_px[14])
                            / max(np.linalg.norm(pts_px[61] - pts_px[291]), 1e-5))
                cnn_prob, cnn_closed = None, None
                if eye_net is not None and crops is not None:
                    with torch.no_grad():
                        logit = float(eye_net(torch.from_numpy(crops[None]).float().div_(255.0)))
                    cnn_prob = 1.0 / (1.0 + float(np.exp(-logit)))
                    cnn_closed = logit >= eye_threshold
                signals = FrameSignals(
                    face=True,
                    cnn_prob=cnn_prob,
                    closed_override=cnn_closed if args.eye_signal == "cnn" else None,
                    ear=ear,
                    blink_score=0.5 * (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0))
                    if bs else None,
                    jaw_open=bs.get("jawOpen") if bs else None,
                    mar=mar,
                    head_yaw=yaw,
                    off_road=(ZONE_TO_ATTENTION[zone] == 2) if zone is not None else None,
                )
            else:
                signals = FrameSignals(face=False)
                if gaze is not None and monitor.last_face_t is not None and (
                    t - monitor.last_face_t > monitor.config.no_face_reset
                ):
                    gaze.reset()

            out = monitor.update(t, signals)
            state_counts[out.state] = state_counts.get(out.state, 0) + 1
            fps_times.append(time.perf_counter())
            fps = (len(fps_times) - 1) / max(fps_times[-1] - fps_times[0], 1e-6)

            if crops is not None and args.save_crops:
                saved_crops.append(crops)
                saved_t.append(t)
            if csv_writer:
                csv_writer.writerow([
                    f"{t:.3f}", int(signals.face), f"{signals.ear or 0:.4f}",
                    f"{signals.blink_score or 0:.3f}", f"{signals.cnn_prob or 0:.3f}",
                    f"{signals.jaw_open or 0:.3f}",
                    f"{signals.head_yaw or 0:.1f}", GAZE_ZONES[zone] if zone is not None else "",
                    int(out.eye_closed), f"{out.perclos:.3f}", f"{out.closure_seconds:.2f}",
                    f"{out.yawning_seconds:.2f}", f"{out.off_road_ratio:.2f}", out.state,
                ] + ([hands_out.state, f"{hands_out.hands_off_seconds:.2f}"] if hands_out else []))
            if t - last_log >= 1.0:
                last_log = t
                print(
                    f"t={t:6.1f}s state={out.state:10s} ear={signals.ear or 0:.3f}"
                    f"/thr={out.ear_threshold:.3f} blink={signals.blink_score or 0:.2f} "
                    f"cnn={signals.cnn_prob or 0:.2f} "
                    f"perclos={out.perclos:.2f} longest={out.longest_closure:.1f}s "
                    f"jaw={signals.jaw_open or 0:.2f} yawns={out.yawns_recent} "
                    f"zone={GAZE_ZONES[zone] if zone is not None else '-'} "
                    f"offroad={out.off_road_ratio:.2f} fps={fps:.1f}"
                    + (f" hands={hands_out.state}" if hands_out else ""),
                    flush=True,
                )

            draw_overlay(frame, out, signals, crops, zone,
                         gaze.confidence if gaze else 0.0, fps, pts_px)
            if hands_out is not None:
                draw_hands(frame, wheel, hands, hands_out, hands_config,
                           calibrator.progress(t) if calibrator.active else None)
                if clicks:
                    cv2.circle(frame, clicks[0], 6, (0, 255, 255), -1)
            if args.save:
                if writer is None:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                             native_fps, (w, h))
                writer.write(frame)
            if not args.no_display:
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = True
                if key == ord("r"):
                    monitor.reset()
                    hands_monitor.reset()
                    if gaze:
                        gaze.reset()
                if key == ord("w") and hand_detector is not None:
                    wheel = None
                    calibrator.start(t)

    capture.release()
    if hand_detector is not None:
        hand_detector.close()
        events = hands_monitor.finish()
        print(f"hands-off events: {len(events)}", flush=True)
        if args.hands_events:
            args.hands_events.parent.mkdir(parents=True, exist_ok=True)
            args.hands_events.write_text(json.dumps(
                [{"label": e.label, "start_s": e.start_s, "end_s": e.end_s,
                  "confirmed_s": e.confirmed_s} for e in events], indent=2))
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
    if args.save_crops and saved_crops:
        args.save_crops.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_crops, crops=np.stack(saved_crops),
                            timestamps_s=np.asarray(saved_t, np.float32))
    cv2.destroyAllWindows()
    total = max(sum(state_counts.values()), 1)
    print("summary frames=%d %s" % (total, " ".join(
        f"{k}={v / total:.1%}" for k, v in sorted(state_counts.items()))), flush=True)


if __name__ == "__main__":
    main()
