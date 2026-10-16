"""Cache causal 5 FPS MediaPipe features and affine crops for DMD videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from adaptive_dms.data import DmdSession, load_drowsiness_session

LEFT_EYE = (362, 263)
RIGHT_EYE = (33, 133)
MOUTH = (61, 291)

EYE_CODES = {
    "unknown": 0,
    "open": 1,
    "close": 2,
    "opening": 3,
    "closing": 4,
    "undefined": 5,
}
YAWN_CODES = {
    "none": 0,
    "yawning_with_hand": 1,
    "yawning_without_hand": 2,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task",
        type=Path,
        default=Path("weights/face_landmarker_v2_with_blendshapes.task"),
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def xy(point: object) -> np.ndarray:
    if isinstance(point, tuple):
        return np.asarray(point, dtype=np.float32)
    return np.asarray((point.x, point.y), dtype=np.float32)  # type: ignore[attr-defined]


def robust_landmarks(
    detector: vision.FaceLandmarker, frame: np.ndarray
) -> list[object] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(mp.ImageFormat.SRGB, rgb))
    if result.face_landmarks:
        return result.face_landmarks[0]
    for angle, code in (
        (180, cv2.ROTATE_180),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ):
        result = detector.detect(mp.Image(mp.ImageFormat.SRGB, cv2.rotate(rgb, code)))
        if not result.face_landmarks:
            continue
        restored: list[object] = []
        for point in result.face_landmarks[0]:
            if angle == 180:
                restored.append((1.0 - point.x, 1.0 - point.y))
            elif angle == 90:
                restored.append((point.y, 1.0 - point.x))
            else:
                restored.append((1.0 - point.y, point.x))
        return restored
    return None


def affine_crop(
    frame: np.ndarray,
    landmarks: list[object],
    corners: tuple[int, int],
    output_size: tuple[int, int],
    width_fraction: float,
    clahe: cv2.CLAHE,
) -> np.ndarray:
    height, width = frame.shape[:2]
    first, second = (xy(landmarks[index]) * (width, height) for index in corners)
    center = (first + second) / 2.0
    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance < 2.0:
        raise ValueError("landmark span is too small")
    crop_width, crop_height = output_size
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    scale = crop_width * width_fraction / distance
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, scale)
    matrix[0, 2] += crop_width / 2.0 - center[0]
    matrix[1, 2] += crop_height / 2.0 - center[1]
    crop = cv2.warpAffine(
        frame,
        matrix,
        output_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return clahe.apply(cv2.medianBlur(gray, 3))


def affine_face_crop(
    frame: np.ndarray,
    landmarks: list[object],
    output_size: tuple[int, int],
    clahe: cv2.CLAHE,
) -> np.ndarray:
    """Align a compact full-face crop while retaining hand-over-mouth context."""
    height, width = frame.shape[:2]
    points = np.asarray([xy(point) for point in landmarks], dtype=np.float32)
    points *= np.asarray((width, height), dtype=np.float32)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    span = np.maximum(maximum - minimum, 2.0)
    left_eye = points[list(LEFT_EYE)].mean(axis=0)
    right_eye = points[list(RIGHT_EYE)].mean(axis=0)
    delta = left_eye - right_eye
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    crop_width, crop_height = output_size
    scale = min(crop_width * 0.82 / span[0], crop_height * 0.82 / span[1])
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, float(scale))
    matrix[0, 2] += crop_width / 2.0 - center[0]
    matrix[1, 2] += crop_height / 2.0 - center[1]
    crop = cv2.warpAffine(
        frame,
        matrix,
        output_size,
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return clahe.apply(cv2.medianBlur(gray, 3))


def sample_indices(total_frames: int, native_fps: float, target_fps: float) -> np.ndarray:
    if total_frames <= 0 or native_fps <= 0 or target_fps <= 0:
        raise ValueError("frame count and FPS values must be positive")
    indices = np.rint(
        np.arange(0.0, total_frames, native_fps / target_fps)
    ).astype(np.int32)
    return np.unique(np.clip(indices, 0, total_frames - 1))


def quality_features(points: np.ndarray) -> np.ndarray:
    xy_points = points[:, :2]
    left = xy_points[list(LEFT_EYE)].mean(axis=0)
    right = xy_points[list(RIGHT_EYE)].mean(axis=0)
    interocular = float(np.linalg.norm(left - right))
    span = np.maximum(xy_points.max(axis=0) - xy_points.min(axis=0), 0.0)
    bbox_area = float(span[0] * span[1])
    inside = float(np.mean(np.all((xy_points >= 0.0) & (xy_points <= 1.0), axis=1)))
    return np.asarray((interocular, bbox_area, inside), dtype=np.float32)


def cache_session(
    session: DmdSession,
    detector: vision.FaceLandmarker,
    output: Path,
    target_fps: float,
    max_samples: int,
) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(session.face_video_path))
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(session.total_frames, video_frames)
    indices = sample_indices(total_frames, native_fps, target_fps)
    if max_samples:
        indices = indices[:max_samples]

    count = len(indices)
    landmarks = np.zeros((count, 478, 3), dtype=np.float16)
    left_eye = np.zeros((count, 32, 64), dtype=np.uint8)
    right_eye = np.zeros((count, 32, 64), dtype=np.uint8)
    mouth = np.zeros((count, 48, 64), dtype=np.uint8)
    face = np.zeros((count, 96, 96), dtype=np.uint8)
    quality = np.zeros((count, 3), dtype=np.float32)
    valid = np.zeros(count, dtype=np.bool_)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))

    sample_position = 0
    frame_index = 0
    while sample_position < count:
        grabbed = capture.grab()
        if not grabbed:
            break
        if frame_index != int(indices[sample_position]):
            frame_index += 1
            continue
        decoded, frame = capture.retrieve()
        if decoded:
            detected = robust_landmarks(detector, frame)
            if detected is not None and len(detected) == 478:
                try:
                    points = np.asarray(
                        [
                            (
                                point[0],
                                point[1],
                                0.0,
                            )
                            if isinstance(point, tuple)
                            else (point.x, point.y, point.z)
                            for point in detected
                        ],
                        dtype=np.float32,
                    )
                    left_eye[sample_position] = affine_crop(
                        frame, detected, LEFT_EYE, (64, 32), 0.70, clahe
                    )
                    right_eye[sample_position] = affine_crop(
                        frame, detected, RIGHT_EYE, (64, 32), 0.70, clahe
                    )
                    mouth[sample_position] = affine_crop(
                        frame, detected, MOUTH, (64, 48), 0.72, clahe
                    )
                    face[sample_position] = affine_face_crop(
                        frame, detected, (96, 96), clahe
                    )
                    landmarks[sample_position] = points.astype(np.float16)
                    quality[sample_position] = quality_features(points)
                    valid[sample_position] = True
                except (ValueError, cv2.error):
                    pass
        sample_position += 1
        frame_index += 1
        if sample_position % 250 == 0 or sample_position == count:
            print(
                f"progress={session.group}/{session.subject} "
                f"samples={sample_position}/{count} valid={int(valid.sum())}",
                flush=True,
            )
    capture.release()

    frame_indices = indices[:sample_position]
    eye_codes = np.asarray(
        [EYE_CODES.get(session.eye_state[index], 0) for index in frame_indices],
        dtype=np.uint8,
    )
    yawn_codes = np.asarray(
        [YAWN_CODES.get(session.yawning[index], 0) for index in frame_indices],
        dtype=np.uint8,
    )
    blink = np.asarray(
        [session.blinking[index] for index in frame_indices], dtype=np.bool_
    )
    metadata = json.dumps(
        {
            "group": session.group,
            "subject": session.subject,
            "native_fps": native_fps,
            "target_fps": target_fps,
            "face_video": str(session.face_video_path),
            "annotation": str(session.annotation_path),
            "eye_codes": EYE_CODES,
            "yawn_codes": YAWN_CODES,
            "subject_metadata": session.metadata,
            "context": session.context,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        frame_indices=frame_indices,
        timestamps_s=(frame_indices / native_fps).astype(np.float32),
        landmarks=landmarks[:sample_position],
        left_eye=left_eye[:sample_position],
        right_eye=right_eye[:sample_position],
        mouth=mouth[:sample_position],
        face=face[:sample_position],
        quality=quality[:sample_position],
        valid=valid[:sample_position],
        eye_state=eye_codes,
        blinking=blink,
        yawning=yawn_codes,
        metadata=np.asarray(metadata),
    )
    temporary.replace(output)
    return int(valid[:sample_position].sum()), sample_position


def main() -> None:
    args = arguments()
    annotations = sorted(args.root.rglob("*_ann_drowsiness.json"))
    if args.max_sessions:
        annotations = annotations[: args.max_sessions]
    if not annotations:
        raise SystemExit(f"No annotations found under {args.root}")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.task)),
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for annotation in annotations:
            session = load_drowsiness_session(annotation)
            output = args.output / session.group / f"{session.subject}_s5.npz"
            if output.exists() and not args.overwrite:
                print(f"skip={output}", flush=True)
                continue
            valid, total = cache_session(
                session, detector, output, args.fps, args.max_samples
            )
            print(
                f"cached={session.group}/{session.subject} valid={valid}/{total} "
                f"output={output}",
                flush=True,
            )


if __name__ == "__main__":
    main()
