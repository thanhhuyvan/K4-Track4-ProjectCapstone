"""Extract MediaPipe 478 landmarks, 3D pose, and BlazeGaze features on DMD gaze sessions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from adaptive_dms.data import (
    GAZE_ZONE_CANONICAL_POG,
    GAZE_ZONES,
    DmdGazeSession,
    load_gaze_session,
)
from adaptive_dms.dms import (
    BlazeGazePyTorch,
    compute_ear,
    compute_face_origin_3d,
    geometry_from_landmarks,
    get_head_vector,
    obtain_eyepatch,
)

LEFT_EYE = (362, 263)
RIGHT_EYE = (33, 133)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache MediaPipe + BlazeGaze features for DMD gaze sessions."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/dmd"),
        help="Root directory containing DMD sessions (*_ann_gaze.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cache_dmd_blazegaze"),
        help="Output directory for cached .npz files",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=Path(
            "weights/face_landmarker_v2_with_blendshapes.task"
        ),
        help="Path to MediaPipe face_landmarker task model",
    )
    parser.add_argument(
        "--keras-weights",
        type=Path,
        default=Path(
            "weights/blazegaze_mpiifacegaze.keras"
        ),
        help="Path to pretrained blazegaze_mpiifacegaze.keras",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Sampling FPS for extraction (default: 10.0 FPS)",
    )
    parser.add_argument(
        "--gaze-valid-only",
        action="store_true",
        help="Only sample frames up to the last valid gaze_zone annotation (plus small tail)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_indices(total_frames: int, native_fps: float, target_fps: float) -> np.ndarray:
    step = max(native_fps / target_fps, 1.0)
    indices = np.rint(np.arange(0.0, total_frames, step)).astype(np.int32)
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


def camera_matrix_from_session(
    session: DmdGazeSession, width: int, height: int
) -> np.ndarray:
    mat_3x4 = session.camera_intrinsics.get("camera_matrix_3x4")
    if isinstance(mat_3x4, list) and len(mat_3x4) == 12:
        mat = np.array(mat_3x4, dtype=np.float32).reshape(3, 4)
        return mat[:, :3].copy()
    focal = float(width)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def cache_gaze_session(
    session: DmdGazeSession,
    detector: vision.FaceLandmarker,
    blazegaze: BlazeGazePyTorch,
    device: torch.device,
    output_path: Path,
    target_fps: float,
    gaze_valid_only: bool,
) -> dict[str, int | float]:
    t0 = time.perf_counter()
    capture = cv2.VideoCapture(str(session.face_video_path))
    native_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 29.76
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = min(session.total_frames, video_frames) if video_frames > 0 else session.total_frames

    if gaze_valid_only:
        valid_zone_frames = [
            i for i, z in enumerate(session.gaze_zone[:total_frames]) if z != "not_valid"
        ]
        if valid_zone_frames:
            total_frames = min(total_frames, max(valid_zone_frames) + int(native_fps * 5))

    indices = sample_indices(total_frames, native_fps, target_fps)
    count = len(indices)
    intrinsics = camera_matrix_from_session(session, width, height)

    landmarks = np.zeros((count, 478, 3), dtype=np.float32)
    face_rts = np.zeros((count, 4, 4), dtype=np.float32)
    head_vectors = np.zeros((count, 3), dtype=np.float32)
    face_origins_3d = np.zeros((count, 3), dtype=np.float32)
    ears = np.zeros((count, 2), dtype=np.float32)
    quality = np.zeros((count, 3), dtype=np.float32)
    valid = np.zeros(count, dtype=np.bool_)
    eye_patches = np.zeros((count, 128, 512, 3), dtype=np.uint8)

    face_width_cm: float | None = None
    sample_pos = 0
    frame_idx = 0

    while sample_pos < count:
        grabbed = capture.grab()
        if not grabbed:
            break
        if frame_idx != int(indices[sample_pos]):
            frame_idx += 1
            continue
        decoded, frame_bgr = capture.retrieve()
        if decoded and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = detector.detect(mp_img)
            if res.face_landmarks and res.facial_transformation_matrixes:
                lm_proto = res.face_landmarks[0]
                if len(lm_proto) == 478:
                    pts_norm = np.array(
                        [(p.x, p.y, p.z) for p in lm_proto], dtype=np.float32
                    )
                    pts_px = pts_norm[:, :2] * np.array([width, height], dtype=np.float32)
                    rt = np.array(
                        res.facial_transformation_matrixes[0], dtype=np.float32
                    ).reshape(4, 4)
                    try:
                        patch = obtain_eyepatch(frame_rgb, pts_px)
                        origin_3d, face_width_cm = compute_face_origin_3d(
                            pts_norm,
                            rt,
                            width=width,
                            height=height,
                            face_width_cm=face_width_cm,
                            intrinsics=intrinsics,
                        )
                        h_vec = get_head_vector(rt)
                        landmarks[sample_pos] = pts_norm
                        face_rts[sample_pos] = rt
                        head_vectors[sample_pos] = h_vec
                        face_origins_3d[sample_pos] = origin_3d
                        ears[sample_pos] = (
                            compute_ear(pts_px, "left"),
                            compute_ear(pts_px, "right"),
                        )
                        quality[sample_pos] = quality_features(pts_norm)
                        eye_patches[sample_pos] = patch
                        valid[sample_pos] = True
                    except (ValueError, cv2.error):
                        pass
        sample_pos += 1
        frame_idx += 1
        if sample_pos % 100 == 0 or sample_pos == count:
            print(
                f"progress={session.group}/{session.subject}/{session.session_id} "
                f"samples={sample_pos}/{count} valid={int(valid.sum())}",
                flush=True,
            )
    capture.release()

    actual_count = sample_pos
    indices = indices[:actual_count]
    landmarks = landmarks[:actual_count]
    face_rts = face_rts[:actual_count]
    head_vectors = head_vectors[:actual_count]
    face_origins_3d = face_origins_3d[:actual_count]
    ears = ears[:actual_count]
    quality = quality[:actual_count]
    valid = valid[:actual_count]
    eye_patches = eye_patches[:actual_count]

    # Run BlazeGaze in fast batches over all extracted samples
    blazegaze_emb = np.zeros((actual_count, 512), dtype=np.float32)
    blazegaze_hidden = np.zeros((actual_count, 16), dtype=np.float32)
    blazegaze_pog = np.zeros((actual_count, 2), dtype=np.float32)

    valid_idx = np.flatnonzero(valid)
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(valid_idx), batch_size):
            sel = valid_idx[start : start + batch_size]
            patches_t = (
                torch.from_numpy(eye_patches[sel]).float().div_(255.0).to(device)
            )
            head_t = torch.from_numpy(head_vectors[sel]).float().to(device)
            origin_t = torch.from_numpy(face_origins_3d[sel]).float().to(device)
            out = blazegaze(patches_t, head_t, origin_t)
            blazegaze_emb[sel] = out["embedding"].cpu().numpy()
            blazegaze_hidden[sel] = out["hidden"].cpu().numpy()
            blazegaze_pog[sel] = out["norm_pog"].cpu().numpy()

    geometry = geometry_from_landmarks(landmarks, quality)
    gaze_zone_id = np.array(
        [session.gaze_zone_id[i] for i in indices], dtype=np.int64
    )
    gaze_zone_str = [session.gaze_zone[i] for i in indices]
    canonical_pog = np.array(
        [GAZE_ZONE_CANONICAL_POG.get(z, (0.0, 0.0)) for z in gaze_zone_str],
        dtype=np.float32,
    )
    blinking = np.array([session.blinking[i] for i in indices], dtype=np.bool_)
    hands = np.array([session.hands_on_wheel[i] for i in indices])

    metadata = {
        "group": session.group,
        "subject": session.subject,
        "session_id": session.session_id,
        "native_fps": native_fps,
        "target_fps": target_fps,
        "width": width,
        "height": height,
        "face_width_cm": face_width_cm,
        "gaze_zones": list(GAZE_ZONES),
        "canonical_pog": GAZE_ZONE_CANONICAL_POG,
        "subject_metadata": session.metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        frame_indices=indices,
        timestamps_s=(indices / native_fps).astype(np.float32),
        landmarks=landmarks.astype(np.float16),
        face_rt=face_rts,
        head_vector=head_vectors,
        face_origin_3d=face_origins_3d,
        ears=ears,
        quality=quality,
        geometry=geometry,
        valid=valid,
        blazegaze_emb=blazegaze_emb,
        blazegaze_hidden=blazegaze_hidden,
        blazegaze_pog=blazegaze_pog,
        gaze_zone_id=gaze_zone_id,
        canonical_pog=canonical_pog,
        blinking=blinking,
        hands_on_wheel=hands,
        metadata=np.asarray(json.dumps(metadata)),
    )
    tmp_path.replace(output_path)
    elapsed = time.perf_counter() - t0
    return {
        "total_samples": actual_count,
        "valid_samples": int(valid.sum()),
        "valid_gaze_zone_samples": int((valid & (gaze_zone_id >= 0)).sum()),
        "elapsed_s": round(elapsed, 2),
    }


def main() -> None:
    args = arguments()
    annotations = sorted(args.root.rglob("*_ann_gaze.json"))
    if not annotations:
        raise SystemExit(f"No *_ann_gaze.json files found under {args.root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blazegaze = BlazeGazePyTorch().to(device).eval()
    blazegaze.load_keras_archive(args.keras_weights)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.task)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for ann in annotations:
            session = load_gaze_session(ann)
            out_file = (
                args.output
                / session.group
                / f"{session.subject}_{session.session_id}.npz"
            )
            if out_file.exists() and not args.overwrite:
                print(f"skip={out_file}", flush=True)
                continue
            stats = cache_gaze_session(
                session=session,
                detector=detector,
                blazegaze=blazegaze,
                device=device,
                output_path=out_file,
                target_fps=args.fps,
                gaze_valid_only=args.gaze_valid_only,
            )
            print(
                f"cached={session.group}/{session.subject}/{session.session_id} "
                f"stats={json.dumps(stats)} output={out_file}",
                flush=True,
            )


if __name__ == "__main__":
    main()

