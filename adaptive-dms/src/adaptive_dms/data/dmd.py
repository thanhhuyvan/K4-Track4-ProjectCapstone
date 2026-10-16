"""Loaders for DMD (Driver Monitoring Dataset) OpenLABEL annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GAZE_ZONES: tuple[str, ...] = (
    "left_mirror",
    "left",
    "front",
    "center_mirror",
    "front_right",
    "right_mirror",
    "right",
    "infotainment",
    "steering_wheel",
)

GAZE_ZONE_TO_ID: dict[str, int] = {zone: idx for idx, zone in enumerate(GAZE_ZONES)}

# Canonical 2D coordinates in normalized driver-perspective space [-0.5, 0.5]^2
# (+X = right, +Y = down, (0, 0) = straight ahead on road 'front')
GAZE_ZONE_CANONICAL_POG: dict[str, tuple[float, float]] = {
    "left_mirror": (-0.38, 0.05),
    "left": (-0.48, 0.00),
    "front": (0.00, 0.00),
    "center_mirror": (0.20, -0.25),
    "front_right": (0.25, 0.00),
    "right_mirror": (0.42, 0.05),
    "right": (0.48, 0.00),
    "infotainment": (0.22, 0.25),
    "steering_wheel": (0.00, 0.30),
}


@dataclass(frozen=True)
class DmdSession:
    group: str
    subject: str
    total_frames: int
    face_video_path: Path
    annotation_path: Path
    eye_state: tuple[str, ...]
    blinking: tuple[bool, ...]
    yawning: tuple[str, ...]
    metadata: dict[str, Any]
    context: dict[str, Any]


@dataclass(frozen=True)
class DmdGazeSession:
    group: str
    subject: str
    session_id: str
    total_frames: int
    face_video_path: Path
    annotation_path: Path
    gaze_zone: tuple[str, ...]
    gaze_zone_id: tuple[int, ...]
    blinking: tuple[bool, ...]
    hands_on_wheel: tuple[str, ...]
    camera_intrinsics: dict[str, Any]
    metadata: dict[str, Any]
    context: dict[str, Any]


def _extract_kv(entries: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for items in entries.values():
        if isinstance(items, list):
            for item in items:
                if "name" in item and "val" in item:
                    result[str(item["name"])] = item["val"]
    return result


def _resolve_video_and_ids(
    annotation_path: Path, openlabel: dict[str, Any]
) -> tuple[str, str, str, int, Path, dict[str, Any]]:
    annotation_path = annotation_path.resolve()
    streams = openlabel.get("streams", {})
    face_stream = streams.get("face_camera", {})
    uri = str(face_stream.get("uri", ""))
    props = face_stream.get("stream_properties", {})
    total_frames = int(props.get("total_frames", 0))
    intrinsics = props.get("intrinsics_pinhole", {})

    uri_parts = Path(uri).parts
    if len(uri_parts) >= 3:
        group, subject, session_id = uri_parts[0], uri_parts[1], uri_parts[2]
    else:
        session_id = annotation_path.parent.name
        subject = annotation_path.parent.parent.name
        group = annotation_path.parent.parent.parent.name

    video_name = Path(uri).name if uri else ""
    candidate = annotation_path.parent / video_name if video_name else None
    if candidate is not None and candidate.exists():
        face_video_path = candidate.resolve()
    else:
        matches = sorted(annotation_path.parent.glob("*_rgb_face.mp4"))
        if matches:
            face_video_path = matches[0].resolve()
        elif candidate is not None:
            face_video_path = candidate.resolve()
        else:
            raise FileNotFoundError(
                f"Cannot find face video for annotation {annotation_path}"
            )

    return group, subject, session_id, total_frames, face_video_path, intrinsics


def load_drowsiness_session(annotation_path: Path | str) -> DmdSession:
    path = Path(annotation_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    openlabel = raw["openlabel"]

    group, subject, _, total_frames, face_video_path, _ = _resolve_video_and_ids(
        path, openlabel
    )

    eye_state = ["unknown"] * total_frames
    blinking = [False] * total_frames
    yawning = ["none"] * total_frames

    for action in openlabel.get("actions", {}).values():
        action_type = str(action.get("type", ""))
        for interval in action.get("frame_intervals", []):
            start = max(0, int(interval["frame_start"]))
            end = min(total_frames - 1, int(interval["frame_end"]))
            if end < start:
                continue
            if action_type.startswith("eyes_state/"):
                label = action_type.split("/", 1)[1].strip().lower()
                for idx in range(start, end + 1):
                    eye_state[idx] = label
            elif action_type == "blinks/blinking":
                for idx in range(start, end + 1):
                    blinking[idx] = True
            elif action_type.startswith("yawning/"):
                raw_yawn = action_type.split("/", 1)[1].strip().lower()
                norm_yawn = "_".join(raw_yawn.split())
                for idx in range(start, end + 1):
                    yawning[idx] = norm_yawn

    objects = openlabel.get("objects", {})
    obj_data = next(iter(objects.values()), {}).get("object_data", {}) if objects else {}
    contexts = openlabel.get("contexts", {})
    ctx_data = (
        next(iter(contexts.values()), {}).get("context_data", {}) if contexts else {}
    )

    return DmdSession(
        group=group,
        subject=subject,
        total_frames=total_frames,
        face_video_path=face_video_path,
        annotation_path=path,
        eye_state=tuple(eye_state),
        blinking=tuple(blinking),
        yawning=tuple(yawning),
        metadata=_extract_kv(obj_data),
        context=_extract_kv(ctx_data),
    )


def load_gaze_session(
    annotation_path: Path | str,
    hands_annotation_path: Path | str | None = None,
) -> DmdGazeSession:
    path = Path(annotation_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    openlabel = raw["openlabel"]

    group, subject, session_id, total_frames, face_video_path, intrinsics = (
        _resolve_video_and_ids(path, openlabel)
    )

    gaze_zone = ["not_valid"] * total_frames
    blinking = [False] * total_frames
    hands_on_wheel = ["unknown"] * total_frames

    for action in openlabel.get("actions", {}).values():
        action_type = str(action.get("type", ""))
        for interval in action.get("frame_intervals", []):
            start = max(0, int(interval["frame_start"]))
            end = min(total_frames - 1, int(interval["frame_end"]))
            if end < start:
                continue
            if action_type.startswith("gaze_zone/"):
                zone = action_type.split("/", 1)[1].strip().lower()
                for idx in range(start, end + 1):
                    gaze_zone[idx] = zone
            elif action_type == "blinks/blinking":
                for idx in range(start, end + 1):
                    blinking[idx] = True

    if hands_annotation_path is None:
        candidates = sorted(path.parent.glob("*_ann_hands.json"))
        if candidates:
            hands_annotation_path = candidates[0]

    if hands_annotation_path is not None and Path(hands_annotation_path).exists():
        hands_raw = json.loads(
            Path(hands_annotation_path).read_text(encoding="utf-8")
        )["openlabel"]
        for action in hands_raw.get("actions", {}).values():
            action_type = str(action.get("type", ""))
            if action_type.startswith("hands_on_wheel/"):
                label = action_type.split("/", 1)[1].strip().lower()
                for interval in action.get("frame_intervals", []):
                    start = max(0, int(interval["frame_start"]))
                    end = min(total_frames - 1, int(interval["frame_end"]))
                    for idx in range(start, end + 1):
                        hands_on_wheel[idx] = label

    gaze_zone_id = tuple(GAZE_ZONE_TO_ID.get(z, -100) for z in gaze_zone)

    objects = openlabel.get("objects", {})
    obj_data = next(iter(objects.values()), {}).get("object_data", {}) if objects else {}
    contexts = openlabel.get("contexts", {})
    ctx_data = (
        next(iter(contexts.values()), {}).get("context_data", {}) if contexts else {}
    )

    return DmdGazeSession(
        group=group,
        subject=subject,
        session_id=session_id,
        total_frames=total_frames,
        face_video_path=face_video_path,
        annotation_path=path,
        gaze_zone=tuple(gaze_zone),
        gaze_zone_id=gaze_zone_id,
        blinking=tuple(blinking),
        hands_on_wheel=tuple(hands_on_wheel),
        camera_intrinsics=intrinsics,
        metadata=_extract_kv(obj_data),
        context=_extract_kv(ctx_data),
    )

