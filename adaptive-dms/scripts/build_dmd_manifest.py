"""Build a face-frame manifest from extracted DMD drowsiness sessions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from adaptive_dms.data import load_drowsiness_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = sorted(args.root.rglob("*_ann_drowsiness.json"))
    if not annotations:
        raise SystemExit(f"No DMD drowsiness annotations found under {args.root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "group",
            "subject",
            "frame_idx",
            "eye_state",
            "blinking",
            "yawning",
            "face_video",
            "annotation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for annotation in annotations:
            session = load_drowsiness_session(annotation)
            if not session.face_video_path.is_file():
                raise FileNotFoundError(session.face_video_path)
            for frame_idx in range(session.total_frames):
                writer.writerow(
                    {
                        "group": session.group,
                        "subject": session.subject,
                        "frame_idx": frame_idx,
                        "eye_state": session.eye_state[frame_idx],
                        "blinking": int(session.blinking[frame_idx]),
                        "yawning": session.yawning[frame_idx],
                        "face_video": session.face_video_path,
                        "annotation": session.annotation_path,
                    }
                )
            summaries.append(
                {
                    "group": session.group,
                    "subject": session.subject,
                    "frames": session.total_frames,
                    "eye_state": dict(Counter(session.eye_state)),
                    "blinking_frames": sum(session.blinking),
                    "yawning": dict(Counter(session.yawning)),
                    "metadata": session.metadata,
                    "context": session.context,
                }
            )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"sessions={len(summaries)} rows={sum(x['frames'] for x in summaries)}")
    print(f"manifest={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
