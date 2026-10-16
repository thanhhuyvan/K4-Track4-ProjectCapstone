"""One timeline for the whole DMS: face/eyes/gaze state and hands state from demo_realtime.py.

    python scripts/demo_realtime.py --hands --log-csv artifacts/run.csv ...
    python scripts/plot_timeline.py artifacts/run.csv --out artifacts/run_timeline.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

COLORS = {  # BGR, same palette as the demo overlay
    "alert": (80, 200, 80),
    "both_hands": (80, 200, 80),
    "only_left": (120, 220, 200),
    "only_right": (120, 220, 200),
    "distracted": (0, 200, 255),
    "yawning": (0, 170, 255),
    "drowsy": (0, 100, 255),
    "microsleep": (0, 0, 255),
    "hands_off": (0, 0, 255),
    "no_face": (160, 160, 160),
    "no_wheel": (160, 160, 160),
}
ROW_H, LABEL_W, WIDTH = 46, 150, 1400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with args.log_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    tracks = [("driver", [r["state"] for r in rows])]
    if rows and "hands_state" in rows[0]:
        tracks.append(("hands", [r["hands_state"] for r in rows]))

    span = max(t[-1] - t[0], 1e-6)
    x = (LABEL_W + (t - t[0]) / span * (WIDTH - LABEL_W - 10)).astype(int)
    legend = sorted({s for _, states in tracks for s in states})
    height = ROW_H * len(tracks) + 70 + 22 * ((len(legend) + 5) // 6)
    img = np.full((height, WIDTH, 3), 255, np.uint8)
    for row, (name, states) in enumerate(tracks):
        y0 = 10 + row * ROW_H
        cv2.putText(img, name, (10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
        for i in range(len(states) - 1):
            cv2.rectangle(img, (x[i], y0), (max(x[i + 1], x[i] + 1), y0 + ROW_H - 8),
                          COLORS.get(states[i], (200, 200, 200)), -1)
    y_axis = 10 + len(tracks) * ROW_H + 18
    for sec in np.arange(0.0, span + 1e-6, max(1.0, round(span / 10))):
        xs = int(LABEL_W + sec / span * (WIDTH - LABEL_W - 10))
        cv2.putText(img, f"{sec:.0f}s", (xs - 10, y_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (90, 90, 90), 1)
    for i, state in enumerate(legend):
        lx, ly = 10 + (i % 6) * 220, y_axis + 25 + 22 * (i // 6)
        cv2.rectangle(img, (lx, ly - 12), (lx + 16, ly + 2), COLORS.get(state, (200, 200, 200)), -1)
        cv2.putText(img, state, (lx + 22, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)

    out = args.out or args.log_csv.with_name(args.log_csv.stem + "_timeline.png")
    cv2.imwrite(str(out), img)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
