"""Offline check of per-driver gaze calibration on the cached DMD gaze sessions.

For every driver: K support frames per zone come from the early part of the
session; evaluation uses later frames of the same driver (buffer dropped).
  spread = support frames spread over the whole early part (best case)
  burst  = K consecutive frames from the start of each zone (what a guided
           3-second calibration in the demo actually produces)
The zero-shot BlazeGaze head is the baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_eval_dmd_blazegaze as tev
from adaptive_dms.data import GAZE_ZONES
from adaptive_dms.dms import BlazeGazePyTorch
from adaptive_dms.gaze_calibration import PROTOTYPES, GazeCalibrator

KEYS = ("zone_9_accuracy", "zone_9_macro_f1", "attention_3_macro_f1",
        "distracted_binary_macro_f1", "pog_norm_l2_error")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache_dmd_blazegaze"))
    parser.add_argument("--keras-weights", type=Path,
                        default=Path("weights/blazegaze_mpiifacegaze.keras"))
    parser.add_argument("--shots", default="1,3,5,10,20,30")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs/gaze_calibration.json"))
    return parser.parse_args()


def pick_support(train_idx: np.ndarray, zone: np.ndarray, k: int, mode: str) -> np.ndarray:
    chosen = []
    for z in range(len(GAZE_ZONES)):
        idx = train_idx[zone[train_idx] == z]
        if len(idx) == 0:
            continue
        if mode == "spread":
            chosen.extend(idx[np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)])
        else:
            chosen.extend(idx[:k])
    return np.asarray(chosen, dtype=np.int64)


def main() -> None:
    args = arguments()
    torch.manual_seed(0)
    mlp = BlazeGazePyTorch().eval()
    mlp.load_keras_archive(args.keras_weights)
    shots = [int(s) for s in args.shots.split(",")]
    files = sorted(args.cache.rglob("*.npz"))
    results: dict[str, dict[str, list[dict[str, float]]]] = {"zeroshot": {"-": []}}
    for mode in ("spread", "burst"):
        results[mode] = {str(k): [] for k in shots}

    for file in files:
        name = f"{file.parent.name}/{file.stem.split('_')[0]}"
        data = tev.load_npz_records(file)
        train_mask, val_mask = tev.temporal_zone_split(
            data["valid"], data["gaze_zone_id"], data["blinking"])
        train_idx, val_idx = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
        zone = data["gaze_zone_id"]

        def evaluate(pog: np.ndarray, pred_zone: np.ndarray) -> dict[str, float]:
            m = tev.evaluate_predictions(zone[val_idx], pred_zone,
                                         data["canonical_pog"][val_idx], pog)
            return {k: m[k] for k in KEYS}

        pog0 = data["blazegaze_pog"][val_idx]
        results["zeroshot"]["-"].append(evaluate(pog0, tev.nearest_zone_from_pog(pog0, PROTOTYPES)))
        for mode in ("spread", "burst"):
            for k in shots:
                support = pick_support(train_idx, zone, k, mode)
                calibrator = GazeCalibrator(mlp.gaze_mlp, steps=args.steps)
                calibrator.fit(data["blazegaze_emb"][support], data["head_vector"][support],
                               data["face_origin_3d"][support], zone[support])
                pog, pred, _ = calibrator.predict(
                    data["blazegaze_emb"][val_idx], data["head_vector"][val_idx],
                    data["face_origin_3d"][val_idx])
                results[mode][str(k)].append(evaluate(pog, pred))
        print(f"done {name}: train {len(train_idx)} eval {len(val_idx)} frames", flush=True)

    def mean(rows: list[dict[str, float]]) -> dict[str, float]:
        return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in KEYS}

    summary = {mode: {k: mean(rows) for k, rows in per.items()} for mode, per in results.items()}
    header = f"{'setting':16s} {'zone9_acc':>9s} {'zone9_F1':>9s} {'att3_F1':>8s} {'dist_F1':>8s} {'PoG_L2':>7s}"
    print("\nmean over drivers\n" + header)
    for mode, per in summary.items():
        for k, m in per.items():
            label = "zero-shot" if mode == "zeroshot" else f"{mode} K={k}"
            print(f"{label:16s} {m['zone_9_accuracy']:9.3f} {m['zone_9_macro_f1']:9.3f} "
                  f"{m['attention_3_macro_f1']:8.3f} {m['distracted_binary_macro_f1']:8.3f} "
                  f"{m['pog_norm_l2_error']:7.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"mean": summary, "per_driver": results}, indent=2),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
