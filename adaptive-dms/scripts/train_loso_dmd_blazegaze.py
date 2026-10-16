"""Leave-one-subject-out evaluation of BlazeGaze DMD gaze-zone models.

For each fold: one subject is test (never seen in training or model selection),
one other subject is validation (checkpoint selection only), the rest train.
Windows never cross session boundaries.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import train_eval_dmd_blazegaze as tev
from adaptive_dms.data import GAZE_ZONE_CANONICAL_POG, GAZE_ZONES
from adaptive_dms.dms import BlazeGazeDmdAdapter, BlazeGazePyTorch

SUMMARY_KEYS = (
    "zone_9_accuracy",
    "zone_9_macro_f1",
    "attention_3_macro_f1",
    "distracted_binary_macro_f1",
    "pog_norm_l2_error",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache_dmd_blazegaze"))
    parser.add_argument(
        "--keras-weights", type=Path, default=Path("weights/blazegaze_mpiifacegaze.keras")
    )
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/loso_blazegaze"))
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--few-shot-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--folds", type=str, default="", help="Comma-separated test subjects; default all"
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_sessions(cache: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return {subject_id: arrays}; subject id is '<group>/<subject>' from gA/1_s6.npz."""
    files = sorted(cache.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz cache found under {cache}")
    sessions: dict[str, dict[str, np.ndarray]] = {}
    for file in files:
        subject = f"{file.parent.name}/{file.stem.split('_')[0]}"
        if subject in sessions:
            raise ValueError(f"Duplicate subject {subject}; one session per subject expected")
        sessions[subject] = tev.load_npz_records(file)
    return sessions


def concat(
    sessions: dict[str, dict[str, np.ndarray]], subjects: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    keys = sessions[subjects[0]].keys()
    data = {k: np.concatenate([sessions[s][k] for s in subjects], axis=0) for k in keys}
    session_id = np.concatenate(
        [np.full(len(sessions[s]["valid"]), i) for i, s in enumerate(subjects)]
    )
    return data, session_id


def usable_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["valid"] & (data["gaze_zone_id"] >= 0) & (~data["blinking"])


_SESSION_ID: np.ndarray | None = None
_original_windows = tev.build_disjoint_windows


def session_safe_windows(indices: np.ndarray, window_length: int) -> list[np.ndarray]:
    """Same as tev.build_disjoint_windows but also breaks at session boundaries."""
    if _SESSION_ID is None or len(indices) == 0:
        return _original_windows(indices, window_length)
    sess = _SESSION_ID[indices]
    breaks = np.where(np.diff(sess) != 0)[0] + 1
    windows: list[np.ndarray] = []
    for part in np.split(indices, breaks):
        windows.extend(_original_windows(part, window_length))
    return windows


tev.build_disjoint_windows = session_safe_windows


def evaluate_adapter(
    model: BlazeGazeDmdAdapter,
    data: dict[str, np.ndarray],
    window_length: int,
    device: torch.device,
    use_visual: bool,
    use_pose: bool,
) -> dict[str, Any]:
    global _SESSION_ID
    _SESSION_ID = None
    windows = tev.build_disjoint_windows(np.flatnonzero(usable_mask(data)), window_length)
    model.eval()
    with torch.no_grad():
        z_logits, p_pred, _, idx = tev.forward_windows(
            model, data, windows, device, use_visual=use_visual, use_pose=use_pose
        )
        valid_idx = np.flatnonzero(data["valid"])
        blink_windows = tev.build_disjoint_windows(valid_idx, window_length)
        _, _, b_logits, idx_b = tev.forward_windows(
            model, data, blink_windows, device, use_visual=use_visual, use_pose=use_pose
        )
    metrics = tev.evaluate_predictions(
        data["gaze_zone_id"][idx],
        z_logits.argmax(dim=-1).cpu().numpy(),
        data["canonical_pog"][idx],
        p_pred.cpu().numpy(),
    )
    metrics["blink_macro_f1"] = round(
        tev.macro_f1(
            data["blinking"][idx_b].astype(np.int64), b_logits.argmax(-1).cpu().numpy(), 2
        ),
        4,
    )
    return metrics


def run_fold(
    sessions: dict[str, dict[str, np.ndarray]],
    test_subject: str,
    args: argparse.Namespace,
    blazegaze: BlazeGazePyTorch,
    prototypes: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    global _SESSION_ID
    others = [s for s in sessions if s != test_subject]
    val_subject = others[0]
    train_subjects = others[1:]
    log(f"fold test={test_subject} val={val_subject} train={train_subjects}")

    fit_subjects = train_subjects + [val_subject]
    fit_data, fit_session = concat(sessions, fit_subjects)
    usable = usable_mask(fit_data)
    val_session = len(fit_subjects) - 1
    train_mask = usable & (fit_session != val_session)
    val_mask = usable & (fit_session == val_session)
    test = sessions[test_subject]
    log(
        f"  frames train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"test={int(usable_mask(test).sum())}"
    )

    result: dict[str, Any] = {
        "test_subject": test_subject,
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "test_zone_counts": {
            GAZE_ZONES[z]: int(c)
            for z, c in enumerate(
                np.bincount(test["gaze_zone_id"][usable_mask(test)], minlength=len(GAZE_ZONES))
            )
            if c > 0
        },
    }

    test_idx = np.flatnonzero(usable_mask(test))
    result["zeroshot"] = tev.evaluate_predictions(
        test["gaze_zone_id"][test_idx],
        tev.nearest_zone_from_pog(test["blazegaze_pog"][test_idx], prototypes),
        test["canonical_pog"][test_idx],
        test["blazegaze_pog"][test_idx],
    )
    log(f"  zeroshot      zone9_f1={result['zeroshot']['zone_9_macro_f1']}")

    # Per-driver calibration: K support frames per zone from the start of the test
    # session, evaluated on the later part (temporal split + buffer).
    calib_train, calib_val = tev.temporal_zone_split(
        test["valid"], test["gaze_zone_id"], test["blinking"]
    )
    result[f"maml_{args.few_shot_k}shot_test_calib"] = tev.run_webeyetrack_maml_few_shot(
        blazegaze.gaze_mlp, test, calib_train, calib_val, args.few_shot_k, prototypes, device
    )
    log(
        f"  maml{args.few_shot_k}shot     zone9_f1="
        f"{result[f'maml_{args.few_shot_k}shot_test_calib']['zone_9_macro_f1']}"
    )

    for name, use_visual, use_pose in (
        ("pose_only", False, True),
        ("visual_only", True, False),
        ("full_adapter", True, True),
    ):
        torch.manual_seed(args.seed)
        _SESSION_ID = fit_session
        model, val_metrics = tev.train_adapter(
            data=fit_data,
            train_mask=train_mask,
            val_mask=val_mask,
            window_length=args.window_length,
            epochs=args.epochs,
            lr=args.learning_rate,
            device=device,
            use_visual=use_visual,
            use_pose=use_pose,
        )
        test_metrics = evaluate_adapter(
            model, test, args.window_length, device, use_visual, use_pose
        )
        result[name] = {"val": val_metrics, "test": test_metrics}
        log(
            f"  {name:13s} val_zone9_f1={val_metrics['zone_9_macro_f1']} "
            f"test_zone9_f1={test_metrics['zone_9_macro_f1']} "
            f"test_att3_f1={test_metrics['attention_3_macro_f1']} "
            f"test_pog={test_metrics['pog_norm_l2_error']} "
            f"test_blink_f1={test_metrics['blink_macro_f1']}"
        )
    return result


def aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    methods = [k for k in folds[0] if isinstance(folds[0][k], dict) and k != "test_zone_counts"]
    out: dict[str, Any] = {}
    for method in methods:
        rows = [f[method]["test"] if "test" in f[method] else f[method] for f in folds]
        out[method] = {
            key: {
                "mean": round(float(np.mean([r[key] for r in rows])), 4),
                "std": round(float(np.std([r[key] for r in rows])), 4),
            }
            for key in SUMMARY_KEYS
        }
    return out


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sessions = load_sessions(args.cache)
    log(f"device={device} subjects={list(sessions)}")
    if len(sessions) < 3:
        raise SystemExit("LOSO needs at least 3 subjects (train, val, test)")

    blazegaze = BlazeGazePyTorch().to(device).eval()
    blazegaze.load_keras_archive(args.keras_weights)
    prototypes = np.array([GAZE_ZONE_CANONICAL_POG[z] for z in GAZE_ZONES], dtype=np.float32)

    fold_subjects = [s.strip() for s in args.folds.split(",") if s.strip()] or list(sessions)
    folds = [
        run_fold(sessions, s, args, blazegaze, prototypes, device) for s in fold_subjects
    ]
    summary = {"config": {k: str(v) for k, v in vars(args).items()}, "folds": folds,
               "mean_over_folds": aggregate(folds)}

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / "loso_result.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("=== mean over folds (test) ===")
    for method, stats in summary["mean_over_folds"].items():
        cells = " ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}" for k, v in stats.items())
        log(f"  {method:22s} {cells}")
    log(f"saved {out}")


if __name__ == "__main__":
    main()
