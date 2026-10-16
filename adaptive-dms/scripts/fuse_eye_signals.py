"""Does fusing CNN + EAR + blendshape beat each signal alone on an unseen subject?

Protocol (no leakage):
  * CNN is the stage-1 model: trained on the train part of the non-test subjects.
  * Fusion (logistic regression) is fitted on the val part of those subjects,
    i.e. frames the CNN never trained on, so its logits are not over-confident.
  * Threshold is chosen on the same val part; everything is reported on the
    test subject, which no component has seen.
All features are causal (trailing windows only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import train_eye_crop as tec
from adaptive_dms.dms import EyeBlinkCNN

FEATURES = ("cnn", "cnn_ma5", "ear", "ear_norm", "ear_norm_ma5", "blend", "blend_ma5")
NO_CNN = ("ear", "ear_norm", "ear_norm_ma5", "blend", "blend_ma5")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/eye_crops_16"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/eye_crop"))
    parser.add_argument("--test-subject", default="gB/7")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gap-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-seconds", type=float, default=30.0)
    return parser.parse_args()


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Causal mean of the last k samples."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)


def open_eye_baseline(ear: np.ndarray, window: int, fallback: float = 0.25) -> np.ndarray:
    """Causal per-person open-eye EAR: 90th percentile of the trailing window."""
    out = np.full(len(ear), fallback, np.float32)
    current = fallback
    for i in range(len(ear)):
        if i % 15 == 0 and i >= 30:
            current = float(np.percentile(ear[max(0, i - window): i + 1], 90))
        out[i] = current
    return out


def build_features(subject: dict[str, np.ndarray], logit: np.ndarray, fps: float,
                   baseline_seconds: float) -> np.ndarray:
    ear = subject["ear"].astype(np.float32)
    ear_norm = ear / np.maximum(open_eye_baseline(ear, int(baseline_seconds * fps)), 1e-3)
    blend = subject["blink_score"].astype(np.float32)
    columns = {
        "cnn": logit, "cnn_ma5": moving_average(logit, 5),
        "ear": ear, "ear_norm": ear_norm, "ear_norm_ma5": moving_average(ear_norm, 5),
        "blend": blend, "blend_ma5": moving_average(blend, 5),
    }
    return np.stack([columns[name] for name in FEATURES], axis=1).astype(np.float32)


class Fusion:
    """Standardised logistic regression."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.columns = [FEATURES.index(n) for n in names]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Fusion":
        x = x[:, self.columns]
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        xt = torch.from_numpy((x - self.mean) / self.std).float()
        yt = torch.from_numpy(y.astype(np.float32))
        self.w = torch.zeros(xt.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        pos_weight = torch.tensor(float(np.clip((len(y) - y.sum()) / max(y.sum(), 1), 1, 8)))
        optimizer = torch.optim.LBFGS([self.w, self.b], max_iter=200)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                xt @ self.w + self.b, yt, pos_weight=pos_weight) + 1e-3 * (self.w ** 2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = (x[:, self.columns] - self.mean) / self.std
        return z @ self.w.detach().numpy() + float(self.b.detach())

    def describe(self) -> dict[str, float]:
        return {n: round(float(w), 3) for n, w in zip(self.names, self.w.detach().numpy())}


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    subjects = tec.load(args.data)
    if args.test_subject not in subjects:
        raise SystemExit(f"--test-subject must be one of {list(subjects)}")
    others = [s for s in subjects if s != args.test_subject]

    splits = [tec.temporal_split(subjects[s], args.val_fraction, args.gap_frames) for s in others]
    train, val_cnn = tec.stack([s[0] for s in splits]), tec.stack([s[1] for s in splits])
    tec.log("training stage-1 CNN (test subject unseen)")
    cnn, best_epoch = tec.fit(train, val_cnn, args, args.epochs)
    tec.log(f"CNN best epoch {best_epoch}")

    def featurise(subject: dict[str, np.ndarray]) -> np.ndarray:
        fps = 1.0 / max(float(np.median(np.diff(subject["timestamps_s"]))), 1e-3)
        return build_features(subject, tec.predict(cnn, subject["crops"]), fps,
                              args.baseline_seconds)

    val_parts = []
    for name in others:
        full = dict(subjects[name], feat=featurise(subjects[name]))
        val_parts.append(tec.temporal_split(full, args.val_fraction, args.gap_frames)[1])
    val = tec.stack(val_parts)
    test = subjects[args.test_subject]
    test_feat = featurise(test)
    seconds = float(test["timestamps_s"][-1] - test["timestamps_s"][0])
    tec.log(f"fusion fit on {len(val['blink'])} unseen val frames "
            f"(closed {int(val['blink'].sum())}); test {len(test['blink'])} "
            f"(closed {int(test['blink'].sum())})")

    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, column, sign in (("cnn", "cnn", 1), ("ear", "ear", -1),
                               ("ear_norm", "ear_norm", -1), ("blendshape", "blend", 1)):
        i = FEATURES.index(column)
        candidates[name] = (sign * val["feat"][:, i], sign * test_feat[:, i])
    coefficients = {}
    for name, names in (("fused_no_cnn", NO_CNN), ("fused_all", FEATURES)):
        model = Fusion(names).fit(val["feat"], val["blink"])
        candidates[name] = (model.score(val["feat"]), model.score(test_feat))
        coefficients[name] = model.describe()

    report: dict[str, object] = {"test_subject": args.test_subject, "cnn_best_epoch": best_epoch,
                                 "coefficients": coefficients}
    tec.log(f"{'signal':13s} {'AUC':>6s} {'F1':>6s} {'prec':>6s} {'rec':>6s} "
            f"{'evt_rec':>8s} {'FA/min':>7s}")
    for name, (val_score, test_score) in candidates.items():
        threshold = tec.best_threshold(val_score, val["blink"])
        r = tec.score_report(test_score, test["blink"], threshold, seconds)
        report[name] = r
        tec.log(f"{name:13s} {r['auc']:6.3f} {r['f1_closed']:6.3f} {r['precision']:6.3f} "
                f"{r['recall']:6.3f} {r['event_recall']:8.3f} {r['false_events_per_min']:7.2f}")
    tec.log(f"coefficients {json.dumps(coefficients)}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "fusion_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
