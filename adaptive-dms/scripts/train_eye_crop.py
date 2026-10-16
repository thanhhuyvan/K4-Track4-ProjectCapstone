"""Train the 16x16 eye-crop CNN (blink / eyes-closed) on the cached DMD crops.

Stage 1: one honest hold-out. One subject is test, the last part of every other
subject is validation (threshold + epoch selection). CNN is compared with the
EAR and MediaPipe-blendshape baselines on the exact same test frames.
Stage 2: retrain on every subject for the demo checkpoint (no LOSO).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adaptive_dms.dms import EyeBlinkCNN


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/eye_crops_16"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/eye_crop"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/eye_cnn_16.pt"))
    parser.add_argument("--test-subject", default="gB/7")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gap-frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load(data: Path) -> dict[str, dict[str, np.ndarray]]:
    subjects = {}
    for file in sorted(data.glob("*.npz")):
        with np.load(file) as z:
            arrays = {k: z[k] for k in z.files if k != "subject"}
            name = str(z["subject"])
        valid = arrays["valid"]
        subjects[name] = {k: v[valid] for k, v in arrays.items() if k != "valid"}
        log(f"{name}: {int(valid.sum())} valid frames, {int(subjects[name]['blink'].sum())} closed")
    if not subjects:
        raise SystemExit(f"no dataset in {data}")
    return subjects


def stack(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def temporal_split(
    subject: dict[str, np.ndarray], val_fraction: float, gap: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = len(subject["blink"])
    cut = int(n * (1.0 - val_fraction))
    train = {k: v[:max(cut - gap, 0)] for k, v in subject.items()}
    val = {k: v[cut:] for k, v in subject.items()}
    return train, val


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def prf(pred: np.ndarray, label: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def best_threshold(score: np.ndarray, label: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    return float(max(candidates, key=lambda t: prf(score >= t, label)[2]))


def runs(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(s, e) for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if e - s >= min_length]


def event_metrics(pred: np.ndarray, label: np.ndarray, seconds: float) -> dict[str, float]:
    gt = runs(label)
    predicted = runs(pred, 2)
    hit = sum(any(ps < ge and pe > gs for ps, pe in predicted) for gs, ge in gt)
    false = sum(not any(ps < ge and pe > gs for gs, ge in gt) for ps, pe in predicted)
    return {
        "gt_events": len(gt),
        "event_recall": round(hit / max(len(gt), 1), 4),
        "false_events_per_min": round(false / max(seconds / 60.0, 1e-6), 2),
    }


def score_report(score: np.ndarray, label: np.ndarray, threshold: float,
                 seconds: float) -> dict[str, float]:
    pred = score >= threshold
    precision, recall, f1 = prf(pred, label)
    return {
        "auc": round(auc(score, label), 4),
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_closed": round(f1, 4),
        **event_metrics(pred, label, seconds),
    }


def augment(x: torch.Tensor) -> torch.Tensor:
    """Brightness/contrast jitter, noise, and up to 1 px shifts per quarter batch."""
    b = x.shape[0]
    contrast = 1.0 + 0.3 * (torch.rand(b, 1, 1, 1) - 0.5) * 2
    brightness = 0.12 * (torch.rand(b, 1, 1, 1) - 0.5) * 2
    x = (x - 0.5) * contrast + 0.5 + brightness + 0.02 * torch.randn_like(x)
    chunks = []
    for part in x.chunk(4):
        dy, dx = random.randint(-1, 1), random.randint(-1, 1)
        chunks.append(torch.roll(part, shifts=(dy, dx), dims=(-2, -1)))
    return torch.cat(chunks).clamp(0.0, 1.0)


@torch.no_grad()
def predict(model: nn.Module, crops: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(crops).float().div_(255.0)
    return torch.cat([model(x[i:i + 2048]) for i in range(0, len(x), 2048)]).numpy()


def fit(train: dict[str, np.ndarray], val: dict[str, np.ndarray] | None,
        args: argparse.Namespace, epochs: int) -> tuple[nn.Module, int]:
    torch.manual_seed(args.seed)
    model = EyeBlinkCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=epochs * int(np.ceil(
            len(train["blink"]) / args.batch_size)))
    x_all = torch.from_numpy(train["crops"]).float().div_(255.0)
    y_all = torch.from_numpy(train["blink"].astype(np.float32))
    pos = float(y_all.sum())
    pos_weight = torch.tensor(float(np.clip((len(y_all) - pos) / max(pos, 1.0), 1.0, 8.0)))
    best, best_state, best_epoch = -1.0, copy.deepcopy(model.state_dict()), epochs
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_all))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            loss = F.binary_cross_entropy_with_logits(
                model(augment(x_all[idx])), y_all[idx], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)
        message = f"  epoch {epoch:02d}/{epochs} loss={total / len(order):.4f}"
        if val is not None:
            val_auc = auc(predict(model, val["crops"]), val["blink"])
            message += f" val_auc={val_auc:.4f}"
            if val_auc > best:
                best, best_epoch = val_auc, epoch
                best_state = copy.deepcopy(model.state_dict())
        log(message)
    if val is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    subjects = load(args.data)
    if args.test_subject not in subjects:
        raise SystemExit(f"--test-subject must be one of {list(subjects)}")

    others = [s for s in subjects if s != args.test_subject]
    splits = [temporal_split(subjects[s], args.val_fraction, args.gap_frames) for s in others]
    train, val = stack([s[0] for s in splits]), stack([s[1] for s in splits])
    test = subjects[args.test_subject]
    log(f"train={len(train['blink'])} (closed {int(train['blink'].sum())}) "
        f"val={len(val['blink'])} (closed {int(val['blink'].sum())}) "
        f"test={args.test_subject} {len(test['blink'])} (closed {int(test['blink'].sum())})")

    log("stage 1: train with hold-out")
    model, best_epoch = fit(train, val, args, args.epochs)
    params = sum(p.numel() for p in model.parameters())
    log(f"best epoch {best_epoch}, parameters {params}")

    val_scores = {"cnn": predict(model, val["crops"]), "ear": -val["ear"],
                  "blendshape": val["blink_score"]}
    test_scores = {"cnn": predict(model, test["crops"]), "ear": -test["ear"],
                   "blendshape": test["blink_score"]}
    seconds = float(test["timestamps_s"][-1] - test["timestamps_s"][0])
    report: dict[str, object] = {
        "test_subject": args.test_subject, "parameters": params, "best_epoch": best_epoch,
        "test_frames": int(len(test["blink"])), "test_closed_frames": int(test["blink"].sum()),
    }
    thresholds = {}
    for name in ("cnn", "ear", "blendshape"):
        thresholds[name] = best_threshold(val_scores[name], val["blink"])
        report[name] = score_report(test_scores[name], test["blink"], thresholds[name], seconds)
        log(f"  {name:10s} {json.dumps(report[name])}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.skip_final:
        return
    log("stage 2: retrain on all subjects for the demo checkpoint")
    everything = stack(list(subjects.values()))
    final, _ = fit(everything, None, args, best_epoch)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final.state_dict(), "size": 16,
                "threshold": thresholds["cnn"], "trained_subjects": list(subjects),
                "holdout_report": report["cnn"]}, args.checkpoint)
    log(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
