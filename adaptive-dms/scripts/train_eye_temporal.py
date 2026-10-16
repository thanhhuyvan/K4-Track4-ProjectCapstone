"""Can temporal context + light attention separate 'eyes closed' from 'looking down'?

Per-frame target: DMD blink label. Inputs are causal 1 s windows of MediaPipe
blendshapes (blink / look-down / look-up / wide / squint), EAR, head pitch/yaw.
All models are tiny; the evaluation is leave-one-subject-out over the 4 videos
(val = last 20% of each training subject, used for epochs and thresholds).

Reported per model: AUC, F1, recall, and the false-closed rate on frames where
the driver looks DOWN (steering wheel / infotainment) but does not blink.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_eye_crop as tec

BLEND_USE = ("eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
             "eyeLookUpLeft", "eyeLookUpRight", "eyeWideLeft", "eyeWideRight",
             "eyeSquintLeft", "eyeSquintRight")
FEATURES = BLEND_USE + ("ear", "pitch", "yaw")
DOWN_ZONES = (7, 8)  # infotainment, steering_wheel


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/eye_feat_16"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs/eye_temporal.json"))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--gap-frames", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def load_all(data: Path) -> dict[str, dict[str, np.ndarray]]:
    subjects = {}
    for file in sorted(data.glob("*.npz")):
        with np.load(file) as z:
            valid = z["valid"]
            index = {str(n): i for i, n in enumerate(z["blend_names"])}
            blend = z["blend"][valid].astype(np.float32)
            euler = z["euler"][valid]
            feat = np.column_stack(
                [blend[:, index[n]] for n in BLEND_USE]
                + [z["ear"][valid], euler[:, 0], euler[:, 1]]
            ).astype(np.float32)
            subjects[str(z["subject"])] = {
                "feat": feat, "blink": z["blink"][valid],
                "zone": z["gaze_zone_id"][valid].astype(np.int64),
            }
    if not subjects:
        raise SystemExit(f"no dataset in {data}")
    return subjects


def make_windows(feat: np.ndarray, window: int) -> np.ndarray:
    padded = np.concatenate([np.repeat(feat[:1], window - 1, axis=0), feat])
    return np.lib.stride_tricks.sliding_window_view(
        padded, (window, feat.shape[1]))[:, 0].astype(np.float32)


class WindowLogistic(nn.Module):
    """Logistic regression on last / mean / max / min of each feature."""

    def __init__(self, dim: int, **_: int) -> None:
        super().__init__()
        self.linear = nn.Linear(4 * dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = torch.cat([x[:, -1], x.mean(1), x.max(1).values, x.min(1).values], dim=-1)
        return self.linear(stats).squeeze(-1)


class FrameLogistic(nn.Module):
    """Same inputs but only the current frame (no temporal context)."""

    def __init__(self, dim: int, **_: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[:, -1]).squeeze(-1)


class GruModel(nn.Module):
    def __init__(self, dim: int, hidden: int = 32, **_: int) -> None:
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0][:, -1]).squeeze(-1)


class AttentionPool(nn.Module):
    """Per-step embedding, one attention head over the window, current frame kept."""

    def __init__(self, dim: int, hidden: int = 32, **_: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU())
        self.query = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        weights = torch.softmax(self.query(h).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * h).sum(1)
        return self.head(torch.cat([pooled, h[:, -1]], dim=-1)).squeeze(-1)


MODELS = {"frame_logistic": FrameLogistic, "window_logistic": WindowLogistic,
          "gru": GruModel, "attention": AttentionPool}


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    t = torch.from_numpy(x)
    return torch.cat([model(t[i:i + 4096]) for i in range(0, len(t), 4096)]).numpy()


def fit(name: str, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
        y_val: np.ndarray, args: argparse.Namespace, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = MODELS[name](x_train.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    x, y = torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))
    pos = float(y.sum())
    weight = torch.tensor(float(np.clip((len(y) - pos) / max(pos, 1.0), 1.0, 8.0)))
    best, state = -1.0, None
    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            loss = F.binary_cross_entropy_with_logits(model(x[idx]), y[idx], pos_weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = tec.auc(predict(model, x_val), y_val)
        if score > best:
            best, state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def evaluate(score_val: np.ndarray, y_val: np.ndarray, score_test: np.ndarray,
             test: dict[str, np.ndarray]) -> dict[str, float]:
    threshold = tec.best_threshold(score_val, y_val)
    pred = score_test >= threshold
    label = test["blink"]
    precision, recall, f1 = tec.prf(pred, label)
    down = np.isin(test["zone"], DOWN_ZONES)

    def false_rate(mask: np.ndarray) -> float:
        return float(pred[mask & ~label].mean()) if (mask & ~label).any() else float("nan")

    return {"auc": tec.auc(score_test, label), "f1": f1, "precision": precision,
            "recall": recall, "fpr_down": false_rate(down), "fpr_other": false_rate(~down),
            "down_frames": int((down & ~label).sum())}


def main() -> None:
    args = arguments()
    subjects = load_all(args.data)
    names = list(subjects)
    blink_idx = [FEATURES.index("eyeBlinkLeft"), FEATURES.index("eyeBlinkRight")]
    down_idx = [FEATURES.index("eyeLookDownLeft"), FEATURES.index("eyeLookDownRight")]
    results: dict[str, list[dict[str, float]]] = {}

    for test_name in names:
        train_names = [n for n in names if n != test_name]
        stats = np.concatenate([subjects[n]["feat"] for n in train_names])
        mean, std = stats.mean(0), stats.std(0) + 1e-6
        windows = {n: make_windows((subjects[n]["feat"] - mean) / std, args.window)
                   for n in names}
        x_tr, y_tr, x_va, y_va = [], [], [], []
        for n in train_names:
            size = len(subjects[n]["blink"])
            cut = int(size * (1 - args.val_fraction))
            x_tr.append(windows[n][:max(cut - args.gap_frames, 0)])
            y_tr.append(subjects[n]["blink"][:max(cut - args.gap_frames, 0)])
            x_va.append(windows[n][cut:])
            y_va.append(subjects[n]["blink"][cut:])
        x_tr, y_tr = np.concatenate(x_tr), np.concatenate(y_tr)
        x_va, y_va = np.concatenate(x_va), np.concatenate(y_va)
        test = subjects[test_name]

        raw_val = np.concatenate([subjects[n]["feat"][int(len(subjects[n]["blink"]) * (
            1 - args.val_fraction)):] for n in train_names])
        for label, fn in (("hand: blink", lambda f: f[:, blink_idx].mean(1)),
                          ("hand: blink - lookDown", lambda f: f[:, blink_idx].mean(1)
                           - f[:, down_idx].mean(1))):
            results.setdefault(label, []).append(
                evaluate(fn(raw_val), y_va, fn(test["feat"]), test))

        for model_name in MODELS:
            runs = []
            for seed in range(args.seeds):
                model = fit(model_name, x_tr, y_tr, x_va, y_va, args, seed)
                runs.append(evaluate(predict(model, x_va), y_va,
                                     predict(model, windows[test_name]), test))
            results.setdefault(model_name, []).append(
                {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]})
        tec.log(f"fold test={test_name} done ({int((test['zone'] >= 7).sum())} frames in down zones)")

    keys = ("auc", "f1", "recall", "fpr_down", "fpr_other")
    tec.log(f"{'model':24s} " + " ".join(f"{k:>9s}" for k in keys))
    for model_name, folds in results.items():
        tec.log(f"{model_name:24s} " + " ".join(
            f"{np.nanmean([f[k] for f in folds]):9.3f}" for k in keys))
    tec.log("per fold (auc / f1 / fpr_down), folds in order " + str(names))
    for model_name, folds in results.items():
        tec.log(f"  {model_name:24s} " + "  ".join(
            f"{f['auc']:.2f}/{f['f1']:.2f}/{f['fpr_down']:.2f}" for f in folds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"subjects": names, "features": FEATURES,
                                    "window": args.window, "results": results}, indent=2),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
