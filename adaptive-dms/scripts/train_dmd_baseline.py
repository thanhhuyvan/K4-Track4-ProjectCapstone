"""Train the tiny causal DMD auxiliary-event baseline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from adaptive_dms.dms import (
    CausalDmsBaseline,
    CausalDmsBaselineV2,
    CausalDmsBaselineV3,
    geometry_from_landmarks,
)


@dataclass
class Session:
    subject: str
    left_eye: np.ndarray
    right_eye: np.ndarray
    mouth: np.ndarray
    face: np.ndarray | None
    geometry: np.ndarray
    valid: np.ndarray
    eye: np.ndarray
    yawn: np.ndarray
    blink: np.ndarray


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        sessions: list[Session],
        subjects: set[str],
        length: int,
        binary_yawn: bool = False,
    ) -> None:
        self.sessions = [session for session in sessions if session.subject in subjects]
        self.length = length
        self.binary_yawn = binary_yawn
        self.windows = [
            (session_index, start)
            for session_index, session in enumerate(self.sessions)
            for start in range(0, len(session.eye) - length + 1, length)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        session_index, start = self.windows[index]
        session = self.sessions[session_index]
        selection = slice(start, start + self.length)
        item = {
            "left_eye": torch.from_numpy(session.left_eye[selection]).float() / 255.0,
            "right_eye": torch.from_numpy(session.right_eye[selection]).float() / 255.0,
            "mouth": torch.from_numpy(session.mouth[selection]).float() / 255.0,
            "geometry": torch.from_numpy(session.geometry[selection]).float(),
            "valid": torch.from_numpy(session.valid[selection]),
            "eye": torch.from_numpy(session.eye[selection]).long(),
            "yawn": torch.from_numpy(
                (session.yawn[selection] > 0).astype(np.int64)
                if self.binary_yawn
                else session.yawn[selection]
            ).long(),
            "blink": torch.from_numpy(session.blink[selection]).long(),
        }
        if session.face is not None:
            item["face"] = torch.from_numpy(session.face[selection]).float() / 255.0
        return item


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-subjects", default="1,5,6")
    parser.add_argument("--val-subjects", default="7")
    parser.add_argument("--test-subjects", default="9")
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument(
        "--model-version", choices=("v1", "v2", "v3"), default="v1"
    )
    parser.add_argument("--yawn-positive-weight", type=float, default=2.0)
    return parser.parse_args()


def load_sessions(cache: Path) -> list[Session]:
    sessions = []
    for path in sorted(cache.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            raw_eye = data["eye_state"].astype(np.int64)
            eye = np.full_like(raw_eye, -100)
            for raw, target in ((1, 0), (2, 1), (3, 2), (4, 3)):
                eye[raw_eye == raw] = target
            sessions.append(
                Session(
                    subject=str(metadata["subject"]),
                    left_eye=data["left_eye"].copy(),
                    right_eye=data["right_eye"].copy(),
                    mouth=data["mouth"].copy(),
                    face=data["face"].copy() if "face" in data.files else None,
                    geometry=geometry_from_landmarks(
                        data["landmarks"], data["quality"]
                    ),
                    valid=data["valid"].copy(),
                    eye=eye,
                    yawn=data["yawning"].astype(np.int64),
                    blink=data["blinking"].astype(np.int64),
                )
            )
    return sessions


def subject_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def balanced_overfit_indices(dataset: SequenceDataset, count: int = 8) -> list[int]:
    categories: dict[str, list[int]] = {"yawn": [], "close": [], "neutral": []}
    for index in range(len(dataset)):
        item = dataset[index]
        if bool((item["yawn"] > 0).any()):
            categories["yawn"].append(index)
        elif bool((item["eye"] == 1).any()):
            categories["close"].append(index)
        else:
            categories["neutral"].append(index)
    selected = categories["yawn"][:3] + categories["close"][:3]
    selected += categories["neutral"][: max(0, count - len(selected))]
    if not selected:
        raise RuntimeError("No overfit windows available")
    return selected[:count]


def class_weights(dataset: SequenceDataset, key: str, classes: int) -> torch.Tensor:
    counts = np.zeros(classes, dtype=np.float64)
    for index in range(len(dataset)):
        target = dataset[index][key].numpy()
        valid = target >= 0
        counts += np.bincount(target[valid], minlength=classes)
    weights = counts.sum() / (classes * np.maximum(counts, 1.0))
    return torch.from_numpy(np.clip(weights, 0.25, 4.0).astype(np.float32))


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    mask = valid & (target >= 0)
    return nn.functional.cross_entropy(logits[mask], target[mask], weight=weights)


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward(
    model: nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return model(
        batch["left_eye"],
        batch["right_eye"],
        batch["mouth"],
        batch["geometry"],
        batch.get("face"),
    )


def loss_for_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = forward(model, batch)
    eye = masked_cross_entropy(output["eye"], batch["eye"], batch["valid"], weights["eye"])
    yawn = masked_cross_entropy(output["yawn"], batch["yawn"], batch["valid"], weights["yawn"])
    blink = masked_cross_entropy(output["blink"], batch["blink"], batch["valid"], weights["blink"])
    return eye + 1.5 * yawn + 0.5 * blink, output


def macro_f1(target: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    scores = []
    for value in range(classes):
        true_positive = np.sum((target == value) & (prediction == value))
        false_positive = np.sum((target != value) & (prediction == value))
        false_negative = np.sum((target == value) & (prediction != value))
        denominator = 2 * true_positive + false_positive + false_negative
        if np.any(target == value) and denominator:
            scores.append(2 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    yawn_classes: int,
    yawn_threshold: float | None = None,
) -> dict[str, float]:
    model.eval()
    targets: dict[str, list[np.ndarray]] = {key: [] for key in ("eye", "yawn", "blink")}
    predictions: dict[str, list[np.ndarray]] = {key: [] for key in targets}
    for raw_batch in loader:
        batch = move(raw_batch, device)
        output = forward(model, batch)
        valid = batch["valid"]
        for key, target_list in targets.items():
            mask = valid & (batch[key] >= 0)
            target_list.append(batch[key][mask].cpu().numpy())
            if key == "yawn" and yawn_classes == 2 and yawn_threshold is not None:
                probability = output[key].softmax(-1)[..., 1]
                prediction = (probability >= yawn_threshold).long()
            else:
                prediction = output[key].argmax(-1)
            predictions[key].append(prediction[mask].cpu().numpy())
    result = {}
    for key, classes in (("eye", 4), ("yawn", yawn_classes), ("blink", 2)):
        target = np.concatenate(targets[key])
        prediction = np.concatenate(predictions[key])
        result[f"{key}_accuracy"] = float(np.mean(target == prediction))
        result[f"{key}_macro_f1"] = macro_f1(target, prediction, classes)
        if key == "yawn":
            binary_target = (target > 0).astype(np.int64)
            binary_prediction = (prediction > 0).astype(np.int64)
            result["yawn_binary_accuracy"] = float(
                np.mean(binary_target == binary_prediction)
            )
            result["yawn_binary_macro_f1"] = macro_f1(
                binary_target, binary_prediction, 2
            )
    result["score"] = (result["eye_macro_f1"] + result["yawn_macro_f1"] + result["blink_macro_f1"]) / 3.0
    return result


@torch.no_grad()
def tune_yawn_threshold(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    targets, probabilities = [], []
    for raw_batch in loader:
        batch = move(raw_batch, device)
        output = forward(model, batch)
        mask = batch["valid"] & (batch["yawn"] >= 0)
        targets.append(batch["yawn"][mask].cpu().numpy())
        probabilities.append(output["yawn"].softmax(-1)[..., 1][mask].cpu().numpy())
    target = np.concatenate(targets)
    probability = np.concatenate(probabilities)
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        score = macro_f1(target, (probability >= threshold).astype(np.int64), 2)
        if score > best[0]:
            best = (score, float(threshold))
    return best[1], best[0]


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    sessions = load_sessions(args.cache)
    binary_yawn = args.model_version != "v1"
    yawn_classes = 2 if binary_yawn else 3
    train = SequenceDataset(
        sessions,
        subject_set(args.train_subjects),
        args.sequence_length,
        binary_yawn=binary_yawn,
    )
    validation = SequenceDataset(
        sessions,
        subject_set(args.val_subjects),
        args.sequence_length,
        binary_yawn=binary_yawn,
    )
    test = SequenceDataset(
        sessions,
        subject_set(args.test_subjects),
        args.sequence_length,
        binary_yawn=binary_yawn,
    )
    if args.overfit:
        train_data: Dataset[dict[str, torch.Tensor]] = Subset(train, balanced_overfit_indices(train))
        validation_data = train_data
    else:
        train_data = train
        validation_data = validation
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=not args.overfit)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    model: nn.Module
    if args.model_version == "v3":
        if any(session.face is None for session in sessions):
            raise RuntimeError("V3 requires caches rebuilt with aligned face crops")
        model = CausalDmsBaselineV3().to(device)
    elif args.model_version == "v2":
        model = CausalDmsBaselineV2().to(device)
    else:
        model = CausalDmsBaseline().to(device)
    weights = {
        "eye": class_weights(train, "eye", 4).to(device),
        "yawn": (
            torch.tensor((1.0, args.yawn_positive_weight), device=device)
            if binary_yawn
            else class_weights(train, "yawn", 3).to(device)
        ),
        "blink": class_weights(train, "blink", 2).to(device),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0 if args.overfit else 1e-4
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(
        json.dumps({key: str(value) for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    log_path = args.run_dir / "metrics.jsonl"
    best_score = -1.0
    stale = 0
    epochs = args.epochs
    print(
        f"device={device} parameters={sum(p.numel() for p in model.parameters())} "
        f"train_windows={len(train_data)} val_windows={len(validation_data)} "
        f"test_windows={len(test)}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for_batch(model, batch, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        metrics = evaluate(model, validation_loader, device, yawn_classes)
        record = {"epoch": epoch, "train_loss": total_loss / batches, **metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps(record), flush=True)
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            stale = 0
            torch.save({"model": model.state_dict(), "record": record}, args.run_dir / "best.pt")
        else:
            stale += 1
        if not args.overfit and stale >= args.patience:
            print(f"early_stop={epoch} best_score={best_score:.6f}", flush=True)
            break

    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final: dict[str, object] = {
        "best_validation": checkpoint["record"],
        "test": evaluate(model, test_loader, device, yawn_classes),
    }
    if binary_yawn:
        threshold, calibration_score = tune_yawn_threshold(
            model, validation_loader, device
        )
        final["calibration"] = {
            "yawn_threshold": threshold,
            "validation_yawn_macro_f1": calibration_score,
        }
        final["test_calibrated"] = evaluate(
            model,
            test_loader,
            device,
            yawn_classes,
            yawn_threshold=threshold,
        )
    torch.save({"model": model.state_dict(), "result": final}, args.run_dir / "final.pt")
    (args.run_dir / "result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
