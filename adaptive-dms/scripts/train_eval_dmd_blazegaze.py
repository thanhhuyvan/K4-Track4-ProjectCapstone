"""Fine-tune and evaluate BlazeGaze on DMD s6 Gaze Zones + Driver Attention."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from adaptive_dms.data import GAZE_ZONE_CANONICAL_POG, GAZE_ZONES
from adaptive_dms.dms import BlazeGazeDmdAdapter, BlazeGazeMlp, BlazeGazePyTorch

# Map 9 DMD gaze zones into 3 high-level DMS driver attention states:
# 0 = on_road (front, front_right)
# 1 = mirror_check (left_mirror, center_mirror, right_mirror)
# 2 = distracted (left, right, infotainment, steering_wheel)
ZONE_TO_ATTENTION_GROUP: dict[int, int] = {
    0: 1,  # left_mirror -> mirror_check
    1: 2,  # left -> distracted
    2: 0,  # front -> on_road
    3: 1,  # center_mirror -> mirror_check
    4: 0,  # front_right -> on_road
    5: 1,  # right_mirror -> mirror_check
    6: 2,  # right -> distracted
    7: 2,  # infotainment -> distracted
    8: 2,  # steering_wheel -> distracted
}
ATTENTION_NAMES = ("on_road", "mirror_check", "distracted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate BlazeGaze on DMD Gaze Zones."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/cache_dmd_blazegaze/gA/1_s6.npz"),
        help="Path to cached DMD session .npz (or directory of .npz files)",
    )
    parser.add_argument(
        "--test-cache",
        type=Path,
        default=None,
        help="Optional external test session .npz for cross-session / cross-driver evaluation",
    )
    parser.add_argument(
        "--keras-weights",
        type=Path,
        default=Path(
            "weights/blazegaze_mpiifacegaze.keras"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/runs/dmd_blazegaze_s6"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--buffer-ratio", type=float, default=0.10)
    parser.add_argument("--window-length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--few-shot-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_npz_records(path: Path) -> dict[str, np.ndarray]:
    files = sorted(path.rglob("*.npz")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .npz cache found at {path}")
    arrays: dict[str, list[np.ndarray]] = {}
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            for key in (
                "frame_indices",
                "timestamps_s",
                "head_vector",
                "face_origin_3d",
                "ears",
                "geometry",
                "valid",
                "blazegaze_emb",
                "blazegaze_hidden",
                "blazegaze_pog",
                "gaze_zone_id",
                "canonical_pog",
                "blinking",
            ):
                arrays.setdefault(key, []).append(data[key].copy())
    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


def temporal_zone_split(
    valid: np.ndarray,
    gaze_zone_id: np.ndarray,
    blinking: np.ndarray,
    train_ratio: float = 0.65,
    buffer_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each gaze zone interval temporally into train, buffer (dropped), and holdout val."""
    usable = valid & (gaze_zone_id >= 0) & (~blinking)
    train_mask = np.zeros_like(usable, dtype=bool)
    val_mask = np.zeros_like(usable, dtype=bool)

    for zone_id in range(len(GAZE_ZONES)):
        idx = np.flatnonzero(usable & (gaze_zone_id == zone_id))
        if len(idx) < 4:
            continue
        n_train = max(1, int(round(len(idx) * train_ratio)))
        n_buf = max(1, int(round(len(idx) * buffer_ratio)))
        val_start = min(len(idx) - 1, n_train + n_buf)
        train_mask[idx[:n_train]] = True
        val_mask[idx[val_start:]] = True

    return train_mask, val_mask


def macro_f1(target: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((target == c) & (pred == c)))
        fp = int(np.sum((target != c) & (pred == c)))
        fn = int(np.sum((target == c) & (pred != c)))
        denom = 2 * tp + fp + fn
        if np.any(target == c):
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate_predictions(
    zone_target: np.ndarray,
    zone_pred: np.ndarray,
    pog_target: np.ndarray,
    pog_pred: np.ndarray,
) -> dict[str, Any]:
    acc_9 = float(np.mean(zone_target == zone_pred))
    f1_9 = macro_f1(zone_target, zone_pred, len(GAZE_ZONES))

    att_target = np.array([ZONE_TO_ATTENTION_GROUP[int(z)] for z in zone_target])
    att_pred = np.array([ZONE_TO_ATTENTION_GROUP[int(z)] for z in zone_pred])
    acc_3 = float(np.mean(att_target == att_pred))
    f1_3 = macro_f1(att_target, att_pred, 3)

    dist_target = (att_target == 2).astype(np.int64)
    dist_pred = (att_pred == 2).astype(np.int64)
    dist_f1 = macro_f1(dist_target, dist_pred, 2)

    pog_l2 = float(np.mean(np.linalg.norm(pog_pred - pog_target, axis=-1)))
    per_zone_acc = {}
    for z_id, z_name in enumerate(GAZE_ZONES):
        mask = zone_target == z_id
        if np.any(mask):
            per_zone_acc[z_name] = round(
                float(np.mean(zone_pred[mask] == z_id)), 4
            )

    return {
        "zone_9_accuracy": round(acc_9, 4),
        "zone_9_macro_f1": round(f1_9, 4),
        "attention_3_accuracy": round(acc_3, 4),
        "attention_3_macro_f1": round(f1_3, 4),
        "distracted_binary_macro_f1": round(dist_f1, 4),
        "pog_norm_l2_error": round(pog_l2, 4),
        "per_zone_accuracy": per_zone_acc,
    }


def nearest_zone_from_pog(
    pog: np.ndarray, prototypes: np.ndarray
) -> np.ndarray:
    dists = np.linalg.norm(pog[:, None, :] - prototypes[None, :, :], axis=-1)
    return np.argmin(dists, axis=-1)


def compute_affine_2x3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_aug = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float32)])
    mat, _, _, _ = np.linalg.lstsq(src_aug, dst, rcond=None)
    return mat.T.astype(np.float32)


def run_webeyetrack_maml_few_shot(
    base_mlp: BlazeGazeMlp,
    data: dict[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    k_shot: int,
    prototypes: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Reproduce WebEyeTrack's MAML inner-loop + Affine transform on K support samples/zone."""
    support_indices = []
    train_idx = np.flatnonzero(train_mask)
    for z_id in range(len(GAZE_ZONES)):
        z_indices = train_idx[data["gaze_zone_id"][train_idx] == z_id]
        if len(z_indices) > 0:
            pick = np.linspace(0, len(z_indices) - 1, min(k_shot, len(z_indices))).astype(int)
            support_indices.extend(z_indices[pick].tolist())
    support_idx = np.array(support_indices, dtype=np.int64)
    val_idx = np.flatnonzero(val_mask)

    mlp = copy.deepcopy(base_mlp).to(device).train()
    emb_s = torch.from_numpy(data["blazegaze_emb"][support_idx]).to(device)
    head_s = torch.from_numpy(data["head_vector"][support_idx]).to(device)
    orig_s = torch.from_numpy(data["face_origin_3d"][support_idx]).to(device)
    target_pog_s = torch.from_numpy(data["canonical_pog"][support_idx]).to(device)

    with torch.no_grad():
        init_preds = mlp(emb_s, head_s, orig_s).cpu().numpy()
    affine = compute_affine_2x3(init_preds, data["canonical_pog"][support_idx])
    affine_t = torch.from_numpy(affine.T).to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, betas=(0.85, 0.9))
    for _ in range(25):
        optimizer.zero_grad()
        raw_pred = mlp(emb_s, head_s, orig_s)
        homog = torch.cat([raw_pred, torch.ones_like(raw_pred[:, :1])], dim=1)
        aligned_pred = homog @ affine_t
        loss = F.smooth_l1_loss(aligned_pred, target_pog_s)
        loss.backward()
        optimizer.step()

    mlp.eval()
    with torch.no_grad():
        raw_s = mlp(emb_s, head_s, orig_s)
        homog_s = torch.cat([raw_s, torch.ones_like(raw_s[:, :1])], dim=1)
        adapted_s = (homog_s @ affine_t).cpu().numpy()
        adapted_prototypes = prototypes.copy()
        for z_id in range(len(GAZE_ZONES)):
            m = data["gaze_zone_id"][support_idx] == z_id
            if np.any(m):
                adapted_prototypes[z_id] = adapted_s[m].mean(axis=0)

        emb_v = torch.from_numpy(data["blazegaze_emb"][val_idx]).to(device)
        head_v = torch.from_numpy(data["head_vector"][val_idx]).to(device)
        orig_v = torch.from_numpy(data["face_origin_3d"][val_idx]).to(device)
        raw_v = mlp(emb_v, head_v, orig_v)
        homog_v = torch.cat([raw_v, torch.ones_like(raw_v[:, :1])], dim=1)
        pog_pred_v = (homog_v @ affine_t).cpu().numpy()

    zone_pred_v = nearest_zone_from_pog(pog_pred_v, adapted_prototypes)
    return evaluate_predictions(
        data["gaze_zone_id"][val_idx],
        zone_pred_v,
        data["canonical_pog"][val_idx],
        pog_pred_v,
    )


def build_disjoint_windows(
    indices: np.ndarray, window_length: int
) -> list[np.ndarray]:
    """Group contiguous or zone-local indices into short causal windows so GRU resets between segments."""
    if len(indices) == 0:
        return []
    breaks = np.where(np.diff(indices) > 3)[0] + 1
    segments = np.split(indices, breaks)
    windows: list[np.ndarray] = []
    for seg in segments:
        if len(seg) <= window_length:
            windows.append(seg)
        else:
            for start in range(0, len(seg), window_length):
                chunk = seg[start : start + window_length]
                if len(chunk) > 0:
                    windows.append(chunk)
    return windows


def forward_windows(
    model: BlazeGazeDmdAdapter,
    data: dict[str, np.ndarray],
    windows: list[np.ndarray],
    device: torch.device,
    use_visual: bool = True,
    use_pose: bool = True,
) -> tuple[Tensor, Tensor, Tensor, np.ndarray]:
    zone_list, pog_list, blink_list, flat_idx = [], [], [], []
    for win in windows:
        flat_idx.append(win)

        def t_win(key: str, active: bool = True) -> Tensor:
            arr = data[key][win].astype(np.float32)
            t = torch.from_numpy(arr).unsqueeze(0).to(device)
            return t if active else torch.zeros_like(t)

        out = model(
            t_win("blazegaze_emb", use_visual),
            t_win("blazegaze_hidden", use_visual),
            t_win("blazegaze_pog", use_visual),
            t_win("head_vector", use_pose),
            t_win("face_origin_3d", use_pose),
            t_win("geometry", True),
        )
        zone_list.append(out["zone_logits"].squeeze(0))
        pog_list.append(out["pog"].squeeze(0))
        blink_list.append(out["blink_logits"].squeeze(0))

    return (
        torch.cat(zone_list, dim=0),
        torch.cat(pog_list, dim=0),
        torch.cat(blink_list, dim=0),
        np.concatenate(flat_idx, axis=0),
    )


def train_adapter(
    data: dict[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    window_length: int,
    epochs: int,
    lr: float,
    device: torch.device,
    use_visual: bool = True,
    use_pose: bool = True,
) -> tuple[BlazeGazeDmdAdapter, dict[str, Any]]:
    model = BlazeGazeDmdAdapter(num_zones=len(GAZE_ZONES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_windows = build_disjoint_windows(np.flatnonzero(train_mask), window_length)
    val_windows = build_disjoint_windows(np.flatnonzero(val_mask), window_length)

    # Disjoint temporal split for blink evaluation (first 70% valid frames train, last 30% val)
    valid_indices = np.flatnonzero(data["valid"])
    n_blink_train = int(len(valid_indices) * 0.70)
    blink_train_windows = build_disjoint_windows(valid_indices[:n_blink_train], window_length)
    blink_val_windows = build_disjoint_windows(valid_indices[n_blink_train:], window_length)

    zone_target_all = torch.from_numpy(data["gaze_zone_id"]).long().to(device)
    pog_target_all = torch.from_numpy(data["canonical_pog"]).float().to(device)
    blink_target_all = torch.from_numpy(data["blinking"].astype(np.int64)).long().to(device)

    train_zones = data["gaze_zone_id"][train_mask]
    counts = np.bincount(train_zones, minlength=len(GAZE_ZONES)).astype(np.float32)
    weights = counts.sum() / (len(GAZE_ZONES) * np.maximum(counts, 1.0))
    zone_weights = torch.from_numpy(np.clip(weights, 0.5, 3.0)).to(device)

    blink_counts = np.bincount(
        data["blinking"][valid_indices[:n_blink_train]].astype(np.int64), minlength=2
    ).astype(np.float32)
    blink_weights = torch.from_numpy(
        np.clip(blink_counts.sum() / (2.0 * np.maximum(blink_counts, 1.0)), 0.5, 6.0)
    ).to(device)

    best_score = -1.0
    best_state = None
    best_metrics: dict[str, Any] = {}

    for _ in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z_logits, p_pred, _, idx_t = forward_windows(
            model, data, train_windows, device, use_visual=use_visual, use_pose=use_pose
        )
        _, _, b_logits, idx_bt = forward_windows(
            model, data, blink_train_windows, device, use_visual=use_visual, use_pose=use_pose
        )

        loss_zone = F.cross_entropy(z_logits, zone_target_all[idx_t], weight=zone_weights)
        loss_pog = F.smooth_l1_loss(p_pred, pog_target_all[idx_t])
        loss_blink = F.cross_entropy(b_logits, blink_target_all[idx_bt], weight=blink_weights)
        loss = loss_zone + 1.5 * loss_pog + 0.5 * loss_blink
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_z_logits, val_p_pred, _, idx_v = forward_windows(
                model, data, val_windows, device, use_visual=use_visual, use_pose=use_pose
            )
            _, _, val_b_logits, idx_bv = forward_windows(
                model, data, blink_val_windows, device, use_visual=use_visual, use_pose=use_pose
            )
            val_z_pred = val_z_logits.argmax(dim=-1).cpu().numpy()
            val_p_np = val_p_pred.cpu().numpy()
            val_b_pred = val_b_logits.argmax(dim=-1).cpu().numpy()

            metrics = evaluate_predictions(
                data["gaze_zone_id"][idx_v],
                val_z_pred,
                data["canonical_pog"][idx_v],
                val_p_np,
            )
            metrics["blink_val_macro_f1"] = round(
                macro_f1(data["blinking"][idx_bv].astype(np.int64), val_b_pred, 2), 4
            )
            score = (
                metrics["zone_9_macro_f1"]
                + 0.5 * metrics["attention_3_macro_f1"]
                + 0.2 * metrics["blink_val_macro_f1"]
                - 0.2 * metrics["pog_norm_l2_error"]
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_npz_records(args.cache)
    train_mask, val_mask = temporal_zone_split(
        data["valid"],
        data["gaze_zone_id"],
        data["blinking"],
        train_ratio=args.train_ratio,
        buffer_ratio=args.buffer_ratio,
    )

    prototypes = np.array(
        [GAZE_ZONE_CANONICAL_POG[z] for z in GAZE_ZONES], dtype=np.float32
    )

    # 1. Zero-shot Pretrained BlazeGaze evaluation
    val_idx = np.flatnonzero(val_mask)
    zs_zone_pred = nearest_zone_from_pog(data["blazegaze_pog"][val_idx], prototypes)
    zeroshot_metrics = evaluate_predictions(
        data["gaze_zone_id"][val_idx],
        zs_zone_pred,
        data["canonical_pog"][val_idx],
        data["blazegaze_pog"][val_idx],
    )

    # 2. WebEyeTrack MAML + Affine Few-Shot adaptation (K shots per zone)
    blazegaze = BlazeGazePyTorch().to(device).eval()
    blazegaze.load_keras_archive(args.keras_weights)
    fewshot_metrics = run_webeyetrack_maml_few_shot(
        base_mlp=blazegaze.gaze_mlp,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        k_shot=args.few_shot_k,
        prototypes=prototypes,
        device=device,
    )

    # 3. Ablation A: BlazeGaze Visual Features Only (no 3D head pose)
    _, visual_only_metrics = train_adapter(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        window_length=args.window_length,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        use_visual=True,
        use_pose=False,
    )

    # 4. Ablation B: 3D Head Pose + Geometry Only (no BlazeGaze visual features)
    _, pose_only_metrics = train_adapter(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        window_length=args.window_length,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        use_visual=False,
        use_pose=True,
    )

    # 5. Full Fine-Tuned BlazeGazeDmdAdapter (BlazeGaze Visual + 3D Pose + Eye Geometry + FiLM + Causal GRU)
    adapter_model, full_adapter_metrics = train_adapter(
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        window_length=args.window_length,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        use_visual=True,
        use_pose=True,
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "best_blazegaze_dmd_adapter.pt"
    torch.save(
        {
            "adapter_state_dict": adapter_model.state_dict(),
            "gaze_zones": list(GAZE_ZONES),
            "canonical_pog": GAZE_ZONE_CANONICAL_POG,
            "attention_names": list(ATTENTION_NAMES),
        },
        checkpoint_path,
    )

    summary: dict[str, Any] = {
        "dataset_summary": {
            "total_cached_frames": int(len(data["valid"])),
            "valid_face_frames": int(data["valid"].sum()),
            "valid_gaze_non_blink_frames": int(
                (data["valid"] & (data["gaze_zone_id"] >= 0) & (~data["blinking"])).sum()
            ),
            "train_frames": int(train_mask.sum()),
            "holdout_val_frames": int(val_mask.sum()),
            "blink_frames": int(data["blinking"].sum()),
        },
        "1_zeroshot_blazegaze": zeroshot_metrics,
        f"2_webeyetrack_maml_{args.few_shot_k}shot": fewshot_metrics,
        "3_ablation_blazegaze_visual_only": visual_only_metrics,
        "4_ablation_pose_geometry_only": pose_only_metrics,
        "5_full_finetuned_blazegaze_dmd_adapter": full_adapter_metrics,
        "saved_checkpoint": str(checkpoint_path),
    }

    if args.test_cache is not None and args.test_cache.exists():
        test_data = load_npz_records(args.test_cache)
        test_usable = np.flatnonzero(
            test_data["valid"]
            & (test_data["gaze_zone_id"] >= 0)
            & (~test_data["blinking"])
        )
        test_windows = build_disjoint_windows(test_usable, args.window_length)
        adapter_model.eval()
        with torch.no_grad():
            t_z_logits, t_p_pred, _, idx_test = forward_windows(
                adapter_model, test_data, test_windows, device, use_visual=True, use_pose=True
            )
            summary["external_test_evaluation"] = evaluate_predictions(
                test_data["gaze_zone_id"][idx_test],
                t_z_logits.argmax(dim=-1).cpu().numpy(),
                test_data["canonical_pog"][idx_test],
                t_p_pred.cpu().numpy(),
            )

    (args.run_dir / "result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
