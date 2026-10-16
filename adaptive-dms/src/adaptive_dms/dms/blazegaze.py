"""PyTorch port of WebEyeTrack's BlazeGaze with direct .keras weight loading and MediaPipe preprocessing."""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# MediaPipe landmark constants matching WebEyeTrack
LEFTMOST_LANDMARK = 356
RIGHTMOST_LANDMARK = 127
LEFT_IRIS_LANDMARKS = (473, 475, 474, 477, 476)
RIGHT_IRIS_LANDMARKS = (468, 470, 469, 472, 471)
LEFT_EYE_HORIZONTAL_LANDMARKS = (362, 263)
RIGHT_EYE_HORIZONTAL_LANDMARKS = (33, 133)
LEFT_EYE_EAR_LANDMARKS = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_EAR_LANDMARKS = (133, 158, 160, 33, 144, 153)
AVERAGE_IRIS_SIZE_CM = 1.2
VERTICAL_FOV_DEGREES = 60.0
NEAR = 1.0
FAR = 10000.0


def compute_ear(facial_landmarks_px: np.ndarray, side: str = "left") -> float:
    indices = (
        LEFT_EYE_EAR_LANDMARKS if side == "left" else RIGHT_EYE_EAR_LANDMARKS
    )
    p1, p2, p3, p4, p5, p6 = (facial_landmarks_px[i, :2] for i in indices)
    denom = 2.0 * float(np.linalg.norm(p1 - p4))
    if denom < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / denom)


def obtain_eyepatch(
    frame_rgb: np.ndarray,
    face_landmarks_px: np.ndarray,
    face_padding_coefs: tuple[float, float] = (0.4, 0.2),
    face_crop_size: int = 512,
    dst_img_size: tuple[int, int] = (512, 128),
) -> np.ndarray:
    """Warp the eye strip to (128, 512, 3) RGB using WebEyeTrack's 4-point homography."""
    lefttop = face_landmarks_px[103, :2]
    leftbottom = face_landmarks_px[150, :2]
    righttop = face_landmarks_px[332, :2]
    rightbottom = face_landmarks_px[379, :2]
    center = face_landmarks_px[4, :2]

    src_pts = np.array(
        [lefttop, leftbottom, rightbottom, righttop], dtype=np.float32
    )
    src_direction = src_pts - center
    src_pts = src_pts + np.array(face_padding_coefs, dtype=np.float32) * src_direction

    dst_pts = np.array(
        [
            [0, 0],
            [0, face_crop_size],
            [face_crop_size, face_crop_size],
            [face_crop_size, 0],
        ],
        dtype=np.float32,
    )

    matrix, _ = cv2.findHomography(src_pts, dst_pts)
    if matrix is None:
        raise ValueError("Homography estimation failed for eye patch")
    warped_face = cv2.warpPerspective(
        frame_rgb, matrix, (face_crop_size, face_crop_size)
    )

    homog_lms = np.vstack(
        (face_landmarks_px[:, :2].T, np.ones((1, face_landmarks_px.shape[0])))
    )
    warped_lms = matrix @ homog_lms
    warped_lms = (warped_lms[:2, :] / np.clip(warped_lms[2, :], 1e-6, None)).T.astype(
        np.int32
    )

    top_y = int(np.clip(warped_lms[151, 1], 0, face_crop_size - 2))
    bottom_y = int(np.clip(warped_lms[195, 1], top_y + 2, face_crop_size))
    eyes_patch = warped_face[top_y:bottom_y, :]
    return cv2.resize(eyes_patch, dst_img_size)


def rotation_matrix_to_euler_angles(r_mat: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(r_mat[0, 0] ** 2 + r_mat[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(float(r_mat[2, 1]), float(r_mat[2, 2]))
        y = math.atan2(float(-r_mat[2, 0]), sy)
        z = math.atan2(float(r_mat[1, 0]), float(r_mat[0, 0]))
    else:
        x = math.atan2(float(-r_mat[1, 2]), float(r_mat[1, 1]))
        y = math.atan2(float(-r_mat[2, 0]), sy)
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def euler_angles_to_rotation_matrix(
    pitch: float, yaw: float, roll: float
) -> np.ndarray:
    rx, ry, rz = np.radians([pitch, yaw, roll])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return r_z @ (r_y @ r_x)


def pitch_yaw_roll_to_gaze_vector(
    pitch: float, yaw: float, roll: float = 0.0
) -> np.ndarray:
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    x = np.sin(yaw_rad) * np.cos(pitch_rad)
    y = -np.sin(pitch_rad)
    z = -np.cos(yaw_rad) * np.cos(pitch_rad)
    vec = np.array([x, y, z], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def get_head_vector(face_rt: np.ndarray) -> np.ndarray:
    pitch, yaw, roll = rotation_matrix_to_euler_angles(face_rt[:3, :3])
    return pitch_yaw_roll_to_gaze_vector(-yaw, pitch, roll)


def estimate_face_width(facial_landmarks_px: np.ndarray) -> float:
    iris_dist = []
    for indices in (LEFT_IRIS_LANDMARKS, RIGHT_IRIS_LANDMARKS):
        leftmost = facial_landmarks_px[indices[4], :2]
        rightmost = facial_landmarks_px[indices[2], :2]
        iris_dist.append(float(np.linalg.norm(leftmost - rightmost)))
    avg_iris = float(np.mean(iris_dist))
    face_width_px = float(
        np.linalg.norm(
            facial_landmarks_px[LEFTMOST_LANDMARK, :2]
            - facial_landmarks_px[RIGHTMOST_LANDMARK, :2]
        )
    )
    ratio = max(avg_iris / max(face_width_px, 1e-5), 1e-5)
    return float(AVERAGE_IRIS_SIZE_CM / ratio)


def create_perspective_matrix(aspect_ratio: float) -> np.ndarray:
    k_degrees_to_radians = np.pi / 180.0
    f = 1.0 / np.tan(k_degrees_to_radians * VERTICAL_FOV_DEGREES / 2.0)
    denom = NEAR - FAR
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect_ratio
    mat[1, 1] = f
    mat[2, 2] = (NEAR + FAR) / denom
    mat[2, 3] = -1.0
    mat[3, 2] = (2.0 * FAR * NEAR) / denom
    return mat


def compute_face_origin_3d(
    face_landmarks_norm: np.ndarray,
    face_rt: np.ndarray,
    width: int,
    height: int,
    face_width_cm: float | None = None,
    intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Compute WebEyeTrack's 3D face origin in cm (vectorized equivalent)."""
    landmarks_px = face_landmarks_norm[:, :2] * np.array(
        [width, height], dtype=np.float32
    )
    if face_width_cm is None:
        face_width_cm = estimate_face_width(landmarks_px)
    if intrinsics is None:
        focal = float(width)
        intrinsics = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    perspective = create_perspective_matrix(aspect_ratio=width / height)
    ndc_x = 2.0 * face_landmarks_norm[:, 0] - 1.0
    ndc_y = 1.0 - 2.0 * face_landmarks_norm[:, 1]
    z_eye = face_landmarks_norm[:, 2]
    x_eye = -(ndc_x * z_eye) / perspective[0, 0]
    y_eye = -(ndc_y * z_eye) / perspective[1, 1]
    relative_mesh = np.stack([x_eye, y_eye, z_eye], axis=-1).astype(np.float32)

    nose = relative_mesh[4]
    relative_mesh = (relative_mesh - nose) * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    span = float(
        np.linalg.norm(
            relative_mesh[LEFTMOST_LANDMARK] - relative_mesh[RIGHTMOST_LANDMARK]
        )
    )
    relative_mesh /= max(span, 1e-6)
    canonical_pts_3d = relative_mesh * float(face_width_cm)

    face_r = face_rt[:3, :3].copy()
    pitch, yaw, roll = rotation_matrix_to_euler_angles(np.linalg.inv(face_r))
    face_r = euler_angles_to_rotation_matrix(-yaw, pitch, roll)
    canonical_pts_3d = canonical_pts_3d @ np.linalg.inv(face_r).T
    scales = np.linalg.norm(face_r, axis=0)
    face_r = face_r / max(float(scales.mean()), 1e-6)

    initial_z = 60.0
    init_transform = np.eye(4, dtype=np.float32)
    init_transform[:3, :3] = face_r
    init_transform[:3, 3] = np.array([0.0, 0.0, initial_z], dtype=np.float32)

    def project_2d(pts_3d: np.ndarray, transform: np.ndarray) -> np.ndarray:
        homog = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)])
        cam = (transform @ homog.T).T[:, :3]
        uvw = (intrinsics @ cam.T).T
        return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)

    proj_2d = project_2d(canonical_pts_3d, init_transform)
    shift_2d = landmarks_px.mean(axis=0) - proj_2d.mean(axis=0)
    shift_3d = np.array(
        [
            shift_2d[0] * initial_z / intrinsics[0, 0],
            shift_2d[1] * initial_z / intrinsics[1, 1],
            0.0,
        ],
        dtype=np.float32,
    )
    final_transform = init_transform.copy()
    final_transform[:3, 3] += shift_3d
    first_translation = final_transform[:3, 3].copy()

    for _ in range(10):
        current_proj = project_2d(canonical_pts_3d, final_transform)
        c_proj = current_proj.mean(axis=0)
        c_det = landmarks_px.mean(axis=0)
        r_proj = np.linalg.norm(current_proj - c_proj, axis=1)
        r_det = np.linalg.norm(landmarks_px - c_det, axis=1)
        valid = r_det > 1e-4
        ratios = r_proj[valid] / r_det[valid]
        scale_factor = float(np.median(ratios)) if np.any(valid) else 1.0
        old_z = float(final_transform[2, 3])
        raw_new_z = old_z * scale_factor
        new_z = old_z + 0.5 * (raw_new_z - old_z)
        if abs(new_z - old_z) < 0.25:
            break
        final_transform[0, 3] = first_translation[0] * (new_z / initial_z)
        final_transform[1, 3] = first_translation[1] * (new_z / initial_z)
        final_transform[2, 3] = new_z

    homog = np.hstack(
        [canonical_pts_3d, np.ones((canonical_pts_3d.shape[0], 1), dtype=np.float32)]
    )
    final_face_pts = (final_transform @ homog.T).T[:, :3]
    left_origin = final_face_pts[list(LEFT_EYE_HORIZONTAL_LANDMARKS)].mean(axis=0)
    right_origin = final_face_pts[list(RIGHT_EYE_HORIZONTAL_LANDMARKS)].mean(axis=0)
    face_origin_3d = ((left_origin + right_origin) / 2.0).astype(np.float32)
    return face_origin_3d, float(face_width_cm)


class _KerasSameConv2d(nn.Module):
    """Conv2d / DepthwiseConv2d with exact TensorFlow 'SAME' padding semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        if self.stride == 1:
            pad_h = max(self.kernel_size - 1, 0)
            pad_w = max(self.kernel_size - 1, 0)
        else:
            out_h = (h + self.stride - 1) // self.stride
            out_w = (w + self.stride - 1) // self.stride
            pad_h = max((out_h - 1) * self.stride + self.kernel_size - h, 0)
            pad_w = max((out_w - 1) * self.stride + self.kernel_size - w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class _SingleBlazeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        self.dw = _KerasSameConv2d(
            in_channels, in_channels, kernel_size=5, stride=stride, groups=in_channels
        )
        self.pw = _KerasSameConv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2) if stride == 2 else None
        self.proj = (
            _KerasSameConv2d(in_channels, out_channels, kernel_size=1, stride=1)
            if stride == 2
            else None
        )

    def forward(self, y: Tensor) -> Tensor:
        x = self.pw(self.dw(y))
        if self.stride == 2 and self.pool is not None and self.proj is not None:
            y = self.proj(self.pool(y))
        return F.relu(x + y)


class _DoubleBlazeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.dw1 = _KerasSameConv2d(
            in_channels, in_channels, kernel_size=5, stride=stride, groups=in_channels
        )
        self.pw1 = _KerasSameConv2d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.dw2 = _KerasSameConv2d(
            mid_channels, mid_channels, kernel_size=5, stride=1, groups=mid_channels
        )
        self.pw2 = _KerasSameConv2d(mid_channels, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2) if stride == 2 else None
        self.proj = (
            _KerasSameConv2d(in_channels, out_channels, kernel_size=1, stride=1)
            if stride == 2
            else None
        )

    def forward(self, y: Tensor) -> Tensor:
        x = F.relu(self.pw1(self.dw1(y)))
        x = self.pw2(self.dw2(x))
        if self.stride == 2 and self.pool is not None and self.proj is not None:
            y = self.proj(self.pool(y))
        return F.relu(x + y)


class BlazeGazeEncoder(nn.Module):
    """Exact PyTorch equivalent of WebEyeTrack's cnn_encoder (outputs 512-d embedding)."""

    def __init__(self) -> None:
        super().__init__()
        self.first_conv = _KerasSameConv2d(3, 24, kernel_size=5, stride=2)
        self.single_1 = _SingleBlazeBlock(24, 24, stride=1)
        self.single_2 = _SingleBlazeBlock(24, 24, stride=1)
        self.single_3 = _SingleBlazeBlock(24, 48, stride=2)
        self.single_4 = _SingleBlazeBlock(48, 48, stride=1)
        self.single_5 = _SingleBlazeBlock(48, 48, stride=1)
        self.double_1 = _DoubleBlazeBlock(48, 24, 96, stride=2)
        self.double_2 = _DoubleBlazeBlock(96, 24, 96, stride=1)
        self.double_3 = _DoubleBlazeBlock(96, 24, 96, stride=1)
        self.double_4 = _DoubleBlazeBlock(96, 24, 96, stride=2)
        self.double_5 = _DoubleBlazeBlock(96, 24, 96, stride=1)
        self.double_6 = _DoubleBlazeBlock(96, 24, 96, stride=1)
        self.squeeze_1 = _KerasSameConv2d(96, 64, kernel_size=3, stride=2)
        self.bn_1 = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.squeeze_2 = _KerasSameConv2d(64, 32, kernel_size=3, stride=2)
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)

    def forward(self, eye_patch: Tensor) -> Tensor:
        """Accepts (B, 3, 128, 512) or (B, 128, 512, 3) float tensor in [0, 1]."""
        if eye_patch.ndim != 4:
            raise ValueError(f"Expected 4D eye_patch, got {eye_patch.shape}")
        if eye_patch.shape[-1] == 3 and eye_patch.shape[1] != 3:
            x = eye_patch.permute(0, 3, 1, 2).contiguous()
        else:
            x = eye_patch
        x = F.relu(self.first_conv(x))
        x = self.single_1(x)
        x = self.single_2(x)
        x = self.single_3(x)
        x = self.single_4(x)
        x = self.single_5(x)
        x = self.double_1(x)
        x = self.double_2(x)
        x = self.double_3(x)
        x = self.double_4(x)
        x = self.double_5(x)
        x = self.double_6(x)
        x = self.bn_1(F.relu(self.squeeze_1(x)))
        x = self.bn_2(F.relu(self.squeeze_2(x)))
        # Flatten in Keras NHWC order so index order matches gaze_mlp's 518-d Dense(16)
        return x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1)


class BlazeGazeMlp(nn.Module):
    """Exact PyTorch equivalent of WebEyeTrack's MAML gaze_mlp (518 -> 16 -> 16 -> 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(518, 16)
        self.dense_2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 2)

    def forward_features(
        self, embedding: Tensor, head_vector: Tensor, face_origin_3d: Tensor
    ) -> tuple[Tensor, Tensor]:
        concat = torch.cat([embedding, head_vector, face_origin_3d], dim=-1)
        h1 = F.relu(self.dense_1(concat))
        h2 = F.relu(self.dense_2(h1))
        pog = self.out(h2)
        return pog, h2

    def forward(
        self, embedding: Tensor, head_vector: Tensor, face_origin_3d: Tensor
    ) -> Tensor:
        pog, _ = self.forward_features(embedding, head_vector, face_origin_3d)
        return pog


class BlazeGazePyTorch(nn.Module):
    """Complete BlazeGaze model with pretrained weight loader from .keras archive."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = BlazeGazeEncoder()
        self.gaze_mlp = BlazeGazeMlp()

    def forward(
        self, eye_patch: Tensor, head_vector: Tensor, face_origin_3d: Tensor
    ) -> dict[str, Tensor]:
        embedding = self.encoder(eye_patch)
        norm_pog, hidden = self.gaze_mlp.forward_features(
            embedding, head_vector, face_origin_3d
        )
        return {
            "embedding": embedding,
            "hidden": hidden,
            "norm_pog": norm_pog,
        }

    def load_keras_archive(self, keras_path: Path | str) -> None:
        path = Path(keras_path)
        with zipfile.ZipFile(path, "r") as zf:
            bio = io.BytesIO(zf.read("model.weights.h5"))

        with h5py.File(bio, "r") as hf:
            enc_grp = hf["layers/functional/layers"]
            mlp_grp = hf["layers/functional_1/layers"]

            def load_conv(mod: _KerasSameConv2d, key: str, depthwise: bool = False) -> None:
                w = np.array(enc_grp[f"{key}/vars/0"], dtype=np.float32)
                b = np.array(enc_grp[f"{key}/vars/1"], dtype=np.float32)
                if depthwise:
                    # Keras depthwise: (H, W, C, 1) -> PyTorch: (C, 1, H, W)
                    w_pt = np.transpose(w, (2, 3, 0, 1))
                else:
                    # Keras conv2d: (H, W, In, Out) -> PyTorch: (Out, In, H, W)
                    w_pt = np.transpose(w, (3, 2, 0, 1))
                mod.conv.weight.data.copy_(torch.from_numpy(w_pt))
                mod.conv.bias.data.copy_(torch.from_numpy(b))

            def load_bn(bn: nn.BatchNorm2d, key: str) -> None:
                gamma = np.array(enc_grp[f"{key}/vars/0"], dtype=np.float32)
                beta = np.array(enc_grp[f"{key}/vars/1"], dtype=np.float32)
                mean = np.array(enc_grp[f"{key}/vars/2"], dtype=np.float32)
                var = np.array(enc_grp[f"{key}/vars/3"], dtype=np.float32)
                bn.weight.data.copy_(torch.from_numpy(gamma))
                bn.bias.data.copy_(torch.from_numpy(beta))
                bn.running_mean.copy_(torch.from_numpy(mean))
                bn.running_var.copy_(torch.from_numpy(var))

            enc = self.encoder
            load_conv(enc.first_conv, "conv2d")

            load_conv(enc.single_1.dw, "depthwise_conv2d", depthwise=True)
            load_conv(enc.single_1.pw, "conv2d_1")

            load_conv(enc.single_2.dw, "depthwise_conv2d_1", depthwise=True)
            load_conv(enc.single_2.pw, "conv2d_2")

            load_conv(enc.single_3.dw, "depthwise_conv2d_2", depthwise=True)
            load_conv(enc.single_3.pw, "conv2d_3")
            assert enc.single_3.proj is not None
            load_conv(enc.single_3.proj, "conv2d_4")

            load_conv(enc.single_4.dw, "depthwise_conv2d_3", depthwise=True)
            load_conv(enc.single_4.pw, "conv2d_5")

            load_conv(enc.single_5.dw, "depthwise_conv2d_4", depthwise=True)
            load_conv(enc.single_5.pw, "conv2d_6")

            load_conv(enc.double_1.dw1, "depthwise_conv2d_5", depthwise=True)
            load_conv(enc.double_1.pw1, "conv2d_7")
            load_conv(enc.double_1.dw2, "depthwise_conv2d_6", depthwise=True)
            load_conv(enc.double_1.pw2, "conv2d_8")
            assert enc.double_1.proj is not None
            load_conv(enc.double_1.proj, "conv2d_9")

            load_conv(enc.double_2.dw1, "depthwise_conv2d_7", depthwise=True)
            load_conv(enc.double_2.pw1, "conv2d_10")
            load_conv(enc.double_2.dw2, "depthwise_conv2d_8", depthwise=True)
            load_conv(enc.double_2.pw2, "conv2d_11")

            load_conv(enc.double_3.dw1, "depthwise_conv2d_9", depthwise=True)
            load_conv(enc.double_3.pw1, "conv2d_12")
            load_conv(enc.double_3.dw2, "depthwise_conv2d_10", depthwise=True)
            load_conv(enc.double_3.pw2, "conv2d_13")

            load_conv(enc.double_4.dw1, "depthwise_conv2d_11", depthwise=True)
            load_conv(enc.double_4.pw1, "conv2d_14")
            load_conv(enc.double_4.dw2, "depthwise_conv2d_12", depthwise=True)
            load_conv(enc.double_4.pw2, "conv2d_15")
            assert enc.double_4.proj is not None
            load_conv(enc.double_4.proj, "conv2d_16")

            load_conv(enc.double_5.dw1, "depthwise_conv2d_13", depthwise=True)
            load_conv(enc.double_5.pw1, "conv2d_17")
            load_conv(enc.double_5.dw2, "depthwise_conv2d_14", depthwise=True)
            load_conv(enc.double_5.pw2, "conv2d_18")

            load_conv(enc.double_6.dw1, "depthwise_conv2d_15", depthwise=True)
            load_conv(enc.double_6.pw1, "conv2d_19")
            load_conv(enc.double_6.dw2, "depthwise_conv2d_16", depthwise=True)
            load_conv(enc.double_6.pw2, "conv2d_20")

            load_conv(enc.squeeze_1, "conv2d_21")
            load_bn(enc.bn_1, "batch_normalization")
            load_conv(enc.squeeze_2, "conv2d_22")
            load_bn(enc.bn_2, "batch_normalization_1")

            for layer, key in (
                (self.gaze_mlp.dense_1, "dense"),
                (self.gaze_mlp.dense_2, "dense_1"),
                (self.gaze_mlp.out, "dense_2"),
            ):
                w = np.array(mlp_grp[f"{key}/vars/0"], dtype=np.float32)
                b = np.array(mlp_grp[f"{key}/vars/1"], dtype=np.float32)
                layer.weight.data.copy_(torch.from_numpy(w.T))
                layer.bias.data.copy_(torch.from_numpy(b))


class BlazeGazeDmdAdapter(nn.Module):
    """Few-shot & fine-tunable head on top of frozen BlazeGaze for DMD Gaze Zones, PoG & Blinks.

    Combines:
    - Frozen 512-d BlazeGaze visual eye embedding (LayerNorm-normalized)
    - Pretrained 16-d MAML hidden representation & 2-d zero-shot norm_pog
    - 3D head_vector (3-d) + normalized face_origin_3d (3-d) + DMS eye/head geometry (10-d)
    """

    def __init__(
        self,
        num_zones: int = 9,
        geometry_dim: int = 10,
        hidden_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.emb_norm = nn.LayerNorm(512)
        self.emb_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # Visual MAML features: pretrained_pog(2) + pretrained_hidden(16) = 18
        self.maml_norm = nn.LayerNorm(18)
        # Pose & geometry features: head_vector(3) + face_origin_3d(3) + geometry(10) = 16
        pose_dim = 3 + 3 + geometry_dim
        self.pose_norm = nn.LayerNorm(pose_dim)

        aux_dim = 18 + pose_dim
        self.film = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.temporal = nn.GRU(hidden_dim + aux_dim, hidden_dim, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.zone_head = nn.Linear(hidden_dim, num_zones)
        self.pog_head = nn.Linear(hidden_dim, 2)

        # Dedicated blink head combining temporal state and causal EAR dynamics
        self.blink_ear_mlp = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
        )
        self.blink_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        embedding: Tensor,
        pretrained_hidden: Tensor,
        pretrained_pog: Tensor,
        head_vector: Tensor,
        face_origin_3d: Tensor,
        geometry: Tensor,
    ) -> dict[str, Tensor]:
        is_seq = embedding.ndim == 3
        if not is_seq:
            embedding = embedding.unsqueeze(1)
            pretrained_hidden = pretrained_hidden.unsqueeze(1)
            pretrained_pog = pretrained_pog.unsqueeze(1)
            head_vector = head_vector.unsqueeze(1)
            face_origin_3d = face_origin_3d.unsqueeze(1)
            geometry = geometry.unsqueeze(1)

        maml_feat = self.maml_norm(torch.cat([pretrained_pog, pretrained_hidden], dim=-1))
        scaled_origin = face_origin_3d / 60.0
        pose_feat = self.pose_norm(torch.cat([head_vector, scaled_origin, geometry], dim=-1))
        aux_normed = torch.cat([maml_feat, pose_feat], dim=-1)

        vis = self.emb_proj(self.emb_norm(embedding))
        gamma, beta = self.film(aux_normed).chunk(2, dim=-1)
        modulated = (1.0 + 0.2 * torch.tanh(gamma)) * vis + 0.2 * torch.tanh(beta)
        fused = torch.cat([modulated, aux_normed], dim=-1)
        temporal_out, _ = self.temporal(fused)
        features = self.out_norm(temporal_out + modulated)

        # Causal EAR dynamics: left_ear, right_ear, mean_ear, diff_ear, causal_delta_ear, mar
        ears = geometry[..., :2]
        mean_ear = ears.mean(dim=-1, keepdim=True)
        diff_ear = (ears[..., :1] - ears[..., 1:2]).abs()
        prev_ear = torch.cat([mean_ear[:, :1], mean_ear[:, :-1]], dim=1)
        delta_ear = mean_ear - prev_ear
        mar = geometry[..., 2:3]
        blink_Scroll = torch.cat([ears, mean_ear, diff_ear, delta_ear, mar], dim=-1)
        blink_logits = self.blink_head(features) + self.blink_ear_mlp(blink_Scroll)

        result = {
            "zone_logits": self.zone_head(features),
            "pog": pretrained_pog + 0.5 * torch.tanh(self.pog_head(features)),
            "blink_logits": blink_logits,
        }
        if not is_seq:
            result = {k: v.squeeze(1) for k, v in result.items()}
        return result

