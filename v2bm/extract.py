"""Monocular video or still image -> 3D human skeleton motion extraction.

Extracts 3D human motion using Google MediaPipe Pose (Apache-2.0) with zero non-commercial
or copyleft dependencies (completely free of non-commercial research models and PyTorch).

Produces skeleton_motion.npz containing 24 Body24 joints per frame in a Y-up, metre-scaled
right-handed coordinate frame (+Y up, +X anatomical right, -Z chest-forward) that meshes
seamlessly with downstream retargeting (retarget.py) via R_BODY_ALIGN.

Biomechanical stabilisation and adaptive filtering:
1. One-Euro (1€) adaptive frequency filtering on 3D landmarks (specialised Z-depth suppression).
2. Left/right bilateral limb swap prevention using lateral shoulder/hip width hysteresis.
3. Rigid bone length hierarchical projection enforcing temporal median rest lengths.
4. Anatomical hinge range-of-motion clamping (posterior knee bend 0–150°, elbow extension).
5. Camera-space root trajectory estimation and foot ground pinning to Y = 0 during stance phases.

Supports both video (--video) and single still image (--image) inputs.
All code, comments, docstrings, and documentation strictly use British English conventions.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

# Ensure Windows stdout/stderr handles Unicode symbols cleanly without charmap crashes
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

from . import checks, paths
from .stabilize_kp2d import (
    BiomechanicalStabiliser,
    OneEuroFilter,
    OneEuroFilter3D,
    estimate_root_trajectory_and_pinning,
)

# Standard 24 Body24 joint order consumed by downstream retargeting modules.
BODY24_JOINT_NAMES = [
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
    "left_hand",       # 22
    "right_hand",      # 23
]

# For a still photo, repeat the recovered pose across a static hold clip
# so that retargeting, .glb export, and preview rendering display a stable pose.
HOLD_FRAMES = 32
HOLD_FPS = 30.0
HOLD_MAX_SIDE = 1920

# Canonical 52 ARKit blendshapes specified by Apple ARKit & Google MediaPipe FaceLandmarker
CANONICAL_ARKIT_BLENDSHAPES: list[str] = [
    # Brows (5)
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    # Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    # Eyes (14)
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    # Jaw (4)
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    # Mouth (23)
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    # Nose (2)
    "noseSneerLeft",
    "noseSneerRight",
    # Tongue (1)
    "tongueOut",
]


@dataclass
class FaceExtractionFrame:
    """Per-frame facial extraction results."""

    landmarks_3d: np.ndarray  # (478, 3) in unprojected full frame normalised space [0, 1]
    blendshapes: np.ndarray  # (52,) raw blendshapes in [0.0, 1.0] matching CANONICAL_ARKIT_BLENDSHAPES
    transform_matrix: np.ndarray  # (4, 4) rigid head pose matrix in camera space
    gaze_vectors: np.ndarray  # (2, 3) left/right iris offset relative to eye centres
    head_bbox: tuple[int, int, int, int] | None  # (x1, y1, box_w, box_h) in pixels


class MediaPipeFaceExtractor:
    """Google MediaPipe Tasks Vision FaceLandmarker (Apache-2.0) facial extractor.

    Extracts 478 3D landmarks, 52 ARKit blendshapes, and 4x4 head transformation matrix.
    Supports both VIDEO and IMAGE running modes.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        video_mode: bool = False,
        device: str = "cpu",
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
        from .cuda_accel import resolve_device

        resolved_model = model_path or checks.ensure_mediapipe_face_model()
        if not resolved_model.exists():
            raise FileNotFoundError(
                f"MediaPipe face model not found at {resolved_model}"
            )

        self.video_mode = video_mode
        self.device = resolve_device(device)
        mode = RunningMode.VIDEO if video_mode else RunningMode.IMAGE

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=mode,
            num_faces=1,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.mp_image = mp.Image
        self.mp_format = mp.ImageFormat

    def close(self) -> None:
        """Explicitly close the underlying MediaPipe FaceLandmarker task runner."""
        if getattr(self, "landmarker", None) is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def detect_frame(
        self, bgr_frame: np.ndarray, timestamp_ms: int | None = None
    ) -> FaceExtractionFrame | None:
        """Run direct face detection on an entire frame (for portrait framing or pre-cropped images)."""
        if bgr_frame is None or bgr_frame.size == 0 or self.landmarker is None:
            return None
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = self.mp_image(image_format=self.mp_format.SRGB, data=rgb)
        if self.video_mode and timestamp_ms is not None:
            res = self.landmarker.detect_for_video(mp_img, int(timestamp_ms))
        else:
            res = self.landmarker.detect(mp_img)

        return self._unpack(res, bgr_frame.shape[1], bgr_frame.shape[0])

    def detect_cropped_roi(
        self,
        bgr_frame: np.ndarray,
        head_roi: tuple[int, int, int, int],
        timestamp_ms: int | None = None,
    ) -> FaceExtractionFrame | None:
        """Run face detection on a square head ROI crop and unproject landmarks to full frame."""
        if bgr_frame is None or bgr_frame.size == 0 or self.landmarker is None:
            return None

        h, w = bgr_frame.shape[:2]
        v0, v1, v2, v3 = head_roi

        # Parse head_roi: standardise on (x_min, y_min, x_max, y_max)
        # If coordinates define an in-frame bounding box where v2 > v0 and v3 > v1
        # and v0 >= 0 and v1 >= 0 and v2 <= w and v3 <= h, parse as (x_min, y_min, x_max, y_max).
        if v2 > v0 and v3 > v1 and v0 >= 0 and v1 >= 0 and v2 <= w and v3 <= h:
            x1, y1 = int(v0), int(v1)
            bw = max(32, int(v2 - v0))
            bh = max(32, int(v3 - v1))
        else:
            # Fallback for explicit (x1, y1, bw, bh) format with potential negative margin padding
            x1, y1 = int(v0), int(v1)
            bw = max(32, int(v2))
            bh = max(32, int(v3))

        bw = max(32, int(bw))
        bh = max(32, int(bh))

        src_x1, src_x2 = max(0, x1), min(w, x1 + bw)
        src_y1, src_y2 = max(0, y1), min(h, y1 + bh)
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return None

        crop_patch = bgr_frame[src_y1:src_y2, src_x1:src_x2]

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, (x1 + bw) - w)
        pad_bottom = max(0, (y1 + bh) - h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop_patch = cv2.copyMakeBorder(
                crop_patch,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        resized = cv2.resize(crop_patch, (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_img = self.mp_image(image_format=self.mp_format.SRGB, data=rgb)

        if self.video_mode and timestamp_ms is not None:
            res = self.landmarker.detect_for_video(mp_img, int(timestamp_ms))
        else:
            res = self.landmarker.detect(mp_img)

        if not res or not res.face_landmarks:
            return None

        flms = res.face_landmarks[0]
        unprojected = np.zeros((478, 3), dtype=np.float32)
        for i in range(478):
            px = flms[i].x * bw
            py = flms[i].y * bh
            orig_px = x1 + px
            orig_py = y1 + py
            unprojected[i, 0] = orig_px / float(w)
            unprojected[i, 1] = orig_py / float(h)
            unprojected[i, 2] = flms[i].z * (bw / float(w))

        if res.face_blendshapes and len(res.face_blendshapes) > 0:
            scores = {cat.category_name: float(cat.score) for cat in res.face_blendshapes[0]}
            bs = np.array(
                [scores.get(name, 0.0) for name in CANONICAL_ARKIT_BLENDSHAPES],
                dtype=np.float32,
            )
        else:
            bs = np.zeros(52, dtype=np.float32)

        mat = (
            np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
            if res.facial_transformation_matrixes
            else np.eye(4, dtype=np.float32)
        )
        gaze = self._compute_gaze(flms)

        return FaceExtractionFrame(
            landmarks_3d=unprojected,
            blendshapes=bs,
            transform_matrix=mat,
            gaze_vectors=gaze,
            head_bbox=(x1, y1, bw, bh),
        )

    @staticmethod
    def _compute_gaze(flms) -> np.ndarray:
        """Compute relative gaze offset vectors for left and right eyes."""
        if isinstance(flms, np.ndarray):
            l_eye_c = 0.5 * (flms[33] + flms[133])
            l_iris = flms[468]
            r_eye_c = 0.5 * (flms[362] + flms[263])
            r_iris = flms[473]
            return np.stack([l_iris - l_eye_c, r_iris - r_eye_c], axis=0).astype(np.float32)

        l_eye_c = np.array(
            [
                0.5 * (flms[33].x + flms[133].x),
                0.5 * (flms[33].y + flms[133].y),
                0.5 * (flms[33].z + flms[133].z),
            ],
            dtype=np.float32,
        )
        l_iris = np.array([flms[468].x, flms[468].y, flms[468].z], dtype=np.float32)

        r_eye_c = np.array(
            [
                0.5 * (flms[362].x + flms[263].x),
                0.5 * (flms[362].y + flms[263].y),
                0.5 * (flms[362].z + flms[263].z),
            ],
            dtype=np.float32,
        )
        r_iris = np.array([flms[473].x, flms[473].y, flms[473].z], dtype=np.float32)

        return np.stack([l_iris - l_eye_c, r_iris - r_eye_c], axis=0).astype(np.float32)

    def _unpack(self, res, width: int, height: int) -> FaceExtractionFrame | None:
        if not res or not res.face_landmarks:
            return None
        flms = res.face_landmarks[0]
        lms = np.zeros((478, 3), dtype=np.float32)
        for i in range(478):
            lms[i, 0] = flms[i].x
            lms[i, 1] = flms[i].y
            lms[i, 2] = flms[i].z

        if res.face_blendshapes and len(res.face_blendshapes) > 0:
            scores = {cat.category_name: float(cat.score) for cat in res.face_blendshapes[0]}
            bs = np.array(
                [scores.get(name, 0.0) for name in CANONICAL_ARKIT_BLENDSHAPES],
                dtype=np.float32,
            )
        else:
            bs = np.zeros(52, dtype=np.float32)

        mat = (
            np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
            if res.facial_transformation_matrixes
            else np.eye(4, dtype=np.float32)
        )
        gaze = self._compute_gaze(flms)

        return FaceExtractionFrame(
            landmarks_3d=lms,
            blendshapes=bs,
            transform_matrix=mat,
            gaze_vectors=gaze,
            head_bbox=None,
        )


def compute_head_roi(
    screen_landmarks_2d: np.ndarray,
    frame_width: int,
    frame_height: int,
    *,
    padding_scale: float = 2.8,
) -> tuple[int, int, int, int]:
    """Calculate square pixel bounding box [x_min, y_min, x_max, y_max] from pose head landmarks."""
    pts = screen_landmarks_2d[:11]
    xs = pts[:, 0] * float(frame_width)
    ys = pts[:, 1] * float(frame_height)

    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    span = max(max_x - min_x, max_y - min_y, 30.0)
    half_box = max(32.0, span * padding_scale / 2.0)

    x_min = max(0, min(frame_width, int(math.floor(cx - half_box))))
    y_min = max(0, min(frame_height, int(math.floor(cy - half_box))))
    x_max = max(0, min(frame_width, int(math.ceil(cx + half_box))))
    y_max = max(0, min(frame_height, int(math.ceil(cy + half_box))))

    if x_max <= x_min:
        x_min = max(0, min(frame_width - 1, x_min))
        x_max = min(frame_width, x_min + 64)
    if y_max <= y_min:
        y_min = max(0, min(frame_height - 1, y_min))
        y_max = min(frame_height, y_min + 64)

    return x_min, y_min, x_max, y_max


def pose_guided_head_roi(
    pose_landmarks: np.ndarray,
    img_width: int,
    img_height: int,
    expansion_factor: float = 2.5,
) -> tuple[int, int, int, int]:
    """Compute square bounding box around head from pose landmarks with expansion.

    pose_landmarks: (33, 3) or (N, 2/3) normalised [0, 1] pose coordinates.
    Returns: (x_min, y_min, x_max, y_max) pixel coordinates strictly clamped to image frame.
    """
    head_indices = [0, 2, 5, 7, 8]
    valid_idx = [i for i in head_indices if i < len(pose_landmarks)]
    if not valid_idx:
        valid_idx = list(range(min(11, len(pose_landmarks))))
    pts = pose_landmarks[valid_idx, :2]

    px = pts[:, 0] * float(img_width)
    py = pts[:, 1] * float(img_height)

    centre_x = float(np.mean(px))
    centre_y = float(np.mean(py))

    span_x = float(np.max(px) - np.min(px))
    span_y = float(np.max(py) - np.min(py))
    base_span = max(span_x, span_y, 20.0)

    half_side = (base_span * expansion_factor) / 2.0

    raw_x_min = int(math.floor(centre_x - half_side))
    raw_y_min = int(math.floor(centre_y - half_side))
    raw_x_max = int(math.ceil(centre_x + half_side))
    raw_y_max = int(math.ceil(centre_y + half_side))

    x_min = max(0, min(img_width, raw_x_min))
    y_min = max(0, min(img_height, raw_y_min))
    x_max = max(0, min(img_width, raw_x_max))
    y_max = max(0, min(img_height, raw_y_max))

    # Guard against non-positive dimensions when landmarks are completely outside frame
    if x_max <= x_min:
        if x_min >= img_width:
            x_min = max(0, img_width - 1)
            x_max = img_width
        else:
            x_max = min(img_width, x_min + 1)

    if y_max <= y_min:
        if y_min >= img_height:
            y_min = max(0, img_height - 1)
            y_max = img_height
        else:
            y_max = min(img_height, y_min + 1)

    return (x_min, y_min, x_max, y_max)


def remap_crop_landmarks(
    landmarks_crop: np.ndarray,
    roi: tuple[int, int, int, int],
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Remap normalized landmarks from crop space [0, 1] back to full-frame space [0, 1]."""
    x_min, y_min, x_max, y_max = roi
    crop_w = x_max - x_min
    crop_h = y_max - y_min

    out = landmarks_crop.copy()
    out[..., 0] = (x_min + landmarks_crop[..., 0] * crop_w) / max(img_width, 1)
    out[..., 1] = (y_min + landmarks_crop[..., 1] * crop_h) / max(img_height, 1)
    if out.shape[-1] >= 3:
        out[..., 2] = landmarks_crop[..., 2] * (crop_w / max(img_width, 1))
    return out


def synthesise_neutral_rest_body24() -> np.ndarray:
    """Synthesise neutral rest Body24 skeleton for portrait fallback mode.

    Returns: (24, 3) array in Y-up, metre-scaled right-handed coordinates.
    """
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [0.0, 1.0, 0.0]       # pelvis
    joints[1] = [-0.10, 0.95, 0.0]    # left_hip
    joints[2] = [0.10, 0.95, 0.0]     # right_hip
    joints[3] = [0.0, 1.15, 0.0]      # spine1
    joints[6] = [0.0, 1.30, 0.0]      # spine2
    joints[9] = [0.0, 1.45, 0.0]      # spine3
    joints[4] = [-0.10, 0.50, 0.0]    # left_knee
    joints[5] = [0.10, 0.50, 0.0]     # right_knee
    joints[7] = [-0.10, 0.10, 0.0]    # left_ankle
    joints[8] = [0.10, 0.10, 0.0]     # right_ankle
    joints[10] = [-0.10, 0.0, -0.15]  # left_foot
    joints[11] = [0.10, 0.0, -0.15]   # right_foot
    joints[12] = [0.0, 1.55, 0.0]     # neck
    joints[15] = [0.0, 1.70, 0.0]     # head
    joints[13] = [-0.08, 1.50, 0.0]   # left_collar
    joints[14] = [0.08, 1.50, 0.0]    # right_collar
    joints[16] = [-0.20, 1.48, 0.0]   # left_shoulder
    joints[17] = [0.20, 1.48, 0.0]    # right_shoulder
    joints[18] = [-0.25, 1.20, 0.0]   # left_elbow
    joints[19] = [0.25, 1.20, 0.0]    # right_elbow
    joints[20] = [-0.28, 0.95, 0.0]   # left_wrist
    joints[21] = [0.28, 0.95, 0.0]    # right_wrist
    joints[22] = [-0.30, 0.90, 0.0]   # left_hand
    joints[23] = [0.30, 0.90, 0.0]    # right_hand
    return joints


# Alias with American spelling
synthesize_neutral_rest_body24 = synthesise_neutral_rest_body24


class OneEuroFilterBlendshapes:
    """Vectorised One-Euro (1€) adaptive low-pass filter across (T, 52) blendshapes."""

    def __init__(
        self,
        fps: float = 30.0,
        min_cutoff: float | np.ndarray = 1.0,
        beta: float | np.ndarray = 0.01,
        d_cutoff: float = 1.0,
        *,
        freq: float | None = None,
    ) -> None:
        self.fps = float(freq if freq is not None else fps)
        self.d_cutoff = float(d_cutoff)
        if isinstance(min_cutoff, np.ndarray):
            self.min_cutoff = min_cutoff.astype(np.float32).copy()
        else:
            self.min_cutoff = np.full(52, float(min_cutoff), dtype=np.float32)

        if isinstance(beta, np.ndarray):
            self.beta = beta.astype(np.float32).copy()
        else:
            self.beta = np.full(52, float(beta), dtype=np.float32)

        if not isinstance(beta, np.ndarray) and beta == 0.01:
            # Specialised velocity sensitivity for ARKit blinks (8: eyeBlinkLeft, 9: eyeBlinkRight)
            for idx in [8, 9]:
                self.min_cutoff[idx] = 1.5
                self.beta[idx] = 0.08

        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @property
    def freq(self) -> float:
        """Sampling rate in Hertz (alias for fps)."""
        return self.fps

    def filter(self, x: np.ndarray, rate: float | None = None) -> np.ndarray:
        """Filter a single frame (52,) blendshapes."""
        fps = rate or self.fps
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        te = 1.0 / max(fps, 1e-4)
        tau_d = 1.0 / (2.0 * math.pi * max(self.d_cutoff, 1e-4))
        alpha_d = float(1.0 / (1.0 + tau_d / te))

        dx = (x - self.x_prev) / te
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        fc = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * math.pi * np.maximum(fc, 1e-4))
        alpha = 1.0 / (1.0 + tau / te)

        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def filter_trajectory(self, blendshapes: np.ndarray) -> np.ndarray:
        """Filter an entire (T, 52) trajectory."""
        T, C = blendshapes.shape
        if T <= 1 or C != 52:
            return blendshapes.copy()
        out = np.zeros_like(blendshapes, dtype=np.float32)
        self.x_prev = None
        self.dx_prev = None
        for t in range(T):
            out[t] = self.filter(blendshapes[t])
        return out


def apply_anatomical_constraints(
    blendshapes: np.ndarray,
    names: Sequence[str] = CANONICAL_ARKIT_BLENDSHAPES,
) -> np.ndarray:
    """Apply biomechanical plausibility constraints, antagonistic muting, and eyelid-gaze coupling."""
    is_1d = blendshapes.ndim == 1
    out = np.clip(np.asarray(blendshapes, dtype=np.float32), 0.0, 1.0).copy()
    if is_1d:
        out = out[None, :]

    name_to_idx = {name: i for i, name in enumerate(names)}
    blink_l = name_to_idx.get("eyeBlinkLeft", 8)
    blink_r = name_to_idx.get("eyeBlinkRight", 9)
    down_l = name_to_idx.get("eyeLookDownLeft", 10)
    down_r = name_to_idx.get("eyeLookDownRight", 11)
    wide_l = name_to_idx.get("eyeWideLeft", 20)
    wide_r = name_to_idx.get("eyeWideRight", 21)
    jaw_l = name_to_idx.get("jawLeft", 23)
    jaw_open = name_to_idx.get("jawOpen", 24)
    jaw_r = name_to_idx.get("jawRight", 25)
    mouth_close = name_to_idx.get("mouthClose", 26)
    mouth_frown_l = name_to_idx.get("mouthFrownLeft", 29)
    mouth_frown_r = name_to_idx.get("mouthFrownRight", 30)
    mouth_l = name_to_idx.get("mouthLeft", 32)
    mouth_pucker = name_to_idx.get("mouthPucker", 37)
    mouth_r = name_to_idx.get("mouthRight", 38)
    mouth_smile_l = name_to_idx.get("mouthSmileLeft", 43)
    mouth_smile_r = name_to_idx.get("mouthSmileRight", 44)
    mouth_stretch_l = name_to_idx.get("mouthStretchLeft", 45)
    mouth_stretch_r = name_to_idx.get("mouthStretchRight", 46)

    # 1. Eyelid-gaze coupling: downward gaze suppresses upper eyelid retraction and false blinks
    out[:, wide_l] *= np.maximum(0.0, 1.0 - out[:, down_l] * 0.8)
    out[:, wide_r] *= np.maximum(0.0, 1.0 - out[:, down_r] * 0.8)
    out[:, blink_l] = np.clip(out[:, blink_l] - 0.45 * out[:, down_l], 0.0, 1.0)
    out[:, blink_r] = np.clip(out[:, blink_r] - 0.45 * out[:, down_r], 0.0, 1.0)

    # 2. Antagonistic muting: eyeWide vs eyeBlink
    out[:, wide_l] *= np.maximum(0.0, 1.0 - out[:, blink_l] * 1.5)
    out[:, wide_r] *= np.maximum(0.0, 1.0 - out[:, blink_r] * 1.5)

    # 3. Antagonistic muting: jawOpen vs mouthClose
    out[:, mouth_close] *= np.maximum(0.0, 1.0 - out[:, jaw_open] * 2.0)

    # 4. Jaw lateral excursion (mutual exclusivity)
    delta_jaw = out[:, jaw_l] - out[:, jaw_r]
    out[:, jaw_l] = np.maximum(0.0, delta_jaw)
    out[:, jaw_r] = np.maximum(0.0, -delta_jaw)

    # 5. Mouth lateral deviation (mutual exclusivity)
    delta_mouth = out[:, mouth_l] - out[:, mouth_r]
    out[:, mouth_l] = np.maximum(0.0, delta_mouth)
    out[:, mouth_r] = np.maximum(0.0, -delta_mouth)

    # 6. Smile vs Frown antagonism
    out[:, mouth_frown_l] *= np.maximum(0.0, 1.0 - 1.5 * out[:, mouth_smile_l])
    out[:, mouth_frown_r] *= np.maximum(0.0, 1.0 - 1.5 * out[:, mouth_smile_r])

    # 7. Pucker vs Stretch antagonism
    stretch_max = np.maximum(out[:, mouth_stretch_l], out[:, mouth_stretch_r])
    out[:, mouth_pucker] *= np.maximum(0.0, 1.0 - stretch_max)

    res = np.clip(out, 0.0, 1.0)
    return res[0] if is_1d else res


# ==============================================================================
# Video & Still Media Helper Functions
# ==============================================================================

def video_duration_seconds(path: Path) -> float | None:
    """Wall-clock duration of a video clip in seconds. Prefers ffprobe, falls back to OpenCV."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration:format=duration",
                "-of",
                "json",
                os.fspath(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            import json

            payload = json.loads(completed.stdout)
            streams = payload.get("streams") or []
            candidates = []
            if streams and streams[0].get("duration"):
                candidates.append(streams[0]["duration"])
            fmt = payload.get("format") or {}
            if fmt.get("duration"):
                candidates.append(fmt["duration"])
            for raw in candidates:
                duration = float(raw)
                if duration > 0:
                    return duration
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass

    try:
        cap = cv2.VideoCapture(os.fspath(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        if fps > 1e-6 and count > 0:
            return count / fps
    except Exception:
        pass
    return None


def video_native_fps(path: Path) -> float | None:
    """Container native frame rate. Returns None if it cannot be determined."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                os.fspath(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            import json

            streams = json.loads(completed.stdout).get("streams") or []
            if streams:
                rate = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate") or "0/1"
                num, _, den = str(rate).partition("/")
                fps = float(num) / float(den or 1)
                if fps > 0:
                    return fps
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        pass

    try:
        cap = cv2.VideoCapture(os.fspath(path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        finally:
            cap.release()
        if fps > 1e-6:
            return fps
    except Exception:
        pass
    return None


def playback_fps(video: Path, n_motion_frames: int, fallback: float = 30.0) -> float:
    """Calculate the FPS that plays `n_motion_frames` over the clip's true duration."""
    n = max(1, int(n_motion_frames))
    duration = video_duration_seconds(video)
    native = video_native_fps(video)
    if duration and duration > 1e-6:
        implied_source_frames = native * duration if native else None
        if (
            implied_source_frames is not None
            and abs(implied_source_frames - n) <= 2
            and native
            and native > 0
        ):
            return float(native)
        return n / duration
    if native and native > 0:
        return float(native)
    return float(fallback) if fallback > 0 else 30.0


def still_to_hold_video(
    image: Path,
    out_mp4: Path,
    *,
    n_frames: int = HOLD_FRAMES,
    fps: float = HOLD_FPS,
    device: str = "cpu",
) -> Path:
    """Encode a still photo into a short static hold clip."""
    from .cuda_accel import ffmpeg_video_encode_args, use_cuda

    image = Path(image)
    frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot decode image: {image}")

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest > HOLD_MAX_SIDE:
        scale = HOLD_MAX_SIDE / longest
        w = int(round(w * scale))
        h = int(round(h * scale))
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    w -= w % 2
    h -= h % 2
    if w < 2 or h < 2:
        raise ValueError(f"Image dimensions are too small: {image}")
    frame = frame[:h, :w]

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        for enc_dev in ([device, "cpu"] if use_cuda(device) else ["cpu"]):
            cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-frames:v",
                str(n_frames),
                *ffmpeg_video_encode_args(enc_dev, crf=18, preset="veryfast"),
                "-movflags",
                "+faststart",
                str(out_mp4),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            assert proc.stdin is not None
            try:
                for _ in range(n_frames):
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
            except Exception:
                pass
            if proc.wait() == 0 and out_mp4.exists():
                return out_mp4

    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_mp4}")
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    return out_mp4


@contextmanager
def temp_hold_video(image: Path, *, device: str = "cpu") -> Iterator[Path]:
    """Context manager writing a temporary hold video deleted upon exit."""
    tmp = Path(tempfile.mkdtemp(prefix="v2bm-hold-"))
    try:
        yield still_to_hold_video(image, tmp / f"{Path(image).stem}.mp4", device=device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# Google MediaPipe 3D Pose Extraction & Mapping Engine
# ==============================================================================

class MediaPipeExtractor:
    """Permissive (Apache-2.0) 3D human pose extractor powered by Google MediaPipe.
 
    Lightweight MediaPipe PoseLandmarker inference engine. Formats output into
    the exact 24 Body24 joint contract in metres.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        video_mode: bool = False,
        device: str = "cpu",
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )
        from .cuda_accel import get_cuda_capabilities, resolve_device, use_cuda

        resolved_model = model_path or checks.ensure_mediapipe_model()
        if not resolved_model.exists():
            raise FileNotFoundError(
                f"MediaPipe pose landmarker model not found at {resolved_model}"
            )

        self.video_mode = video_mode
        self.device = resolve_device(device)
        mode = RunningMode.VIDEO if video_mode else RunningMode.IMAGE
        landmarker = None

        if use_cuda(self.device) and get_cuda_capabilities().mediapipe_gpu_delegate:
            try:
                gpu_options = PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(resolved_model),
                        delegate=BaseOptions.Delegate.GPU,
                    ),
                    running_mode=mode,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False,
                )
                landmarker = PoseLandmarker.create_from_options(gpu_options)
            except Exception:
                landmarker = None

        if landmarker is None:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(resolved_model)),
                running_mode=mode,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            landmarker = PoseLandmarker.create_from_options(options)

        self.landmarker = landmarker
        self.mp_image_module = mp.Image
        self.mp_image_format = mp.ImageFormat

    def close(self) -> None:
        """Explicitly close the underlying MediaPipe PoseLandmarker task runner."""
        if getattr(self, "landmarker", None) is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    @staticmethod
    def _unpack_result(result) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not result.pose_world_landmarks or not result.pose_landmarks:
            return None

        w_lms = result.pose_world_landmarks[0]
        s_lms = result.pose_landmarks[0]

        w_coords = np.zeros((33, 3), dtype=np.float32)
        s_coords = np.zeros((33, 3), dtype=np.float32)
        vis = np.zeros(33, dtype=np.float32)

        for i in range(33):
            # MediaPipe world coordinates:
            # +X_mp: screen right (anatomical left of subject)
            # +Y_mp: down (towards ground)
            # +Z_mp: into scene (away from camera)
            w_coords[i, 0] = w_lms[i].x
            w_coords[i, 1] = w_lms[i].y
            w_coords[i, 2] = w_lms[i].z

            s_coords[i, 0] = s_lms[i].x
            s_coords[i, 1] = s_lms[i].y
            s_coords[i, 2] = s_lms[i].z

            vis[i] = getattr(w_lms[i], "visibility", 1.0) or 1.0

        return w_coords, s_coords, vis

    def detect_frame(
        self, bgr_frame: np.ndarray, timestamp_ms: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Detect human pose on a single BGR frame (or video frame with timestamp_ms).

        Returns:
            tuple of (world_landmarks (33, 3), screen_landmarks (33, 3), visibility (33,))
            or None if no person was detected.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = self.mp_image_module(
            image_format=self.mp_image_format.SRGB, data=rgb
        )
        if self.video_mode and timestamp_ms is not None:
            result = self.landmarker.detect_for_video(mp_img, int(timestamp_ms))
        else:
            result = self.landmarker.detect(mp_img)

        return self._unpack_result(result)

    @staticmethod
    def map_to_body24(
        world_landmarks: np.ndarray, visibilities: np.ndarray
    ) -> np.ndarray:
        """Transform MediaPipe landmarks into 24 Body24 joints in human world space.

        Coordinate transformation:
            Proper rotation rotZ(180°):
                p_human = (-x_mp, -y_mp, z_mp)
            Produces:
                +Y = Up (against gravity)
                +X = Anatomical right
                -Z = Chest-forward (facing direction in capture frame)
            Seamlessly converted to Humanoid rest rig via R_BODY_ALIGN ([-x, y, -z]).

        Mathematical derivation for all 24 joints:
            0:  pelvis        = 0.5 * (hip_left + hip_right)
            1:  left_hip      = landmark 23
            2:  right_hip     = landmark 24
            3:  spine1        = pelvis + 0.25 * (neck - pelvis)
            4:  left_knee     = landmark 25
            5:  right_knee    = landmark 26
            6:  spine2        = pelvis + 0.55 * (neck - pelvis)
            7:  left_ankle    = landmark 27
            8:  right_ankle   = landmark 28
            9:  spine3        = pelvis + 0.82 * (neck - pelvis)
            10: left_foot     = landmark 31 (foot_index / toe ball)
            11: right_foot    = landmark 32 (foot_index / toe ball)
            12: neck          = 0.5 * (shoulder_left + shoulder_right)
            13: left_collar   = spine3 + 0.35 * (shoulder_left - spine3)
            14: right_collar  = spine3 + 0.35 * (shoulder_right - spine3)
            15: head          = ear midpoint (7, 8) if visible; fallback from nose
            16: left_shoulder = landmark 11
            17: right_shoulder= landmark 12
            18: left_elbow    = landmark 13
            19: right_elbow   = landmark 14
            20: left_wrist    = landmark 15
            21: right_wrist   = landmark 16
            22: left_hand     = 0.5 * (pinky_17 + index_19)
            23: right_hand    = 0.5 * (pinky_18 + index_20)
        """
        w = np.asarray(world_landmarks, dtype=np.float32)

        # Coordinate transformation: rotZ(180°) -> (-x, -y, z)
        p = np.zeros_like(w)
        p[:, 0] = -w[:, 0]
        p[:, 1] = -w[:, 1]
        p[:, 2] = w[:, 2]

        j = np.zeros((24, 3), dtype=np.float32)

        # Root and hips
        pelvis = 0.5 * (p[23] + p[24])
        j[0] = pelvis
        j[1] = p[23]  # left_hip
        j[2] = p[24]  # right_hip

        # Torso upright vector
        neck = 0.5 * (p[11] + p[12])
        j[12] = neck

        # Spine column
        torso_vec = neck - pelvis
        j[3] = pelvis + 0.25 * torso_vec  # spine1 (lower lumbar)
        j[6] = pelvis + 0.55 * torso_vec  # spine2 (mid thoracic)
        j[9] = pelvis + 0.82 * torso_vec  # spine3 (upper thoracic)

        # Lower extremities
        j[4] = p[25]   # left_knee
        j[5] = p[26]   # right_knee
        j[7] = p[27]   # left_ankle
        j[8] = p[28]   # right_ankle
        j[10] = p[31]  # left_foot (toe ball)
        j[11] = p[32]  # right_foot (toe ball)

        # Clavicles / Collars
        j[13] = j[9] + 0.35 * (p[11] - j[9])  # left_collar
        j[14] = j[9] + 0.35 * (p[12] - j[9])  # right_collar

        # Cranial head centre
        v_ear = min(visibilities[7], visibilities[8])
        if v_ear > 0.3:
            j[15] = 0.5 * (p[7] + p[8])
        else:
            # Fallback relative to nose: ears are slightly cranial (+Y) and posterior (+Z)
            j[15] = p[0] + np.array([0.0, 0.04, 0.08], dtype=np.float32)

        # Upper extremities
        j[16] = p[11]  # left_shoulder
        j[17] = p[12]  # right_shoulder
        j[18] = p[13]  # left_elbow
        j[19] = p[14]  # right_elbow
        j[20] = p[15]  # left_wrist
        j[21] = p[16]  # right_wrist

        # Palm plates
        j[22] = 0.5 * (p[17] + p[19])  # left_hand
        j[23] = 0.5 * (p[18] + p[20])  # right_hand

        return j


# ==============================================================================
# High-Level Motion Extraction Pipeline
# ==============================================================================

def extract_motion(
    video: Path,
    output_npz: Path,
    *,
    work_dir: Path | None = None,
    device: str = "cpu",
    static_camera: bool = True,
    verbose: bool = True,
    source_image: Path | None = None,
    framing_mode: str = "auto",
    capture_face: bool = True,
) -> Path:
    """Extract 3D human motion and facial performance, saving skeleton and face motion payloads.

    Args:
        video: Path to input video or hold clip.
        output_npz: Destination path for skeleton_motion.npz.
        work_dir: Working cache directory.
        device: Execution device ('cpu', 'cuda', 'mps', 'auto').
        static_camera: Flag indicating static camera perspective.
        verbose: Verbose logging output.
        source_image: Optional path if the run originated from a still photo.
        framing_mode: Framing mode ('auto', 'full_body', 'portrait').
        capture_face: Flag indicating whether to extract facial performance.

    Returns:
        Path to the generated skeleton_motion.npz.
    """
    from .cuda_accel import map_to_body24_batch_cuda, resolve_device, use_cupy

    active_device = resolve_device(device)
    video = Path(video).resolve()
    is_still_source = source_image is not None or paths.is_image(video)
    actual_image = Path(source_image).resolve() if source_image is not None else (
        video if paths.is_image(video) else None
    )

    extractor = MediaPipeExtractor(device=active_device)
    face_extractor = MediaPipeFaceExtractor(device=active_device) if capture_face else None

    mode = framing_mode.lower().strip() if framing_mode else "auto"

    # --------------------------------------------------------------------------
    # Case A: Static Still Photo Extraction
    # --------------------------------------------------------------------------
    if is_still_source and actual_image is not None and actual_image.exists():
        if verbose:
            print(
                f"extracting static 3D pose and face from still: {actual_image.name} (device={active_device})"
            )

        frame = cv2.imread(str(actual_image), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image: {actual_image}")

        det = None
        if mode != "portrait":
            det = extractor.detect_frame(frame)

        if det is not None:
            framing = "full_body"
            w_coords, s_coords, vis = det
            if use_cupy(active_device):
                single_pose_body24 = map_to_body24_batch_cuda(
                    w_coords[None, ...], vis[None, ...]
                )[0]
            else:
                single_pose_body24 = extractor.map_to_body24(w_coords, vis)

            from .stabilize_kp2d import clamp_anatomical_hinges

            single_pose_body24 = clamp_anatomical_hinges(
                single_pose_body24[None, ...], device=active_device
            )[0]

            # Pose-guided head ROI crop for facial extraction
            if capture_face and face_extractor is not None:
                head_roi = compute_head_roi(s_coords, frame.shape[1], frame.shape[0])
                face_det = face_extractor.detect_cropped_roi(frame, head_roi)
                if face_det is None:
                    face_det = face_extractor.detect_frame(frame)
            else:
                face_det = None
        else:
            # Portrait fallback
            if capture_face and face_extractor is not None:
                face_det = face_extractor.detect_frame(frame)
            else:
                face_det = None
            if face_det is None and mode != "portrait":
                extractor.close()
                if face_extractor is not None:
                    face_extractor.close()
                raise RuntimeError(
                    f"MediaPipe failed to detect a human subject or face in image: {actual_image}"
                )
            framing = "portrait"
            single_pose_body24 = synthesise_neutral_rest_body24()
            if face_det is not None:
                r_head = face_det.transform_matrix[:3, :3]
                rest_offset = single_pose_body24[15] - single_pose_body24[12]
                single_pose_body24[15] = single_pose_body24[12] + r_head @ rest_offset
            w_coords = np.zeros((33, 3), dtype=np.float32)
            s_coords = np.zeros((33, 3), dtype=np.float32)

        extractor.close()
        if face_extractor is not None:
            face_extractor.close()

        n_frames = HOLD_FRAMES
        fps = HOLD_FPS
        joints_3d = np.tile(single_pose_body24[None, ...], (n_frames, 1, 1))
        screen_lms = np.tile(s_coords[None, ...], (n_frames, 1, 1))
        world_lms = np.tile(w_coords[None, ...], (n_frames, 1, 1))

        if framing == "full_body":
            root_translation, root_orientation = estimate_root_trajectory_and_pinning(
                joints_3d, screen_lms, fps=fps, device=active_device
            )
        else:
            root_translation = np.zeros((n_frames, 3), dtype=np.float32)
            root_orientation = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[None, :], (n_frames, 1)
            )

        timestamps = np.arange(n_frames, dtype=np.float64) / float(fps)

        if face_det is not None:
            raw_blendshapes = np.tile(face_det.blendshapes[None, ...], (n_frames, 1))
            face_landmarks_3d = np.tile(face_det.landmarks_3d[None, ...], (n_frames, 1, 1))
            head_mats = np.tile(face_det.transform_matrix[None, ...], (n_frames, 1, 1))
            gaze_vectors = np.tile(face_det.gaze_vectors[None, ...], (n_frames, 1, 1))
        else:
            raw_blendshapes = np.zeros((n_frames, 52), dtype=np.float32)
            face_landmarks_3d = np.zeros((n_frames, 478, 3), dtype=np.float32)
            head_mats = np.tile(np.eye(4, dtype=np.float32)[None, ...], (n_frames, 1, 1))
            gaze_vectors = np.zeros((n_frames, 2, 3), dtype=np.float32)

    # --------------------------------------------------------------------------
    # Case B: Video Motion Sequence Extraction
    # --------------------------------------------------------------------------
    else:
        if not video.exists():
            raise FileNotFoundError(f"Input video not found: {video}")

        if verbose:
            print(
                f"extracting 3D motion and facial performance from video: {video.name} (device={active_device})"
            )

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

        video_extractor = MediaPipeExtractor(video_mode=True, device=active_device)
        video_face_extractor = (
            MediaPipeFaceExtractor(video_mode=True, device=active_device)
            if capture_face
            else None
        )
        step_ms = max(1, int(round(1000.0 / (raw_fps if raw_fps > 1e-3 else 30.0))))

        world_list: list[np.ndarray | None] = []
        screen_list: list[np.ndarray | None] = []
        vis_list: list[np.ndarray | None] = []
        face_list: list[FaceExtractionFrame | None] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = frame_idx * step_ms

            det = None
            if mode != "portrait":
                det = video_extractor.detect_frame(frame, timestamp_ms=ts_ms)
                if det is None:
                    det = extractor.detect_frame(frame)

            face_det = None
            if det is not None:
                w_coords, s_coords, vis = det
                world_list.append(w_coords)
                screen_list.append(s_coords)
                vis_list.append(vis)

                if capture_face and video_face_extractor is not None:
                    head_roi = compute_head_roi(s_coords, frame.shape[1], frame.shape[0])
                    face_det = video_face_extractor.detect_cropped_roi(
                        frame, head_roi, timestamp_ms=ts_ms
                    )
                    if face_det is None and face_extractor is not None:
                        face_det = face_extractor.detect_frame(frame)
            else:
                world_list.append(None)
                screen_list.append(None)
                vis_list.append(None)

                if capture_face and video_face_extractor is not None:
                    face_det = video_face_extractor.detect_frame(frame, timestamp_ms=ts_ms)
                    if face_det is None and face_extractor is not None:
                        face_det = face_extractor.detect_frame(frame)

            face_list.append(face_det)
            frame_idx += 1

        cap.release()
        video_extractor.close()
        if video_face_extractor is not None:
            video_face_extractor.close()
        extractor.close()
        if face_extractor is not None:
            face_extractor.close()

        n_frames = len(face_list)
        if n_frames == 0:
            raise RuntimeError(f"Video contains 0 decodable frames: {video}")

        valid_pose_indices = [i for i, w in enumerate(world_list) if w is not None]
        valid_face_indices = [i for i, f in enumerate(face_list) if f is not None]

        fps = playback_fps(video, n_frames, fallback=raw_fps or HOLD_FPS)
        timestamps = np.arange(n_frames, dtype=np.float64) / float(fps)

        if valid_pose_indices and mode != "portrait":
            framing = "full_body"

            # Fill leading missing frames from the first valid detection
            first_valid = valid_pose_indices[0]
            for i in range(first_valid):
                world_list[i] = world_list[first_valid].copy()
                screen_list[i] = screen_list[first_valid].copy()
                vis_list[i] = vis_list[first_valid].copy()

            # Fill trailing missing frames from the last valid detection
            last_valid = valid_pose_indices[-1]
            for i in range(last_valid + 1, n_frames):
                world_list[i] = world_list[last_valid].copy()
                screen_list[i] = screen_list[last_valid].copy()
                vis_list[i] = vis_list[last_valid].copy()

            # Linearly interpolate any interior gaps between valid detections
            for k in range(len(valid_pose_indices) - 1):
                i0 = valid_pose_indices[k]
                i1 = valid_pose_indices[k + 1]
                if i1 - i0 > 1:
                    for i in range(i0 + 1, i1):
                        alpha = float(i - i0) / float(i1 - i0)
                        world_list[i] = (
                            (1.0 - alpha) * world_list[i0] + alpha * world_list[i1]
                        ).astype(np.float32)
                        screen_list[i] = (
                            (1.0 - alpha) * screen_list[i0] + alpha * screen_list[i1]
                        ).astype(np.float32)
                        vis_list[i] = (
                            (1.0 - alpha) * vis_list[i0] + alpha * vis_list[i1]
                        ).astype(np.float32)

            world_lms = np.array(world_list, dtype=np.float32)
            screen_lms = np.array(screen_list, dtype=np.float32)
            vis_array = np.array(vis_list, dtype=np.float32)

            # Map to Body24 24 joints in human coordinates (GPU batch or CPU)
            if use_cupy(active_device):
                raw_body24_joints = map_to_body24_batch_cuda(world_lms, vis_array)
            else:
                raw_body24_joints = np.zeros((n_frames, 24, 3), dtype=np.float32)
                for i in range(n_frames):
                    raw_body24_joints[i] = extractor.map_to_body24(
                        world_lms[i], vis_array[i]
                    )

            # 1. Adaptive One-Euro (1€) filter on 3D coordinates
            filter_3d = OneEuroFilter3D(
                freq=fps,
                min_cutoff_xy=1.0,
                beta_xy=0.005,
                min_cutoff_z=0.5,
                beta_z=0.05,
            )
            filtered_joints = filter_3d.filter_trajectory(raw_body24_joints, rate=fps)

            distal_filter = OneEuroFilter3D(
                freq=fps,
                min_cutoff_xy=0.85,
                beta_xy=0.004,
                min_cutoff_z=0.35,
                beta_z=0.008,
            )
            filtered_joints[:, 18:24, :] = distal_filter.filter_trajectory(
                raw_body24_joints[:, 18:24, :], rate=fps
            )

            stabiliser = BiomechanicalStabiliser()
            joints_3d, swap_flags = stabiliser.stabilise(
                filtered_joints, screen_lms, device=active_device
            )

            if verbose and int(swap_flags.sum()) > 0:
                print(
                    f"biomechanical stabiliser corrected {int(swap_flags.sum())}/{n_frames} "
                    f"left/right bilateral limb flips"
                )

            from .ik_optimiser import optimise_trajectory_windowed

            joints_3d = optimise_trajectory_windowed(
                joints_3d,
                landmarks_2d=screen_lms,
                visibilities=vis_array,
                fps=fps,
                device=active_device,
                verbose=verbose,
            )

            root_translation, root_orientation = estimate_root_trajectory_and_pinning(
                joints_3d, screen_lms, fps=fps, device=active_device
            )

        elif valid_face_indices or mode == "portrait":
            framing = "portrait"
            rest_body24 = synthesise_neutral_rest_body24()
            joints_3d = np.tile(rest_body24[None, ...], (n_frames, 1, 1))
            screen_lms = None
            world_lms = None
            root_translation = np.zeros((n_frames, 3), dtype=np.float32)
            root_orientation = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[None, :], (n_frames, 1)
            )

        else:
            raise RuntimeError(
                f"No human pose or face detected across any of the {n_frames} frames in {video.name}"
            )

        # Process and interpolate face detections
        if valid_face_indices:
            first_f = valid_face_indices[0]
            for i in range(first_f):
                face_list[i] = face_list[first_f]

            last_f = valid_face_indices[-1]
            for i in range(last_f + 1, n_frames):
                face_list[i] = face_list[last_f]

            for k in range(len(valid_face_indices) - 1):
                i0 = valid_face_indices[k]
                i1 = valid_face_indices[k + 1]
                if i1 - i0 > 1:
                    f0 = face_list[i0]
                    f1 = face_list[i1]
                    assert f0 is not None and f1 is not None
                    for i in range(i0 + 1, i1):
                        alpha = float(i - i0) / float(i1 - i0)
                        interp_bs = (1.0 - alpha) * f0.blendshapes + alpha * f1.blendshapes
                        interp_lms = (1.0 - alpha) * f0.landmarks_3d + alpha * f1.landmarks_3d
                        interp_mat = (1.0 - alpha) * f0.transform_matrix + alpha * f1.transform_matrix
                        interp_gaze = (1.0 - alpha) * f0.gaze_vectors + alpha * f1.gaze_vectors
                        face_list[i] = FaceExtractionFrame(
                            landmarks_3d=interp_lms.astype(np.float32),
                            blendshapes=interp_bs.astype(np.float32),
                            transform_matrix=interp_mat.astype(np.float32),
                            gaze_vectors=interp_gaze.astype(np.float32),
                            head_bbox=None,
                        )

            raw_blendshapes = np.stack(
                [f.blendshapes for f in face_list if f is not None], axis=0
            ).astype(np.float32)
            face_landmarks_3d = np.stack(
                [f.landmarks_3d for f in face_list if f is not None], axis=0
            ).astype(np.float32)
            head_mats = np.stack(
                [f.transform_matrix for f in face_list if f is not None], axis=0
            ).astype(np.float32)
            gaze_vectors = np.stack(
                [f.gaze_vectors for f in face_list if f is not None], axis=0
            ).astype(np.float32)
        else:
            raw_blendshapes = np.zeros((n_frames, 52), dtype=np.float32)
            face_landmarks_3d = np.zeros((n_frames, 478, 3), dtype=np.float32)
            head_mats = np.tile(np.eye(4, dtype=np.float32)[None, ...], (n_frames, 1, 1))
            gaze_vectors = np.zeros((n_frames, 2, 3), dtype=np.float32)

        # In portrait mode, articulate head joint (15) with rigid head rotation matrix
        if framing == "portrait":
            rest_body24 = synthesise_neutral_rest_body24()
            rest_head_vec = rest_body24[15] - rest_body24[12]
            for t in range(n_frames):
                r_head = head_mats[t][:3, :3]
                joints_3d[t, 15] = joints_3d[t, 12] + r_head @ rest_head_vec

    # --------------------------------------------------------------------------
    # Facial Temporal Smoothing & Anatomical Plausibility Constraints
    # --------------------------------------------------------------------------
    from .cuda_accel import apply_anatomical_constraints_cuda, filter_blendshapes_cuda

    filtered_bs = filter_blendshapes_cuda(
        raw_blendshapes,
        fps=fps,
        min_cutoff=1.0,
        beta=0.01,
        d_cutoff=1.0,
        device=active_device,
    )
    constrained_bs = apply_anatomical_constraints_cuda(
        filtered_bs,
        device=active_device,
    )

    # Convert head transformation matrices to continuous quaternions
    from .rig.fbx_skeleton import matrix_to_quat_wxyz
    from .rig.kinematics import continue_quaternion

    head_quats = np.zeros((n_frames, 4), dtype=np.float32)
    head_trans = np.zeros((n_frames, 3), dtype=np.float32)
    prev_q: np.ndarray | None = None
    for t in range(n_frames):
        rmat = head_mats[t][:3, :3]
        q = matrix_to_quat_wxyz(rmat)
        q, _ = continue_quaternion(q, prev_q)
        prev_q = q
        head_quats[t] = q.astype(np.float32)
        head_trans[t] = head_mats[t][:3, 3].astype(np.float32)

    # --------------------------------------------------------------------------
    # Save face_motion.npz in Run Output Directory
    # --------------------------------------------------------------------------
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    face_motion_npz = output_npz.parent / "face_motion.npz"

    np.savez_compressed(
        face_motion_npz,
        blendshapes=constrained_bs.astype(np.float32),
        blendshape_names=np.array(CANONICAL_ARKIT_BLENDSHAPES),
        face_landmarks_3d=face_landmarks_3d.astype(np.float32),
        head_rotation_quat=head_quats.astype(np.float32),
        head_translation=head_trans.astype(np.float32),
        gaze_vectors=gaze_vectors.astype(np.float32),
        fps=np.float32(fps),
        timestamps=timestamps.astype(np.float64),
        framing=np.array(framing),
    )

    # --------------------------------------------------------------------------
    # Save skeleton_motion.npz Payload (with Mirrored Facial Channels)
    # --------------------------------------------------------------------------
    payload = dict(
        joints_3d=joints_3d.astype(np.float32),
        joint_names=np.array(BODY24_JOINT_NAMES),
        fps=np.float32(fps),
        timestamps=timestamps.astype(np.float64),
        root_translation=root_translation.astype(np.float32),
        root_orientation=root_orientation.astype(np.float32),
        video=np.array(str(video)),
        device=np.array(str(active_device)),
        face_blendshapes=constrained_bs.astype(np.float32),
        face_blendshape_names=np.array(CANONICAL_ARKIT_BLENDSHAPES),
        head_rotation_quat=head_quats.astype(np.float32),
        framing=np.array(framing),
    )
    if screen_lms is not None:
        payload["screen_landmarks"] = screen_lms.astype(np.float32)
    if "world_lms" in locals() and world_lms is not None:
        payload["world_landmarks"] = world_lms.astype(np.float32)
    if is_still_source and actual_image is not None:
        payload["image"] = np.array(str(actual_image))

    np.savez_compressed(output_npz, **payload)

    if verbose:
        print(
            f"saved {output_npz} ({len(joints_3d)} frames @ {fps:.2f} fps, framing={framing})"
        )
        print(f"saved {face_motion_npz} (52 ARKit blendshapes)")

    return output_npz
