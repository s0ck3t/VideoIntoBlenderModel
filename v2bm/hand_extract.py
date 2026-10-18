"""21-point hand landmark extraction powered by Google MediaPipe Hand Landmarker.

Permissive (Apache-2.0) hand extraction pipeline featuring:
- Self-contained verification and automated download of official Google weights.
- Pose-guided wrist Region-Of-Interest (ROI) cropping with scale-adaptive bounding radius.
- Bilinear upscaling for distal hand crops below detector resolution.
- Full-frame and local-frame coordinate remapping.
- Vectorised One-Euro (1€) adaptive frequency low-pass filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Sequence
import urllib.request

import cv2
import numpy as np

from .human_motion import BODY24_JOINTS
from .stabilize_kp2d import OneEuroFilter3D

OFFICIAL_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
EXPECTED_MODEL_SIZE = 7_819_105  # bytes (~7.8 MB)
MIN_MODEL_SIZE = 7_000_000


def ensure_hand_landmarker_model(target_path: Path | str | None = None) -> Path:
    """Ensure Google MediaPipe HandLandmarker model is downloaded and return its path.

    Downloads from official Google storage if missing or truncated (< 7.0 MB).
    """
    if target_path is None:
        target_path = (
            Path(__file__).resolve().parent.parent
            / "weights"
            / "mediapipe"
            / "hand_landmarker.task"
        )
    else:
        target_path = Path(target_path)

    if not target_path.exists() or target_path.stat().st_size < MIN_MODEL_SIZE:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            OFFICIAL_HAND_LANDMARKER_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "VideoIntoBlenderModel/0.2.0"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)

    if not target_path.exists() or target_path.stat().st_size < MIN_MODEL_SIZE:
        size = target_path.stat().st_size if target_path.exists() else 0
        raise RuntimeError(
            f"Failed to acquire MediaPipe HandLandmarker model at {target_path} (size={size} bytes)"
        )
    return target_path


@dataclass
class HandLandmarks:
    """21-point hand landmarks for a single hand in a frame."""

    landmarks_3d: np.ndarray  # (21, 3) metric coords in Humanoid world frame (+X left, +Y up, +Z forward)
    landmarks_screen: np.ndarray  # (21, 3) normalised screen coords in [0, 1]^2
    confidence: float  # detection confidence score in [0, 1]
    side: str  # "left" or "right"


@dataclass
class ExtractedHandMotion:
    """Temporal hand trajectories across a video sequence."""

    left_landmarks_3d: np.ndarray  # (T, 21, 3)
    right_landmarks_3d: np.ndarray  # (T, 21, 3)
    left_screen: np.ndarray  # (T, 21, 3)
    right_screen: np.ndarray  # (T, 21, 3)
    left_present: np.ndarray  # (T,) bool
    right_present: np.ndarray  # (T,) bool
    fps: float


def compute_wrist_roi(
    frame_w: int,
    frame_h: int,
    p_wrist: np.ndarray,
    p_index: np.ndarray,
    p_elbow: np.ndarray,
) -> tuple[int, int, int, int]:
    """Calculate pose-guided wrist bounding box [x1, y1, x2, y2] in pixel coordinates.

    Uses wrist, index finger, and elbow positions with heading projection and scale clamping.
    """
    pw = np.asarray(p_wrist, dtype=np.float64)[:2]
    pi = np.asarray(p_index, dtype=np.float64)[:2]
    pe = np.asarray(p_elbow, dtype=np.float64)[:2]

    # Convert normalised coordinates [0, 1] to pixel space if needed
    if pw[0] <= 1.0 and pw[1] <= 1.0 and frame_w > 1 and frame_h > 1:
        pw = pw * np.array([frame_w, frame_h], dtype=np.float64)
        pi = pi * np.array([frame_w, frame_h], dtype=np.float64)
        pe = pe * np.array([frame_w, frame_h], dtype=np.float64)

    d_wi = float(np.linalg.norm(pi - pw))
    d_we = float(np.linalg.norm(pw - pe))

    if d_wi > 5.0:
        radius = max(d_wi * 1.8, d_we * 0.45, 45.0)
        u_dir = (pi - pw) / d_wi
    elif d_we > 5.0:
        radius = max(d_we * 0.65, 45.0)
        u_dir = (pw - pe) / d_we
    else:
        radius = 45.0
        u_dir = np.array([0.0, 1.0], dtype=np.float64)

    centre = pw + 0.35 * radius * u_dir

    x1 = int(np.clip(math.floor(centre[0] - radius), 0, frame_w - 1))
    y1 = int(np.clip(math.floor(centre[1] - radius), 0, frame_h - 1))
    x2 = int(np.clip(math.ceil(centre[0] + radius), x1 + 1, frame_w))
    y2 = int(np.clip(math.ceil(centre[1] + radius), y1 + 1, frame_h))

    return x1, y1, x2, y2


def crop_wrist_roi(
    rgb_frame: np.ndarray,
    wrist_2d: np.ndarray,
    index_2d: np.ndarray,
    elbow_2d: np.ndarray,
    min_size: int = 192,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract and upscale pose-guided wrist crop from an RGB frame.

    Returns the cropped (and bilinearly upscaled if necessary) image and the bounding box.
    """
    h, w = rgb_frame.shape[:2]
    x1, y1, x2, y2 = compute_wrist_roi(w, h, wrist_2d, index_2d, elbow_2d)
    crop = rgb_frame[y1:y2, x1:x2]

    ch, cw = crop.shape[:2]
    if max(ch, cw) < min_size and ch > 0 and cw > 0:
        crop = cv2.resize(crop, (min_size, min_size), interpolation=cv2.INTER_LINEAR)

    return crop, (x1, y1, x2, y2)


def remap_crop_to_full(
    crop_landmarks: np.ndarray,
    roi_box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Remap normalised crop landmark coordinates (u, v, z) to full-frame normalised space."""
    x1, y1, x2, y2 = roi_box
    crop_w = float(max(1, x2 - x1))
    crop_h = float(max(1, y2 - y1))

    lms = np.asarray(crop_landmarks, dtype=np.float64).copy()
    u_full = (x1 + lms[:, 0] * crop_w) / float(frame_w)
    v_full = (y1 + lms[:, 1] * crop_h) / float(frame_h)
    z_full = lms[:, 2] * (crop_w / float(frame_w))

    return np.column_stack([u_full, v_full, z_full])


class MediaPipeHandExtractor:
    """Permissive (Apache-2.0) 21-point 3D hand landmark extractor powered by Google MediaPipe."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.2,
        min_presence_confidence: float = 0.2,
        min_tracking_confidence: float = 0.2,
        device: str = "cpu",
        filter_landmarks: bool = True,
    ) -> None:
        self.model_path = ensure_hand_landmarker_model(model_path)
        self.device = device
        self.filter_landmarks = filter_landmarks
        self.num_hands = num_hands

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Initialise 21 One-Euro filters per hand for temporal stabilisation
        if self.filter_landmarks:
            self._filters_left_3d = [
                OneEuroFilter3D(
                    freq=30.0,
                    min_cutoff_xy=1.0,
                    beta_xy=0.05,
                    min_cutoff_z=0.8,
                    beta_z=0.05,
                )
                for _ in range(21)
            ]
            self._filters_right_3d = [
                OneEuroFilter3D(
                    freq=30.0,
                    min_cutoff_xy=1.0,
                    beta_xy=0.05,
                    min_cutoff_z=0.8,
                    beta_z=0.05,
                )
                for _ in range(21)
            ]
        else:
            self._filters_left_3d = None
            self._filters_right_3d = None

    def __enter__(self) -> MediaPipeHandExtractor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly release MediaPipe HandLandmarker resources."""
        if getattr(self, "landmarker", None) is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def reset_filters(self) -> None:
        """Reset internal temporal One-Euro filter states."""
        if self._filters_left_3d is not None:
            for f in self._filters_left_3d:
                f.reset()
        if self._filters_right_3d is not None:
            for f in self._filters_right_3d:
                f.reset()

    def _parse_pose_landmarks(
        self, pose_landmarks_2d: np.ndarray, side: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Extract wrist, index, and elbow 2D positions for the given side."""
        arr = np.asarray(pose_landmarks_2d, dtype=np.float64)
        if len(arr) >= 33:
            # MediaPipe Pose 33-landmark layout
            if side == "left":
                idx_w, idx_i, idx_e = 15, 19, 13
            else:
                idx_w, idx_i, idx_e = 16, 20, 14
            return arr[idx_w, :2], arr[idx_i, :2], arr[idx_e, :2]
        elif len(arr) == 24:
            # Body24 joints layout
            prefix = "left" if side == "left" else "right"
            idx_w = BODY24_JOINTS.index(f"{prefix}_wrist")
            idx_i = BODY24_JOINTS.index(f"{prefix}_hand")
            idx_e = BODY24_JOINTS.index(f"{prefix}_elbow")
            return arr[idx_w, :2], arr[idx_i, :2], arr[idx_e, :2]
        return None

    def extract_frame(
        self,
        rgb_frame: np.ndarray,
        pose_landmarks_2d: np.ndarray | None = None,
        timestamp_ms: int = 0,
    ) -> dict[str, HandLandmarks | None]:
        """Extract left and right hand landmarks from a single RGB frame.

        Returns dictionary with keys "left" and "right".
        """
        if self.landmarker is None:
            raise RuntimeError("MediaPipeHandExtractor is closed.")

        frame_h, frame_w = rgb_frame.shape[:2]
        results: dict[str, HandLandmarks | None] = {"left": None, "right": None}

        # 1. Pose-guided Region-of-Interest extraction when 2D pose is provided
        if pose_landmarks_2d is not None:
            for side in ("left", "right"):
                pts = self._parse_pose_landmarks(pose_landmarks_2d, side)
                if pts is None:
                    continue
                pw, pi, pe = pts
                crop, roi_box = crop_wrist_roi(rgb_frame, pw, pi, pe, min_size=192)
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                mp_image = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(crop),
                )
                res = self.landmarker.detect(mp_image)

                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    # Best detection in the crop
                    crop_lms = np.array(
                        [[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]],
                        dtype=np.float64,
                    )
                    screen_lms = remap_crop_to_full(crop_lms, roi_box, frame_w, frame_h)

                    # 3D metric landmarks: MediaPipe (+X right, +Y down, +Z into camera)
                    # mapped to Humanoid world frame (+X left, +Y up, +Z forward)
                    raw_3d = np.array(
                        [[lm.x, -lm.y, -lm.z] for lm in res.hand_world_landmarks[0]],
                        dtype=np.float64,
                    )

                    # Temporal One-Euro filtering
                    if self.filter_landmarks:
                        filters = (
                            self._filters_left_3d
                            if side == "left"
                            else self._filters_right_3d
                        )
                        if filters is not None:
                            filtered_3d = np.zeros_like(raw_3d)
                            for j in range(21):
                                filtered_3d[j] = filters[j].filter(raw_3d[j])
                            raw_3d = filtered_3d

                    conf = 0.90
                    if res.handedness and len(res.handedness) > 0 and len(res.handedness[0]) > 0:
                        conf = float(res.handedness[0][0].score)

                    results[side] = HandLandmarks(
                        landmarks_3d=raw_3d,
                        landmarks_screen=screen_lms,
                        confidence=conf,
                        side=side,
                    )

        # 2. Fallback to full-frame detection if hands remain unresolved
        if results["left"] is None or results["right"] is None:
            mp_full = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
            res_full = self.landmarker.detect(mp_full)

            if res_full.hand_landmarks:
                for i, hand_lms in enumerate(res_full.hand_landmarks):
                    # Determine side from 2D pose wrist proximity if available, else handedness
                    side_label = "left"
                    matched_by_pose = False
                    if pose_landmarks_2d is not None:
                        pts_l = self._parse_pose_landmarks(pose_landmarks_2d, "left")
                        pts_r = self._parse_pose_landmarks(pose_landmarks_2d, "right")
                        if pts_l is not None and pts_r is not None:
                            pw_l = pts_l[0].copy()
                            pw_r = pts_r[0].copy()
                            if float(np.linalg.norm(pw_l)) > 1e-4 and float(np.linalg.norm(pw_r)) > 1e-4:
                                if np.max(np.abs(pw_l)) > 2.0 or np.max(np.abs(pw_r)) > 2.0:
                                    pw_l = np.array([pw_l[0] / max(frame_w, 1), pw_l[1] / max(frame_h, 1)])
                                    pw_r = np.array([pw_r[0] / max(frame_w, 1), pw_r[1] / max(frame_h, 1)])
                                hw = np.array([hand_lms[0].x, hand_lms[0].y], dtype=np.float64)
                                d_l = float(np.linalg.norm(hw - pw_l))
                                d_r = float(np.linalg.norm(hw - pw_r))
                                if min(d_l, d_r) > 0.35:
                                    continue
                                side_label = "left" if d_l <= d_r else "right"
                                matched_by_pose = True

                    if not matched_by_pose:
                        if (
                            res_full.handedness
                            and len(res_full.handedness) > i
                            and len(res_full.handedness[i]) > 0
                        ):
                            name = res_full.handedness[i][0].category_name.lower()
                            # MediaPipe defines handedness from performer's perspective
                            side_label = "left" if "left" in name else "right"
                        else:
                            wrist_x = hand_lms[0].x
                            side_label = "left" if wrist_x > 0.5 else "right"

                    if results[side_label] is not None:
                        continue

                    screen_lms = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float64
                    )
                    raw_3d = np.array(
                        [[lm.x, -lm.y, -lm.z] for lm in res_full.hand_world_landmarks[i]],
                        dtype=np.float64,
                    )

                    if self.filter_landmarks:
                        filters = (
                            self._filters_left_3d
                            if side_label == "left"
                            else self._filters_right_3d
                        )
                        if filters is not None:
                            filtered_3d = np.zeros_like(raw_3d)
                            for j in range(21):
                                filtered_3d[j] = filters[j].filter(raw_3d[j])
                            raw_3d = filtered_3d

                    conf = 0.85
                    if (
                        res_full.handedness
                        and len(res_full.handedness) > i
                        and len(res_full.handedness[i]) > 0
                    ):
                        conf = float(res_full.handedness[i][0].score)

                    results[side_label] = HandLandmarks(
                        landmarks_3d=raw_3d,
                        landmarks_screen=screen_lms,
                        confidence=conf,
                        side=side_label,
                    )

        return results

    def extract_video(
        self,
        video_path: Path | str,
        pose_motion: Any | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> ExtractedHandMotion:
        """Extract and temporally stabilise hand trajectories across an entire video."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = native_fps if native_fps > 0 else 30.0

        left_lms_list: list[np.ndarray] = []
        right_lms_list: list[np.ndarray] = []
        left_screen_list: list[np.ndarray] = []
        right_screen_list: list[np.ndarray] = []
        left_present_list: list[bool] = []
        right_present_list: list[bool] = []

        self.reset_filters()
        frame_idx = 0

        # Retrieve 2D screen pose landmarks sequence if present on motion object
        pose_2d_seq = None
        if pose_motion is not None:
            if (
                hasattr(pose_motion, "screen_landmarks")
                and pose_motion.screen_landmarks is not None
            ):
                pose_2d_seq = pose_motion.screen_landmarks
            elif (
                hasattr(pose_motion, "keypoints_2d")
                and pose_motion.keypoints_2d is not None
            ):
                pose_2d_seq = pose_motion.keypoints_2d

        last_valid_left_3d = np.zeros((21, 3), dtype=np.float64)
        last_valid_right_3d = np.zeros((21, 3), dtype=np.float64)
        last_valid_left_screen = np.zeros((21, 3), dtype=np.float64)
        last_valid_right_screen = np.zeros((21, 3), dtype=np.float64)

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                p2d = (
                    pose_2d_seq[frame_idx]
                    if pose_2d_seq is not None and frame_idx < len(pose_2d_seq)
                    else None
                )
                res = self.extract_frame(rgb_frame, pose_landmarks_2d=p2d)

                # Left hand processing
                if res["left"] is not None:
                    last_valid_left_3d = res["left"].landmarks_3d.copy()
                    last_valid_left_screen = res["left"].landmarks_screen.copy()
                    left_lms_list.append(last_valid_left_3d)
                    left_screen_list.append(last_valid_left_screen)
                    left_present_list.append(True)
                else:
                    left_lms_list.append(last_valid_left_3d.copy())
                    left_screen_list.append(last_valid_left_screen.copy())
                    left_present_list.append(False)

                # Right hand processing
                if res["right"] is not None:
                    last_valid_right_3d = res["right"].landmarks_3d.copy()
                    last_valid_right_screen = res["right"].landmarks_screen.copy()
                    right_lms_list.append(last_valid_right_3d)
                    right_screen_list.append(last_valid_right_screen)
                    right_present_list.append(True)
                else:
                    right_lms_list.append(last_valid_right_3d.copy())
                    right_screen_list.append(last_valid_right_screen.copy())
                    right_present_list.append(False)

                frame_idx += 1
                if progress_callback is not None and total_frames > 0:
                    progress_callback(min(1.0, frame_idx / float(total_frames)))
        finally:
            cap.release()

        n_out = len(left_lms_list)
        if n_out == 0:
            left_3d = np.zeros((0, 21, 3), dtype=np.float64)
            right_3d = np.zeros((0, 21, 3), dtype=np.float64)
            left_scr = np.zeros((0, 21, 3), dtype=np.float64)
            right_scr = np.zeros((0, 21, 3), dtype=np.float64)
            left_p = np.zeros((0,), dtype=bool)
            right_p = np.zeros((0,), dtype=bool)
        else:
            left_3d = np.asarray(left_lms_list, dtype=np.float64)
            right_3d = np.asarray(right_lms_list, dtype=np.float64)
            left_scr = np.asarray(left_screen_list, dtype=np.float64)
            right_scr = np.asarray(right_screen_list, dtype=np.float64)
            left_p = np.asarray(left_present_list, dtype=bool)
            right_p = np.asarray(right_present_list, dtype=bool)

            # Bridge short interior detection dropouts (<= 15 frames) via linear interpolation
            max_gap = 15
            for lms_3d, lms_scr, pres in (
                (left_3d, left_scr, left_p),
                (right_3d, right_scr, right_p),
            ):
                valid_idx = np.flatnonzero(pres)
                if len(valid_idx) >= 2:
                    for k in range(len(valid_idx) - 1):
                        t0 = int(valid_idx[k])
                        t1 = int(valid_idx[k + 1])
                        gap = t1 - t0 - 1
                        if 1 <= gap <= max_gap:
                            for t in range(t0 + 1, t1):
                                alpha = float(t - t0) / float(t1 - t0)
                                lms_3d[t] = (1.0 - alpha) * lms_3d[t0] + alpha * lms_3d[t1]
                                lms_scr[t] = (1.0 - alpha) * lms_scr[t0] + alpha * lms_scr[t1]
                                pres[t] = True

        return ExtractedHandMotion(
            left_landmarks_3d=left_3d,
            right_landmarks_3d=right_3d,
            left_screen=left_scr,
            right_screen=right_scr,
            left_present=left_p,
            right_present=right_p,
            fps=fps,
        )
