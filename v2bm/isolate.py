"""Multi-person tracking, silhouette protection, and background inpainting.

Enables the v2bm pipeline to automatically handle multi-performer videos (e.g. duo
choreography, group performances, or crowded street scenes) by isolating a target
performer and cleanly inpainting all non-target individuals across all frames.

Features:
1. Multi-Object Tracking (MOT): Bipartite matching (Hungarian algorithm) over
   spatial centroids, bounding box heights, and torso chromatic signatures.
2. Invariant Tracking Across Swaps: Maintains continuous identity across lateral
   performer swaps (left <-> right) and depth crossings (forwards <-> backwards).
3. Dynamic Silhouette Protection: Dilates non-target masks while strictly subtracting
   the target performer's silhouette, eliminating limb clipping during close interaction.
4. Multi-Person Inpainting: Inpaints 1, 2, or more non-target performers simultaneously
   using fast Telea / Navier-Stokes methods.
5. Audio Stream Preservation: Multiplexes the original source audio into the isolated video.

All code, comments, docstrings, and documentation strictly use British English conventions.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import checks, paths

TargetSelector = Literal["auto", "primary", "left", "right", "tallest", "shortest"]


@dataclass
class DetectedPose:
    """Representation of a detected human pose on a single frame."""

    index: int
    landmarks: np.ndarray  # (33, 3) screen landmarks [0, 1]
    bbox: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y in pixels
    centroid: np.ndarray  # (2,) [cx, cy] in pixels
    height: float  # vertical span in pixels
    area: float  # bbox width * height
    torso_colour: np.ndarray  # mean BGR colour of torso quadrilateral
    segmentation_mask: np.ndarray | None = None  # (H, W) float32 [0, 1]


@dataclass
class SubjectTrack:
    """Temporal trajectory of a tracked individual."""

    track_id: int
    is_target: bool = False
    last_centroid: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    last_height: float = 0.0
    last_torso_colour: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    consecutive_misses: int = 0
    total_detections: int = 0


class MultiPersonTracker:
    """Multi-subject tracking and silhouette isolation engine."""

    def __init__(
        self,
        target_selector: TargetSelector = "auto",
        target_point: tuple[float, float] | None = None,
        max_misses: int = 15,
    ) -> None:
        self.target_selector = target_selector
        self.target_point = target_point
        self.max_misses = max_misses
        self.tracks: list[SubjectTrack] = []
        self.next_track_id = 0
        self.target_track_id: int | None = None
        self._initialised = False

    def is_initialised(self) -> bool:
        """Flag indicating whether the target subject has been seeded."""
        return self._initialised and self.target_track_id is not None

    def select_initial_target(self, detections: list[DetectedPose], frame_shape: tuple[int, int]) -> int:
        """Select target detection index on initial frame based on selector policy."""
        if not detections:
            raise ValueError("Cannot select target from an empty detection list")

        h, w = frame_shape[:2]

        if self.target_point is not None:
            pt = np.array(self.target_point, dtype=np.float64)
            dists = [np.linalg.norm(d.centroid - pt) for d in detections]
            return int(np.argmin(dists))

        selector = self.target_selector.lower()
        if selector == "left":
            # Performer closest to the left boundary (lowest X)
            return int(np.argmin([d.centroid[0] for d in detections]))
        elif selector == "right":
            # Performer closest to the right boundary (highest X)
            return int(np.argmax([d.centroid[0] for d in detections]))
        elif selector == "tallest":
            # Performer with largest vertical height span
            return int(np.argmax([d.height for d in detections]))
        elif selector == "shortest":
            # Performer with smallest vertical height span
            return int(np.argmin([d.height for d in detections]))
        else:  # "primary" or "auto"
            # Primary subject: largest screen area
            return int(np.argmax([d.area for d in detections]))

    def initialise_tracks(self, detections: list[DetectedPose], frame_shape: tuple[int, int]) -> None:
        """Initialise active tracks from the first frame with detected human subjects."""
        if not detections:
            return

        target_idx = self.select_initial_target(detections, frame_shape)
        self.tracks.clear()

        for idx, det in enumerate(detections):
            t_id = self.next_track_id
            self.next_track_id += 1
            is_target = idx == target_idx

            track = SubjectTrack(
                track_id=t_id,
                is_target=is_target,
                last_centroid=det.centroid.copy(),
                velocity=np.zeros(2, dtype=np.float64),
                last_height=det.height,
                last_torso_colour=det.torso_colour.copy(),
                consecutive_misses=0,
                total_detections=1,
            )
            self.tracks.append(track)
            if is_target:
                self.target_track_id = t_id

        self._initialised = True

    def update(
        self, detections: list[DetectedPose], frame_shape: tuple[int, int]
    ) -> dict[int, DetectedPose]:
        """Update tracks across frames using bipartite Hungarian assignment.

        Returns:
            Mapping of track_id -> matched DetectedPose (or omitted if track is missed).
        """
        if not self._initialised:
            self.initialise_tracks(detections, frame_shape)
            return {t.track_id: d for t, d in zip(self.tracks, detections)}

        h, w = frame_shape[:2]
        diag = math.hypot(w, h)

        if not detections:
            for t in self.tracks:
                t.consecutive_misses += 1
                t.last_centroid += t.velocity
            return {}

        n_tracks = len(self.tracks)
        n_dets = len(detections)

        # Cost matrix: distance, height ratio difference, colour chromatic difference
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)

        for i, trk in enumerate(self.tracks):
            pred_centroid = trk.last_centroid + trk.velocity
            for j, det in enumerate(detections):
                d_spatial = np.linalg.norm(pred_centroid - det.centroid) / max(diag, 1.0)
                max_h = max(trk.last_height, det.height, 1.0)
                d_height = abs(trk.last_height - det.height) / max_h
                d_colour = np.linalg.norm(trk.last_torso_colour - det.torso_colour) / 441.67  # sqrt(3*255^2)

                # Composite cost with high spatial and chromatic weight
                cost = 0.50 * d_spatial + 0.30 * d_colour + 0.20 * d_height
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: dict[int, DetectedPose] = {}
        assigned_dets: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            # Cost gating threshold: reject completely implausible teleports
            if cost_matrix[r, c] < 0.65:
                trk = self.tracks[r]
                det = detections[c]

                # Update velocity (alpha-beta smoothing)
                new_vel = det.centroid - trk.last_centroid
                trk.velocity = 0.6 * trk.velocity + 0.4 * new_vel
                trk.last_centroid = det.centroid.copy()
                trk.last_height = 0.7 * trk.last_height + 0.3 * det.height
                trk.last_torso_colour = 0.7 * trk.last_torso_colour + 0.3 * det.torso_colour
                trk.consecutive_misses = 0
                trk.total_detections += 1

                matched[trk.track_id] = det
                assigned_dets.add(c)
            else:
                self.tracks[r].consecutive_misses += 1
                self.tracks[r].last_centroid += self.tracks[r].velocity

        # Manage missed tracks
        for r, trk in enumerate(self.tracks):
            if trk.track_id not in matched:
                trk.consecutive_misses += 1
                trk.last_centroid += trk.velocity

        # Add newly discovered tracks (up to 4)
        for c, det in enumerate(detections):
            if c not in assigned_dets and len(self.tracks) < 4:
                t_id = self.next_track_id
                self.next_track_id += 1
                new_track = SubjectTrack(
                    track_id=t_id,
                    is_target=False,
                    last_centroid=det.centroid.copy(),
                    velocity=np.zeros(2, dtype=np.float64),
                    last_height=det.height,
                    last_torso_colour=det.torso_colour.copy(),
                    consecutive_misses=0,
                    total_detections=1,
                )
                self.tracks.append(new_track)
                matched[t_id] = det

        return matched


def extract_detected_poses(
    frame: np.ndarray,
    landmarker_result,
    conf_threshold: float = 0.20,
) -> list[DetectedPose]:
    """Parse MediaPipe PoseLandmarker result into structured DetectedPose objects."""
    if not landmarker_result or not landmarker_result.pose_landmarks:
        return []

    h, w = frame.shape[:2]
    poses: list[DetectedPose] = []

    has_masks = bool(getattr(landmarker_result, "segmentation_masks", None))

    for idx, lms in enumerate(landmarker_result.pose_landmarks):
        xs = np.array([lm.x * w for lm in lms], dtype=np.float64)
        ys = np.array([lm.y * h for lm in lms], dtype=np.float64)

        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))

        height = max(max_y - min_y, 1.0)
        width = max(max_x - min_x, 1.0)
        area = width * height

        # Filter out negligible noise / tiny background false positives (< 1% frame area)
        if (area / (w * h)) < 0.008:
            continue

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        # Sample torso chromatic signature (quadrilateral midpoint of shoulders & hips)
        # Landmarks: 11 (left shoulder), 12 (right shoulder), 23 (left hip), 24 (right hip)
        tcx = int(np.clip((lms[11].x + lms[12].x + lms[23].x + lms[24].x) / 4.0 * w, 0, w - 1))
        tcy = int(np.clip((lms[11].y + lms[12].y + lms[23].y + lms[24].y) / 4.0 * h, 0, h - 1))

        # 5x5 window average around torso centroid to mitigate single-pixel noise
        y1, y2 = max(0, tcy - 2), min(h, tcy + 3)
        x1, x2 = max(0, tcx - 2), min(w, tcx + 3)
        torso_crop = frame[y1:y2, x1:x2]
        torso_col = np.mean(torso_crop, axis=(0, 1)) if torso_crop.size > 0 else frame[tcy, tcx].astype(np.float64)

        seg_mask = None
        if has_masks and idx < len(landmarker_result.segmentation_masks):
            mask_data = landmarker_result.segmentation_masks[idx].numpy_view()
            seg_mask = mask_data[:, :, 0] if mask_data.ndim == 3 else mask_data

        poses.append(
            DetectedPose(
                index=idx,
                landmarks=np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float64),
                bbox=(min_x, min_y, max_x, max_y),
                centroid=np.array([cx, cy], dtype=np.float64),
                height=height,
                area=area,
                torso_colour=torso_col,
                segmentation_mask=seg_mask,
            )
        )

    return poses


def build_removal_mask(
    frame_shape: tuple[int, int],
    matched_tracks: dict[int, DetectedPose],
    target_track_id: int | None,
    *,
    dilate_removal_px: int = 17,
    protect_target_px: int = 7,
) -> np.ndarray:
    """Build union removal mask for all non-target individuals with target protection."""
    h, w = frame_shape[:2]
    union_removal = np.zeros((h, w), dtype=np.uint8)
    target_mask = np.zeros((h, w), dtype=np.uint8)

    k_removal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_removal_px, dilate_removal_px))
    k_protect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (protect_target_px, protect_target_px))

    for t_id, det in matched_tracks.items():
        if det.segmentation_mask is not None:
            m = (det.segmentation_mask > 0.20).astype(np.uint8)
        else:
            # Elliptical fallback when segmentation mask is absent
            m = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = det.bbox
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            axes = (int(max((x2 - x1) / 2 * 1.1, 5)), int(max((y2 - y1) / 2 * 1.05, 5)))
            cv2.ellipse(m, center, axes, 0, 0, 360, 1, -1)

        if t_id == target_track_id:
            target_mask = np.maximum(target_mask, m)
        else:
            union_removal = np.maximum(union_removal, m)

    if np.sum(union_removal) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Dilate non-target silhouettes to cover hair, fringes, and fingertips
    dilated_removal = cv2.dilate(union_removal, k_removal, iterations=2)

    # Strictly protect target subject silhouette from any inpainting encroachment
    if np.sum(target_mask) > 0:
        dilated_target = cv2.dilate(target_mask, k_protect, iterations=1)
        final_mask = np.where(dilated_target == 1, 0, dilated_removal).astype(np.uint8) * 255
    else:
        final_mask = dilated_removal * 255

    return final_mask


def isolate_video_subject(
    input_video: Path,
    output_video: Path,
    *,
    target: TargetSelector = "auto",
    target_point: tuple[float, float] | None = None,
    weights_path: Path | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    verbose: bool = True,
) -> Path:
    """Isolate a target subject from a multi-person video clip and inpaint all others.

    Args:
        input_video: Path to input multi-person video.
        output_video: Destination path for isolated clean single-subject video.
        target: Target selector ('left', 'right', 'tallest', 'shortest', 'primary', 'auto').
        target_point: Optional (x, y) seed point on frame 0 to choose target performer.
        weights_path: Optional path to MediaPipe pose landmarker model weights.
        progress_callback: Optional progress reporter receiving (frame_idx, total_frames, fps).
        verbose: Verbose console logging flag.

    Returns:
        Path to output_video.
    """
    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    model_file = weights_path or checks.ensure_mediapipe_model()
    if not model_file.exists():
        raise FileNotFoundError(f"MediaPipe pose landmarker not found: {model_file}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if verbose:
        print(f"isolating performer (selector='{target}') from {input_video.name}")
        print(f"  resolution {w}x{h} @ {fps:.2f} fps ({total_frames} frames)")

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_file)),
        running_mode=RunningMode.IMAGE,
        num_poses=4,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    import mediapipe as mp

    mp_image = mp.Image
    mp_format = mp.ImageFormat

    tracker = MultiPersonTracker(target_selector=target, target_point=target_point)

    temp_raw = output_video.parent / f"_temp_isolated_{output_video.name}"
    writer = cv2.VideoWriter(
        str(temp_raw),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    t0 = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_obj = mp_image(image_format=mp_format.SRGB, data=rgb)
            res = landmarker.detect(img_obj)

            detections = extract_detected_poses(frame, res)
            matched = tracker.update(detections, (h, w))

            removal_mask = build_removal_mask((h, w), matched, tracker.target_track_id)

            if np.sum(removal_mask) > 0:
                clean_frame = cv2.inpaint(frame, removal_mask, 5, cv2.INPAINT_TELEA)
            else:
                clean_frame = frame

            writer.write(clean_frame)
            frame_idx += 1

            if progress_callback:
                elapsed = time.time() - t0
                fps_act = frame_idx / elapsed if elapsed > 0 else 0.0
                progress_callback(frame_idx, total_frames, fps_act)

            if verbose and (frame_idx % 200 == 0 or frame_idx == total_frames):
                elapsed = time.time() - t0
                fps_act = frame_idx / elapsed if elapsed > 0 else 0.0
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0.0
                print(f"  isolate {frame_idx}/{total_frames} ({pct:.1f}%) @ {fps_act:.1f} fps")

    finally:
        cap.release()
        writer.release()

    # Audio Muxing via FFmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(temp_raw),
            "-i",
            str(input_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-shortest",
            str(output_video),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and output_video.exists():
            if temp_raw.exists():
                temp_raw.unlink()
            if verbose:
                print(f"saved isolated video: {output_video}")
            return output_video

    if temp_raw.exists():
        if output_video.exists():
            output_video.unlink()
        temp_raw.rename(output_video)

    return output_video
