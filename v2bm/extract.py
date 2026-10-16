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

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
) -> Path:
    """Encode a still photo into a short static hold clip."""
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
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for _ in range(n_frames):
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"FFmpeg failed writing {out_mp4}")
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
def temp_hold_video(image: Path) -> Iterator[Path]:
    """Context manager writing a temporary hold video deleted upon exit."""
    tmp = Path(tempfile.mkdtemp(prefix="v2bm-hold-"))
    try:
        yield still_to_hold_video(image, tmp / f"{Path(image).stem}.mp4")
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

    def __init__(self, model_path: Path | None = None, *, video_mode: bool = False) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        resolved_model = model_path or checks.ensure_mediapipe_model()
        if not resolved_model.exists():
            raise FileNotFoundError(
                f"MediaPipe pose landmarker model not found at {resolved_model}"
            )

        self.video_mode = video_mode
        mode = RunningMode.VIDEO if video_mode else RunningMode.IMAGE
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.mp_image_module = mp.Image
        self.mp_image_format = mp.ImageFormat

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
    work_dir: Path,
    device: str = "cpu",
    static_camera: bool = True,
    verbose: bool = True,
    source_image: Path | None = None,
) -> Path:
    """Extract 3D human motion and write the skeleton_motion.npz payload.

    Args:
        video: Path to input video or hold clip.
        output_npz: Destination path for skeleton_motion.npz.
        work_dir: Working cache directory.
        device: Device flag ('cpu', 'cuda', 'mps' - MediaPipe runs on CPU).
        static_camera: Flag indicating static camera perspective.
        verbose: Verbose logging output.
        source_image: Optional path if the run originated from a still photo.

    Returns:
        Path to the generated skeleton_motion.npz.
    """
    video = Path(video).resolve()
    is_still_source = source_image is not None or paths.is_image(video)
    actual_image = Path(source_image).resolve() if source_image is not None else (
        video if paths.is_image(video) else None
    )

    extractor = MediaPipeExtractor()

    # --------------------------------------------------------------------------
    # Case A: Static Still Photo Extraction
    # --------------------------------------------------------------------------
    if is_still_source and actual_image is not None and actual_image.exists():
        if verbose:
            print(f"extracting static 3D pose from still: {actual_image.name}")

        frame = cv2.imread(str(actual_image), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot decode image: {actual_image}")

        det = extractor.detect_frame(frame)
        if det is None:
            raise RuntimeError(
                f"MediaPipe failed to detect a human subject in image: {actual_image}"
            )

        w_coords, s_coords, vis = det
        single_pose_body24 = extractor.map_to_body24(w_coords, vis)

        # Enforce anatomical hinge range-of-motion clamping on the static pose
        from .stabilize_kp2d import clamp_anatomical_hinges

        single_pose_body24 = clamp_anatomical_hinges(single_pose_body24[None, ...])[0]

        # Duplicate static pose across HOLD_FRAMES to form an animated hold clip
        n_frames = HOLD_FRAMES
        fps = HOLD_FPS
        joints_3d = np.tile(single_pose_body24[None, ...], (n_frames, 1, 1))
        screen_lms = np.tile(s_coords[None, ...], (n_frames, 1, 1))
        world_lms = np.tile(w_coords[None, ...], (n_frames, 1, 1))

        # Root translation with foot ground pinning to Y = 0
        root_translation, root_orientation = estimate_root_trajectory_and_pinning(
            joints_3d, screen_lms, fps=fps
        )
        timestamps = np.arange(n_frames, dtype=np.float64) / float(fps)

    # --------------------------------------------------------------------------
    # Case B: Video Motion Sequence Extraction
    # --------------------------------------------------------------------------
    else:
        if not video.exists():
            raise FileNotFoundError(f"Input video not found: {video}")

        if verbose:
            print(f"extracting 3D motion from video: {video.name}")

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

        video_extractor = MediaPipeExtractor(video_mode=True)
        step_ms = max(1, int(round(1000.0 / (raw_fps if raw_fps > 1e-3 else 30.0))))

        world_list: list[np.ndarray | None] = []
        screen_list: list[np.ndarray | None] = []
        vis_list: list[np.ndarray | None] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = frame_idx * step_ms
            det = video_extractor.detect_frame(frame, timestamp_ms=ts_ms)
            if det is None:
                det = extractor.detect_frame(frame)
            if det is not None:
                w_coords, s_coords, vis = det
                world_list.append(w_coords)
                screen_list.append(s_coords)
                vis_list.append(vis)
            else:
                world_list.append(None)
                screen_list.append(None)
                vis_list.append(None)
            frame_idx += 1

        cap.release()
        n_frames = len(world_list)
        if n_frames == 0:
            raise RuntimeError(f"Video contains 0 decodable frames: {video}")

        # Interpolate frames where detection momentarily failed
        valid_indices = [i for i, w in enumerate(world_list) if w is not None]
        if not valid_indices:
            raise RuntimeError(
                f"No human pose detected across any of the {n_frames} frames in {video.name}"
            )

        # Fill leading missing frames from the first valid detection
        first_valid = valid_indices[0]
        for i in range(first_valid):
            world_list[i] = world_list[first_valid].copy()
            screen_list[i] = screen_list[first_valid].copy()
            vis_list[i] = vis_list[first_valid].copy()

        # Fill trailing missing frames from the last valid detection
        last_valid = valid_indices[-1]
        for i in range(last_valid + 1, n_frames):
            world_list[i] = world_list[last_valid].copy()
            screen_list[i] = screen_list[last_valid].copy()
            vis_list[i] = vis_list[last_valid].copy()

        # Linearly interpolate any interior gaps between valid detections
        for k in range(len(valid_indices) - 1):
            i0 = valid_indices[k]
            i1 = valid_indices[k + 1]
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

        # Map to Body24 24 joints in human coordinates
        raw_body24_joints = np.zeros((n_frames, 24, 3), dtype=np.float32)
        screen_lms = np.zeros((n_frames, 33, 3), dtype=np.float32)
        for i in range(n_frames):
            raw_body24_joints[i] = extractor.map_to_body24(world_list[i], vis_list[i])
            screen_lms[i] = screen_list[i]

        fps = playback_fps(video, n_frames, fallback=raw_fps or HOLD_FPS)
        timestamps = np.arange(n_frames, dtype=np.float64) / float(fps)

        # 1. Adaptive One-Euro (1€) filter on 3D coordinates
        filter_3d = OneEuroFilter3D(
            freq=fps,
            min_cutoff_xy=1.0,
            beta_xy=0.005,
            min_cutoff_z=0.5,
            beta_z=0.05,
        )
        filtered_joints = filter_3d.filter_trajectory(raw_body24_joints, rate=fps)

        # Apply stronger depth-jitter suppression on distal upper-limb joints
        # (elbows, wrists, hands: indices 18..23) where monocular Z variance is highest
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

        # 2. Three-stage biomechanical stabilisation:
        #    - Left/right limb swap prevention via hysteresis & Euclidean tracking
        #    - Rigid bone length hierarchical projection to temporal median rest lengths
        #    - Anatomical hinge range-of-motion clamping (knees & elbows)
        stabiliser = BiomechanicalStabiliser()
        joints_3d, swap_flags = stabiliser.stabilise(filtered_joints, screen_lms)

        if verbose and int(swap_flags.sum()) > 0:
            print(
                f"biomechanical stabiliser corrected {int(swap_flags.sum())}/{n_frames} "
                f"left/right bilateral limb flips"
            )

        # 3. Sliding-window 2D-reprojection-constrained IK trajectory optimisation
        from .ik_optimiser import optimise_trajectory_windowed

        world_lms = np.array(world_list, dtype=np.float32)
        vis_array = np.array(vis_list, dtype=np.float32)
        joints_3d = optimise_trajectory_windowed(
            joints_3d,
            landmarks_2d=screen_lms,
            visibilities=vis_array,
            fps=fps,
            verbose=verbose,
        )

        # 4. Camera-space root trajectory estimation & foot ground pinning to Y = 0
        root_translation, root_orientation = estimate_root_trajectory_and_pinning(
            joints_3d, screen_lms, fps=fps
        )

    # --------------------------------------------------------------------------
    # Save skeleton_motion.npz Payload
    # --------------------------------------------------------------------------
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        joints_3d=joints_3d.astype(np.float32),
        joint_names=np.array(BODY24_JOINT_NAMES),
        fps=np.float32(fps),
        timestamps=timestamps.astype(np.float64),
        root_translation=root_translation.astype(np.float32),
        root_orientation=root_orientation.astype(np.float32),
        video=np.array(str(video)),
    )
    if screen_lms is not None:
        payload["screen_landmarks"] = screen_lms.astype(np.float32)
    if "world_lms" in locals() and world_lms is not None:
        payload["world_landmarks"] = world_lms.astype(np.float32)
    if is_still_source and actual_image is not None:
        payload["image"] = np.array(str(actual_image))

    np.savez_compressed(output_npz, **payload)

    if verbose:
        print(f"saved {output_npz} ({len(joints_3d)} frames @ {fps:.2f} fps)")

    return output_npz
