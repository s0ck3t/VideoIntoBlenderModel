"""Load and query the skeleton_motion.npz produced by extract.py.

Human world coordinate frame:
    +Y = Up (against gravity)
    +X = Anatomical right
    -Z = Chest-forward (facing direction in capture frame)
    Units = Metres (m), right-handed system.

Seamlessly converted to the Humanoid FBX rest coordinate frame via R_BODY_ALIGN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HumanMotion:
    joints_3d: np.ndarray  # (F, J, 3) metres, Y-up human world space
    joint_names: list[str]
    fps: float
    timestamps: np.ndarray
    root_translation: np.ndarray | None
    path: Path
    root_orientation: np.ndarray | None = None
    world_landmarks: np.ndarray | None = None
    screen_landmarks: np.ndarray | None = None
    face_blendshapes: np.ndarray | None = None  # (F, 52) float32 in [0.0, 1.0]
    face_blendshape_names: list[str] | None = None  # (52,) ARKit names
    head_rotation_quat: np.ndarray | None = None  # (F, 4) float32 wxyz
    gaze_vectors: np.ndarray | None = None  # (F, 2, 3) float32
    framing: str | None = None  # "full_body" or "portrait"

    @property
    def n_frames(self) -> int:
        """Total frame count of the motion sequence."""
        return int(self.joints_3d.shape[0])

    def index(self, name: str) -> int:
        """Return the integer joint index for a standard Body24 joint name."""
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"joint {name!r} not in {self.joint_names}") from exc

    def joint(self, frame: int, name: str) -> np.ndarray:
        """Return the (3,) 3D position of joint `name` at frame index `frame`."""
        return np.asarray(self.joints_3d[frame, self.index(name)], dtype=np.float64)


def load_human_motion(path: Path) -> HumanMotion:
    """Load extracted skeletal motion payload from a skeleton_motion.npz archive.

    Also automatically discovers companion face_motion.npz in the same directory,
    or reads embedded facial channels if present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the extraction step first.")
    data = np.load(path, allow_pickle=True)
    joints = np.asarray(data["joints_3d"], dtype=np.float64)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joints_3d must be (F, J, 3), got {joints.shape}")
    names = [str(x) for x in data["joint_names"]]
    fps = float(data["fps"]) if "fps" in data.files else 30.0
    if "timestamps" in data.files:
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    else:
        timestamps = np.arange(len(joints), dtype=np.float64) / fps
    root = (
        np.asarray(data["root_translation"], dtype=np.float64)
        if "root_translation" in data.files
        else None
    )
    root_orient = (
        np.asarray(data["root_orientation"], dtype=np.float64)
        if "root_orientation" in data.files
        else None
    )
    world_lms = (
        np.asarray(data["world_landmarks"], dtype=np.float64)
        if "world_landmarks" in data.files
        else None
    )
    screen_lms = (
        np.asarray(data["screen_landmarks"], dtype=np.float64)
        if "screen_landmarks" in data.files
        else None
    )

    # Load facial motion payload from companion face_motion.npz or embedded keys
    face_bs = None
    face_names = None
    head_quats = None
    gaze_vecs = None
    framing = None

    face_npz = path.parent / "face_motion.npz"
    if face_npz.exists():
        try:
            with np.load(face_npz, allow_pickle=True) as fdata:
                if "blendshapes" in fdata.files:
                    face_bs = np.asarray(fdata["blendshapes"], dtype=np.float32)
                if "blendshape_names" in fdata.files:
                    face_names = [str(x) for x in fdata["blendshape_names"]]
                if "head_rotation_quat" in fdata.files:
                    head_quats = np.asarray(fdata["head_rotation_quat"], dtype=np.float32)
                if "gaze_vectors" in fdata.files:
                    gaze_vecs = np.asarray(fdata["gaze_vectors"], dtype=np.float32)
                if "framing" in fdata.files:
                    framing = str(fdata["framing"])
        except Exception:
            pass

    # Fallback to embedded keys in skeleton_motion.npz if companion missed any fields
    if face_bs is None and "face_blendshapes" in data.files:
        face_bs = np.asarray(data["face_blendshapes"], dtype=np.float32)
    if face_names is None and "face_blendshape_names" in data.files:
        face_names = [str(x) for x in data["face_blendshape_names"]]
    if head_quats is None and "head_rotation_quat" in data.files:
        head_quats = np.asarray(data["head_rotation_quat"], dtype=np.float32)
    if gaze_vecs is None and "gaze_vectors" in data.files:
        gaze_vecs = np.asarray(data["gaze_vectors"], dtype=np.float32)
    if framing is None and "framing" in data.files:
        framing = str(data["framing"])

    return HumanMotion(
        joints_3d=joints,
        joint_names=names,
        fps=fps,
        timestamps=timestamps,
        root_translation=root,
        path=path,
        root_orientation=root_orient,
        world_landmarks=world_lms,
        screen_landmarks=screen_lms,
        face_blendshapes=face_bs,
        face_blendshape_names=face_names,
        head_rotation_quat=head_quats,
        gaze_vectors=gaze_vecs,
        framing=framing,
    )


def source_video_path(motion: HumanMotion) -> Path | None:
    """The input video recorded inside skeleton_motion.npz, if still on disk."""
    npz = np.load(motion.path, allow_pickle=True)
    if "video" in npz.files:
        path = Path(str(npz["video"]))
        if path.exists():
            return path
    return None


def source_image_path(motion: HumanMotion) -> Path | None:
    """The input still photo recorded inside skeleton_motion.npz, if still on disk."""
    npz = np.load(motion.path, allow_pickle=True)
    if "image" in npz.files:
        path = Path(str(npz["image"]))
        if path.exists():
            return path
    return None
