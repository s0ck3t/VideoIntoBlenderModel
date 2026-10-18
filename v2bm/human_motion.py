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


# Canonical 24 Body24 joint order matching extract.py and downstream retargeting modules.
BODY24_JOINTS: tuple[str, ...] = (
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
)
BODY24_JOINT_NAMES: list[str] = list(BODY24_JOINTS)


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
    head_rotation_quat: np.ndarray | None = None

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
    """Load extracted skeletal motion payload from a skeleton_motion.npz archive."""
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
    head_rot_quat = (
        np.asarray(data["head_rotation_quat"], dtype=np.float64)
        if "head_rotation_quat" in data.files
        else None
    )
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
        head_rotation_quat=head_rot_quat,
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
