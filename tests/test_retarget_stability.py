"""Tier 2: Mathematical Verification — Retargeting Stability & Quaternion Continuity.

Verifies:
- twist_events == 0 (Zero unexpected 180° twist flips across continuous motion sequences)
- quat_sign_flips == 0 (Zero antipodal quaternion sign flips / C0 discontinuities in generated tracks)
- Continuity of swing-twist solver and aim stability
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from v2bm.human_motion import HumanMotion
from v2bm.rig.kinematics import (
    continue_quaternion,
    pick_closest_aim_twist,
)
from v2bm.rig.retarget import (
    FrameRetarget,
    retarget_one_frame,
)
from tests.test_fk_skin import create_synthetic_humanoid_rig

REPO_ROOT = Path(__file__).resolve().parent.parent


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz unit quaternion to 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


BODY24_JOINTS = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


def create_smooth_synthetic_motion(num_frames: int = 60, fps: float = 30.0) -> HumanMotion:
    """Generate a continuous, smooth human motion sequence in meters.

    Frame 0 is an upright pose, smoothly flexing arms over 60 frames.
    """
    joints_3d = np.zeros((num_frames, 24, 3), dtype=np.float64)

    # Base T-pose positions in meters (Human convention: +Y up, +X right, -Z forward or +Z forward)
    # Matching Humanoid proportions scaled to meters
    base_pos: dict[str, tuple[float, float, float]] = {
        "pelvis": (0.0, 1.02, 0.0),
        "left_hip": (-0.10, 0.96, 0.0),
        "right_hip": (0.10, 0.96, 0.0),
        "spine1": (0.0, 1.12, 0.0),
        "left_knee": (-0.10, 0.52, 0.0),
        "right_knee": (0.10, 0.52, 0.0),
        "spine2": (0.0, 1.24, 0.0),
        "left_ankle": (-0.10, 0.10, 0.0),
        "right_ankle": (0.10, 0.10, 0.0),
        "spine3": (0.0, 1.36, 0.0),
        "left_foot": (-0.10, 0.03, -0.14),
        "right_foot": (0.10, 0.03, -0.14),
        "neck": (0.0, 1.50, 0.0),
        "left_collar": (-0.05, 1.48, 0.0),
        "right_collar": (0.05, 1.48, 0.0),
        "head": (0.0, 1.60, 0.0),
        "left_shoulder": (-0.18, 1.48, 0.0),
        "right_shoulder": (0.18, 1.48, 0.0),
        "left_elbow": (-0.45, 1.48, 0.0),
        "right_elbow": (0.45, 1.48, 0.0),
        "left_wrist": (-0.72, 1.48, 0.0),
        "right_wrist": (0.72, 1.48, 0.0),
        "left_hand": (-0.85, 1.48, 0.0),
        "right_hand": (0.85, 1.48, 0.0),
    }

    joint_indices = {name: i for i, name in enumerate(BODY24_JOINTS)}

    for t in range(num_frames):
        theta = float(np.sin(2.0 * np.pi * t / num_frames) * 0.3)  # Smooth oscillation
        for name, pos in base_pos.items():
            idx = joint_indices[name]
            p = np.array(pos, dtype=np.float64)
            # Smoothly raise/lower forearms
            if "elbow" in name or "wrist" in name or "hand" in name:
                p[1] += theta * 0.2
                p[2] += theta * 0.1
            joints_3d[t, idx] = p

    timestamps = np.arange(num_frames, dtype=np.float64) / fps
    root_translation = joints_3d[:, 0, :].copy()

    return HumanMotion(
        joints_3d=joints_3d,
        joint_names=BODY24_JOINTS,
        fps=fps,
        timestamps=timestamps,
        root_translation=root_translation,
        path=Path("synthetic_motion.npz"),
    )


def test_continue_quaternion_so3_and_geodesic_continuity() -> None:
    """Verify continue_quaternion enforces geodesic continuity without altering SO(3) rotation."""
    # 1. Verify antipodal quaternions produce the exact same rotation matrix
    q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
    r_pos = quat_wxyz_to_matrix(q)
    r_neg = quat_wxyz_to_matrix(-q)
    diff = float(np.linalg.norm(r_pos - r_neg))
    assert diff < 1e-12, f"Antipodal quaternions must yield identical SO(3) matrices: {diff:.3e}"

    # 2. Verify sign flip detection when dot product is negative
    q_prev = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_curr = np.array([-0.99, 0.05, 0.0, 0.0], dtype=np.float64)
    q_curr /= np.linalg.norm(q_curr)

    q_cont, did_flip = continue_quaternion(q_curr, q_prev)
    assert did_flip is True, "Expected did_flip=True when dot(q_curr, q_prev) < 0"
    assert float(np.dot(q_cont, q_prev)) > 0.0, "Expected positive dot product after continuation"

    # 3. Verify no flip when already continuous
    q_cont_2, did_flip_2 = continue_quaternion(q_cont, q_prev)
    assert did_flip_2 is False, "Expected did_flip=False when dot(q_cont, q_prev) >= 0"
    assert np.allclose(q_cont_2, q_cont)


def test_pick_closest_aim_twist_stability() -> None:
    """Verify pick_closest_aim_twist avoids spurious 180° flips under small incremental motion."""
    # Test bone aiming along Y axis
    axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    r_prev = np.eye(3, dtype=np.float64)

    # Slight rotation of 5 degrees around X
    ang = np.radians(5.0)
    c, s = np.cos(ang), np.sin(ang)
    r_step = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    r_pick, did_twist = pick_closest_aim_twist(r_step, r_prev, axis)
    assert did_twist is False, "Small perturbation must not trigger a 180° twist flip"
    assert np.allclose(r_pick, r_step)


def test_retargeting_stability_zero_twist_and_sign_flips() -> None:
    """Verify zero twist flips and zero quaternion sign discontinuities across smooth motion."""
    rig = create_synthetic_humanoid_rig()
    motion = create_smooth_synthetic_motion(num_frames=60, fps=30.0)

    total_twist_flips: int = 0
    total_sign_flips: int = 0

    prev_r: dict[str, np.ndarray] | None = None
    prev_q: dict[str, np.ndarray] | None = None

    history_q: dict[str, list[np.ndarray]] = {name: [] for name in rig.order}

    for frame in range(motion.n_frames):
        res: FrameRetarget = retarget_one_frame(
            rig,
            motion,
            frame,
            prev_r_anim=prev_r,
            prev_q_anim=prev_q,
        )

        total_twist_flips += len(res.twist_flips)
        total_sign_flips += len(res.sign_flips)

        for name, q in res.q_anim.items():
            history_q[name].append(q)

        prev_r = res.r_anim_raw
        prev_q = res.q_anim

    # Criterion 1: twist_events == 0 across continuous motion
    assert total_twist_flips == 0, (
        f"Expected zero twist flips on smooth motion, got {total_twist_flips}"
    )

    # Criterion 2: Check resulting quaternion continuity for all bones
    # For every consecutive pair (q_t, q_{t-1}), dot product must be >= 0
    discontinuities = 0
    for name, q_list in history_q.items():
        for t in range(1, len(q_list)):
            dot_prod = float(np.dot(q_list[t], q_list[t - 1]))
            if dot_prod < -1e-6:
                discontinuities += 1

    assert discontinuities == 0, (
        f"Detected {discontinuities} antipodal quaternion sign flips in generated tracks "
        f"(expected quat_sign_flips == 0)"
    )
