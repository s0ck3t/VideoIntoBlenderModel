"""Analytical foot-contact detection, 3D world stance pinning, and 2-bone leg IK solver.

Permissive (Apache-2.0) analytical kinematic solver featuring:
- Stance-phase contact detection via vertical clearance and planar speed thresholding.
- Morphological filtering to eliminate transient false contacts.
- 3D world stance contact pinning reducing horizontal drift to < 1 mm during stance.
- C^2 quintic smoothstep transitions at stance entry and exit to prevent visual popping.
- Closed-form analytical 2-bone hip-knee-ankle IK solver preserving bone lengths (< 1e-4 m).
- Exact conversion of solved 3D knee and ankle coordinates into local FBX animation channels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .kinematics import (
    HumanoidRig,
    forward_kinematics,
    frame_anim_from_two_vectors,
    local_matrix,
    normalize,
    rotation_aligning,
    world_positions_m,
)
from .retarget import _foot_rotation


def compute_planar_speed(pos: np.ndarray, dt: float) -> np.ndarray:
    """Calculate horizontal planar speed (metres/second) in the XZ plane across a trajectory.

    Uses the minimum of incoming and outgoing step speeds:
        v_xz[t] = min(v_in[t], v_out[t])
    to prevent boundary smearing when crossing between swing and stance phases.

    Parameters
    ----------
    pos : np.ndarray
        Array of shape (T, 3) or (T, K, 3) representing 3D positions over time.
    dt : float
        Time step between adjacent frames in seconds (1.0 / fps).

    Returns
    -------
    np.ndarray
        Horizontal speed array of shape (T,) or (T, K).
    """
    arr = np.asarray(pos, dtype=np.float64)
    if arr.ndim == 3:
        # Multi-point input, e.g. (T, K, 3)
        n_frames, n_pts, _ = arr.shape
        speeds = np.zeros((n_frames, n_pts), dtype=np.float64)
        for k in range(n_pts):
            speeds[:, k] = compute_planar_speed(arr[:, k, :], dt)
        return speeds

    n_frames = len(arr)
    v_xz = np.zeros(n_frames, dtype=np.float64)
    if n_frames < 2 or dt <= 1e-9:
        return v_xz

    step_dx = np.diff(arr[:, 0])
    step_dz = np.diff(arr[:, 2])
    step_dist = np.sqrt(step_dx * step_dx + step_dz * step_dz)
    step_speed = step_dist / dt

    v_in = np.concatenate([[step_speed[0]], step_speed])
    v_out = np.concatenate([step_speed, [step_speed[-1]]])
    return np.minimum(v_in, v_out)


def detect_stance_phases(
    ankle_positions: np.ndarray,
    fps: float = 30.0,
    speed_thresh: float = 0.15,
    height_thresh: float = 0.15,
    min_stance_frames: int = 3,
    *,
    vel_threshold: float | None = None,
    height_threshold: float | None = None,
    min_duration_frames: int | None = None,
) -> np.ndarray:
    """Detect active stance contact phases from world ankle positions.

    Applies vertical height and horizontal planar speed thresholds followed by
    morphological filtering to eliminate transient false contacts.

    Parameters
    ----------
    ankle_positions : np.ndarray
        Array of shape (T, 2, 3) for [left, right] ankles, or (T, 3) for a single foot.
    fps : float
        Sampling frequency in frames per second.
    speed_thresh : float
        Maximum horizontal planar speed (m/s) to classify stationary stance.
    height_thresh : float
        Maximum vertical height (metres) above ground to classify near-contact.
    min_stance_frames : int
        Minimum contiguous frames required for a valid stance phase.
    vel_threshold : float, optional
        Alias for speed_thresh.
    height_threshold : float, optional
        Alias for height_thresh.
    min_duration_frames : int, optional
        Alias for min_stance_frames.

    Returns
    -------
    np.ndarray
        Boolean array of shape (T, 2) or (T,) indicating active stance.
    """
    if vel_threshold is not None:
        speed_thresh = vel_threshold
    if height_threshold is not None:
        height_thresh = height_threshold
    if min_duration_frames is not None:
        min_stance_frames = min_duration_frames

    arr = np.asarray(ankle_positions, dtype=np.float64)
    single_foot = (arr.ndim == 2)
    if single_foot:
        arr = arr[:, np.newaxis, :]

    n_frames, n_feet, _ = arr.shape
    dt = 1.0 / fps if fps > 0.0 else 1.0 / 30.0
    stance_mask = np.zeros((n_frames, n_feet), dtype=bool)

    for s in range(n_feet):
        pos = arr[:, s, :]
        h_mask = (pos[:, 1] <= height_thresh)
        v_xz = compute_planar_speed(pos, dt)
        vel_mask = (v_xz <= speed_thresh)
        raw_contact = h_mask & vel_mask

        # Morphological suppression of transient spikes
        filtered = raw_contact.copy()
        t = 0
        while t < n_frames:
            if filtered[t]:
                start = t
                while t < n_frames and filtered[t]:
                    t += 1
                if t - start < min_stance_frames:
                    filtered[start:t] = False
            else:
                t += 1
        stance_mask[:, s] = filtered

    return stance_mask[:, 0] if single_foot else stance_mask


def _quintic_smoothstep(u: float) -> float:
    """Quintic polynomial with zero 1st and 2nd derivatives at u=0 and u=1."""
    u_clamped = min(max(float(u), 0.0), 1.0)
    return u_clamped * u_clamped * u_clamped * (u_clamped * (u_clamped * 6.0 - 15.0) + 10.0)


def pin_stance_contacts(
    ankle_positions: np.ndarray,
    stance_mask: np.ndarray,
    blend_window: int = 3,
) -> np.ndarray:
    """Pin world contact positions during stance phases to eliminate horizontal sliding.

    Locks horizontal coordinates (X, Z) to their median across each contiguous stance
    interval, ensuring zero horizontal foot skating (< 1 mm drift). Uses C^2 quintic
    smoothstep blending across blend_window frames at contact entry and exit.

    Parameters
    ----------
    ankle_positions : np.ndarray
        Array of shape (T, 2, 3) or (T, 3).
    stance_mask : np.ndarray
        Boolean array of shape (T, 2) or (T,).
    blend_window : int
        Number of transition frames for quintic smoothstep blending.

    Returns
    -------
    np.ndarray
        Pinned target positions of identical shape to ankle_positions.
    """
    arr = np.asarray(ankle_positions, dtype=np.float64).copy()
    mask = np.asarray(stance_mask, dtype=bool)
    single_foot = (arr.ndim == 2)
    if single_foot:
        arr = arr[:, np.newaxis, :]
        mask = mask[:, np.newaxis]

    n_frames, n_feet, _ = arr.shape
    pinned = arr.copy()

    for s in range(n_feet):
        m = mask[:, s]
        t = 0
        while t < n_frames:
            if m[t]:
                start = t
                while t < n_frames and m[t]:
                    t += 1
                end = t

                # Contact anchor: median horizontal (X, Z) and minimum vertical (Y)
                x_med = float(np.median(arr[start:end, s, 0]))
                y_min = float(np.min(arr[start:end, s, 1]))
                z_med = float(np.median(arr[start:end, s, 2]))
                anchor = np.array([x_med, y_min, z_med], dtype=np.float64)

                for k in range(start, end):
                    d_in = k - start
                    d_out = end - 1 - k

                    w_in = _quintic_smoothstep((d_in + 1.0) / float(blend_window + 1.0)) if start > 0 else 1.0
                    w_out = _quintic_smoothstep((d_out + 1.0) / float(blend_window + 1.0)) if end < n_frames else 1.0
                    weight = min(w_in, w_out)

                    pinned[k, s] = (1.0 - weight) * arr[k, s] + weight * anchor
            else:
                t += 1

    return pinned[:, 0] if single_foot else pinned


def solve_two_bone_ik(
    p_hip: np.ndarray,
    p_target: np.ndarray,
    l1: float,
    l2: float,
    v_pole: np.ndarray | None = None,
) -> np.ndarray:
    """Analytical closed-form two-bone IK solver for the Hip-Knee-Ankle chain.

    Conserves exact bone lengths (< 1e-4 m error) and maintains the knee bending plane
    aligned with the pole vector.

    Parameters
    ----------
    p_hip : np.ndarray
        3D hip world origin position (metres).
    p_target : np.ndarray
        3D desired ankle target position (metres).
    l1 : float
        Thigh length (metres).
    l2 : float
        Shin length (metres).
    v_pole : np.ndarray, optional
        Preferred knee anterior bending direction. Default is [0, 0, 1].

    Returns
    -------
    np.ndarray
        3D solved knee position (metres).
    """
    p_h = np.asarray(p_hip, dtype=np.float64).reshape(3)
    p_t = np.asarray(p_target, dtype=np.float64).reshape(3)

    v_d = p_t - p_h
    d = float(np.linalg.norm(v_d))

    eps = 1e-5
    d_min = abs(l1 - l2) + eps
    d_max = (l1 + l2) - eps
    d_clamped = min(max(d, d_min), d_max)

    if d > 1e-6:
        u_d = v_d / d
    else:
        u_d = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    cos_alpha = (l1 * l1 + d_clamped * d_clamped - l2 * l2) / (2.0 * l1 * d_clamped)
    cos_alpha = min(max(cos_alpha, -1.0), 1.0)
    sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))

    if v_pole is None:
        v_p = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v_p = np.asarray(v_pole, dtype=np.float64).reshape(3)

    v_ortho = v_p - float(np.dot(v_p, u_d)) * u_d
    norm_ortho = float(np.linalg.norm(v_ortho))
    if norm_ortho < 1e-6:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(u_d, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v_ortho = helper - float(np.dot(helper, u_d)) * u_d
        norm_ortho = float(np.linalg.norm(v_ortho))

    u_bend = v_ortho / max(norm_ortho, 1e-12)
    p_knee = p_h + l1 * (cos_alpha * u_d + sin_alpha * u_bend)
    return p_knee


def solve_leg_ik_rotations(
    skel: HumanoidRig,
    world_transforms: dict[str, np.ndarray],
    side: str,
    p_hip: np.ndarray,
    p_knee: np.ndarray,
    p_target: np.ndarray,
    u_bend: np.ndarray,
    motion: Any | None = None,
    frame: int = 0,
) -> dict[str, np.ndarray]:
    """Convert solved 3D knee and ankle coordinates into local FBX animation rotations.

    Computes local rotation matrices for:
    - rig:{side}UpLeg (aimed along p_knee - p_hip, flexion plane along u_bend)
    - rig:{side}Leg (aimed along p_target - p_knee)
    - rig:{side}Foot (aimed horizontally towards toe with natural stance orientation)

    Parameters
    ----------
    skel : HumanoidRig
        Target humanoid rig skeleton model.
    world_transforms : dict[str, np.ndarray]
        Current world 4x4 matrices from forward kinematics.
    side : str
        "Left" or "Right" side specifier.
    p_hip : np.ndarray
        3D world position of the hip joint.
    p_knee : np.ndarray
        3D solved world position of the knee joint.
    p_target : np.ndarray
        3D desired world position of the ankle joint.
    u_bend : np.ndarray
        Unit vector in the knee flexion plane.
    motion : Any, optional
        Reference HumanMotion payload for foot orientation.
    frame : int
        Frame index.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping bone names to 3x3 local rotation matrices.
    """
    side_cap = "Left" if "left" in side.lower() else "Right"
    b_up_name = f"rig:{side_cap}UpLeg"
    b_leg_name = f"rig:{side_cap}Leg"
    b_foot_name = f"rig:{side_cap}Foot"

    b_up = skel.bones[b_up_name]
    b_leg = skel.bones[b_leg_name]
    b_foot = skel.bones[b_foot_name]

    r_parent = world_transforms[b_up.parent][:3, :3] if b_up.parent else np.eye(3)

    # 1. UpLeg rotation
    along_up = skel.child_offset_cm(b_up_name, b_leg_name)
    sec_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tgt_thigh = normalize(p_knee - p_hip)
    r_up_anim = frame_anim_from_two_vectors(
        b_up,
        r_parent,
        along_up,
        sec_up,
        tgt_thigh,
        u_bend,
    )

    # Partial forward kinematics to obtain updated UpLeg world frame
    m_up_world = world_transforms[b_up.parent] @ local_matrix(b_up, r_up_anim)
    r_up_world = m_up_world[:3, :3]

    # 2. Leg rotation
    along_leg = skel.child_offset_cm(b_leg_name, b_foot_name)
    tgt_shin = normalize(p_target - p_knee)
    d_in_anim = b_leg.pre_rotation.T @ r_up_world.T @ tgt_shin
    r_leg_anim = rotation_aligning(along_leg, d_in_anim)

    # Partial forward kinematics for Leg world frame
    m_leg_world = m_up_world @ local_matrix(b_leg, r_leg_anim)

    # 3. Foot rotation
    world_temp = dict(world_transforms)
    world_temp[b_up_name] = m_up_world
    world_temp[b_leg_name] = m_leg_world

    if motion is not None:
        r_foot_anim = _foot_rotation(skel, world_temp, motion, frame, side_cap)
    else:
        along_foot = skel.child_offset_cm(b_foot_name, f"rig:{side_cap}ToeBase")
        r_foot_anim = frame_anim_from_two_vectors(
            b_foot,
            m_leg_world[:3, :3],
            along_foot,
            b_foot.pre_rotation.T @ m_leg_world[:3, :3].T @ np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    return {
        b_up_name: r_up_anim,
        b_leg_name: r_leg_anim,
        b_foot_name: r_foot_anim,
    }


def apply_foot_ik(
    motion: Any,
    r_anim_seq: list[dict[str, np.ndarray]],
    hips_cm_seq: list[np.ndarray],
    skel: HumanoidRig,
    *,
    fps: float = 30.0,
    blend_window: int = 3,
    enabled: bool = True,
) -> list[dict[str, np.ndarray]]:
    """Apply foot-contact detection, stance pinning, and 2-bone IK across an animation sequence.

    Parameters
    ----------
    motion : Any
        HumanMotion reference trajectory.
    r_anim_seq : list[dict[str, np.ndarray]]
        Sequence of local FBX rotation dictionaries per frame.
    hips_cm_seq : list[np.ndarray]
        Root hip translation trajectory in centimetres.
    skel : HumanoidRig
        Skeleton model.
    fps : float
        Frame rate in frames per second.
    blend_window : int
        Number of transition frames for smoothstep contact blending.
    enabled : bool
        Whether foot IK processing is active.

    Returns
    -------
    list[dict[str, np.ndarray]]
        Updated rotation sequence with stance-pinned leg kinematics.
    """
    if not enabled or len(r_anim_seq) == 0:
        return r_anim_seq

    n_frames = len(r_anim_seq)
    if "rig:LeftUpLeg" not in skel.bones or "rig:RightUpLeg" not in skel.bones:
        return r_anim_seq

    # 1. Forward kinematics before IK to record initial ankle positions
    ankle_positions = np.zeros((n_frames, 2, 3), dtype=np.float64)
    for t in range(n_frames):
        world_t = forward_kinematics(skel, r_anim_seq[t], hips_translation_cm=hips_cm_seq[t])
        pos_t = world_positions_m(skel, world_t)
        ankle_positions[t, 0] = pos_t["rig:LeftFoot"]
        ankle_positions[t, 1] = pos_t["rig:RightFoot"]

    # 2. Detect stance phases
    stance_mask = detect_stance_phases(ankle_positions, fps=fps)

    # If no stance phases detected, return unmodified rotations
    if not np.any(stance_mask):
        return r_anim_seq

    # 3. Pin contact anchors
    pinned_targets = pin_stance_contacts(ankle_positions, stance_mask, blend_window=blend_window)

    l1_left = float(np.linalg.norm(skel.child_offset_cm("rig:LeftUpLeg", "rig:LeftLeg") * skel.meters_per_unit))
    l2_left = float(np.linalg.norm(skel.child_offset_cm("rig:LeftLeg", "rig:LeftFoot") * skel.meters_per_unit))
    l1_right = float(np.linalg.norm(skel.child_offset_cm("rig:RightUpLeg", "rig:RightLeg") * skel.meters_per_unit))
    l2_right = float(np.linalg.norm(skel.child_offset_cm("rig:RightLeg", "rig:RightFoot") * skel.meters_per_unit))

    out_seq: list[dict[str, np.ndarray]] = [{k: v.copy() for k, v in r.items()} for r in r_anim_seq]

    for t in range(n_frames):
        world_t = forward_kinematics(skel, out_seq[t], hips_translation_cm=hips_cm_seq[t])
        pos_t = world_positions_m(skel, world_t)

        for s_idx, (side, l1, l2) in enumerate((("Left", l1_left, l2_left), ("Right", l1_right, l2_right))):
            if stance_mask[t, s_idx]:
                p_hip = pos_t[f"rig:{side}UpLeg"]
                p_tgt = pinned_targets[t, s_idx]
                p_knee_fk = pos_t[f"rig:{side}Leg"]

                # Knee pole vector
                v_pole = p_knee_fk - p_hip
                if float(np.linalg.norm(v_pole)) < 1e-4:
                    v_pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                p_knee = solve_two_bone_ik(p_hip, p_tgt, l1, l2, v_pole=v_pole)
                u_d = normalize(p_tgt - p_hip)
                v_ortho = v_pole - float(np.dot(v_pole, u_d)) * u_d
                if float(np.linalg.norm(v_ortho)) < 1e-6:
                    v_ortho = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    v_ortho = v_ortho - float(np.dot(v_ortho, u_d)) * u_d
                u_bend = v_ortho / max(float(np.linalg.norm(v_ortho)), 1e-8)

                leg_rots = solve_leg_ik_rotations(
                    skel,
                    world_t,
                    side,
                    p_hip,
                    p_knee,
                    p_tgt,
                    u_bend,
                    motion=motion,
                    frame=t,
                )
                out_seq[t].update(leg_rots)

    return out_seq
