"""Per-frame retarget of a whole clip: Human -> Humanoid Skeleton -> character rig.

    human skeleton_motion.npz
            |  retarget_one_frame (swing/aim, no bone scaling, no IK)
    Humanoid Skeleton R_anim
            |  rest-relative transfer from the T-pose calibration
    character rig R_anim  ->  FK  ->  linear blend skinning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, binary_opening

from .kinematics import (
    forward_kinematics,
    human_to_rig,
    normalize,
    quat_wxyz,
    rotation_aligning,
    world_positions_m,
)
from .retarget import _quat_wxyz_to_matrix, _slerp, joint_rig, retarget_one_frame
from .foot_ik import apply_foot_ik, solve_two_bone_ik
from .hand_retarget import (
    _rodrigues,
    get_athletic_guard_rotations,
    solve_hand_and_finger_rotations,
    solve_wrist_rotations,
)
from .skinned_mesh import HumanoidRigAsset, bone_palette, skin_mesh
from .rig_retarget import RigMapping


def _human_reference_hip_height_m(motion) -> float:
    """Estimate the human subject's standing hip height in metres from 3D leg segments."""
    joints = getattr(motion, "joints_3d", None)
    if joints is not None and len(joints) > 0:
        arr = np.asarray(joints, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[1] >= 12:
            l_len = (
                np.linalg.norm(arr[:, 4] - arr[:, 1], axis=-1)
                + np.linalg.norm(arr[:, 7] - arr[:, 4], axis=-1)
                + 0.4 * np.linalg.norm(arr[:, 10] - arr[:, 7], axis=-1)
            )
            r_len = (
                np.linalg.norm(arr[:, 5] - arr[:, 2], axis=-1)
                + np.linalg.norm(arr[:, 8] - arr[:, 5], axis=-1)
                + 0.4 * np.linalg.norm(arr[:, 11] - arr[:, 8], axis=-1)
            )
            leg_m = float(np.median(0.5 * (l_len + r_len)))
            if 0.45 <= leg_m <= 1.25:
                return leg_m
    rt = getattr(motion, "root_translation", None)
    if rt is not None and len(rt) > 0:
        p90 = float(np.percentile(rt[:, 1], 90))
        if p90 > 0.35:
            return p90
    return 0.82


def compute_hips_translation_cm(motion, frame: int, rig) -> np.ndarray:
    """Calculate absolute hips root translation in rig space (cm).

    Preserves character rest height firmly above ground plane Y = 0,
    incorporating camera-space root trajectory when available.
    """
    rest_hips_cm = rig.bones[rig.root].lcl_translation_cm.copy()
    if getattr(motion, "root_translation", None) is not None:
        rt = motion.root_translation
        if len(rt) > frame:
            ref_h_m = _human_reference_hip_height_m(motion)
            rest_h_m = float(rest_hips_cm[1]) * rig.meters_per_unit
            s_y = rest_h_m / ref_h_m if ref_h_m > 1e-4 else 1.0
            p_human = np.array([
                float(rt[frame, 0]) * s_y,
                float(rt[frame, 1]) * s_y,
                float(rt[frame, 2]) * s_y,
            ], dtype=np.float64)
            p_humanoid_m = human_to_rig(p_human)
            return p_humanoid_m / rig.meters_per_unit

    p_pelvis = motion.joint(frame, "pelvis")
    if np.linalg.norm(p_pelvis) >= 0.05:
        return human_to_rig(p_pelvis) / rig.meters_per_unit

    return rest_hips_cm


def _compute_airborne_lift_m(
    motion, frames: list[int], return_intervals: bool = False
) -> np.ndarray | tuple[np.ndarray, list[tuple[int, int]]]:
    """Estimate vertical airborne jump elevation (metres >= 0) from 2D screen landmarks.

    When all lower-body landmarks (hips, knees, ankles, heels, toes) rise significantly
    above the local stance ground envelope in screen space (e.g. during a flying kick
    or aerial jump), returns the corresponding elevation in metres. During normal stance,
    walking, or floor-work, returns 0.0 so the lowest contact point stays pinned to Y = 0.
    """
    n_out = len(frames)
    sl = getattr(motion, "screen_landmarks", None)
    if sl is None or n_out < 16:
        zeros = np.zeros(n_out, dtype=np.float64)
        return (zeros, []) if return_intervals else zeros

    arr = np.asarray(sl, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 33 or max(frames, default=0) >= len(arr):
        zeros = np.zeros(n_out, dtype=np.float64)
        return (zeros, []) if return_intervals else zeros

    sub = arr[frames]
    lower_idxs = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    max_screen_y = np.clip(sub[:, lower_idxs, 1].max(axis=1), 0.0, 2.50)

    neck_2d = 0.5 * (sub[:, 11, :2] + sub[:, 12, :2])
    pelvis_2d = 0.5 * (sub[:, 23, :2] + sub[:, 24, :2])
    torso_2d = np.linalg.norm(neck_2d - pelvis_2d, axis=-1)
    torso_2d_filtered = np.convolve(
        np.pad(torso_2d, 2, mode="edge"),
        np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64),
        mode="valid",
    )
    med_torso_2d = np.maximum(torso_2d_filtered, 0.12)
    meters_per_unit_2d = 0.52 / med_torso_2d

    fps = float(getattr(motion, "fps", 24.0) or 24.0)
    dt = 1.0 / max(fps, 1.0)

    l_foot_pts = sub[:, [27, 29, 31], :2]
    r_foot_pts = sub[:, [28, 30, 32], :2]

    def _foot_planar_speed(pts: np.ndarray) -> np.ndarray:
        c = pts.mean(axis=1)
        v = np.zeros(n_out, dtype=np.float64)
        if n_out > 2:
            v[1:-1] = np.linalg.norm(c[2:] - c[:-2], axis=1) / (2.0 * dt)
            v[0] = v[1]
            v[-1] = v[-2]
        return v

    v_l = _foot_planar_speed(l_foot_pts)
    v_r = _foot_planar_speed(r_foot_pts)
    v_min = np.minimum(v_l, v_r)

    j3d = getattr(motion, "joints_3d", None)
    if j3d is not None and len(j3d) > 0 and np.asarray(j3d).shape[1] >= 12:
        j_sub = np.asarray(j3d, dtype=np.float64)[frames]
        l_ext = -j_sub[:, [7, 10], 1].min(axis=1)
        r_ext = -j_sub[:, [8, 11], 1].min(axis=1)
        max_standing_ext = max(
            float(np.percentile(l_ext, 90)),
            float(np.percentile(r_ext, 90)),
            0.65,
        )
        is_contact_l = (l_ext >= 0.82 * max_standing_ext) & (v_l <= 0.65)
        is_contact_r = (r_ext >= 0.82 * max_standing_ext) & (v_r <= 0.65)
    else:
        is_contact_l = v_l <= 0.35
        is_contact_r = v_r <= 0.35

    # Ground contact includes upright stance, crouched floor contact, and close-range camera framing
    is_crouch_contact = (v_min <= 0.25) & (pelvis_2d[:, 1] >= 0.62)
    is_close_contact = max_screen_y >= 1.15
    is_grounded = is_contact_l | is_contact_r | is_crouch_contact | is_close_contact

    flight_mask = ~is_grounded
    flight_clean = binary_closing(flight_mask, structure=np.ones(5))
    flight_clean = binary_opening(flight_clean, structure=np.ones(8))

    flight_intervals: list[tuple[int, int]] = []
    t = 0
    while t < n_out:
        if flight_clean[t]:
            t_start = t
            while t < n_out and flight_clean[t]:
                t += 1
            t_end = t - 1
            flight_intervals.append((t_start, t_end))
        else:
            t += 1

    grounded_indices = np.where(~flight_clean)[0]
    if len(grounded_indices) == 0:
        grounded_indices = np.arange(n_out)

    win = max(16, int(round(2.0 * fps)))
    ground_y = np.zeros(n_out, dtype=np.float64)
    for t in range(n_out):
        t0 = max(0, t - win)
        t1 = min(n_out, t + win + 1)
        local_grounded = grounded_indices[
            (grounded_indices >= t0) & (grounded_indices <= t1)
        ]
        if len(local_grounded) > 0:
            ground_y[t] = float(np.percentile(max_screen_y[local_grounded], 85))
        else:
            nearest_idx = grounded_indices[np.argmin(np.abs(grounded_indices - t))]
            ground_y[t] = float(max_screen_y[nearest_idx])

    # Interpolate ground baseline across aerial flight gaps to eliminate window collapse
    for (t_start, t_end) in flight_intervals:
        g_takeoff = ground_y[max(0, t_start - 1)]
        g_touchdown = ground_y[min(n_out - 1, t_end + 1)]
        span = max(1, t_end - t_start + 2)
        for k in range(t_start, t_end + 1):
            alpha = (k - (t_start - 1)) / float(span)
            ground_y[k] = (1.0 - alpha) * g_takeoff + alpha * g_touchdown

    deadband = 0.05
    raw_lift_m = np.maximum(0.0, ground_y - max_screen_y - deadband) * meters_per_unit_2d
    raw_lift_m[~flight_clean] = 0.0

    if n_out >= 5:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
        lift_m = np.convolve(np.pad(raw_lift_m, 2, mode="edge"), kernel, mode="valid")
    else:
        lift_m = raw_lift_m
    lift_m[lift_m < 0.02] = 0.0

    if return_intervals:
        return lift_m, flight_intervals
    return lift_m


@dataclass
class RetargetedAnimation:
    frames: list[int]
    fps: float
    r_anim_skeleton: list[dict[str, np.ndarray]]
    r_anim_rig: list[dict[str, np.ndarray]]
    hips_cm_skeleton: list[np.ndarray]
    hips_cm_rig: list[np.ndarray]
    rig_skel_pos: list[dict[str, np.ndarray]]
    rig_pos: list[dict[str, np.ndarray]]
    human_pos: list[dict[str, np.ndarray]]
    twist_events: list[dict] = field(default_factory=list)
    sign_events: list[dict] = field(default_factory=list)
    morph_weights: np.ndarray | None = None  # (N, M) float32 in [0.0, 1.0]
    airborne_lift_m: np.ndarray | None = None  # (N,) float64 >= 0.0
    framing: str = "full_body"

    @property
    def n(self) -> int:
        return len(self.frames)


from .fbx_skeleton import FINGER_BONE_SUFFIXES, TWIST_BONE_SUFFIXES

UPPER_LIMB_BONES = frozenset(
    {
        "rig:LeftShoulder",
        "rig:RightShoulder",
        "rig:LeftArm",
        "rig:RightArm",
        "rig:LeftForeArm",
        "rig:RightForeArm",
        "rig:LeftHand",
        "rig:RightHand",
        "rig:LeftFoot",
        "rig:RightFoot",
    }
    | {f"rig:{s}" for s in TWIST_BONE_SUFFIXES}
    | {f"rig:{s}" for s in FINGER_BONE_SUFFIXES}
)


def _quat_wxyz_to_mat3(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to a 3x3 orthogonal rotation matrix."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        res = q0 + t * (q1 - q0)
        norm = float(np.linalg.norm(res))
        return res / norm if norm > 1e-12 else q0
    theta_0 = np.arccos(min(1.0, max(-1.0, dot)))
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s0 = np.cos(theta_t) - dot * sin_theta_t / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * q0 + s1 * q1
    norm = float(np.linalg.norm(res))
    return res / norm if norm > 1e-12 else q0


def _smooth_rotation_trajectories(
    r_seq: list[dict[str, np.ndarray]], bone_names: list[str], device: str = "cpu"
) -> list[dict[str, np.ndarray]]:
    """Zero-phase geodesic quaternion filter to suppress high-frequency limb jitter."""
    from ..cuda_accel import smooth_rotation_trajectories_cuda, use_cupy
    from .kinematics import continue_quaternion, quat_wxyz

    if use_cupy(device):
        return smooth_rotation_trajectories_cuda(r_seq, bone_names, UPPER_LIMB_BONES)

    n = len(r_seq)
    if n < 3:
        return r_seq

    out_seq: list[dict[str, np.ndarray]] = [{k: v.copy() for k, v in r.items()} for r in r_seq]
    w_upper = np.array([0.08, 0.24, 0.36, 0.24, 0.08], dtype=np.float64)
    w_body = np.array([0.0, 0.15, 0.70, 0.15, 0.0], dtype=np.float64)

    for bname in bone_names:
        q_track = np.zeros((n, 4), dtype=np.float64)
        q_prev = None
        for t in range(n):
            rmat = r_seq[t].get(bname, np.eye(3))
            q, _ = continue_quaternion(quat_wxyz(rmat), q_prev)
            q_track[t] = q
            q_prev = q

        # Skip if static across the clip
        if np.allclose(q_track, q_track[0], atol=1e-7):
            continue

        weights = w_upper if bname in UPPER_LIMB_BONES else w_body
        passes = 2 if bname in UPPER_LIMB_BONES else 1
        q_curr = q_track.copy()
        for _ in range(passes):
            q_next = np.zeros_like(q_curr)
            for t in range(n):
                acc = np.zeros(4, dtype=np.float64)
                q_ref = q_curr[t]
                for k_idx, offset in enumerate((-2, -1, 0, 1, 2)):
                    w = weights[k_idx]
                    if w <= 0.0:
                        continue
                    tk = min(max(t + offset, 0), n - 1)
                    qk = q_curr[tk]
                    if float(np.dot(q_ref, qk)) < 0.0:
                        qk = -qk
                    acc += w * qk
                norm_acc = float(np.linalg.norm(acc))
                q_next[t] = acc / norm_acc if norm_acc > 1e-12 else q_ref
            q_curr = q_next

        # Enforce physiological angular jump limit (<= 24.5 deg/frame) on limb tracks
        if bname in UPPER_LIMB_BONES:
            max_rad = np.radians(24.5)
            for _ in range(4):
                # Forward pass
                for t in range(1, n):
                    dot = float(np.dot(q_curr[t], q_curr[t - 1]))
                    if dot < 0.0:
                        q_curr[t] = -q_curr[t]
                        dot = -dot
                    ang = 2.0 * np.arccos(min(1.0, max(-1.0, dot)))
                    if ang > max_rad:
                        q_curr[t] = _slerp(q_curr[t - 1], q_curr[t], max_rad / ang)
                # Backward pass
                for t in range(n - 2, -1, -1):
                    dot = float(np.dot(q_curr[t], q_curr[t + 1]))
                    if dot < 0.0:
                        q_curr[t] = -q_curr[t]
                        dot = -dot
                    ang = 2.0 * np.arccos(min(1.0, max(-1.0, dot)))
                    if ang > max_rad:
                        q_curr[t] = _slerp(q_curr[t + 1], q_curr[t], max_rad / ang)

        for t in range(n):
            out_seq[t][bname] = _quat_wxyz_to_mat3(q_curr[t])

    return out_seq


def apply_floor_reach_ik(
    r_skeleton: list[dict[str, np.ndarray]],
    hips_list: list[np.ndarray],
    skel,
    fps: float = 30.0,
) -> list[dict[str, np.ndarray]]:
    """Adjust arm reach and scapular depression during grounded three-point landings.

    When the character is in a grounded crouch and a hand reaches near the floor plane
    (y_wrist < 0.42 m and bounded planar velocity), applies downward scapular depression
    to the active clavicle and closed-form two-bone reach IK to the arm so the wrist
    reaches within 25.0 cm of the floor plane without floor penetration.
    """
    out = [dict(r) for r in r_skeleton]
    n_frames = len(out)
    if n_frames == 0:
        return out

    wrist_positions = {"Left": np.zeros((n_frames, 3)), "Right": np.zeros((n_frames, 3))}
    for t in range(n_frames):
        wpos = world_positions_m(skel, forward_kinematics(skel, out[t], hips_translation_cm=hips_list[t]))
        if "rig:LeftHand" in wpos:
            wrist_positions["Left"][t] = wpos["rig:LeftHand"]
        if "rig:RightHand" in wpos:
            wrist_positions["Right"][t] = wpos["rig:RightHand"]

    dt = 1.0 / fps if fps > 0.0 else 1.0 / 30.0

    for side in ("Left", "Right"):
        w_name = f"rig:{side}Hand"
        sh_name = f"rig:{side}Shoulder"
        arm_name = f"rig:{side}Arm"
        fore_name = f"rig:{side}ForeArm"

        if (
            sh_name not in skel.bones
            or arm_name not in skel.bones
            or fore_name not in skel.bones
            or w_name not in skel.bones
        ):
            continue

        along_local = np.array([1.0, 0.0, 0.0]) if side == "Left" else np.array([-1.0, 0.0, 0.0])
        dep_axis = normalize(np.array([0.0, -0.3, -1.0 if side == "Left" else 1.0]))

        pos_arr = wrist_positions[side]

        for t in range(n_frames):
            p_wrist = pos_arr[t]
            y_wrist = float(p_wrist[1])

            # Check ground proximity
            if y_wrist >= 0.42:
                continue

            wpos_t = world_positions_m(skel, forward_kinematics(skel, out[t], hips_translation_cm=hips_list[t]))
            p_sh = wpos_t[sh_name]
            y_sh = float(p_sh[1])

            # Must be reaching downwards from shoulder
            if y_wrist >= y_sh - 0.22:
                continue

            t0, t1 = max(0, t - 1), min(n_frames - 1, t + 1)
            delta_t = (t1 - t0) * dt if t1 > t0 else dt
            v_xz = float(np.hypot(pos_arr[t1, 0] - pos_arr[t0, 0], pos_arr[t1, 2] - pos_arr[t0, 2]) / delta_t)

            if v_xz >= 0.65:
                continue

            w_dist = float(np.clip((0.42 - y_wrist) / 0.22, 0.0, 1.0))
            w_vel = float(np.clip((0.65 - v_xz) / 0.25, 0.0, 1.0))
            weight = w_dist * w_vel

            if weight <= 0.01:
                continue

            # 1. Scapular depression on active shoulder
            r_sh0 = out[t][sh_name]
            r_dep = _rodrigues(dep_axis, np.radians(30.0 * weight))
            out[t][sh_name] = r_sh0 @ r_dep

            # 2. Re-compute transforms after shoulder depression
            fk_mid = forward_kinematics(skel, out[t], hips_translation_cm=hips_list[t])
            wpos_mid = world_positions_m(skel, fk_mid)
            p_arm = wpos_mid[arm_name]
            p_fore = wpos_mid[fore_name]
            p_w_curr = wpos_mid[w_name]

            l1 = float(np.linalg.norm(p_fore - p_arm))
            l2 = float(np.linalg.norm(p_w_curr - p_fore))

            # Target wrist position pulled down towards floor contact
            target_y = (1.0 - weight) * y_wrist + weight * 0.15
            target_wrist = np.array([p_w_curr[0], target_y, p_w_curr[2]])

            v_pole = normalize(p_fore - p_arm)
            p_elbow_new = solve_two_bone_ik(p_arm, target_wrist, l1, l2, v_pole=v_pole)

            # Re-aim upper arm
            r_parent_arm = fk_mid[sh_name][:3, :3]
            u_arm_tgt = normalize(p_elbow_new - p_arm)
            u_arm_local = r_parent_arm.T @ u_arm_tgt
            out[t][arm_name] = rotation_aligning(along_local, u_arm_local)

            # Re-aim forearm
            fk_arm = forward_kinematics(skel, out[t], hips_translation_cm=hips_list[t])
            r_parent_fore = fk_arm[arm_name][:3, :3]
            u_fore_tgt = normalize(target_wrist - p_elbow_new)
            u_fore_local = r_parent_fore.T @ u_fore_tgt
            out[t][fore_name] = rotation_aligning(along_local, u_fore_local)

    return out


def build_animation(
    motion,
    mapping: RigMapping,
    frames: list[int],
    *,
    target_morph_names: list[str] | None = None,
    verbose: bool = True,
    device: str = "cpu",
    hand_data: Any | None = None,
    enable_foot_ik: bool = True,
) -> RetargetedAnimation:
    """Retarget Human -> Humanoid, then carry the rest-relative motion onto the rig."""
    from .kinematics import quat_wxyz

    skel = mapping.skeleton_rig
    anim = RetargetedAnimation(
        frames=frames,
        fps=float(motion.fps),
        r_anim_skeleton=[],
        r_anim_rig=[],
        hips_cm_skeleton=[],
        hips_cm_rig=[],
        rig_skel_pos=[],
        rig_pos=[],
        human_pos=[],
        framing=str(getattr(motion, "framing", "full_body") or "full_body"),
    )
    r_prev = q_prev = None
    raw_r_skeleton: list[dict[str, np.ndarray]] = []
    hips_list: list[np.ndarray] = []
    h_source = hand_data if hand_data is not None else getattr(motion, "hand_data", None)

    core_arm_bones = frozenset({
        "rig:LeftForeArm",
        "rig:RightForeArm",
    })
    prev_aux_left: dict[str, np.ndarray] = {}
    prev_aux_right: dict[str, np.ndarray] = {}
    q_ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    guard_left_mats = get_athletic_guard_rotations("Left", rest_rig=mapping.target_rig)
    guard_right_mats = get_athletic_guard_rotations("Right", rest_rig=mapping.target_rig)
    q_guard_left = {k: quat_wxyz(m) for k, m in guard_left_mats.items()}
    q_guard_right = {k: quat_wxyz(m) for k, m in guard_right_mats.items()}
    has_target_fingers = bool(guard_left_mats or guard_right_mats)

    for i, frame in enumerate(frames):
        result = retarget_one_frame(skel, motion, frame, prev_r_anim=r_prev, prev_q_anim=q_prev)

        # Integrate hand and finger retargeting when landmarks are available
        if h_source is not None:
            l_lms = None
            r_lms = None
            if hasattr(h_source, "left_landmarks_3d") and hasattr(h_source, "left_present"):
                l_lms = h_source.left_landmarks_3d[frame] if h_source.left_present[frame] else None
                r_lms = h_source.right_landmarks_3d[frame] if h_source.right_present[frame] else None
            elif hasattr(h_source, "landmarks") and isinstance(h_source.landmarks, dict):
                l_val = h_source.landmarks.get("left")
                r_val = h_source.landmarks.get("right")
                l_lms = l_val[frame] if l_val is not None and len(l_val) > frame else l_val
                r_lms = r_val[frame] if r_val is not None and len(r_val) > frame else r_val
            elif isinstance(h_source, dict):
                l_val = h_source.get("left")
                r_val = h_source.get("right")
                l_lms = l_val[frame] if l_val is not None and len(l_val) > frame else l_val
                r_lms = r_val[frame] if r_val is not None and len(r_val) > frame else r_val

            if l_lms is not None or r_lms is not None:
                h_rots = solve_hand_and_finger_rotations(
                    hand_data_left=l_lms,
                    hand_data_right=r_lms,
                    motion=motion,
                    frame=frame,
                    rest_rig=mapping.target_rig,
                )
                # Apply auxiliary twist, 3-DoF wrist & finger bones without overwriting elbow swing
                for b_key, b_mat in h_rots.items():
                    if b_key in core_arm_bones:
                        continue
                    result.r_anim[b_key] = b_mat
                    if "Left" in b_key:
                        prev_aux_left[b_key] = b_mat
                    elif "Right" in b_key:
                        prev_aux_right[b_key] = b_mat

            # Smoothly decay auxiliary hand/finger channels towards athletic guard posture across dropouts
            if l_lms is None and has_target_fingers:
                if not prev_aux_left:
                    for b_key, b_mat in guard_left_mats.items():
                        prev_aux_left[b_key] = b_mat
                        result.r_anim[b_key] = b_mat
                else:
                    for b_key, b_mat in list(prev_aux_left.items()):
                        q_target = q_guard_left.get(b_key, q_ident)
                        q_dec = _slerp(quat_wxyz(b_mat), q_target, 0.20)
                        mat_dec = _quat_wxyz_to_matrix(q_dec)
                        prev_aux_left[b_key] = mat_dec
                        result.r_anim[b_key] = mat_dec
            if r_lms is None and has_target_fingers:
                if not prev_aux_right:
                    for b_key, b_mat in guard_right_mats.items():
                        prev_aux_right[b_key] = b_mat
                        result.r_anim[b_key] = b_mat
                else:
                    for b_key, b_mat in list(prev_aux_right.items()):
                        q_target = q_guard_right.get(b_key, q_ident)
                        q_dec = _slerp(quat_wxyz(b_mat), q_target, 0.20)
                        mat_dec = _quat_wxyz_to_matrix(q_dec)
                        prev_aux_right[b_key] = mat_dec
                        result.r_anim[b_key] = mat_dec
        else:
            # Fallback 3-DoF wrist articulation from Pose world landmarks if available
            wl = getattr(motion, "world_landmarks", None)
            if wl is not None and len(wl) > frame and wl.shape[1] >= 23:
                l_raw = wl[frame, [15, 19, 17, 21]]
                l_hum = np.array([l_raw[:, 0], -l_raw[:, 1], -l_raw[:, 2]]).T
                r_raw = wl[frame, [16, 20, 18, 22]]
                r_hum = np.array([r_raw[:, 0], -r_raw[:, 1], -r_raw[:, 2]]).T
                rw_l = solve_wrist_rotations(l_hum, "Left", rest_rig=mapping.target_rig)
                rw_r = solve_wrist_rotations(r_hum, "Right", rest_rig=mapping.target_rig)
                if "rig:LeftHand" in mapping.target_rig.bones:
                    result.r_anim["rig:LeftHand"] = rw_l
                if "rig:RightHand" in mapping.target_rig.bones:
                    result.r_anim["rig:RightHand"] = rw_r

            # When finger landmarks are not tracked, maintain canonical athletic guard posture for articulated hands
            if has_target_fingers:
                for b_key, b_mat in guard_left_mats.items():
                    result.r_anim[b_key] = b_mat
                for b_key, b_mat in guard_right_mats.items():
                    result.r_anim[b_key] = b_mat

        hips_cm = compute_hips_translation_cm(motion, frame, skel)
        raw_r_skeleton.append(result.r_anim)
        hips_list.append(hips_cm)
        anim.human_pos.append({n: joint_rig(motion, frame, n) for n in motion.joint_names})
        if result.twist_flips:
            anim.twist_events.append({"frame": frame, "bones": result.twist_flips})
        if result.sign_flips:
            anim.sign_events.append({"frame": frame, "bones": result.sign_flips})

        r_prev, q_prev = result.r_anim_raw, result.q_anim
        if verbose and (i == 0 or (i + 1) % 200 == 0 or i == len(frames) - 1):
            print(f"  retarget {i + 1}/{len(frames)}")

    all_bone_names = list(
        dict.fromkeys(
            list(mapping.controlled)
            + [b_key for r_frame in raw_r_skeleton for b_key in r_frame.keys()]
        )
    )
    smoothed_r_skeleton = _smooth_rotation_trajectories(
        raw_r_skeleton, all_bone_names, device=device
    )

    # Apply stance foot contact locking and closed-form 2-bone leg IK
    if enable_foot_ik:
        smoothed_r_skeleton = apply_foot_ik(
            motion,
            smoothed_r_skeleton,
            hips_list,
            skel,
            fps=float(motion.fps),
            enabled=enable_foot_ik,
        )

    # Apply three-point landing floor reach IK and scapular depression
    smoothed_r_skeleton = apply_floor_reach_ik(
        smoothed_r_skeleton,
        hips_list,
        skel,
        fps=float(motion.fps),
    )

    rest_skel_pos = world_positions_m(skel, forward_kinematics(skel, {}))
    rest_sole_skel_m = max(0.0, min(float(p[1]) for p in rest_skel_pos.values()))
    rest_rig_pos = world_positions_m(
        mapping.target_rig, forward_kinematics(mapping.target_rig, {})
    )
    rest_sole_rig_m = max(0.0, min(float(p[1]) for p in rest_rig_pos.values()))

    airborne_lift_m, flight_intervals = _compute_airborne_lift_m(
        motion, frames, return_intervals=True
    )

    r_rig_list: list[dict[str, np.ndarray]] = []
    hips_rig_list: list[np.ndarray] = []
    raw_y_skel_cm = np.zeros(len(frames), dtype=np.float64)
    raw_y_rig_cm = np.zeros(len(frames), dtype=np.float64)
    floor_min_skel_cm = np.zeros(len(frames), dtype=np.float64)
    floor_min_rig_cm = np.zeros(len(frames), dtype=np.float64)

    for i, frame in enumerate(frames):
        lift_m = float(airborne_lift_m[i])
        r_skel = smoothed_r_skeleton[i]
        hips_cm = hips_list[i].copy()
        skel_zero = world_positions_m(
            skel, forward_kinematics(skel, r_skel, hips_translation_cm=np.zeros(3))
        )
        min_rel_y_skel = min(float(p[1]) for p in skel_zero.values())
        floor_min_skel_cm[i] = (-min_rel_y_skel + rest_sole_skel_m) / skel.meters_per_unit
        raw_y_skel_cm[i] = (-min_rel_y_skel + rest_sole_skel_m + lift_m) / skel.meters_per_unit

        r_rig = mapping.apply_rest_relative_rotations(r_skel)
        for b_name, b_rot in r_skel.items():
            if b_name not in r_rig and b_name in mapping.target_rig.bones:
                r_rig[b_name] = b_rot
        hips_rig = mapping.hips_translation_cm(hips_cm)
        rig_zero = mapping.humanoid_rig_positions_m(r_rig, np.zeros(3))
        min_rel_y_rig = min(float(p[1]) for p in rig_zero.values())
        floor_min_rig_cm[i] = (-min_rel_y_rig + rest_sole_rig_m) / mapping.target_rig.meters_per_unit
        raw_y_rig_cm[i] = (-min_rel_y_rig + rest_sole_rig_m + lift_m) / mapping.target_rig.meters_per_unit

        r_rig_list.append(r_rig)
        hips_rig_list.append(hips_rig)

    # Fit smooth piecewise ballistic parabolic arc across confirmed aerial flight phases
    y_ballistic_skel = raw_y_skel_cm.copy()
    y_ballistic_rig = raw_y_rig_cm.copy()
    for (t0, t1) in flight_intervals:
        if t1 - t0 < 8:
            continue
        t_apex_skel = t0 + int(np.argmax(raw_y_skel_cm[t0 : t1 + 1]))
        y_peak_skel = raw_y_skel_cm[t_apex_skel]
        y0_skel = raw_y_skel_cm[t0]
        y1_skel = raw_y_skel_cm[t1]

        t_apex_rig = t0 + int(np.argmax(raw_y_rig_cm[t0 : t1 + 1]))
        y_peak_rig = raw_y_rig_cm[t_apex_rig]
        y0_rig = raw_y_rig_cm[t0]
        y1_rig = raw_y_rig_cm[t1]

        for k in range(t0, t1 + 1):
            if k <= t_apex_skel:
                denom_s = float((t_apex_skel - t0) ** 2) if t_apex_skel > t0 else 1.0
                arc_s = y_peak_skel - ((y_peak_skel - y0_skel) / denom_s) * float((k - t_apex_skel) ** 2)
            else:
                denom_s = float((t1 - t_apex_skel) ** 2) if t1 > t_apex_skel else 1.0
                arc_s = y_peak_skel - ((y_peak_skel - y1_skel) / denom_s) * float((k - t_apex_skel) ** 2)
            y_ballistic_skel[k] = max(arc_s, floor_min_skel_cm[k])

            if k <= t_apex_rig:
                denom_r = float((t_apex_rig - t0) ** 2) if t_apex_rig > t0 else 1.0
                arc_r = y_peak_rig - ((y_peak_rig - y0_rig) / denom_r) * float((k - t_apex_rig) ** 2)
            else:
                denom_r = float((t1 - t_apex_rig) ** 2) if t1 > t_apex_rig else 1.0
                arc_r = y_peak_rig - ((y_peak_rig - y1_rig) / denom_r) * float((k - t_apex_rig) ** 2)
            y_ballistic_rig[k] = max(arc_r, floor_min_rig_cm[k])

    # Smooth vertical hip height while guaranteeing strict non-penetration below ground plane Y = 0
    n_f = len(frames)
    smooth_y_skel_cm = y_ballistic_skel.copy()
    smooth_y_rig_cm = y_ballistic_rig.copy()
    if n_f >= 3:
        for t in range(n_f):
            t0, t1, t2 = max(0, t - 1), t, min(n_f - 1, t + 1)
            avg_skel = 0.25 * y_ballistic_skel[t0] + 0.50 * y_ballistic_skel[t1] + 0.25 * y_ballistic_skel[t2]
            avg_rig = 0.25 * y_ballistic_rig[t0] + 0.50 * y_ballistic_rig[t1] + 0.25 * y_ballistic_rig[t2]
            smooth_y_skel_cm[t] = max(avg_skel, floor_min_skel_cm[t])
            smooth_y_rig_cm[t] = max(avg_rig, floor_min_rig_cm[t])

    ballistic_lift_m = np.maximum(
        0.0,
        (smooth_y_rig_cm - floor_min_rig_cm) * mapping.target_rig.meters_per_unit,
    )
    anim.airborne_lift_m = ballistic_lift_m

    for i, frame in enumerate(frames):
        lift_m = float(ballistic_lift_m[i])
        r_skel = smoothed_r_skeleton[i]
        hips_cm = hips_list[i].copy()
        hips_cm[1] = smooth_y_skel_cm[i]
        mix_pos = world_positions_m(
            skel, forward_kinematics(skel, r_skel, hips_translation_cm=hips_cm)
        )
        r_rig = r_rig_list[i]
        hips_rig = hips_rig_list[i].copy()
        hips_rig[1] = smooth_y_rig_cm[i]
        rig_pos = mapping.humanoid_rig_positions_m(r_rig, hips_rig)

        # Anchor pelvis-centred human_pos onto the shared ground plane and XZ trajectory
        h_pos = anim.human_pos[i]
        if float(np.linalg.norm(h_pos.get("pelvis", np.zeros(3)))) < 0.05:
            min_h_y = min(float(p[1]) for p in h_pos.values())
            hips_m = hips_cm * skel.meters_per_unit
            offset_m = np.array([hips_m[0], -min_h_y + lift_m, hips_m[2]], dtype=np.float64)
            anim.human_pos[i] = {k: p + offset_m for k, p in h_pos.items()}

        anim.r_anim_skeleton.append(r_skel)
        anim.r_anim_rig.append(r_rig)
        anim.hips_cm_skeleton.append(hips_cm)
        anim.hips_cm_rig.append(hips_rig)
        anim.rig_skel_pos.append(mix_pos)
        anim.rig_pos.append(rig_pos)

    # Retarget facial blendshapes if present in motion and target names provided
    retargeted_morphs = None
    if getattr(motion, "face_blendshapes", None) is not None and target_morph_names is not None:
        from .retarget import retarget_facial_blendshapes

        bs = motion.face_blendshapes
        if bs.ndim == 2:
            if max(frames) < len(bs):
                bs = bs[frames]
            elif len(bs) == len(frames):
                pass
            else:
                bs = bs[:len(frames)]
        retargeted_morphs = retarget_facial_blendshapes(bs, target_morph_names)
    anim.morph_weights = retargeted_morphs

    return anim


def skin_frames(
    asset: HumanoidRigAsset,
    anim: RetargetedAnimation,
    indices: list[int],
    device: str = "cpu",
) -> list[list[np.ndarray]]:
    """Linear blend skinning of the character mesh (metres) for the given indices."""
    from ..cuda_accel import skin_frames_cuda, use_cupy

    scale = asset.rig.meters_per_unit
    if use_cupy(device) and len(indices) > 0 and len(asset.meshes) > 0:
        return skin_frames_cuda(asset, anim, indices)

    out: list[list[np.ndarray]] = []
    for i in indices:
        world = forward_kinematics(
            asset.rig, anim.r_anim_rig[i], hips_translation_cm=anim.hips_cm_rig[i]
        )
        frame_verts = []
        mw = anim.morph_weights[i] if anim.morph_weights is not None else None
        for mesh in asset.meshes:
            mesh_mw = mw if getattr(mesh, "morph_deltas_pos", None) is not None else None
            palette = bone_palette(world, mesh, asset.rig.order)
            v = skin_mesh(mesh, palette, morph_weights=mesh_mw, device=device) * scale
            frame_verts.append(v)
        out.append(frame_verts)
    return out


def pin_character_mesh_to_ground(
    asset: HumanoidRigAsset, anim: RetargetedAnimation, device: str = "cpu"
) -> None:
    """Align vertical hip translations so the skinned character mesh rests cleanly on Y = 0 (or airborne jump height)."""
    if not asset.meshes or anim.n == 0:
        return
    skinned = skin_frames(asset, anim, list(range(anim.n)), device=device)
    lift_arr = (
        anim.airborne_lift_m
        if anim.airborne_lift_m is not None and len(anim.airborne_lift_m) == anim.n
        else np.zeros(anim.n, dtype=np.float64)
    )
    raw_delta_m = np.array(
        [-float(np.concatenate(v, axis=0)[:, 1].min()) + float(lift_arr[idx]) for idx, v in enumerate(skinned)],
        dtype=np.float64,
    )
    smooth_delta_m = raw_delta_m.copy()
    if anim.n >= 3:
        for t in range(anim.n):
            t0, t1, t2 = max(0, t - 1), t, min(anim.n - 1, t + 1)
            avg = 0.25 * raw_delta_m[t0] + 0.50 * raw_delta_m[t1] + 0.25 * raw_delta_m[t2]
            smooth_delta_m[t] = max(avg, raw_delta_m[t])

    mpu = float(asset.rig.meters_per_unit)
    for i in range(anim.n):
        dy_m = float(smooth_delta_m[i])
        anim.hips_cm_rig[i][1] += dy_m / mpu
        shift_vec = np.array([0.0, dy_m, 0.0], dtype=np.float64)
        anim.rig_pos[i] = {k: p + shift_vec for k, p in anim.rig_pos[i].items()}


def core_bones_only(pos: dict[str, np.ndarray], edges) -> dict[str, np.ndarray]:
    """Drop the finger chains so the rig panel draws the same graph as the edges."""
    keep = {name for edge in edges for name in edge}
    return {name: p for name, p in pos.items() if name in keep}


def all_points(seq: list[dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack([p for pos in seq for p in pos.values()])

