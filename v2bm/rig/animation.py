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

from .kinematics import forward_kinematics, human_to_rig, world_positions_m
from .retarget import joint_rig, retarget_one_frame
from .foot_ik import apply_foot_ik
from .hand_retarget import solve_hand_and_finger_rotations
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

    @property
    def n(self) -> int:
        return len(self.frames)


UPPER_LIMB_BONES = frozenset({
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
})


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
    )
    r_prev = q_prev = None
    raw_r_skeleton: list[dict[str, np.ndarray]] = []
    hips_list: list[np.ndarray] = []
    h_source = hand_data if hand_data is not None else getattr(motion, "hand_data", None)

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
                    rest_rig=None,
                )
                result.r_anim.update(h_rots)

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

    smoothed_r_skeleton = _smooth_rotation_trajectories(
        raw_r_skeleton, list(mapping.controlled), device=device
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

    rest_skel_pos = world_positions_m(skel, forward_kinematics(skel, {}))
    rest_sole_skel_m = max(0.0, min(float(p[1]) for p in rest_skel_pos.values()))
    rest_rig_pos = world_positions_m(
        mapping.target_rig, forward_kinematics(mapping.target_rig, {})
    )
    rest_sole_rig_m = max(0.0, min(float(p[1]) for p in rest_rig_pos.values()))

    r_rig_list: list[dict[str, np.ndarray]] = []
    hips_rig_list: list[np.ndarray] = []
    raw_y_skel_cm = np.zeros(len(frames), dtype=np.float64)
    raw_y_rig_cm = np.zeros(len(frames), dtype=np.float64)

    for i, frame in enumerate(frames):
        r_skel = smoothed_r_skeleton[i]
        hips_cm = hips_list[i].copy()
        skel_zero = world_positions_m(
            skel, forward_kinematics(skel, r_skel, hips_translation_cm=np.zeros(3))
        )
        min_rel_y_skel = min(float(p[1]) for p in skel_zero.values())
        raw_y_skel_cm[i] = (-min_rel_y_skel + rest_sole_skel_m) / skel.meters_per_unit

        r_rig = mapping.apply_rest_relative_rotations(r_skel)
        for b_name, b_rot in r_skel.items():
            if b_name not in r_rig and b_name in mapping.target_rig.bones:
                r_rig[b_name] = b_rot
        hips_rig = mapping.hips_translation_cm(hips_cm)
        rig_zero = mapping.humanoid_rig_positions_m(r_rig, np.zeros(3))
        min_rel_y_rig = min(float(p[1]) for p in rig_zero.values())
        raw_y_rig_cm[i] = (-min_rel_y_rig + rest_sole_rig_m) / mapping.target_rig.meters_per_unit

        r_rig_list.append(r_rig)
        hips_rig_list.append(hips_rig)

    # Smooth vertical hip height while guaranteeing non-penetration below ground plane Y = 0
    n_f = len(frames)
    smooth_y_skel_cm = raw_y_skel_cm.copy()
    smooth_y_rig_cm = raw_y_rig_cm.copy()
    if n_f >= 3:
        for t in range(n_f):
            t0, t1, t2 = max(0, t - 1), t, min(n_f - 1, t + 1)
            avg_skel = 0.25 * raw_y_skel_cm[t0] + 0.50 * raw_y_skel_cm[t1] + 0.25 * raw_y_skel_cm[t2]
            avg_rig = 0.25 * raw_y_rig_cm[t0] + 0.50 * raw_y_rig_cm[t1] + 0.25 * raw_y_rig_cm[t2]
            smooth_y_skel_cm[t] = max(avg_skel, raw_y_skel_cm[t])
            smooth_y_rig_cm[t] = max(avg_rig, raw_y_rig_cm[t])

    for i, frame in enumerate(frames):
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
            offset_m = np.array([hips_m[0], -min_h_y, hips_m[2]], dtype=np.float64)
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
    """Align vertical hip translations so the skinned character mesh rests cleanly on Y = 0."""
    if not asset.meshes or anim.n == 0:
        return
    skinned = skin_frames(asset, anim, list(range(anim.n)), device=device)
    raw_delta_m = np.array(
        [-float(np.concatenate(v, axis=0)[:, 1].min()) for v in skinned],
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

