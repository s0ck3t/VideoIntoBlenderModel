"""Human → Humanoid per-frame retarget.

FK is T * Rpre * R_anim. Humanoid rest bone lengths are never scaled.
Optional previous-frame rotations pick the 180-degree twist closest to the
last frame and enforce quaternion sign continuity. No temporal smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import (
    HumanoidRig,
    anim_from_world_rotation,
    continue_quaternion,
    forward_kinematics,
    frame_anim_from_axes,
    frame_anim_from_two_vectors,
    human_to_rig,
    local_aim_axis,
    normalize,
    orthonormal_frame,
    pick_closest_aim_twist,
    quat_wxyz,
    swing_anim_to_direction,
    world_positions_m,
)

CONTROLLED = (
    "rig:Hips",
    "rig:Spine",
    "rig:Spine1",
    "rig:Spine2",
    "rig:Neck",
    "rig:Head",
    "rig:LeftShoulder",
    "rig:RightShoulder",
    "rig:LeftArm",
    "rig:RightArm",
    "rig:LeftForeArm",
    "rig:RightForeArm",
    "rig:LeftHand",
    "rig:RightHand",
    "rig:LeftUpLeg",
    "rig:RightUpLeg",
    "rig:LeftLeg",
    "rig:RightLeg",
    "rig:LeftFoot",
    "rig:RightFoot",
    "rig:LeftToeBase",
    "rig:RightToeBase",
)

# Humanoid bone → (human parent joint, human child joint) used as the aim vector.
# Spine2 uses a chest frame (not a single swing) so clavicle attachments keep the right twist.
#
# Neck/Head must not aim along Body24 neck→head. That vector is a short (~9 cm)
# skull-centre offset, often 50–70° from vertical, and flops Humanoid's longer
# neck/head bones. spine3→head is the stable head-up (skull above the chest)
# while still using the Human head joint. Joint mapping remains
# neck→rig:Neck and head→rig:Head.
#
# Humanoid Neck→Head and Head→HeadTop rest translations are not along local +Y
# (they include a forward +Z offset). Aiming that offset at head-up tilts the
# neck/head bones. Neck/Head therefore swing local +Y.
NECK_HEAD = ("rig:Neck", "rig:Head")
NECK_HEAD_AIM_LOCAL = np.array([0.0, 1.0, 0.0])

SWING_TARGETS = {
    "rig:Spine": ("spine1", "spine2"),
    "rig:Spine1": ("spine2", "spine3"),
    "rig:Neck": ("spine3", "head"),
    "rig:Head": ("spine3", "head"),
    "rig:LeftShoulder": ("left_collar", "left_shoulder"),
    "rig:LeftArm": ("left_shoulder", "left_elbow"),
    "rig:LeftForeArm": ("left_elbow", "left_wrist"),
    "rig:RightShoulder": ("right_collar", "right_shoulder"),
    "rig:RightArm": ("right_shoulder", "right_elbow"),
    "rig:RightForeArm": ("right_elbow", "right_wrist"),
    "rig:LeftUpLeg": ("left_hip", "left_knee"),
    "rig:LeftLeg": ("left_knee", "left_ankle"),
    "rig:RightUpLeg": ("right_hip", "right_knee"),
    "rig:RightLeg": ("right_knee", "right_ankle"),
}

SWING_TARGETS_NO_CLAVICLE = {
    "rig:Spine": ("spine1", "spine2"),
    "rig:Spine1": ("spine2", "spine3"),
    "rig:Spine2": ("spine3", "neck"),
    "rig:LeftArm": ("left_shoulder", "left_elbow"),
    "rig:RightArm": ("right_shoulder", "right_elbow"),
}

# Child that defines this bone's rest axis (local +Y after PreRotation).
AIM_CHILD = {
    "rig:Spine": "rig:Spine1",
    "rig:Spine1": "rig:Spine2",
    "rig:Spine2": "rig:Neck",
    "rig:Neck": "rig:Head",
    "rig:Head": "rig:HeadTop_End",
    "rig:LeftShoulder": "rig:LeftArm",
    "rig:LeftArm": "rig:LeftForeArm",
    "rig:LeftForeArm": "rig:LeftHand",
    "rig:RightShoulder": "rig:RightArm",
    "rig:RightArm": "rig:RightForeArm",
    "rig:RightForeArm": "rig:RightHand",
    "rig:LeftUpLeg": "rig:LeftLeg",
    "rig:LeftLeg": "rig:LeftFoot",
    "rig:LeftFoot": "rig:LeftToeBase",
    "rig:LeftToeBase": "rig:LeftToe_End",
    "rig:RightUpLeg": "rig:RightLeg",
    "rig:RightLeg": "rig:RightFoot",
    "rig:RightFoot": "rig:RightToeBase",
    "rig:RightToeBase": "rig:RightToe_End",
}


def swing_aim_local(rig: HumanoidRig, bone_name: str) -> np.ndarray:
    """Rest axis that swing_anim_to_direction aligns with the Human target."""
    if bone_name in NECK_HEAD:
        return NECK_HEAD_AIM_LOCAL.copy()
    return rig.child_offset_cm(bone_name, AIM_CHILD[bone_name])


RIG_TO_HUMAN = {
    "rig:Hips": "pelvis",
    "rig:Spine": "spine1",
    "rig:Spine1": "spine2",
    "rig:Spine2": "spine3",
    "rig:Neck": "neck",
    "rig:Head": "head",
    "rig:LeftShoulder": "left_collar",
    "rig:LeftArm": "left_shoulder",
    "rig:LeftForeArm": "left_elbow",
    "rig:LeftHand": "left_wrist",
    "rig:RightShoulder": "right_collar",
    "rig:RightArm": "right_shoulder",
    "rig:RightForeArm": "right_elbow",
    "rig:RightHand": "right_wrist",
    "rig:LeftUpLeg": "left_hip",
    "rig:LeftLeg": "left_knee",
    "rig:LeftFoot": "left_ankle",
    "rig:LeftToeBase": "left_foot",
    "rig:RightUpLeg": "right_hip",
    "rig:RightLeg": "right_knee",
    "rig:RightFoot": "right_ankle",
    "rig:RightToeBase": "right_foot",
}

ARM_POSITION_MARKERS = {
    "left_elbow": "rig:LeftForeArm",
    "left_wrist": "rig:LeftHand",
    "right_elbow": "rig:RightForeArm",
    "right_wrist": "rig:RightHand",
}

LEG_POSITION_MARKERS = {
    "left_hip": "rig:LeftUpLeg",
    "left_knee": "rig:LeftLeg",
    "left_ankle": "rig:LeftFoot",
    "right_hip": "rig:RightUpLeg",
    "right_knee": "rig:RightLeg",
    "right_ankle": "rig:RightFoot",
}

LEG_BONES = (
    "rig:LeftUpLeg",
    "rig:LeftLeg",
    "rig:LeftFoot",
    "rig:LeftToeBase",
    "rig:LeftToe_End",
    "rig:RightUpLeg",
    "rig:RightLeg",
    "rig:RightFoot",
    "rig:RightToeBase",
    "rig:RightToe_End",
)

WORLD_UP = np.array([0.0, 1.0, 0.0])
HAND_LOCAL_AXIS = np.array([0.0, 1.0, 0.0])

RIG_EDGES = [
    ("rig:Hips", "rig:Spine"),
    ("rig:Spine", "rig:Spine1"),
    ("rig:Spine1", "rig:Spine2"),
    ("rig:Spine2", "rig:Neck"),
    ("rig:Neck", "rig:Head"),
    ("rig:Head", "rig:HeadTop_End"),
    ("rig:Spine2", "rig:LeftShoulder"),
    ("rig:LeftShoulder", "rig:LeftArm"),
    ("rig:LeftArm", "rig:LeftForeArm"),
    ("rig:LeftForeArm", "rig:LeftHand"),
    ("rig:Spine2", "rig:RightShoulder"),
    ("rig:RightShoulder", "rig:RightArm"),
    ("rig:RightArm", "rig:RightForeArm"),
    ("rig:RightForeArm", "rig:RightHand"),
    ("rig:Hips", "rig:LeftUpLeg"),
    ("rig:LeftUpLeg", "rig:LeftLeg"),
    ("rig:LeftLeg", "rig:LeftFoot"),
    ("rig:LeftFoot", "rig:LeftToeBase"),
    ("rig:LeftToeBase", "rig:LeftToe_End"),
    ("rig:Hips", "rig:RightUpLeg"),
    ("rig:RightUpLeg", "rig:RightLeg"),
    ("rig:RightLeg", "rig:RightFoot"),
    ("rig:RightFoot", "rig:RightToeBase"),
    ("rig:RightToeBase", "rig:RightToe_End"),
]

HUMAN_EDGES = [
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("spine3", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("spine3", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
]

LENGTH_EDGES = [
    ("rig:Spine2", "rig:LeftShoulder"),
    ("rig:LeftShoulder", "rig:LeftArm"),
    ("rig:LeftArm", "rig:LeftForeArm"),
    ("rig:LeftForeArm", "rig:LeftHand"),
    ("rig:Spine2", "rig:RightShoulder"),
    ("rig:RightShoulder", "rig:RightArm"),
    ("rig:RightArm", "rig:RightForeArm"),
    ("rig:RightForeArm", "rig:RightHand"),
    ("rig:Hips", "rig:Spine"),
    ("rig:Spine", "rig:Spine1"),
    ("rig:Spine1", "rig:Spine2"),
    ("rig:Spine2", "rig:Neck"),
    ("rig:Neck", "rig:Head"),
    ("rig:Head", "rig:HeadTop_End"),
    ("rig:Hips", "rig:LeftUpLeg"),
    ("rig:LeftUpLeg", "rig:LeftLeg"),
    ("rig:LeftLeg", "rig:LeftFoot"),
    ("rig:LeftFoot", "rig:LeftToeBase"),
    ("rig:Hips", "rig:RightUpLeg"),
    ("rig:RightUpLeg", "rig:RightLeg"),
    ("rig:RightLeg", "rig:RightFoot"),
    ("rig:RightFoot", "rig:RightToeBase"),
]


@dataclass
class FrameRetarget:
    r_anim: dict[str, np.ndarray]
    q_anim: dict[str, np.ndarray]
    r_anim_raw: dict[str, np.ndarray] = field(default_factory=dict)
    twist_flips: list[str] = field(default_factory=list)
    sign_flips: list[str] = field(default_factory=list)


def joint_rig(motion, frame: int, name: str) -> np.ndarray:
    return human_to_rig(motion.joint(frame, name))


def human_dir(motion, frame: int, start: str, end: str) -> np.ndarray:
    return joint_rig(motion, frame, end) - joint_rig(motion, frame, start)


def verify_rest_fk(rig: HumanoidRig, atol: float = 1e-4) -> float:
    """Identity R_anim must reproduce the FBX rest pose (PreRotation included)."""
    world = forward_kinematics(rig, {})
    err = 0.0
    for name, bone in rig.bones.items():
        err = max(err, float(np.linalg.norm(world[name] - bone.rest_world)))
    if err > atol:
        raise RuntimeError(f"rest FK mismatch {err:.4g} > {atol} — PreRotation composition is wrong")
    return err


def rig_proportioned_from(
    rig: HumanoidRig,
    motion,
    frame: int,
    rest_pos: dict[str, np.ndarray],
    root_name: str,
    root_pos_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rebuild descendants of root_name with Human directions and Humanoid rest lengths."""
    target: dict[str, np.ndarray] = {root_name: np.asarray(root_pos_m, dtype=np.float64).copy()}
    for name in rig.order:
        if name == root_name:
            continue
        parent = rig.bones[name].parent
        if parent is None or parent not in target:
            continue
        length = float(np.linalg.norm(rest_pos[name] - rest_pos[parent]))
        if name in RIG_TO_HUMAN and parent in RIG_TO_HUMAN:
            direction = human_dir(motion, frame, RIG_TO_HUMAN[parent], RIG_TO_HUMAN[name])
        else:
            direction = rest_pos[name] - rest_pos[parent]
        target[name] = target[parent] + normalize(direction) * length
    return target


def position_errors(
    posed: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    markers: dict[str, str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label, bone in markers.items():
        delta = posed[bone] - target[bone]
        out[label] = {
            "error_m": float(np.linalg.norm(delta)),
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2]),
            "rig_bone": bone,
        }
    return out


def _hips_rotation(rig: HumanoidRig, motion, frame: int) -> np.ndarray:
    rest_world = forward_kinematics(rig, {})
    rest_pos = world_positions_m(rig, rest_world)
    r_rest_frame = orthonormal_frame(
        rest_pos["rig:LeftUpLeg"] - rest_pos["rig:RightUpLeg"],
        rest_pos["rig:Neck"] - rest_pos["rig:Hips"],
    )
    r_tgt_frame = orthonormal_frame(
        joint_rig(motion, frame, "left_hip") - joint_rig(motion, frame, "right_hip"),
        joint_rig(motion, frame, "neck") - joint_rig(motion, frame, "pelvis"),
    )
    r_hips_world = (r_tgt_frame @ r_rest_frame.T) @ rest_world["rig:Hips"][:3, :3]
    return anim_from_world_rotation(rig.bones["rig:Hips"], np.eye(3), r_hips_world)


def _spine2_chest_rotation(rig: HumanoidRig, world: dict[str, np.ndarray], motion, frame: int) -> np.ndarray:
    bone = rig.bones["rig:Spine2"]
    parent_r = world[bone.parent][:3, :3] if bone.parent else np.eye(3)
    rest_left = rig.child_offset_cm("rig:Spine2", "rig:LeftShoulder") - rig.child_offset_cm(
        "rig:Spine2", "rig:RightShoulder"
    )
    rest_up = rig.child_offset_cm("rig:Spine2", "rig:Neck")
    tgt_left = joint_rig(motion, frame, "left_collar") - joint_rig(motion, frame, "right_collar")
    tgt_up = joint_rig(motion, frame, "neck") - joint_rig(motion, frame, "spine3")

    # Guard against monocular keypoint inversion when neck dips below spine3:
    # preserve upright spinal column alignment instead of flipping the ribcage upside-down.
    if np.dot(tgt_up, WORLD_UP) < 0.05:
        spine_col = joint_rig(motion, frame, "neck") - joint_rig(motion, frame, "pelvis")
        if np.dot(spine_col, WORLD_UP) > 0.05:
            tgt_up = spine_col
        else:
            tgt_up = WORLD_UP.copy()

    return frame_anim_from_axes(
        bone,
        parent_r,
        rest_left,
        rest_up,
        tgt_left,
        tgt_up,
    )


def _foot_rotation(rig: HumanoidRig, world: dict[str, np.ndarray], motion, frame: int, side: str) -> np.ndarray:
    foot = f"rig:{side}Foot"
    toe = f"rig:{side}ToeBase"
    bone = rig.bones[foot]
    parent_r = world[bone.parent][:3, :3] if bone.parent else np.eye(3)
    along_local = rig.child_offset_cm(foot, toe)
    r_rest = parent_r @ bone.pre_rotation
    up_local = r_rest.T @ WORLD_UP
    ankle = "left_ankle" if side == "Left" else "right_ankle"
    foot_j = "left_foot" if side == "Left" else "right_foot"
    tgt_along = normalize(human_dir(motion, frame, ankle, foot_j), fallback=np.array([0.0, 0.0, 1.0]))
    # Bound foot pitch away from the vertical WORLD_UP pole so Gram-Schmidt never flips 180 degrees
    horiz = np.array([tgt_along[0], 0.0, tgt_along[2]], dtype=np.float64)
    n_horiz = float(np.linalg.norm(horiz))
    if n_horiz < 0.35:
        # Fallback horizontal heading from pelvis anterior vector
        v_left_hip = joint_rig(motion, frame, "left_hip") - joint_rig(motion, frame, "right_hip")
        v_up_spine = joint_rig(motion, frame, "neck") - joint_rig(motion, frame, "pelvis")
        fwd = np.cross(v_left_hip, v_up_spine)
        fwd[1] = 0.0
        u_fwd = normalize(fwd, fallback=np.array([0.0, 0.0, 1.0]))
        # Continuous smooth blend with pelvis forward heading during turns
        w_raw = max(0.0, min(1.0, (n_horiz - 0.05) / 0.30))
        u_horiz = normalize(w_raw * horiz + (1.0 - w_raw) * u_fwd, fallback=u_fwd)
        tgt_along = normalize(0.50 * u_horiz + 0.50 * tgt_along)
    tgt_up = WORLD_UP.copy()
    return frame_anim_from_two_vectors(
        bone,
        parent_r,
        along_local,
        up_local,
        tgt_along,
        tgt_up,
    )


def _forearm_rotation(
    rig: HumanoidRig,
    world: dict[str, np.ndarray],
    motion,
    frame: int,
    side: str,
    prev_r_anim: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Couple forearm axial twist to palm normal using two-vector orientation alignment."""
    forearm = f"rig:{side}ForeArm"
    hand = f"rig:{side}Hand"
    bone = rig.bones[forearm]
    parent_r = world[bone.parent][:3, :3] if bone.parent else np.eye(3)
    along_local = rig.child_offset_cm(forearm, hand)

    # In rest T-pose: forearm aims along +/-X, palm normal points downwards (-Y)
    up_local = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    elbow_j = "left_elbow" if side == "Left" else "right_elbow"
    wrist_j = "left_wrist" if side == "Left" else "right_wrist"
    palm_j = "left_hand" if side == "Left" else "right_hand"

    tgt_along = human_dir(motion, frame, elbow_j, wrist_j)
    if float(np.linalg.norm(tgt_along)) < 1e-4:
        return np.eye(3)
    u_along = normalize(tgt_along)

    # Form palm normal vector n_palm from wrist, pinky, and index landmarks
    n_palm: np.ndarray | None = None
    if (
        hasattr(motion, "world_landmarks")
        and motion.world_landmarks is not None
        and len(motion.world_landmarks) > frame
    ):
        w_lms = motion.world_landmarks[frame]
        idx_w = 15 if side == "Left" else 16
        idx_p = 17 if side == "Left" else 18
        idx_i = 19 if side == "Left" else 20

        # Transform to Humanoid world frame (rotZ(180) then R_BODY_ALIGN: [x, -y, -z])
        pw = np.array([w_lms[idx_w, 0], -w_lms[idx_w, 1], -w_lms[idx_w, 2]], dtype=np.float64)
        pp = np.array([w_lms[idx_p, 0], -w_lms[idx_p, 1], -w_lms[idx_p, 2]], dtype=np.float64)
        pi = np.array([w_lms[idx_i, 0], -w_lms[idx_i, 1], -w_lms[idx_i, 2]], dtype=np.float64)

        v_palm = 0.5 * (pp + pi) - pw
        v_lat = pi - pp
        raw_norm = np.cross(v_palm, v_lat)
        if float(np.linalg.norm(raw_norm)) > 1e-4:
            n_palm = normalize(raw_norm)
            if side == "Left":
                n_palm = -n_palm

    if n_palm is None:
        # Fallback using 24 Body24 joints: elbow -> wrist -> hand
        v_hand = human_dir(motion, frame, wrist_j, palm_j)
        if float(np.linalg.norm(v_hand)) > 1e-4:
            raw_norm = np.cross(u_along, normalize(v_hand))
            if float(np.linalg.norm(raw_norm)) > 1e-4:
                n_palm = normalize(raw_norm)

    if n_palm is None:
        n_palm = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    # Maintain palm normal orientation continuity against previous frame
    if prev_r_anim is not None and forearm in prev_r_anim:
        prev_world_up = (parent_r @ prev_r_anim[forearm]) @ up_local
        if float(np.linalg.norm(prev_world_up)) > 1e-4:
            u_prev_up = normalize(prev_world_up)
            if float(np.dot(n_palm, u_prev_up)) < 0.0:
                n_palm = -n_palm

    # Base canonical zero-twist swing aiming elbow -> wrist
    r_swing = swing_anim_to_direction(rig, world, forearm, hand, u_along)

    # Use frame_anim_from_two_vectors to compute palm-coupled orientation, then
    # blend a gentle bounded axial pronation/supination twist (<= 20 deg) onto r_swing
    r_two_vec = frame_anim_from_two_vectors(
        bone,
        parent_r,
        along_local,
        up_local,
        u_along,
        n_palm,
    )
    r_delta = r_swing.T @ r_two_vec
    trace_val = float(np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0))
    twist_angle = float(np.arccos(trace_val))
    if twist_angle > 1e-4:
        # Attenuate axial palm twist to 20% and clamp at 18 degrees so 3 cm finger noise
        # never destabilises the forearm aim or elbow silhouette
        target_twist = min(0.20 * twist_angle, float(np.radians(18.0)))
        q_swing = quat_wxyz(r_swing)
        q_two, _ = continue_quaternion(quat_wxyz(r_two_vec), q_swing)
        alpha = target_twist / twist_angle
        q_blend = (1.0 - alpha) * q_swing + alpha * q_two
        n_q = float(np.linalg.norm(q_blend))
        if n_q > 1e-8:
            q_blend /= n_q
            w, x, y, z = (float(v) for v in q_blend)
            return np.array(
                [
                    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                    [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                    [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )

    return r_swing


def _hand_rotation(rig: HumanoidRig, world: dict[str, np.ndarray], motion, frame: int, side: str) -> np.ndarray:
    """Aim Humanoid Hand along its true rest axis with forearm-bounded wrist flexion."""
    hand = f"rig:{side}Hand"
    elbow_j = "left_elbow" if side == "Left" else "right_elbow"
    wrist_j = "left_wrist" if side == "Left" else "right_wrist"
    palm_j = "left_hand" if side == "Left" else "right_hand"
    tgt = human_dir(motion, frame, wrist_j, palm_j)
    if float(np.linalg.norm(tgt)) < 1e-4:
        return np.eye(3)

    forearm_vec = human_dir(motion, frame, elbow_j, wrist_j)
    if float(np.linalg.norm(forearm_vec)) > 1e-4:
        u_forearm = normalize(forearm_vec)
        u_hand = normalize(tgt)
        # Blend noisy ~5 cm palm vector with stable ~25 cm forearm direction
        u_blend = normalize(0.55 * u_forearm + 0.45 * u_hand, fallback=u_forearm)
        # Bound physiological wrist deflection to at most 25 degrees from forearm axis
        max_rad = np.radians(25.0)
        cos_ang = float(np.clip(np.dot(u_forearm, u_blend), -1.0, 1.0))
        if np.arccos(cos_ang) > max_rad:
            ortho = u_blend - cos_ang * u_forearm
            n_ortho = float(np.linalg.norm(ortho))
            if n_ortho > 1e-6:
                ortho /= n_ortho
                u_blend = np.cos(max_rad) * u_forearm + np.sin(max_rad) * ortho
        tgt = u_blend

    return swing_anim_to_direction(rig, world, hand, None, tgt, d_local=local_aim_axis(rig, hand))


# Bones whose rotation is built from two independent human directions or from
# the canonical minimum-angle (zero-twist) swing `rotation_aligning`.
FULLY_DETERMINED = frozenset(CONTROLLED)


def retarget_one_frame(
    rig: HumanoidRig,
    motion,
    frame: int,
    *,
    clavicles: bool = True,
    legs: bool = True,
    prev_r_anim: dict[str, np.ndarray] | None = None,
    prev_q_anim: dict[str, np.ndarray] | None = None,
) -> FrameRetarget:
    hips_t_cm = joint_rig(motion, frame, "pelvis") / rig.meters_per_unit
    r_anim = {name: np.eye(3) for name in rig.order}
    q_anim = {name: np.array([1.0, 0.0, 0.0, 0.0]) for name in rig.order}
    twist_flips: list[str] = []
    sign_flips: list[str] = []

    def fk() -> dict[str, np.ndarray]:
        return forward_kinematics(rig, r_anim, hips_translation_cm=hips_t_cm)

    def commit(name: str, rotation: np.ndarray) -> None:
        q_prev = None if prev_q_anim is None else prev_q_anim.get(name)
        if name in FULLY_DETERMINED:
            r_pick, did_twist = np.asarray(rotation, dtype=np.float64), False
        else:
            axis = swing_aim_local(rig, name) if name in AIM_CHILD or name in NECK_HEAD else local_aim_axis(rig, name)
            r_prev = None if prev_r_anim is None else prev_r_anim.get(name)
            r_pick, did_twist = pick_closest_aim_twist(rotation, r_prev, axis)
        q = quat_wxyz(r_pick)
        q, did_sign = continue_quaternion(q, q_prev)
        r_anim[name] = r_pick
        q_anim[name] = q
        if did_twist:
            twist_flips.append(name)
        # Record only actual discontinuities where dot product remains negative
        if q_prev is not None and float(np.dot(q, q_prev)) < -1e-6:
            sign_flips.append(name)

    def apply_swings(targets: dict[str, tuple[str, str]]) -> None:
        for bone_name, (h_a, h_b) in targets.items():
            rotation = swing_anim_to_direction(
                rig,
                fk(),
                bone_name,
                AIM_CHILD.get(bone_name),
                human_dir(motion, frame, h_a, h_b),
                d_local=swing_aim_local(rig, bone_name),
            )
            commit(bone_name, rotation)

    commit("rig:Hips", _hips_rotation(rig, motion, frame))
    if clavicles:
        apply_swings(
            {
                "rig:Spine": SWING_TARGETS["rig:Spine"],
                "rig:Spine1": SWING_TARGETS["rig:Spine1"],
            }
        )
        commit("rig:Spine2", _spine2_chest_rotation(rig, fk(), motion, frame))
        apply_swings(
            {
                "rig:Neck": SWING_TARGETS["rig:Neck"],
                "rig:Head": SWING_TARGETS["rig:Head"],
                "rig:LeftShoulder": SWING_TARGETS["rig:LeftShoulder"],
                "rig:LeftArm": SWING_TARGETS["rig:LeftArm"],
                "rig:RightShoulder": SWING_TARGETS["rig:RightShoulder"],
                "rig:RightArm": SWING_TARGETS["rig:RightArm"],
            }
        )
        commit("rig:LeftForeArm", _forearm_rotation(rig, fk(), motion, frame, "Left", prev_r_anim))
        commit("rig:RightForeArm", _forearm_rotation(rig, fk(), motion, frame, "Right", prev_r_anim))
        commit("rig:LeftHand", _hand_rotation(rig, fk(), motion, frame, "Left"))
        commit("rig:RightHand", _hand_rotation(rig, fk(), motion, frame, "Right"))
    else:
        apply_swings(SWING_TARGETS_NO_CLAVICLE)
        apply_swings(
            {
                "rig:Neck": SWING_TARGETS["rig:Neck"],
                "rig:Head": SWING_TARGETS["rig:Head"],
            }
        )
        commit("rig:LeftForeArm", _forearm_rotation(rig, fk(), motion, frame, "Left", prev_r_anim))
        commit("rig:RightForeArm", _forearm_rotation(rig, fk(), motion, frame, "Right", prev_r_anim))
        commit("rig:LeftHand", _hand_rotation(rig, fk(), motion, frame, "Left"))
        commit("rig:RightHand", _hand_rotation(rig, fk(), motion, frame, "Right"))
    if legs:
        apply_swings(
            {
                "rig:LeftUpLeg": SWING_TARGETS["rig:LeftUpLeg"],
                "rig:LeftLeg": SWING_TARGETS["rig:LeftLeg"],
                "rig:RightUpLeg": SWING_TARGETS["rig:RightUpLeg"],
                "rig:RightLeg": SWING_TARGETS["rig:RightLeg"],
            }
        )
        commit("rig:LeftFoot", _foot_rotation(rig, fk(), motion, frame, "Left"))
        commit("rig:LeftToeBase", np.eye(3))
        commit("rig:RightFoot", _foot_rotation(rig, fk(), motion, frame, "Right"))
        commit("rig:RightToeBase", np.eye(3))
    r_raw = {name: np.asarray(rot, dtype=np.float64).copy() for name, rot in r_anim.items()}
    return FrameRetarget(
        r_anim=r_anim,
        q_anim=q_anim,
        r_anim_raw=r_raw,
        twist_flips=twist_flips,
        sign_flips=sign_flips,
    )
