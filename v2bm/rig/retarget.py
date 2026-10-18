"""Human → Humanoid per-frame retarget.

FK is T * Rpre * R_anim. Humanoid rest bone lengths are never scaled.
Optional previous-frame rotations pick the 180-degree twist closest to the
last frame and enforce quaternion sign continuity. No temporal smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fbx_skeleton import matrix_to_quat_wxyz
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

# Canonical 52 ARKit blendshapes specified by Apple ARKit & Google MediaPipe FaceLandmarker
CANONICAL_ARKIT_BLENDSHAPES: list[str] = [
    # Brows (5)
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    # Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    # Eyes (14)
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    # Jaw (4)
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    # Mouth (23)
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    # Nose (2)
    "noseSneerLeft",
    "noseSneerRight",
    # Tongue (1)
    "tongueOut",
]

# Morph Sensitivity Profile (amplify subtle facial movements)
DEFAULT_BLENDSHAPE_SENSITIVITIES: dict[str, float] = {
    # Brows (articulate brow movement clearly)
    "browDownLeft": 1.20,
    "browDownRight": 1.20,
    "browInnerUp": 1.25,
    "browOuterUpLeft": 1.20,
    "browOuterUpRight": 1.20,
    # Cheeks
    "cheekPuff": 1.25,
    "cheekSquintLeft": 1.15,
    "cheekSquintRight": 1.15,
    # Eyes (blinks preserved without dampening)
    "eyeBlinkLeft": 1.00,
    "eyeBlinkRight": 1.00,
    "eyeWideLeft": 1.10,
    "eyeWideRight": 1.10,
    # Jaw
    "jawOpen": 1.10,
    "jawForward": 1.10,
    # Mouth
    "mouthSmileLeft": 1.15,
    "mouthSmileRight": 1.15,
    "mouthDimpleLeft": 1.15,
    "mouthDimpleRight": 1.15,
    "mouthFrownLeft": 1.20,
    "mouthFrownRight": 1.20,
    "mouthPucker": 1.15,
    "mouthFunnel": 1.15,
}

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


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    q0_arr = np.asarray(q0, dtype=np.float64).reshape(-1)
    q1_arr = np.asarray(q1, dtype=np.float64).reshape(-1)
    dot = float(np.dot(q0_arr, q1_arr))
    if dot < 0.0:
        q1_arr = -q1_arr
        dot = -dot
    if dot > 0.9995:
        res = q0_arr + t * (q1_arr - q0_arr)
        norm = float(np.linalg.norm(res))
        return res / norm if norm > 1e-12 else q0_arr
    theta_0 = np.arccos(min(1.0, max(-1.0, dot)))
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s0 = np.cos(theta_t) - dot * sin_theta_t / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * q0_arr + s1 * q1_arr
    norm = float(np.linalg.norm(res))
    return res / norm if norm > 1e-12 else q0_arr


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 orthonormal rotation matrix."""
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = arr / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _hips_rotation(rig: HumanoidRig, motion, frame: int) -> np.ndarray:
    """Solve pelvic orientation with decoupled sagittal pitch clamped to [-15 deg, +20 deg].

    Prevents forward thoracic flexion (bows) from pitching the pelvis excessively forward
    and kicking the legs backward.
    """
    bone = rig.bones["rig:Hips"]
    rest_world = forward_kinematics(rig, {})
    rest_pos = world_positions_m(rig, rest_world)
    r_rest_frame = orthonormal_frame(
        rest_pos["rig:LeftUpLeg"] - rest_pos["rig:RightUpLeg"],
        rest_pos["rig:Spine"] - rest_pos["rig:Hips"],
    )

    v_lat = joint_rig(motion, frame, "left_hip") - joint_rig(motion, frame, "right_hip")
    u_lat = normalize(v_lat, fallback=np.array([1.0, 0.0, 0.0]))

    v_spine_lower = joint_rig(motion, frame, "spine1") - joint_rig(motion, frame, "pelvis")
    if float(np.linalg.norm(v_spine_lower)) < 1e-4:
        v_spine_lower = WORLD_UP.copy()

    u_fwd_horiz = normalize(np.cross(u_lat, WORLD_UP), fallback=np.array([0.0, 0.0, 1.0]))

    pitch_sin = float(np.dot(v_spine_lower, u_fwd_horiz))
    pitch_cos = float(np.dot(v_spine_lower, WORLD_UP))
    pitch_rad = float(np.arctan2(pitch_sin, pitch_cos))

    min_pitch = float(np.radians(-15.0))
    max_pitch = float(np.radians(20.0))
    pitch_clamped = float(np.clip(pitch_rad, min_pitch, max_pitch))

    u_pelv_up = normalize(
        np.cos(pitch_clamped) * WORLD_UP + np.sin(pitch_clamped) * u_fwd_horiz,
        fallback=WORLD_UP,
    )

    r_tgt_frame = orthonormal_frame(u_lat, u_pelv_up)
    r_hips_world = (r_tgt_frame @ r_rest_frame.T) @ rest_world["rig:Hips"][:3, :3]
    return anim_from_world_rotation(bone, np.eye(3), r_hips_world)


def _chest_target_frame(motion, frame: int) -> np.ndarray:
    """Target thoracic chest frame computed from collar lateral axis and upper spinal column."""
    tgt_left = joint_rig(motion, frame, "left_collar") - joint_rig(motion, frame, "right_collar")
    tgt_up = joint_rig(motion, frame, "neck") - joint_rig(motion, frame, "spine3")

    # Guard against monocular keypoint inversion when neck dips below spine3:
    # preserve upright spinal column alignment instead of flipping the ribcage upside-down.
    u_up = normalize(tgt_up, fallback=WORLD_UP)
    if float(np.dot(u_up, WORLD_UP)) < -0.10:
        spine_col = joint_rig(motion, frame, "neck") - joint_rig(motion, frame, "pelvis")
        u_spine = normalize(spine_col, fallback=WORLD_UP)
        if float(np.dot(u_spine, WORLD_UP)) > 0.05:
            tgt_up = spine_col
        else:
            tgt_up = WORLD_UP.copy()

    return orthonormal_frame(tgt_left, tgt_up)


def _shoulder_rotation(
    rig: HumanoidRig,
    world: dict[str, np.ndarray],
    motion,
    frame: int,
    side: str,
) -> np.ndarray:
    """Clavicle rotation coupling base collar aim with scapular elevation and protraction."""
    shoulder_bone_name = f"rig:{side}Shoulder"

    collar_j = "left_collar" if side == "Left" else "right_collar"
    shoulder_j = "left_shoulder" if side == "Left" else "right_shoulder"
    elbow_j = "left_elbow" if side == "Left" else "right_elbow"

    tgt_clavicle = human_dir(motion, frame, collar_j, shoulder_j)
    d_local = swing_aim_local(rig, shoulder_bone_name)
    r_base = swing_anim_to_direction(
        rig,
        world,
        shoulder_bone_name,
        AIM_CHILD.get(shoulder_bone_name),
        tgt_clavicle,
        d_local=d_local,
    )

    tgt_arm = human_dir(motion, frame, shoulder_j, elbow_j)
    if float(np.linalg.norm(tgt_arm)) < 1e-4:
        return r_base

    u_arm = normalize(tgt_arm)
    u_down = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    cos_elev = float(np.clip(np.dot(u_arm, u_down), -1.0, 1.0))
    theta_elev = float(np.arccos(cos_elev))

    thresh = float(np.radians(60.0))
    r_elev = np.eye(3)
    if theta_elev > thresh:
        delta = theta_elev - thresh
        phi_elev = 0.35 * delta
        sign_z = 1.0 if side == "Left" else -1.0
        c, s = np.cos(sign_z * phi_elev), np.sin(sign_z * phi_elev)
        r_elev = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    fwd_comp = float(u_arm[2])
    cross_comp = -float(u_arm[0]) if side == "Left" else float(u_arm[0])
    r_prot = np.eye(3)
    if fwd_comp > 0.1 and cross_comp > 0.0:
        reach_factor = min(1.0, (fwd_comp + cross_comp) * 0.5)
        phi_prot = min(float(np.radians(15.0)), float(np.radians(15.0)) * reach_factor)
        sign_y = -1.0 if side == "Left" else 1.0
        cy, sy = np.cos(sign_y * phi_prot), np.sin(sign_y * phi_prot)
        r_prot = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)

    return r_base @ (r_prot @ r_elev)


def _cranial_rotations(
    rig: HumanoidRig,
    world: dict[str, np.ndarray],
    motion,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve 3-DoF relative cranial rotation and distribute 40% to Neck and 60% to Head."""
    r_cranium_world: np.ndarray | None = None

    # 1. Direct quaternion if present on motion
    if (
        hasattr(motion, "head_rotation_quat")
        and motion.head_rotation_quat is not None
        and len(motion.head_rotation_quat) > frame
        and motion.head_rotation_quat[frame] is not None
    ):
        q_head = np.asarray(motion.head_rotation_quat[frame], dtype=np.float64).reshape(-1)
        if len(q_head) == 4 and float(np.linalg.norm(q_head)) > 1e-4:
            r_cranium_world = _quat_wxyz_to_matrix(q_head)

    # 2. Cranial landmarks fallback from MediaPipe world landmarks
    if r_cranium_world is None and (
        hasattr(motion, "world_landmarks")
        and motion.world_landmarks is not None
        and len(motion.world_landmarks) > frame
    ):
        w_lms = motion.world_landmarks[frame]
        # Landmarks: 0: nose, 7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right
        p_nose = np.array([w_lms[0, 0], -w_lms[0, 1], -w_lms[0, 2]], dtype=np.float64)
        p_lear = np.array([w_lms[7, 0], -w_lms[7, 1], -w_lms[7, 2]], dtype=np.float64)
        p_rear = np.array([w_lms[8, 0], -w_lms[8, 1], -w_lms[8, 2]], dtype=np.float64)
        p_lmouth = np.array([w_lms[9, 0], -w_lms[9, 1], -w_lms[9, 2]], dtype=np.float64)
        p_rmouth = np.array([w_lms[10, 0], -w_lms[10, 1], -w_lms[10, 2]], dtype=np.float64)

        v_lat = p_lear - p_rear
        if float(np.linalg.norm(v_lat)) > 1e-4:
            u_lat = normalize(v_lat)
            m_ears = 0.5 * (p_lear + p_rear)
            m_mouth = 0.5 * (p_lmouth + p_rmouth)
            v_fwd = 0.70 * (p_nose - m_ears) + 0.30 * (m_mouth - m_ears)
            v_fwd_rej = v_fwd - u_lat * float(np.dot(v_fwd, u_lat))
            if float(np.linalg.norm(v_fwd_rej)) > 1e-4:
                u_fwd = normalize(v_fwd_rej)
                u_up = normalize(np.cross(u_fwd, u_lat))
                r_cranium_world = np.column_stack([u_lat, u_up, u_fwd])

    # 3. Fallback to single-vector swing spine3 -> head
    if r_cranium_world is None:
        u_up = normalize(human_dir(motion, frame, "spine3", "head"), fallback=WORLD_UP)
        u_lat_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        u_fwd = normalize(np.cross(u_lat_hint, u_up), fallback=np.array([0.0, 0.0, 1.0]))
        u_lat = normalize(np.cross(u_up, u_fwd))
        u_up = normalize(np.cross(u_fwd, u_lat))
        r_cranium_world = np.column_stack([u_lat, u_up, u_fwd])

    # Ocular-cervical coupling when gaze vectors are available
    gv = getattr(motion, "gaze_vectors", None)
    if gv is not None and len(gv) > frame and gv[frame] is not None:
        gaze_vec = np.asarray(gv[frame], dtype=np.float64)
        if gaze_vec.shape == (2, 3) and float(np.linalg.norm(gaze_vec)) > 1e-4:
            gx = 0.5 * (float(gaze_vec[0, 0]) + float(gaze_vec[1, 0]))
            gy = 0.5 * (float(gaze_vec[0, 1]) + float(gaze_vec[1, 1]))
            gz = max(0.5 * (float(gaze_vec[0, 2]) + float(gaze_vec[1, 2])), 0.1)
            yaw_micro = float(np.arctan2(gx, gz) * 0.12)
            pitch_micro = float(np.arctan2(-gy, gz) * 0.10)
            cy, sy = np.cos(yaw_micro), np.sin(yaw_micro)
            cp, sp = np.cos(pitch_micro), np.sin(pitch_micro)
            r_micro = np.array(
                [
                    [cy, 0.0, sy],
                    [-sp * sy, cp, sp * cy],
                    [-cp * sy, -sp, cp * cy],
                ],
                dtype=np.float64,
            )
            r_cranium_world = r_cranium_world @ r_micro

    # Relative cranial rotation from Spine2 thoracic cage
    parent_r = world["rig:Spine2"][:3, :3]
    r_rel = parent_r.T @ r_cranium_world
    if float(np.linalg.det(r_rel)) < 0.0:
        u, _, vt = np.linalg.svd(r_rel)
        r_rel = u @ vt

    q_rel = quat_wxyz(r_rel)
    if q_rel[0] < 0.0:
        q_rel = -q_rel

    q_I = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_neck = _slerp(q_I, q_rel, 0.40)
    q_head = _slerp(q_I, q_rel, 0.60)

    # Solve Neck anim rotation (parent is Spine2)
    r_target_neck_world = parent_r @ rig.bones["rig:Neck"].pre_rotation @ _quat_wxyz_to_matrix(q_neck)
    r_neck_anim = anim_from_world_rotation(rig.bones["rig:Neck"], parent_r, r_target_neck_world)

    # Solve Head anim rotation (parent is Neck)
    r_head_anim = anim_from_world_rotation(rig.bones["rig:Head"], r_target_neck_world, r_cranium_world)

    return r_neck_anim, r_head_anim


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
    u_up = normalize(tgt_up, fallback=WORLD_UP)
    if float(np.dot(u_up, WORLD_UP)) < -0.10:
        spine_col = joint_rig(motion, frame, "neck") - joint_rig(motion, frame, "pelvis")
        u_spine = normalize(spine_col, fallback=WORLD_UP)
        if float(np.dot(u_spine, WORLD_UP)) > 0.05:
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


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz unit quaternion to 3x3 rotation matrix."""
    return _quat_wxyz_to_matrix(q)


def build_morph_target_mapping(
    target_names: list[str],
    source_names: list[str] = CANONICAL_ARKIT_BLENDSHAPES,
) -> list[int]:
    """Map target mesh shape key names to canonical source blendshape indices.

    Supports exact matches, case-insensitive matches, and prefix stripping
    (e.g. 'rig:', 'blendshape.', 'facs_', 'basis.', 'mesh_').
    Returns list of source indices of length len(target_names). -1 indicates unmapped.
    """
    cleaned_source = {s.lower(): i for i, s in enumerate(source_names)}
    mapping: list[int] = []

    prefixes = ("rig:", "blendshape.", "facs_", "basis.", "mesh_")

    for tname in target_names:
        # 1. Exact match
        if tname in source_names:
            mapping.append(source_names.index(tname))
            continue

        # 2. Case-insensitive match
        t_lower = tname.lower()
        if t_lower in cleaned_source:
            mapping.append(cleaned_source[t_lower])
            continue

        # 3. Strip prefix and match
        matched_idx = -1
        for pfx in prefixes:
            if t_lower.startswith(pfx):
                stripped = t_lower[len(pfx):]
                if stripped in cleaned_source:
                    matched_idx = cleaned_source[stripped]
                    break
        mapping.append(matched_idx)

    return mapping


def retarget_facial_blendshapes(
    extracted_blendshapes: np.ndarray,
    target_morph_names: list[str] | None,
    source_names: list[str] = CANONICAL_ARKIT_BLENDSHAPES,
    *,
    sensitivities: dict[str, float] | None = None,
) -> np.ndarray | None:
    """Retarget extracted (T, 52) or (52,) ARKit blendshapes to target mesh shape keys.

    Applies name mapping, sensitivity scaling, and blink snapping.
    Returns: ndarray of shape (T, M) or (M,) bounded in [0.0, 1.0].
    """
    if target_morph_names is None or len(target_morph_names) == 0:
        return None

    is_1d = extracted_blendshapes.ndim == 1
    raw = np.atleast_2d(extracted_blendshapes).astype(np.float32)
    T, S = raw.shape
    M = len(target_morph_names)

    mapping = build_morph_target_mapping(target_morph_names, source_names)
    weights = np.zeros((T, M), dtype=np.float32)

    active_sens = DEFAULT_BLENDSHAPE_SENSITIVITIES if sensitivities is None else sensitivities

    idx_blink_l = source_names.index("eyeBlinkLeft") if "eyeBlinkLeft" in source_names else -1
    idx_blink_r = source_names.index("eyeBlinkRight") if "eyeBlinkRight" in source_names else -1

    for m, src_idx in enumerate(mapping):
        if src_idx < 0 or src_idx >= S:
            continue
        col = raw[:, src_idx].copy()
        sname = source_names[src_idx]
        scale = active_sens.get(sname, 1.0)
        col *= scale

        # Non-linear blink preservation snap for w >= 0.65
        if src_idx in (idx_blink_l, idx_blink_r):
            mask = col >= 0.65
            if np.any(mask):
                norm_part = (col[mask] - 0.65) / 0.35
                col[mask] = 0.65 + 0.35 * np.power(norm_part, 0.75)

        weights[:, m] = np.clip(col, 0.0, 1.0)

    return weights[0] if is_1d else weights


def retarget_head_neck_orientation(
    rig: HumanoidRig,
    world: dict[str, np.ndarray],
    head_quat_wxyz: np.ndarray,
    gaze_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Couple extracted head orientation and gaze into rig:Neck (35%) and rig:Head (65%).

    head_quat_wxyz: (4,) unit quaternion wxyz in MediaPipe camera space.
    gaze_vector: (2, 3) left/right eye gaze displacement vectors.
    Returns: (r_anim_neck, r_anim_head) 3x3 local rotation matrices.
    """
    r_cam = quat_to_matrix(head_quat_wxyz)
    c_mat = np.diag([1.0, -1.0, -1.0])
    r_head_world = c_mat @ r_cam @ c_mat.T

    if gaze_vector is not None:
        gx = 0.5 * (float(gaze_vector[0, 0]) + float(gaze_vector[1, 0]))
        gy = 0.5 * (float(gaze_vector[0, 1]) + float(gaze_vector[1, 1]))
        gz = max(0.5 * (float(gaze_vector[0, 2]) + float(gaze_vector[1, 2])), 0.1)
        yaw_micro = float(np.arctan2(gx, gz) * 0.12)
        pitch_micro = float(np.arctan2(-gy, gz) * 0.10)
        cy, sy = np.cos(yaw_micro), np.sin(yaw_micro)
        cp, sp = np.cos(pitch_micro), np.sin(pitch_micro)
        r_micro = np.array([
            [cy, 0.0, sy],
            [-sp * sy, cp, sp * cy],
            [-cp * sy, -sp, cp * cy],
        ], dtype=np.float64)
        r_head_world = r_head_world @ r_micro

    bone_neck = rig.bones["rig:Neck"]
    bone_head = rig.bones["rig:Head"]
    r_spine2 = world.get(bone_neck.parent, np.eye(4))[:3, :3] if bone_neck.parent else np.eye(3)

    r_cervical = r_spine2.T @ r_head_world
    q_cervical = matrix_to_quat_wxyz(r_cervical)

    q_ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_neck = _slerp(q_ident, q_cervical, 0.35)
    q_head = _slerp(q_ident, q_cervical, 0.65)

    r_neck_world = r_spine2 @ quat_to_matrix(q_neck)
    r_head_final = r_neck_world @ quat_to_matrix(q_head)

    r_anim_neck = anim_from_world_rotation(bone_neck, r_spine2, r_neck_world)
    r_anim_head = anim_from_world_rotation(bone_head, r_neck_world, r_head_final)

    return r_anim_neck, r_anim_head


def retarget_one_frame(
    rig: HumanoidRig,
    motion,
    frame: int,
    *,
    clavicles: bool = True,
    legs: bool = True,
    prev_r_anim: dict[str, np.ndarray] | None = None,
    prev_q_anim: dict[str, np.ndarray] | None = None,
    hand_data: dict | None = None,
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

    # Progressive Spine Curvature & Axial Torsion (25% Spine, 35% Spine1, 40% Spine2)
    r_chest_target = _chest_target_frame(motion, frame)
    r_pelvis_world = fk()["rig:Hips"][:3, :3]
    r_torso_rel = r_pelvis_world.T @ r_chest_target
    if float(np.linalg.det(r_torso_rel)) < 0.0:
        u, _, vt = np.linalg.svd(r_torso_rel)
        r_torso_rel = u @ vt

    q_torso = quat_wxyz(r_torso_rel)
    if q_torso[0] < 0.0:
        q_torso = -q_torso

    q_I = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_spine = _slerp(q_I, q_torso, 0.25)

    r_target_spine_world = r_pelvis_world @ _quat_wxyz_to_matrix(q_spine)
    commit(
        "rig:Spine",
        anim_from_world_rotation(rig.bones["rig:Spine"], r_pelvis_world, r_target_spine_world),
    )

    r_target_spine1_world = r_pelvis_world @ _quat_wxyz_to_matrix(_slerp(q_I, q_torso, 0.60))
    commit(
        "rig:Spine1",
        anim_from_world_rotation(rig.bones["rig:Spine1"], fk()["rig:Spine"][:3, :3], r_target_spine1_world),
    )

    commit(
        "rig:Spine2",
        anim_from_world_rotation(rig.bones["rig:Spine2"], fk()["rig:Spine1"][:3, :3], r_chest_target),
    )

    # 3-DoF Cranial Solver (Neck 40% / Head 60% + ocular-cervical gaze micro-rotation)
    r_neck, r_head = _cranial_rotations(rig, fk(), motion, frame)
    commit("rig:Neck", r_neck)
    commit("rig:Head", r_head)

    if clavicles:
        # Scapular Rhythm Clavicles & Arms
        commit("rig:LeftShoulder", _shoulder_rotation(rig, fk(), motion, frame, "Left"))
        apply_swings({"rig:LeftArm": SWING_TARGETS["rig:LeftArm"]})
        commit("rig:RightShoulder", _shoulder_rotation(rig, fk(), motion, frame, "Right"))
        apply_swings({"rig:RightArm": SWING_TARGETS["rig:RightArm"]})

        commit("rig:LeftForeArm", _forearm_rotation(rig, fk(), motion, frame, "Left", prev_r_anim))
        commit("rig:RightForeArm", _forearm_rotation(rig, fk(), motion, frame, "Right", prev_r_anim))
        commit("rig:LeftHand", _hand_rotation(rig, fk(), motion, frame, "Left"))
        commit("rig:RightHand", _hand_rotation(rig, fk(), motion, frame, "Right"))
    else:
        apply_swings(
            {
                "rig:LeftArm": SWING_TARGETS_NO_CLAVICLE["rig:LeftArm"],
                "rig:RightArm": SWING_TARGETS_NO_CLAVICLE["rig:RightArm"],
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
