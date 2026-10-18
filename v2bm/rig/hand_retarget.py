"""High-fidelity palm normal, forearm twist, 3-DoF wrist, and 30-bone finger retargeting.

Permissive (Apache-2.0) closed-form kinematic solver featuring:
- Metacarpal palm normal estimation from the rigid carpal triangle (wrist 0, index MCP 5, pinky MCP 17).
- Closed-form forearm pronation/supination twist (bounded to [-85 deg, +85 deg]).
- Physiological twist bone splitting across ForeArm and ForeArmTwist.
- 3-DoF wrist flexion/extension ([-70 deg, +80 deg]) and deviation ([-20 deg, +30 deg]).
- Closed-form 30-bone phalangeal finger retargeting with curl ([0 deg, 95 deg]) and splay ([+/-15 deg]) bounds.
- 100% backward compatibility when retargeting onto 25-core-bone rigs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fbx_skeleton import FINGER_BONE_SUFFIXES, TWIST_BONE_SUFFIXES
from ..human_motion import BODY24_JOINTS
from .kinematics import HumanoidRig, normalize

# Anatomical landmark indices per digit
FINGER_CHAINS = {
    "Thumb": [1, 2, 3, 4],  # CMC, MCP, IP, Tip
    "Index": [5, 6, 7, 8],  # MCP, PIP, DIP, Tip
    "Middle": [9, 10, 11, 12],  # MCP, PIP, DIP, Tip
    "Ring": [13, 14, 15, 16],  # MCP, PIP, DIP, Tip
    "Pinky": [17, 18, 19, 20],  # MCP, PIP, DIP, Tip
}

# Physiological anatomical angle limits in radians
TWIST_LIMIT_RAD = math.radians(85.0)  # +/- 85 degrees
WRIST_FLEX_MIN_RAD = math.radians(-70.0)  # Extension
WRIST_FLEX_MAX_RAD = math.radians(80.0)  # Flexion
WRIST_DEV_MIN_RAD = math.radians(-20.0)  # Radial deviation
WRIST_DEV_MAX_RAD = math.radians(30.0)  # Ulnar deviation

FINGER_CURL_MIN_RAD = math.radians(0.0)  # Full extension
FINGER_CURL_MAX_RAD = math.radians(95.0)  # Full fist flexion
FINGER_SPLAY_MIN_RAD = math.radians(-15.0)  # Adduction
FINGER_SPLAY_MAX_RAD = math.radians(15.0)  # Abduction

THUMB_OPPOSITION_MAX_RAD = math.radians(60.0)
THUMB_FLEX_MAX_RAD = math.radians(70.0)
THUMB_IP_MAX_RAD = math.radians(80.0)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Analytical Rodrigues formula returning an orthogonal SO(3) matrix."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-10 or abs(angle) < 1e-10:
        return np.eye(3, dtype=np.float64)
    u = axis / norm
    c = float(math.cos(angle))
    s = float(math.sin(angle))
    ux, uy, uz = u[0], u[1], u[2]
    k = np.array(
        [
            [0.0, -uz, uy],
            [uz, 0.0, -ux],
            [-uy, ux, 0.0],
        ],
        dtype=np.float64,
    )
    r = np.eye(3, dtype=np.float64) + s * k + (1.0 - c) * (k @ k)
    # Ensure numerical orthonormality
    u_mat, _, vt_mat = np.linalg.svd(r)
    r_ortho = u_mat @ vt_mat
    if float(np.linalg.det(r_ortho)) < 0.0:
        u_mat[:, -1] *= -1.0
        r_ortho = u_mat @ vt_mat
    return r_ortho


def compute_palm_normal(hand_landmarks_21: np.ndarray, side: str) -> np.ndarray:
    """Compute the unit palmar (ventral) normal vector from the metacarpal triangle.

    Formed by wrist (0), index MCP (5), and pinky MCP (17).
    Both left and right conventions produce a normal pointing outward from the palm
    (ventral direction, towards -Y in standard T-pose).
    """
    lms = np.asarray(hand_landmarks_21, dtype=np.float64)
    p0 = lms[0]
    p5 = lms[5]
    p17 = lms[17]

    v_idx = p5 - p0
    v_pky = p17 - p0

    side_low = side.lower()
    if "left" in side_low:
        n = np.cross(v_pky, v_idx)
    else:
        n = np.cross(v_idx, v_pky)

    norm = float(np.linalg.norm(n))
    if norm < 1e-6:
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return n / norm


def solve_forearm_twist(
    palm_normal: np.ndarray,
    forearm_vector: np.ndarray,
    rest_palm_normal: np.ndarray | None = None,
    side: str = "left",
) -> float:
    """Calculate signed axial forearm pronation/supination twist angle in radians.

    Clamped strictly to [-85 deg, +85 deg].
    """
    u_f = normalize(forearm_vector, fallback=np.array([1.0, 0.0, 0.0]))
    n_palm = normalize(palm_normal, fallback=np.array([0.0, -1.0, 0.0]))
    n_ref = normalize(
        rest_palm_normal
        if rest_palm_normal is not None
        else np.array([0.0, -1.0, 0.0], dtype=np.float64)
    )

    # Project palm normal and rest normal onto the plane orthogonal to forearm axis
    n_perp = n_palm - float(np.dot(n_palm, u_f)) * u_f
    n_ref_perp = n_ref - float(np.dot(n_ref, u_f)) * u_f

    norm_perp = float(np.linalg.norm(n_perp))
    norm_ref = float(np.linalg.norm(n_ref_perp))
    if norm_perp < 1e-6 or norm_ref < 1e-6:
        return 0.0

    u_perp = n_perp / norm_perp
    u_ref_perp = n_ref_perp / norm_ref

    sin_theta = float(np.dot(np.cross(u_ref_perp, u_perp), u_f))
    cos_theta = float(np.dot(u_ref_perp, u_perp))
    theta = math.atan2(sin_theta, cos_theta)

    # Anatomical clamp to [-85 deg, +85 deg]
    return float(np.clip(theta, -TWIST_LIMIT_RAD, TWIST_LIMIT_RAD))


def split_forearm_twist(
    twist_angle: float, side: str = "left", rest_rig: HumanoidRig | None = None
) -> dict[str, float]:
    """Distribute forearm twist between ForeArm and optional ForeArmTwist helper bone.

    Splits 50/50 if ForeArmTwist exists on target rig; otherwise assigns 100% to ForeArm.
    """
    side_cap = "Left" if "left" in side.lower() else "Right"
    forearm_bone = f"rig:{side_cap}ForeArm"
    twist_bone = f"rig:{side_cap}ForeArmTwist"

    has_twist_bone = False
    if rest_rig is None:
        has_twist_bone = True
    elif hasattr(rest_rig, "bones") and twist_bone in rest_rig.bones:
        has_twist_bone = True

    if has_twist_bone:
        half = twist_angle * 0.5
        return {forearm_bone: half, twist_bone: half}
    return {forearm_bone: twist_angle}


def solve_wrist_rotations(
    hand_landmarks_21: np.ndarray,
    side: str,
    forearm_quat: np.ndarray | None = None,
    rest_rig: HumanoidRig | None = None,
) -> np.ndarray:
    """Calculate 3-DoF wrist rotation matrix for rig:LeftHand / rig:RightHand.

    Enforces anatomical limits:
    - Flexion / Extension: [-70 deg, +80 deg]
    - Radial / Ulnar Deviation: [-20 deg, +30 deg]
    Returns an orthogonal SO(3) matrix.
    """
    lms = np.asarray(hand_landmarks_21, dtype=np.float64)
    p0 = lms[0]
    p9 = lms[9]
    v_hand = p9 - p0

    if float(np.linalg.norm(v_hand)) < 1e-6:
        return np.eye(3, dtype=np.float64)
    u_hand = normalize(v_hand)

    side_cap = "Left" if "left" in side.lower() else "Right"
    n_ref_palm = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    u_ref_fwd = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if side_cap == "Left"
        else np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    )
    u_ref_lat = normalize(np.cross(u_ref_fwd, n_ref_palm))

    # 1. Flexion / Extension (sagittal bend towards palm)
    sin_flex = float(np.dot(u_hand, n_ref_palm))
    flex_angle = math.asin(float(np.clip(sin_flex, -1.0, 1.0)))
    flex_clamped = float(np.clip(flex_angle, WRIST_FLEX_MIN_RAD, WRIST_FLEX_MAX_RAD))

    # 2. Deviation (palmar plane lateral bend)
    sin_dev = float(np.dot(u_hand, u_ref_lat))
    dev_angle = math.asin(float(np.clip(sin_dev, -1.0, 1.0)))
    dev_clamped = float(np.clip(dev_angle, WRIST_DEV_MIN_RAD, WRIST_DEV_MAX_RAD))

    r_flex = _rodrigues(u_ref_lat, flex_clamped)
    r_dev = _rodrigues(n_ref_palm, dev_clamped)
    r_wrist = r_dev @ r_flex

    # Validate SO(3) orthonormality
    u_mat, _, vt_mat = np.linalg.svd(r_wrist)
    r_ortho = u_mat @ vt_mat
    if float(np.linalg.det(r_ortho)) < 0.0:
        u_mat[:, -1] *= -1.0
        r_ortho = u_mat @ vt_mat
    return r_ortho


def compute_finger_curl_and_splay(
    hand_landmarks_21: np.ndarray, side: str
) -> dict[str, dict[str, Any]]:
    """Analyse anatomical curl and splay angles across all five digits."""
    lms = np.asarray(hand_landmarks_21, dtype=np.float64)
    n_palm = compute_palm_normal(lms, side)
    p0 = lms[0]
    p9 = lms[9]
    u_meta = normalize(p9 - p0, fallback=np.array([1.0, 0.0, 0.0]))
    u_lat = normalize(np.cross(u_meta, n_palm), fallback=np.array([0.0, 0.0, 1.0]))

    results: dict[str, dict[str, Any]] = {}

    for finger, chain in FINGER_CHAINS.items():
        p_mcp, p_pip, p_dip, p_tip = (
            lms[chain[0]],
            lms[chain[1]],
            lms[chain[2]],
            lms[chain[3]],
        )
        v1 = p_pip - p_mcp
        v2 = p_dip - p_pip
        v3 = p_tip - p_dip

        u1 = normalize(v1, fallback=u_meta)
        u2 = normalize(v2, fallback=u1)
        u3 = normalize(v3, fallback=u2)

        if finger == "Thumb":
            # Thumb kinematics
            opp_sin = float(np.dot(u1, n_palm))
            splay = float(
                np.clip(
                    math.asin(float(np.clip(opp_sin, -1.0, 1.0))),
                    0.0,
                    THUMB_OPPOSITION_MAX_RAD,
                )
            )

            c1 = math.acos(float(np.clip(np.dot(u_meta, u1), -1.0, 1.0)))
            c2 = math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))
            c3 = math.acos(float(np.clip(np.dot(u2, u3), -1.0, 1.0)))

            curl1 = float(np.clip(c1, 0.0, THUMB_FLEX_MAX_RAD))
            curl2 = float(np.clip(c2, 0.0, THUMB_IP_MAX_RAD))
            curl3 = float(np.clip(c3, 0.0, THUMB_IP_MAX_RAD))
        else:
            # Fingers: Index, Middle, Ring, Pinky
            # Curl 1 (MCP): signed atan2 between palmar projection and metacarpal projection
            curl1_sin = float(np.dot(u1, n_palm))
            curl1_cos = float(np.dot(u1, u_meta))
            curl1_raw = math.atan2(curl1_sin, curl1_cos)
            curl1 = float(np.clip(curl1_raw, FINGER_CURL_MIN_RAD, FINGER_CURL_MAX_RAD))

            # Splay angle in palm plane
            u1_palm = u1 - float(np.dot(u1, n_palm)) * n_palm
            norm_u1 = float(np.linalg.norm(u1_palm))
            if norm_u1 > 1e-4:
                u1_palm /= norm_u1
                splay_sin = float(np.dot(np.cross(u_meta, u1_palm), n_palm))
                splay_cos = float(np.dot(u_meta, u1_palm))
                if curl1_cos < 0.0:
                    splay_sin = -splay_sin
                    splay_cos = -splay_cos
                splay_raw = math.atan2(splay_sin, splay_cos)
            else:
                splay_raw = 0.0
            splay = float(np.clip(splay_raw, FINGER_SPLAY_MIN_RAD, FINGER_SPLAY_MAX_RAD))

            # Curl 2 (PIP hinge): signed angle around lateral flexion axis u_lat
            c2_cos = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
            c2_sin = float(np.dot(np.cross(u1, u2), u_lat))
            curl2_raw = math.atan2(c2_sin, c2_cos)
            curl2 = float(np.clip(curl2_raw, FINGER_CURL_MIN_RAD, FINGER_CURL_MAX_RAD))

            # Curl 3 (DIP hinge): signed angle around lateral flexion axis u_lat
            c3_cos = float(np.clip(np.dot(u2, u3), -1.0, 1.0))
            c3_sin = float(np.dot(np.cross(u2, u3), u_lat))
            curl3_raw = math.atan2(c3_sin, c3_cos)
            curl3 = float(np.clip(curl3_raw, FINGER_CURL_MIN_RAD, FINGER_CURL_MAX_RAD))

        results[finger] = {
            "curl_rad": [curl1, curl2, curl3],
            "splay_rad": splay,
            "curl_deg": [math.degrees(c) for c in (curl1, curl2, curl3)],
            "splay_deg": math.degrees(splay),
        }

    return results


def solve_finger_rotations(
    hand_landmarks_21: np.ndarray,
    side: str,
    hand_world_quat: np.ndarray | None = None,
    rest_rig: HumanoidRig | None = None,
) -> dict[str, np.ndarray]:
    """Solve local SO(3) rotations for all 15 phalangeal finger bones on one hand.

    Returns dict mapping canonical bone names e.g. rig:LeftHandIndex1 to (3, 3) rotation matrices.
    Enforces anatomical curl ([0 deg, 95 deg]) and splay ([+/-15 deg]) bounds.
    """
    lms = np.asarray(hand_landmarks_21, dtype=np.float64)
    side_cap = "Left" if "left" in side.lower() else "Right"
    analysis = compute_finger_curl_and_splay(lms, side)

    n_palm = compute_palm_normal(lms, side)
    p0 = lms[0]
    p9 = lms[9]
    u_meta = normalize(p9 - p0, fallback=np.array([1.0, 0.0, 0.0]))
    u_lat = normalize(np.cross(u_meta, n_palm), fallback=np.array([0.0, 0.0, 1.0]))

    rotations: dict[str, np.ndarray] = {}

    for finger, data in analysis.items():
        curls = data["curl_rad"]
        splay = data["splay_rad"]

        # Phalanx 1 (MCP): Splay + Curl
        r_splay = _rodrigues(n_palm, splay)
        r_curl1 = _rodrigues(u_lat, curls[0])
        r1 = r_splay @ r_curl1

        # Phalanx 2 (PIP): 1-DoF Hinge Curl
        r2 = _rodrigues(u_lat, curls[1])

        # Phalanx 3 (DIP): 1-DoF Hinge Curl
        r3 = _rodrigues(u_lat, curls[2])

        bone1 = f"rig:{side_cap}Hand{finger}1"
        bone2 = f"rig:{side_cap}Hand{finger}2"
        bone3 = f"rig:{side_cap}Hand{finger}3"

        rotations[bone1] = r1
        rotations[bone2] = r2
        rotations[bone3] = r3

    return rotations


def solve_hand_and_finger_rotations(
    hand_data_left: Any = None,
    hand_data_right: Any = None,
    motion: Any | None = None,
    frame: int = 0,
    rest_rig: HumanoidRig | None = None,
) -> dict[str, np.ndarray]:
    """Unified solver generating forearm twist, wrist, and finger rotations.

    Filters output by available bones if rest_rig is supplied, preserving 100%
    backward compatibility when targeting standard 25-core-bone rigs.
    """
    results: dict[str, np.ndarray] = {}

    for side, data in (("Left", hand_data_left), ("Right", hand_data_right)):
        if data is None:
            continue
        lms = getattr(data, "landmarks_3d", None)
        if lms is None:
            lms = np.asarray(data, dtype=np.float64)
        if lms.shape != (21, 3):
            continue

        # 1. Metacarpal palm normal
        n_palm = compute_palm_normal(lms, side)

        # 2. Forearm twist
        forearm_vec = None
        if (
            motion is not None
            and hasattr(motion, "joints_3d")
            and motion.joints_3d is not None
            and len(motion.joints_3d) > frame
        ):
            side_low = side.lower()
            try:
                if hasattr(motion, "index") and callable(motion.index):
                    idx_w = motion.index(f"{side_low}_wrist")
                    idx_e = motion.index(f"{side_low}_elbow")
                else:
                    idx_w = BODY24_JOINTS.index(f"{side_low}_wrist")
                    idx_e = BODY24_JOINTS.index(f"{side_low}_elbow")
                forearm_vec = (
                    motion.joints_3d[frame, idx_w] - motion.joints_3d[frame, idx_e]
                )
            except Exception:
                forearm_vec = None

        if forearm_vec is None:
            forearm_vec = (
                np.array([1.0, 0.0, 0.0], dtype=np.float64)
                if side == "Left"
                else np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            )

        twist_angle = solve_forearm_twist(n_palm, forearm_vec, side=side)
        twist_splits = split_forearm_twist(twist_angle, side=side, rest_rig=rest_rig)
        u_f = normalize(forearm_vec)
        for bone_name, angle in twist_splits.items():
            results[bone_name] = _rodrigues(u_f, angle)

        # 3. 3-DoF Wrist orientation
        r_wrist = solve_wrist_rotations(lms, side, rest_rig=rest_rig)
        results[f"rig:{side}Hand"] = r_wrist

        # 4. 15 Phalangeal finger bones
        finger_rots = solve_finger_rotations(lms, side, rest_rig=rest_rig)
        results.update(finger_rots)

    # 5. Filter for target rig backward compatibility
    if rest_rig is not None and hasattr(rest_rig, "bones"):
        rig_bone_set = set(rest_rig.bones.keys())
        results = {k: v for k, v in results.items() if k in rig_bone_set}

    return results
