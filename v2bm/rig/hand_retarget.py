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
from .kinematics import HumanoidRig, human_to_rig, normalize

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
WRIST_TWIST_MIN_RAD = math.radians(-25.0)  # Pronation/supination axial twist lower limit
WRIST_TWIST_MAX_RAD = math.radians(25.0)  # Pronation/supination axial twist upper limit

FINGER_CURL_MIN_RAD = math.radians(0.0)  # Full extension
FINGER_CURL_MAX_RAD = math.radians(95.0)  # Full fist flexion
FINGER_SPLAY_MIN_RAD = math.radians(-15.0)  # Adduction
FINGER_SPLAY_MAX_RAD = math.radians(15.0)  # Abduction

THUMB_OPPOSITION_MAX_RAD = math.radians(60.0)
THUMB_FLEX_MAX_RAD = math.radians(70.0)
THUMB_IP_MAX_RAD = math.radians(80.0)

# Canonical athletic guard curl angles (flexion in radians)
# MCP ~30 deg, PIP ~45 deg, DIP ~25 deg, Thumb ~25 deg
GUARD_CURL_MCP_RAD = math.radians(30.0)
GUARD_CURL_PIP_RAD = math.radians(45.0)
GUARD_CURL_DIP_RAD = math.radians(25.0)
GUARD_THUMB_CMC_RAD = math.radians(20.0)
GUARD_THUMB_MCP_RAD = math.radians(25.0)
GUARD_THUMB_IP_RAD = math.radians(25.0)
GUARD_THUMB_OPP_RAD = math.radians(15.0)


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
    - Pronation / Supination: [-25 deg, +25 deg]
    Returns an orthogonal SO(3) matrix.
    """
    lms = np.asarray(hand_landmarks_21, dtype=np.float64)
    if lms.ndim == 2 and lms.shape[0] == 21:
        p0 = lms[0]
        p9 = lms[9]
        v_hand = p9 - p0
    elif lms.ndim == 2 and lms.shape[0] >= 3:
        p0 = lms[0]
        p_mid = 0.5 * (lms[1] + lms[2])
        v_hand = p_mid - p0
    else:
        return np.eye(3, dtype=np.float64)

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

    # 3. Pronation / Supination (axial twist around longitudinal forearm/hand axis)
    if lms.shape[0] == 21:
        n_palm = compute_palm_normal(lms, side)
    elif lms.shape[0] >= 3:
        v_idx = lms[1] - lms[0]
        v_pky = lms[2] - lms[0]
        raw_n = np.cross(v_pky, v_idx) if side_cap == "Left" else np.cross(v_idx, v_pky)
        n_palm = normalize(raw_n, fallback=n_ref_palm)
    else:
        n_palm = n_ref_palm

    n_palm_proj = n_palm - float(np.dot(n_palm, u_ref_fwd)) * u_ref_fwd
    norm_palm_proj = float(np.linalg.norm(n_palm_proj))
    if norm_palm_proj > 1e-4:
        u_palm_proj = n_palm_proj / norm_palm_proj
        sin_twist = float(np.dot(np.cross(n_ref_palm, u_palm_proj), u_ref_fwd))
        cos_twist = float(np.dot(n_ref_palm, u_palm_proj))
        twist_angle = math.atan2(sin_twist, cos_twist)
        twist_clamped = float(np.clip(twist_angle, WRIST_TWIST_MIN_RAD, WRIST_TWIST_MAX_RAD))
    else:
        twist_clamped = 0.0

    r_flex = _rodrigues(u_ref_lat, flex_clamped)
    r_dev = _rodrigues(n_ref_palm, dev_clamped)
    r_twist = _rodrigues(u_ref_fwd, twist_clamped)
    r_wrist = r_twist @ r_dev @ r_flex

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

            # Curl 2 (PIP hinge): 3D hinge angle with palmar sign check
            c2_cos = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
            c2_sin = float(np.dot(np.cross(u1, u2), u_lat))
            curl2_raw = math.acos(c2_cos) if c2_sin >= -1e-4 else -math.acos(c2_cos)
            curl2 = float(np.clip(curl2_raw, FINGER_CURL_MIN_RAD, FINGER_CURL_MAX_RAD))

            # Curl 3 (DIP hinge): 3D hinge angle with palmar sign check
            c3_cos = float(np.clip(np.dot(u2, u3), -1.0, 1.0))
            c3_sin = float(np.dot(np.cross(u2, u3), u_lat))
            curl3_raw = math.acos(c3_cos) if c3_sin >= -1e-4 else -math.acos(c3_cos)
            curl3 = float(np.clip(curl3_raw, FINGER_CURL_MIN_RAD, FINGER_CURL_MAX_RAD))

            # Collateral ligaments tighten under MCP flexion, reducing splay in a closed fist
            if curl1 > math.radians(30.0):
                splay *= max(0.25, 1.0 - 0.65 * (curl1 / FINGER_CURL_MAX_RAD))

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

    # Construct bone-local rotations around fixed T-pose local phalanx axes
    n_palm_local = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    u_lat_local = (
        np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if side_cap == "Left"
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    # Inter-finger median regularisation across the 4 non-thumb digits to suppress single-digit noise
    four_fingers = ("Index", "Middle", "Ring", "Pinky")
    med_curls = [
        float(np.median([analysis[f]["curl_rad"][j] for f in four_fingers]))
        for j in range(3)
    ]

    rotations: dict[str, np.ndarray] = {}

    for finger, data in analysis.items():
        curls = list(data["curl_rad"])
        splay = float(data["splay_rad"])

        if finger in four_fingers:
            # Blend 70% individual digit curl with 30% 4-finger median curl
            curls = [0.70 * curls[j] + 0.30 * med_curls[j] for j in range(3)]
            # Anatomical flexor tendon coupling: prevent stiff-PIP L-claw when MCP is strongly flexed
            if curls[0] > math.radians(35.0) and curls[1] < 0.65 * curls[0]:
                total = curls[0] + curls[1] + curls[2]
                curls[0] = min(math.radians(68.0), 0.36 * total)
                curls[1] = min(math.radians(82.0), 0.42 * total)
                curls[2] = min(math.radians(60.0), 0.22 * total)
            else:
                curls[0] = min(curls[0], math.radians(72.0))
        else:
            # Keep thumb opposition and flexion natural without over-folding into palm
            splay = min(splay, math.radians(32.0))
            curls[0] = min(curls[0], math.radians(38.0))
            curls[1] = min(curls[1], math.radians(48.0))
            curls[2] = min(curls[2], math.radians(48.0))

        # Phalanx 1 (MCP): Splay + Curl in local bone frame
        r_splay = _rodrigues(n_palm_local, splay)
        r_curl1 = _rodrigues(u_lat_local, curls[0])
        r1 = r_splay @ r_curl1

        # Phalanx 2 (PIP): 1-DoF Hinge Curl in local bone frame
        r2 = _rodrigues(u_lat_local, curls[1])

        # Phalanx 3 (DIP): 1-DoF Hinge Curl in local bone frame
        r3 = _rodrigues(u_lat_local, curls[2])

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
                raw_forearm = (
                    motion.joints_3d[frame, idx_w] - motion.joints_3d[frame, idx_e]
                )
                forearm_vec = human_to_rig(raw_forearm)
            except Exception:
                forearm_vec = None

        local_shaft = (
            np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if side == "Left"
            else np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        )
        if forearm_vec is None:
            forearm_vec = local_shaft.copy()

        twist_angle = solve_forearm_twist(n_palm, forearm_vec, side=side)
        twist_splits = split_forearm_twist(twist_angle, side=side, rest_rig=rest_rig)
        max_distal_twist = math.radians(25.0)
        for bone_name, angle in twist_splits.items():
            clamped_angle = (
                float(np.clip(angle, -max_distal_twist, max_distal_twist))
                if motion is not None and "Twist" in bone_name
                else angle
            )
            results[bone_name] = _rodrigues(local_shaft, clamped_angle)

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


def get_athletic_guard_rotations(
    side: str, rest_rig: HumanoidRig | None = None
) -> dict[str, np.ndarray]:
    """Generate canonical athletic guard rest curl posture for all 15 phalanges.

    Produces non-zero natural flexion:
    - Metacarpophalangeal (MCP) joints: ~30 deg flexion
    - Proximal interphalangeal (PIP) joints: ~45 deg flexion
    - Distal interphalangeal (DIP) joints: ~25 deg flexion
    - Pollex (Thumb) joints: ~15 deg opposition, ~20-25 deg flexion
    Every finger joint exhibits curl > 15.0 deg.
    """
    side_cap = "Left" if "left" in side.lower() else "Right"
    n_palm_local = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    u_lat_local = (
        np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if side_cap == "Left"
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    guard_configs = {
        "Index": (GUARD_CURL_MCP_RAD, GUARD_CURL_PIP_RAD, GUARD_CURL_DIP_RAD, 0.0),
        "Middle": (GUARD_CURL_MCP_RAD, GUARD_CURL_PIP_RAD, GUARD_CURL_DIP_RAD, 0.0),
        "Ring": (math.radians(32.0), math.radians(48.0), GUARD_CURL_DIP_RAD, math.radians(-2.0)),
        "Pinky": (math.radians(35.0), math.radians(50.0), GUARD_CURL_DIP_RAD, math.radians(-4.0)),
    }

    rotations: dict[str, np.ndarray] = {}
    for finger, (c1, c2, c3, splay) in guard_configs.items():
        r1 = _rodrigues(n_palm_local, splay) @ _rodrigues(u_lat_local, c1)
        r2 = _rodrigues(u_lat_local, c2)
        r3 = _rodrigues(u_lat_local, c3)
        rotations[f"rig:{side_cap}Hand{finger}1"] = r1
        rotations[f"rig:{side_cap}Hand{finger}2"] = r2
        rotations[f"rig:{side_cap}Hand{finger}3"] = r3

    # Pollex (Thumb)
    r_th1 = _rodrigues(n_palm_local, GUARD_THUMB_OPP_RAD) @ _rodrigues(u_lat_local, GUARD_THUMB_CMC_RAD)
    r_th2 = _rodrigues(u_lat_local, GUARD_THUMB_MCP_RAD)
    r_th3 = _rodrigues(u_lat_local, GUARD_THUMB_IP_RAD)
    rotations[f"rig:{side_cap}HandThumb1"] = r_th1
    rotations[f"rig:{side_cap}HandThumb2"] = r_th2
    rotations[f"rig:{side_cap}HandThumb3"] = r_th3

    if rest_rig is not None and hasattr(rest_rig, "bones"):
        rig_bone_set = set(rest_rig.bones.keys())
        rotations = {k: v for k, v in rotations.items() if k in rig_bone_set}

    return rotations
