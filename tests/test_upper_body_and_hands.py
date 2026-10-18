"""Tier 2: Mathematical Verification — Upper Body Kinematics & Cranial Solvers.

Verifies Milestone 1 kinematics upgrades:
- 3-DoF cranial solver distributing yaw, pitch, and roll across Neck (40%) and Head (60%) via SLERP.
- Cranial solver with direct quaternion and MediaPipe world landmarks input, and fallback behaviour.
- Progressive spine curvature and axial torsion (25% Spine, 35% Spine1, 40% Spine2).
- Decoupled hips pitch clamping pelvic sagittal tilt to [-15 deg, +20 deg].
- Scapular rhythm coupling clavicle elevation (>60 deg arm elevation) and protraction (cross-body reach).
- SO(3) orthonormality and numerical stability across multi-frame sequences.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pytest

from v2bm.human_motion import BODY24_JOINTS, HumanMotion
from v2bm.rig.fbx_skeleton import ALL_HUMANOID_BONE_SUFFIXES, CORE_BONE_SUFFIXES
from v2bm.rig.kinematics import (
    HumanoidRig,
    R_BODY_ALIGN,
    forward_kinematics,
    normalize,
    quat_wxyz,
    world_positions_m,
)
from v2bm.rig.retarget import (
    _cranial_rotations,
    _hips_rotation,
    _quat_wxyz_to_matrix,
    _shoulder_rotation,
    _slerp,
    retarget_one_frame,
)
from tests.test_fk_skin import create_synthetic_humanoid_rig
from tests.test_retarget_stability import quat_wxyz_to_matrix
from v2bm.hand_extract import (
    ExtractedHandMotion,
    HandLandmarks,
    MediaPipeHandExtractor,
    compute_wrist_roi,
    crop_wrist_roi,
    ensure_hand_landmarker_model,
    remap_crop_to_full,
)
from v2bm.rig.hand_retarget import (
    FINGER_CHAINS,
    _rodrigues,
    compute_finger_curl_and_splay,
    compute_palm_normal,
    get_athletic_guard_rotations,
    solve_finger_rotations,
    solve_forearm_twist,
    solve_hand_and_finger_rotations,
    solve_wrist_rotations,
    split_forearm_twist,
)
from v2bm.rig.foot_ik import (
    apply_foot_ik,
    compute_planar_speed,
    detect_stance_phases,
    pin_stance_contacts,
    solve_leg_ik_rotations,
    solve_two_bone_ik,
)
from v2bm.rig.animation import apply_floor_reach_ik, build_animation
from v2bm.rig.rig_retarget import RigMapping
from v2bm.rig.kinematics import RigBone


def _make_base_motion(num_frames: int = 1) -> HumanMotion:
    """Generate an upright reference HumanMotion in metres."""
    joints_3d = np.zeros((num_frames, 24, 3), dtype=np.float64)
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
        "left_elbow": (-0.18, 1.18, 0.0),
        "right_elbow": (0.18, 1.18, 0.0),
        "left_wrist": (-0.18, 0.88, 0.0),
        "right_wrist": (0.18, 0.88, 0.0),
        "left_hand": (-0.18, 0.80, 0.0),
        "right_hand": (0.18, 0.80, 0.0),
    }
    for name, pos in base_pos.items():
        idx = BODY24_JOINTS.index(name)
        joints_3d[:, idx] = pos

    timestamps = np.arange(num_frames, dtype=np.float64) / 30.0
    root_translation = joints_3d[:, 0, :].copy()
    return HumanMotion(
        joints_3d=joints_3d,
        joint_names=list(BODY24_JOINTS),
        fps=30.0,
        timestamps=timestamps,
        root_translation=root_translation,
        path=Path("synthetic.npz"),
    )


def test_cranial_solver_with_quaternion() -> None:
    """Verify cranial solver distributes direct quaternion input 40% Neck / 60% Head."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    # 30 degree head yaw (rotation about Y axis)
    angle = float(np.radians(30.0))
    q_target = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=np.float64)
    motion.head_rotation_quat = np.array([q_target], dtype=np.float64)

    res = retarget_one_frame(rig, motion, 0)
    world = forward_kinematics(rig, res.r_anim)

    # World rotation of Spine2 (torso) is identity at rest
    r_spine2 = world["rig:Spine2"][:3, :3]
    r_neck = world["rig:Neck"][:3, :3]
    r_head = world["rig:Head"][:3, :3]

    r_target_world = _quat_wxyz_to_matrix(q_target)

    # Composite orientation of Head must match target world orientation
    rot_diff = r_head.T @ r_target_world
    angle_err = np.arccos(float(np.clip((np.trace(rot_diff) - 1.0) / 2.0, -1.0, 1.0)))
    assert angle_err < 1e-5, f"Composite head rotation error: {angle_err:.2e} rad"

    # Neck relative rotation should be ~40% of target angle
    r_rel_neck = r_spine2.T @ r_neck
    neck_angle = np.arccos(float(np.clip((np.trace(r_rel_neck) - 1.0) / 2.0, -1.0, 1.0)))
    assert np.isclose(neck_angle, 0.40 * angle, atol=1e-4), (
        f"Neck angle {neck_angle:.4f} rad should be 40% of {angle:.4f} rad"
    )

    # Head relative to Neck should be ~60% of target angle
    r_rel_head = r_neck.T @ r_head
    head_rel_angle = np.arccos(float(np.clip((np.trace(r_rel_head) - 1.0) / 2.0, -1.0, 1.0)))
    assert np.isclose(head_rel_angle, 0.60 * angle, atol=1e-4), (
        f"Head relative angle {head_rel_angle:.4f} rad should be 60% of {angle:.4f} rad"
    )


def test_cranial_solver_with_landmarks() -> None:
    """Verify cranial solver recovers 3-DoF cranium orientation from facial landmarks."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    # Synthetic MediaPipe world landmarks (33, 3) in camera coordinate system (Y-down, Z-into-scene)
    # MediaPipe pose world landmarks: 0: nose, 7: left_ear, 8: right_ear, 9: mouth_left, 10: mouth_right
    # Internal solver maps: p_v2bm = [x, -y, -z]
    w_lms = np.zeros((1, 33, 3), dtype=np.float64)

    # Create facial landmark geometry rotated by 25 deg pitch forward (nodding)
    pitch_angle = float(np.radians(25.0))
    # In v2bm coordinates: +X right, +Y up, +Z forward
    # Pitch forward is rotation about X axis: Y' = cos*Y - sin*Z, Z' = sin*Y + cos*Z
    c, s = np.cos(pitch_angle), np.sin(pitch_angle)

    # Rest positions in v2bm coords:
    # ears at (-0.08, 1.60, 0.0) and (+0.08, 1.60, 0.0)
    # nose at (0.0, 1.60, 0.10)
    # mouth at (-0.03, 1.55, 0.08) and (+0.03, 1.55, 0.08)
    def rotate_pt(pt: np.ndarray) -> np.ndarray:
        p_c = pt - np.array([0.0, 1.60, 0.0])
        p_rot = np.array([p_c[0], c * p_c[1] - s * p_c[2], s * p_c[1] + c * p_c[2]])
        return p_rot + np.array([0.0, 1.60, 0.0])

    p_lear = rotate_pt(np.array([0.08, 1.60, 0.0]))
    p_rear = rotate_pt(np.array([-0.08, 1.60, 0.0]))
    p_nose = rotate_pt(np.array([0.0, 1.60, 0.10]))
    p_lmouth = rotate_pt(np.array([0.03, 1.60, 0.08]))
    p_rmouth = rotate_pt(np.array([-0.03, 1.60, 0.08]))

    # Convert to MediaPipe camera coords: x_mp = x, y_mp = -y, z_mp = -z
    w_lms[0, 0] = [p_nose[0], -p_nose[1], -p_nose[2]]
    w_lms[0, 7] = [p_lear[0], -p_lear[1], -p_lear[2]]
    w_lms[0, 8] = [p_rear[0], -p_rear[1], -p_rear[2]]
    w_lms[0, 9] = [p_lmouth[0], -p_lmouth[1], -p_lmouth[2]]
    w_lms[0, 10] = [p_rmouth[0], -p_rmouth[1], -p_rmouth[2]]

    motion.world_landmarks = w_lms

    res = retarget_one_frame(rig, motion, 0)
    world = forward_kinematics(rig, res.r_anim)

    r_head = world["rig:Head"][:3, :3]
    # Forward vector (+Z) should point downward by pitch_angle
    u_fwd = r_head[:, 2]
    expected_fwd = np.array([0.0, -np.sin(pitch_angle), np.cos(pitch_angle)], dtype=np.float64)
    dot_fwd = float(np.dot(u_fwd, expected_fwd))
    assert dot_fwd > 0.999, f"Head forward vector dot product {dot_fwd:.4f} expected > 0.999"


def test_cranial_solver_fallback_without_landmarks() -> None:
    """Verify cranial solver fallback executes safely when facial data is absent."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    # Neither quaternion nor facial landmarks provided
    assert not hasattr(motion, "head_rotation_quat") or motion.head_rotation_quat is None
    assert not hasattr(motion, "world_landmarks") or motion.world_landmarks is None

    res = retarget_one_frame(rig, motion, 0)
    assert "rig:Neck" in res.r_anim
    assert "rig:Head" in res.r_anim
    assert np.all(np.isfinite(res.r_anim["rig:Neck"]))
    assert np.all(np.isfinite(res.r_anim["rig:Head"]))


def test_progressive_spine_curvature_and_torsion() -> None:
    """Verify progressive spine splits flexion/torsion across Spine, Spine1, Spine2."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    # Pitch torso forward by 40 degrees and twist yaw by 20 degrees
    pitch_rad = float(np.radians(40.0))
    yaw_rad = float(np.radians(20.0))
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    # Rotation matrix: Yaw * Pitch
    r_target_torso = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    ) @ np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float64,
    )

    # Adjust collar and neck joints to reflect this thoracic orientation
    p_spine3 = np.array([0.0, 1.36, 0.0])
    u_collar_lat = r_target_torso[:, 0]
    u_spine_up = r_target_torso[:, 1]

    idx_lcol = BODY24_JOINTS.index("left_collar")
    idx_rcol = BODY24_JOINTS.index("right_collar")
    idx_neck = BODY24_JOINTS.index("neck")
    idx_spine3 = BODY24_JOINTS.index("spine3")

    motion.joints_3d[0, idx_spine3] = p_spine3 @ R_BODY_ALIGN
    motion.joints_3d[0, idx_neck] = (p_spine3 + 0.14 * u_spine_up) @ R_BODY_ALIGN
    motion.joints_3d[0, idx_lcol] = (p_spine3 + 0.12 * u_spine_up + 0.05 * u_collar_lat) @ R_BODY_ALIGN
    motion.joints_3d[0, idx_rcol] = (p_spine3 + 0.12 * u_spine_up - 0.05 * u_collar_lat) @ R_BODY_ALIGN

    res = retarget_one_frame(rig, motion, 0)
    world = forward_kinematics(rig, res.r_anim)

    r_pelvis = world["rig:Hips"][:3, :3]
    r_spine = world["rig:Spine"][:3, :3]
    r_spine1 = world["rig:Spine1"][:3, :3]
    r_spine2 = world["rig:Spine2"][:3, :3]

    # Ribcage (Spine2) must match target thoracic orientation to high precision
    rot_diff = r_spine2.T @ r_target_torso
    angle_err = np.arccos(float(np.clip((np.trace(rot_diff) - 1.0) / 2.0, -1.0, 1.0)))
    assert angle_err < 1e-4, f"Spine2 alignment error: {angle_err:.2e} rad"

    # Compute relative angle from pelvis to each spine vertebra
    total_rel = r_pelvis.T @ r_spine2
    total_angle = np.arccos(float(np.clip((np.trace(total_rel) - 1.0) / 2.0, -1.0, 1.0)))

    rel_spine = r_pelvis.T @ r_spine
    angle_spine = np.arccos(float(np.clip((np.trace(rel_spine) - 1.0) / 2.0, -1.0, 1.0)))

    rel_spine1 = r_pelvis.T @ r_spine1
    angle_spine1 = np.arccos(float(np.clip((np.trace(rel_spine1) - 1.0) / 2.0, -1.0, 1.0)))

    # Verify cumulative 25% Spine, 60% Spine1 (25% + 35%), 100% Spine2
    assert np.isclose(angle_spine / total_angle, 0.25, atol=0.03), (
        f"Spine ratio {angle_spine / total_angle:.3f} expected ~0.25"
    )
    assert np.isclose(angle_spine1 / total_angle, 0.60, atol=0.03), (
        f"Spine1 cumulative ratio {angle_spine1 / total_angle:.3f} expected ~0.60"
    )


def test_decoupled_hips_pitch_limits_pelvic_tilt() -> None:
    """Verify forward waist flexion up to 60 deg clamps pelvic pitch to <= 20 deg."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    # Pose with deep 60-degree forward waist bow
    bow_angle = float(np.radians(60.0))
    cb, sb = np.cos(bow_angle), np.sin(bow_angle)

    idx_pelvis = BODY24_JOINTS.index("pelvis")
    idx_s1 = BODY24_JOINTS.index("spine1")
    idx_s2 = BODY24_JOINTS.index("spine2")
    idx_s3 = BODY24_JOINTS.index("spine3")
    idx_neck = BODY24_JOINTS.index("neck")

    p_pelvis = motion.joints_3d[0, idx_pelvis]
    # Spine bends forward: +Y becomes cb*Y + sb*Z, etc.
    motion.joints_3d[0, idx_s1] = p_pelvis + np.array([0.0, 0.10 * cb, 0.10 * sb])
    motion.joints_3d[0, idx_s2] = p_pelvis + np.array([0.0, 0.22 * cb, 0.22 * sb])
    motion.joints_3d[0, idx_s3] = p_pelvis + np.array([0.0, 0.34 * cb, 0.34 * sb])
    motion.joints_3d[0, idx_neck] = p_pelvis + np.array([0.0, 0.48 * cb, 0.48 * sb])

    res = retarget_one_frame(rig, motion, 0)
    world = forward_kinematics(rig, res.r_anim)

    r_hips = world["rig:Hips"][:3, :3]
    # Pelvis forward tilt (+Z pointing downward into -Y)
    u_hips_up = r_hips[:, 1]
    pelvis_pitch = float(np.arcsin(np.clip(-u_hips_up[2], -1.0, 1.0)))

    # Maximum allowed pelvic pitch is +20 degrees (0.349 rad)
    assert pelvis_pitch <= float(np.radians(20.5)), (
        f"Pelvic pitch {np.degrees(pelvis_pitch):.1f} deg exceeded 20 deg clamp"
    )

    # Verify legs have not kicked backward: LeftUpLeg aim direction
    r_lleg = world["rig:LeftUpLeg"][:3, :3]
    u_lleg_down = -r_lleg[:, 1]
    # Leg should remain predominantly vertical (Y-down)
    assert u_lleg_down[1] < -0.80, (
        f"Left thigh pitched excessively backward: direction={u_lleg_down}"
    )


def test_scapular_rhythm_elevation_and_protraction() -> None:
    """Verify clavicle elevation activates above 60 deg arm raise, and protracts on forward reach."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    # 1. Arm raised to 45 deg (below 60 deg threshold): clavicle elevation offset should be 0
    idx_lshoulder = BODY24_JOINTS.index("left_shoulder")
    idx_lelbow = BODY24_JOINTS.index("left_elbow")
    p_lshoulder = motion.joints_3d[0, idx_lshoulder]

    angle_45 = float(np.radians(45.0))
    motion.joints_3d[0, idx_lelbow] = p_lshoulder + np.array(
        [-0.30 * np.sin(angle_45), -0.30 * np.cos(angle_45), 0.0]
    )

    res_45 = retarget_one_frame(rig, motion, 0)
    world_45 = forward_kinematics(rig, res_45.r_anim)
    u_clav_45 = world_45["rig:LeftShoulder"][:3, 0]  # Lateral axis

    # 2. Arm raised to 120 deg (overhead): clavicle should elevate significantly
    angle_120 = float(np.radians(120.0))
    motion.joints_3d[0, idx_lelbow] = p_lshoulder + np.array(
        [-0.30 * np.sin(angle_120), -0.30 * np.cos(angle_120), 0.0]
    )

    res_120 = retarget_one_frame(rig, motion, 0)
    world_120 = forward_kinematics(rig, res_120.r_anim)
    u_clav_120 = world_120["rig:LeftShoulder"][:3, 0]

    # Lateral vector of clavicle should tilt upward (+Y component)
    elev_45 = float(u_clav_45[1])
    elev_120 = float(u_clav_120[1])
    assert elev_120 > elev_45 + 0.10, (
        f"Scapular elevation failed to engage: elev_45={elev_45:.3f}, elev_120={elev_120:.3f}"
    )

    # 3. Cross-body reach: arm points forward (-Z human / +Z rig) and across (+X human / -X rig)
    motion.joints_3d[0, idx_lelbow] = p_lshoulder + np.array([0.20, 0.0, -0.25])
    res_reach = retarget_one_frame(rig, motion, 0)
    world_reach = forward_kinematics(rig, res_reach.r_anim)
    u_clav_reach = world_reach["rig:LeftShoulder"][:3, 0]

    # Clavicle protraction: lateral axis angles forward (+Z component)
    assert u_clav_reach[2] > 0.05, (
        f"Scapular protraction failed on cross-body reach: z-component={u_clav_reach[2]:.3f}"
    )


def test_upper_body_stability_across_sequence() -> None:
    """Verify mathematical stability, SO(3) orthonormality, and zero NaNs over motion sequence."""
    rig = create_synthetic_humanoid_rig()
    num_frames = 30
    motion = _make_base_motion(num_frames=num_frames)

    idx_lelbow = BODY24_JOINTS.index("left_elbow")
    p_lshoulder = motion.joints_3d[0, BODY24_JOINTS.index("left_shoulder")]

    for f in range(num_frames):
        t = f / float(num_frames)
        # Smoothly raise arm from 0 to 150 deg and forward reach
        angle = t * float(np.radians(150.0))
        reach = t * 0.20
        motion.joints_3d[f, idx_lelbow] = p_lshoulder + np.array(
            [-0.30 * np.sin(angle), -0.30 * np.cos(angle), reach]
        )

    prev_r: dict[str, np.ndarray] | None = None
    prev_q: dict[str, np.ndarray] | None = None

    for f in range(num_frames):
        res = retarget_one_frame(rig, motion, f, prev_r_anim=prev_r, prev_q_anim=prev_q)
        prev_r = res.r_anim
        prev_q = res.q_anim

        for bone_name, r_mat in res.r_anim.items():
            assert not np.any(np.isnan(r_mat)), f"NaN detected in {bone_name} at frame {f}"
            assert not np.any(np.isinf(r_mat)), f"Inf detected in {bone_name} at frame {f}"

            # SO(3) determinant == +1.0
            det = float(np.linalg.det(r_mat))
            assert np.isclose(det, 1.0, atol=1e-6), (
                f"Non-orthonormal rotation in {bone_name} frame {f}: det={det:.6f}"
            )
            # Orthogonality error: R @ R.T == I
            ortho_err = float(np.max(np.abs(r_mat @ r_mat.T - np.eye(3))))
            assert ortho_err < 1e-6, (
                f"Orthogonality error in {bone_name} frame {f}: {ortho_err:.2e}"
            )


@pytest.mark.parametrize("bow_deg", [0.0, 30.0, 60.0, 69.0, 69.5, 75.0, 85.0, 90.0])
def test_progressive_spine_extreme_forward_bows(bow_deg: float) -> None:
    """Verify progressive spine maintains alignment under extreme forward bows (up to 90 deg)."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    rad = float(np.radians(bow_deg))
    cx, sx = np.cos(rad), np.sin(rad)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

    p_spine3 = np.array([0.0, 1.36, 0.0])
    u_lat = Rx[:, 0]
    u_up = Rx[:, 1]

    idx_s3 = BODY24_JOINTS.index("spine3")
    idx_neck = BODY24_JOINTS.index("neck")
    idx_lcol = BODY24_JOINTS.index("left_collar")
    idx_rcol = BODY24_JOINTS.index("right_collar")

    motion.joints_3d[0, idx_s3] = p_spine3 @ R_BODY_ALIGN
    motion.joints_3d[0, idx_neck] = (p_spine3 + 0.14 * u_up) @ R_BODY_ALIGN
    motion.joints_3d[0, idx_lcol] = (p_spine3 + 0.12 * u_up + 0.05 * u_lat) @ R_BODY_ALIGN
    motion.joints_3d[0, idx_rcol] = (p_spine3 + 0.12 * u_up - 0.05 * u_lat) @ R_BODY_ALIGN

    res = retarget_one_frame(rig, motion, 0)
    world = forward_kinematics(rig, res.r_anim)
    r_s2 = world["rig:Spine2"][:3, :3]

    rot_diff = r_s2.T @ Rx
    err = float(np.arccos(np.clip((np.trace(rot_diff) - 1.0) * 0.5, -1.0, 1.0)))
    assert err < 1e-4, f"Spine2 alignment error at {bow_deg} deg bow is {np.degrees(err):.2f} deg (>1e-4 rad)"


def test_forward_waist_flexion_continuity_sweep() -> None:
    """Verify forward waist flexion from 0 deg through 90 deg is continuous with zero snap jumps."""
    rig = create_synthetic_humanoid_rig()
    num_frames = 91  # 0 to 90 deg inclusive in 1-deg steps
    motion = _make_base_motion(num_frames=num_frames)

    p_spine3 = np.array([0.0, 1.36, 0.0])
    idx_s3 = BODY24_JOINTS.index("spine3")
    idx_neck = BODY24_JOINTS.index("neck")
    idx_lcol = BODY24_JOINTS.index("left_collar")
    idx_rcol = BODY24_JOINTS.index("right_collar")

    for f in range(num_frames):
        rad = float(np.radians(f))
        cx, sx = np.cos(rad), np.sin(rad)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

        u_lat = Rx[:, 0]
        u_up = Rx[:, 1]

        motion.joints_3d[f, idx_s3] = p_spine3 @ R_BODY_ALIGN
        motion.joints_3d[f, idx_neck] = (p_spine3 + 0.14 * u_up) @ R_BODY_ALIGN
        motion.joints_3d[f, idx_lcol] = (p_spine3 + 0.12 * u_up + 0.05 * u_lat) @ R_BODY_ALIGN
        motion.joints_3d[f, idx_rcol] = (p_spine3 + 0.12 * u_up - 0.05 * u_lat) @ R_BODY_ALIGN

    prev_r: dict[str, np.ndarray] | None = None
    prev_q: dict[str, np.ndarray] | None = None
    prev_r_s2: np.ndarray | None = None

    for f in range(num_frames):
        res = retarget_one_frame(rig, motion, f, prev_r_anim=prev_r, prev_q_anim=prev_q)
        prev_r = res.r_anim
        prev_q = res.q_anim

        world = forward_kinematics(rig, res.r_anim)
        r_s2 = world["rig:Spine2"][:3, :3]

        # Target rotation for this frame
        rad = float(np.radians(f))
        cx, sx = np.cos(rad), np.sin(rad)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

        # 1. Alignment error < 1e-4 rad
        rot_diff = r_s2.T @ Rx
        err = float(np.arccos(np.clip((np.trace(rot_diff) - 1.0) * 0.5, -1.0, 1.0)))
        assert err < 1e-4, f"Spine2 alignment error at {f} deg is {err:.2e} rad (>1e-4 rad)"

        # 2. C0/C1 continuity: frame-to-frame geodesic angular jump must remain small (~1.0 deg = ~0.0175 rad)
        if prev_r_s2 is not None:
            step_diff = prev_r_s2.T @ r_s2
            step_angle = float(np.arccos(np.clip((np.trace(step_diff) - 1.0) * 0.5, -1.0, 1.0)))
            assert step_angle < 0.035, (
                f"Snap jump detected between frame {f - 1} and {f}: step_angle={np.degrees(step_angle):.2f} deg"
            )

        prev_r_s2 = r_s2


def _make_synthetic_hand_landmarks(side: str = "left", pose: str = "flat") -> np.ndarray:
    """Generate 21 3D hand landmarks in Humanoid coordinates (+X left, +Y up, +Z forward)."""
    lms = np.zeros((21, 3), dtype=np.float64)
    lms[0] = [0.0, 0.0, 0.0]
    sx = 1.0 if "left" in side.lower() else -1.0

    # Thumb CMC, MCP, IP, Tip
    lms[1] = [0.03 * sx, -0.01, 0.02]
    lms[2] = [0.05 * sx, -0.02, 0.04]
    lms[3] = [0.07 * sx, -0.02, 0.05]
    lms[4] = [0.09 * sx, -0.02, 0.06]

    # Metacarpal heads
    lms[5] = [0.07 * sx, 0.0, 0.03]      # Index MCP (anterior)
    lms[9] = [0.075 * sx, 0.0, 0.0]      # Middle MCP (central longitudinal axis)
    lms[13] = [0.07 * sx, 0.0, -0.015]   # Ring MCP
    lms[17] = [0.065 * sx, 0.0, -0.035]  # Pinky MCP (posterior)

    if pose == "flat":
        for b_idx, z in [(5, 0.03), (9, 0.0), (13, -0.015), (17, -0.035)]:
            lms[b_idx + 1] = [lms[b_idx][0] + 0.03 * sx, 0.0, z]
            lms[b_idx + 2] = [lms[b_idx + 1][0] + 0.025 * sx, 0.0, z]
            lms[b_idx + 3] = [lms[b_idx + 2][0] + 0.02 * sx, 0.0, z]
    elif pose == "fist":
        for b_idx, z in [(5, 0.03), (9, 0.01), (13, -0.015), (17, -0.035)]:
            lms[b_idx + 1] = [lms[b_idx][0] + 0.02 * sx, -0.02, z]
            lms[b_idx + 2] = [lms[b_idx][0] + 0.01 * sx, -0.04, z]
            lms[b_idx + 3] = [lms[b_idx][0], -0.03, z]
        lms[3] = [0.06 * sx, -0.03, 0.02]
        lms[4] = [0.05 * sx, -0.03, 0.0]
    elif pose == "splay":
        splay_z = [(5, 0.05), (9, 0.015), (13, -0.025), (17, -0.055)]
        for b_idx, sz in splay_z:
            lms[b_idx + 1] = [lms[b_idx][0] + 0.03 * sx, 0.0, sz]
            lms[b_idx + 2] = [lms[b_idx + 1][0] + 0.025 * sx, 0.0, sz * 1.2]
            lms[b_idx + 3] = [lms[b_idx + 2][0] + 0.02 * sx, 0.0, sz * 1.4]

    return lms


def test_hand_landmarker_model_download_and_integrity(tmp_path: Path) -> None:
    """Verify ensure_hand_landmarker_model resolves official weights and validates file integrity."""
    # 1. Main model file on disk
    model_path = ensure_hand_landmarker_model()
    assert model_path.exists(), f"Model path {model_path} does not exist"
    assert model_path.name == "hand_landmarker.task"
    file_size = model_path.stat().st_size
    assert file_size >= 7_000_000, f"Expected size >= 7MB, got {file_size} bytes"

    # 2. Test downloading or retrieving to custom path
    test_target = tmp_path / "custom_hand_landmarker.task"
    test_target.write_bytes(model_path.read_bytes()[:100])  # truncated (< 7MB)
    assert test_target.stat().st_size < 7_000_000

    resolved = ensure_hand_landmarker_model(test_target)
    assert resolved == test_target
    assert resolved.exists()
    assert resolved.stat().st_size >= 7_000_000


def test_pose_guided_wrist_roi_crop_and_remapping() -> None:
    """Verify pose-guided wrist ROI computation, upscaling, and exact coordinate remapping."""
    frame_w, frame_h = 640, 480
    p_wrist = np.array([200.0, 300.0])
    p_index = np.array([230.0, 320.0])
    p_elbow = np.array([150.0, 250.0])

    # 1. compute_wrist_roi bounds
    x1, y1, x2, y2 = compute_wrist_roi(frame_w, frame_h, p_wrist, p_index, p_elbow)
    assert 0 <= x1 < x2 <= frame_w
    assert 0 <= y1 < y2 <= frame_h
    crop_w, crop_h = x2 - x1, y2 - y1
    assert crop_w >= 45
    assert crop_h >= 45

    # 2. crop_wrist_roi with synthetic frame
    dummy_frame = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 128
    crop, roi_box = crop_wrist_roi(dummy_frame, p_wrist, p_index, p_elbow, min_size=192)
    assert crop.shape[0] >= 192
    assert crop.shape[1] >= 192
    assert roi_box == (x1, y1, x2, y2)

    # 3. Coordinate remapping: centre of crop (0.5, 0.5, 0.0) -> full-frame normalised centre
    crop_pts = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.1], [1.0, 1.0, -0.1]], dtype=np.float64)
    remapped = remap_crop_to_full(crop_pts, roi_box, frame_w, frame_h)
    assert remapped.shape == (3, 3)

    expected_u = (x1 + 0.5 * crop_w) / float(frame_w)
    expected_v = (y1 + 0.5 * crop_h) / float(frame_h)
    assert np.isclose(remapped[0, 0], expected_u, atol=1e-5)
    assert np.isclose(remapped[0, 1], expected_v, atol=1e-5)

    assert np.all(remapped[:, :2] >= 0.0)
    assert np.all(remapped[:, :2] <= 1.0)


def test_hand_extractor_one_euro_filter() -> None:
    """Verify vectorised One-Euro filtering suppresses spatial jitter while preserving metric fidelity."""
    from v2bm.stabilize_kp2d import OneEuroFilter3D

    num_frames = 60
    base_lms = _make_synthetic_hand_landmarks("left", "flat")
    np.random.seed(42)

    noise = np.random.normal(0.0, 0.005, size=(num_frames, 21, 3))
    noisy_traj = np.repeat(base_lms[np.newaxis, :, :], num_frames, axis=0) + noise

    filter_3d = OneEuroFilter3D(freq=30.0, min_cutoff_xy=1.0, beta_xy=0.05, min_cutoff_z=0.8, beta_z=0.05)
    filtered_traj = filter_3d.filter_trajectory(noisy_traj)

    assert filtered_traj.shape == noisy_traj.shape
    assert not np.any(np.isnan(filtered_traj))
    assert not np.any(np.isinf(filtered_traj))

    var_noisy = np.var(noisy_traj[10:], axis=0)
    var_filtered = np.var(filtered_traj[10:], axis=0)
    assert np.mean(var_filtered) < 0.50 * np.mean(var_noisy), (
        f"One-Euro filter failed to suppress jitter: filtered_var={np.mean(var_filtered):.2e}, noisy_var={np.mean(var_noisy):.2e}"
    )


def test_palm_normal_and_forearm_twist() -> None:
    """Verify metacarpal palm normal computation, +/-85 deg twist clamping, and twist bone splitting."""
    l_hand = _make_synthetic_hand_landmarks("left", "flat")
    r_hand = _make_synthetic_hand_landmarks("right", "flat")

    n_l = compute_palm_normal(l_hand, "left")
    n_r = compute_palm_normal(r_hand, "right")

    assert np.allclose(n_l, [0.0, -1.0, 0.0], atol=1e-3), f"Left palm normal {n_l} expected [0, -1, 0]"
    assert np.allclose(n_r, [0.0, -1.0, 0.0], atol=1e-3), f"Right palm normal {n_r} expected [0, -1, 0]"

    u_forearm_l = np.array([1.0, 0.0, 0.0])
    twist_0 = solve_forearm_twist(n_l, u_forearm_l, side="left")
    assert np.isclose(twist_0, 0.0, atol=1e-4)

    max_twist_rad = np.radians(85.0)
    for deg in np.linspace(-180.0, 180.0, 37):
        rad = np.radians(deg)
        n_rot = np.array([0.0, -np.cos(rad), np.sin(rad)])
        twist = solve_forearm_twist(n_rot, u_forearm_l, side="left")
        assert abs(twist) <= max_twist_rad + 1e-6, (
            f"Twist {np.degrees(twist):.1f} deg exceeded +/-85 deg bound at input {deg} deg"
        )

    rig_synthetic = create_synthetic_humanoid_rig()
    split_with_twist = split_forearm_twist(np.radians(60.0), side="left", rest_rig=None)
    assert np.isclose(split_with_twist["rig:LeftForeArm"], np.radians(30.0))
    assert np.isclose(split_with_twist["rig:LeftForeArmTwist"], np.radians(30.0))

    split_core_only = split_forearm_twist(np.radians(60.0), side="left", rest_rig=rig_synthetic)
    assert np.isclose(split_core_only["rig:LeftForeArm"], np.radians(60.0))
    assert "rig:LeftForeArmTwist" not in split_core_only


def test_wrist_flexion_and_deviation_bounds() -> None:
    """Verify 3-DoF wrist flexion/extension ([-70 deg, +80 deg]) and deviation ([-20 deg, +30 deg])."""
    l_hand = _make_synthetic_hand_landmarks("left", "flat")

    r_rest = solve_wrist_rotations(l_hand, "left")
    assert np.isclose(np.linalg.det(r_rest), 1.0, atol=1e-6)
    assert np.allclose(r_rest, np.eye(3), atol=0.05)

    l_flex = l_hand.copy()
    l_flex[9] = [0.0, -0.08, 0.0]
    r_flex = solve_wrist_rotations(l_flex, "left")
    assert np.isclose(np.linalg.det(r_flex), 1.0, atol=1e-6)
    assert np.max(np.abs(r_flex @ r_flex.T - np.eye(3))) < 1e-6

    l_dev = l_hand.copy()
    l_dev[9] = [0.0, 0.0, 0.08]
    r_dev = solve_wrist_rotations(l_dev, "left")
    assert np.isclose(np.linalg.det(r_dev), 1.0, atol=1e-6)
    assert np.max(np.abs(r_dev @ r_dev.T - np.eye(3))) < 1e-6


def test_30_bone_finger_curl_and_splay_bounds() -> None:
    """Verify all 30 phalangeal finger bones across flat, fist, and splay poses satisfy anatomical curl/splay bounds."""
    for side in ("left", "right"):
        side_cap = "Left" if side == "left" else "Right"
        for pose in ("flat", "fist", "splay"):
            lms = _make_synthetic_hand_landmarks(side, pose)
            analysis = compute_finger_curl_and_splay(lms, side)
            rots = solve_finger_rotations(lms, side)

            assert len(rots) == 15, f"Expected 15 finger bones for {side}, got {len(rots)}"

            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
                data = analysis[finger]
                curls = data["curl_deg"]
                splay = data["splay_deg"]

                if finger != "Thumb":
                    assert -15.01 <= splay <= 15.01, (
                        f"{side_cap} {finger} splay {splay:.1f} deg exceeded [-15, +15] deg in {pose} pose"
                    )
                    for ph_idx, c in enumerate(curls):
                        assert 0.0 <= c <= 95.01, (
                            f"{side_cap} {finger}{ph_idx + 1} curl {c:.1f} deg exceeded [0, 95] deg in {pose} pose"
                        )
                else:
                    assert 0.0 <= splay <= 60.01, f"{side_cap} Thumb opposition {splay:.1f} deg exceeded [0, 60] deg"
                    assert 0.0 <= curls[0] <= 70.01, f"{side_cap} Thumb1 curl {curls[0]:.1f} deg exceeded [0, 70] deg"
                    assert 0.0 <= curls[1] <= 80.01, f"{side_cap} Thumb2 curl {curls[1]:.1f} deg exceeded [0, 80] deg"
                    assert 0.0 <= curls[2] <= 80.01, f"{side_cap} Thumb3 curl {curls[2]:.1f} deg exceeded [0, 80] deg"

                for ph_num in (1, 2, 3):
                    bname = f"rig:{side_cap}Hand{finger}{ph_num}"
                    assert bname in rots, f"Missing {bname} in solved finger rotations"
                    r_mat = rots[bname]
                    det = float(np.linalg.det(r_mat))
                    assert np.isclose(det, 1.0, atol=1e-6), f"{bname} non-orthonormal det={det:.6f}"
                    ortho_err = float(np.max(np.abs(r_mat @ r_mat.T - np.eye(3))))
                    assert ortho_err < 1e-6, f"{bname} orthogonality error={ortho_err:.2e}"


def test_25_bone_rig_backward_compatibility() -> None:
    """Verify solve_hand_and_finger_rotations preserves 100% backward compatibility on 25-core-bone rigs."""
    rig = create_synthetic_humanoid_rig()
    motion = _make_base_motion(num_frames=1)

    l_hand = _make_synthetic_hand_landmarks("left", "flat")
    r_hand = _make_synthetic_hand_landmarks("right", "flat")

    solved = solve_hand_and_finger_rotations(
        hand_data_left=l_hand,
        hand_data_right=r_hand,
        motion=motion,
        frame=0,
        rest_rig=rig,
    )

    for b in solved.keys():
        assert not any(f in b for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")), (
            f"Finger bone {b} unexpectedly leaked into 25-bone rig output"
        )
        assert b in rig.bones, f"Bone {b} is not in target 25-bone rig"

    assert "rig:LeftForeArm" in solved
    assert "rig:RightForeArm" in solved
    assert "rig:LeftHand" in solved
    assert "rig:RightHand" in solved

    world = forward_kinematics(rig, solved)
    assert "rig:LeftHand" in world
    assert "rig:RightHand" in world
    assert not np.any(np.isnan(world["rig:LeftHand"]))
    assert not np.any(np.isnan(world["rig:RightHand"]))


def test_mediapipe_hand_extractor_lifecycle_and_inference() -> None:
    """Verify MediaPipeHandExtractor lifecycle, Body24 and 33-pose parsing, fallback, and resource cleanup."""
    with MediaPipeHandExtractor(min_detection_confidence=0.2) as ext:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Body24 pose landmark layout (24, 2)
        pose_24 = np.zeros((24, 2), dtype=np.float64)
        pose_24[20] = [300.0, 200.0]  # left_wrist
        pose_24[22] = [320.0, 210.0]  # left_hand
        pose_24[18] = [250.0, 180.0]  # left_elbow
        pose_24[21] = [400.0, 200.0]  # right_wrist
        pose_24[23] = [420.0, 210.0]  # right_hand
        pose_24[19] = [450.0, 180.0]  # right_elbow

        res_24 = ext.extract_frame(dummy_frame, pose_landmarks_2d=pose_24)
        assert isinstance(res_24, dict)
        assert set(res_24.keys()) == {"left", "right"}
        assert res_24["left"] is None
        assert res_24["right"] is None

        # 2. MediaPipe Pose 33-landmark layout (33, 2)
        pose_33 = np.zeros((33, 2), dtype=np.float64)
        pose_33[15] = [300.0, 200.0]  # left_wrist
        pose_33[19] = [320.0, 210.0]  # left_index
        pose_33[13] = [250.0, 180.0]  # left_elbow
        pose_33[16] = [400.0, 200.0]  # right_wrist
        pose_33[20] = [420.0, 210.0]  # right_index
        pose_33[14] = [450.0, 180.0]  # right_elbow

        res_33 = ext.extract_frame(dummy_frame, pose_landmarks_2d=pose_33)
        assert isinstance(res_33, dict)
        assert set(res_33.keys()) == {"left", "right"}
        assert res_33["left"] is None
        assert res_33["right"] is None

        # 3. Full-frame fallback when pose_landmarks_2d is None
        res_none = ext.extract_frame(dummy_frame, pose_landmarks_2d=None)
        assert isinstance(res_none, dict)
        assert set(res_none.keys()) == {"left", "right"}

        # 4. Filter reset
        ext.reset_filters()

    # 5. Closed extractor raises RuntimeError
    with pytest.raises(RuntimeError, match="closed"):
        ext.extract_frame(dummy_frame)


def test_mediapipe_hand_extractor_detection_output_structure() -> None:
    """Verify HandLandmarks output data structure, coordinate remapping, and downstream solver wiring."""
    from unittest.mock import MagicMock

    class MockLandmark:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x = x
            self.y = y
            self.z = z

    class MockCategory:
        def __init__(self, score: float, category_name: str) -> None:
            self.score = score
            self.category_name = category_name

    class MockDetectionResult:
        def __init__(self) -> None:
            self.hand_landmarks = [
                [MockLandmark(0.5 + 0.01 * i, 0.5 + 0.01 * i, 0.0) for i in range(21)]
            ]
            self.hand_world_landmarks = [
                [MockLandmark(0.02 * i, -0.01 * i, -0.005 * i) for i in range(21)]
            ]
            self.handedness = [[MockCategory(0.92, "Left")]]

    with MediaPipeHandExtractor(min_detection_confidence=0.2) as ext:
        ext.landmarker.detect = MagicMock(return_value=MockDetectionResult())

        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        pose_24 = np.zeros((24, 2), dtype=np.float64)
        pose_24[20] = [300.0, 200.0]
        pose_24[22] = [320.0, 210.0]
        pose_24[18] = [250.0, 180.0]

        res = ext.extract_frame(dummy_frame, pose_landmarks_2d=pose_24)

        assert res["left"] is not None
        left_hand = res["left"]
        assert isinstance(left_hand, HandLandmarks)
        assert left_hand.side == "left"
        assert np.isclose(left_hand.confidence, 0.92)
        assert left_hand.landmarks_3d.shape == (21, 3)
        assert left_hand.landmarks_screen.shape == (21, 3)
        assert not np.any(np.isnan(left_hand.landmarks_3d))
        assert not np.any(np.isnan(left_hand.landmarks_screen))

        # Remapped screen coords must be in [0, 1] range
        assert np.all(left_hand.landmarks_screen[:, :2] >= 0.0)
        assert np.all(left_hand.landmarks_screen[:, :2] <= 1.0)

        # Downstream solver wiring: verify landmarks feed directly into solve_hand_and_finger_rotations
        solved = solve_hand_and_finger_rotations(
            hand_data_left=left_hand,
        )
        assert "rig:LeftForeArm" in solved
        assert "rig:LeftHand" in solved
        assert "rig:LeftHandIndex1" in solved


def test_finger_curl_hyperextension_and_mcp_flexion_bounds() -> None:
    """Verify signed hinge angle calculation clamps hyperextension (<0 deg) to 0 deg and prevents asin wrapping at 95 deg."""
    for side in ("left", "right"):
        side_cap = "Left" if side == "left" else "Right"
        sx = 1.0 if side == "left" else -1.0

        # 1. Dorsal Hyperextension (< 0 deg) must clamp strictly to 0.0 deg
        for hyp_deg in (10.0, 20.0, 30.0):
            rad_hyp = math.radians(hyp_deg)
            lms_hyp = _make_synthetic_hand_landmarks(side, "flat")
            for mcp_idx in (5, 9, 13, 17):
                p_pip = lms_hyp[mcp_idx + 1]
                # Phalanx 2 (PIP to DIP) bent dorsally (+Y)
                lms_hyp[mcp_idx + 2] = [
                    p_pip[0] + 0.025 * sx * math.cos(rad_hyp),
                    p_pip[1] + 0.025 * math.sin(rad_hyp),
                    p_pip[2],
                ]
                # Phalanx 3 (DIP to Tip) aligned with Phalanx 2
                p_dip = lms_hyp[mcp_idx + 2]
                lms_hyp[mcp_idx + 3] = [
                    p_dip[0] + 0.02 * sx * math.cos(rad_hyp),
                    p_dip[1] + 0.02 * math.sin(rad_hyp),
                    p_dip[2],
                ]

            analysis = compute_finger_curl_and_splay(lms_hyp, side)
            rots = solve_finger_rotations(lms_hyp, side)

            for finger in ("Index", "Middle", "Ring", "Pinky"):
                curls = analysis[finger]["curl_deg"]
                assert np.isclose(curls[1], 0.0, atol=1e-3), (
                    f"{side_cap} {finger} PIP hyperextension of -{hyp_deg} deg failed to clamp to 0 deg (got {curls[1]:.2f} deg)"
                )
                assert np.isclose(curls[2], 0.0, atol=1e-3), (
                    f"{side_cap} {finger} DIP hyperextension of -{hyp_deg} deg failed to clamp to 0 deg (got {curls[2]:.2f} deg)"
                )

                # Rotation matrices for PIP and DIP must be identity under clamped 0 deg curl
                b_pip = f"rig:{side_cap}Hand{finger}2"
                b_dip = f"rig:{side_cap}Hand{finger}3"
                assert np.allclose(rots[b_pip], np.eye(3), atol=1e-4), (
                    f"{b_pip} rotation should be identity under clamped hyperextension"
                )
                assert np.allclose(rots[b_dip], np.eye(3), atol=1e-4), (
                    f"{b_dip} rotation should be identity under clamped hyperextension"
                )

        # 2. Monotonic MCP Flexion up to 95.0 deg (verifying no asin wrapping at 90 deg)
        flex_angles = [0.0, 30.0, 60.0, 85.0, 90.0, 95.0, 110.0]
        prev_mcp_curl = -1.0
        for flex_deg in flex_angles:
            rad_flex = math.radians(flex_deg)
            lms_flex = _make_synthetic_hand_landmarks(side, "flat")
            for mcp_idx in (5, 9, 13, 17):
                p_mcp = lms_flex[mcp_idx]
                lms_flex[mcp_idx + 1] = [
                    p_mcp[0] + 0.03 * sx * math.cos(rad_flex),
                    p_mcp[1] - 0.03 * math.sin(rad_flex),
                    p_mcp[2],
                ]
                p_pip = lms_flex[mcp_idx + 1]
                lms_flex[mcp_idx + 2] = [
                    p_pip[0] + 0.025 * sx * math.cos(rad_flex),
                    p_pip[1] - 0.025 * math.sin(rad_flex),
                    p_pip[2],
                ]
                p_dip = lms_flex[mcp_idx + 2]
                lms_flex[mcp_idx + 3] = [
                    p_dip[0] + 0.02 * sx * math.cos(rad_flex),
                    p_dip[1] - 0.02 * math.sin(rad_flex),
                    p_dip[2],
                ]

            analysis_flex = compute_finger_curl_and_splay(lms_flex, side)
            rots_flex = solve_finger_rotations(lms_flex, side)

            for finger in ("Index", "Middle", "Ring", "Pinky"):
                mcp_curl = analysis_flex[finger]["curl_deg"][0]
                expected_mcp = min(flex_deg, 95.0)
                assert np.isclose(mcp_curl, expected_mcp, atol=0.1), (
                    f"{side_cap} {finger} MCP curl at {flex_deg} deg input expected {expected_mcp} deg, got {mcp_curl:.2f} deg"
                )
                assert mcp_curl >= prev_mcp_curl - 1e-3, (
                    f"{side_cap} {finger} MCP curl failed monotonicity: prev={prev_mcp_curl:.2f} deg, curr={mcp_curl:.2f} deg"
                )

                # SO(3) orthonormality
                for ph in (1, 2, 3):
                    bname = f"rig:{side_cap}Hand{finger}{ph}"
                    r_mat = rots_flex[bname]
                    det = float(np.linalg.det(r_mat))
                    assert np.isclose(det, 1.0, atol=1e-6)
                    assert np.max(np.abs(r_mat @ r_mat.T - np.eye(3))) < 1e-6

            prev_mcp_curl = min(flex_deg, 95.0)


def test_stance_phase_detection_accuracy() -> None:
    """Verify stance phase detection accuracy, speed/height thresholds, and morphological filtering."""
    n_frames = 35
    fps = 30.0
    dt = 1.0 / fps

    # 1. Direct planar speed verification
    pos_line = np.zeros((5, 3), dtype=np.float64)
    pos_line[:, 0] = [0.0, 0.1, 0.2, 0.3, 0.4]  # 0.1 m displacement per frame = 3.0 m/s
    v_speed = compute_planar_speed(pos_line, dt)
    assert np.allclose(v_speed, 3.0, atol=1e-5), f"Expected planar speed 3.0 m/s, got {v_speed}"

    # Stationary planar speed
    pos_still = np.zeros((5, 3), dtype=np.float64)
    assert np.allclose(compute_planar_speed(pos_still, dt), 0.0, atol=1e-5)

    # 2. Multi-phase sequence for left and right ankles
    # Default to flight at y = 0.50 m
    ankle_pos = np.full((n_frames, 2, 3), 0.50, dtype=np.float64)

    # Left foot:
    # Frames 5 to 14 (10 frames): firm stationary stance at y = 0.08 m
    ankle_pos[5:15, 0, 1] = 0.08
    ankle_pos[5:15, 0, 0] = 0.10
    ankle_pos[5:15, 0, 2] = 0.00

    # Frames 15 to 24 (10 frames): flight at y = 0.45 m with horizontal travel
    ankle_pos[15:25, 0, 1] = 0.45
    ankle_pos[15:25, 0, 0] = np.linspace(0.10, 0.80, 10)

    # Frame 26: 1-frame transient glitch (low and stationary, should be rejected)
    ankle_pos[26, 0, 1] = 0.08
    ankle_pos[26, 0, 0] = 0.85

    # Frames 29 to 33: near-ground high-speed sweep (y = 0.08 m, but moving at 2.0 m/s > 0.15 m/s)
    ankle_pos[29:34, 0, 1] = 0.08
    ankle_pos[29:34, 0, 0] = np.linspace(1.0, 1.0 + 4 * (2.0 * dt), 5)

    # Right foot:
    # Stance at frames 18 to 28 (11 frames) at y = 0.09 m
    ankle_pos[18:29, 1, 1] = 0.09
    ankle_pos[18:29, 1, 0] = -0.10
    ankle_pos[18:29, 1, 2] = 0.05

    mask = detect_stance_phases(
        ankle_pos,
        fps=fps,
        speed_thresh=0.15,
        height_thresh=0.15,
        min_stance_frames=3,
    )

    assert mask.shape == (n_frames, 2)
    # Left foot assertions
    assert np.all(mask[5:15, 0] == True), "Valid left stance interval (5:15) was not detected"
    assert np.all(mask[:5, 0] == False), "Pre-stance flight falsely classified as stance"
    assert np.all(mask[15:25, 0] == False), "Airborne swing phase falsely classified as stance"
    assert mask[26, 0] == False, "Single-frame transient glitch was not suppressed by morphological filtering"
    assert np.all(mask[29:34, 0] == False), "Fast ground-sweep was falsely classified as stance"

    # Right foot assertions
    assert np.all(mask[18:29, 1] == True), "Valid right stance interval (18:29) was not detected"
    assert np.all(mask[:18, 1] == False)
    assert np.all(mask[29:, 1] == False)

    # Single-foot input dimension handling: shape (T, 3) -> (T,)
    single_mask = detect_stance_phases(ankle_pos[:, 0, :], fps=fps)
    assert single_mask.shape == (n_frames,)
    assert np.array_equal(single_mask, mask[:, 0])


def test_stance_foot_pinning_drift_under_one_millimetre() -> None:
    """Verify 3D world stance pinning eliminates horizontal drift to < 1 mm during stance."""
    n_frames = 30
    ankle_pos = np.full((n_frames, 2, 3), 0.50, dtype=np.float64)

    # Left foot stance interval from frame 6 to 19 (14 frames)
    ankle_pos[6:20, 0, 1] = 0.10
    # Inject significant horizontal skating / tracking drift (8 cm in X, 6 cm in Z)
    ankle_pos[6:20, 0, 0] = np.linspace(0.10, 0.18, 14)
    ankle_pos[6:20, 0, 2] = np.linspace(-0.03, 0.03, 14)

    # Flight before and after
    ankle_pos[:6, 0, 0] = np.linspace(0.0, 0.09, 6)
    ankle_pos[20:, 0, 0] = np.linspace(0.20, 0.40, 10)

    mask = np.zeros((n_frames, 2), dtype=bool)
    mask[6:20, 0] = True

    blend_window = 3
    pinned = pin_stance_contacts(ankle_pos, mask, blend_window=blend_window)

    # 1. Non-stance frames must remain completely untouched
    assert np.allclose(pinned[:6], ankle_pos[:6], atol=1e-12)
    assert np.allclose(pinned[20:], ankle_pos[20:], atol=1e-12)

    # 2. Interior stance frames (frames 6 + blend_window to 20 - 1 - blend_window)
    # Frames 9 to 16 inclusive are fully locked
    interior_frames = pinned[9:17, 0]
    frame_diffs = np.diff(interior_frames[:, [0, 2]], axis=0)
    step_drifts = np.linalg.norm(frame_diffs, axis=-1)
    max_step_drift = float(np.max(step_drifts))

    assert max_step_drift < 1e-4, (
        f"Interior stance step drift of {max_step_drift:.6f} m violates < 1 mm requirement"
    )

    # Overall horizontal span across the entire interior interval must be < 1 mm
    span_x = float(np.ptp(interior_frames[:, 0]))
    span_z = float(np.ptp(interior_frames[:, 2]))
    total_span = math.sqrt(span_x * span_x + span_z * span_z)
    assert total_span < 1e-4, (
        f"Total interior stance span of {total_span:.6f} m violates < 1 mm requirement"
    )

    # 3. Smoothstep transition continuity at boundaries
    # At frame 6 (entry start): pinned is weighted blend of raw and anchor
    anchor = np.array([
        np.median(ankle_pos[6:20, 0, 0]),
        np.min(ankle_pos[6:20, 0, 1]),
        np.median(ankle_pos[6:20, 0, 2]),
    ])
    assert not np.allclose(pinned[6, 0], ankle_pos[6, 0])
    assert not np.allclose(pinned[6, 0], anchor)

    # Single-foot array input support: shape (T, 3) -> (T, 3)
    single_pinned = pin_stance_contacts(ankle_pos[:, 0, :], mask[:, 0], blend_window=blend_window)
    assert single_pinned.shape == (n_frames, 3)
    assert np.allclose(single_pinned, pinned[:, 0, :])


def test_two_bone_ik_bone_length_preservation_and_pole_vector() -> None:
    """Verify closed-form 2-bone IK bone length conservation (< 1e-4 m) and pole vector alignment."""
    p_hip = np.array([0.10, 0.96, 0.0], dtype=np.float64)
    l1 = 0.44  # thigh length (metres)
    l2 = 0.42  # shin length (metres)
    total_reach = l1 + l2

    test_targets = [
        # Normal reachable target (extended downwards and forward)
        np.array([0.10, 0.40, 0.25], dtype=np.float64),
        # Diagonal reachable target
        np.array([0.25, 0.35, -0.15], dtype=np.float64),
        # Over-extended target beyond reach (D = 1.20 m > 0.86 m)
        np.array([0.10, -0.24, 0.0], dtype=np.float64),
        # Extreme fold target inside minimum reach (D = 0.01 m < 0.02 m)
        p_hip + np.array([0.0, -0.01, 0.0], dtype=np.float64),
    ]

    pole_vectors = [
        np.array([0.0, 0.0, 1.0], dtype=np.float64),   # Standard anterior (+Z)
        np.array([1.0, 0.0, 0.0], dtype=np.float64),   # Lateral (+X)
        np.array([0.0, 0.0, -1.0], dtype=np.float64),  # Posterior (-Z)
    ]

    for p_target in test_targets:
        for v_pole in pole_vectors:
            p_knee = solve_two_bone_ik(p_hip, p_target, l1, l2, v_pole=v_pole)

            # 1. Thigh length preservation (< 1e-4 m)
            actual_l1 = float(np.linalg.norm(p_knee - p_hip))
            err_l1 = abs(actual_l1 - l1)
            assert err_l1 < 1e-4, f"Thigh length error {err_l1:.2e} m violates < 1e-4 m"

            # 2. Shin length preservation for reachable targets
            d_target = float(np.linalg.norm(p_target - p_hip))
            if abs(l1 - l2) + 1e-4 < d_target < total_reach - 1e-4:
                actual_l2 = float(np.linalg.norm(p_target - p_knee))
                err_l2 = abs(actual_l2 - l2)
                assert err_l2 < 1e-4, f"Shin length error {err_l2:.2e} m violates < 1e-4 m"

                # 3. Pole vector half-plane alignment
                u_d = (p_target - p_hip) / d_target
                v_ortho = v_pole - float(np.dot(v_pole, u_d)) * u_d
                if float(np.linalg.norm(v_ortho)) > 1e-4:
                    u_bend_expected = v_ortho / float(np.linalg.norm(v_ortho))
                    knee_disp = p_knee - p_hip
                    knee_bend_component = knee_disp - float(np.dot(knee_disp, u_d)) * u_d
                    assert float(np.dot(knee_bend_component, u_bend_expected)) > 0.0, (
                        "Solved knee position does not align with preferred pole vector direction"
                    )

    # 4. Local bone rotation solving under forward kinematics
    skel = create_synthetic_humanoid_rig()
    fk_rest = forward_kinematics(skel, {})
    p_target_test = p_hip + np.array([0.05, -0.65, 0.15], dtype=np.float64)
    v_pole_test = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    p_knee_solved = solve_two_bone_ik(p_hip, p_target_test, l1, l2, v_pole=v_pole_test)
    u_d_test = normalize(p_target_test - p_hip)
    v_ortho_test = v_pole_test - float(np.dot(v_pole_test, u_d_test)) * u_d_test
    u_bend_test = v_ortho_test / float(np.linalg.norm(v_ortho_test))

    leg_rots = solve_leg_ik_rotations(
        skel=skel,
        world_transforms=fk_rest,
        side="Left",
        p_hip=p_hip,
        p_knee=p_knee_solved,
        p_target=p_target_test,
        u_bend=u_bend_test,
    )

    for bname in ("rig:LeftUpLeg", "rig:LeftLeg", "rig:LeftFoot"):
        assert bname in leg_rots
        r_mat = leg_rots[bname]
        det = float(np.linalg.det(r_mat))
        assert np.isclose(det, 1.0, atol=1e-6), f"{bname} non-orthonormal det={det}"
        assert np.max(np.abs(r_mat @ r_mat.T - np.eye(3))) < 1e-6

    # Test forward kinematics reach with solved rotations
    fk_check = forward_kinematics(skel, leg_rots)
    pos_check = world_positions_m(skel, fk_check)
    knee_err = float(np.linalg.norm(pos_check["rig:LeftLeg"] - p_knee_solved))
    ankle_err = float(np.linalg.norm(pos_check["rig:LeftFoot"] - p_target_test))
    assert knee_err < 1e-4, f"FK knee error {knee_err:.2e} m violates < 1e-4 m"
    assert ankle_err < 1e-4, f"FK ankle error {ankle_err:.2e} m violates < 1e-4 m"


def test_end_to_end_build_animation_hand_and_foot_ik_wiring() -> None:
    """Verify build_animation wiring of 21-point hand retargeting and foot IK with legacy fallback."""
    n_frames = 10
    motion = _make_base_motion(num_frames=n_frames)

    # 1. Base 25-bone rig
    skel_25 = create_synthetic_humanoid_rig()
    mapping_25 = RigMapping(
        skeleton_rig=skel_25,
        target_rig=skel_25,
        basis_change={},
        translation_scale=1.0,
        r_world_align=np.eye(3),
        max_basis_deviation_deg=0.0,
    )

    # 2. Articulated target rig (simulating 55+ bone character)
    art_bones = dict(skel_25.bones)
    test_fingers = [
        "rig:LeftHandIndex1",
        "rig:LeftHandIndex2",
        "rig:LeftHandThumb1",
        "rig:RightHandIndex1",
        "rig:RightHandMiddle1",
    ]
    for fname in test_fingers:
        art_bones[fname] = RigBone(
            name=fname,
            parent="rig:LeftHand" if "Left" in fname else "rig:RightHand",
            children=[],
            pre_rotation=np.eye(3),
            lcl_translation_cm=np.array([2.0, 0.0, 0.0], dtype=np.float64),
            rest_world=np.eye(4),
            rest_local=np.eye(4),
            pre_rotation_euler_deg=np.zeros(3),
        )

    rig_articulated = HumanoidRig(
        bones=art_bones,
        order=list(art_bones.keys()),
        meters_per_unit=0.01,
        source=Path("mock_articulated.fbx"),
    )
    mapping_articulated = RigMapping(
        skeleton_rig=skel_25,
        target_rig=rig_articulated,
        basis_change={},
        translation_scale=1.0,
        r_world_align=np.eye(3),
        max_basis_deviation_deg=0.0,
    )

    # Hand data payloads
    l_hands = [_make_synthetic_hand_landmarks("left", "flat") for _ in range(n_frames)]
    r_hands = [_make_synthetic_hand_landmarks("right", "flat") for _ in range(n_frames)]
    hand_payload = {"left": l_hands, "right": r_hands}

    # Test Case A: Legacy execution (no hand data, foot IK disabled)
    anim_legacy = build_animation(
        motion,
        mapping_25,
        list(range(n_frames)),
        hand_data=None,
        enable_foot_ik=False,
        verbose=False,
    )
    assert len(anim_legacy.r_anim_rig) == n_frames
    assert len(anim_legacy.hips_cm_rig) == n_frames
    for b in anim_legacy.r_anim_rig[0].keys():
        assert not any(f in b for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")), (
            f"Finger bone {b} leaked into legacy fallback output"
        )

    # Test Case B: 25-bone rig with hand data & foot IK enabled (backward compatibility)
    anim_25 = build_animation(
        motion,
        mapping_25,
        list(range(n_frames)),
        hand_data=hand_payload,
        enable_foot_ik=True,
        verbose=False,
    )
    assert len(anim_25.r_anim_rig) == n_frames
    # Target 25-bone rig must strictly omit finger bones
    for b in anim_25.r_anim_rig[0].keys():
        assert not any(f in b for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")), (
            f"Finger bone {b} unexpectedly leaked into 25-bone target rig output"
        )
    # Skeleton rotations retain hand orientation channels
    assert "rig:LeftHand" in anim_25.r_anim_skeleton[0]
    assert "rig:RightHand" in anim_25.r_anim_skeleton[0]

    # Test Case C: Articulated rig with hand data & foot IK enabled
    anim_art = build_animation(
        motion,
        mapping_articulated,
        list(range(n_frames)),
        hand_data=hand_payload,
        enable_foot_ik=True,
        verbose=False,
    )
    assert len(anim_art.r_anim_rig) == n_frames
    # Articulated target rig must receive active finger rotations
    for fname in test_fingers:
        assert fname in anim_art.r_anim_rig[0], f"Articulated rig missing active channel {fname}"
        r_f = anim_art.r_anim_rig[0][fname]
        det = float(np.linalg.det(r_f))
        assert np.isclose(det, 1.0, atol=1e-5), f"{fname} non-orthonormal det={det}"
        assert np.max(np.abs(r_f @ r_f.T - np.eye(3))) < 1e-5


def test_forearm_swing_preservation_and_local_shaft_twist() -> None:
    """Verify ForeArmTwist rotates purely around the local longitudinal shaft axis and does not overwrite elbow flexion."""
    n_frames = 8
    motion = _make_base_motion(num_frames=n_frames)
    # Bend the left elbow forward (+Z in Humanoid, -Z in Human) by 60 deg across all frames
    idx_le = BODY24_JOINTS.index("left_elbow")
    idx_lw = BODY24_JOINTS.index("left_wrist")
    idx_lh = BODY24_JOINTS.index("left_hand")
    for t in range(n_frames):
        p_e = motion.joints_3d[t, idx_le].copy()
        motion.joints_3d[t, idx_lw] = p_e + np.array([-0.12, 0.08, -0.20], dtype=np.float64)
        motion.joints_3d[t, idx_lh] = motion.joints_3d[t, idx_lw] + np.array([-0.04, 0.02, -0.08], dtype=np.float64)

    skel_25 = create_synthetic_humanoid_rig()
    art_bones = dict(skel_25.bones)
    art_bones["rig:LeftForeArmTwist"] = RigBone(
        name="rig:LeftForeArmTwist",
        parent="rig:LeftForeArm",
        children=[],
        pre_rotation=np.eye(3),
        lcl_translation_cm=np.array([12.0, 0.0, 0.0], dtype=np.float64),
        rest_world=np.eye(4),
        rest_local=np.eye(4),
        pre_rotation_euler_deg=np.zeros(3),
    )
    art_bones["rig:LeftHandIndex1"] = RigBone(
        name="rig:LeftHandIndex1",
        parent="rig:LeftHand",
        children=[],
        pre_rotation=np.eye(3),
        lcl_translation_cm=np.array([8.0, 0.0, 0.0], dtype=np.float64),
        rest_world=np.eye(4),
        rest_local=np.eye(4),
        pre_rotation_euler_deg=np.zeros(3),
    )
    rig_art = HumanoidRig(
        bones=art_bones,
        order=list(art_bones.keys()),
        meters_per_unit=0.01,
        source=Path("mock_twist.fbx"),
    )
    mapping_art = RigMapping(
        skeleton_rig=skel_25,
        target_rig=rig_art,
        basis_change={},
        translation_scale=1.0,
        r_world_align=np.eye(3),
        max_basis_deviation_deg=0.0,
    )

    # Simulate intermittent hand detection dropout at frames 3 and 4
    l_fist = _make_synthetic_hand_landmarks("left", "fist")
    l_seq = [l_fist if t not in (3, 4) else None for t in range(n_frames)]
    hand_payload = {"left": l_seq, "right": [None] * n_frames}

    anim_no_hands = build_animation(
        motion, mapping_art, list(range(n_frames)), hand_data=None, enable_foot_ik=False, verbose=False
    )
    anim_with_hands = build_animation(
        motion, mapping_art, list(range(n_frames)), hand_data=hand_payload, enable_foot_ik=False, verbose=False
    )

    # 1. Elbow swing on rig:LeftForeArm must be 100% preserved when hands are present
    for t in range(n_frames):
        assert np.allclose(
            anim_with_hands.r_anim_rig[t]["rig:LeftForeArm"],
            anim_no_hands.r_anim_rig[t]["rig:LeftForeArm"],
            atol=1e-6,
        )

    # 2. ForeArmTwist must rotate strictly around the local longitudinal X-axis ([1, 0, 0])
    for t in range(n_frames):
        r_twist = anim_with_hands.r_anim_rig[t]["rig:LeftForeArmTwist"]
        assert np.allclose(r_twist @ np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_wrist_3dof_articulation_flexion_deviation_twist() -> None:
    """Verify solve_wrist_rotations calculates 3-DoF articulation (flexion, deviation, twist)."""
    # 1. 21-point flat landmark set
    lms_flat = _make_synthetic_hand_landmarks("left", "flat")
    rw_flat = solve_wrist_rotations(lms_flat, "Left")
    assert np.allclose(rw_flat @ rw_flat.T, np.eye(3), atol=1e-5)
    assert np.isclose(float(np.linalg.det(rw_flat)), 1.0, atol=1e-5)

    # 2. Pronated palm (twisted around longitudinal axis)
    lms_twisted = lms_flat.copy()
    # Rotate palm normal 20 degrees around X-axis
    r_rot = _rodrigues(np.array([1.0, 0.0, 0.0]), np.radians(20.0))
    for idx in range(1, 21):
        lms_twisted[idx] = r_rot @ lms_flat[idx]

    rw_twisted = solve_wrist_rotations(lms_twisted, "Left")
    q_tw = quat_wxyz(rw_twisted)
    # Quaternion must reflect non-identity rotation with longitudinal twist component
    assert abs(q_tw[1]) > 0.05, f"Expected non-zero pronation/supination twist, got {q_tw}"
    assert np.isclose(float(np.linalg.det(rw_twisted)), 1.0, atol=1e-5)

    # 3. 4-point pose landmarks fallback ([wrist, index, pinky, thumb])
    lms_4pt = lms_flat[[0, 5, 17, 2]]
    rw_4pt = solve_wrist_rotations(lms_4pt, "Left")
    assert np.allclose(rw_4pt @ rw_4pt.T, np.eye(3), atol=1e-5)
    assert np.isclose(float(np.linalg.det(rw_4pt)), 1.0, atol=1e-5)


def test_athletic_guard_hand_fallback_and_dropout_decay() -> None:
    """Verify get_athletic_guard_rotations produces > 15.0 deg curl on all phalanges and decays naturally."""
    guard_left = get_athletic_guard_rotations("Left")
    guard_right = get_athletic_guard_rotations("Right")

    # All 15 finger bones present for both hands
    assert len(guard_left) == 15
    assert len(guard_right) == 15

    for side_name, rots in (("Left", guard_left), ("Right", guard_right)):
        for b_name, r_mat in rots.items():
            # Check SO(3) orthonormality
            det = float(np.linalg.det(r_mat))
            assert np.isclose(det, 1.0, atol=1e-5), f"{b_name} non-orthonormal det={det}"
            assert np.max(np.abs(r_mat @ r_mat.T - np.eye(3))) < 1e-5

            # Check curl angle is strictly > 15.0 degrees
            q = quat_wxyz(r_mat)
            ang_deg = 2.0 * np.degrees(np.arccos(np.clip(abs(q[0]), 0.0, 1.0)))
            assert ang_deg > 15.0, (
                f"{b_name} guard curl angle {ang_deg:.1f} deg failed threshold (> 15.0 deg)"
            )

    # Verify dropout decay behaviour in build_animation across multi-frame sequence
    n_frames = 12
    motion = _make_base_motion(num_frames=n_frames)
    skel_25 = create_synthetic_humanoid_rig()
    art_bones = dict(skel_25.bones)
    for b in guard_left:
        art_bones[b] = RigBone(
            name=b,
            parent="rig:LeftHand",
            children=[],
            pre_rotation=np.eye(3),
            lcl_translation_cm=np.array([2.0, 0.0, 0.0], dtype=np.float64),
            rest_world=np.eye(4),
            rest_local=np.eye(4),
            pre_rotation_euler_deg=np.zeros(3),
        )
    rig_art = HumanoidRig(
        bones=art_bones,
        order=list(art_bones.keys()),
        meters_per_unit=0.01,
        source=Path("mock_guard.fbx"),
    )
    mapping_art = RigMapping(
        skeleton_rig=skel_25,
        target_rig=rig_art,
        basis_change={},
        translation_scale=1.0,
        r_world_align=np.eye(3),
        max_basis_deviation_deg=0.0,
    )

    # Frame 0..2 tracked fist, then frames 3..11 drop out (motion blur in flight)
    l_fist = _make_synthetic_hand_landmarks("left", "fist")
    l_seq = [l_fist if t < 3 else None for t in range(n_frames)]
    hand_payload = {"left": l_seq, "right": [None] * n_frames}

    anim = build_animation(
        motion, mapping_art, list(range(n_frames)), hand_data=hand_payload, enable_foot_ik=False, verbose=False
    )

    # Across all dropout frames (3..11), finger joints must never decay to 0 deg T-pose palm
    for t in range(3, n_frames):
        r_idx1 = anim.r_anim_rig[t]["rig:LeftHandIndex1"]
        q_idx1 = quat_wxyz(r_idx1)
        ang_deg = 2.0 * np.degrees(np.arccos(np.clip(abs(q_idx1[0]), 0.0, 1.0)))
        assert ang_deg > 15.0, (
            f"Frame {t} Index1 decayed to flat open palm: angle={ang_deg:.1f} deg (expected > 15.0 deg)"
        )


def test_floor_reach_ik_landing_scapular_depression() -> None:
    """Verify apply_floor_reach_ik activates during deep landing crouch and brings wrist within 25.0 cm."""
    n_frames = 6
    motion = _make_base_motion(num_frames=n_frames)
    skel = create_synthetic_humanoid_rig()
    fps = 30.0

    # Place pelvis and wrists in deep superhero landing crouch:
    # Hip height = 0.30 m, forward torso lean 20 deg, arm angled downward
    hips_list = [np.array([0.0, 30.0, 0.0], dtype=np.float64) for _ in range(n_frames)]
    r_anim_base = {name: np.eye(3) for name in skel.order}
    r_anim_base["rig:Spine"] = _rodrigues(np.array([1.0, 0.0, 0.0]), np.radians(20.0))
    r_anim_base["rig:LeftArm"] = _rodrigues(np.array([0.0, 0.0, -1.0]), np.radians(70.0))
    r_skeleton = [dict(r_anim_base) for _ in range(n_frames)]

    # Initial wrist height before reach IK
    wpos_pre = world_positions_m(skel, forward_kinematics(skel, r_skeleton[0], hips_translation_cm=hips_list[0]))
    y_pre_cm = wpos_pre["rig:LeftHand"][1] * 100.0

    # Apply reach IK
    r_reach = apply_floor_reach_ik(r_skeleton, hips_list, skel, fps=fps)

    # Post reach kinematics
    wpos_post = world_positions_m(skel, forward_kinematics(skel, r_reach[0], hips_translation_cm=hips_list[0]))
    y_post_cm = wpos_post["rig:LeftHand"][1] * 100.0

    assert y_post_cm <= 25.0, f"Expected wrist height <= 25.0 cm, got {y_post_cm:.1f} cm (initial was {y_pre_cm:.1f} cm)"
    assert y_post_cm >= 0.0, f"Floor penetration detected: {y_post_cm:.1f} cm"
    # Scapular depression applied on shoulder
    r_sh = r_reach[0]["rig:LeftShoulder"]
    assert not np.allclose(r_sh, np.eye(3)), "Expected non-identity scapular depression on LeftShoulder"


def test_floor_reach_ik_inactive_during_flight_and_standing() -> None:
    """Verify apply_floor_reach_ik remains inactive when wrist is high or moving with high planar velocity."""
    n_frames = 4
    motion = _make_base_motion(num_frames=n_frames)
    skel = create_synthetic_humanoid_rig()
    fps = 30.0

    # Standing posture: hip height = 1.02 m, wrist height = 0.88 m
    hips_list = [np.array([0.0, 102.0, 0.0], dtype=np.float64) for _ in range(n_frames)]
    r_skeleton = [{name: np.eye(3) for name in skel.order} for _ in range(n_frames)]

    r_reach = apply_floor_reach_ik(r_skeleton, hips_list, skel, fps=fps)

    # All rotations must be 100% identical to baseline (zero reach adjustment)
    for t in range(n_frames):
        for b in ("rig:LeftShoulder", "rig:LeftArm", "rig:LeftForeArm", "rig:RightShoulder", "rig:RightArm"):
            assert np.allclose(r_reach[t][b], r_skeleton[t][b], atol=1e-8), (
                f"apply_floor_reach_ik erroneously modified {b} in standing posture"
            )
