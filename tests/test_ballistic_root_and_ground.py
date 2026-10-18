"""Unit tests for Milestone 2: Ballistic Root Trajectory & Ground Baseline.

Verifies:
1. Graceful fallback on empty, short, or invalid screen landmarks.
2. Zero airborne lift during normal stance, walking, and grounded crouches.
3. Robust ground baseline interpolation across sustained aerial flight phases (> 40 frames).
4. Elimination of mid-air hip sink during flight phases.
5. Strict floor non-penetration constraint (y_min >= 0.0) across all frames.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from v2bm.human_motion import HumanMotion, BODY24_JOINTS
from v2bm.rig.animation import _compute_airborne_lift_m, build_animation
from v2bm.rig.kinematics import HumanoidRig, RigBone, forward_kinematics, world_positions_m
from v2bm.rig.rig_retarget import RigMapping


from tests.test_fk_skin import create_synthetic_humanoid_rig


def test_airborne_lift_graceful_fallback_on_empty_or_short_motion() -> None:
    """Verify that motions with None, empty, or short landmarks return zero lift safely."""
    n_frames = 10
    motion_short = HumanMotion(
        joints_3d=np.zeros((n_frames, 24, 3), dtype=np.float64),
        joint_names=list(BODY24_JOINTS),
        fps=24.0,
        timestamps=np.linspace(0, 1, n_frames),
        root_translation=np.zeros((n_frames, 3), dtype=np.float64),
        path=Path("mock.npz"),
        screen_landmarks=None,
    )
    lift = _compute_airborne_lift_m(motion_short, list(range(n_frames)))
    assert isinstance(lift, np.ndarray)
    assert lift.shape == (n_frames,)
    assert np.all(lift == 0.0)

    # Test return_intervals=True on short motion
    lift, intervals = _compute_airborne_lift_m(motion_short, list(range(n_frames)), return_intervals=True)
    assert isinstance(lift, np.ndarray)
    assert intervals == []


def test_airborne_lift_zero_during_grounded_standing_and_crouch() -> None:
    """Verify that grounded upright standing and deep floor crouches evaluate to zero lift."""
    n_frames = 60
    fps = 24.0
    sl = np.zeros((n_frames, 33, 3), dtype=np.float64)

    # Torso: neck at Y = 0.20, pelvis at Y = 0.50 (torso length = 0.30 in screen space)
    sl[:, 11, :2] = [0.45, 0.20]
    sl[:, 12, :2] = [0.55, 0.20]
    sl[:, 23, :2] = [0.47, 0.50]
    sl[:, 24, :2] = [0.53, 0.50]

    # Stance feet resting firmly on floor envelope Y = 0.95
    lower_idxs = [27, 28, 29, 30, 31, 32]
    for idx in lower_idxs:
        sl[:, idx, 1] = 0.95

    j3d = np.zeros((n_frames, 24, 3), dtype=np.float64)
    # Pelvis at origin, feet at -0.82 m below pelvis
    j3d[:, [7, 8, 10, 11], 1] = -0.82

    motion = HumanMotion(
        joints_3d=j3d,
        joint_names=list(BODY24_JOINTS),
        fps=fps,
        timestamps=np.linspace(0, n_frames / fps, n_frames),
        root_translation=np.zeros((n_frames, 3), dtype=np.float64),
        path=Path("mock.npz"),
        screen_landmarks=sl,
    )

    lift, intervals = _compute_airborne_lift_m(motion, list(range(n_frames)), return_intervals=True)
    assert np.all(lift == 0.0), f"Expected 0.0 lift during standing, got max: {lift.max():.3f} m"
    assert len(intervals) == 0, f"Expected 0 flight intervals during standing, got: {intervals}"


def test_airborne_lift_sustained_flight_interpolation() -> None:
    """Verify that a 50-frame aerial flight gap (> 40 frames) does not suffer sliding-window collapse."""
    n_frames = 100
    fps = 24.0
    sl = np.zeros((n_frames, 33, 3), dtype=np.float64)
    j3d = np.zeros((n_frames, 24, 3), dtype=np.float64)

    # Common torso: length 0.30 screen space
    sl[:, 11, :2] = [0.45, 0.20]
    sl[:, 12, :2] = [0.55, 0.20]
    sl[:, 23, :2] = [0.47, 0.50]
    sl[:, 24, :2] = [0.53, 0.50]

    # Pre-jump stance (F00 to F20): ground Y = 1.00
    # Flight phase (F25 to F75): feet rise to Y = 0.70 (30 cm above ground)
    # Post-jump stance (F80 to F99): ground Y = 1.00
    for idx in [27, 28, 29, 30, 31, 32]:
        sl[:22, idx, 1] = 1.00
        sl[25:75, idx, 1] = 0.70
        sl[78:, idx, 1] = 1.00
    # Smooth transitions
    sl[22:25, :, 1] = np.linspace(1.00, 0.70, 3)[:, None]
    sl[75:78, :, 1] = np.linspace(0.70, 1.00, 3)[:, None]

    # Leg extension: -0.80 m during stance, tucked to -0.35 m during flight
    j3d[:22, [7, 8, 10, 11], 1] = -0.80
    j3d[25:75, [7, 8, 10, 11], 1] = -0.35
    j3d[78:, [7, 8, 10, 11], 1] = -0.80

    motion = HumanMotion(
        joints_3d=j3d,
        joint_names=list(BODY24_JOINTS),
        fps=fps,
        timestamps=np.linspace(0, n_frames / fps, n_frames),
        root_translation=np.zeros((n_frames, 3), dtype=np.float64),
        path=Path("mock.npz"),
        screen_landmarks=sl,
    )

    lift, intervals = _compute_airborne_lift_m(motion, list(range(n_frames)), return_intervals=True)
    assert len(intervals) >= 1, f"Expected flight interval to be detected, got: {intervals}"
    t_start, t_end = intervals[0]
    assert t_start <= 26 and t_end >= 73, f"Flight interval [{t_start}, {t_end}] did not span flight phase"

    # Sustained flight in middle of jump (F45 to F55) must maintain positive lift (no collapse)
    mid_jump_lift = lift[45:55]
    assert np.all(mid_jump_lift > 0.30), (
        f"Mid-jump lift collapsed! Expected > 0.30 m, got values: {mid_jump_lift}"
    )


def test_build_animation_strict_floor_non_penetration() -> None:
    """Verify build_animation strictly enforces floor clearance y_min >= 0 across all frames."""
    n_frames = 30
    fps = 24.0
    rig = create_synthetic_humanoid_rig()
    mapping = RigMapping(
        skeleton_rig=rig,
        target_rig=rig,
        basis_change={},
        translation_scale=1.0,
        r_world_align=np.eye(3),
        max_basis_deviation_deg=0.0,
    )

    from tests.test_upper_body_and_hands import _make_base_motion
    motion = _make_base_motion(n_frames)
    # Deep knee bend where lowest foot dips significantly
    motion.joints_3d[:, [4, 5], 1] -= 0.20
    motion.joints_3d[:, [7, 8], 1] -= 0.35
    motion.joints_3d[:, [10, 11], 1] -= 0.40

    anim = build_animation(motion, mapping, list(range(n_frames)), verbose=False, enable_foot_ik=False)
    assert len(anim.hips_cm_rig) == n_frames

    for f in range(n_frames):
        pos = mapping.humanoid_rig_positions_m(anim.r_anim_rig[f], anim.hips_cm_rig[f])
        min_y = min(float(p[1]) for p in pos.values())
        assert min_y >= -1e-5, f"Floor penetration detected at frame {f}: min_y = {min_y:.4f} m"


def test_flying_kick_biomechanics_telemetry() -> None:
    """Verify quantitative biomechanical criteria on the Flying kick dataset if available."""
    run_dir = Path(r"G:\My Drive\02_Ventures & Projects\WebProjects\VideoIntoBlenderModel\Flying kick")
    motion_path = run_dir / "skeleton_motion.npz"
    rig_path = Path(r"assets\rigs\renderpeople_sophia.fbx")
    if not motion_path.exists() or not rig_path.exists():
        pytest.skip("Flying kick dataset or Sophia rig not available in environment.")

    from v2bm.human_motion import load_human_motion
    from v2bm.rig.tpose_calibration import load_calibration
    from v2bm.rig.rig_retarget import build_mapping

    motion = load_human_motion(motion_path)
    cal = load_calibration(rig_path)
    mapping = build_mapping(cal, verbose=False)
    frames = list(range(motion.n_frames))

    anim = build_animation(motion, mapping, frames, verbose=False, enable_foot_ik=True)
    hips_y = np.array([float(h[1]) for h in anim.hips_cm_rig])

    # 1. Apex height at F117 >= 70.0 cm
    assert hips_y[117] >= 70.0, f"Hips Y at apex F117 ({hips_y[117]:.1f} cm) below 70.0 cm threshold"

    # 2. Mid-air hip drop between tuck (F085) and kick extension (F117-F125) <= 8.0 cm
    drop_to_117 = hips_y[85] - hips_y[117]
    drop_to_125 = hips_y[85] - hips_y[125]
    assert drop_to_117 <= 8.0, f"Hip drop F085 -> F117 ({drop_to_117:.1f} cm) exceeded 8.0 cm"
    assert drop_to_125 <= 8.0, f"Hip drop F085 -> F125 ({drop_to_125:.1f} cm) exceeded 8.0 cm"

    # 3. Floor clearance across all frames strictly non-penetrating (>= 0.0 cm)
    min_floor_clearance = min(
        min(float(p[1]) for p in mapping.humanoid_rig_positions_m(anim.r_anim_rig[f], anim.hips_cm_rig[f]).values())
        for f in range(motion.n_frames)
    )
    assert min_floor_clearance >= -1e-5, f"Floor penetration detected: {min_floor_clearance*100:.2f} cm"
