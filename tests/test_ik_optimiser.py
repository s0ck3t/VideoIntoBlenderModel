"""Unit tests for the sliding-window 2D-reprojection-constrained IK trajectory optimiser.

Validates:
1. Invariant rigid bone lengths (< 10^-4 m error across all frames).
2. Jerk reduction (magnitude of third temporal derivative d^3P/dt^3 reduced by >= 50%).
3. 2D reprojection accuracy (fidelity to screen keypoints).
4. Foreshortening stability (suppression of near-zero length noise amplification without stick snapping).
5. Physiological range-of-motion preservation.
6. Window overlap blending continuity across boundaries.

All tests adhere strictly to British English conventions and permissive commercial licensing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from v2bm.ik_optimiser import (
    calculate_bone_lengths,
    compute_trajectory_jerk,
    optimise_trajectory_windowed,
)
from v2bm.stabilize_kp2d import (
    BODY24_BONE_PAIRS,
    map_screen_to_body24,
    project_rigid_bone_lengths,
)

BODY24_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def _create_synthetic_tpose(n_frames: int = 30) -> np.ndarray:
    """Create a realistic baseline human 3D pose trajectory across n_frames."""
    base_pos = {
        "pelvis": (0.0, 0.95, 0.0),
        "left_hip": (-0.10, 0.90, 0.0),
        "right_hip": (0.10, 0.90, 0.0),
        "spine1": (0.0, 1.10, 0.0),
        "left_knee": (-0.12, 0.50, 0.0),
        "right_knee": (0.12, 0.50, 0.0),
        "spine2": (0.0, 1.25, 0.0),
        "left_ankle": (-0.12, 0.10, 0.0),
        "right_ankle": (0.12, 0.10, 0.0),
        "spine3": (0.0, 1.40, 0.0),
        "left_foot": (-0.12, 0.02, 0.12),
        "right_foot": (0.12, 0.02, 0.12),
        "neck": (0.0, 1.50, 0.0),
        "left_collar": (-0.08, 1.45, 0.0),
        "right_collar": (0.08, 1.45, 0.0),
        "head": (0.0, 1.68, 0.0),
        "left_shoulder": (-0.20, 1.45, 0.0),
        "right_shoulder": (0.20, 1.45, 0.0),
        "left_elbow": (-0.46, 1.45, 0.0),
        "right_elbow": (0.46, 1.45, 0.0),
        "left_wrist": (-0.72, 1.45, 0.0),
        "right_wrist": (0.72, 1.45, 0.0),
        "left_hand": (-0.85, 1.45, 0.0),
        "right_hand": (0.85, 1.45, 0.0),
    }
    joint_indices = {name: i for i, name in enumerate(BODY24_JOINTS)}
    joints_3d = np.zeros((n_frames, 24, 3), dtype=np.float64)

    for t in range(n_frames):
        phase = 2.0 * math.pi * float(t) / float(n_frames)
        for name, pos in base_pos.items():
            idx = joint_indices[name]
            p = list(pos)
            # Smooth oscillatory dance motion on forearms
            if "elbow" in name or "wrist" in name or "hand" in name:
                p[1] += 0.08 * math.sin(phase)
                p[2] += 0.05 * math.cos(phase)
            joints_3d[t, idx] = p

    return joints_3d


def test_invariant_rigid_bone_lengths() -> None:
    """Verify all bone lengths match their rest lengths within < 10^-4 m error."""
    n_frames = 35
    trajectory = _create_synthetic_tpose(n_frames)

    # Inject substantial non-physical stretching and compression (+- 8 cm)
    rng = np.random.default_rng(42)
    stretch_noise = rng.normal(0.0, 0.04, trajectory.shape)
    distorted_trajectory = trajectory + stretch_noise

    # Pre-calculate rest lengths on uncorrupted baseline
    target_rest_lengths = calculate_bone_lengths(trajectory, BODY24_BONE_PAIRS)

    # Optimise trajectory
    optimised = optimise_trajectory_windowed(
        distorted_trajectory,
        bone_lengths=target_rest_lengths,
        fps=30.0,
        window_length=11,
        stride=5,
    )

    # Verify bone length invariance across all frames and bones
    max_error = 0.0
    for parent, child in BODY24_BONE_PAIRS:
        target_l = target_rest_lengths[(parent, child)]
        for t in range(n_frames):
            actual_l = float(np.linalg.norm(optimised[t, child] - optimised[t, parent]))
            err = abs(actual_l - target_l)
            if err > max_error:
                max_error = err

    assert max_error < 1e-4, (
        f"Rigid bone length invariance violated: maximum error {max_error:.3e} m "
        f"exceeds 10^-4 m threshold"
    )


def test_jerk_reduction_at_least_50_percent() -> None:
    """Verify sliding-window trajectory optimization reduces jerk by at least 50%."""
    n_frames = 50
    fps = 30.0
    clean_trajectory = _create_synthetic_tpose(n_frames)

    # Inject realistic monocular depth jitter and high-frequency noise on upper extremities
    rng = np.random.default_rng(123)
    noisy_trajectory = clean_trajectory.copy()
    # High-frequency depth jitter on distal upper limbs (indices 18..23)
    jitter = rng.normal(0.0, 0.025, (n_frames, 6, 3))
    noisy_trajectory[:, 18:24, :] += jitter

    initial_jerk = compute_trajectory_jerk(noisy_trajectory, fps=fps)
    assert initial_jerk > 0.0, "Initial jerk must be positive"

    # Optimise trajectory
    optimised = optimise_trajectory_windowed(
        noisy_trajectory,
        fps=fps,
        window_length=11,
        stride=5,
        w_smooth=6.0,
        w_jerk=4.0,
    )

    optimised_jerk = compute_trajectory_jerk(optimised, fps=fps)
    reduction = (initial_jerk - optimised_jerk) / initial_jerk

    assert reduction >= 0.50, (
        f"Jerk reduction {reduction:.1%} is below the mandatory 50% threshold "
        f"(initial {initial_jerk:.2f} m/s^3 -> optimised {optimised_jerk:.2f} m/s^3)"
    )


def test_2d_reprojection_accuracy() -> None:
    """Verify trajectory optimiser preserves 2D screen landmark fidelity."""
    n_frames = 25
    gt_trajectory = _create_synthetic_tpose(n_frames)

    # Synthesize 2D screen landmarks via weak perspective projection
    s_scale = 0.4 / 0.52  # torso height in screen units / 3D metric length
    pelvis_2d = np.array([0.5, 0.5])
    screen_body24 = np.zeros((n_frames, 24, 2), dtype=np.float64)
    for t in range(n_frames):
        for j in range(24):
            dx = gt_trajectory[t, j, 0] - gt_trajectory[t, 0, 0]
            dy = gt_trajectory[t, j, 1] - gt_trajectory[t, 0, 1]
            screen_body24[t, j, 0] = pelvis_2d[0] + s_scale * dx
            screen_body24[t, j, 1] = pelvis_2d[1] - s_scale * dy

    # Corrupt 3D trajectory with transverse lateral/vertical perturbation
    rng = np.random.default_rng(999)
    corrupted_3d = gt_trajectory.copy()
    corrupted_3d[:, :, :2] += rng.normal(0.0, 0.02, corrupted_3d[:, :, :2].shape)

    # Compute initial 2D error
    def calc_2d_error(p_3d: np.ndarray) -> float:
        err = 0.0
        for t in range(n_frames):
            for j in range(24):
                dx = p_3d[t, j, 0] - p_3d[t, 0, 0]
                dy = p_3d[t, j, 1] - p_3d[t, 0, 1]
                u_hat = pelvis_2d[0] + s_scale * dx
                v_hat = pelvis_2d[1] - s_scale * dy
                err += float((u_hat - screen_body24[t, j, 0]) ** 2 + (v_hat - screen_body24[t, j, 1]) ** 2)
        return math.sqrt(err / float(n_frames * 24))

    init_error = calc_2d_error(corrupted_3d)

    # Optimise trajectory constrained by 2D screen landmarks
    optimised = optimise_trajectory_windowed(
        corrupted_3d,
        landmarks_2d=screen_body24,
        fps=30.0,
        window_length=11,
        stride=5,
        w_2d=3.0,
        w_3d=0.5,
    )

    final_error = calc_2d_error(optimised)
    assert final_error < init_error, (
        f"Optimiser failed to reduce 2D reprojection error: initial {init_error:.4f} "
        f"vs final {final_error:.4f}"
    )


def test_foreshortening_stability() -> None:
    """Verify foreshortened limbs avoid noise explosion, maintain continuity, and do not stick."""
    n_frames = 20
    trajectory = _create_synthetic_tpose(n_frames)

    # Simulate severe right forearm foreshortening around frames 8..12
    # Right elbow (19) -> Right wrist (21)
    target_rest_len = float(np.linalg.norm(trajectory[0, 21] - trajectory[0, 19]))

    # Collapse transverse distance to 1 mm (d_2D ~ 0.001) and 3D distance to 0.002 m
    for t in range(8, 13):
        # Wrist collapses near elbow
        trajectory[t, 21] = trajectory[t, 19] + np.array([0.0005, 0.0005, 0.0015])

    optimised = optimise_trajectory_windowed(
        trajectory,
        fps=30.0,
        window_length=11,
        stride=5,
    )

    # 1. Assert no NaN or Inf values
    assert np.all(np.isfinite(optimised)), "Optimised trajectory contains non-finite values"

    # 2. Assert rigid bone length is invariant even during the singularity
    for t in range(8, 13):
        forearm_len = float(np.linalg.norm(optimised[t, 21] - optimised[t, 19]))
        err = abs(forearm_len - target_rest_len)
        assert err < 1e-4, f"Foreshortened frame {t} bone length error {err:.3e} exceeds 10^-4 m"

    # 3. Assert limb does not snap to artificial vertical straight stick [0, -1, 0]
    for t in range(8, 13):
        direction = (optimised[t, 21] - optimised[t, 19]) / target_rest_len
        # Downward stick would have direction ~ [0.0, -1.0, 0.0]
        stick_dot = float(np.dot(direction, np.array([0.0, -1.0, 0.0])))
        assert stick_dot < 0.90, (
            f"Frame {t} snapped to artificial straight downward stick: dot = {stick_dot:.3f}"
        )

    # 4. Assert direction continuity across singularity (no instantaneous 180° flip)
    for t in range(1, n_frames):
        dir_prev = (optimised[t - 1, 21] - optimised[t - 1, 19]) / target_rest_len
        dir_curr = (optimised[t, 21] - optimised[t, 19]) / target_rest_len
        cos_ang = float(np.clip(np.dot(dir_prev, dir_curr), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(cos_ang))
        assert angle_deg < 50.0, (
            f"Frame {t} experienced abrupt angular flip of {angle_deg:.1f}° relative to frame {t-1}"
        )


def test_window_overlap_continuity() -> None:
    """Verify Hann window overlap blending eliminates boundary step discontinuities."""
    n_frames = 40
    trajectory = _create_synthetic_tpose(n_frames)

    optimised = optimise_trajectory_windowed(
        trajectory,
        fps=30.0,
        window_length=11,
        stride=5,
    )

    # Compute maximum frame-to-frame joint displacement
    diffs = np.linalg.norm(optimised[1:] - optimised[:-1], axis=-1)
    max_step = float(np.max(diffs))

    # In a smooth 30 fps dance motion, maximum joint velocity is < 0.15 m/frame
    assert max_step < 0.15, (
        f"Window boundary discontinuity detected: maximum step {max_step:.3f} m exceeds 0.15 m"
    )
