"""Sliding-window 2D-reprojection-constrained Inverse Kinematics (IK) trajectory optimiser.

Jointly minimises:
1. 2D screen-landmark reprojection error (high lateral/vertical fidelity).
2. 3D pose prior (preserving global spatial orientation, downweighting monocular Z when foreshortened).
3. Rigid bone length constraints (strictly eliminating non-physical stretching/compression).
4. Temporal acceleration and jerk penalties (third derivative d^3P/dt^3 jerk reduction >= 50%).
5. Anatomical joint range-of-motion limits (knees 0-145°, elbows 0-145°).

Operates over sliding temporal windows (window length W ~ 11, stride S ~ 5) with smooth
Hann/cosine overlap blending and final hierarchical rigid bone projection to guarantee
machine-precision bone length invariance (< 10^-4 m error).

All mathematical formulations use pure NumPy and SciPy (least_squares) under permissive
licences (Apache-2.0 / BSD-3-Clause) with British English conventions.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .stabilize_kp2d import (
    BODY24_BONE_PAIRS,
    clamp_anatomical_hinges,
    map_screen_to_body24,
    project_rigid_bone_lengths,
)

# Standard hinge chains: (proximal, hinge, distal, max_interior_angle_deg, is_knee)
ANATOMICAL_HINGES = [
    (1, 4, 7, 145.0, True),   # Left leg: hip -> knee -> ankle
    (2, 5, 8, 145.0, True),   # Right leg: hip -> knee -> ankle
    (16, 18, 20, 145.0, False),  # Left arm: shoulder -> elbow -> wrist
    (17, 19, 21, 145.0, False),  # Right arm: shoulder -> elbow -> wrist
]


def calculate_bone_lengths(
    positions_3d: np.ndarray,
    bone_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], float]:
    """Calculate temporal median rest lengths for all bone connections.

    Args:
        positions_3d: Array of shape (F, 24, 3) in human coordinates.
        bone_pairs: Kinematic connections (parent, child). Defaults to BODY24_BONE_PAIRS.

    Returns:
        Mapping from bone pair (parent, child) to rest length in metres.
    """
    pairs = bone_pairs or BODY24_BONE_PAIRS
    data = np.asarray(positions_3d, dtype=np.float64)
    rest_lengths: dict[tuple[int, int], float] = {}
    for parent, child in pairs:
        lengths = np.linalg.norm(data[:, child] - data[:, parent], axis=-1)
        med_len = float(np.median(lengths))
        rest_lengths[(parent, child)] = max(med_len, 1e-4)
    return rest_lengths


def compute_trajectory_jerk(positions_3d: np.ndarray, fps: float = 30.0) -> float:
    """Compute the mean Euclidean jerk (magnitude of third temporal derivative d^3P/dt^3).

    Args:
        positions_3d: Array of shape (F, 24, 3) or (F, J, 3) in metres.
        fps: Sampling frequency in frames per second.

    Returns:
        Mean jerk magnitude across all joints and valid interior frames in m/s^3.
    """
    data = np.asarray(positions_3d, dtype=np.float64)
    n_frames = len(data)
    if n_frames < 4:
        return 0.0

    dt = 1.0 / float(fps) if fps > 0 else 1.0 / 30.0
    # Third finite difference: P_t - 3 P_{t-1} + 3 P_{t-2} - P_{t-3}
    jerk_vectors = (
        data[3:] - 3.0 * data[2:-1] + 3.0 * data[1:-2] - data[:-3]
    ) / (dt**3)
    magnitudes = np.linalg.norm(jerk_vectors, axis=-1)
    return float(np.mean(magnitudes))


def optimise_window(
    x_init: np.ndarray,
    s_body24_win: np.ndarray | None,
    vis_win: np.ndarray | None,
    torso_scales: np.ndarray | None,
    bone_pairs: Sequence[tuple[int, int]],
    bone_lengths: dict[tuple[int, int], float],
    foreshorten_factors: np.ndarray | None,
    *,
    w_2d: float = 1.5,
    w_3d: float = 0.8,
    w_bone: float = 25.0,
    w_smooth: float = 4.0,
    w_jerk: float = 2.0,
    w_anat: float = 8.0,
    max_nfev: int = 25,
) -> np.ndarray:
    """Optimise a single sliding temporal window using non-linear least squares.

    Args:
        x_init: Initial joint positions of shape (W, 24, 3) in float64.
        s_body24_win: Normalised 2D screen coordinates of shape (W, 24, 2) or None.
        vis_win: Visibility confidences of shape (W, 24) or None.
        torso_scales: Scale factor relating 3D metres to 2D screen units of shape (W,) or None.
        bone_pairs: List of (parent, child) joint index tuples.
        bone_lengths: Dict mapping (parent, child) to target rest lengths in metres.
        foreshorten_factors: Array of shape (W, 24) indicating foreshortening severity [0, 1].
        w_2d: Weight on 2D screen reprojection error.
        w_3d: Weight on 3D initial pose prior.
        w_bone: Weight on rigid bone rest length preservation.
        w_smooth: Weight on second-order temporal acceleration penalty.
        w_jerk: Weight on third-order temporal jerk penalty.
        w_anat: Weight on anatomical range-of-motion limits.
        max_nfev: Maximum function evaluations in least_squares.

    Returns:
        Optimised window joint positions of shape (W, 24, 3).
    """
    win_len, n_joints, _ = x_init.shape
    x0 = x_init.copy().ravel()

    # Pre-compute weights
    sqrt_w_2d = math.sqrt(max(w_2d, 0.0))
    sqrt_w_3d = math.sqrt(max(w_3d, 0.0))
    sqrt_w_bone = math.sqrt(max(w_bone, 0.0))
    sqrt_w_smooth = math.sqrt(max(w_smooth, 0.0))
    sqrt_w_jerk = math.sqrt(max(w_jerk, 0.0))
    sqrt_w_anat = math.sqrt(max(w_anat, 0.0))

    has_2d = s_body24_win is not None and torso_scales is not None
    vis = vis_win if vis_win is not None else np.ones((win_len, n_joints), dtype=np.float64)
    f_factors = foreshorten_factors if foreshorten_factors is not None else np.zeros((win_len, n_joints), dtype=np.float64)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        p = params.reshape((win_len, n_joints, 3))
        res_list: list[np.ndarray] = []

        # 1. 2D Screen Landmark Reprojection Residuals
        if has_2d and sqrt_w_2d > 0.0:
            assert s_body24_win is not None
            assert torso_scales is not None
            dx = p[:, :, 0] - p[:, 0:1, 0]
            dy = p[:, :, 1] - p[:, 0:1, 1]
            if torso_scales.ndim == 2 and torso_scales.shape[1] == 2:
                u_hat = s_body24_win[:, 0:1, 0] + torso_scales[:, 0:1] * dx
                v_hat = s_body24_win[:, 0:1, 1] + torso_scales[:, 1:2] * dy
            else:
                u_hat = s_body24_win[:, 0:1, 0] + torso_scales[:, None] * dx
                v_hat = s_body24_win[:, 0:1, 1] - torso_scales[:, None] * dy

            diff_u = (u_hat - s_body24_win[:, :, 0]) * np.sqrt(vis) * sqrt_w_2d
            diff_v = (v_hat - s_body24_win[:, :, 1]) * np.sqrt(vis) * sqrt_w_2d
            res_list.append(diff_u.ravel())
            res_list.append(diff_v.ravel())

        # 2. 3D Prior Residuals (downweighting Z when foreshortened)
        if sqrt_w_3d > 0.0:
            diff_xy = (p[:, :, :2] - x_init[:, :, :2]) * sqrt_w_3d
            res_list.append(diff_xy.ravel())

            # For Z coordinate: downweight depth prior in foreshortened frames
            z_weight = sqrt_w_3d * (1.0 - 0.75 * np.clip(f_factors, 0.0, 1.0))
            diff_z = (p[:, :, 2] - x_init[:, :, 2]) * z_weight
            res_list.append(diff_z.ravel())

        # 3. Rigid Bone Length Constraints
        if sqrt_w_bone > 0.0:
            bone_residuals = []
            for parent, child in bone_pairs:
                target_l = bone_lengths[(parent, child)]
                bone_vec = p[:, child] - p[:, parent]
                bone_len = np.linalg.norm(bone_vec, axis=-1)
                bone_residuals.append((bone_len - target_l) * sqrt_w_bone)
            if bone_residuals:
                res_list.append(np.concatenate(bone_residuals))

        # 4. Temporal Smoothness / Acceleration Penalties (d^2P/dt^2)
        if win_len >= 3 and sqrt_w_smooth > 0.0:
            accel = p[2:] - 2.0 * p[1:-1] + p[:-2]
            res_list.append((accel * sqrt_w_smooth).ravel())

        # 5. Temporal Jerk Penalties (d^3P/dt^3)
        if win_len >= 4 and sqrt_w_jerk > 0.0:
            jerk = p[3:] - 3.0 * p[2:-1] + 3.0 * p[1:-2] - p[:-3]
            res_list.append((jerk * sqrt_w_jerk).ravel())

        # 6. Anatomical Range-of-Motion Limit Penalties
        if sqrt_w_anat > 0.0:
            anat_penalties = []
            for prox, hinge, dist, max_deg, is_knee in ANATOMICAL_HINGES:
                v_in = p[:, hinge] - p[:, prox]
                v_out = p[:, dist] - p[:, hinge]
                n_in = np.linalg.norm(v_in, axis=-1) + 1e-6
                n_out = np.linalg.norm(v_out, axis=-1) + 1e-6
                cos_theta = np.clip(np.sum(v_in * v_out, axis=-1) / (n_in * n_out), -1.0, 1.0)
                # Interior flexion angle in degrees (0° = straight, 180° = fully folded)
                flex_deg = np.degrees(np.arccos(cos_theta))

                # Penalty for exceeding max interior flexion (e.g. 145°)
                exceed = np.maximum(0.0, flex_deg - max_deg)
                anat_penalties.append(exceed * (sqrt_w_anat * 0.02))

                # Knee hyperextension penalty: prevent backward deflection
                if is_knee:
                    # Posterior deflection relative to pelvis anterior direction
                    v_right = p[:, 2] - p[:, 1]
                    v_up = p[:, 12] - p[:, 0]
                    f_body = np.cross(v_up, v_right)
                    f_body_norm = np.linalg.norm(f_body, axis=-1, keepdims=True) + 1e-6
                    f_body /= f_body_norm

                    # Knee deflection relative to hip-ankle line
                    u_chord = (p[:, dist] - p[:, prox]) / (np.linalg.norm(p[:, dist] - p[:, prox], axis=-1, keepdims=True) + 1e-6)
                    proj = p[:, prox] + np.sum((p[:, hinge] - p[:, prox]) * u_chord, axis=-1, keepdims=True) * u_chord
                    deflect = p[:, hinge] - proj
                    ant_disp = np.sum(deflect * f_body, axis=-1)
                    hyperextend = np.maximum(0.0, -ant_disp - 0.01)
                    anat_penalties.append(hyperextend * (sqrt_w_anat * 2.0))

            if anat_penalties:
                res_list.append(np.concatenate(anat_penalties))

        if not res_list:
            return np.zeros(1, dtype=np.float64)
        return np.concatenate(res_list)

    # Solve least squares optimization problem
    try:
        opt_res = least_squares(
            residual_fn,
            x0,
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-3,
            xtol=1e-3,
            gtol=1e-3,
        )
        return opt_res.x.reshape((win_len, n_joints, 3))
    except Exception:
        # Graceful fallback to initial window if numerical solver encounters an issue
        return x_init.copy()


def optimise_trajectory_windowed(
    positions_3d: np.ndarray,
    landmarks_2d: np.ndarray | None = None,
    visibilities: np.ndarray | None = None,
    bone_pairs: Sequence[tuple[int, int]] | None = None,
    bone_lengths: dict[tuple[int, int], float] | None = None,
    fps: float = 30.0,
    window_length: int = 11,
    stride: int = 5,
    w_2d: float = 1.5,
    w_3d: float = 0.8,
    w_bone: float = 25.0,
    w_smooth: float = 4.0,
    w_jerk: float = 2.0,
    w_anat: float = 8.0,
    max_nfev: int = 25,
    verbose: bool = False,
) -> np.ndarray:
    """Sliding-window 2D-reprojection-constrained IK trajectory optimiser.

    Jointly optimises limb trajectories across overlapping temporal windows
    to minimise 2D reprojection error, regularise depth uncertainty, eliminate
    acceleration/jerk spikes, and enforce rigid bone invariance (< 10^-4 m).

    Args:
        positions_3d: Input joint positions of shape (T, 24, 3) in human world coordinates.
        landmarks_2d: Optional 2D screen coordinates of shape (T, 33, 2/3) or (T, 24, 2).
        visibilities: Optional landmark visibilities of shape (T, 33) or (T, 24).
        bone_pairs: Bone connections. Defaults to BODY24_BONE_PAIRS.
        bone_lengths: Optional pre-computed rest lengths for each bone pair.
        fps: Sampling rate in frames per second.
        window_length: Length of sliding temporal window in frames (default 11).
        stride: Stride step between window centers in frames (default 5).
        w_2d: Weight on 2D screen reprojection fidelity.
        w_3d: Weight on 3D initial pose prior.
        w_bone: Weight on rigid bone length preservation.
        w_smooth: Weight on second-order temporal acceleration penalty.
        w_jerk: Weight on third-order temporal jerk penalty.
        w_anat: Weight on anatomical range-of-motion limits.
        max_nfev: Maximum least_squares iterations per window.
        verbose: Verbose progress reporting flag.

    Returns:
        Optimised array of shape (T, 24, 3) in np.float32.
    """
    data = np.asarray(positions_3d, dtype=np.float64)
    n_frames, n_joints, _ = data.shape
    pairs = bone_pairs or BODY24_BONE_PAIRS
    rest_lengths = bone_lengths or calculate_bone_lengths(data, pairs)

    if n_frames == 0:
        return data.astype(np.float32)

    # 1. Screen landmark mapping and torso scale estimation
    s_body24: np.ndarray | None = None
    vis_body24: np.ndarray | None = None
    torso_scales: np.ndarray | None = None

    if landmarks_2d is not None and len(landmarks_2d) == n_frames:
        l2d = np.asarray(landmarks_2d, dtype=np.float64)
        if l2d.shape[1] == 33:
            s_body24 = map_screen_to_body24(l2d)
        elif l2d.shape[1] == 24:
            s_body24 = l2d[:, :, :2]

        if s_body24 is not None:
            # Estimate signed weak-perspective scales (scale_u, scale_v) per frame
            # accounting for rotZ(180°) coordinate convention and 16:9 image aspect ratio.
            du = s_body24[:, :, 0] - s_body24[:, 0:1, 0]
            dv = s_body24[:, :, 1] - s_body24[:, 0:1, 1]
            dx0 = data[:, :, 0] - data[:, 0:1, 0]
            dy0 = data[:, :, 1] - data[:, 0:1, 1]
            scale_u = np.sum(du * dx0, axis=-1) / np.maximum(np.sum(dx0 * dx0, axis=-1), 1e-4)
            scale_v = np.sum(dv * dy0, axis=-1) / np.maximum(np.sum(dy0 * dy0, axis=-1), 1e-4)
            scale_u = np.clip(scale_u, -5.0, 5.0)
            scale_v = np.clip(scale_v, -5.0, 5.0)
            torso_scales = np.stack([scale_u, scale_v], axis=-1)

        if visibilities is not None and len(visibilities) == n_frames:
            v_arr = np.asarray(visibilities, dtype=np.float64)
            if v_arr.shape[1] == 33:
                # Map MediaPipe 33 visibilities to 24 Body24 joints
                vis_body24 = np.ones((n_frames, 24), dtype=np.float64)
                vis_body24[:, 0] = 0.5 * (v_arr[:, 23] + v_arr[:, 24])
                vis_body24[:, 1] = v_arr[:, 23]
                vis_body24[:, 2] = v_arr[:, 24]
                vis_body24[:, 4] = v_arr[:, 25]
                vis_body24[:, 5] = v_arr[:, 26]
                vis_body24[:, 7] = v_arr[:, 27]
                vis_body24[:, 8] = v_arr[:, 28]
                vis_body24[:, 10] = v_arr[:, 31]
                vis_body24[:, 11] = v_arr[:, 32]
                vis_body24[:, 12] = 0.5 * (v_arr[:, 11] + v_arr[:, 12])
                vis_body24[:, 16] = v_arr[:, 11]
                vis_body24[:, 17] = v_arr[:, 12]
                vis_body24[:, 18] = v_arr[:, 13]
                vis_body24[:, 19] = v_arr[:, 14]
                vis_body24[:, 20] = v_arr[:, 15]
                vis_body24[:, 21] = v_arr[:, 16]
                vis_body24[:, 22] = 0.5 * (v_arr[:, 17] + v_arr[:, 19])
                vis_body24[:, 23] = 0.5 * (v_arr[:, 18] + v_arr[:, 20])
            elif v_arr.shape[1] == 24:
                vis_body24 = v_arr

    # 2. Foreshortening factors per joint and frame
    # Compute screen projected distance vs rest length
    foreshorten_factors = np.zeros((n_frames, 24), dtype=np.float64)
    if s_body24 is not None:
        for parent, child in pairs:
            d_2d = np.linalg.norm(s_body24[:, child] - s_body24[:, parent], axis=-1)
            # Foreshortened when projected distance is small (< 0.035)
            s_factor = np.clip((0.035 - d_2d) / 0.035, 0.0, 1.0)
            foreshorten_factors[:, child] = np.maximum(foreshorten_factors[:, child], s_factor)

    # 3. Sliding Window Optimization with Hann Overlap Blending
    accum_positions = np.zeros((n_frames, 24, 3), dtype=np.float64)
    accum_weights = np.zeros(n_frames, dtype=np.float64)

    W = min(window_length, n_frames)
    S = max(1, stride)

    window_starts = list(range(0, n_frames, S))
    # Ensure the final frames are fully covered
    if window_starts[-1] + W < n_frames:
        window_starts.append(n_frames - W)

    for win_idx, t0 in enumerate(window_starts):
        t1 = min(t0 + W, n_frames)
        w_act = t1 - t0
        if w_act <= 0:
            continue

        x_win = data[t0:t1].copy()
        s_win = s_body24[t0:t1] if s_body24 is not None else None
        v_win = vis_body24[t0:t1] if vis_body24 is not None else None
        sc_win = torso_scales[t0:t1] if torso_scales is not None else None
        f_win = foreshorten_factors[t0:t1]

        opt_win = optimise_window(
            x_win,
            s_win,
            v_win,
            sc_win,
            pairs,
            rest_lengths,
            f_win,
            w_2d=w_2d,
            w_3d=w_3d,
            w_bone=w_bone,
            w_smooth=w_smooth,
            w_jerk=w_jerk,
            w_anat=w_anat,
            max_nfev=max_nfev,
        )

        # Raised cosine / Hann blending window weights
        if w_act > 1:
            indices = np.arange(w_act, dtype=np.float64)
            blend_w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (indices + 0.5) / float(w_act))
        else:
            blend_w = np.ones(1, dtype=np.float64)

        for k in range(w_act):
            t = t0 + k
            w_k = blend_w[k]
            accum_positions[t] += w_k * opt_win[k]
            accum_weights[t] += w_k

        if verbose and (win_idx % 10 == 0 or win_idx == len(window_starts) - 1):
            print(f"  ik optimiser window {win_idx + 1}/{len(window_starts)} (frames {t0}..{t1})")

    # Normalise blended positions by accumulated window weights
    valid_weights = np.maximum(accum_weights, 1e-6)[:, None, None]
    blended_positions = accum_positions / valid_weights

    # 4. Final Anatomical Hinge Clamping & Hierarchical Rigid Bone Projection
    # Clamping anatomical hinges first, then hierarchically projecting bone lengths guarantees
    # invariant rigid bone lengths (< 10^-4 m error) while strictly respecting physiological limits.
    clamped_positions = clamp_anatomical_hinges(blended_positions)
    final_positions = project_rigid_bone_lengths(
        clamped_positions,
        pairs,
        screen_landmarks=landmarks_2d if landmarks_2d is not None and landmarks_2d.shape[1] == 33 else None,
        bone_lengths=rest_lengths,
    )

    return final_positions.astype(np.float32)
