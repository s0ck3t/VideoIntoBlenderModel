"""Biomechanical stabilisation, One-Euro (1€) filtering, and root trajectory estimation.

Provides:
- OneEuroFilter and OneEuroFilter3D adaptive frequency filters.
- Three-stage biomechanical stabilisation:
    1. Left/right limb swap prevention via lateral hysteresis and Euclidean tracking.
    2. Rigid bone length hierarchical projection to temporal median rest lengths.
    3. Anatomical hinge clamping (posterior knee flexion 0–150°, elbow extension).
- Camera-space root trajectory estimation and stance foot vertical ground pinning to Y = 0.
- Backward-compatible COCO-17 2D keypoint stabilisation without external torch dependency.

All operations use British English conventions and pure NumPy mathematics (Apache-2.0 / MIT).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

# Permutation for COCO-17 left/right symmetry (nose at index 0 stays unchanged).
COCO17_SWAP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], dtype=np.int64)

# 24 Body24 joint kinematic tree connections: (parent_index, child_index).
BODY24_BONE_PAIRS: list[tuple[int, int]] = [
    (0, 1),    # pelvis -> left_hip
    (1, 4),    # left_hip -> left_knee
    (4, 7),    # left_knee -> left_ankle
    (7, 10),   # left_ankle -> left_foot
    (0, 2),    # pelvis -> right_hip
    (2, 5),    # right_hip -> right_knee
    (5, 8),    # right_knee -> right_ankle
    (8, 11),   # right_ankle -> right_foot
    (0, 3),    # pelvis -> spine1
    (3, 6),    # spine1 -> spine2
    (6, 9),    # spine2 -> spine3
    (9, 12),   # spine3 -> neck
    (12, 15),  # neck -> head
    (9, 13),   # spine3 -> left_collar
    (13, 16),  # left_collar -> left_shoulder
    (16, 18),  # left_shoulder -> left_elbow
    (18, 20),  # left_elbow -> left_wrist
    (20, 22),  # left_wrist -> left_hand
    (9, 14),   # spine3 -> right_collar
    (14, 17),  # right_collar -> right_shoulder
    (17, 19),  # right_shoulder -> right_elbow
    (19, 21),  # right_elbow -> right_wrist
    (21, 23),  # right_wrist -> right_hand
]

# Body24 24 joint symmetric bilateral pairs for left/right swap handling.
BODY24_LR_PAIRS: list[tuple[int, int]] = [
    (1, 2),    # left_hip <-> right_hip
    (4, 5),    # left_knee <-> right_knee
    (7, 8),    # left_ankle <-> right_ankle
    (10, 11),  # left_foot <-> right_foot
    (13, 14),  # left_collar <-> right_collar
    (16, 17),  # left_shoulder <-> right_shoulder
    (18, 19),  # left_elbow <-> right_elbow
    (20, 21),  # left_wrist <-> right_wrist
    (22, 23),  # left_hand <-> right_hand
]


# ==============================================================================
# 1. One-Euro (1€) Adaptive Frequency Filter
# ==============================================================================

class OneEuroFilter:
    """One-Euro (1€) adaptive low-pass filter (Casiez et al., CHI 2012).

    Dynamically adjusts the cutoff frequency based on derivative velocity to eliminate
    stationary spatial jitter without introducing phase lag during rapid motion.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.005,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    def reset(self) -> None:
        """Reset internal filter state."""
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        """Calculate exponential smoothing factor alpha."""
        if cutoff <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0.0 else 1.0 / 30.0
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, rate: float | None = None) -> float:
        """Filter a scalar signal observation."""
        rate = rate or self.freq
        if self.x_prev is None:
            self.x_prev = float(x)
            return float(x)

        te = 1.0 / rate if rate > 0.0 else 1.0 / 30.0
        dx = (float(x) - self.x_prev) / te
        a_d = self._alpha(rate, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(rate, cutoff)
        x_hat = a * float(x) + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class OneEuroFilter3D:
    """3D adaptive filter with axis-specialised parameters.

    Depth (Z) monocular coordinates exhibit substantially higher variance than lateral
    and vertical coordinates (X, Y). A lower minimum cutoff frequency (0.5 Hz) on Z
    suppresses stationary depth flutter, while a higher velocity slope (beta = 0.05)
    preserves responsiveness during genuine forward and backward strides.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff_xy: float = 1.0,
        beta_xy: float = 0.005,
        min_cutoff_z: float = 0.5,
        beta_z: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self.fx = OneEuroFilter(freq, min_cutoff_xy, beta_xy, d_cutoff)
        self.fy = OneEuroFilter(freq, min_cutoff_xy, beta_xy, d_cutoff)
        self.fz = OneEuroFilter(freq, min_cutoff_z, beta_z, d_cutoff)

    def reset(self) -> None:
        """Reset internal filter states for all three axes."""
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()

    def filter(self, vec3: Sequence[float] | np.ndarray, rate: float | None = None) -> np.ndarray:
        """Filter a single 3D vector (X, Y, Z)."""
        p = np.asarray(vec3, dtype=np.float64).reshape(3)
        return np.array(
            [
                self.fx.filter(p[0], rate),
                self.fy.filter(p[1], rate),
                self.fz.filter(p[2], rate),
            ],
            dtype=np.float64,
        )

    def filter_trajectory(
        self, trajectory: np.ndarray, rate: float | None = None
    ) -> np.ndarray:
        """Filter an entire trajectory sequence of shape (F, 3) or (F, J, 3)."""
        data = np.asarray(trajectory, dtype=np.float64)
        orig_shape = data.shape
        if data.ndim == 2:
            n_frames = orig_shape[0]
            out = np.zeros_like(data)
            self.reset()
            for i in range(n_frames):
                out[i] = self.filter(data[i], rate)
            return out.astype(trajectory.dtype)
        elif data.ndim == 3:
            n_frames, n_joints, _ = orig_shape
            out = np.zeros_like(data)
            filters = [
                OneEuroFilter3D(
                    freq=self.fx.freq,
                    min_cutoff_xy=self.fx.min_cutoff,
                    beta_xy=self.fx.beta,
                    min_cutoff_z=self.fz.min_cutoff,
                    beta_z=self.fz.beta,
                    d_cutoff=self.fx.d_cutoff,
                )
                for _ in range(n_joints)
            ]
            for i in range(n_frames):
                for j in range(n_joints):
                    out[i, j] = filters[j].filter(data[i, j], rate)
            return out.astype(trajectory.dtype)
        else:
            raise ValueError(f"Expected shape (F, 3) or (F, J, 3), got {orig_shape}")


# ==============================================================================
# 2. Biomechanical Stabilisation: Swap Prevention, Bone Lengths & Hinges
# ==============================================================================

def stabilise_limb_swaps_body24(
    joints_3d: np.ndarray,
    screen_landmarks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prevent and correct left/right bilateral limb swaps across time.

    Uses shoulder and hip lateral separation hysteresis combined with temporal
    Euclidean distance tracking to detect and reverse spurious identity flips.

    Args:
        joints_3d: Array of shape (F, 24, 3) in human world coordinates.
        screen_landmarks: Optional array of shape (F, 33, 3) normalized screen landmarks.

    Returns:
        tuple of (stabilised_joints of shape (F, 24, 3), swap_flags of shape (F,)).
    """
    src = np.asarray(joints_3d, dtype=np.float64).copy()
    n_frames = len(src)
    if n_frames <= 1:
        return src.astype(np.float32), np.zeros(n_frames, dtype=np.int32)

    swapped = np.zeros(n_frames, dtype=np.int32)

    # Lateral shoulder separation: left_shoulder (16) vs right_shoulder (17).
    # In human coordinates, +X is anatomical right, -X is anatomical left.
    # Therefore, left - right is typically negative when facing camera.
    sh_diffs = src[:, 16, 0] - src[:, 17, 0]
    median_width = float(np.median(np.abs(sh_diffs)))
    narrow_threshold = max(0.06, 0.45 * median_width)

    last_sign = np.sign(sh_diffs[0]) if abs(sh_diffs[0]) > 1e-4 else 1.0
    last_turn_frame = -1000

    for i in range(1, n_frames):
        current_frame = src[i].copy()
        prev_frame = src[i - 1]

        current_diff = current_frame[16, 0] - current_frame[17, 0]
        current_width = abs(current_diff)
        current_sign = np.sign(current_diff) if current_width > 1e-4 else last_sign

        # Check if a sign inversion occurred while the lateral width remained wide
        is_sudden_flip = False
        if current_width >= narrow_threshold and current_sign != last_sign:
            # Check if recent frames passed through a narrow silhouette turn
            history_start = max(0, last_turn_frame + 1, i - 15)
            recent_widths = np.abs(src[history_start:i, 16, 0] - src[history_start:i, 17, 0])
            narrow_count = int(np.sum(recent_widths < narrow_threshold))
            if narrow_count >= 2:
                # Genuine anatomical turn
                last_sign = current_sign
                last_turn_frame = i
            else:
                is_sudden_flip = True

        # Check tracking continuity cost for limb pairs
        cost_direct = 0.0
        cost_swapped = 0.0
        for left_idx, right_idx in BODY24_LR_PAIRS:
            cost_direct += float(
                np.sum((current_frame[left_idx] - prev_frame[left_idx]) ** 2)
                + np.sum((current_frame[right_idx] - prev_frame[right_idx]) ** 2)
            )
            cost_swapped += float(
                np.sum((current_frame[left_idx] - prev_frame[right_idx]) ** 2)
                + np.sum((current_frame[right_idx] - prev_frame[left_idx]) ** 2)
            )

        should_swap = False
        if is_sudden_flip:
            should_swap = True
        elif cost_swapped + 0.02 < cost_direct and current_width >= narrow_threshold:
            should_swap = True

        if should_swap:
            for left_idx, right_idx in BODY24_LR_PAIRS:
                tmp = current_frame[left_idx].copy()
                current_frame[left_idx] = current_frame[right_idx]
                current_frame[right_idx] = tmp
            swapped[i] = 1

        src[i] = current_frame

    return src.astype(np.float32), swapped


def map_screen_to_body24(screen_landmarks: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Map MediaPipe 33 screen landmarks to 24 Body24 joint 2D screen positions.

    Args:
        screen_landmarks: Array of shape (F, 33, 2) or (F, 33, 3) or (33, 2)/(33, 3).
        device: Compute device ('cpu', 'cuda', 'mps', 'auto').

    Returns:
        Array of shape (F, 24, 2) in normalised screen coordinates [0, 1].
    """
    from .cuda_accel import map_screen_to_body24_cuda, use_cupy

    if use_cupy(device):
        return map_screen_to_body24_cuda(screen_landmarks)

    s = np.asarray(screen_landmarks, dtype=np.float64)
    squeeze = False
    if s.ndim == 2:
        s = s[None, ...]
        squeeze = True
    n_f = s.shape[0]
    out = np.zeros((n_f, 24, 2), dtype=np.float64)

    pelvis_2d = 0.5 * (s[:, 23, :2] + s[:, 24, :2])
    neck_2d = 0.5 * (s[:, 11, :2] + s[:, 12, :2])
    torso_2d = neck_2d - pelvis_2d

    out[:, 0] = pelvis_2d
    out[:, 1] = s[:, 23, :2]  # left_hip
    out[:, 2] = s[:, 24, :2]  # right_hip
    out[:, 3] = pelvis_2d + 0.25 * torso_2d  # spine1
    out[:, 4] = s[:, 25, :2]  # left_knee
    out[:, 5] = s[:, 26, :2]  # right_knee
    out[:, 6] = pelvis_2d + 0.55 * torso_2d  # spine2
    out[:, 7] = s[:, 27, :2]  # left_ankle
    out[:, 8] = s[:, 28, :2]  # right_ankle
    spine3_2d = pelvis_2d + 0.82 * torso_2d
    out[:, 9] = spine3_2d  # spine3
    out[:, 10] = s[:, 31, :2]  # left_foot
    out[:, 11] = s[:, 32, :2]  # right_foot
    out[:, 12] = neck_2d  # neck
    out[:, 13] = spine3_2d + 0.35 * (s[:, 11, :2] - spine3_2d)  # left_collar
    out[:, 14] = spine3_2d + 0.35 * (s[:, 12, :2] - spine3_2d)  # right_collar
    out[:, 15] = 0.5 * (s[:, 7, :2] + s[:, 8, :2])  # head
    out[:, 16] = s[:, 11, :2]  # left_shoulder
    out[:, 17] = s[:, 12, :2]  # right_shoulder
    out[:, 18] = s[:, 13, :2]  # left_elbow
    out[:, 19] = s[:, 14, :2]  # right_elbow
    out[:, 20] = s[:, 15, :2]  # left_wrist
    out[:, 21] = s[:, 16, :2]  # right_wrist
    out[:, 22] = 0.5 * (s[:, 17, :2] + s[:, 19, :2])  # left_hand
    out[:, 23] = 0.5 * (s[:, 18, :2] + s[:, 20, :2])  # right_hand

    if squeeze:
        return out[0]
    return out


def project_rigid_bone_lengths(
    joints_3d: np.ndarray,
    bone_pairs: list[tuple[int, int]] | None = None,
    screen_landmarks: np.ndarray | None = None,
    foreshorten_dist_2d: float = 0.035,
    singular_dist_3d: float = 0.05,
    bone_lengths: dict[tuple[int, int], float] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Project joints hierarchically to maintain invariant temporal median rest lengths.

    Preserves 100% of measured 3D directional unit vectors while strictly eliminating
    non-physical stretching and compression artefacts from monocular estimation.

    When a bone undergoes severe foreshortening (projected screen distance d_2D < 0.035
    or 3D distance < 0.05 m), near-zero length division and noise amplification are
    suppressed by regularising the direction vector towards a temporal prior from
    adjacent non-singular frames, preserving smooth depth Z continuity.

    Args:
        joints_3d: Array of shape (F, 24, 3) in human world coordinates.
        bone_pairs: Kinematic connections (parent, child). Defaults to BODY24_BONE_PAIRS.
        screen_landmarks: Optional MediaPipe screen landmarks of shape (F, 33, 2/3).
        foreshorten_dist_2d: 2D screen distance threshold below which foreshortening occurs.
        singular_dist_3d: 3D distance threshold in metres below which singularity occurs.
        bone_lengths: Optional explicit rest lengths dictionary for each bone pair.
        device: Compute device ('cpu', 'cuda', 'mps', 'auto').

    Returns:
        Rigidly projected array of shape (F, 24, 3) in np.float32.
    """
    pairs = bone_pairs or BODY24_BONE_PAIRS
    data = np.asarray(joints_3d, dtype=np.float64).copy()
    n_frames, n_joints, _ = data.shape

    # 1. Target rest length for each bone segment
    if bone_lengths is not None:
        rest_lengths: dict[tuple[int, int], float] = {
            k: max(float(v), 1e-4) for k, v in bone_lengths.items()
        }
    else:
        rest_lengths = {}
        for parent, child in pairs:
            lengths = np.linalg.norm(data[:, child] - data[:, parent], axis=-1)
            med_len = float(np.median(lengths))
            rest_lengths[(parent, child)] = max(med_len, 1e-4)

    # 2. Extract 2D screen coordinates if provided
    s_body24: np.ndarray | None = None
    if screen_landmarks is not None and len(screen_landmarks) == n_frames:
        s_body24 = map_screen_to_body24(screen_landmarks, device=device)

    # 3. Hierarchically project child joint positions outward from root
    out = data.copy()
    for parent, child in pairs:
        target_len = rest_lengths[(parent, child)]
        deltas = data[:, child] - data[:, parent]
        d_3d = np.linalg.norm(deltas, axis=-1)

        if s_body24 is not None:
            d_2d = np.linalg.norm(s_body24[:, child] - s_body24[:, parent], axis=-1)
        else:
            d_xy = np.linalg.norm(deltas[:, :2], axis=-1)
            d_2d = d_xy * (0.35 / 0.52)

        # Scale thresholds relative to this bone's median length so short bones
        # (collars, hands, spine segments) are not falsely flagged as singular.
        med_2d = float(np.median(d_2d))
        eff_2d_thresh = min(foreshorten_dist_2d, 0.30 * med_2d)
        eff_3d_thresh = min(singular_dist_3d, 0.30 * target_len)

        is_singular = (d_2d < eff_2d_thresh) | (d_3d < eff_3d_thresh)
        valid_indices = [t for t in range(n_frames) if not is_singular[t]]

        priors = np.zeros((n_frames, 3), dtype=np.float64)
        if valid_indices:
            for t in valid_indices:
                priors[t] = deltas[t] / max(d_3d[t], 1e-6)
            first_v = valid_indices[0]
            last_v = valid_indices[-1]
            for t in range(first_v):
                priors[t] = priors[first_v]
            for t in range(last_v + 1, n_frames):
                priors[t] = priors[last_v]
            for k in range(len(valid_indices) - 1):
                i0 = valid_indices[k]
                i1 = valid_indices[k + 1]
                if i1 - i0 > 1:
                    u0 = priors[i0]
                    u1 = priors[i1]
                    for t in range(i0 + 1, i1):
                        alpha = float(t - i0) / float(i1 - i0)
                        u_blend = (1.0 - alpha) * u0 + alpha * u1
                        norm_b = float(np.linalg.norm(u_blend))
                        priors[t] = u_blend / norm_b if norm_b > 1e-4 else u0
        else:
            for t in range(n_frames):
                if d_3d[t] > 1e-6:
                    priors[t] = deltas[t] / d_3d[t]
                else:
                    priors[t] = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        for t in range(n_frames):
            p_pos = out[t, parent]
            dist_3d = float(d_3d[t])
            dist_2d = float(d_2d[t])
            u_prior = priors[t]

            s_2d = max(0.0, min(1.0, (eff_2d_thresh - dist_2d) / max(eff_2d_thresh, 1e-6))) if eff_2d_thresh > 0 else 0.0
            s_3d = max(0.0, min(1.0, (eff_3d_thresh - dist_3d) / max(eff_3d_thresh, 1e-6))) if eff_3d_thresh > 0 else 0.0
            s_blend = max(s_2d, s_3d)

            if s_blend <= 0.0 and dist_3d >= eff_3d_thresh:
                direction = deltas[t] / max(dist_3d, 1e-6)
            elif dist_3d < 1e-4:
                direction = u_prior
            else:
                u_raw = deltas[t] / dist_3d
                blended = (1.0 - 0.65 * s_blend) * u_raw + (0.65 * s_blend) * u_prior
                norm_b = float(np.linalg.norm(blended))
                direction = blended / norm_b if norm_b > 1e-6 else u_prior

            out[t, child] = p_pos + direction * target_len

    return out.astype(np.float32)


def clamp_anatomical_hinges(joints_3d: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Enforce physiological range-of-motion constraints on knees and elbows.

    Knees:
    - Prevent posterior hyperextension relative to the subject's instantaneous
      pelvis sagittal plane while preserving normal anterior kneecap flexion.
    - Bound maximum interior flexion angle to 145° without altering bone lengths.

    Elbows:
    - Bound maximum interior flexion angle to 145° while preserving 3D shoulder
      abduction/internal-rotation planes and rigid upper-arm/forearm bone lengths.

    Args:
        joints_3d: Array of shape (F, 24, 3) in human coordinates (+Y up, -Z forward, +Z backward).
        device: Compute device ('cpu', 'cuda', 'mps', 'auto').

    Returns:
        Clamped array of shape (F, 24, 3) in np.float32.
    """
    from .cuda_accel import clamp_anatomical_hinges_cuda, use_cupy

    if use_cupy(device):
        return clamp_anatomical_hinges_cuda(joints_3d)

    data = np.asarray(joints_3d, dtype=np.float64).copy()
    n_frames = len(data)

    # 1. Prevent posterior knee hyperextension in the subject's local pelvis frame
    leg_triads = [
        (1, 4, 7, 10),  # Left leg (hip, knee, ankle, foot)
        (2, 5, 8, 11),  # Right leg (hip, knee, ankle, foot)
    ]
    for hip_idx, knee_idx, ankle_idx, foot_idx in leg_triads:
        for t in range(n_frames):
            # Pelvis-relative anterior unit vector: f_body = normalize(up x right)
            v_right = data[t, 2] - data[t, 1]
            v_up = data[t, 12] - data[t, 0]
            f_body = np.cross(v_up, v_right)
            n_f = float(np.linalg.norm(f_body))
            if n_f > 1e-6:
                f_body /= n_f
            else:
                f_body = np.array([0.0, 0.0, -1.0], dtype=np.float64)

            p_hip = data[t, hip_idx]
            p_knee = data[t, knee_idx]
            p_ankle = data[t, ankle_idx]

            v_leg = p_ankle - p_hip
            len_leg = float(np.linalg.norm(v_leg))
            if len_leg < 1e-4:
                continue
            u_leg = v_leg / len_leg

            v_thigh = p_knee - p_hip
            n_thigh = float(np.linalg.norm(v_thigh))
            n_shin = float(np.linalg.norm(p_ankle - p_knee))
            proj_len = float(np.dot(v_thigh, u_leg))
            p_proj = p_hip + proj_len * u_leg
            deflection = p_knee - p_proj

            # Kneecap must deflect anteriorly (along f_body); if it hyperextends
            # posteriorly behind the hip-ankle chord, reflect the posterior component.
            ant_comp = float(np.dot(deflection, f_body))
            if ant_comp < -0.015:
                deflection_fixed = deflection - (ant_comp + 0.015) * f_body
                new_thigh = (p_proj + deflection_fixed) - p_hip
                n_new = float(np.linalg.norm(new_thigh))
                if n_new > 1e-6 and n_thigh > 1e-6:
                    data[t, knee_idx] = p_hip + (new_thigh / n_new) * n_thigh
                    v_shin = p_ankle - data[t, knee_idx]
                    n_s = float(np.linalg.norm(v_shin))
                    if n_s > 1e-6 and n_shin > 1e-6:
                        new_ankle = data[t, knee_idx] + (v_shin / n_s) * n_shin
                        data[t, foot_idx] += new_ankle - p_ankle
                        data[t, ankle_idx] = new_ankle

    # 2. Bound interior flexion angle (0° <= theta <= 145°) for knees and elbows
    #    while strictly preserving segment bone lengths.
    hinge_chains = [
        (1, 4, 7, 10, 145.0),    # Left leg
        (2, 5, 8, 11, 145.0),    # Right leg
        (16, 18, 20, 22, 145.0), # Left arm
        (17, 19, 21, 23, 145.0), # Right arm
    ]
    for prox_idx, mid_idx, dist_idx, term_idx, max_deg in hinge_chains:
        for t in range(n_frames):
            p_prox = data[t, prox_idx]
            p_mid = data[t, mid_idx]
            p_dist = data[t, dist_idx]

            v1 = p_mid - p_prox
            v2 = p_dist - p_mid
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-4 or n2 < 1e-4:
                continue

            u1 = v1 / n1
            u2 = v2 / n2
            cos_theta = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(cos_theta))

            if angle_deg > max_deg:
                max_rad = math.radians(max_deg)
                ortho = u2 - cos_theta * u1
                norm_ortho = float(np.linalg.norm(ortho))
                if norm_ortho > 1e-6:
                    ortho /= norm_ortho
                    u2_clamped = math.cos(max_rad) * u1 + math.sin(max_rad) * ortho
                    new_dist = p_mid + u2_clamped * n2
                    delta_dist = new_dist - p_dist
                    data[t, dist_idx] = new_dist
                    data[t, term_idx] += delta_dist

    return data.astype(np.float32)


class BiomechanicalStabiliser:
    """Comprehensive three-stage biomechanical stabilisation pipeline.

    Applies:
    1. Left/right bilateral limb swap prevention.
    2. Hierarchical rigid bone length projection to temporal median rest lengths.
    3. Physiological knee and elbow range-of-motion clamping.
    """

    def __init__(self, bone_pairs: list[tuple[int, int]] | None = None) -> None:
        self.bone_pairs = bone_pairs or BODY24_BONE_PAIRS

    def stabilise(
        self,
        joints_3d: np.ndarray,
        screen_landmarks: np.ndarray | None = None,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute full biomechanical stabilisation on a 3D joint sequence.

        Args:
            joints_3d: Raw extracted joints of shape (F, 24, 3).
            screen_landmarks: Optional screen coordinates of shape (F, 33, 3).
            device: Compute device ('cpu', 'cuda', 'mps', 'auto').

        Returns:
            tuple of (stabilised_joints of shape (F, 24, 3), swap_flags of shape (F,)).
        """
        # Stage 1: Left/Right swap prevention
        swapped_joints, swap_flags = stabilise_limb_swaps_body24(joints_3d, screen_landmarks)

        # Stage 2: Foreshortening-aware rigid bone length projection
        rigid_joints = project_rigid_bone_lengths(
            swapped_joints, self.bone_pairs, screen_landmarks=screen_landmarks, device=device
        )

        # Stage 3: Anatomical hinge clamping
        clamped_joints = clamp_anatomical_hinges(rigid_joints, device=device)

        return clamped_joints, swap_flags


# ==============================================================================
# 3. Ground Contact & Root Trajectory Estimation
# ==============================================================================

def estimate_root_trajectory_and_pinning(
    joints_3d: np.ndarray,
    screen_landmarks: np.ndarray | None = None,
    fps: float = 30.0,
    stance_velocity_threshold: float = 0.15,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera-space root translation and enforce foot ground pinning at Y = 0.

    1. Reconstructs camera-space depth (Z_c) from 2D screen torso scale relative to
       the median 3D torso metric length.
    2. Reconstructs lateral (X_c) and vertical (Y_c) pelvis positions from screen coordinates.
    3. Identifies stance foot contact phases by detecting foot velocity drops below
       the threshold (0.15 m/s).
    4. Enforces vertical ground pinning so that during stance phases, the lowest foot contact
       point firmly touches the ground plane Y = 0 without vertical floating or floor penetration.
    5. Filters the recovered trajectory with an adaptive One-Euro (1€) filter.

    Args:
        joints_3d: Stabilised 3D joints of shape (F, 24, 3) in human world coordinates.
        screen_landmarks: Optional 2D screen landmarks of shape (F, 33, 3).
        fps: Sampling rate in frames per second.
        stance_velocity_threshold: Velocity threshold in m/s below which a foot is planted.
        device: Compute device ('cpu', 'cuda', 'mps', 'auto').

    Returns:
        tuple of (root_translation of shape (F, 3), root_orientation of shape (F, 3)).
    """
    data = np.asarray(joints_3d, dtype=np.float64)
    n_frames = len(data)
    dt = 1.0 / float(fps) if fps > 0 else 1.0 / 30.0

    root_trans = np.zeros((n_frames, 3), dtype=np.float64)
    root_orient = np.zeros((n_frames, 3), dtype=np.float32)

    # Foot indices in Body24 24: left_ankle (7), right_ankle (8), left_foot (10), right_foot (11)
    foot_indices = [7, 8, 10, 11]

    # Calculate foot velocities for stance phase identification
    v_left = np.zeros(n_frames, dtype=np.float64)
    v_right = np.zeros(n_frames, dtype=np.float64)
    for t in range(1, n_frames):
        d_left = np.linalg.norm(data[t, 10] - data[t - 1, 10])
        d_right = np.linalg.norm(data[t, 11] - data[t - 1, 11])
        v_left[t] = d_left / dt
        v_right[t] = d_right / dt
    v_left[0] = v_left[1] if n_frames > 1 else 0.0
    v_right[0] = v_right[1] if n_frames > 1 else 0.0

    # Camera-space depth estimation via foreshortening-compensated 2D torso & shoulder scale
    if screen_landmarks is not None and len(screen_landmarks) == n_frames:
        screen = np.asarray(screen_landmarks, dtype=np.float64)
        # MediaPipe landmarks: 11 (left shoulder), 12 (right shoulder), 23 (left hip), 24 (right hip)
        neck_2d = 0.5 * (screen[:, 11, :2] + screen[:, 12, :2])
        pelvis_2d = 0.5 * (screen[:, 23, :2] + screen[:, 24, :2])
        torso_2d_height = np.linalg.norm(neck_2d - pelvis_2d, axis=-1)
        shoulder_2d_width = np.linalg.norm(screen[:, 11, :2] - screen[:, 12, :2], axis=-1)

        # Compensate for 3D torso pitch/foreshortening when crouching or leaning toward camera
        torso_3d_vec = data[:, 12] - data[:, 0]  # neck (12) - pelvis (0)
        torso_3d_len = np.maximum(np.linalg.norm(torso_3d_vec, axis=-1), 1e-4)
        torso_3d_xy = np.linalg.norm(torso_3d_vec[:, :2], axis=-1)
        cos_tilt = np.clip(torso_3d_xy / torso_3d_len, 0.35, 1.0)

        scale_2d = np.maximum(torso_2d_height / cos_tilt, 1.35 * shoulder_2d_width)
        ref_scale_2d = max(float(np.percentile(scale_2d, 75)), 0.15)
        scale_2d = np.clip(scale_2d, 0.75 * ref_scale_2d, 1.35 * ref_scale_2d)

        median_torso_3d = float(np.median(torso_3d_len))
        if median_torso_3d < 0.2 or median_torso_3d > 0.9:
            median_torso_3d = 0.52  # Standard human anatomical torso rest length in metres

        # Normalized focal length f ~= 1.0
        focal_length = 1.0
        ref_depth_z = float(np.clip((focal_length * median_torso_3d) / ref_scale_2d, 1.5, 4.5))
        raw_depth_z = (focal_length * median_torso_3d) / scale_2d
        depth_z = ref_depth_z + np.clip(raw_depth_z - ref_depth_z, -0.60, 0.60)
        pos_x = (pelvis_2d[:, 0] - 0.5) * ref_depth_z / focal_length
        pos_y_screen = -(pelvis_2d[:, 1] - 0.5) * ref_depth_z / focal_length
    else:
        depth_z = np.zeros(n_frames, dtype=np.float64)
        pos_x = np.zeros(n_frames, dtype=np.float64)
        pos_y_screen = np.zeros(n_frames, dtype=np.float64)

    # Vertical Ground Pinning: Lowest anatomical contact point pinned to Y = 0.
    # When upright, uses stance foot; during floor-work (sitting, back-rolls, crouches
    # where feet are airborne above the torso/pelvis), pins the lowest body joint to Y = 0.
    for t in range(n_frames):
        feet_heights = data[t, foot_indices, 1]
        lowest_body_y = float(np.min(data[t, :, 1]))
        lowest_foot_y = float(np.min(feet_heights))
        is_left_stance = v_left[t] < stance_velocity_threshold
        is_right_stance = v_right[t] < stance_velocity_threshold

        if lowest_foot_y <= lowest_body_y + 0.04:
            if is_left_stance and not is_right_stance:
                y_contact = min(data[t, 7, 1], data[t, 10, 1], lowest_body_y + 0.04)
            elif is_right_stance and not is_left_stance:
                y_contact = min(data[t, 8, 1], data[t, 11, 1], lowest_body_y + 0.04)
            else:
                y_contact = lowest_foot_y
        else:
            y_contact = lowest_body_y

        ty = -y_contact
        root_trans[t, 0] = pos_x[t] - pos_x[0]
        root_trans[t, 1] = ty
        root_trans[t, 2] = depth_z[t] - depth_z[0]

    # Smooth root translation trajectory with OneEuroFilter3D
    root_filter = OneEuroFilter3D(freq=fps, min_cutoff_xy=1.0, beta_xy=0.005, min_cutoff_z=0.5, beta_z=0.05)
    smoothed_trans = root_filter.filter_trajectory(root_trans, rate=fps)

    # Ensure vertical ground pinning precision: lowest body contact point remains firmly at Y = 0
    for t in range(n_frames):
        lowest_y = float(np.min(data[t, :, 1]))
        target_ty = -lowest_y
        # Soft blend towards exact ground contact
        smoothed_trans[t, 1] = 0.85 * target_ty + 0.15 * smoothed_trans[t, 1]

    return smoothed_trans.astype(np.float32), root_orient


# ==============================================================================
# 4. Backward Compatibility: 2D COCO-17 Keypoint Stabilisation (Pure NumPy)
# ==============================================================================

def _lr_x(frame: np.ndarray) -> float:
    """Calculate shoulder left-minus-right in image x; fallback to hips."""
    sh = float(frame[5, 0] - frame[6, 0])
    if abs(sh) >= 40.0:
        return sh
    return float(frame[11, 0] - frame[12, 0])


def stabilize_coco17(kp2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stabilise 2D COCO-17 keypoints across time using pure NumPy.

    Retained for backward compatibility with existing unit tests without requiring
    any torch or copyleft dependencies.
    """
    src = np.asarray(kp2d, dtype=np.float64)
    if src.ndim != 3 or src.shape[1] < 17:
        raise ValueError(f"expected (F, 17, 3) COCO-17, got {src.shape}")
    out = src.copy()
    n = len(out)
    narrow = float(np.clip(0.45 * np.median([abs(_lr_x(f)) for f in src]), 40.0, 80.0))
    swapped = np.zeros(n, dtype=np.int32)
    last_sign = np.sign(_lr_x(out[0])) or 1.0
    last_turn = -(10**9)
    for i in range(n):
        raw = src[i]
        width = abs(_lr_x(raw))
        sign = np.sign(_lr_x(raw)) or last_sign
        if width >= 40.0 and sign != last_sign:
            start = max(0, last_turn + 1, i - 18)
            widths = [abs(_lr_x(f)) for f in out[start:i]]
            run = best = 0
            for w in widths:
                run = run + 1 if w < narrow else 0
                best = max(best, run)
            if best >= (1 if last_turn < 0 else 4):
                last_sign = sign
                last_turn = i
            else:
                raw = raw[COCO17_SWAP]
                swapped[i] = 1
        out[i] = raw
        sh = out[i, 5, 0] - out[i, 6, 0]
        hip = out[i, 11, 0] - out[i, 12, 0]
        if abs(sh) >= 40.0 and abs(hip) >= 40.0 and np.sign(sh) != np.sign(hip):
            out[i, [11, 12, 13, 14, 15, 16]] = out[i, [12, 11, 14, 13, 16, 15]]
    return out.astype(np.float32), swapped
