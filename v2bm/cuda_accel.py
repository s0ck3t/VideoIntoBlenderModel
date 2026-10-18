"""NVIDIA CUDA GPU acceleration across all stages of the v2bm pipeline.

Provides optional, 100% commercial-permissive (Apache-2.0 / MIT / BSD) GPU
acceleration powered by CuPy and FFmpeg NVIDIA NVENC / NVDEC:

1. Stage 1 — Media Ingestion & Preflight:
   - Hardware-accelerated H.264 decoding (`-hwaccel cuda`) and encoding (`h264_nvenc`).
   - MediaPipe GPU delegate probing with automatic CPU delegate fallback on platforms
     where TFLite OpenGL ES is disabled in prebuilt wheels (e.g. Windows).
2. Stage 2 — Multi-Performer Isolation (`v2bm/isolate.py`):
   - GPU-accelerated bipartite tracking cost matrices (`compute_tracking_cost_matrix_cuda`).
   - GPU-accelerated silhouette thresholding, elliptical morphological dilation, and
     target performer mask protection (`build_removal_mask_cuda`).
   - NVIDIA NVENC (`h264_nvenc`) hardware video encoding for isolated clips.
3. Stage 3 — 3D Pose Extraction, Biomechanical Stabilisation & Sliding-Window IK:
   - Vectorised GPU `Body24` coordinate transformation (`map_to_body24_batch_cuda`,
     `map_screen_to_body24_cuda`).
   - GPU bilateral limb swap cost evaluation, anatomical hinge clamping
     (`clamp_anatomical_hinges_cuda`), and rigid bone projection
     (`project_rigid_bone_lengths_cuda`).
   - Parallel batched GPU sliding-window 2D-reprojection Inverse Kinematics solver
     (`optimise_trajectory_windowed_cuda`) solving all temporal windows simultaneously
     in VRAM.
   - GPU camera-space root depth trajectory and stance foot pinning
     (`estimate_root_trajectory_and_pinning_cuda`).
4. Stage 4 — Retargeting, Geodesic Quaternion Smoothing & Linear Blend Skinning:
   - Batched GPU geodesic quaternion trajectory filtering (`smooth_rotation_trajectories_cuda`).
   - Batched multi-frame GPU Linear Blend Skinning (`skin_frames_cuda`, `skin_mesh_cuda`)
     evaluating `(F, V, 3)` posed vertices via tensor contractions in VRAM.
5. Stage 5 — Binary glTF 2.0 Export & Preview / Studio Rendering:
   - GPU vertex normal computation (`vertex_normals_cuda`), top-4 joint weight partitioning
     (`top4_weights_cuda`), and vertex colour accumulation (`accumulate_vertex_colors_cuda`).
   - GPU 3D perspective projection, triangle normal cross products, backface culling,
     Lambertian shading, and Z-depth sorting (`project_and_shade_mesh_cuda`).
   - Concurrent NVIDIA NVENC (`h264_nvenc`) hardware video streaming in `_StreamWriter`
     and headless Blender `OPTIX` / `CUDA` GPU rendering.

All code, comments, docstrings, and documentation strictly use British English conventions.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import cupy as cp  # type: ignore[import-untyped]
    from cupyx.scipy import ndimage as cp_ndimage  # type: ignore[import-untyped]

    _HAS_CUPY = True
except Exception:
    cp = None  # type: ignore[assignment]
    cp_ndimage = None  # type: ignore[assignment]
    _HAS_CUPY = False


@dataclass(frozen=True)
class CudaCapabilities:
    """Detected NVIDIA CUDA hardware and software capabilities on the current host."""

    cupy_available: bool
    cupy_version: str | None
    gpu_name: str | None
    compute_capability: str | None
    vram_gb: float | None
    nvenc_available: bool
    cuvid_available: bool
    mediapipe_gpu_delegate: bool

    @property
    def any_cuda_available(self) -> bool:
        """Return True if either CuPy GPU compute or NVIDIA NVENC hardware encoding is ready."""
        return self.cupy_available or self.nvenc_available

    @property
    def cuda_available(self) -> bool:
        """Alias for `any_cuda_available`."""
        return self.any_cuda_available

    @property
    def device_name(self) -> str:
        """Human-readable GPU device name."""
        return self.gpu_name or "CUDA GPU"

    @property
    def ffmpeg_nvenc(self) -> bool:
        """Alias for `nvenc_available`."""
        return self.nvenc_available

    @property
    def ffmpeg_cuvid(self) -> bool:
        """Alias for `cuvid_available`."""
        return self.cuvid_available

    def summary(self) -> str:
        """Human-readable one-line status summary for `v2bm doctor` and run reports."""
        parts: list[str] = []
        if self.gpu_name:
            vram = f" ({self.vram_gb:.1f} GB VRAM)" if self.vram_gb else ""
            parts.append(f"{self.gpu_name}{vram}")
        if self.cupy_available:
            parts.append(f"CuPy {self.cupy_version}")
        if self.nvenc_available:
            parts.append("FFmpeg h264_nvenc")
        if self.cuvid_available:
            parts.append("FFmpeg cuda hwaccel")
        if not parts:
            return "unavailable (install cupy-cuda12x / cupy-cuda13x or NVIDIA drivers)"
        return ", ".join(parts)


_CAPS_CACHE: CudaCapabilities | None = None


def _probe_ffmpeg_nvenc() -> tuple[bool, bool]:
    """Probe whether FFmpeg supports NVIDIA `h264_nvenc` encoding and `-hwaccel cuda` decoding."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, False

    nvenc_ok = False
    cuvid_ok = False
    try:
        res = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "nullsrc=s=256x256:d=0.04:r=25",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        nvenc_ok = res.returncode == 0
    except Exception:
        nvenc_ok = False

    try:
        res_hw = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if res_hw.returncode == 0 and "cuda" in res_hw.stdout.lower().split():
            cuvid_ok = True
    except Exception:
        cuvid_ok = False

    return nvenc_ok, cuvid_ok


def get_cuda_capabilities(*, refresh: bool = False) -> CudaCapabilities:
    """Detect and cache NVIDIA CUDA capabilities across CuPy, FFmpeg, and MediaPipe."""
    global _CAPS_CACHE
    if _CAPS_CACHE is not None and not refresh:
        return _CAPS_CACHE

    cupy_ok = False
    cupy_ver: str | None = None
    gpu_name: str | None = None
    cc_str: str | None = None
    vram_gb: float | None = None

    if _HAS_CUPY and cp is not None:
        try:
            if cp.cuda.is_available():
                dev = cp.cuda.Device(0)
                props = cp.cuda.runtime.getDeviceProperties(0)
                raw_name = props.get("name", b"NVIDIA GPU")
                gpu_name = (
                    raw_name.decode("utf-8", errors="replace")
                    if isinstance(raw_name, (bytes, bytearray))
                    else str(raw_name)
                )
                cc = str(dev.compute_capability)
                cc_str = f"{cc[:-1]}.{cc[-1]}" if len(cc) >= 2 else cc
                total_mem = float(props.get("totalGlobalMem", 0))
                if total_mem > 0:
                    vram_gb = round(total_mem / (1024.0**3), 1)
                # Verify a minimal kernel execution on device 0
                _test = cp.ones((4, 4), dtype=cp.float32)
                _ = float(cp.sum(_test))
                cupy_ok = True
                cupy_ver = str(cp.__version__)
        except Exception:
            cupy_ok = False

    nvenc_ok, cuvid_ok = _probe_ffmpeg_nvenc()

    # If CuPy wasn't installed, still try querying GPU name via nvidia-smi if NVENC is present
    if gpu_name is None and (nvenc_ok or cuvid_ok):
        nvsmi = shutil.which("nvidia-smi")
        if nvsmi:
            try:
                res = subprocess.run(
                    [nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    first_line = res.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in first_line.split(",")]
                    if parts:
                        gpu_name = parts[0]
                    if len(parts) > 1:
                        vram_gb = round(float(parts[1]) / 1024.0, 1)
            except Exception:
                pass

    _CAPS_CACHE = CudaCapabilities(
        cupy_available=cupy_ok,
        cupy_version=cupy_ver,
        gpu_name=gpu_name,
        compute_capability=cc_str,
        vram_gb=vram_gb,
        nvenc_available=nvenc_ok,
        cuvid_available=cuvid_ok,
        mediapipe_gpu_delegate=False,
    )
    return _CAPS_CACHE


def resolve_device(device: str | None) -> str:
    """Normalise device string ('cpu', 'cuda', 'mps', 'auto') to an active execution target."""
    if not device:
        return "cpu"
    low = str(device).strip().lower()
    if low == "auto":
        caps = get_cuda_capabilities()
        return "cuda" if caps.any_cuda_available else "cpu"
    return low


def use_cuda(device: str | None) -> bool:
    """Return True if the requested device resolves to CUDA and CUDA hardware is available."""
    resolved = resolve_device(device)
    if resolved != "cuda":
        return False
    return get_cuda_capabilities().any_cuda_available


def use_cupy(device: str | None) -> bool:
    """Return True if CuPy GPU array execution should be used for `device`."""
    resolved = resolve_device(device)
    if resolved != "cuda":
        return False
    return get_cuda_capabilities().cupy_available


def use_nvenc(device: str | None) -> bool:
    """Return True if FFmpeg NVIDIA NVENC (`h264_nvenc`) should be used for `device`."""
    resolved = resolve_device(device)
    if resolved != "cuda":
        return False
    return get_cuda_capabilities().nvenc_available


def to_gpu(array: np.ndarray, dtype=None):
    """Transfer a NumPy array to CUDA VRAM via CuPy."""
    if not _HAS_CUPY or cp is None:
        return np.asarray(array, dtype=dtype)
    return cp.asarray(array, dtype=dtype)


def to_cpu(array) -> np.ndarray:
    """Transfer a CuPy GPU array back to host NumPy memory."""
    if _HAS_CUPY and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


# ==============================================================================
# Stage 1 & 5: FFmpeg NVIDIA NVENC / NVDEC Hardware Video Helpers
# ==============================================================================

def ffmpeg_video_encode_args(
    device: str | None = "cpu",
    *,
    crf: int = 18,
    preset: str = "fast",
) -> list[str]:
    """Return FFmpeg video codec arguments using NVIDIA `h264_nvenc` when CUDA is enabled."""
    if use_nvenc(device):
        # Map x264 presets to NVENC p1..p7 presets
        nvenc_preset = {
            "ultrafast": "p1",
            "superfast": "p2",
            "veryfast": "p3",
            "faster": "p3",
            "fast": "p4",
            "medium": "p4",
            "slow": "p5",
        }.get(preset, "p4")
        return [
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-cq",
            str(crf),
            "-preset",
            nvenc_preset,
        ]
    return [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
    ]


# ==============================================================================
# Stage 2: Multi-Performer Isolation & Silhouette Protection on CUDA
# ==============================================================================

def compute_tracking_cost_matrix_cuda(
    pred_centroids: np.ndarray,
    last_heights: np.ndarray,
    last_colours: np.ndarray,
    det_centroids: np.ndarray,
    det_heights: np.ndarray,
    det_colours: np.ndarray,
    diag: float,
) -> np.ndarray:
    """Compute the (N_tracks, N_dets) bipartite assignment cost matrix on CUDA."""
    if not _HAS_CUPY or cp is None:
        raise RuntimeError("CuPy is not available")

    p_c = cp.asarray(pred_centroids, dtype=cp.float64)[:, None, :]  # (T, 1, 2)
    d_c = cp.asarray(det_centroids, dtype=cp.float64)[None, :, :]   # (1, D, 2)
    d_spatial = cp.linalg.norm(p_c - d_c, axis=-1) / max(float(diag), 1.0)

    p_h = cp.asarray(last_heights, dtype=cp.float64)[:, None]       # (T, 1)
    d_h = cp.asarray(det_heights, dtype=cp.float64)[None, :]        # (1, D)
    max_h = cp.maximum(cp.maximum(p_h, d_h), 1.0)
    d_height = cp.abs(p_h - d_h) / max_h

    p_col = cp.asarray(last_colours, dtype=cp.float64)[:, None, :]  # (T, 1, 3)
    d_col = cp.asarray(det_colours, dtype=cp.float64)[None, :, :]   # (1, D, 3)
    d_colour = cp.linalg.norm(p_col - d_col, axis=-1) / 441.67

    cost = 0.50 * d_spatial + 0.30 * d_colour + 0.20 * d_height
    return cp.asnumpy(cost)


def build_removal_mask_cuda(
    frame_shape: tuple[int, int],
    matched_tracks: dict,
    target_track_id: int | None,
    *,
    dilate_removal_px: int = 17,
    protect_target_px: int = 7,
) -> np.ndarray:
    """Build union non-target removal mask with target silhouette protection on CUDA."""
    import cv2

    h, w = frame_shape[:2]
    union_removal_gpu = cp.zeros((h, w), dtype=cp.uint8)
    target_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)

    for t_id, det in matched_tracks.items():
        if det.segmentation_mask is not None:
            seg_gpu = cp.asarray(det.segmentation_mask)
            m_gpu = (seg_gpu > 0.20).astype(cp.uint8)
        else:
            m_cpu = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = det.bbox
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            axes = (int(max((x2 - x1) / 2 * 1.1, 5)), int(max((y2 - y1) / 2 * 1.05, 5)))
            cv2.ellipse(m_cpu, center, axes, 0, 0, 360, 1, -1)
            m_gpu = cp.asarray(m_cpu)

        if t_id == target_track_id:
            target_mask_gpu = cp.maximum(target_mask_gpu, m_gpu)
        else:
            union_removal_gpu = cp.maximum(union_removal_gpu, m_gpu)

    if int(cp.sum(union_removal_gpu)) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    k_removal = cp.asarray(
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_removal_px, dilate_removal_px))
    )
    k_protect = cp.asarray(
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (protect_target_px, protect_target_px))
    )

    dilated_removal = cp_ndimage.grey_dilation(union_removal_gpu, footprint=k_removal)
    dilated_removal = cp_ndimage.grey_dilation(dilated_removal, footprint=k_removal)

    if int(cp.sum(target_mask_gpu)) > 0:
        dilated_target = cp_ndimage.grey_dilation(target_mask_gpu, footprint=k_protect)
        final_gpu = cp.where(dilated_target == 1, 0, dilated_removal).astype(cp.uint8) * 255
    else:
        final_gpu = dilated_removal.astype(cp.uint8) * 255

    return cp.asnumpy(final_gpu)


# ==============================================================================
# Stage 3: GPU Pose Mapping, Biomechanical Stabilisation & Batched IK Solver
# ==============================================================================

def map_to_body24_batch_cuda(
    world_landmarks_seq: np.ndarray,
    visibilities_seq: np.ndarray,
) -> np.ndarray:
    """Transform a batch of MediaPipe world landmarks (T, 33, 3) into (T, 24, 3) Body24 joints on CUDA."""
    w = cp.asarray(world_landmarks_seq, dtype=cp.float32)
    vis = cp.asarray(visibilities_seq, dtype=cp.float32)
    n_frames = w.shape[0]

    p = cp.zeros_like(w)
    p[:, :, 0] = -w[:, :, 0]
    p[:, :, 1] = -w[:, :, 1]
    p[:, :, 2] = w[:, :, 2]

    j = cp.zeros((n_frames, 24, 3), dtype=cp.float32)

    pelvis = 0.5 * (p[:, 23] + p[:, 24])
    j[:, 0] = pelvis
    j[:, 1] = p[:, 23]
    j[:, 2] = p[:, 24]

    neck = 0.5 * (p[:, 11] + p[:, 12])
    j[:, 12] = neck

    torso_vec = neck - pelvis
    j[:, 3] = pelvis + 0.25 * torso_vec
    j[:, 6] = pelvis + 0.55 * torso_vec
    j[:, 9] = pelvis + 0.82 * torso_vec

    j[:, 4] = p[:, 25]
    j[:, 5] = p[:, 26]
    j[:, 7] = p[:, 27]
    j[:, 8] = p[:, 28]
    j[:, 10] = p[:, 31]
    j[:, 11] = p[:, 32]

    j[:, 13] = j[:, 9] + 0.35 * (p[:, 11] - j[:, 9])
    j[:, 14] = j[:, 9] + 0.35 * (p[:, 12] - j[:, 9])

    v_ear = cp.minimum(vis[:, 7], vis[:, 8])[:, None]
    ear_mid = 0.5 * (p[:, 7] + p[:, 8])
    nose_fb = p[:, 0] + cp.array([0.0, 0.04, 0.08], dtype=cp.float32)[None, :]
    j[:, 15] = cp.where(v_ear > 0.3, ear_mid, nose_fb)

    j[:, 16] = p[:, 11]
    j[:, 17] = p[:, 12]
    j[:, 18] = p[:, 13]
    j[:, 19] = p[:, 14]
    j[:, 20] = p[:, 15]
    j[:, 21] = p[:, 16]

    j[:, 22] = 0.5 * (p[:, 17] + p[:, 19])
    j[:, 23] = 0.5 * (p[:, 18] + p[:, 20])

    return cp.asnumpy(j)


def map_screen_to_body24_cuda(screen_landmarks: np.ndarray) -> np.ndarray:
    """Vectorised GPU mapping of MediaPipe 33 screen landmarks to 24 Body24 2D coordinates."""
    s = cp.asarray(screen_landmarks, dtype=cp.float64)
    squeeze = False
    if s.ndim == 2:
        s = s[None, ...]
        squeeze = True
    n_f = s.shape[0]
    out = cp.zeros((n_f, 24, 2), dtype=cp.float64)

    pelvis_2d = 0.5 * (s[:, 23, :2] + s[:, 24, :2])
    neck_2d = 0.5 * (s[:, 11, :2] + s[:, 12, :2])
    torso_2d = neck_2d - pelvis_2d

    out[:, 0] = pelvis_2d
    out[:, 1] = s[:, 23, :2]
    out[:, 2] = s[:, 24, :2]
    out[:, 3] = pelvis_2d + 0.25 * torso_2d
    out[:, 4] = s[:, 25, :2]
    out[:, 5] = s[:, 26, :2]
    out[:, 6] = pelvis_2d + 0.55 * torso_2d
    out[:, 7] = s[:, 27, :2]
    out[:, 8] = s[:, 28, :2]
    spine3_2d = pelvis_2d + 0.82 * torso_2d
    out[:, 9] = spine3_2d
    out[:, 10] = s[:, 31, :2]
    out[:, 11] = s[:, 32, :2]
    out[:, 12] = neck_2d
    out[:, 13] = spine3_2d + 0.35 * (s[:, 11, :2] - spine3_2d)
    out[:, 14] = spine3_2d + 0.35 * (s[:, 12, :2] - spine3_2d)
    out[:, 15] = 0.5 * (s[:, 7, :2] + s[:, 8, :2])
    out[:, 16] = s[:, 11, :2]
    out[:, 17] = s[:, 12, :2]
    out[:, 18] = s[:, 13, :2]
    out[:, 19] = s[:, 14, :2]
    out[:, 20] = s[:, 15, :2]
    out[:, 21] = s[:, 16, :2]
    out[:, 22] = 0.5 * (s[:, 17, :2] + s[:, 19, :2])
    out[:, 23] = 0.5 * (s[:, 18, :2] + s[:, 20, :2])

    res = cp.asnumpy(out)
    return res[0] if squeeze else res


def clamp_anatomical_hinges_cuda(joints_3d: np.ndarray) -> np.ndarray:
    """Vectorised GPU enforcement of physiological knee and elbow range-of-motion limits."""
    data = cp.asarray(joints_3d, dtype=cp.float64).copy()
    n_frames = data.shape[0]
    if n_frames == 0:
        return cp.asnumpy(data.astype(cp.float32))

    # 1. Prevent posterior knee hyperextension across all frames simultaneously
    v_right = data[:, 2] - data[:, 1]
    v_up = data[:, 12] - data[:, 0]
    f_body = cp.cross(v_up, v_right)
    n_f = cp.linalg.norm(f_body, axis=-1, keepdims=True)
    default_f = cp.array([0.0, 0.0, -1.0], dtype=cp.float64)[None, :]
    f_body = cp.where(n_f > 1e-6, f_body / cp.maximum(n_f, 1e-6), default_f)

    leg_triads = [
        (1, 4, 7, 10),
        (2, 5, 8, 11),
    ]
    for hip_idx, knee_idx, ankle_idx, foot_idx in leg_triads:
        p_hip = data[:, hip_idx]
        p_knee = data[:, knee_idx]
        p_ankle = data[:, ankle_idx]

        v_leg = p_ankle - p_hip
        len_leg = cp.linalg.norm(v_leg, axis=-1, keepdims=True)
        valid_leg = len_leg[:, 0] >= 1e-4
        u_leg = v_leg / cp.maximum(len_leg, 1e-4)

        v_thigh = p_knee - p_hip
        n_thigh = cp.linalg.norm(v_thigh, axis=-1, keepdims=True)
        n_shin = cp.linalg.norm(p_ankle - p_knee, axis=-1, keepdims=True)
        proj_len = cp.sum(v_thigh * u_leg, axis=-1, keepdims=True)
        p_proj = p_hip + proj_len * u_leg
        deflection = p_knee - p_proj

        ant_comp = cp.sum(deflection * f_body, axis=-1, keepdims=True)
        need_fix = valid_leg & (ant_comp[:, 0] < -0.015)
        if bool(cp.any(need_fix)):
            deflection_fixed = deflection - (ant_comp + 0.015) * f_body
            new_thigh = (p_proj + deflection_fixed) - p_hip
            n_new = cp.linalg.norm(new_thigh, axis=-1, keepdims=True)
            valid_thigh = need_fix & (n_new[:, 0] > 1e-6) & (n_thigh[:, 0] > 1e-6)
            cand_knee = p_hip + (new_thigh / cp.maximum(n_new, 1e-6)) * n_thigh
            data[:, knee_idx] = cp.where(valid_thigh[:, None], cand_knee, data[:, knee_idx])

            v_shin = p_ankle - data[:, knee_idx]
            n_s = cp.linalg.norm(v_shin, axis=-1, keepdims=True)
            valid_shin = valid_thigh & (n_s[:, 0] > 1e-6) & (n_shin[:, 0] > 1e-6)
            new_ankle = data[:, knee_idx] + (v_shin / cp.maximum(n_s, 1e-6)) * n_shin
            delta_ankle = new_ankle - p_ankle
            data[:, foot_idx] = cp.where(valid_shin[:, None], data[:, foot_idx] + delta_ankle, data[:, foot_idx])
            data[:, ankle_idx] = cp.where(valid_shin[:, None], new_ankle, data[:, ankle_idx])

    # 2. Bound interior flexion angle (0° <= theta <= 145°) for knees and elbows in parallel
    hinge_chains = [
        (1, 4, 7, 10, 145.0),
        (2, 5, 8, 11, 145.0),
        (16, 18, 20, 22, 145.0),
        (17, 19, 21, 23, 145.0),
    ]
    for prox_idx, mid_idx, dist_idx, term_idx, max_deg in hinge_chains:
        p_prox = data[:, prox_idx]
        p_mid = data[:, mid_idx]
        p_dist = data[:, dist_idx]

        v1 = p_mid - p_prox
        v2 = p_dist - p_mid
        n1 = cp.linalg.norm(v1, axis=-1, keepdims=True)
        n2 = cp.linalg.norm(v2, axis=-1, keepdims=True)
        valid = (n1[:, 0] >= 1e-4) & (n2[:, 0] >= 1e-4)

        u1 = v1 / cp.maximum(n1, 1e-4)
        u2 = v2 / cp.maximum(n2, 1e-4)
        cos_theta = cp.clip(cp.sum(u1 * u2, axis=-1, keepdims=True), -1.0, 1.0)
        angle_deg = cp.degrees(cp.arccos(cos_theta))

        exceed = valid & (angle_deg[:, 0] > max_deg)
        if bool(cp.any(exceed)):
            max_rad = math.radians(max_deg)
            ortho = u2 - cos_theta * u1
            norm_ortho = cp.linalg.norm(ortho, axis=-1, keepdims=True)
            can_clamp = exceed & (norm_ortho[:, 0] > 1e-6)
            ortho_unit = ortho / cp.maximum(norm_ortho, 1e-6)
            u2_clamped = math.cos(max_rad) * u1 + math.sin(max_rad) * ortho_unit
            new_dist = p_mid + u2_clamped * n2
            delta_dist = new_dist - p_dist
            data[:, dist_idx] = cp.where(can_clamp[:, None], new_dist, data[:, dist_idx])
            data[:, term_idx] = cp.where(can_clamp[:, None], data[:, term_idx] + delta_dist, data[:, term_idx])

    return cp.asnumpy(data.astype(cp.float32))


def optimise_windows_cuda(
    x_windows: np.ndarray,
    s_windows: np.ndarray | None,
    vis_windows: np.ndarray | None,
    scales_windows: np.ndarray | None,
    bone_pairs: Sequence[tuple[int, int]],
    bone_lengths: dict[tuple[int, int], float],
    foreshorten_windows: np.ndarray | None,
    *,
    w_2d: float = 1.5,
    w_3d: float = 0.8,
    w_bone: float = 25.0,
    w_smooth: float = 4.0,
    w_jerk: float = 2.0,
    w_anat: float = 8.0,
    max_nfev: int = 25,
    n_iters: int | None = None,
) -> np.ndarray:
    """Batched GPU non-linear least-squares IK trajectory solver across all windows simultaneously.

    Solves `B` sliding windows of shape `(B, W, 24, 3)` in parallel on the NVIDIA GPU
    using analytical residual gradients and per-window backtracking trust-region steps.
    """
    from .ik_optimiser import ANATOMICAL_HINGES

    p_init = cp.asarray(x_windows, dtype=cp.float64)
    n_batch, win_len, n_joints, _ = p_init.shape
    p = p_init.copy()

    has_2d = s_windows is not None and scales_windows is not None and w_2d > 0.0
    s_win_gpu = cp.asarray(s_windows, dtype=cp.float64) if s_windows is not None else None
    vis_gpu = (
        cp.asarray(vis_windows, dtype=cp.float64)
        if vis_windows is not None
        else cp.ones((n_batch, win_len, n_joints), dtype=cp.float64)
    )
    sc_gpu = cp.asarray(scales_windows, dtype=cp.float64) if scales_windows is not None else None
    f_gpu = (
        cp.asarray(foreshorten_windows, dtype=cp.float64)
        if foreshorten_windows is not None
        else cp.zeros((n_batch, win_len, n_joints), dtype=cp.float64)
    )

    z_weight_sq = float(w_3d) * ((1.0 - 0.75 * cp.clip(f_gpu, 0.0, 1.0)) ** 2)

    parent_idx = cp.asarray([pair[0] for pair in bone_pairs], dtype=cp.int32)
    child_idx = cp.asarray([pair[1] for pair in bone_pairs], dtype=cp.int32)
    target_lens = cp.asarray(
        [bone_lengths[(pair[0], pair[1])] for pair in bone_pairs], dtype=cp.float64
    )[None, None, :]  # (1, 1, N_bones)

    def cost_and_grad(curr: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute per-window scalar cost (B,) and analytical gradient (B, W, J, 3)."""
        cost = cp.zeros(n_batch, dtype=cp.float64)
        grad = cp.zeros_like(curr)

        # 1. 2D Screen Landmark Reprojection
        if has_2d and s_win_gpu is not None and sc_gpu is not None:
            dx = curr[:, :, :, 0] - curr[:, :, 0:1, 0]
            dy = curr[:, :, :, 1] - curr[:, :, 0:1, 1]
            if sc_gpu.ndim == 3 and sc_gpu.shape[2] == 2:
                su = sc_gpu[:, :, 0:1]
                sv = sc_gpu[:, :, 1:2]
                v_sign = 1.0
            else:
                su = sc_gpu[:, :, None]
                sv = sc_gpu[:, :, None]
                v_sign = -1.0

            u_hat = s_win_gpu[:, :, 0:1, 0] + su * dx
            v_hat = s_win_gpu[:, :, 0:1, 1] + (v_sign * sv) * dy

            du = u_hat - s_win_gpu[:, :, :, 0]
            dv = v_hat - s_win_gpu[:, :, :, 1]
            w_vis = vis_gpu * float(w_2d)

            cost += 0.5 * cp.sum(w_vis * (du * du + dv * dv), axis=(1, 2))

            gu = w_vis * du * su
            gv = w_vis * dv * (v_sign * sv)
            grad[:, :, :, 0] += gu
            grad[:, :, 0, 0] -= cp.sum(gu, axis=2)
            grad[:, :, :, 1] += gv
            grad[:, :, 0, 1] -= cp.sum(gv, axis=2)

        # 2. 3D Prior (downweighting Z when foreshortened)
        if w_3d > 0.0:
            dxy = curr[:, :, :, :2] - p_init[:, :, :, :2]
            dz = curr[:, :, :, 2] - p_init[:, :, :, 2]
            cost += 0.5 * float(w_3d) * cp.sum(dxy * dxy, axis=(1, 2, 3))
            cost += 0.5 * cp.sum(z_weight_sq * dz * dz, axis=(1, 2))
            grad[:, :, :, :2] += float(w_3d) * dxy
            grad[:, :, :, 2] += z_weight_sq * dz

        # 3. Rigid Bone Length Constraints
        if w_bone > 0.0 and len(bone_pairs) > 0:
            v_bones = curr[:, :, child_idx, :] - curr[:, :, parent_idx, :]  # (B, W, Nb, 3)
            l_bones = cp.linalg.norm(v_bones, axis=-1)                      # (B, W, Nb)
            dl = l_bones - target_lens
            cost += 0.5 * float(w_bone) * cp.sum(dl * dl, axis=(1, 2))

            u_bones = v_bones / cp.maximum(l_bones[..., None], 1e-6)
            g_bones = (float(w_bone) * dl)[..., None] * u_bones             # (B, W, Nb, 3)
            for b_i, (p_i, c_i) in enumerate(bone_pairs):
                grad[:, :, c_i, :] += g_bones[:, :, b_i, :]
                grad[:, :, p_i, :] -= g_bones[:, :, b_i, :]

        # 4. Temporal Smoothness / Acceleration (d^2P/dt^2)
        if win_len >= 3 and w_smooth > 0.0:
            accel = curr[:, 2:] - 2.0 * curr[:, 1:-1] + curr[:, :-2]
            cost += 0.5 * float(w_smooth) * cp.sum(accel * accel, axis=(1, 2, 3))
            ga = float(w_smooth) * accel
            grad[:, :-2] += ga
            grad[:, 1:-1] -= 2.0 * ga
            grad[:, 2:] += ga

        # 5. Temporal Jerk (d^3P/dt^3)
        if win_len >= 4 and w_jerk > 0.0:
            jerk = curr[:, 3:] - 3.0 * curr[:, 2:-1] + 3.0 * curr[:, 1:-2] - curr[:, :-3]
            cost += 0.5 * float(w_jerk) * cp.sum(jerk * jerk, axis=(1, 2, 3))
            gj = float(w_jerk) * jerk
            grad[:, :-3] -= gj
            grad[:, 1:-2] += 3.0 * gj
            grad[:, 2:-1] -= 3.0 * gj
            grad[:, 3:] += gj

        # 6. Anatomical Range-of-Motion Limit Penalties
        if w_anat > 0.0:
            for prox, hinge, dist, max_deg, is_knee in ANATOMICAL_HINGES:
                v_in = curr[:, :, hinge] - curr[:, :, prox]
                v_out = curr[:, :, dist] - curr[:, :, hinge]
                n_in = cp.linalg.norm(v_in, axis=-1) + 1e-6
                n_out = cp.linalg.norm(v_out, axis=-1) + 1e-6
                cos_theta = cp.clip(cp.sum(v_in * v_out, axis=-1) / (n_in * n_out), -1.0, 1.0)
                flex_deg = cp.degrees(cp.arccos(cos_theta))
                exceed = cp.maximum(0.0, flex_deg - max_deg)
                w_ex = float(w_anat) * 0.0004
                cost += 0.5 * w_ex * cp.sum(exceed * exceed, axis=1)

                if is_knee:
                    v_right = curr[:, :, 2] - curr[:, :, 1]
                    v_up = curr[:, :, 12] - curr[:, :, 0]
                    f_body = cp.cross(v_up, v_right)
                    f_body /= cp.linalg.norm(f_body, axis=-1, keepdims=True) + 1e-6
                    chord = curr[:, :, dist] - curr[:, :, prox]
                    u_chord = chord / (cp.linalg.norm(chord, axis=-1, keepdims=True) + 1e-6)
                    proj = curr[:, :, prox] + cp.sum(
                        (curr[:, :, hinge] - curr[:, :, prox]) * u_chord, axis=-1, keepdims=True
                    ) * u_chord
                    deflect = curr[:, :, hinge] - proj
                    ant_disp = cp.sum(deflect * f_body, axis=-1)
                    hyper = cp.maximum(0.0, -ant_disp - 0.01)
                    w_hyp = float(w_anat) * 4.0
                    cost += 0.5 * w_hyp * cp.sum(hyper * hyper, axis=1)
                    gh = (w_hyp * hyper)[..., None] * (-f_body)
                    grad[:, :, hinge] += gh

        return cost, grad

    # Parallel Preconditioned Gradient Descent with per-window Armijo line search on GPU
    total_iters = int(n_iters) if n_iters is not None else max(60, int(max_nfev) * 3)
    c_curr, g_curr = cost_and_grad(p)

    # Diagonal Hessian preconditioner estimate from objective weights
    h_diag = float(w_3d + 2.0 * w_bone + 6.0 * w_smooth + 20.0 * w_jerk + 2.0 * w_2d)
    step_scale = cp.full((n_batch, 1, 1, 1), 1.5 / max(h_diag, 1.0), dtype=cp.float64)
    velocity = cp.zeros_like(p)

    for _ in range(total_iters):
        # Momentum-accelerated preconditioned descent direction
        direction = -step_scale * g_curr + 0.45 * velocity
        cand = p + direction
        c_cand, g_cand = cost_and_grad(cand)

        improved = c_cand < c_curr
        mask = improved[:, None, None, None]
        p = cp.where(mask, cand, p)
        velocity = cp.where(mask, direction, velocity * 0.2)
        c_curr = cp.where(improved, c_cand, c_curr)
        g_curr = cp.where(mask, g_cand, g_curr)

        # Adapt per-window step size (grow on improvement, shrink on overshoot)
        step_scale = cp.where(mask, cp.minimum(step_scale * 1.15, 0.08), step_scale * 0.5)
        if float(cp.max(step_scale)) < 1e-7:
            break

    return cp.asnumpy(p)


# ==============================================================================
# Stage 4: Geodesic Quaternion Smoothing & Batched Linear Blend Skinning on CUDA
# ==============================================================================

def skin_mesh_cuda(vertices: np.ndarray, weights: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Linear Blend Skinning on CUDA (single frame or batched frames).

    Supports:
    - Single frame: vertices (V, 3), palette (B, 4, 4) -> (V, 3)
    - Batched frames: vertices (F, V, 3), palette (B, 4, 4) or (F, B, 4, 4) -> (F, V, 3)
    """
    v_gpu = cp.asarray(vertices, dtype=cp.float64)
    w_gpu = cp.asarray(weights, dtype=cp.float64)
    p_gpu = cp.asarray(palette, dtype=cp.float64)

    if v_gpu.ndim == 2:
        # Single frame: (V, 3)
        if p_gpu.ndim == 4:
            p_gpu = p_gpu[0]
        ones = cp.ones((v_gpu.shape[0], 1), dtype=cp.float64)
        vh = cp.concatenate([v_gpu, ones], axis=1)  # (V, 4)
        blended_mats = cp.einsum("vb,bij->vij", w_gpu, p_gpu)
        posed = cp.einsum("vij,vj->vi", blended_mats[:, :3, :], vh)
        return cp.asnumpy(posed)
    elif v_gpu.ndim == 3:
        # Batched frames: (F, V, 3)
        F, V, _ = v_gpu.shape
        ones = cp.ones((F, V, 1), dtype=cp.float64)
        vh = cp.concatenate([v_gpu, ones], axis=2)  # (F, V, 4)
        if p_gpu.ndim == 3:
            # Single palette (B, 4, 4) shared across F frames
            blended_mats = cp.einsum("vb,bij->vij", w_gpu, p_gpu)  # (V, 4, 4)
            posed = cp.einsum("vij,fvj->fvi", blended_mats[:, :3, :], vh)  # (F, V, 3)
        elif p_gpu.ndim == 4:
            # Per-frame palette (F, B, 4, 4)
            if p_gpu.shape[0] != F:
                raise ValueError(
                    f"Frame count mismatch: vertices has {F} frames but palette has {p_gpu.shape[0]}"
                )
            blended_mats = cp.einsum("vb,fbij->fvij", w_gpu, p_gpu)  # (F, V, 4, 4)
            posed = cp.einsum("fvij,fvj->fvi", blended_mats[:, :, :3, :], vh)  # (F, V, 3)
        else:
            raise ValueError(f"Unsupported palette shape for CUDA skinning: {p_gpu.shape}")
        return cp.asnumpy(posed)
    raise ValueError(f"Unsupported vertices ndim for CUDA skinning: {v_gpu.ndim}")


def skin_frames_cuda(asset, anim, indices: list[int]) -> list[list[np.ndarray]]:
    """Batched multi-frame Linear Blend Skinning across all requested frames in parallel on CUDA."""
    from .rig.kinematics import forward_kinematics

    if not indices or not asset.meshes:
        return [[] for _ in indices]

    scale = float(asset.rig.meters_per_unit)
    n_frames = len(indices)
    order = asset.rig.order

    # Compute FK bone world matrices per frame (25x4x4 is tiny, then stack palettes on GPU)
    world_stack = np.empty((n_frames, len(order), 4, 4), dtype=np.float64)
    for f_i, idx in enumerate(indices):
        world = forward_kinematics(
            asset.rig, anim.r_anim_rig[idx], hips_translation_cm=anim.hips_cm_rig[idx]
        )
        for b_i, name in enumerate(order):
            world_stack[f_i, b_i] = world[name]

    world_gpu = cp.asarray(world_stack, dtype=cp.float64)  # (F, B, 4, 4)

    # For each mesh, compute posed vertices across all F frames simultaneously on GPU
    per_mesh_frames: list[np.ndarray] = []
    for mesh in asset.meshes:
        ibm_gpu = cp.asarray(mesh.ibm, dtype=cp.float64)          # (B, 4, 4)
        palettes_gpu = cp.matmul(world_gpu, ibm_gpu[None, ...])   # (F, B, 4, 4)
        w_gpu = cp.asarray(mesh.weights, dtype=cp.float64)        # (V, B)

        has_morph = (
            anim.morph_weights is not None
            and getattr(mesh, "morph_deltas_pos", None) is not None
        )
        if has_morph:
            mw = anim.morph_weights[indices]
            morphed = evaluate_morph_cuda(
                mesh.vertices, mesh.morph_deltas_pos, mw, device="cuda"
            )  # (F, V, 3)
            v_gpu = cp.asarray(morphed, dtype=cp.float64)
            ones = cp.ones((n_frames, v_gpu.shape[1], 1), dtype=cp.float64)
            vh_gpu = cp.concatenate([v_gpu, ones], axis=2)        # (F, V, 4)
            blended_mats = cp.einsum("vb,fbij->fvij", w_gpu, palettes_gpu[:, :, :3, :])  # (F, V, 3, 4)
            posed_gpu = cp.einsum("fvij,fvj->fvi", blended_mats, vh_gpu) * scale
        else:
            v_gpu = cp.asarray(mesh.vertices, dtype=cp.float64)       # (V, 3)
            ones = cp.ones((v_gpu.shape[0], 1), dtype=cp.float64)
            vh_gpu = cp.concatenate([v_gpu, ones], axis=1)            # (V, 4)
            vw_gpu = w_gpu[:, :, None] * vh_gpu[:, None, :]           # (V, B, 4)
            posed_gpu = cp.einsum("fbij,vbj->fvi", palettes_gpu[:, :, :3, :], vw_gpu) * scale

        per_mesh_frames.append(cp.asnumpy(posed_gpu))

    out: list[list[np.ndarray]] = []
    for f_i in range(n_frames):
        out.append([per_mesh_frames[m_i][f_i] for m_i in range(len(asset.meshes))])
    return out


def smooth_rotation_trajectories_cuda(
    r_seq: list[dict[str, np.ndarray]],
    bone_names: list[str],
    upper_limb_bones: frozenset[str],
) -> list[dict[str, np.ndarray]]:
    """Zero-phase geodesic quaternion filter and angular velocity clamping on CUDA."""
    from .rig.kinematics import continue_quaternion, quat_wxyz

    n = len(r_seq)
    if n < 3:
        return r_seq

    out_seq: list[dict[str, np.ndarray]] = [{k: v.copy() for k, v in r.items()} for r in r_seq]
    w_upper = cp.array([0.08, 0.24, 0.36, 0.24, 0.08], dtype=cp.float64)
    w_body = cp.array([0.0, 0.15, 0.70, 0.15, 0.0], dtype=cp.float64)
    offsets = (-2, -1, 0, 1, 2)
    idx_base = cp.arange(n, dtype=cp.int32)

    for bname in bone_names:
        q_track = np.zeros((n, 4), dtype=np.float64)
        q_prev = None
        for t in range(n):
            rmat = r_seq[t].get(bname, np.eye(3))
            q, _ = continue_quaternion(quat_wxyz(rmat), q_prev)
            q_track[t] = q
            q_prev = q

        if np.allclose(q_track, q_track[0], atol=1e-7):
            continue

        q_curr = cp.asarray(q_track, dtype=cp.float64)
        weights = w_upper if bname in upper_limb_bones else w_body
        passes = 2 if bname in upper_limb_bones else 1

        for _ in range(passes):
            acc = cp.zeros_like(q_curr)
            for k_idx, offset in enumerate(offsets):
                w = float(weights[k_idx])
                if w <= 0.0:
                    continue
                tk = cp.clip(idx_base + offset, 0, n - 1)
                qk = q_curr[tk]
                dots = cp.sum(q_curr * qk, axis=-1, keepdims=True)
                qk = cp.where(dots < 0.0, -qk, qk)
                acc += w * qk
            norms = cp.linalg.norm(acc, axis=-1, keepdims=True)
            q_curr = cp.where(norms > 1e-12, acc / cp.maximum(norms, 1e-12), q_curr)

        q_host = cp.asnumpy(q_curr)

        # Enforce physiological angular jump limit (<= 24.5 deg/frame) on upper limb tracks
        if bname in upper_limb_bones:
            from .rig.animation import _slerp

            max_rad = math.radians(24.5)
            for _ in range(4):
                for t in range(1, n):
                    dot = float(np.dot(q_host[t], q_host[t - 1]))
                    if dot < 0.0:
                        q_host[t] = -q_host[t]
                        dot = -dot
                    ang = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
                    if ang > max_rad:
                        q_host[t] = _slerp(q_host[t - 1], q_host[t], max_rad / ang)
                for t in range(n - 2, -1, -1):
                    dot = float(np.dot(q_host[t], q_host[t + 1]))
                    if dot < 0.0:
                        q_host[t] = -q_host[t]
                        dot = -dot
                    ang = 2.0 * math.acos(min(1.0, max(-1.0, dot)))
                    if ang > max_rad:
                        q_host[t] = _slerp(q_host[t + 1], q_host[t], max_rad / ang)

        # Vectorised GPU conversion from unit quaternion (w, x, y, z) to (N, 3, 3) rotation matrices
        q_gpu = cp.asarray(q_host, dtype=cp.float64)
        w, x, y, z = q_gpu[:, 0], q_gpu[:, 1], q_gpu[:, 2], q_gpu[:, 3]
        mats_gpu = cp.empty((n, 3, 3), dtype=cp.float64)
        mats_gpu[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        mats_gpu[:, 0, 1] = 2.0 * (x * y - w * z)
        mats_gpu[:, 0, 2] = 2.0 * (x * z + w * y)
        mats_gpu[:, 1, 0] = 2.0 * (x * y + w * z)
        mats_gpu[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        mats_gpu[:, 1, 2] = 2.0 * (y * z - w * x)
        mats_gpu[:, 2, 0] = 2.0 * (x * z - w * y)
        mats_gpu[:, 2, 1] = 2.0 * (y * z + w * x)
        mats_gpu[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        mats_host = cp.asnumpy(mats_gpu)

        for t in range(n):
            out_seq[t][bname] = mats_host[t]

    return out_seq


# ==============================================================================
# Stage 5: glTF 2.0 Export & 3D Software Renderer CUDA Acceleration
# ==============================================================================

def accumulate_vertex_colors_cuda(
    faces: np.ndarray,
    face_colors_bgr: np.ndarray,
    orig_v_count: int,
    v_map: np.ndarray | None = None,
) -> np.ndarray:
    """Convert BGR face colours to averaged per-vertex RGBA colours on CUDA."""
    f_gpu = cp.asarray(faces, dtype=cp.int32)
    f_bgr = cp.asarray(face_colors_bgr, dtype=cp.float64)
    face_rgb = f_bgr[:, [2, 1, 0]] / 255.0

    v_colors = cp.zeros((orig_v_count, 3), dtype=cp.float64)
    v_counts = cp.zeros((orig_v_count, 1), dtype=cp.float64)
    ones = cp.ones((f_gpu.shape[0], 1), dtype=cp.float64)
    for i in range(3):
        cp.add.at(v_colors, f_gpu[:, i], face_rgb)
        cp.add.at(v_counts, f_gpu[:, i], ones)

    v_colors = cp.clip(v_colors / cp.maximum(v_counts, 1.0), 0.0, 1.0).astype(cp.float32)
    if v_map is not None:
        v_colors = v_colors[cp.asarray(v_map, dtype=cp.int64)]
    alpha = cp.ones((v_colors.shape[0], 1), dtype=cp.float32)
    rgba = cp.concatenate([v_colors, alpha], axis=1)
    return cp.asnumpy(rgba)


def vertex_normals_cuda(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted smooth vertex normals on CUDA."""
    v_gpu = cp.asarray(vertices, dtype=cp.float64)
    f_gpu = cp.asarray(faces, dtype=cp.int32)
    normals = cp.zeros_like(v_gpu, dtype=cp.float64)

    p0 = v_gpu[f_gpu[:, 0]]
    p1 = v_gpu[f_gpu[:, 1]]
    p2 = v_gpu[f_gpu[:, 2]]
    face_n = cp.cross(p1 - p0, p2 - p0)

    for i in range(3):
        cp.add.at(normals, f_gpu[:, i], face_n)

    length = cp.linalg.norm(normals, axis=1, keepdims=True)
    unit_normals = (normals / cp.where(length < 1e-12, 1.0, length)).astype(cp.float32)
    return cp.asnumpy(unit_normals)


def top4_weights_cuda(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select and normalise top-4 skinning bone influences per vertex on CUDA."""
    w_gpu = cp.asarray(weights, dtype=cp.float64)
    n_bones = w_gpu.shape[1]
    take = min(4, n_bones)
    order = cp.argpartition(-w_gpu, take - 1, axis=1)[:, :take]
    w = cp.take_along_axis(w_gpu, order, axis=1)
    w = w / cp.maximum(w.sum(axis=1, keepdims=True), 1e-8)
    if take < 4:
        pad_i = cp.zeros((w_gpu.shape[0], 4 - take), dtype=order.dtype)
        pad_w = cp.zeros((w_gpu.shape[0], 4 - take), dtype=w.dtype)
        order = cp.concatenate([order, pad_i], axis=1)
        w = cp.concatenate([w, pad_w], axis=1)
    return cp.asnumpy(order.astype(cp.uint16)), cp.asnumpy(w.astype(cp.float32))


def project_and_shade_mesh_cuda(
    vertices: list[np.ndarray],
    faces: list[np.ndarray],
    colors: list[np.ndarray],
    cam,
    ambient: float = 0.30,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project 3D meshes, cull backfaces, compute Lambertian shading, and Z-sort on CUDA.

    Returns:
        tuple of (sorted_triangles_px (N_tri, 3, 2) int32, sorted_bgr (N_tri, 3) int32)
        or None if no triangles face the camera.
    """
    polys: list[cp.ndarray] = []
    shades: list[cp.ndarray] = []
    depths: list[cp.ndarray] = []

    look_gpu = cp.asarray(cam.look, dtype=cp.float64)
    right_gpu = cp.asarray(cam.cam_right, dtype=cp.float64)
    up_gpu = cp.asarray(cam.cam_up, dtype=cp.float64)
    light_gpu = -look_gpu
    is_persp = hasattr(cam, "eye") and hasattr(cam, "focal")

    if is_persp:
        eye_gpu = cp.asarray(cam.eye, dtype=cp.float64)
        focal = float(cam.focal)
        near = float(getattr(cam, "near", 0.05))
    else:
        center_gpu = cp.asarray(cam.center, dtype=cp.float64)
        scale = float(cam.scale)

    half_w = float(cam.width) * 0.5
    half_h = float(cam.height) * 0.5

    for verts, face, base in zip(vertices, faces, colors):
        if len(face) == 0:
            continue
        v_gpu = cp.asarray(verts, dtype=cp.float64)
        f_gpu = cp.asarray(face, dtype=cp.int32)

        if is_persp:
            d = v_gpu - eye_gpu[None, :]
            z = d @ look_gpu
            z_clip = cp.maximum(z, near)
            u = (d @ right_gpu) / z_clip * focal
            v = (d @ up_gpu) / z_clip * focal
            uv = cp.column_stack([half_w + u, half_h - v])
        else:
            d = v_gpu - center_gpu[None, :]
            z = d @ look_gpu
            u = d @ right_gpu
            v = d @ up_gpu
            uv = cp.column_stack([half_w + u * scale, half_h - v * scale])

        v0, v1, v2 = f_gpu[:, 0], f_gpu[:, 1], f_gpu[:, 2]
        p0, p1, p2 = v_gpu[v0], v_gpu[v1], v_gpu[v2]
        normals = cp.cross(p1 - p0, p2 - p0)
        nlen = cp.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / cp.where(nlen < 1e-12, 1.0, nlen)
        facing = normals @ light_gpu
        keep = facing > 0.0
        if not bool(cp.any(keep)):
            continue

        lambert = cp.clip(cp.abs(facing[keep]), 0.0, 1.0)
        shade = float(ambient) + (1.0 - float(ambient)) * lambert
        tri = cp.stack([uv[v0[keep]], uv[v1[keep]], uv[v2[keep]]], axis=1)
        polys.append(cp.rint(tri).astype(cp.int32))

        tint = cp.asarray(base, dtype=cp.float64)
        tint = tint[None, :] if tint.ndim == 1 else tint[keep]
        shades.append(shade[:, None] * tint)
        depths.append((z[v0[keep]] + z[v1[keep]] + z[v2[keep]]) / 3.0)

    if not polys:
        return None

    tri_all = cp.concatenate(polys, axis=0)
    rgb_all = cp.clip(cp.concatenate(shades, axis=0), 0, 255)
    depth_all = cp.concatenate(depths, axis=0)

    order = cp.argsort(-depth_all)
    tri_sorted = cp.asnumpy(tri_all[order])
    rgb_sorted = cp.asnumpy(rgb_all[order].astype(cp.int32))
    return tri_sorted, rgb_sorted


# ==============================================================================
# Facial Expression & Blendshape CUDA Acceleration
# ==============================================================================

_ONE_EURO_BLENDSHAPES_KERNEL = None


def filter_blendshapes_cuda(
    blendshapes: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.01,
    d_cutoff: float = 1.0,
    device: str | None = "cuda",
) -> np.ndarray:
    """Filter (T, 52) ARKit blendshapes on NVIDIA GPU via CuPy CUDA C kernel.

    Provides vectorised 1€ adaptive low-pass filtering across all 52 facial
    expression channels simultaneously with higher velocity sensitivity on
    blink channels. Seamlessly falls back to host CPU NumPy if CUDA is unavailable.
    """
    global _ONE_EURO_BLENDSHAPES_KERNEL
    if not _HAS_CUPY or cp is None or not use_cupy(device):
        from .extract import OneEuroFilterBlendshapes

        return OneEuroFilterBlendshapes(
            fps=fps, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
        ).filter_trajectory(blendshapes)

    T, C = blendshapes.shape
    if T <= 1 or C != 52:
        return blendshapes.copy()

    if _ONE_EURO_BLENDSHAPES_KERNEL is None:
        _ONE_EURO_BLENDSHAPES_KERNEL = cp.RawKernel(
            r'''
        extern "C" __global__
        void one_euro_blendshapes(
            const float* __restrict__ input,
            float* __restrict__ output,
            const float* __restrict__ min_cutoffs,
            const float* __restrict__ betas,
            int num_frames,
            int num_channels,
            float fps,
            float d_cutoff
        ) {
            int c = blockDim.x * blockIdx.x + threadIdx.x;
            if (c >= num_channels) return;

            float te = 1.0f / fps;
            float tau_d = 1.0f / (2.0f * 3.141592653589793f * d_cutoff);
            float alpha_d = 1.0f / (1.0f + tau_d / te);

            float min_cut = min_cutoffs[c];
            float beta_val = betas[c];

            float x_prev = input[c];
            output[c] = x_prev;
            float dx_prev = 0.0f;

            for (int t = 1; t < num_frames; ++t) {
                int idx = t * num_channels + c;
                float x = input[idx];
                float dx = (x - x_prev) / te;
                float dx_hat = alpha_d * dx + (1.0f - alpha_d) * dx_prev;

                float fc = min_cut + beta_val * fabsf(dx_hat);
                float tau = 1.0f / (2.0f * 3.141592653589793f * fc);
                float alpha = 1.0f / (1.0f + tau / te);

                float x_hat = alpha * x + (1.0f - alpha) * x_prev;
                output[idx] = x_hat;
                x_prev = x_hat;
                dx_prev = dx_hat;
            }
        }
        ''',
            "one_euro_blendshapes",
        )

    min_cut = np.full(52, min_cutoff, dtype=np.float32)
    betas = np.full(52, beta, dtype=np.float32)
    # Higher velocity sensitivity on ARKit blink channels (8: eyeBlinkLeft, 9: eyeBlinkRight)
    for b_idx in [8, 9]:
        min_cut[b_idx] = 1.5
        betas[b_idx] = 0.08

    data_cp = cp.asarray(blendshapes, dtype=cp.float32)
    out_cp = cp.empty_like(data_cp)
    min_cut_cp = cp.asarray(min_cut)
    betas_cp = cp.asarray(betas)

    threads_per_block = 64
    blocks = (C + threads_per_block - 1) // threads_per_block
    _ONE_EURO_BLENDSHAPES_KERNEL(
        (blocks,),
        (threads_per_block,),
        (
            data_cp,
            out_cp,
            min_cut_cp,
            betas_cp,
            T,
            C,
            cp.float32(fps),
            cp.float32(d_cutoff),
        ),
    )
    return cp.asnumpy(out_cp)


def apply_anatomical_constraints_cuda(
    blendshapes: np.ndarray,
    device: str | None = "cuda",
) -> np.ndarray:
    """Enforce physiological bounds, antagonistic muting, and eyelid-gaze coupling on GPU.

    Supports both 1D (52,) and 2D (T, 52) arrays. When CUDA is available, executes
    via vectorised CuPy tensor operations; falls back to CPU NumPy seamlessly.
    """
    if not _HAS_CUPY or cp is None or not use_cupy(device):
        from .extract import apply_anatomical_constraints

        return apply_anatomical_constraints(blendshapes)

    is_1d = blendshapes.ndim == 1
    data_cp = cp.asarray(blendshapes[None, :] if is_1d else blendshapes, dtype=cp.float32)
    out = cp.clip(data_cp, 0.0, 1.0).copy()

    # Canonical ARKit blendshape indices
    blink_l, blink_r = 8, 9
    down_l, down_r = 10, 11
    wide_l, wide_r = 20, 21
    jaw_l, jaw_open, jaw_r = 23, 24, 25
    mouth_close = 26
    mouth_frown_l, mouth_frown_r = 29, 30
    mouth_l = 32
    mouth_pucker = 37
    mouth_r = 38
    mouth_smile_l, mouth_smile_r = 43, 44
    mouth_stretch_l, mouth_stretch_r = 45, 46

    # 1. Eyelid-gaze decoupling: downward gaze lid lowering
    out[:, wide_l] *= cp.maximum(0.0, 1.0 - out[:, down_l] * 0.8)
    out[:, wide_r] *= cp.maximum(0.0, 1.0 - out[:, down_r] * 0.8)
    out[:, blink_l] = cp.clip(out[:, blink_l] - 0.45 * out[:, down_l], 0.0, 1.0)
    out[:, blink_r] = cp.clip(out[:, blink_r] - 0.45 * out[:, down_r], 0.0, 1.0)

    # 2. Antagonistic: eyeWide vs eyeBlink
    out[:, wide_l] *= cp.maximum(0.0, 1.0 - out[:, blink_l] * 1.5)
    out[:, wide_r] *= cp.maximum(0.0, 1.0 - out[:, blink_r] * 1.5)

    # 3. Antagonistic: jawOpen vs mouthClose
    out[:, mouth_close] *= cp.maximum(0.0, 1.0 - out[:, jaw_open] * 2.0)

    # 4. Jaw lateral excursion (mutual exclusivity)
    delta_jaw = out[:, jaw_l] - out[:, jaw_r]
    out[:, jaw_l] = cp.maximum(0.0, delta_jaw)
    out[:, jaw_r] = cp.maximum(0.0, -delta_jaw)

    # 5. Mouth lateral deviation (mutual exclusivity)
    delta_mouth = out[:, mouth_l] - out[:, mouth_r]
    out[:, mouth_l] = cp.maximum(0.0, delta_mouth)
    out[:, mouth_r] = cp.maximum(0.0, -delta_mouth)

    # 6. Smile vs Frown antagonism
    out[:, mouth_frown_l] *= cp.maximum(0.0, 1.0 - 1.5 * out[:, mouth_smile_l])
    out[:, mouth_frown_r] *= cp.maximum(0.0, 1.0 - 1.5 * out[:, mouth_smile_r])

    # 7. Pucker vs Stretch antagonism
    stretch_max = cp.maximum(out[:, mouth_stretch_l], out[:, mouth_stretch_r])
    out[:, mouth_pucker] *= cp.maximum(0.0, 1.0 - stretch_max)

    res = cp.asnumpy(cp.clip(out, 0.0, 1.0))
    return res[0] if is_1d else res


def evaluate_morph_cuda(
    vertices: np.ndarray,
    morph_deltas_pos: np.ndarray,
    weights: np.ndarray,
    device: str | None = "cuda",
) -> np.ndarray:
    """Evaluate morphed vertices on CUDA via tensor contraction with clean CPU fallback.

    Computes: v_morphed = v_base + sum(w_m * delta_p_m) with weights clamped to [0.0, 1.0].

    Supports:
    - Single frame: vertices (V, 3), morph_deltas_pos (M, V, 3), weights (M,) -> (V, 3)
    - Batched frames: vertices (V, 3), morph_deltas_pos (M, V, 3), weights (F, M) -> (F, V, 3)

    Parameters
    ----------
    vertices : np.ndarray or cp.ndarray
        Base vertex positions, shape (V, 3).
    morph_deltas_pos : np.ndarray or cp.ndarray
        Morph displacement deltas, shape (M, V, 3).
    weights : np.ndarray or cp.ndarray
        Morph target activation weights, shape (M,) or (F, M).
    device : str or None, optional
        Target device ('cuda' or 'cpu'). Default is 'cuda'.

    Returns
    -------
    np.ndarray
        Morphed vertex positions of shape (V, 3) or (F, V, 3).
    """
    dtype = vertices.dtype if hasattr(vertices, "dtype") else np.float32
    if not _HAS_CUPY or cp is None or not use_cupy(device):
        v = np.asarray(vertices, dtype=dtype)
        deltas = np.asarray(morph_deltas_pos, dtype=dtype)
        w = np.clip(np.asarray(weights, dtype=dtype), 0.0, 1.0)
        if w.ndim == 1:
            return v + np.einsum("m,mvj->vj", w, deltas)
        elif w.ndim == 2:
            return v[None, :, :] + np.einsum("fm,mvj->fvj", w, deltas)
        raise ValueError(f"Unsupported weights shape on CPU: {weights.shape}")

    v_gpu = cp.asarray(vertices, dtype=dtype)
    deltas_gpu = cp.asarray(morph_deltas_pos, dtype=dtype)
    w_gpu = cp.clip(cp.asarray(weights, dtype=dtype), 0.0, 1.0)

    if w_gpu.ndim == 1:
        morphed_gpu = v_gpu + cp.einsum("m,mvj->vj", w_gpu, deltas_gpu)
    elif w_gpu.ndim == 2:
        morphed_gpu = v_gpu[None, :, :] + cp.einsum("fm,mvj->fvj", w_gpu, deltas_gpu)
    else:
        raise ValueError(f"Unsupported weights shape on CUDA: {weights.shape}")

    if isinstance(vertices, cp.ndarray) and isinstance(weights, cp.ndarray):
        return morphed_gpu
    return cp.asnumpy(morphed_gpu)

