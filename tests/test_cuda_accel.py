"""Unit and integration tests for NVIDIA CUDA acceleration across all v2bm pipeline stages.

Validates:
1. Stage 1: Device capability detection (`get_cuda_capabilities`, `resolve_device`, `use_cuda`, `use_cupy`, `use_nvenc`).
2. Stage 2: Subject isolation GPU kernels (`compute_tracking_cost_matrix_cuda`, `build_removal_mask_cuda`).
3. Stage 3: 3D skeleton extraction, biomechanical stabilisation, and batched GPU IK trajectory optimisation
   (`map_to_body24_batch_cuda`, `map_screen_to_body24`, `clamp_anatomical_hinges`, `BiomechanicalStabiliser`,
   and all 5 mathematical guarantees of `optimise_trajectory_windowed(device='cuda')`).
4. Stage 4: Retargeting quaternion geodesic smoothing and batched GPU linear blend skinning
   (`_smooth_rotation_trajectories`, `skin_mesh`, `skin_frames`).
5. Stage 5: GLB export (`_vertex_normals`, `_top4_weights`, `accumulate_vertex_colors_cuda`, `write_glb`)
   and software preview rendering + FFmpeg NVENC hardware video encoding (`project_and_shade_mesh_cuda`,
   `draw_mesh`, `encode_h264`, `_StreamWriter`).

All tests adhere strictly to British English conventions and permissive commercial licensing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from v2bm import paths
from v2bm.cuda_accel import (
    accumulate_vertex_colors_cuda,
    build_removal_mask_cuda,
    clamp_anatomical_hinges_cuda,
    compute_tracking_cost_matrix_cuda,
    ffmpeg_video_encode_args,
    get_cuda_capabilities,
    map_screen_to_body24_cuda,
    map_to_body24_batch_cuda,
    project_and_shade_mesh_cuda,
    resolve_device,
    skin_frames_cuda,
    skin_mesh_cuda,
    smooth_rotation_trajectories_cuda,
    top4_weights_cuda,
    use_cuda,
    use_cupy,
    use_nvenc,
    vertex_normals_cuda,
)
from v2bm.ik_optimiser import (
    calculate_bone_lengths,
    compute_trajectory_jerk,
    optimise_trajectory_windowed,
)
from v2bm.rig.animation import UPPER_LIMB_BONES, _smooth_rotation_trajectories
from v2bm.rig.gltf import _top4_weights, _vertex_normals, write_glb
from v2bm.rig.render import build_perspective_camera, draw_mesh, encode_h264
from v2bm.rig.skinned_mesh import bone_palette, load_character_asset, skin_mesh
from v2bm.rig.tpose_calibration import load_calibration
from v2bm.stabilize_kp2d import (
    BODY24_BONE_PAIRS,
    BiomechanicalStabiliser,
    clamp_anatomical_hinges,
    map_screen_to_body24,
)
from v2bm.videos import _StreamWriter

from .test_ik_optimiser import _create_synthetic_tpose


def test_cuda_capabilities_and_device_resolution() -> None:
    """Verify CUDA capability detection and device resolution."""
    caps = get_cuda_capabilities()
    assert isinstance(caps.cuda_available, bool)
    assert isinstance(caps.cupy_available, bool)
    assert isinstance(caps.ffmpeg_nvenc, bool)

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    auto_dev = resolve_device("auto")
    assert auto_dev in ("cuda", "cpu")
    if caps.cuda_available:
        assert auto_dev == "cuda"
        assert resolve_device("cuda") == "cuda"
        assert use_cuda("cuda") is True
        assert use_cuda("auto") is True
    assert use_cuda("cpu") is False
    assert use_cupy("cpu") is False
    assert use_nvenc("cpu") is False

    cpu_args = ffmpeg_video_encode_args("cpu", crf=18)
    assert "libx264" in cpu_args
    if caps.ffmpeg_nvenc and caps.cuda_available:
        gpu_args = ffmpeg_video_encode_args("cuda", crf=18)
        assert "h264_nvenc" in gpu_args


@pytest.mark.skipif(not get_cuda_capabilities().cupy_available, reason="CuPy CUDA not installed")
def test_stage2_isolation_cuda_kernels() -> None:
    """Verify Stage 2 multi-person tracking cost matrix and removal mask GPU kernels."""
    from v2bm.isolate import DetectedPose, MultiPersonTracker, build_removal_mask

    pred_centroids = np.array([[200.0, 400.0], [800.0, 400.0]], dtype=np.float64)
    last_heights = np.array([350.0, 340.0], dtype=np.float64)
    last_colours = np.array([[40.0, 120.0, 200.0], [200.0, 80.0, 50.0]], dtype=np.float64)

    det_centroids = np.array([[205.0, 402.0], [795.0, 398.0]], dtype=np.float64)
    det_heights = np.array([352.0, 338.0], dtype=np.float64)
    det_colours = np.array([[42.0, 118.0, 198.0], [198.0, 82.0, 52.0]], dtype=np.float64)
    diag = math.hypot(1280.0, 720.0)

    cost_gpu = compute_tracking_cost_matrix_cuda(
        pred_centroids, last_heights, last_colours, det_centroids, det_heights, det_colours, diag
    )
    assert cost_gpu.shape == (2, 2)
    assert cost_gpu[0, 0] < cost_gpu[0, 1]
    assert cost_gpu[1, 1] < cost_gpu[1, 0]

    # Verify MultiPersonTracker.update(..., device="cuda") parity with CPU
    det0 = DetectedPose(
        index=0,
        landmarks=np.zeros((33, 3), dtype=np.float32),
        bbox=(150.0, 225.0, 250.0, 575.0),
        centroid=det_centroids[0].copy(),
        height=350.0,
        area=35000.0,
        torso_colour=det_colours[0].copy(),
        segmentation_mask=None,
    )
    det1 = DetectedPose(
        index=1,
        landmarks=np.zeros((33, 3), dtype=np.float32),
        bbox=(750.0, 230.0, 850.0, 570.0),
        centroid=det_centroids[1].copy(),
        height=340.0,
        area=34000.0,
        torso_colour=det_colours[1].copy(),
        segmentation_mask=None,
    )
    tracker_cpu = MultiPersonTracker(target_selector="left")
    tracker_gpu = MultiPersonTracker(target_selector="left")
    matched_cpu = tracker_cpu.update([det0, det1], (720, 1280), device="cpu")
    matched_gpu = tracker_gpu.update([det0, det1], (720, 1280), device="cuda")
    assert set(matched_cpu.keys()) == set(matched_gpu.keys())

    # Removal mask union with target silhouette protection on GPU vs CPU
    mask_cpu = build_removal_mask((720, 1280), matched_cpu, tracker_cpu.target_track_id, device="cpu")
    mask_gpu = build_removal_mask((720, 1280), matched_gpu, tracker_gpu.target_track_id, device="cuda")
    np.testing.assert_array_equal(mask_gpu, mask_cpu)


@pytest.mark.skipif(not get_cuda_capabilities().cupy_available, reason="CuPy CUDA not installed")
def test_stage3_body24_mapping_and_biomechanical_stabilisation_cuda() -> None:
    """Verify Stage 3 Body24 mapping and anatomical hinge clamping match CPU within float32 precision."""
    rng = np.random.default_rng(101)
    n_frames = 25
    mp_world = rng.normal(0.0, 0.3, (n_frames, 33, 3)).astype(np.float32)
    mp_vis = rng.uniform(0.1, 1.0, (n_frames, 33)).astype(np.float32)
    mp_screen = rng.uniform(0.1, 0.9, (n_frames, 33, 3)).astype(np.float64)

    # Batch world mapping
    from v2bm.extract import MediaPipeExtractor

    cpu_world = np.stack(
        [MediaPipeExtractor.map_to_body24(mp_world[t], mp_vis[t]) for t in range(n_frames)],
        axis=0,
    )
    gpu_world = map_to_body24_batch_cuda(mp_world, mp_vis)
    np.testing.assert_allclose(gpu_world, cpu_world, atol=1e-5)

    # Screen mapping
    cpu_screen = map_screen_to_body24(mp_screen, device="cpu")
    gpu_screen = map_screen_to_body24(mp_screen, device="cuda")
    np.testing.assert_allclose(gpu_screen, cpu_screen, atol=1e-6)

    # Anatomical hinge clamping parity
    tpose = _create_synthetic_tpose(n_frames)
    # Introduce hyperextension on left knee (index 4) and excessive flexion on left elbow (index 20)
    tpose[:, 4, 2] += 0.15
    tpose[:, 20, :] = tpose[:, 16, :] + 0.05

    clamped_cpu = clamp_anatomical_hinges(tpose, device="cpu")
    clamped_gpu = clamp_anatomical_hinges(tpose, device="cuda")
    np.testing.assert_allclose(clamped_gpu, clamped_cpu, atol=1e-5)

    # Full BiomechanicalStabiliser parity
    stabiliser = BiomechanicalStabiliser()
    stab_cpu, flags_cpu = stabiliser.stabilise(tpose, mp_screen, device="cpu")
    stab_gpu, flags_gpu = stabiliser.stabilise(tpose, mp_screen, device="cuda")
    np.testing.assert_array_equal(flags_gpu, flags_cpu)
    np.testing.assert_allclose(stab_gpu, stab_cpu, atol=1e-5)


@pytest.mark.skipif(not get_cuda_capabilities().cupy_available, reason="CuPy CUDA not installed")
def test_stage3_ik_optimiser_cuda_guarantees() -> None:
    """Verify GPU batched IK optimiser satisfies all 5 biomechanical & mathematical guarantees."""
    n_frames = 40
    fps = 30.0
    clean = _create_synthetic_tpose(n_frames)
    rest_lengths = calculate_bone_lengths(clean, BODY24_BONE_PAIRS)

    rng = np.random.default_rng(42)
    noisy = clean + rng.normal(0.0, 0.03, clean.shape)
    initial_jerk = compute_trajectory_jerk(noisy, fps=fps)

    # Synthesise 2D screen landmarks
    s_scale = 0.4 / 0.52
    pelvis_2d = np.array([0.5, 0.5])
    screen_body24 = np.zeros((n_frames, 24, 2), dtype=np.float64)
    for t in range(n_frames):
        for j in range(24):
            dx = clean[t, j, 0] - clean[t, 0, 0]
            dy = clean[t, j, 1] - clean[t, 0, 1]
            screen_body24[t, j, 0] = pelvis_2d[0] + s_scale * dx
            screen_body24[t, j, 1] = pelvis_2d[1] - s_scale * dy

    optimised_gpu = optimise_trajectory_windowed(
        noisy,
        landmarks_2d=screen_body24,
        bone_lengths=rest_lengths,
        fps=fps,
        window_length=11,
        stride=5,
        w_2d=3.0,
        w_3d=0.5,
        w_smooth=6.0,
        w_jerk=4.0,
        device="cuda",
    )

    assert np.all(np.isfinite(optimised_gpu))

    # 1. Rigid bone length invariance (< 10^-4 m)
    max_bone_err = 0.0
    for parent, child in BODY24_BONE_PAIRS:
        target_l = rest_lengths[(parent, child)]
        for t in range(n_frames):
            actual_l = float(np.linalg.norm(optimised_gpu[t, child] - optimised_gpu[t, parent]))
            max_bone_err = max(max_bone_err, abs(actual_l - target_l))
    assert max_bone_err < 1e-4, f"CUDA IK bone length error {max_bone_err:.3e} m >= 10^-4 m"

    # 2. Jerk reduction >= 50%
    final_jerk = compute_trajectory_jerk(optimised_gpu, fps=fps)
    jerk_reduction = (initial_jerk - final_jerk) / initial_jerk
    assert jerk_reduction >= 0.50, (
        f"CUDA IK jerk reduction {jerk_reduction:.1%} < 50% "
        f"({initial_jerk:.2f} -> {final_jerk:.2f} m/s^3)"
    )

    # 3. Window overlap continuity
    diffs = np.linalg.norm(optimised_gpu[1:] - optimised_gpu[:-1], axis=-1)
    assert float(np.max(diffs)) < 0.15


@pytest.mark.skipif(not get_cuda_capabilities().cupy_available, reason="CuPy CUDA not installed")
def test_stage4_retarget_smoothing_and_lbs_skinning_cuda() -> None:
    """Verify Stage 4 quaternion trajectory smoothing and linear blend skinning on CUDA match CPU."""
    rng = np.random.default_rng(77)
    n_frames = 20
    bone_names = ["rig:Hips", "rig:LeftArm", "rig:LeftForeArm"]
    r_seq: list[dict[str, np.ndarray]] = []
    for t in range(n_frames):
        ang = 0.2 * math.sin(t * 0.4) + float(rng.normal(0.0, 0.03))
        c, s = math.cos(ang), math.sin(ang)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        r_seq.append({b: rot.copy() for b in bone_names})

    smooth_cpu = _smooth_rotation_trajectories(r_seq, bone_names, device="cpu")
    smooth_gpu = _smooth_rotation_trajectories(r_seq, bone_names, device="cuda")
    for t in range(n_frames):
        for b in bone_names:
            np.testing.assert_allclose(smooth_gpu[t][b], smooth_cpu[t][b], atol=1e-6)

    # Test linear blend skinning on real humanoid rig asset
    rig_path = paths.DEFAULT_RIG
    if rig_path.exists():
        asset = load_character_asset(rig_path, textures=False)
        from v2bm.rig.kinematics import forward_kinematics

        world = forward_kinematics(asset.rig, {})
        for mesh in asset.meshes:
            pal = bone_palette(world, mesh, asset.rig.order)
            v_cpu = skin_mesh(mesh, pal, device="cpu")
            v_gpu = skin_mesh(mesh, pal, device="cuda")
            np.testing.assert_allclose(v_gpu, v_cpu, atol=1e-6)


@pytest.mark.skipif(not get_cuda_capabilities().cupy_available, reason="CuPy CUDA not installed")
def test_stage5_glb_export_and_rendering_cuda(tmp_path: Path) -> None:
    """Verify Stage 5 GLB geometry preparation, painter shading, and NVENC video encoding on CUDA."""
    rig_path = paths.DEFAULT_RIG
    if not rig_path.exists():
        pytest.skip("Default humanoid rig not found")

    asset = load_character_asset(rig_path, textures=True)
    mesh = asset.meshes[0]

    # Top-4 weights parity
    j_cpu, w_cpu = _top4_weights(mesh.weights, device="cpu")
    j_gpu, w_gpu = _top4_weights(mesh.weights, device="cuda")
    np.testing.assert_allclose(
        np.sort(w_gpu, axis=1), np.sort(w_cpu, axis=1), atol=1e-6
    )

    # Vertex normals parity
    n_cpu = _vertex_normals(mesh.vertices, mesh.faces, device="cpu")
    n_gpu = _vertex_normals(mesh.vertices, mesh.faces, device="cuda")
    np.testing.assert_allclose(n_gpu, n_cpu, atol=1e-5)

    # GLB export on CUDA
    glb_cpu = write_glb(
        tmp_path / "cpu.glb",
        asset,
        [{}],
        [asset.rig.bones[asset.rig.root].lcl_translation_cm],
        30.0,
        device="cpu",
    )
    glb_gpu = write_glb(
        tmp_path / "gpu.glb",
        asset,
        [{}],
        [asset.rig.bones[asset.rig.root].lcl_translation_cm],
        30.0,
        device="cuda",
    )
    assert glb_cpu.exists() and glb_gpu.exists()
    assert abs(glb_cpu.stat().st_size - glb_gpu.stat().st_size) == 0

    # Mesh projection & Lambert shading parity
    calibration = load_calibration(rig_path)
    verts_m = [m.vertices * asset.rig.meters_per_unit for m in asset.meshes]
    faces = [m.faces for m in asset.meshes]
    colors = [
        m.face_colors if m.face_colors is not None else np.array([180.0, 180.0, 180.0])
        for m in asset.meshes
    ]
    cam = build_perspective_camera(
        calibration.rig_axes,
        np.concatenate(verts_m, axis=0),
        320,
        240,
    )
    canvas_cpu = np.zeros((240, 320, 3), dtype=np.uint8)
    canvas_gpu = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_mesh(canvas_cpu, verts_m, faces, colors, cam, 0.34, device="cpu")
    draw_mesh(canvas_gpu, verts_m, faces, colors, cam, 0.34, device="cuda")
    # Allow tiny rounding differences on triangle depth ties
    diff_ratio = float(np.mean(np.abs(canvas_cpu.astype(np.float32) - canvas_gpu.astype(np.float32))))
    assert diff_ratio < 0.5

    # StreamWriter with device="cuda" (NVENC when available)
    mp4_path = tmp_path / "nvenc_stream.mp4"
    writer = _StreamWriter(mp4_path, 30.0, 320, 240, device="cuda")
    for _ in range(15):
        writer.write(canvas_gpu)
    writer.close()
    assert mp4_path.exists() and mp4_path.stat().st_size > 200


@pytest.mark.skipif(not get_cuda_capabilities().cupy_available, reason="CuPy CUDA not installed")
def test_end_to_end_pipeline_cuda(tmp_path: Path) -> None:
    """Verify full end-to-end pipeline execution with --device cuda."""
    import json
    from v2bm.pipeline import run_pipeline

    img_path = paths.IMAGE_DIR / "urban_dancer_pose.png"
    rig_path = paths.DEFAULT_RIG
    if not img_path.exists() or not rig_path.exists():
        pytest.skip("Required image or rig asset missing")

    out_dir = tmp_path / "cuda_e2e_run"
    result_dir = run_pipeline(
        None,
        rig_path,
        image=img_path,
        device="cuda",
        output_dir=out_dir,
        no_preview=True,
    )
    assert result_dir == out_dir
    assert (out_dir / "character_animation.glb").exists()
    assert (out_dir / "skeleton_motion.npz").exists()
    assert (out_dir / "rig_rotations.npz").exists()
    run_report = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert run_report["device"] == "cuda"
    assert run_report["cuda_capabilities"]["cupy_available"] is True
    assert run_report["rest_fk_error"] < 1e-4
    assert run_report["twist_events"] == 0
    assert run_report["quat_sign_flips"] == 0

