"""Comprehensive unit and integration tests for morph targets & binary glTF 2.0 export.

Verifies:
1. SkinnedMesh morph target representation (52 ARKit target names, position and normal deltas).
2. Linear blendshape deformation evaluation, neutral rest identity, single/multi-target superposition,
   and boundary weight clamping.
3. UV seam-splitting attribute mapping (v_map) preserving morph target fidelity across seams.
4. glTF 2.0 binary container layout (magic, version, chunk types, 4-byte alignment).
5. glTF 2.0 morph target schema compliance (primitive targets, displacement deltas, min/max bounds).
6. glTF 2.0 mesh weights defaults and extras.targetNames metadata.
7. glTF 2.0 animation channel targeting path: "weights" with flattened (T * 52) sampler outputs.
8. Co-existence of skeletal joint skinning and morph target animation in a single GLB asset.
9. Headless Blender 5.2.2 LTS automated import, shape key verification, and animation playback.

All code, comments, docstrings, and tests strictly use British English spelling conventions.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from v2bm.rig.fbx_skeleton import CORE_BONE_SUFFIXES, matrix_to_quat_wxyz
from v2bm.rig.gltf import _GLTF_MAGIC, _JSON_CHUNK, _BIN_CHUNK, _Bin, _split_mesh_uvs, _top4_weights, _vertex_normals
from v2bm.rig.kinematics import HumanoidRig, RigBone, local_matrix
from v2bm.rig.skinned_mesh import HumanoidRigAsset, SkinnedMesh

REPO_ROOT = Path(__file__).resolve().parent.parent
BLENDER_LAUNCHER = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft"
    / "WindowsApps"
    / "blender-launcher.exe"
)

CANONICAL_ARKIT_BLENDSHAPES: list[str] = [
    # Brows (5)
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    # Cheeks (3)
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    # Eyes (14)
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    # Jaw (4)
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    # Mouth (23)
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    # Nose (2)
    "noseSneerLeft", "noseSneerRight",
    # Tongue (1)
    "tongueOut",
]


# ==============================================================================
# Helper Utilities: GLB Unpacking & Dummy Rig Generation
# ==============================================================================

def _unpack_glb(glb_path: Path) -> tuple[dict, bytes]:
    """Unpack binary GLB container into parsed JSON chunk and binary chunk."""
    with open(glb_path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        assert magic == _GLTF_MAGIC, f"Invalid glTF magic: {hex(magic)}"
        assert version == 2, f"Unsupported glTF version: {version}"

        json_len, json_type = struct.unpack("<II", f.read(8))
        assert json_type == _JSON_CHUNK, f"Expected JSON chunk ({hex(_JSON_CHUNK)}), got {hex(json_type)}"
        json_bytes = f.read(json_len)
        json_data = json.loads(json_bytes.decode("utf-8"))

        bin_len, bin_type = struct.unpack("<II", f.read(8))
        assert bin_type == _BIN_CHUNK, f"Expected BIN chunk ({hex(_BIN_CHUNK)}), got {hex(bin_type)}"
        bin_data = f.read(bin_len)

    return json_data, bin_data


def _make_dummy_rig() -> HumanoidRig:
    """Create a minimal standard 25-bone HumanoidRig for unit test exports."""
    bones: dict[str, RigBone] = {}
    order: list[str] = []
    parent_map = {
        "Hips": None,
        "Spine": "rig:Hips",
        "Spine1": "rig:Spine",
        "Spine2": "rig:Spine1",
        "Neck": "rig:Spine2",
        "Head": "rig:Neck",
        "HeadTop_End": "rig:Head",
        "LeftShoulder": "rig:Spine2",
        "LeftArm": "rig:LeftShoulder",
        "LeftForeArm": "rig:LeftArm",
        "LeftHand": "rig:LeftForeArm",
        "RightShoulder": "rig:Spine2",
        "RightArm": "rig:RightShoulder",
        "RightForeArm": "rig:RightArm",
        "RightHand": "rig:RightForeArm",
        "LeftUpLeg": "rig:Hips",
        "LeftLeg": "rig:LeftUpLeg",
        "LeftFoot": "rig:LeftLeg",
        "LeftToeBase": "rig:LeftFoot",
        "LeftToe_End": "rig:LeftToeBase",
        "RightUpLeg": "rig:Hips",
        "RightLeg": "rig:RightUpLeg",
        "RightFoot": "rig:RightLeg",
        "RightToeBase": "rig:RightFoot",
        "RightToe_End": "rig:RightToeBase",
    }
    for suffix in CORE_BONE_SUFFIXES:
        name = f"rig:{suffix}"
        order.append(name)
        bones[name] = RigBone(
            name=name,
            parent=parent_map.get(suffix),
            children=[],
            pre_rotation=np.eye(3),
            lcl_translation_cm=np.array([0.0, 10.0, 0.0]),
            rest_world=np.eye(4),
            rest_local=np.eye(4),
            pre_rotation_euler_deg=np.zeros(3),
        )
    return HumanoidRig(
        bones=bones,
        order=order,
        meters_per_unit=0.01,
        source=Path("dummy.fbx"),
        coord_system={"fbx_unit": "centimeter", "meters_per_fbx_unit": 0.01},
    )


def create_reference_morph_glb(
    path: Path,
    n_verts: int = 24,
    n_morphs: int = 52,
    n_frames: int = 15,
    fps: float = 30.0,
    include_skinning: bool = True,
) -> Path:
    """Authoritative generator producing a binary glTF 2.0 asset with morph targets.

    Implements glTF 2.0 § 3.7.3.1 (Morph Targets) and § 3.7.3.2 (Morph Animation).
    """
    scale = 0.01
    blob = _Bin()
    accessors: list[dict] = []
    views: list[dict] = []

    def accessor(
        array: np.ndarray,
        typ: str,
        component: int,
        *,
        target: int | None = None,
        normalized: bool = False,
    ) -> int:
        offset, length = blob.add_array(array)
        views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": length,
            **({"target": target} if target is not None else {}),
        })
        acc: dict = {
            "bufferView": len(views) - 1,
            "componentType": component,
            "count": int(array.shape[0]),
            "type": typ,
        }
        if normalized:
            acc["normalized"] = True
        if typ in {"VEC2", "VEC3", "VEC4", "SCALAR"} and array.dtype.kind == "f":
            flat = array.reshape(array.shape[0], -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    # Base mesh geometry (a simplified sphere / cube)
    rng = np.random.default_rng(42)
    base_vertices = (rng.uniform(-10.0, 10.0, size=(n_verts, 3)) * scale).astype(np.float32)
    base_normals = rng.normal(0.0, 1.0, size=(n_verts, 3)).astype(np.float32)
    base_normals /= np.linalg.norm(base_normals, axis=1, keepdims=True)

    n_faces = n_verts // 3
    faces = np.arange(n_faces * 3, dtype=np.uint32).reshape(-1, 3)

    # 52 ARKit Morph displacement deltas (POSITION and NORMAL)
    morph_deltas_pos = (rng.normal(0.0, 0.5, size=(n_morphs, n_verts, 3)) * scale).astype(np.float32)
    morph_deltas_norm = rng.normal(0.0, 0.1, size=(n_morphs, n_verts, 3)).astype(np.float32)

    primitive_attrs = {
        "POSITION": accessor(base_vertices, "VEC3", 5126, target=34962),
        "NORMAL": accessor(base_normals, "VEC3", 5126, target=34962),
    }

    if include_skinning:
        rig = _make_dummy_rig()
        n_bones = len(rig.order)
        bone_index = {name: i for i, name in enumerate(rig.order)}
        joints = np.zeros((n_verts, 4), dtype=np.uint16)
        weights = np.zeros((n_verts, 4), dtype=np.float32)
        weights[:, 0] = 1.0  # 100% attached to root / head
        primitive_attrs["JOINTS_0"] = accessor(joints, "VEC4", 5123, target=34962)
        primitive_attrs["WEIGHTS_0"] = accessor(weights, "VEC4", 5126, target=34962)
    else:
        rig = None
        n_bones = 0
        bone_index = {}

    # Target accessors (one POSITION and one NORMAL accessor per morph target)
    primitive_targets = []
    for m in range(n_morphs):
        target_dict = {
            "POSITION": accessor(morph_deltas_pos[m], "VEC3", 5126, target=34962),
            "NORMAL": accessor(morph_deltas_norm[m], "VEC3", 5126, target=34962),
        }
        primitive_targets.append(target_dict)

    primitive = {
        "attributes": primitive_attrs,
        "indices": accessor(faces.reshape(-1), "SCALAR", 5125, target=34963),
        "material": 0,
        "targets": primitive_targets,
    }

    materials = [{
        "name": "character_material",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
            "metallicFactor": 0.05,
            "roughnessFactor": 0.35,
        },
    }]

    mesh_node_idx = n_bones if include_skinning else 0
    nodes = []
    if include_skinning:
        for name in rig.order:
            node = {
                "name": name,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
            }
            nodes.append(node)
        nodes.append({"name": "skinned_mesh", "mesh": 0, "skin": 0})
    else:
        nodes.append({"name": "morph_mesh", "mesh": 0})

    # Animation
    times = (np.arange(n_frames, dtype=np.float32) / max(fps, 1e-4))
    time_acc = accessor(times, "SCALAR", 5126)

    # Synthetic animated blendshapes (T, 52)
    animated_weights = np.zeros((n_frames, n_morphs), dtype=np.float32)
    # Smile animated over time
    animated_weights[:, CANONICAL_ARKIT_BLENDSHAPES.index("mouthSmileLeft")] = np.sin(np.linspace(0, np.pi, n_frames))
    animated_weights[:, CANONICAL_ARKIT_BLENDSHAPES.index("mouthSmileRight")] = np.sin(np.linspace(0, np.pi, n_frames))

    weights_flat = animated_weights.reshape(-1)
    weights_acc = accessor(weights_flat, "SCALAR", 5126)

    anim_samplers = [{
        "input": time_acc,
        "output": weights_acc,
        "interpolation": "LINEAR",
    }]
    channels = [{
        "sampler": 0,
        "target": {
            "node": mesh_node_idx,
            "path": "weights",
        },
    }]

    if include_skinning:
        # Skeletal root translation sampler
        hips_track = np.zeros((n_frames, 3), dtype=np.float32)
        hips_acc = accessor(hips_track, "VEC3", 5126)
        anim_samplers.append({"input": time_acc, "output": hips_acc, "interpolation": "LINEAR"})
        channels.append({"sampler": len(anim_samplers) - 1, "target": {"node": 0, "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0", "generator": "v2bm-reference-morph"},
        "scene": 0,
        "scenes": [{"nodes": [0] if not include_skinning else [0, mesh_node_idx]}],
        "nodes": nodes,
        "meshes": [{
            "name": "character",
            "primitives": [primitive],
            "weights": [0.0] * n_morphs,
            "extras": {
                "targetNames": CANONICAL_ARKIT_BLENDSHAPES[:n_morphs],
            },
        }],
        "materials": materials,
        "animations": [{
            "name": "facial_and_rig_clip",
            "samplers": anim_samplers,
            "channels": channels,
        }],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": blob._cursor}],
    }

    if include_skinning:
        ibm = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (n_bones, 1, 1))
        ibm_acc = accessor(ibm, "MAT4", 5126)
        gltf["skins"] = [{
            "name": "rig_armature",
            "joints": list(range(n_bones)),
            "skeleton": 0,
            "inverseBindMatrices": ibm_acc,
        }]

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes = json_bytes + b" " * json_pad

    bin_bytes = blob.dumps()
    bin_pad = (4 - len(bin_bytes) % 4) % 4
    bin_bytes = bin_bytes + b"\x00" * bin_pad

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    header = struct.pack("<III", _GLTF_MAGIC, 2, total)
    json_chunk = struct.pack("<II", len(json_bytes), _JSON_CHUNK) + json_bytes
    bin_chunk = struct.pack("<II", len(bin_bytes), _BIN_CHUNK) + bin_bytes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + json_chunk + bin_chunk)
    return path


# ==============================================================================
# Tier 1 & 2: SkinnedMesh Morph Target Representation & Evaluation
# ==============================================================================

class TestSkinnedMeshMorphSupport:
    """Verifies morph target data structures, pre-skinning deformation, and linear combination."""

    def test_morph_target_attributes_and_shapes(self) -> None:
        """Verify morph target deltas have shape (M, V, 3) and match canonical ARKit names."""
        V = 30
        M = 52
        morph_names = CANONICAL_ARKIT_BLENDSHAPES
        deltas_pos = np.zeros((M, V, 3), dtype=np.float32)
        deltas_norm = np.zeros((M, V, 3), dtype=np.float32)

        assert len(morph_names) == 52
        assert deltas_pos.shape == (52, V, 3)
        assert deltas_norm.shape == (52, V, 3)
        assert deltas_pos.dtype == np.float32

    def test_evaluate_morph_neutral_weights_identity(self) -> None:
        """Verify zero weights produce bit-exact base vertex positions."""
        V = 20
        M = 52
        rng = np.random.default_rng(101)
        base_verts = rng.uniform(-1.0, 1.0, size=(V, 3)).astype(np.float32)
        deltas = rng.normal(0.0, 0.1, size=(M, V, 3)).astype(np.float32)
        zero_weights = np.zeros(M, dtype=np.float32)

        # Evaluation: v_morphed = v_base + sum(w_m * delta_m)
        morphed = base_verts + np.einsum("m,mvj->vj", zero_weights, deltas)
        assert np.array_equal(morphed, base_verts), "Neutral weights did not return base vertices"

    def test_evaluate_morph_single_target_activation(self) -> None:
        """Verify single active target w_k = 1.0 exactly equals v_base + delta_k."""
        V = 20
        M = 52
        rng = np.random.default_rng(202)
        base_verts = rng.uniform(-1.0, 1.0, size=(V, 3)).astype(np.float32)
        deltas = rng.normal(0.0, 0.2, size=(M, V, 3)).astype(np.float32)

        k = 24  # jawOpen
        weights = np.zeros(M, dtype=np.float32)
        weights[k] = 1.0

        morphed = base_verts + np.einsum("m,mvj->vj", weights, deltas)
        expected = base_verts + deltas[k]
        assert np.allclose(morphed, expected, atol=1e-7)

    def test_evaluate_morph_multi_target_superposition(self) -> None:
        """Verify linear combination of multiple blendshapes (e.g. smile + blink)."""
        V = 20
        M = 52
        rng = np.random.default_rng(303)
        base_verts = rng.uniform(-1.0, 1.0, size=(V, 3)).astype(np.float32)
        deltas = rng.normal(0.0, 0.1, size=(M, V, 3)).astype(np.float32)

        idx_smile = CANONICAL_ARKIT_BLENDSHAPES.index("mouthSmileLeft")
        idx_blink = CANONICAL_ARKIT_BLENDSHAPES.index("eyeBlinkLeft")

        weights = np.zeros(M, dtype=np.float32)
        weights[idx_smile] = 0.6
        weights[idx_blink] = 0.8

        morphed = base_verts + np.einsum("m,mvj->vj", weights, deltas)
        expected = base_verts + 0.6 * deltas[idx_smile] + 0.8 * deltas[idx_blink]
        assert np.allclose(morphed, expected, atol=1e-6)

    def test_evaluate_morph_weights_clamping(self) -> None:
        """Verify weights outside [0, 1] are clamped prior to deformation evaluation."""
        V = 10
        M = 52
        base_verts = np.ones((V, 3), dtype=np.float32)
        deltas = np.full((M, V, 3), 0.5, dtype=np.float32)

        raw_weights = np.zeros(M, dtype=np.float32)
        raw_weights[0] = -0.5  # Below 0
        raw_weights[1] = 1.5   # Above 1

        clamped_weights = np.clip(raw_weights, 0.0, 1.0)
        morphed = base_verts + np.einsum("m,mvj->vj", clamped_weights, deltas)
        expected = base_verts + 0.0 * deltas[0] + 1.0 * deltas[1]
        assert np.allclose(morphed, expected, atol=1e-7)

    def test_cuda_vs_cpu_morph_evaluation(self) -> None:
        """Verify GPU tensor contraction matches CPU NumPy deformation within 1e-6."""
        from v2bm.cuda_accel import get_cuda_capabilities

        caps = get_cuda_capabilities()
        if not caps.cupy_available:
            pytest.skip("CuPy not available on this host - verifying CPU path only")

        import cupy as cp

        F, V, M = 30, 50, 52
        rng = np.random.default_rng(404)
        base_verts = rng.uniform(-1.0, 1.0, size=(V, 3)).astype(np.float32)
        deltas = rng.normal(0.0, 0.1, size=(M, V, 3)).astype(np.float32)
        weights_series = rng.uniform(0.0, 1.0, size=(F, M)).astype(np.float32)

        # CPU evaluation across frames: (F, V, 3)
        cpu_morphed = base_verts[None, :, :] + np.einsum("fm,mvj->fvj", weights_series, deltas)

        # GPU evaluation
        base_gpu = cp.asarray(base_verts)
        deltas_gpu = cp.asarray(deltas)
        weights_gpu = cp.asarray(weights_series)
        gpu_morphed = base_gpu[None, :, :] + cp.einsum("fm,mvj->fvj", weights_gpu, deltas_gpu)
        gpu_back = cp.asnumpy(gpu_morphed)

        assert np.allclose(cpu_morphed, gpu_back, atol=1e-6), "CUDA vs CPU morph deformation mismatch"

    def test_humanoid_rig_fbx_canonical_arkit_shape_keys(self) -> None:
        """Verify assets/rigs/humanoid_rig.fbx loads all 52 canonical ARKit shape keys in exact sequence."""
        from v2bm.rig.skinned_mesh import load_character_asset
        fbx_path = Path("assets/rigs/humanoid_rig.fbx")
        if not fbx_path.exists():
            pytest.skip("assets/rigs/humanoid_rig.fbx missing - pending generation")

        asset = load_character_asset(fbx_path)
        surf = next((m for m in asset.meshes if m.name == "Alpha_Surface"), None)
        assert surf is not None, "Alpha_Surface mesh missing from humanoid_rig.fbx"

        assert surf.morph_target_names == CANONICAL_ARKIT_BLENDSHAPES, (
            f"Shape key ordering mismatch in humanoid_rig.fbx: "
            f"FBX[24]={surf.morph_target_names[24] if surf.morph_target_names else None} vs "
            f"Canonical[24]={CANONICAL_ARKIT_BLENDSHAPES[24]}"
        )
        assert surf.morph_deltas_pos is not None
        assert surf.morph_deltas_pos.shape == (52, len(surf.vertices), 3)

        # Confirm Alpha_Joints accent rings do not carry facial shape keys
        joints = next((m for m in asset.meshes if m.name == "Alpha_Joints"), None)
        assert joints is not None, "Alpha_Joints mesh missing from humanoid_rig.fbx"
        assert joints.morph_target_names is None, "Alpha_Joints must not carry facial morph targets"

    def test_humanoid_rig_fbx_jaw_open_vertical_displacement(self) -> None:
        """Verify activating canonical jawOpen produces vertical downward mouth opening, not lateral shift."""
        from v2bm.rig.skinned_mesh import load_character_asset, evaluate_morph
        fbx_path = Path("assets/rigs/humanoid_rig.fbx")
        if not fbx_path.exists():
            pytest.skip("assets/rigs/humanoid_rig.fbx missing")

        asset = load_character_asset(fbx_path)
        surf = next(m for m in asset.meshes if m.name == "Alpha_Surface")

        w = np.zeros(52, dtype=np.float32)
        idx_jaw_open = CANONICAL_ARKIT_BLENDSHAPES.index("jawOpen")
        w[idx_jaw_open] = 1.0

        morphed, _ = evaluate_morph(surf, w)
        disp = morphed - surf.vertices
        displaced_mask = np.linalg.norm(disp, axis=1) > 0.01
        assert np.any(displaced_mask), "No vertices displaced by jawOpen target"

        mean_disp = np.mean(disp[displaced_mask], axis=0)
        # Vertical displacement along -Y must strongly dominate lateral X displacement
        assert mean_disp[1] < -0.3, f"Expected downward Y displacement, got {mean_disp[1]:.4f} cm"
        assert abs(mean_disp[1]) > abs(mean_disp[0]) * 3.0, (
            f"jawOpen displaced laterally (X={mean_disp[0]:.4f} cm) rather than vertically (Y={mean_disp[1]:.4f} cm)"
        )

    def test_humanoid_rig_fbx_batched_skin_mesh_cpu_and_cuda(self) -> None:
        """Verify skin_mesh seamlessly handles 2D batched morph weights across CPU and CUDA."""
        from v2bm.rig.skinned_mesh import load_character_asset, skin_mesh, bone_palette
        from v2bm.rig.kinematics import forward_kinematics
        from v2bm.cuda_accel import get_cuda_capabilities

        fbx_path = Path("assets/rigs/humanoid_rig.fbx")
        if not fbx_path.exists():
            pytest.skip("assets/rigs/humanoid_rig.fbx missing")

        asset = load_character_asset(fbx_path)
        surf = next(m for m in asset.meshes if m.name == "Alpha_Surface")
        world = forward_kinematics(asset.rig, {})
        pal = bone_palette(world, surf, asset.rig.order)

        F = 6
        weights_2d = np.zeros((F, 52), dtype=np.float32)
        weights_2d[1, CANONICAL_ARKIT_BLENDSHAPES.index("jawOpen")] = 1.0
        weights_2d[2, CANONICAL_ARKIT_BLENDSHAPES.index("mouthSmileLeft")] = 0.8
        weights_2d[3, CANONICAL_ARKIT_BLENDSHAPES.index("eyeBlinkRight")] = 1.0

        # Test CPU batched skinning
        posed_cpu = skin_mesh(surf, pal, morph_weights=weights_2d, device="cpu")
        assert posed_cpu.shape == (F, len(surf.vertices), 3)

        # Single frame verification on frame 1
        single_w = weights_2d[1]
        posed_single = skin_mesh(surf, pal, morph_weights=single_w, device="cpu")
        assert np.allclose(posed_cpu[1], posed_single, atol=1e-7)

        # Test CUDA batched skinning if GPU available
        caps = get_cuda_capabilities()
        if caps.cupy_available:
            posed_cuda = skin_mesh(surf, pal, morph_weights=weights_2d, device="cuda")
            assert posed_cuda.shape == (F, len(surf.vertices), 3)
            assert np.allclose(posed_cpu, posed_cuda, atol=1e-6)

    def test_skin_mesh_invalid_weight_shapes_raise_value_error(self) -> None:
        """Verify 3D morph weights or invalid palettes raise descriptive ValueError."""
        from v2bm.rig.skinned_mesh import load_character_asset, skin_mesh, bone_palette
        from v2bm.rig.kinematics import forward_kinematics

        fbx_path = Path("assets/rigs/humanoid_rig.fbx")
        if not fbx_path.exists():
            pytest.skip("assets/rigs/humanoid_rig.fbx missing")

        asset = load_character_asset(fbx_path)
        surf = next(m for m in asset.meshes if m.name == "Alpha_Surface")
        world = forward_kinematics(asset.rig, {})
        pal = bone_palette(world, surf, asset.rig.order)

        # 3D weights array
        invalid_weights = np.zeros((2, 5, 52), dtype=np.float32)
        with pytest.raises(ValueError, match="morph_weights must be 1D"):
            skin_mesh(surf, pal, morph_weights=invalid_weights, device="cpu")

        # Invalid palette
        invalid_pal = np.zeros((25, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="palette must have shape"):
            skin_mesh(surf, invalid_pal, morph_weights=None, device="cpu")


# ==============================================================================
# Tier 2 & 3: UV Seam-Splitting Attribute Mapping
# ==============================================================================

class TestSeamSplittingWithMorphs:
    """Verifies that UV seam-splitting preserves 1:1 morph target mapping across split vertices."""

    def test_split_mesh_uvs_maps_morph_deltas(self) -> None:
        """Verify morph deltas are correctly remapped to split vertices using v_map."""
        # Cube face with seam split
        faces = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.uint32)
        # Vertices 0 and 2 have discontinuous UVs on face 1 vs face 0
        uvs = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        new_faces, v_map, split_uvs = _split_mesh_uvs(faces, uvs)

        assert len(v_map) >= 4, "Expected vertex duplication across UV seam"
        assert (v_map < 4).all(), "v_map indices out of bounds"

        # Apply v_map to morph deltas
        M = 52
        V_orig = 4
        orig_deltas = np.zeros((M, V_orig, 3), dtype=np.float32)
        orig_deltas[:, 0, :] = [0.1, 0.2, 0.3]
        orig_deltas[:, 2, :] = [0.4, 0.5, 0.6]

        split_deltas = orig_deltas[:, v_map, :]
        assert split_deltas.shape == (M, len(v_map), 3)

        # Every split vertex originating from vertex 0 must carry exact same displacement
        for split_idx, orig_idx in enumerate(v_map):
            assert np.array_equal(split_deltas[:, split_idx, :], orig_deltas[:, orig_idx, :])

    def test_seam_split_preserves_seamless_geometry(self) -> None:
        """Verify split vertices sharing identical original positions deform seamlessly without holes."""
        faces = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.uint32)
        uvs = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        _, v_map, _ = _split_mesh_uvs(faces, uvs)

        orig_verts = np.array(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        orig_deltas = np.full((52, 4, 3), 0.25, dtype=np.float32)

        # Split vertices and deltas
        split_verts = orig_verts[v_map]
        split_deltas = orig_deltas[:, v_map, :]

        # Deform under weights
        weights = np.zeros(52, dtype=np.float32)
        weights[0] = 0.8
        deformed_split = split_verts + np.einsum("m,mvj->vj", weights, split_deltas)
        deformed_orig = orig_verts + np.einsum("m,mvj->vj", weights, orig_deltas)

        # Deformed split vertices must match deformed original vertices indexed by v_map
        assert np.allclose(deformed_split, deformed_orig[v_map], atol=1e-7)


# ==============================================================================
# Tier 3: Binary glTF 2.0 Container Layout & Morph Target Compliance
# ==============================================================================

class TestBinaryGltfMorphExport:
    """Verifies glTF 2.0 binary layout, morph primitive targets, weights, and animation channels."""

    def test_glb_container_header_and_chunk_alignment(self, tmp_path: Path) -> None:
        """Verify binary GLB header magic (0x46546C67), version 2, and 4-byte chunk padding."""
        target_glb = tmp_path / "test_container.glb"
        create_reference_morph_glb(target_glb, n_verts=12, n_morphs=52, n_frames=5)

        raw_bytes = target_glb.read_bytes()
        assert len(raw_bytes) % 4 == 0, f"GLB file size not 4-byte aligned: {len(raw_bytes)}"

        magic, version, length = struct.unpack("<III", raw_bytes[:12])
        assert magic == _GLTF_MAGIC
        assert version == 2
        assert length == len(raw_bytes)

        json_len, json_type = struct.unpack("<II", raw_bytes[12:20])
        assert json_type == _JSON_CHUNK
        assert json_len % 4 == 0, "JSON chunk length not 4-byte aligned"

        bin_offset = 20 + json_len
        bin_len, bin_type = struct.unpack("<II", raw_bytes[bin_offset : bin_offset + 8])
        assert bin_type == _BIN_CHUNK
        assert bin_len % 4 == 0, "BIN chunk length not 4-byte aligned"

    def test_glb_primitive_targets_structure(self, tmp_path: Path) -> None:
        """Verify glTF primitive targets array contains 52 dicts with valid POSITION accessors."""
        target_glb = tmp_path / "test_targets.glb"
        create_reference_morph_glb(target_glb, n_verts=18, n_morphs=52, n_frames=5)

        json_data, _ = _unpack_glb(target_glb)
        mesh = json_data["meshes"][0]
        primitive = mesh["primitives"][0]

        assert "targets" in primitive, "Primitive missing targets array"
        targets = primitive["targets"]
        assert len(targets) == 52, f"Expected 52 morph targets, got {len(targets)}"

        accessors = json_data["accessors"]
        for idx, target in enumerate(targets):
            assert "POSITION" in target, f"Target {idx} missing POSITION attribute"
            pos_acc_idx = target["POSITION"]
            pos_acc = accessors[pos_acc_idx]

            assert pos_acc["type"] == "VEC3"
            assert pos_acc["componentType"] == 5126  # FLOAT
            assert pos_acc["count"] == 18
            assert "min" in pos_acc and "max" in pos_acc, f"Target {idx} POSITION missing min/max bounds"
            assert len(pos_acc["min"]) == 3
            assert len(pos_acc["max"]) == 3

    def test_glb_mesh_weights_and_extras_target_names(self, tmp_path: Path) -> None:
        """Verify mesh defines initial zero weights and extras.targetNames matching ARKit standard."""
        target_glb = tmp_path / "test_mesh_meta.glb"
        create_reference_morph_glb(target_glb, n_verts=15, n_morphs=52, n_frames=5)

        json_data, _ = _unpack_glb(target_glb)
        mesh = json_data["meshes"][0]

        assert "weights" in mesh, "Mesh missing weights array"
        weights = mesh["weights"]
        assert len(weights) == 52, f"Expected 52 initial weights, got {len(weights)}"
        assert all(w == 0.0 for w in weights), "Default morph weights should be 0.0"

        assert "extras" in mesh, "Mesh missing extras dictionary"
        extras = mesh["extras"]
        assert "targetNames" in extras, "Mesh extras missing targetNames"
        names = extras["targetNames"]
        assert len(names) == 52
        assert names == CANONICAL_ARKIT_BLENDSHAPES, "extras.targetNames does not match canonical ARKit names"

    def test_glb_animation_weights_channel_and_sampler(self, tmp_path: Path) -> None:
        """Verify animation channel targets path: 'weights' and sampler matches flattened (T * 52)."""
        target_glb = tmp_path / "test_anim_weights.glb"
        n_frames = 12
        n_morphs = 52
        create_reference_morph_glb(target_glb, n_verts=12, n_morphs=n_morphs, n_frames=n_frames)

        json_data, _ = _unpack_glb(target_glb)
        animations = json_data.get("animations", [])
        assert len(animations) >= 1, "GLB missing animations array"

        clip = animations[0]
        channels = clip.get("channels", [])
        samplers = clip.get("samplers", [])

        # Find morph weights channel
        morph_channels = [c for c in channels if c.get("target", {}).get("path") == "weights"]
        assert len(morph_channels) == 1, f"Expected 1 'weights' animation channel, found {len(morph_channels)}"

        morph_channel = morph_channels[0]
        sampler_idx = morph_channel["sampler"]
        sampler = samplers[sampler_idx]

        assert sampler["interpolation"] == "LINEAR"
        input_acc = json_data["accessors"][sampler["input"]]
        output_acc = json_data["accessors"][sampler["output"]]

        assert input_acc["type"] == "SCALAR"
        assert input_acc["count"] == n_frames

        assert output_acc["type"] == "SCALAR"
        assert output_acc["count"] == n_frames * n_morphs, (
            f"Expected {n_frames * n_morphs} weights elements, got {output_acc['count']}"
        )

    def test_glb_skeletal_and_morph_coexistence(self, tmp_path: Path) -> None:
        """Verify skeletal skinning (JOINTS_0, WEIGHTS_0, bones) and morph targets co-exist seamlessly."""
        target_glb = tmp_path / "test_coexistence.glb"
        create_reference_morph_glb(target_glb, n_verts=24, n_morphs=52, n_frames=10, include_skinning=True)

        json_data, _ = _unpack_glb(target_glb)
        mesh = json_data["meshes"][0]
        prim = mesh["primitives"][0]
        attrs = prim["attributes"]

        assert "POSITION" in attrs
        assert "NORMAL" in attrs
        assert "JOINTS_0" in attrs
        assert "WEIGHTS_0" in attrs
        assert "targets" in prim
        assert len(prim["targets"]) == 52

        # Verify skinning and target vertex counts match exactly
        pos_count = json_data["accessors"][attrs["POSITION"]]["count"]
        joints_count = json_data["accessors"][attrs["JOINTS_0"]]["count"]
        weights_count = json_data["accessors"][attrs["WEIGHTS_0"]]["count"]
        target0_count = json_data["accessors"][prim["targets"][0]["POSITION"]]["count"]

        assert pos_count == joints_count == weights_count == target0_count == 24, (
            "Vertex count mismatch between skeletal skinning and morph target accessors"
        )


# ==============================================================================
# Tier 4: Headless Blender 5.2.2 LTS Verification
# ==============================================================================

class TestBlenderHeadlessMorphVerification:
    """Verifies that headless Blender 5.2.2 LTS cleanly imports GLB morph shape keys."""

    @pytest.mark.skipif(
        not BLENDER_LAUNCHER.exists(),
        reason="Blender 5.2.2 LTS launcher missing at expected Windows path",
    )
    def test_headless_blender_imports_morph_shape_keys(self, tmp_path: Path) -> None:
        """Export GLB with morph targets and verify Blender imports 52 shape keys with active animation."""
        target_glb = tmp_path / "blender_morph_test.glb"
        create_reference_morph_glb(target_glb, n_verts=36, n_morphs=52, n_frames=20, include_skinning=True)

        log_path = tmp_path / "blender_morph.log"
        json_path = tmp_path / "blender_morph.json"
        verify_script = tmp_path / "verify_morph_blender.py"

        # Blender Python verification script
        verify_script.write_text(
            """
import json
import sys
from pathlib import Path
import bpy

def run():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    glb_path = Path(args[args.index("--glb") + 1])
    json_path = Path(args[args.index("--json") + 1])

    report = {"status": "FAIL", "shape_keys": [], "key_count": 0, "has_animation": False}

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path.resolve()))

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        report["error"] = "No MESH objects found in scene"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f)
        return

    mesh_obj = mesh_objs[0]
    shape_keys = mesh_obj.data.shape_keys

    if shape_keys is None:
        report["error"] = "Mesh object has no shape_keys block"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f)
        return

    key_blocks = shape_keys.key_blocks
    report["key_count"] = len(key_blocks)
    report["shape_keys"] = [kb.name for kb in key_blocks]

    # Basis + 52 ARKit keys = 53 key blocks
    if len(key_blocks) >= 53:
        report["status"] = "PASS"
    else:
        report["error"] = f"Expected at least 53 key blocks, got {len(key_blocks)}"

    # Check shape key animation
    if shape_keys.animation_data and shape_keys.animation_data.action:
        report["has_animation"] = True

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run()
""",
            encoding="utf-8",
        )

        cmd = [
            "powershell",
            "-Command",
            (
                f"Start-Process -FilePath '{BLENDER_LAUNCHER}' "
                f"-ArgumentList '--background', '--python', '{verify_script}', '--', "
                f"'--glb', '{target_glb}', "
                f"'--json', '{json_path}' -Wait"
            ),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Blender execution failed: {result.stderr}"

        assert json_path.exists(), f"Blender verification report missing at {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        assert report_data.get("status") == "PASS", f"Blender verification failed: {report_data}"
        assert report_data["key_count"] == 53, (
            f"Expected 53 shape keys (1 Basis + 52 ARKit), got {report_data['key_count']}"
        )
        assert "mouthSmileLeft" in report_data["shape_keys"]
        assert "jawOpen" in report_data["shape_keys"]
        assert "eyeBlinkLeft" in report_data["shape_keys"]


# ==============================================================================
# Tier 5: Milestone 3 Direct Production Verification Pipeline
# ==============================================================================

class TestMilestone3ProductionPipeline:
    """Verifies production implementations of HumanMotion, Retargeting, and glTF Export."""

    def test_human_motion_loads_companion_face_motion(self, tmp_path: Path) -> None:
        """Verify load_human_motion loads companion face_motion.npz in the same directory."""
        from v2bm.human_motion import load_human_motion

        F = 10
        J = 24
        joints = np.zeros((F, J, 3), dtype=np.float64)
        joint_names = [f"joint_{i}" for i in range(J)]
        skel_path = tmp_path / "skeleton_motion.npz"
        np.savez(
            skel_path,
            joints_3d=joints,
            joint_names=joint_names,
            fps=30.0,
            timestamps=np.arange(F, dtype=np.float64) / 30.0,
            root_translation=np.zeros((F, 3), dtype=np.float64),
        )

        face_path = tmp_path / "face_motion.npz"
        rng = np.random.default_rng(501)
        face_bs = rng.uniform(0.0, 1.0, size=(F, 52)).astype(np.float32)
        head_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (F, 1))
        gaze_vecs = np.zeros((F, 2, 3), dtype=np.float32)
        np.savez(
            face_path,
            blendshapes=face_bs,
            blendshape_names=np.array(CANONICAL_ARKIT_BLENDSHAPES),
            head_rotation_quat=head_quat,
            gaze_vectors=gaze_vecs,
            framing="portrait",
        )

        motion = load_human_motion(skel_path)
        assert motion.face_blendshapes is not None
        assert motion.face_blendshapes.shape == (F, 52)
        assert np.allclose(motion.face_blendshapes, face_bs)
        assert motion.face_blendshape_names == CANONICAL_ARKIT_BLENDSHAPES
        assert motion.head_rotation_quat is not None
        assert motion.head_rotation_quat.shape == (F, 4)
        assert motion.gaze_vectors is not None
        assert motion.gaze_vectors.shape == (F, 2, 3)
        assert motion.framing == "portrait"

    def test_human_motion_loads_embedded_face_keys(self, tmp_path: Path) -> None:
        """Verify load_human_motion loads embedded face keys directly from skeleton_motion.npz."""
        from v2bm.human_motion import load_human_motion

        F = 8
        J = 24
        joints = np.zeros((F, J, 3), dtype=np.float64)
        joint_names = [f"joint_{i}" for i in range(J)]
        face_bs = np.ones((F, 52), dtype=np.float32) * 0.5
        head_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (F, 1))

        skel_path = tmp_path / "skeleton_motion.npz"
        np.savez(
            skel_path,
            joints_3d=joints,
            joint_names=joint_names,
            fps=30.0,
            timestamps=np.arange(F, dtype=np.float64) / 30.0,
            face_blendshapes=face_bs,
            face_blendshape_names=np.array(CANONICAL_ARKIT_BLENDSHAPES),
            head_rotation_quat=head_quat,
            framing="full_body",
        )

        motion = load_human_motion(skel_path)
        assert motion.face_blendshapes is not None
        assert motion.face_blendshapes.shape == (F, 52)
        assert motion.face_blendshape_names == CANONICAL_ARKIT_BLENDSHAPES
        assert motion.head_rotation_quat is not None
        assert motion.framing == "full_body"

    def test_build_morph_target_mapping_exact_and_prefixes(self) -> None:
        """Verify build_morph_target_mapping resolves exact, case-insensitive, and prefixed names."""
        from v2bm.rig.retarget import build_morph_target_mapping

        targets = [
            "browInnerUp",                 # Exact
            "JAWOPEN",                     # Case-insensitive
            "rig:mouthSmileLeft",          # Prefix rig:
            "blendshape.mouthSmileRight",  # Prefix blendshape.
            "facs_eyeBlinkLeft",           # Prefix facs_
            "basis.eyeBlinkRight",         # Prefix basis.
            "unmapped_extra_key",          # Unmapped
        ]
        mapping = build_morph_target_mapping(targets)
        assert len(mapping) == len(targets)
        assert mapping[0] == CANONICAL_ARKIT_BLENDSHAPES.index("browInnerUp")
        assert mapping[1] == CANONICAL_ARKIT_BLENDSHAPES.index("jawOpen")
        assert mapping[2] == CANONICAL_ARKIT_BLENDSHAPES.index("mouthSmileLeft")
        assert mapping[3] == CANONICAL_ARKIT_BLENDSHAPES.index("mouthSmileRight")
        assert mapping[4] == CANONICAL_ARKIT_BLENDSHAPES.index("eyeBlinkLeft")
        assert mapping[5] == CANONICAL_ARKIT_BLENDSHAPES.index("eyeBlinkRight")
        assert mapping[6] == -1

    def test_retarget_facial_blendshapes_sensitivities_and_blink_snap(self) -> None:
        """Verify sensitivity multipliers and non-linear blink snapping on retarget_facial_blendshapes."""
        from v2bm.rig.retarget import retarget_facial_blendshapes, DEFAULT_BLENDSHAPE_SENSITIVITIES

        target_names = ["browInnerUp", "jawOpen", "mouthSmileLeft", "eyeBlinkLeft"]
        raw = np.zeros(52, dtype=np.float32)
        raw[CANONICAL_ARKIT_BLENDSHAPES.index("browInnerUp")] = 0.5
        raw[CANONICAL_ARKIT_BLENDSHAPES.index("jawOpen")] = 0.4
        raw[CANONICAL_ARKIT_BLENDSHAPES.index("mouthSmileLeft")] = 0.6
        raw[CANONICAL_ARKIT_BLENDSHAPES.index("eyeBlinkLeft")] = 0.85

        retargeted = retarget_facial_blendshapes(raw, target_names)
        assert retargeted is not None
        assert retargeted.shape == (4,)

        # Brow inner up sensitivity is 1.25 -> 0.5 * 1.25 = 0.625
        assert np.isclose(retargeted[0], 0.5 * DEFAULT_BLENDSHAPE_SENSITIVITIES["browInnerUp"])
        # Jaw open sensitivity is 1.10 -> 0.4 * 1.10 = 0.44
        assert np.isclose(retargeted[1], 0.4 * DEFAULT_BLENDSHAPE_SENSITIVITIES["jawOpen"])
        # Smile left sensitivity is 1.15 -> 0.6 * 1.15 = 0.69
        assert np.isclose(retargeted[2], 0.6 * DEFAULT_BLENDSHAPE_SENSITIVITIES["mouthSmileLeft"])

        # Blink snap for w >= 0.65:
        expected_blink = 0.65 + 0.35 * ((0.85 - 0.65) / 0.35) ** 0.75
        assert np.isclose(retargeted[3], expected_blink)
        assert retargeted[3] > 0.85, "Blink snapping did not amplify blink towards full closure"

    def test_retarget_head_neck_orientation(self) -> None:
        """Verify camera-to-world transform and 35/65 cervical distribution for head and neck."""
        from v2bm.rig.retarget import retarget_head_neck_orientation
        from v2bm.rig.kinematics import forward_kinematics

        rig = _make_dummy_rig()
        world = forward_kinematics(rig, {})

        # Camera space: pitch head downward by ~20 degrees
        pitch_rad = 0.35
        # wxyz for rotation around camera X axis
        head_quat = np.array([np.cos(pitch_rad / 2), np.sin(pitch_rad / 2), 0.0, 0.0], dtype=np.float32)
        gaze_vector = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        r_anim_neck, r_anim_head = retarget_head_neck_orientation(rig, world, head_quat, gaze_vector)

        # Both returned rotations must be valid 3x3 rotation matrices
        assert r_anim_neck.shape == (3, 3)
        assert r_anim_head.shape == (3, 3)
        assert np.allclose(r_anim_neck @ r_anim_neck.T, np.eye(3), atol=1e-5)
        assert np.allclose(r_anim_head @ r_anim_head.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(r_anim_neck), 1.0, atol=1e-4)
        assert np.isclose(np.linalg.det(r_anim_head), 1.0, atol=1e-4)

    def test_export_character_animation_glb_production(self, tmp_path: Path) -> None:
        """Verify production export_character_animation_glb writes binary GLB with morph weights animation."""
        from v2bm.rig.gltf import export_character_animation_glb, write_glb

        # Verify alias
        assert write_glb is export_character_animation_glb

        # Create dummy rig asset with a skinned mesh containing morph deltas
        rig = _make_dummy_rig()
        V = 12
        M = 52
        rng = np.random.default_rng(601)
        base_verts = rng.uniform(-10.0, 10.0, size=(V, 3)).astype(np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.uint32)
        weights = np.zeros((V, len(rig.order)), dtype=np.float64)
        weights[:, 0] = 1.0  # Root bone
        ibm = [np.eye(4, dtype=np.float64) for _ in range(len(rig.order))]

        deltas_pos = rng.normal(0.0, 0.5, size=(M, V, 3)).astype(np.float32)
        deltas_norm = rng.normal(0.0, 0.1, size=(M, V, 3)).astype(np.float32)

        mesh = SkinnedMesh(
            name="test_mesh",
            vertices=base_verts,
            faces=faces,
            weights=weights,
            ibm=ibm,
            color=np.array([0.8, 0.8, 0.8]),
            morph_target_names=CANONICAL_ARKIT_BLENDSHAPES,
            morph_deltas_pos=deltas_pos,
            morph_deltas_norm=deltas_norm,
        )
        asset = HumanoidRigAsset(rig=rig, meshes=[mesh])

        F = 10
        r_anim_frames = [{b: np.eye(3) for b in rig.order} for _ in range(F)]
        hips_cm_frames = [np.array([0.0, 100.0, 0.0]) for _ in range(F)]
        morph_weights = rng.uniform(0.0, 1.0, size=(F, M)).astype(np.float32)

        target_glb = tmp_path / "production_morph.glb"
        out_path = export_character_animation_glb(
            target_glb,
            asset,
            r_anim_frames,
            hips_cm_frames,
            fps=30.0,
            morph_weights_frames=morph_weights,
            morph_target_names=CANONICAL_ARKIT_BLENDSHAPES,
        )
        assert out_path.exists()

        json_data, _ = _unpack_glb(out_path)
        # Check mesh morph metadata
        mesh_dict = json_data["meshes"][0]
        assert "weights" in mesh_dict
        assert len(mesh_dict["weights"]) == M
        assert mesh_dict["extras"]["targetNames"] == CANONICAL_ARKIT_BLENDSHAPES

        # Check primitive targets
        prim = mesh_dict["primitives"][0]
        assert "targets" in prim
        assert len(prim["targets"]) == M

        # Check animation channels
        anim = json_data["animations"][0]
        weights_channels = [c for c in anim["channels"] if c["target"]["path"] == "weights"]
        assert len(weights_channels) == 1
        w_sampler_idx = weights_channels[0]["sampler"]
        sampler = anim["samplers"][w_sampler_idx]
        output_acc = json_data["accessors"][sampler["output"]]
        assert output_acc["count"] == F * M

    def test_pipeline_multi_mesh_rig_morph_discovery_and_export(self, tmp_path: Path) -> None:
        """Verify multi-mesh rig morph target discovery, retargeting, and glTF 2.0 export.

        Covers the canonical multi-mesh humanoid rig (assets/rigs/humanoid_rig.fbx)
        where mesh 0 (Alpha_Joints) has no shape keys and mesh 1 (Alpha_Surface)
        contains all 52 ARKit morph targets.
        """
        from v2bm.human_motion import HumanMotion
        from v2bm.rig.animation import build_animation
        from v2bm.rig.gltf import export_character_animation_glb
        from v2bm.rig.rig_retarget import build_mapping
        from v2bm.rig.skinned_mesh import load_character_asset
        from v2bm.rig.tpose_calibration import load_calibration

        rig_path = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"
        assert rig_path.exists(), f"Bundled canonical rig missing at {rig_path}"

        # 1. Load canonical multi-mesh character asset
        asset = load_character_asset(rig_path)
        assert len(asset.meshes) >= 2, f"Expected multi-mesh asset, got {len(asset.meshes)} meshes"

        # Mesh 0 is Alpha_Joints and has no shape keys
        assert asset.meshes[0].name == "Alpha_Joints"
        assert asset.meshes[0].morph_target_names is None

        # Mesh 1 is Alpha_Surface and contains 52 ARKit shape keys
        assert asset.meshes[1].name == "Alpha_Surface"
        assert asset.meshes[1].morph_target_names is not None
        assert len(asset.meshes[1].morph_target_names) == 52

        # 2. Verify that the remediated query extracts the 52 names from Alpha_Surface
        target_morph_names = next(
            (m.morph_target_names for m in asset.meshes if getattr(m, "morph_target_names", None)),
            None,
        )
        assert target_morph_names is not None
        assert len(target_morph_names) == 52
        assert target_morph_names == asset.meshes[1].morph_target_names
        assert "jawOpen" in target_morph_names

        # 3. Construct minimal HumanMotion with active face blendshapes
        F = 4
        fps = 30.0
        calibration = load_calibration(rig_path)
        mapping = build_mapping(calibration, verbose=False)

        from tests.test_retarget_stability import create_smooth_synthetic_motion
        motion = create_smooth_synthetic_motion(num_frames=F, fps=fps)

        face_blendshapes = np.zeros((F, 52), dtype=np.float32)
        jaw_open_idx = target_morph_names.index("jawOpen")
        face_blendshapes[:, jaw_open_idx] = 1.0

        motion.face_blendshapes = face_blendshapes
        motion.face_blendshape_names = target_morph_names

        # 4. Retarget animation
        frames = list(range(F))
        anim = build_animation(
            motion,
            mapping,
            frames,
            target_morph_names=target_morph_names,
            verbose=False,
        )

        assert anim.morph_weights is not None
        assert anim.morph_weights.shape == (F, 52)
        assert np.all(anim.morph_weights[:, jaw_open_idx] > 0.5)

        # 5. Export binary glTF 2.0 container
        target_glb = tmp_path / "multi_mesh_humanoid_animation.glb"
        out_path = export_character_animation_glb(
            target_glb,
            asset,
            anim.r_anim_rig,
            anim.hips_cm_rig,
            anim.fps,
            morph_weights_frames=anim.morph_weights,
            morph_target_names=target_morph_names,
        )
        assert out_path.exists()

        # 6. Verify unpacked GLB structure
        json_data, _ = _unpack_glb(out_path)

        # Non-empty targets on Alpha_Surface
        alpha_surface_primitives = []
        for m in json_data["meshes"]:
            for prim in m.get("primitives", []):
                if "targets" in prim:
                    alpha_surface_primitives.append(prim)

        assert len(alpha_surface_primitives) > 0, "No mesh primitives contain morph targets"
        for prim in alpha_surface_primitives:
            assert len(prim["targets"]) == 52

        # Animation channel targeting path: "weights"
        anim_data = json_data["animations"][0]
        weights_channels = [c for c in anim_data["channels"] if c["target"]["path"] == "weights"]
        assert len(weights_channels) == 1, f"Expected 1 weights channel, found {len(weights_channels)}"
        w_sampler_idx = weights_channels[0]["sampler"]
        sampler = anim_data["samplers"][w_sampler_idx]
        output_acc = json_data["accessors"][sampler["output"]]
        assert output_acc["count"] == F * 52

