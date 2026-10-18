"""Unit and integration tests for GLB UV texture and material image embedding.

Verifies:
1. SkinnedMesh correctly extracts and exposes UV coordinates, texture image bytes,
   and detected MIME types from FBX assets.
2. Companion texture loading from the filesystem when textures are not embedded.
3. Seam-splitting algorithm preserves 1:1 attribute mapping, correct index bounds,
   and partition of unity for skinning weights across split vertices.
4. Binary GLB export embeds TEXCOORD_0, images, textures, samplers, and materials
   referencing baseColorTexture with byte-exact image fidelity.
5. Graceful fallback when textures are absent (retaining COLOR_0 and baseColorFactor).
6. Headless Blender 5.2.2 LTS cleanly imports the textured GLB model with full fidelity.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from v2bm.rig.fbx_skeleton import CORE_BONE_SUFFIXES
from v2bm.rig.gltf import _split_mesh_uvs, write_glb
from v2bm.rig.kinematics import HumanoidRig, RigBone
from v2bm.rig.skinned_mesh import (
    HumanoidRigAsset,
    SkinnedMesh,
    _detect_image_mime,
    _find_companion_texture,
    load_character_asset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FBX_ASSET_PATH = REPO_ROOT / "assets" / "rigs" / "humanoid_rig.fbx"
BLENDER_LAUNCHER = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Microsoft"
    / "WindowsApps"
    / "blender-launcher.exe"
)
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_blender_glb.py"


def _unpack_glb(glb_path: Path) -> tuple[dict, bytes]:
    """Unpack binary GLB container into its parsed JSON chunk and binary chunk."""
    with open(glb_path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67, f"Invalid glTF magic: {hex(magic)}"
        assert version == 2, f"Unsupported glTF version: {version}"

        json_len, json_type = struct.unpack("<II", f.read(8))
        assert json_type == 0x4E4F534A, f"Expected JSON chunk (0x4E4F534A), got {hex(json_type)}"
        json_bytes = f.read(json_len)
        json_data = json.loads(json_bytes.decode("utf-8"))

        bin_len, bin_type = struct.unpack("<II", f.read(8))
        assert bin_type == 0x004E4942, f"Expected BIN chunk (0x004E4942), got {hex(bin_type)}"
        bin_data = f.read(bin_len)

    return json_data, bin_data


def _make_dummy_rig() -> HumanoidRig:
    """Create a minimal 25-bone HumanoidRig for unit test exports."""
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


# ---------------------------------------------------------------------------
# Test 1: MIME Type Detection & Companion Texture Discovery
# ---------------------------------------------------------------------------


def test_detect_image_mime() -> None:
    """Verify MIME detection across PNG and JPEG magic bytes and filename extensions."""
    png_magic = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    jpeg_magic = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    random_bytes = b"NOT_AN_IMAGE_HEADER_12345678"

    assert _detect_image_mime(png_magic) == "image/png"
    assert _detect_image_mime(jpeg_magic) == "image/jpeg"
    assert _detect_image_mime(random_bytes, "texture.png") == "image/png"
    assert _detect_image_mime(random_bytes, "photo.jpg") == "image/jpeg"
    assert _detect_image_mime(random_bytes, "photo.jpeg") == "image/jpeg"
    assert _detect_image_mime(random_bytes, "unknown.bin") == "image/png"


def test_companion_texture_resolution(tmp_path: Path) -> None:
    """Verify companion image files are discovered in the FBX folder."""
    fbx_file = tmp_path / "character.fbx"
    fbx_file.write_bytes(b"DummyFBXContent")

    companion_png = tmp_path / "character.png"
    dummy_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    companion_png.write_bytes(dummy_png_bytes)

    raw_bytes, mime = _find_companion_texture(fbx_file, "MeshBody")
    assert raw_bytes == dummy_png_bytes
    assert mime == "image/png"


# ---------------------------------------------------------------------------
# Test 2: Asset Loading with Embedded Textures from humanoid_rig.fbx
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FBX_ASSET_PATH.exists(), reason="humanoid_rig.fbx missing")
def test_skinned_mesh_texture_extraction() -> None:
    """Verify load_character_asset populates UVs, raw texture bytes, and MIME."""
    asset = load_character_asset(FBX_ASSET_PATH, textures=True)
    assert len(asset.meshes) == 2, f"Expected 2 meshes, got {len(asset.meshes)}"

    surface_mesh = next((m for m in asset.meshes if m.name == "Alpha_Surface"), None)
    assert surface_mesh is not None, "Alpha_Surface mesh missing from asset"
    assert surface_mesh.uvs is not None, "UV coordinates missing from Alpha_Surface"
    assert surface_mesh.uvs.shape == (len(surface_mesh.faces), 3, 2), (
        f"Expected UV shape {(len(surface_mesh.faces), 3, 2)}, got {surface_mesh.uvs.shape}"
    )
    assert surface_mesh.texture_image is not None, "Alpha_Surface texture image bytes missing"
    assert len(surface_mesh.texture_image) > 0, "Alpha_Surface texture image bytes empty"
    assert surface_mesh.texture_mime == "image/png"
    assert surface_mesh.texture_image.startswith(b"\x89PNG\r\n\x1a\n"), (
        "Alpha_Surface texture bytes do not start with PNG magic"
    )
    assert surface_mesh.face_colors is not None, (
        "Software preview face_colors must be preserved"
    )

    joints_mesh = next((m for m in asset.meshes if m.name == "Alpha_Joints"), None)
    assert joints_mesh is not None, "Alpha_Joints mesh missing from asset"
    assert joints_mesh.uvs is not None, "UV coordinates missing from Alpha_Joints"
    assert joints_mesh.texture_image is not None, "Alpha_Joints texture image bytes missing"
    assert joints_mesh.texture_mime == "image/png"


# ---------------------------------------------------------------------------
# Test 3: UV Seam-Splitting Algorithm & Mathematical Properties
# ---------------------------------------------------------------------------


def test_uv_seam_splitting_properties() -> None:
    """Verify seam splitting duplicates only boundary vertices and preserves properties."""
    # Construct a simple quad with two triangles sharing an edge, but split in UV space
    # Vertex positions: 0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    # Triangle 0 has UVs in [0, 0.5], Triangle 1 has UVs in [0.5, 1.0] (discontinuous along shared edge 0-2)
    uvs = np.array([
        [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
        [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0]],
    ], dtype=np.float64)

    split_faces, v_map, split_uvs = _split_mesh_uvs(faces, uvs)

    assert split_faces.shape == faces.shape
    assert len(v_map) == len(split_uvs)
    # Triangles 0 and 1 have completely distinct UV coordinates at vertices 0 and 2,
    # so shared vertices 0 and 2 must be split into distinct glTF vertices
    assert len(v_map) == 6
    assert np.all(split_faces < len(v_map)), "Face indices exceed split vertex array bounds"


@pytest.mark.skipif(not FBX_ASSET_PATH.exists(), reason="humanoid_rig.fbx missing")
def test_seam_splitting_on_character_rig() -> None:
    """Verify seam splitting on humanoid_rig.fbx preserves skinning partition of unity."""
    asset = load_character_asset(FBX_ASSET_PATH, textures=True)
    surface_mesh = next(m for m in asset.meshes if m.name == "Alpha_Surface")

    split_faces, v_map, split_uvs = _split_mesh_uvs(surface_mesh.faces, surface_mesh.uvs)

    # 1. Split vertex count must be strictly greater than original geometric count due to UV seams
    assert len(v_map) > len(surface_mesh.vertices), (
        f"Expected split vertices > {len(surface_mesh.vertices)}, got {len(v_map)}"
    )
    # Empirical count for Alpha_Surface (15,893 for base mannequin; 15,996 with facial features)
    assert len(v_map) in (15893, 15996), f"Expected 15893 or 15996 split vertices, got {len(v_map)}"

    # 2. Partition of unity must be strictly preserved on replicated skinning weights
    split_weights = surface_mesh.weights[v_map]
    weight_sums = split_weights.sum(axis=1)
    max_dev = float(np.max(np.abs(weight_sums - 1.0)))
    assert max_dev < 1e-5, f"Partition of unity violated on split vertices: {max_dev:.3e}"

    # 3. UV coordinates must be valid float32 values within range
    assert np.all(np.isfinite(split_uvs)), "Non-finite values encountered in split UVs"


# ---------------------------------------------------------------------------
# Test 4: Binary GLB Export with Full Image Embedding Inspection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FBX_ASSET_PATH.exists(), reason="humanoid_rig.fbx missing")
def test_glb_export_texture_embedding(tmp_path: Path) -> None:
    """Verify exported binary GLB contains valid TEXCOORD_0, images, textures, and materials."""
    asset = load_character_asset(FBX_ASSET_PATH, textures=True)
    out_glb = tmp_path / "textured_character.glb"

    r_anim = [{name: np.eye(3) for name in asset.rig.order}]
    hips = [np.zeros(3)]
    write_glb(out_glb, asset, r_anim, hips, fps=30.0)

    assert out_glb.exists() and out_glb.stat().st_size > 0, "GLB output file missing or empty"

    json_data, bin_data = _unpack_glb(out_glb)

    # Check top-level arrays
    assert "images" in json_data and len(json_data["images"]) == 2, (
        f"Expected 2 embedded images, got {json_data.get('images')}"
    )
    assert "textures" in json_data and len(json_data["textures"]) == 2, (
        f"Expected 2 texture entries, got {json_data.get('textures')}"
    )
    assert "samplers" in json_data and len(json_data["samplers"]) == 1, (
        f"Expected 1 sampler entry, got {json_data.get('samplers')}"
    )

    # Check sampler parameters: Linear, LinearMipmapLinear, Repeat, Repeat
    sampler = json_data["samplers"][0]
    assert sampler["magFilter"] == 9729, "Sampler magFilter must be LINEAR (9729)"
    assert sampler["minFilter"] == 9987, "Sampler minFilter must be LINEAR_MIPMAP_LINEAR (9987)"
    assert sampler["wrapS"] == 10497, "Sampler wrapS must be REPEAT (10497)"
    assert sampler["wrapT"] == 10497, "Sampler wrapT must be REPEAT (10497)"

    # Verify binary image bytes match the original textures byte-for-byte
    for i, img in enumerate(json_data["images"]):
        bv_idx = img["bufferView"]
        bv = json_data["bufferViews"][bv_idx]
        assert "target" not in bv, "glTF image bufferView must NOT specify GPU target"

        offset = bv["byteOffset"]
        length = bv["byteLength"]
        extracted_bytes = bin_data[offset : offset + length]

        mesh_source = asset.meshes[i]
        assert extracted_bytes == mesh_source.texture_image, (
            f"Embedded binary bytes for Image {i} do not match original texture bytes"
        )
        assert img["mimeType"] == mesh_source.texture_mime

    # Verify material pbrMetallicRoughness references baseColorTexture
    for i, mat in enumerate(json_data["materials"]):
        pbr = mat.get("pbrMetallicRoughness", {})
        assert "baseColorTexture" in pbr, f"Material {i} missing baseColorTexture"
        tex_idx = pbr["baseColorTexture"]["index"]
        assert 0 <= tex_idx < len(json_data["textures"])
        assert pbr["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0], (
            "baseColorFactor must be [1,1,1,1] when baseColorTexture is active"
        )

    # Verify mesh primitive attributes include TEXCOORD_0
    for i, prim in enumerate(json_data["meshes"][0]["primitives"]):
        attrs = prim["attributes"]
        assert "TEXCOORD_0" in attrs, f"Primitive {i} missing TEXCOORD_0 attribute"
        assert "POSITION" in attrs
        assert "NORMAL" in attrs
        assert "JOINTS_0" in attrs
        assert "WEIGHTS_0" in attrs

        # Verify all primitive accessors have identical vertex count
        counts = [json_data["accessors"][acc_idx]["count"] for acc_idx in attrs.values()]
        assert len(set(counts)) == 1, (
            f"Primitive {i} attribute vertex counts must be identical: {counts}"
        )


# ---------------------------------------------------------------------------
# Test 5: Fallback Support for Untextured Meshes
# ---------------------------------------------------------------------------


def test_untextured_mesh_graceful_fallback(tmp_path: Path) -> None:
    """Verify export gracefully falls back when mesh has no texture image."""
    rig = _make_dummy_rig()

    # Construct synthetic mesh with face colours but NO texture image
    n_verts = 8
    n_bones = len(rig.order)
    vertices = np.random.randn(n_verts, 3).astype(np.float64) * 50.0
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]
    ], dtype=np.int32)
    weights = np.zeros((n_verts, n_bones), dtype=np.float64)
    weights[:, 0] = 1.0
    ibm = np.repeat(np.eye(4)[None, ...], n_bones, axis=0)
    color = np.array([0.5, 0.5, 0.5])
    face_colors = np.array([
        [200, 100, 50], [200, 100, 50], [50, 100, 200], [50, 100, 200]
    ], dtype=np.uint8)

    untextured_mesh = SkinnedMesh(
        name="UntexturedBody",
        vertices=vertices,
        faces=faces,
        weights=weights,
        ibm=ibm,
        color=color,
        face_colors=face_colors,
        texture_image=None,
        texture_mime=None,
        uvs=None,
    )
    asset = HumanoidRigAsset(rig=rig, meshes=[untextured_mesh])

    out_glb = tmp_path / "untextured_fallback.glb"
    r_anim = [{name: np.eye(3) for name in rig.order}]
    hips = [np.zeros(3)]
    write_glb(out_glb, asset, r_anim, hips, fps=30.0)

    json_data, _ = _unpack_glb(out_glb)

    # When no texture images exist, images and textures must not be present in top-level JSON
    assert "images" not in json_data or len(json_data["images"]) == 0
    assert "textures" not in json_data or len(json_data["textures"]) == 0

    # Material must NOT have baseColorTexture
    mat = json_data["materials"][0]
    pbr = mat.get("pbrMetallicRoughness", {})
    assert "baseColorTexture" not in pbr

    # Primitive attributes should include COLOR_0 for software fallback and NO TEXCOORD_0
    prim = json_data["meshes"][0]["primitives"][0]
    assert "COLOR_0" in prim["attributes"]
    assert "TEXCOORD_0" not in prim["attributes"]


# ---------------------------------------------------------------------------
# Test 6: Headless Blender 5.2.2 LTS Verification of Textured GLB
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not BLENDER_LAUNCHER.exists(),
    reason="Blender 5.2.2 LTS launcher missing at expected Windows path",
)
@pytest.mark.skipif(not FBX_ASSET_PATH.exists(), reason="humanoid_rig.fbx missing")
def test_headless_blender_textured_glb_verification(tmp_path: Path) -> None:
    """Export textured GLB and verify headless Blender LTS imports it with STATUS: PASS."""
    asset = load_character_asset(FBX_ASSET_PATH, textures=True)
    target_glb = tmp_path / "blender_verify_textured.glb"

    r_anim = [{name: np.eye(3) for name in asset.rig.order} for _ in range(5)]
    hips = [np.array([0.0, 95.0, 0.0]) for _ in range(5)]
    write_glb(target_glb, asset, r_anim, hips, fps=30.0)

    log_path = tmp_path / "blender_test.log"
    json_path = tmp_path / "blender_test.json"
    render_path = tmp_path / "blender_render.png"

    cmd = [
        "powershell",
        "-Command",
        (
            f"Start-Process -FilePath '{BLENDER_LAUNCHER}' "
            f"-ArgumentList '--background', '--python', '{VERIFY_SCRIPT}', '--', "
            f"'--glb', '{target_glb}', "
            f"'--log', '{log_path}', "
            f"'--json', '{json_path}', "
            f"'--render', '{render_path}' -Wait"
        ),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"PowerShell execution failed: {result.stderr}"

    assert log_path.exists(), f"Blender log file missing at {log_path}"
    log_content = log_path.read_text(encoding="utf-8")
    assert "STATUS: PASS" in log_content, (
        f"Blender verification failed. Log content:\n{log_content}"
    )

    assert json_path.exists(), f"Blender JSON output missing at {json_path}"
    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)

    assert data.get("status") == "PASS", f"Expected status PASS, got: {data}"
    assert data["armature"]["bone_count"] == 25, (
        f"Expected 25 bones, got {data['armature']['bone_count']}"
    )
    assert render_path.exists() and render_path.stat().st_size > 0, (
        "Rendered verification frame missing or empty"
    )
