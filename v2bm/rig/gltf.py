"""Write a skinned Humanoid character + its animation as a .glb (glTF 2.0).

Blender opens this in one import: mesh, skin weights, and the clip. Node TRS
is the full local Humanoid transform `T * Rpre * R_anim` (glTF has no
PreRotation), so the importer's linear-blend skin matches this repo's.

Units are metres (Y-up). Vertex influences are the top four weights.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .fbx_skeleton import matrix_to_quat_wxyz
from .kinematics import continue_quaternion, local_matrix
from .skinned_mesh import HumanoidRigAsset

_GLTF_MAGIC = 0x46546C67
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942


class _Bin:
    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self._cursor = 0

    def add(self, data: bytes) -> tuple[int, int]:
        offset = self._cursor
        self._parts.append(data)
        self._cursor += len(data)
        pad = (4 - len(data) % 4) % 4
        if pad:
            self._parts.append(b"\x00" * pad)
            self._cursor += pad
        return offset, len(data)

    def add_array(self, array: np.ndarray) -> tuple[int, int]:
        return self.add(np.ascontiguousarray(array).tobytes())

    def dumps(self) -> bytes:
        return b"".join(self._parts)


def _quat_xyzw(rmat: np.ndarray) -> np.ndarray:
    wxyz = matrix_to_quat_wxyz(rmat)
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)


def _scale_affine(mat: np.ndarray, scale: float) -> np.ndarray:
    """Metres from centimetres: R stays, translation is multiplied by `scale`."""
    out = np.asarray(mat, dtype=np.float64).reshape(4, 4).copy()
    out[:3, 3] *= scale
    return out


def _top4_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_bones = weights.shape[1]
    take = min(4, n_bones)
    order = np.argpartition(-weights, take - 1, axis=1)[:, :take]
    w = np.take_along_axis(weights, order, axis=1)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
    if take < 4:
        pad_i = np.zeros((weights.shape[0], 4 - take), dtype=order.dtype)
        pad_w = np.zeros((weights.shape[0], 4 - take), dtype=w.dtype)
        order = np.concatenate([order, pad_i], axis=1)
        w = np.concatenate([w, pad_w], axis=1)
    return order.astype(np.uint16), w.astype(np.float32)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_n = np.cross(p1 - p0, p2 - p0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.where(length < 1e-12, 1.0, length)).astype(np.float32)


def _split_mesh_uvs(
    faces: np.ndarray,
    uvs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split vertices along UV seams so each face corner has a unique vertex index.

    glTF 2.0 requires 1:1 correspondence between vertex positions, normals,
    weights, joints, and TEXCOORD_0. FBX UVs are mapped per polygon vertex
    (by polygon corner), so vertices on texture seams have different UV coordinates
    on different faces.

    Returns:
        new_faces: (F, 3) uint32 face indices pointing into split vertex arrays.
        v_map: (N,) int64 mapping from split vertex index to original vertex index.
        split_uvs: (N, 2) float32 glTF UV coordinates (with inverted V axis).
    """
    flat_faces = faces.reshape(-1)
    flat_uvs = uvs.reshape(-1, 2)
    n_corners = flat_faces.shape[0]

    # glTF UVs have origin at top-left, FBX at bottom-left: v_gltf = 1.0 - v_fbx
    u_gltf = flat_uvs[:, 0].astype(np.float32)
    v_gltf = (1.0 - flat_uvs[:, 1]).astype(np.float32)

    remap: dict[tuple[int, int, int], int] = {}
    unique_orig_v: list[int] = []
    unique_uvs: list[tuple[float, float]] = []
    new_indices = np.empty(n_corners, dtype=np.uint32)

    for i in range(n_corners):
        orig_v = int(flat_faces[i])
        u_val = float(u_gltf[i])
        v_val = float(v_gltf[i])
        key = (orig_v, int(round(u_val * 1_000_000.0)), int(round(v_val * 1_000_000.0)))
        idx = remap.get(key)
        if idx is None:
            idx = len(unique_orig_v)
            remap[key] = idx
            unique_orig_v.append(orig_v)
            unique_uvs.append((u_val, v_val))
        new_indices[i] = idx

    return (
        new_indices.reshape(faces.shape),
        np.asarray(unique_orig_v, dtype=np.int64),
        np.asarray(unique_uvs, dtype=np.float32),
    )


def write_glb(
    path: Path,
    asset: HumanoidRigAsset,
    r_anim_frames: list[dict[str, np.ndarray]],
    hips_cm_frames: list[np.ndarray],
    fps: float,
) -> Path:
    """Skinned character + clip → a single `.glb` file."""
    rig = asset.rig
    scale = float(rig.meters_per_unit)
    identity = np.eye(3)
    n_bones = len(rig.order)
    bone_index = {name: i for i, name in enumerate(rig.order)}

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

    nodes: list[dict] = []
    for name in rig.order:
        bone = rig.bones[name]
        rest = local_matrix(bone, identity)
        kids = [bone_index[c] for c in bone.children if c in bone_index]
        node = {
            "name": name,
            "translation": [float(v) * scale for v in rest[:3, 3]],
            "rotation": [float(v) for v in _quat_xyzw(rest[:3, :3])],
        }
        if kids:
            node["children"] = kids
        nodes.append(node)

    mesh_node = len(nodes)
    nodes.append({"name": "skinned_mesh", "mesh": 0, "skin": 0})

    ibm = np.stack(
        [_scale_affine(asset.meshes[0].ibm[i], scale) for i in range(n_bones)],
        axis=0,
    ).astype(np.float32)
    # glTF stores matrices column-major.
    ibm_acc = accessor(np.ascontiguousarray(ibm.transpose(0, 2, 1)), "MAT4", 5126)

    images: list[dict] = []
    textures: list[dict] = []
    tex_samplers: list[dict] = []
    image_cache: dict[bytes, int] = {}

    primitives = []
    materials = []
    for mesh in asset.meshes:
        color = np.clip(np.asarray(mesh.color, dtype=np.float64).reshape(3), 0, 1)
        is_joint = "joint" in mesh.name.lower()
        metallic = 0.60 if is_joint else 0.05
        roughness = 0.30 if is_joint else 0.35

        has_uvs = mesh.uvs is not None and len(mesh.uvs) == len(mesh.faces)
        has_texture = mesh.texture_image is not None and len(mesh.texture_image) > 0
        has_face_colors = (
            mesh.face_colors is not None and len(mesh.face_colors) == len(mesh.faces)
        )

        if has_uvs:
            faces, v_map, split_uvs = _split_mesh_uvs(mesh.faces, mesh.uvs)
            verts = (mesh.vertices[v_map] * scale).astype(np.float32)
            joints, weights = _top4_weights(mesh.weights[v_map])
            orig_normals = _vertex_normals(
                (mesh.vertices * scale).astype(np.float64), mesh.faces
            )
            normals = orig_normals[v_map]
        else:
            verts = (mesh.vertices * scale).astype(np.float32)
            faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
            joints, weights = _top4_weights(mesh.weights)
            normals = _vertex_normals(verts.astype(np.float64), mesh.faces)
            v_map = None

        attrs = {
            "POSITION": accessor(verts, "VEC3", 5126, target=34962),
            "NORMAL": accessor(normals, "VEC3", 5126, target=34962),
            "JOINTS_0": accessor(joints, "VEC4", 5123, target=34962),
            "WEIGHTS_0": accessor(weights, "VEC4", 5126, target=34962),
        }

        if has_uvs:
            attrs["TEXCOORD_0"] = accessor(split_uvs, "VEC2", 5126, target=34962)

        mat_def: dict = {
            "name": mesh.name,
            "pbrMetallicRoughness": {
                "metallicFactor": metallic,
                "roughnessFactor": roughness,
            },
        }

        if has_texture:
            img_key = mesh.texture_image
            if img_key not in image_cache:
                img_offset, img_length = blob.add(mesh.texture_image)
                # bufferView for an embedded image must not specify GPU target per glTF 2.0
                views.append({
                    "buffer": 0,
                    "byteOffset": img_offset,
                    "byteLength": img_length,
                })
                img_view_idx = len(views) - 1
                mime_type = mesh.texture_mime or "image/png"
                images.append({"bufferView": img_view_idx, "mimeType": mime_type})
                img_idx = len(images) - 1

                if not tex_samplers:
                    # Linear, LinearMipmapLinear, Repeat, Repeat
                    tex_samplers.append({
                        "magFilter": 9729,
                        "minFilter": 9987,
                        "wrapS": 10497,
                        "wrapT": 10497,
                    })
                sampler_idx = 0
                textures.append({"sampler": sampler_idx, "source": img_idx})
                tex_idx = len(textures) - 1
                image_cache[img_key] = tex_idx
            else:
                tex_idx = image_cache[img_key]

            mat_def["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            mat_def["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx}
        else:
            base_color_factor = (
                [1.0, 1.0, 1.0, 1.0]
                if has_face_colors
                else [float(color[0]), float(color[1]), float(color[2]), 1.0]
            )
            mat_def["pbrMetallicRoughness"]["baseColorFactor"] = base_color_factor

            if has_face_colors:
                # Convert BGR (0..255) face colours to per-vertex RGBA (0..1)
                face_rgb = (mesh.face_colors[:, [2, 1, 0]] / 255.0).astype(np.float64)
                orig_v_count = mesh.vertices.shape[0]
                v_colors = np.zeros((orig_v_count, 3), dtype=np.float64)
                v_counts = np.zeros((orig_v_count, 1), dtype=np.float64)
                for i in range(3):
                    np.add.at(v_colors, mesh.faces[:, i], face_rgb)
                    np.add.at(v_counts, mesh.faces[:, i], 1.0)
                v_colors = v_colors / np.maximum(v_counts, 1.0)
                v_colors = np.clip(v_colors, 0.0, 1.0).astype(np.float32)
                if v_map is not None:
                    v_colors = v_colors[v_map]
                v_colors_rgba = np.concatenate(
                    [v_colors, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1
                )
                attrs["COLOR_0"] = accessor(v_colors_rgba, "VEC4", 5126, target=34962)

        materials.append(mat_def)

        primitives.append({
            "attributes": attrs,
            "indices": accessor(faces.reshape(-1), "SCALAR", 5125, target=34963),
            "material": len(materials) - 1,
        })

    n = len(r_anim_frames)
    times = (np.arange(n, dtype=np.float32) / max(float(fps), 1e-6))
    time_acc = accessor(times, "SCALAR", 5126)

    channels = []
    anim_samplers = []
    prev_q: dict[str, np.ndarray] = {}
    rot_tracks = {name: np.zeros((n, 4), dtype=np.float32) for name in rig.order}
    hips_track = np.zeros((n, 3), dtype=np.float32)
    for i, (r_anim, hips) in enumerate(zip(r_anim_frames, hips_cm_frames)):
        hips_track[i] = np.asarray(hips, dtype=np.float64).reshape(3) * scale
        for name in rig.order:
            bone = rig.bones[name]
            local = local_matrix(
                bone,
                r_anim.get(name, identity),
                hips if name == rig.root else None,
            )
            q_wxyz, _ = continue_quaternion(matrix_to_quat_wxyz(local[:3, :3]), prev_q.get(name))
            prev_q[name] = q_wxyz
            rot_tracks[name][i] = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

    def add_sampler(values: np.ndarray, typ: str) -> int:
        anim_samplers.append({"input": time_acc, "output": accessor(values, typ, 5126), "interpolation": "LINEAR"})
        return len(anim_samplers) - 1

    hips_s = add_sampler(hips_track.astype(np.float32), "VEC3")
    channels.append({"sampler": hips_s, "target": {"node": bone_index[rig.root], "path": "translation"}})
    for name in rig.order:
        sid = add_sampler(rot_tracks[name], "VEC4")
        channels.append({"sampler": sid, "target": {"node": bone_index[name], "path": "rotation"}})

    gltf = {
        "asset": {"version": "2.0", "generator": "v2bm"},
        "scene": 0,
        "scenes": [{"nodes": [bone_index[rig.root], mesh_node]}],
        "nodes": nodes,
        "meshes": [{"name": "character", "primitives": primitives}],
        "materials": materials,
        "skins": [{
            "name": "rigs",
            "joints": list(range(n_bones)),
            "skeleton": bone_index[rig.root],
            "inverseBindMatrices": ibm_acc,
        }],
        "animations": [{"name": "rig_clip", "samplers": anim_samplers, "channels": channels}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": blob._cursor}],
    }
    if images:
        gltf["images"] = images
    if textures:
        gltf["textures"] = textures
    if tex_samplers:
        gltf["samplers"] = tex_samplers

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
